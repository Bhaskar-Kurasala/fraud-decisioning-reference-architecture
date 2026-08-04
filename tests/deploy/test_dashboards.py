"""Dashboards and alert rules against the metrics the code actually exports.

This is the load-bearing test of Epic E6, and the reason is a specific failure:
dashboards drift from code **silently**. Nothing breaks when an instrument is renamed,
nothing fails CI, no exception is raised. The panel simply renders "No data", and the
first person to find out is an on-call engineer at 3am who then has to decide whether
"No data" means the system is quiet or the system is dead. During an incident those two
readings lead to opposite actions.

So every metric name in every PromQL expression -- dashboards and alert rules alike --
must either be exported by `fraudlens` today or appear on the explicit allowlist below,
which names the epic that will export it. There is no third option and no wildcard.

The allowlist is not a loophole. It is the *contract* between E6 and E7/E8: those epics
have to export exactly these names, and if they choose different ones this test fails and
the dashboards get fixed in the same commit rather than a quarter later.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from prometheus_client import REGISTRY
from prometheus_client.metrics import MetricWrapperBase

# Importing is what registers the instruments on the default registry, and both deployment
# units have to be imported here even though they never run in the same process: Prometheus
# scrapes them as two jobs and Grafana renders them on one board, so the correspondence
# this file checks is against the union.
#
# The serving import also demonstrates the property `test_tracing` asserts directly: it
# opens no connection, because tracing is inert without an injected provider.
import fraudlens.monitoring.report
import fraudlens.serving.app  # noqa: F401
from fraudlens.serving import metrics

REPO = Path(__file__).resolve().parents[2]
DASHBOARDS = REPO / "deploy" / "grafana" / "dashboards"
RULES = REPO / "deploy" / "prometheus" / "rules"
RUNBOOKS = REPO / "docs" / "runbooks"

# Metrics that Prometheus itself synthesises for every scraped target. `up` in particular
# is the only signal available when the exporter is the thing that broke.
_PROMETHEUS_SYNTHETIC = frozenset({"up", "scrape_duration_seconds", "scrape_samples_scraped"})

# --------------------------------------------------------------------------------------
# Metrics the dashboards reference that nothing exports YET, with the epic that owns each.
#
# Every entry is a quantity that **cannot be observed from a single decision**, for one of
# two structural reasons, and stubbing any of them would put a permanent zero on a
# dashboard — which reads as "healthy" and is worse than an empty panel that says why it
# is empty. Same reasoning as the ledger writing NULL rather than 0.0 for a degraded
# decision's score.
#
#   needs a window   — one event cannot be compared with another it never saw
#                      (duplicates, out-of-order, offline-vs-online skew).
#   needs an outcome — the money is only known once the chargeback settles, 30-90 days on.
#
# Note what is deliberately NOT on this list any more. Drift, calibration, maturity and
# label-dependent model metrics all resolved when E7's `monitoring.report` exporter landed,
# and the dashboards were rewritten onto the names that exporter actually uses rather than
# the names E6 had guessed. That correction is the entire point of this test existing: the
# guessed names would have rendered "No data" forever and nothing would have failed.
# --------------------------------------------------------------------------------------
E7_PENDING = {
    # --- Tier 0/4: needs a window over more than one event
    "fraudlens_training_serving_skew_total": "E7",
    "fraudlens_duplicate_transactions_total": "E7",
    "fraudlens_out_of_order_events_total": "E7",
    # --- Tier 2: needs a realised outcome per transaction rather than an aggregate
    "fraudlens_decile_lift": "E7",
    # --- Tier 1: needs matured labels before a cost can be realised
    "fraudlens_net_value_saved_usd_per_day": "E7",
    "fraudlens_realized_cost_usd_total": "E7",
    "fraudlens_expected_cost_usd_total": "E7",
    "fraudlens_fraud_loss_usd_total": "E7",
    "fraudlens_chargebacks_total": "E7",
    "fraudlens_false_decline_cost_usd_total": "E7",
    "fraudlens_cost_per_decision_usd": "E7",
    # Tenure is computed inside `decisioning` and is not a label on the serving counters,
    # so segment approval is derived from the ledger rather than from the request path.
    "fraudlens_approval_rate": "E7",
    # --- Tier 2: the flywheel's own numbers. No challenger has been run.
    "fraudlens_challenger_cost_delta_usd": "E8",
    "fraudlens_challenger_cost_delta_ci_low": "E8",
    "fraudlens_challenger_cost_delta_ci_high": "E8",
    "fraudlens_days_since_retrain": "E8",
    "fraudlens_promotion_gate_failures_total": "E8",
    # --- Tier 4: a versioned assumption artifact with an owner, not a runtime observation
    "fraudlens_assumption_age_days": "E9",
}

# PromQL vocabulary. Anything here is syntax, not a metric name.
_PROMQL_WORDS = frozenset(
    """abs absent absent_over_time avg avg_over_time bottomk bool by ceil changes clamp
    clamp_max clamp_min count count_over_time count_values day_of_month day_of_week
    day_of_year days_in_month delta deriv exp floor group group_left group_right
    histogram_quantile holt_winters hour idelta ignoring increase inf instant irate
    label_join label_replace last_over_time ln log10 log2 max max_over_time min
    min_over_time minute month nan offset on predict_linear present_over_time quantile
    quantile_over_time rate resets round scalar sgn sort sort_desc sqrt start stddev
    stddev_over_time stdvar stdvar_over_time sum sum_over_time time timestamp topk
    unless vector without year and or le""".split()  # noqa: SIM905
)

# Order matters: label matchers and ranges are stripped before identifiers are extracted,
# so `{stage="total"}` and `[5m]` never contribute a false metric name.
_STRIPPERS = (
    re.compile(r"\{[^}]*\}"),
    re.compile(r"\[[^\]]*\]"),
    re.compile(r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)"),
    re.compile(r"\"[^\"]*\""),
)
_IDENTIFIER = re.compile(r"\b[a-zA-Z_:][a-zA-Z0-9_:]*\b")


def _metric_names(expr: str) -> set[str]:
    """Metric names referenced by a PromQL expression."""
    for stripper in _STRIPPERS:
        expr = stripper.sub(" ", expr)
    return {name for name in _IDENTIFIER.findall(expr) if name not in _PROMQL_WORDS}


def _exported_names() -> set[str]:
    """Every sample name `fraudlens` can actually serve on `/metrics`.

    Derived from the live registry rather than from a hand-maintained list, because a
    hand-maintained list drifts in exactly the way this whole test exists to prevent.
    """
    names: set[str] = set()
    for family in REGISTRY.collect():
        names.add(family.name)
        suffixes = {
            "counter": ("_total", "_created"),
            "histogram": ("_bucket", "_sum", "_count", "_created"),
            "summary": ("_sum", "_count", "_created"),
        }.get(family.type, ())
        names.update(family.name + suffix for suffix in suffixes)
        names.update(sample.name for metric in family.samples for sample in [metric])
    return names


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rule_files() -> list[Path]:
    return sorted(RULES.glob("*.yml"))


def _all_rules() -> list[dict[str, Any]]:
    return [
        rule for path in _rule_files() for group in _yaml(path)["groups"] for rule in group["rules"]
    ]


def _alerts() -> list[dict[str, Any]]:
    return [rule for rule in _all_rules() if "alert" in rule]


def _recorded_names() -> set[str]:
    return {rule["record"] for rule in _all_rules() if "record" in rule}


def _dashboards() -> list[tuple[Path, dict[str, Any]]]:
    return [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(DASHBOARDS.glob("*.json"))
    ]


def _targets(dashboard: dict[str, Any]) -> list[tuple[str, str]]:
    """(panel title, expr) for every query on the dashboard."""
    return [
        (panel.get("title", "<untitled>"), target["expr"])
        for panel in dashboard["panels"]
        for target in panel.get("targets", [])
        if "expr" in target
    ]


# ======================================================================================
# The correspondence between what is displayed and what is emitted.
# ======================================================================================


def test_every_dashboard_metric_is_exported_or_explicitly_pending() -> None:
    """The failure this file exists to prevent: a panel that renders "No data" forever."""
    known = _exported_names() | _recorded_names() | set(E7_PENDING) | _PROMETHEUS_SYNTHETIC
    unknown: list[str] = []
    for path, dashboard in _dashboards():
        for title, expr in _targets(dashboard):
            for name in _metric_names(expr) - known:
                unknown.append(f"{path.name} :: {title} :: {name}")
    assert not unknown, "dashboard panels reference metrics nobody exports:\n" + "\n".join(unknown)


def test_every_alert_metric_is_exported_or_explicitly_pending() -> None:
    """An alert on a metric nobody exports never fires, and looks identical to one that is quiet."""
    known = _exported_names() | _recorded_names() | set(E7_PENDING) | _PROMETHEUS_SYNTHETIC
    unknown: list[str] = []
    for rule in _all_rules():
        name = rule.get("alert") or rule["record"]
        unknown.extend(f"{name} :: {m}" for m in _metric_names(rule["expr"]) - known)
    assert not unknown, "rules reference metrics nobody exports:\n" + "\n".join(unknown)


def test_every_serving_instrument_appears_somewhere() -> None:
    """The other direction: an instrument nobody looks at is cost without benefit.

    Emitting a metric costs cardinality, exposition size on the checkout path, and a
    maintenance obligation. If no dashboard and no rule reads it, it is not observability
    -- it is exhaust, and it should be deleted rather than kept in case.

    Scoped to `fraudlens.serving.metrics`, which is what E6 owns and what runs inside the
    150 ms budget. The monitoring workers' own exporter coverage is E7's test to write:
    asserting it here would make this file fail whenever a sibling epic adds a gauge, and
    a cross-epic test that fails for reasons the reader cannot act on gets marked xfail
    and then stops protecting anything.
    """
    referenced: set[str] = set()
    for _, dashboard in _dashboards():
        for _, expr in _targets(dashboard):
            referenced |= _metric_names(expr)
    for rule in _all_rules():
        referenced |= _metric_names(rule["expr"])

    owned = {
        instrument._name
        for instrument in vars(metrics).values()
        if isinstance(instrument, MetricWrapperBase)
    }
    # A histogram or counter is "read" through any of its derived series, so a prefix match
    # is the right test: `_bucket`, `_sum` and `_total` all count as reading the family.
    unread = sorted(name for name in owned if not any(r.startswith(name) for r in referenced))
    assert not unread, f"serving instruments exported but never displayed: {unread}"


# ======================================================================================
# The seven alerts of design spec §5.2, and their runbooks.
# ======================================================================================

_EXPECTED_ALERTS = {
    "FraudLensScoreDrift",
    "FraudLensDataBreakage",
    "FraudLensLatencySLOBreach",
    "FraudLensDeclineAnomaly",
    "FraudLensCalibrationDecay",
    "FraudLensTrainingServingSkew",
    "FraudLensPromotionGateFailure",
}


def test_exactly_the_seven_alerts_page() -> None:
    """§5.2: "Only these page; everything else is dashboard-only."

    Locked as a test because alert lists grow. Every addition is individually defensible
    and the aggregate is a pager nobody reads, at which point the two that matter are lost
    with the rest.
    """
    assert {alert["alert"] for alert in _alerts()} == _EXPECTED_ALERTS


@pytest.mark.parametrize("alert", _alerts(), ids=lambda a: str(a["alert"]))
def test_each_alert_links_a_runbook_that_exists(alert: dict[str, Any]) -> None:
    """A page with no runbook is a page that starts with ten minutes of orientation."""
    url = alert["annotations"]["runbook_url"]
    assert (REPO / url).is_file(), f"{alert['alert']} points at a missing runbook: {url}"


@pytest.mark.parametrize("alert", _alerts(), ids=lambda a: str(a["alert"]))
def test_each_alert_carries_severity_and_a_summary(alert: dict[str, Any]) -> None:
    assert alert["labels"]["severity"] == "page"
    assert alert["annotations"]["summary"].strip()
    assert alert["annotations"]["description"].strip()


@pytest.mark.parametrize("path", sorted(RUNBOOKS.glob("*.md")), ids=lambda p: p.name)
def test_each_runbook_answers_the_five_questions(path: Path) -> None:
    """Symptom, cause, first query, remediation, and who decides.

    The last one is the one that gets left out and the one that costs the most time: an
    engineer who has diagnosed the problem in four minutes and then spends forty deciding
    whether they are allowed to act. In this system that matters more than usual, because
    several of the obvious remediations -- move the threshold, promote the challenger --
    are decisions about money that on-call must not make alone.
    """
    body = path.read_text(encoding="utf-8").lower()
    for section in ("symptom", "cause", "query", "remediation", "who decides"):
        assert section in body, f"{path.name} has no '{section}' section"


def test_every_runbook_is_referenced_by_an_alert() -> None:
    """An orphaned runbook is documentation for a page that cannot happen."""
    linked = {Path(a["annotations"]["runbook_url"]).name for a in _alerts()}
    assert {p.name for p in RUNBOOKS.glob("*.md")} == linked


# ======================================================================================
# Dashboard hygiene: the things a reader has to be told, on the panel, in the moment.
# ======================================================================================


def test_the_four_dashboards_are_provisioned() -> None:
    """§5.1, verbatim. The numbering is the reading order, not decoration."""
    assert [p.name for p, _ in _dashboards()] == [
        "00-executive.json",
        "01-model-health.json",
        "02-serving-slo.json",
        "03-data-quality.json",
    ]


def test_the_executive_board_is_the_landing_page() -> None:
    """ADR-0002 inverts the usual order: business P&L first, golden signals one layer down.

    All four golden signals can be green while the system is at its most expensive -- a
    model 0.0021 AUC below champion costs $4.36M/yr. If the landing page ever silently
    reverts to a serving dashboard, that argument has been lost without anyone deciding to
    lose it.
    """
    ini = (REPO / "deploy" / "grafana" / "grafana.ini").read_text(encoding="utf-8")
    assert "00-executive.json" in ini
    assert "fraudlens-executive" in ini


@pytest.mark.parametrize("path,dashboard", _dashboards(), ids=lambda x: getattr(x, "name", ""))
def test_panels_pinned_to_the_provisioned_datasource(path: Path, dashboard: dict[str, Any]) -> None:
    """A panel whose datasource is "whatever is default today" renders empty on a fresh stack."""
    for panel in dashboard["panels"]:
        if not panel.get("targets"):
            continue
        assert panel["datasource"]["uid"] == "fraudlens-prometheus", panel["title"]


@pytest.mark.parametrize("path,dashboard", _dashboards(), ids=lambda x: getattr(x, "name", ""))
def test_pending_panels_say_so_rather_than_sitting_empty(
    path: Path, dashboard: dict[str, Any]
) -> None:
    """An empty panel is ambiguous; an empty panel labelled AWAITING E7 is information.

    This is the same reasoning as the NULL-not-zero rule in the ledger, applied to a
    dashboard: silence that could be read as calm has to be labelled as silence.
    """
    for panel in dashboard["panels"]:
        pending = {
            name
            for _, expr in [(panel.get("title", ""), t["expr"]) for t in panel.get("targets", [])]
            for name in _metric_names(expr)
            if name in E7_PENDING
        }
        if not pending:
            continue
        blurb = f"{panel.get('title', '')} {panel.get('description', '')}".upper()
        assert "AWAITING" in blurb, f"{path.name}: {panel.get('title')} queries {pending} silently"


@pytest.mark.parametrize("path,dashboard", _dashboards(), ids=lambda x: getattr(x, "name", ""))
def test_every_querying_panel_explains_itself(path: Path, dashboard: dict[str, Any]) -> None:
    """A panel with no description is a number with no unit, no baseline and no owner."""
    for panel in dashboard["panels"]:
        if panel.get("targets"):
            assert panel.get("description", "").strip(), f"{path.name}: {panel.get('title')}"


def test_the_assumption_dependent_panel_states_its_assumption() -> None:
    """§11: making the softest number the most visible one is the point.

    False-decline cost is proportional to P(churn|declined), which has no data support in
    IEEE-CIS at all. A panel that reports it as a measurement, without saying so, converts
    an assumption into a fact somewhere between the dashboard and the board pack.
    """
    executive = json.loads((DASHBOARDS / "00-executive.json").read_text(encoding="utf-8"))
    panels = [p for p in executive["panels"] if "false-decline" in p.get("title", "").lower()]
    assert panels, "the executive board has no false-decline cost panel"
    for panel in panels:
        description = panel["description"]
        assert "ASSUMPTION" in description.upper()
        assert "P(churn|declined)" in description


# ======================================================================================
# The configuration itself parses and points where it claims to.
# ======================================================================================


def test_prometheus_scrapes_the_api_and_loads_the_rules() -> None:
    config = _yaml(REPO / "deploy" / "prometheus" / "prometheus.yml")
    jobs = {job["job_name"] for job in config["scrape_configs"]}
    assert "fraudlens-api" in jobs
    assert any("rules" in pattern for pattern in config["rule_files"])
    # The three-sigma decline rule needs enough samples inside a 10m window for a
    # standard deviation to be a statistic rather than a coin flip.
    assert config["global"]["scrape_interval"] == "15s"


def test_dashboards_are_not_editable_in_the_browser() -> None:
    """A dashboard edited during an incident cannot be reproduced afterwards -- and the
    panel that was changed is invariably the one whose reading drove the decision under
    review."""
    provider = _yaml(REPO / "deploy" / "grafana" / "provisioning" / "dashboards" / "fraudlens.yml")
    assert provider["providers"][0]["allowUiUpdates"] is False
    for _, dashboard in _dashboards():
        assert dashboard["editable"] is False


def test_dashboard_uids_are_unique_and_pinned() -> None:
    """Links between boards and from alerts are by uid; a generated uid breaks them on
    every reprovision."""
    uids = [dashboard["uid"] for _, dashboard in _dashboards()]
    assert len(set(uids)) == len(uids)
    assert all(uid.startswith("fraudlens-") for uid in uids)
