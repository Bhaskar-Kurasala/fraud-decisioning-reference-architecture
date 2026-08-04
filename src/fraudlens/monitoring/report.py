"""The periodic monitoring job: read the ledger, compute the tiers, emit the gauges.

Runs on a schedule, off the request path. The import-linter contract enforces that
direction (§9a, `pyproject.toml`): a window-wide PSI scan inside the 150 ms decision
budget is not a slow endpoint, it is an outage, and the layering makes it a build failure
instead of a production discovery.

Two dependency notes worth stating rather than discovering:

- `prometheus-client` and `sqlalchemy` are declared under the `serving` and `streaming`
  extras. This package needs both and has no extra of its own; a `monitoring` extra is
  owed in `pyproject.toml`, which E7 does not own. Until then, the monitoring worker must
  be installed with `[serving,streaming]`.
- Instruments are defined here rather than in `serving.metrics`, which is another agent's
  module. They follow its conventions exactly — `fraudlens_` prefix, `_total` on counters,
  seconds on durations — because E6's dashboards and alert rules reference the names.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from prometheus_client import Counter, Gauge
from sklearn.metrics import average_precision_score, roc_auc_score
from sqlalchemy import Engine, select

from fraudlens.monitoring.baseline import Baseline
from fraudlens.monitoring.calibration import CalibrationReport, calibration_report
from fraudlens.monitoring.drift import DriftResult, numeric_drift
from fraudlens.monitoring.maturity import (
    DEFAULT_MATURITY_FLOOR,
    ImmatureWindowError,
    LabelMaturity,
    label_maturity,
    require_mature,
)
from fraudlens.streaming.schema import ACTIONS, decision_ledger, revealed_labels

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

# The baseline feature name the calibrated probability is captured under. Tier 0's
# earliest warning is score-distribution PSI, so the score is treated as a monitored
# feature like any other rather than given its own code path.
SCORE_FEATURE = "calibrated_probability"

DRIFT_PSI = Gauge("fraudlens_drift_psi", "PSI against the training baseline.", ("feature",))
DRIFT_KS = Gauge("fraudlens_drift_ks", "KS statistic against the baseline.", ("feature",))
NULL_RATE = Gauge("fraudlens_feature_null_rate", "Null rate, monitored feature.", ("feature",))
UNSEEN_RATE = Gauge("fraudlens_unseen_level_rate", "Rows on an unseen level.", ("feature",))
ACTION_RATE = Gauge("fraudlens_action_rate", "Share of decisions per action.", ("action",))
DEGRADED_RATE = Gauge("fraudlens_degraded_decision_rate", "Share decided with no model score.")
LABEL_MATURITY = Gauge("fraudlens_label_maturity_ratio", "Share of window labels that landed.")
# Labelled by horizon rather than one gauge per horizon: E6's panel plots the curve, and a
# curve assembled from four differently-named series cannot be graphed as one.
MATURITY_CURVE = Gauge(
    "fraudlens_label_maturity_curve_ratio",
    "Share labelled by t+horizon, among rows old enough for the horizon to have elapsed.",
    ("horizon_days",),
)
CALIBRATION_ECE = Gauge("fraudlens_calibration_ece", "ECE on matured labels.")
CALIBRATION_SLOPE = Gauge("fraudlens_calibration_slope", "Cox calibration slope; 1.0 is perfect.")
CALIBRATION_INTERCEPT = Gauge("fraudlens_calibration_intercept", "Cox intercept; 0.0 is perfect.")
OBSERVED_FRAUD_RATE = Gauge("fraudlens_observed_fraud_rate", "Fraud rate on matured labels.")
# Tier 0, not Tier 2: the moments of the score distribution need no labels, so they are
# emitted on every run including the ones where Tier 2 is refused. §5 lists "mean, p95
# predicted probability" as a leading indicator precisely because it is the thing still
# measurable during the 30-90 days when nothing else is.
PREDICTED_FRAUD_RATE = Gauge(
    "fraudlens_predicted_fraud_rate", "Mean predicted probability over scored decisions."
)
PREDICTED_P95 = Gauge(
    "fraudlens_predicted_probability_p95", "95th percentile predicted probability."
)
MODEL_AUC = Gauge("fraudlens_model_auc", "ROC AUC on matured labels.")
MODEL_PR_AUC = Gauge("fraudlens_model_pr_auc", "PR-AUC on matured labels; primary at 3.5% base.")
# The baseline id rides as a label on a constant 1 so a drift panel can show which
# yardstick produced it. A PSI without its baseline id is an unfalsifiable claim.
BASELINE_INFO = Gauge("fraudlens_monitoring_baseline_info", "Always 1.", ("baseline_id",))
RUNS = Counter("fraudlens_monitoring_runs_total", "Monitoring runs, by outcome.", ("outcome",))
TIER2_REFUSALS = Counter(
    "fraudlens_monitoring_tier2_refusals_total", "Windows refused as immature."
)
RUN_DURATION = Gauge("fraudlens_monitoring_run_duration_seconds", "Wall time of the last run.")


@dataclass(frozen=True, slots=True)
class Tier2Metrics:
    """Label-dependent performance, and the maturity that licenses reading it."""

    n: int
    auc: float
    pr_auc: float
    calibration: CalibrationReport
    maturity_stamp: str


@dataclass(frozen=True, slots=True)
class MonitoringReport:
    """One run's output. `tier2` is None exactly when `tier2_refusal` explains why."""

    window_start: dt.datetime
    window_end: dt.datetime
    as_of: dt.datetime
    baseline_id: str
    n_decisions: int
    n_degraded: int
    action_rates: dict[str, float]
    # Tier 0. Held on the report rather than read off `tier2.calibration`, where an
    # identical `predicted_fraud_rate` also lives: that one is scoped to the matured
    # subset and disappears when Tier 2 is refused. These are over every scored decision
    # in the window and survive the refusal, which is the point of a leading indicator.
    # NaN when nothing was scored — an outage, and a distinct state from "mean is zero".
    mean_predicted_probability: float
    p95_predicted_probability: float
    score_drift: DriftResult | None
    maturity: LabelMaturity
    tier2: Tier2Metrics | None
    tier2_refusal: str | None


def matured_performance(
    y_true: IntArray,
    y_prob: FloatArray,
    maturity: LabelMaturity,
    *,
    floor: float = DEFAULT_MATURITY_FLOOR,
) -> Tier2Metrics:
    """AUC / PR-AUC / ECE over matured labels, or `ImmatureWindowError`.

    The maturity check is a parameter of this function rather than a caller's
    responsibility because every way of making it the caller's responsibility has been
    tried and the caller forgets. Same ordering as the promotion gate (`models.gate`):
    maturity is a precondition, and nothing downstream of a failed precondition is
    computed, because a misleading number in an audit log is worse than an absent one.
    """
    require_mature(maturity, floor)
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be the same length")
    return Tier2Metrics(
        n=len(y_true),
        auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        calibration=calibration_report(y_true, y_prob),
        maturity_stamp=maturity.stamp(),
    )


def _read_window(
    engine: Engine, window_start: dt.datetime, window_end: dt.datetime, as_of: dt.datetime
) -> list[dict[str, Any]]:
    """Decisions in the window, left-joined to whatever labels have landed by `as_of`.

    The `revealed_at <= as_of` predicate is applied in the join rather than after it.
    Filtering afterwards would drop the row entirely instead of leaving it unlabelled,
    which silently shrinks the maturity denominator — the failure would make an immature
    window report 100% matured, defeating the gate.
    """
    joined = decision_ledger.outerjoin(
        revealed_labels,
        (revealed_labels.c.transaction_id == decision_ledger.c.transaction_id)
        & (revealed_labels.c.revealed_at <= as_of),
    )
    stmt = (
        select(
            decision_ledger.c.transaction_at,
            decision_ledger.c.calibrated_probability,
            decision_ledger.c.action,
            decision_ledger.c.degraded,
            revealed_labels.c.is_fraud,
            revealed_labels.c.revealed_at,
        )
        .select_from(joined)
        .where(decision_ledger.c.transaction_at >= window_start)
        .where(decision_ledger.c.transaction_at < window_end)
        .order_by(decision_ledger.c.transaction_at)
    )
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(stmt).mappings().all()]


def run_report(
    engine: Engine,
    baseline: Baseline,
    *,
    window_start: dt.datetime,
    window_end: dt.datetime,
    as_of: dt.datetime,
    maturity_floor: float = DEFAULT_MATURITY_FLOOR,
) -> MonitoringReport:
    """Compute Tier 0 unconditionally and Tier 2 only if the labels permit."""
    started = time.perf_counter()
    rows = _read_window(engine, window_start, window_end, as_of)
    maturity = label_maturity(
        [_as_datetime(r["transaction_at"]) for r in rows],
        [_as_optional_datetime(r["revealed_at"]) for r in rows],
        as_of=as_of,
        window_start=window_start,
        window_end=window_end,
    )
    scored = [r for r in rows if r["calibrated_probability"] is not None]
    probabilities = np.array([float(r["calibrated_probability"]) for r in scored], dtype=np.float64)

    # Score drift only. §5 Tier 0 also asks for per-feature PSI, and `drift` computes it,
    # but the ledger stores an input *hash*, not the feature values — deliberately, since
    # 225 features per row would multiply the audit trail's size for a use case that is
    # not audit. Per-feature drift therefore has to be run against the feature store the
    # scorer read from, which is E8's problem, not this job's.
    reference = baseline.numeric.get(SCORE_FEATURE)
    if reference is None:
        raise ValueError(
            f"baseline {baseline.baseline_id[:12]} has no {SCORE_FEATURE!r} reference; "
            "score-distribution PSI is the Tier 0 leading indicator and cannot be skipped"
        )
    score_drift = numeric_drift(reference, probabilities) if probabilities.size else None

    tier2, refusal = _tier2(scored, maturity, maturity_floor)
    RUN_DURATION.set(time.perf_counter() - started)
    RUNS.labels(outcome="refused" if refusal else "complete").inc()
    return MonitoringReport(
        window_start=window_start,
        window_end=window_end,
        as_of=as_of,
        baseline_id=baseline.baseline_id,
        n_decisions=len(rows),
        n_degraded=sum(1 for r in rows if bool(r["degraded"])),
        action_rates=_action_rates(rows),
        mean_predicted_probability=float(probabilities.mean())
        if probabilities.size
        else float("nan"),
        p95_predicted_probability=float(np.quantile(probabilities, 0.95))
        if probabilities.size
        else float("nan"),
        score_drift=score_drift,
        maturity=maturity,
        tier2=tier2,
        tier2_refusal=refusal,
    )


def _tier2(
    scored: Sequence[dict[str, Any]], maturity: LabelMaturity, floor: float
) -> tuple[Tier2Metrics | None, str | None]:
    labelled = [r for r in scored if r["is_fraud"] is not None]
    if not labelled:
        return None, f"no matured labels in the window ({maturity.stamp()})"
    y_true = np.array([int(bool(r["is_fraud"])) for r in labelled], dtype=np.int_)
    y_prob = np.array([float(r["calibrated_probability"]) for r in labelled], dtype=np.float64)
    try:
        return matured_performance(y_true, y_prob, maturity, floor=floor), None
    except (ImmatureWindowError, ValueError) as refusal:
        TIER2_REFUSALS.inc()
        return None, str(refusal)


def _action_rates(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    total = len(rows) or 1
    counts = dict.fromkeys(ACTIONS, 0)
    for row in rows:
        counts[str(row["action"])] += 1
    return {action: count / total for action, count in counts.items()}


def _as_datetime(value: Any) -> dt.datetime:
    assert isinstance(value, dt.datetime)  # noqa: S101 — column type guarantees this
    return value


def _as_optional_datetime(value: Any) -> dt.datetime | None:
    return None if value is None else _as_datetime(value)


def emit(report: MonitoringReport) -> None:
    """Publish the report to Prometheus.

    Refused Tier 2 gauges are set to NaN, not left alone. A gauge that is simply not
    updated keeps its last value forever, so a monitoring run that refused to compute ECE
    would leave yesterday's reassuring 0.003 on the dashboard indefinitely — the panel
    would look healthiest exactly when it had stopped measuring. NaN renders as a gap,
    which is the honest signal.
    """
    BASELINE_INFO.labels(baseline_id=report.baseline_id).set(1)
    DEGRADED_RATE.set(report.n_degraded / report.n_decisions if report.n_decisions else 0.0)
    for action, rate in report.action_rates.items():
        ACTION_RATE.labels(action=action).set(rate)
    if report.score_drift is not None:
        _emit_drift(report.score_drift)
    # Emitted before the Tier 2 block and outside its guard: these are label-free, and a
    # Tier 2 refusal must not take the leading indicators down with it.
    PREDICTED_FRAUD_RATE.set(report.mean_predicted_probability)
    PREDICTED_P95.set(report.p95_predicted_probability)
    LABEL_MATURITY.set(report.maturity.ratio)
    for point in report.maturity.curve:
        MATURITY_CURVE.labels(horizon_days=str(point.horizon_days)).set(point.ratio)

    tier2 = report.tier2
    nan = float("nan")
    MODEL_AUC.set(tier2.auc if tier2 else nan)
    MODEL_PR_AUC.set(tier2.pr_auc if tier2 else nan)
    CALIBRATION_ECE.set(tier2.calibration.ece if tier2 else nan)
    CALIBRATION_SLOPE.set(tier2.calibration.slope if tier2 else nan)
    CALIBRATION_INTERCEPT.set(tier2.calibration.intercept if tier2 else nan)
    OBSERVED_FRAUD_RATE.set(tier2.calibration.observed_fraud_rate if tier2 else nan)


def _emit_drift(drift: DriftResult) -> None:
    DRIFT_PSI.labels(feature=drift.feature).set(drift.psi)
    NULL_RATE.labels(feature=drift.feature).set(drift.null_rate)
    UNSEEN_RATE.labels(feature=drift.feature).set(drift.unseen_level_rate)
    if drift.ks is not None:
        DRIFT_KS.labels(feature=drift.feature).set(drift.ks)
