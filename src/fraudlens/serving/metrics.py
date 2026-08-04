"""Prometheus instruments for the scoring path, and the one call that records a decision.

ADR-0002 ranks observability last of five, and the reason it is on the list at all is
that this system's failure mode is *silence*: labels arrive 30-90 days late, so a broken
model looks exactly like a working one for a quarter. Every instrument here is therefore
a leading indicator -- something that moves before the loss lands.

**What is deliberately not here.** The design spec's §5 catalog is larger than this file,
and the gap is not an oversight. An instrument this path cannot honestly fill would read
zero forever, and a zero on a dashboard is indistinguishable from "healthy" -- the same
reasoning that makes `decisioning._fallback` write NULL rather than 0.0 for a score it
does not have. A placeholder biases the dashboard toward calm at exactly the moment it
should not. So the following are named as E7's and left absent rather than stubbed:

* Anything needing a **baseline**: score PSI, per-feature drift, null rate vs training,
  unseen-level rate. The serving process has no training distribution and acquiring one
  would put an artifact load on the checkout path. The `fraudlens_calibrated_probability`
  buckets below are the serving half of the PSI computation; E7 owns the other half.
* Anything needing **matured labels**: AUC, PR-AUC, ECE, realized cost, fraud loss,
  false-decline cost, net value saved. At decision time the label does not exist. It
  arrives 30-90 days later, against the ledger, which is where E7 will compute it.
* Anything needing a **window**: training-serving skew (offline vs online score on the
  same row), duplicate rate, out-of-order rate. One request cannot see two of anything.
* **Expected cost per decision.** This one is genuinely observable synchronously -- the
  EV argmax computes it -- but `decisioning` does not return the cost arms and widening
  its `Outcome` for a metric is not this epic's change to make. Noted so E7 does not
  assume it is impossible; it is merely elsewhere.
* **Analyst queue depth and value per analyst-hour.** Not deferred -- *not applicable*.
  This deployment runs the three-action policy (`decisioning.INCLUDE_REVIEW = False`),
  so there is no review arm and no queue. An empty queue-depth panel would imply a queue
  that is keeping up.

Module-level singletons on the default registry. Metric objects are process-global by
construction in `prometheus_client`; creating them per app instance would raise on the
second `create_app()` and, worse, would reset counters on a reload.
"""

from __future__ import annotations

from collections.abc import Sequence

from prometheus_client import Counter, Gauge, Histogram

DECISIONS = Counter(
    "fraudlens_decisions_total",
    "Decisions served, by action and whether the model path or the fallback produced it.",
    labelnames=("action", "degraded"),
)

# Split from DECISIONS by cause rather than folded in as a label, because this is the
# metric an on-call alert fires on: a nonzero rate here means adverse decisions are being
# made without a score, which is safe but expensive and must not persist unnoticed.
DEGRADATIONS = Counter(
    "fraudlens_degraded_decisions_total",
    "Decisions produced by the fail-safe rule path, by cause.",
    labelnames=("reason",),
)

# Buckets straddle the §4.3 per-stage allocations (60/25/15 ms) and the 150 ms total, so
# the histogram can answer "what fraction of the budget did this stage use" directly
# rather than by interpolation between two far-apart buckets.
_STAGE_BUCKETS = (0.001, 0.005, 0.010, 0.015, 0.025, 0.050, 0.060, 0.100, 0.150, 0.300, 1.0)

STAGE_SECONDS = Histogram(
    "fraudlens_decision_stage_seconds",
    "Wall time per pipeline stage, plus the end-to-end total.",
    labelnames=("stage",),
    buckets=_STAGE_BUCKETS,
)

# §4.3: "Breaches counted as an SLO metric, not merely logged." A log line is invisible
# to an error budget; a counter is what a burn-rate alert can be written against.
BUDGET_BREACHES = Counter(
    "fraudlens_latency_budget_breaches_total",
    "Stage executions that exceeded their §4.3 latency allocation.",
    labelnames=("stage",),
)

# Resolution concentrated where the decision boundaries actually sit. Findings §3: the
# break-even probability runs from 0.369 on a $500+ basket to 0.740 on a new account, so
# uniform 0.1-wide buckets would put the entire decision-relevant range in three of them
# and we would be unable to see the score distribution move across a boundary.
_PROBABILITY_BUCKETS = (
    0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0
)  # fmt: skip

CALIBRATED_PROBABILITY = Histogram(
    "fraudlens_calibrated_probability",
    "Calibrated P(fraud) of scored decisions. Degraded decisions are absent, not zero.",
    buckets=_PROBABILITY_BUCKETS,
)

# Amount buckets chosen at the published break-even inflection points (findings §3:
# <$25 needs 0.731 to decline, $500+ needs 0.369) so "decline $ by amount band" falls out
# of the buckets. Rejected: an `amount_band` label. The band boundaries already exist in
# `serving.reasons`, and a second copy here would be a second definition of a business
# constant -- the exact failure §9a's one-source-of-truth rule is about. Buckets are a
# resolution choice; a label would have been a claim about what a band *is*.
_AMOUNT_BUCKETS = (10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 5000.0)

DECISION_AMOUNT_USD = Histogram(
    "fraudlens_decision_amount_usd",
    "Basket value flowing through each decision arm. `_sum` is GMV, `_bucket` is by band.",
    labelnames=("action", "degraded"),
    buckets=_AMOUNT_BUCKETS,
)

# The `degraded` label on the money, not just on the count, is the point. ADR-0002 lists
# the cost of the fail-safe arm as "false declines during degradation; must be measured,
# not assumed" -- and what makes that cost is dollars denied, not decisions denied.

REASON_CODES = Counter(
    "fraudlens_reason_codes_total",
    "Reason codes emitted, by code. A decision contributes one increment per code.",
    labelnames=("code",),
)

# Bounded by the closed taxonomy in `serving.reasons` (13 members), so the label is safe.
# Note what this is *not*: §5's "reason-code coverage" is not emitted, because the
# contract enforces `min_length=1` on the response and a coverage gauge would read 1.0
# forever. The mix is the useful signal instead -- a surge in
# UNKNOWN_TENURE_PRICED_AS_MEDIAN means the upstream tenure lookup broke, which is the
# closest thing this path has to a per-feature null rate.

REQUESTS = Counter(
    "fraudlens_requests_total",
    "HTTP responses by route template and status class-carrying code.",
    labelnames=("route", "status"),
)

# Route *template*, never the raw path: an unauthenticated scanner walking /admin, /.env
# and friends would otherwise mint a new time series per probe and take the registry down
# with it. Unmatched paths collapse to a single "unmatched" label for the same reason.
# 422 on /v1/decide is not noise -- `contracts` runs strict, extra="forbid" validation, so
# a 422 rate *is* the Tier 4 schema-violation rate.

LEDGER_WRITE_FAILURES = Counter(
    "fraudlens_ledger_write_failures_total",
    "Decisions served but not persisted to the audit trail.",
)

# The decision still went out and the customer was still charged or declined, so this is
# not an error rate -- it is an auditability incident rate. It was previously only a log
# line, which meant the one number a regulator would ask for ("how many of your decisions
# can you not reconstruct?") was recoverable only by grepping. §5's Tier 4 "% decisions
# with complete lineage" is the ledger-side view of the same quantity; this is the
# serving-side half and the two must agree.

MODEL_LOADED = Gauge(
    "fraudlens_model_loaded",
    "1 if the scoring artifact loaded at startup, 0 if the process is serving fail-safe.",
)

MODEL_LOAD_SECONDS = Gauge(
    "fraudlens_model_load_seconds",
    "Wall time of the one startup artifact load. 0 if the load failed.",
)

# A Gauge rather than a Histogram because loading happens exactly once per process and
# there is no reload path: a histogram over a single observation is a gauge with extra
# series. Restarts and memory are deliberately absent -- `prometheus_client` already
# exports `process_start_time_seconds` and `process_resident_memory_bytes`, and
# re-exporting them under our own names would give two numbers that can disagree.


def observe_decision(
    *,
    action: str,
    degraded: bool,
    amount: float,
    calibrated_probability: float | None,
    reason_codes: Sequence[str],
) -> None:
    """Record one decision across every instrument that observes decisions.

    One call rather than five at the call site, so an instrument added here cannot be
    added to the response path and forgotten on the fail-safe path -- which is the path
    that matters and the one a scattered call site always misses.

    Takes primitives rather than an `Outcome`: `decisioning` imports this module, so
    depending on its types here would close the cycle.
    """
    degraded_label = str(degraded).lower()
    DECISIONS.labels(action=action, degraded=degraded_label).inc()
    DECISION_AMOUNT_USD.labels(action=action, degraded=degraded_label).observe(amount)
    for code in reason_codes:
        REASON_CODES.labels(code=code).inc()
    if calibrated_probability is not None:
        # Guarded, not defaulted. A degraded decision has no probability and observing 0.0
        # would pull the serving score distribution toward "confidently legitimate"
        # precisely during an outage -- the drift dashboard would go calm as the model
        # died. Absence is the honest reading and E7's PSI must treat it that way too.
        CALIBRATED_PROBABILITY.observe(calibrated_probability)
