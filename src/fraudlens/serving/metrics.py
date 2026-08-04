"""Prometheus instruments for the scoring path.

ADR-0002 ranks observability last of five, and the reason it is on the list at all is
that this system's failure mode is *silence*: labels arrive 30-90 days late, so a broken
model looks exactly like a working one for a quarter. Every instrument here is therefore
a leading indicator -- something that moves before the loss lands.

Module-level singletons on the default registry. Metric objects are process-global by
construction in `prometheus_client`; creating them per app instance would raise on the
second `create_app()` and, worse, would reset counters on a reload.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

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
