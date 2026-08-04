"""OpenTelemetry spans for the decision path, derived from the §4.3 stage timings.

Why spans when §4.3 is already a histogram: the histogram says *that* inference is slow,
across all traffic. It cannot say that the slow requests are the ones decided by
`champion-v8`. So every span here carries the lineage attributes -- model, policy and
feature version, plus whether the decision was degraded -- and a p99 tail becomes
attributable to an artifact rather than to a time range. That is the whole reason this
module exists; a trace with no lineage on it would be a prettier version of the
histogram.

**No exporter by default.** With no `TracerProvider` wired, the OTel API returns a
non-recording tracer and everything below is a handful of attribute lookups. That is
what keeps `import fraudlens.serving.app` free of sockets in a unit test, and there is
deliberately no environment variable that can switch one on behind our backs: the
provider is a `create_app` parameter, exactly like the model loader and the clock.

**The stage spans are materialised from `StageTimings`, not measured a second time.**
Two clocks timing the same three stages will disagree eventually -- different overhead,
different sampling -- and then the trace and the SLO histogram tell different stories
about the same request, which is the one thing an incident cannot afford. Rejected
alternative: `with tracer.start_as_current_span(...)` inside `decisioning._scored`. It
reads better, and it would have put a tracing import on the path that prices
transactions and given us the two-clock problem for free.

The reconstruction is sound because the three stages run back to back inside `_scored`
with no work between them, so laying them end to end from the request anchor reproduces
the real timeline. The residual between their sum and `total` is the 50 ms of
deserialisation/validation/response overhead §4.3 budgets, and it shows up as the
untimed tail of the parent span -- which is where it belongs, since no stage owns it.
"""

from __future__ import annotations

from collections.abc import Mapping

from opentelemetry.trace import Span, Tracer, TracerProvider, get_tracer
from opentelemetry.util.types import AttributeValue

from fraudlens.serving.contracts import DecideRequest, VersionSet
from fraudlens.serving.decisioning import Outcome
from fraudlens.serving.latency import TOTAL, StageTimings

# Names the spans by the same constants the budget uses, so "the span" and "the budget
# line" are the same string and cannot drift apart in a query.
_INSTRUMENTATION = "fraudlens.serving"
_NS_PER_MS = 1_000_000


def tracer_for(provider: TracerProvider | None) -> Tracer:
    """The tracer to instrument with. `None` means the global provider, which is a no-op.

    Not a fallback to a configured default: an unwired service must produce no spans and
    open no connections, and the way to get real spans is to pass a provider in.
    """
    if provider is None:
        return get_tracer(_INSTRUMENTATION)
    return provider.get_tracer(_INSTRUMENTATION)


def record_decision_trace(
    tracer: Tracer,
    span: Span,
    *,
    request: DecideRequest,
    outcome: Outcome,
    versions: VersionSet,
    timings: StageTimings,
    anchor_ns: int,
) -> None:
    """Attribute the parent span and emit the §4.3 stage spans beneath it.

    One entry point rather than two, so the fail-safe path cannot end up with a traced
    total and untraced stages -- the branch that matters most is the one a scattered call
    site always half-instruments.
    """
    attributes = _lineage_attributes(
        transaction_id=request.transaction_id,
        action=outcome.action,
        degraded=outcome.degraded,
        budget_breached=timings.breached,
        versions=versions,
    )
    span.set_attributes(attributes)
    _stage_spans(tracer, anchor_ns=anchor_ns, stage_ms=timings.milliseconds, attributes=attributes)


def _lineage_attributes(
    *,
    transaction_id: int,
    action: str,
    degraded: bool,
    budget_breached: bool,
    versions: VersionSet,
) -> dict[str, AttributeValue]:
    """The attributes every span in a decision trace carries.

    The version triple is on *every* span, not only the parent, because the question a
    slow trace has to answer is "which artifact was slow" and a sampled backend may well
    show a stage span on its own. Repeating three short strings per span is cheaper than
    a join the on-call engineer has to invent at 3am.

    `degraded` is here for the opposite reason it is on the response: a degraded decision
    skipped inference and calibration entirely, so its `total` span is fast for a bad
    reason. Without this attribute the fail-safe path would quietly improve our latency
    percentiles during an outage.
    """
    return {
        "fraudlens.transaction_id": transaction_id,
        "fraudlens.action": action,
        "fraudlens.degraded": degraded,
        "fraudlens.budget_breached": budget_breached,
        "fraudlens.model_version": versions.model_version,
        "fraudlens.policy_version": versions.policy_version,
        "fraudlens.feature_version": versions.feature_version,
        # `config_hash` is deliberately not here. It is a 64-character digest that would
        # be repeated on every span of every request for a value that changes at deploy
        # time; the ledger already pins it per decision, which is where a dispute reads it.
    }


def _stage_spans(
    tracer: Tracer,
    *,
    anchor_ns: int,
    stage_ms: Mapping[str, float],
    attributes: Mapping[str, AttributeValue],
) -> None:
    """Emit one child span per §4.3 stage, laid end to end from `anchor_ns`.

    Called while the parent span is current, so the children attach to it through the
    OTel context rather than through an explicit parent argument.

    `TOTAL` is skipped: it is the parent. Stages absent from `stage_ms` are skipped
    rather than emitted with zero duration -- on the degraded path inference never ran,
    and a zero-length span would claim it ran instantly, which is the same lie the
    ledger avoids by writing NULL instead of 0.0 for a missing score.
    """
    start_ns = anchor_ns
    for stage, milliseconds in stage_ms.items():
        if stage == TOTAL:
            continue
        end_ns = start_ns + int(milliseconds * _NS_PER_MS)
        span = tracer.start_span(stage, start_time=start_ns, attributes=dict(attributes))
        span.end(end_time=end_ns)
        start_ns = end_ns
