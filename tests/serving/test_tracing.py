"""Spans for the decision path: the shape, the lineage, and the silence by default.

Two properties are load-bearing here and both are easy to lose silently.

The first is that an unwired service exports nothing. A tracing setup that reaches for an
env var, or that falls back to a "sensible default" exporter, turns every unit test in
this repository into a test that needs a network -- and the failure shows up as a slow
CI run months later, not as a red test now.

The second is that the spans and the §4.3 latency histogram tell the same story. They are
derived from the same `StageTimings` measurements precisely so they cannot diverge, and
the test that asserts they agree is what keeps that true if someone later decides to
"just time it properly" inside the decisioning path.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan

from fraudlens.serving.latency import BUDGET_MS, CALIBRATION_POLICY, FEATURES, INFERENCE, TOTAL
from fraudlens.serving.tracing import tracer_for
from tests.serving.conftest import StubScorer, make_client, request_body


def _decide(exporter: InMemorySpanExporter, **kwargs: object) -> list[ReadableSpan]:
    """Serve one decision through a recording provider and return the finished spans.

    In-memory exporter, never OTLP: the exporter is the seam this module is about, and a
    test that needed a collector running would be evidence against the design.

    `SimpleSpanProcessor`, not `Batch`: batch flushes on a timer, so the assertions would
    race the exporter and the test would be flaky in the way that gets tests deleted.
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with make_client(tracer_provider=provider, **kwargs) as client:  # type: ignore[arg-type]
        client.post("/v1/decide", json=request_body())
    provider.shutdown()
    return list(exporter.get_finished_spans())


def test_no_provider_means_no_recording_and_no_exporter() -> None:
    """The default must be inert. Importing the app must not be able to open a socket.

    `tracer_for(None)` goes to the global provider, which is OTel's no-op unless something
    has installed one. A non-recording span is the observable proof: it carries no
    attributes, allocates nothing, and there is nowhere for it to be sent.
    """
    tracer = tracer_for(None)
    with tracer.start_as_current_span("probe") as span:
        assert isinstance(span, NonRecordingSpan)
        assert span.is_recording() is False


def test_spans_match_the_budget_stages_one_for_one() -> None:
    """Span names are the §4.3 budget lines, so a trace and the SLO use the same vocabulary.

    Asserted against `BUDGET_MS` itself rather than a literal list: if someone adds a stage
    to the budget without tracing it, the two views of the request silently disagree about
    what the request even consists of, and this is the test that stops that.
    """
    exporter = InMemorySpanExporter()
    finished = _decide(exporter, scorer=StubScorer(0.01))

    assert {span.name for span in finished} == set(BUDGET_MS)


def test_stage_spans_are_children_of_the_total_span() -> None:
    exporter = InMemorySpanExporter()
    finished = _decide(exporter, scorer=StubScorer(0.01))
    by_name = {span.name: span for span in finished}
    parent = by_name[TOTAL]

    for stage in (FEATURES, INFERENCE, CALIBRATION_POLICY):
        assert by_name[stage].parent is not None
        assert by_name[stage].parent.span_id == parent.context.span_id  # type: ignore[union-attr]


def test_every_span_carries_the_lineage_needed_to_blame_an_artifact() -> None:
    """The reason for tracing at all: a slow span must name the artifact that was slow.

    On every span, not only the parent. A sampled backend will show a stage span on its
    own, and "inference was slow" without a model version is a fact nobody can act on.
    """
    exporter = InMemorySpanExporter()
    finished = _decide(exporter, scorer=StubScorer(0.01, version="champion-v7"))

    for span in finished:
        attributes = dict(span.attributes or {})
        assert attributes["fraudlens.model_version"] == "champion-v7"
        assert attributes["fraudlens.policy_version"] == "ev-argmax-3action-v1+rules-ladder-v1"
        assert attributes["fraudlens.feature_version"] == "request-supplied-v1"
        assert attributes["fraudlens.degraded"] is False
        assert attributes["fraudlens.transaction_id"] == request_body()["transaction_id"]


def test_the_degraded_path_is_traced_and_says_so() -> None:
    """A fail-safe decision skips inference and calibration, which makes it *fast*.

    Without the `degraded` attribute an outage would look like a latency improvement, and
    the stages that never ran would appear as instantaneous successes. So: the attribute
    is present, and the spans for stages that did not run are absent rather than zero --
    the same choice the ledger makes when it writes NULL instead of 0.0 for a missing
    score.
    """
    exporter = InMemorySpanExporter()
    finished = _decide(exporter)  # no scorer: the load fails, every decision degrades
    by_name = {span.name: span for span in finished}

    assert set(by_name) == {TOTAL}
    assert dict(by_name[TOTAL].attributes or {})["fraudlens.degraded"] is True
    assert dict(by_name[TOTAL].attributes or {})["fraudlens.action"] == "challenge"


def test_span_durations_agree_with_the_latency_report() -> None:
    """One measurement, two views. This is the property the module exists to guarantee.

    If the spans were timed independently they would drift from the histogram, and during
    an incident the trace and the SLO would disagree about the same request. Here the
    scripted clock says features took 10 ms and inference 20 ms, and the spans must say
    exactly that -- not approximately, because they are the same numbers.
    """
    # total start, features in/out, inference in/out, calibration in/out, total end.
    remaining = [0.0, 0.0, 0.010, 0.010, 0.030, 0.030, 0.035, 0.040]
    last = [0.0]

    def elapsed() -> float:
        if remaining:
            last[0] = remaining.pop(0)
        return last[0]

    exporter = InMemorySpanExporter()
    finished = _decide(exporter, scorer=StubScorer(0.01), elapsed=elapsed)
    durations_ms = {s.name: (s.end_time - s.start_time) / 1e6 for s in finished}  # type: ignore[operator]

    assert durations_ms[FEATURES] == pytest.approx(10.0, abs=0.01)
    assert durations_ms[INFERENCE] == pytest.approx(20.0, abs=0.01)
    assert durations_ms[CALIBRATION_POLICY] == pytest.approx(5.0, abs=0.01)


def test_stage_spans_are_laid_end_to_end_inside_the_parent() -> None:
    """Contiguity is an assumption, so it gets a test.

    The reconstruction only reproduces the real timeline because the three stages run back
    to back in `decisioning._scored` with nothing between them. If work is ever inserted
    between two stages, the spans become a plausible-looking fiction -- and the residual
    that should have shown up as untimed parent tail would instead be attributed to a
    stage that did not spend it.
    """
    exporter = InMemorySpanExporter()
    finished = _decide(exporter, scorer=StubScorer(0.01))
    by_name = {span.name: span for span in finished}

    ordered = [by_name[FEATURES], by_name[INFERENCE], by_name[CALIBRATION_POLICY]]
    for earlier, later in pairwise(ordered):
        assert earlier.end_time == later.start_time
    assert by_name[TOTAL].start_time is not None
    assert ordered[0].start_time >= by_name[TOTAL].start_time  # type: ignore[operator]
