"""Per-stage timing and breach counting against design spec §4.3.

The elapsed-time source is injected, so these assert on a scripted clock rather than on a
sleep: a latency test that sleeps is slow, flaky, and tests the test runner's scheduler.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from prometheus_client import REGISTRY

from fraudlens.serving.latency import BUDGET_MS, CALIBRATION_POLICY, FEATURES, INFERENCE, TOTAL
from tests.serving.conftest import StubScorer, make_client, request_body


def _scripted(*seconds: float) -> Callable[[], float]:
    """An elapsed-time source that walks a fixed sequence, then holds the last value."""
    remaining = list(seconds)
    last = [0.0]

    def elapsed() -> float:
        if remaining:
            last[0] = remaining.pop(0)
        return last[0]

    return elapsed


def _breaches(stage: str) -> float:
    value = REGISTRY.get_sample_value("fraudlens_latency_budget_breaches_total", {"stage": stage})
    return value or 0.0


def test_budget_matches_the_spec_allocation() -> None:
    """§4.3 verbatim: features 60 ms, inference 25 ms, calibration+policy 15 ms, total 150 ms.

    Locked as a test because these numbers are an SLO, and an SLO that can be edited
    without anyone noticing is a preference.
    """
    assert BUDGET_MS == {FEATURES: 60.0, INFERENCE: 25.0, CALIBRATION_POLICY: 15.0, TOTAL: 150.0}
    # The three stages leave headroom inside the 150 ms for the 50 ms of overhead the spec
    # budgets for deserialisation, validation and the response write.
    assert BUDGET_MS[FEATURES] + BUDGET_MS[INFERENCE] + BUDGET_MS[CALIBRATION_POLICY] == 100.0


def test_each_stage_is_reported_separately() -> None:
    """ "The request took 400 ms" is not actionable; the allocation is what localises it."""
    # total start, features in/out, inference in/out, calibration in/out, total end.
    elapsed = _scripted(0.0, 0.0, 0.010, 0.010, 0.030, 0.030, 0.035, 0.040)
    with make_client(scorer=StubScorer(0.01), elapsed=elapsed) as client:
        latency = client.post("/v1/decide", json=request_body()).json()["latency"]

    # approx: the durations are differences of binary floats, so 0.030 - 0.010 is
    # 19.999999999999996 ms. Exact equality here would be testing IEEE 754, not the budget.
    assert latency["features_ms"] == pytest.approx(10.0)
    assert latency["inference_ms"] == pytest.approx(20.0)
    assert latency["calibration_policy_ms"] == pytest.approx(5.0)
    assert latency["total_ms"] == pytest.approx(40.0)
    assert latency["budget_breached"] is False


def test_a_breach_is_counted_not_merely_logged() -> None:
    """§4.3: "Breaches counted as an SLO metric, not merely logged."

    A log line is invisible to an error budget; a counter is what a burn-rate alert can be
    written against.
    """
    before = _breaches(INFERENCE)
    # Inference takes 90 ms against a 25 ms allocation; everything else is instant.
    elapsed = _scripted(0.0, 0.0, 0.0, 0.0, 0.090, 0.090, 0.090, 0.090)
    with make_client(scorer=StubScorer(0.01), elapsed=elapsed) as client:
        latency = client.post("/v1/decide", json=request_body()).json()["latency"]

    assert latency["inference_ms"] == pytest.approx(90.0)
    assert latency["budget_breached"] is True
    assert _breaches(INFERENCE) == before + 1.0


def test_a_within_budget_request_counts_no_breach() -> None:
    before = _breaches(TOTAL)
    with make_client(scorer=StubScorer(0.01), elapsed=_scripted(0.0)) as client:
        client.post("/v1/decide", json=request_body())
    assert _breaches(TOTAL) == before


def test_timings_are_recorded_even_when_scoring_fails() -> None:
    """A slow-failing dependency must not look free in the histogram.

    The features stage completed before inference raised, so its cost is real and must be
    reported; the stages that never ran report zero, which `degraded` disambiguates.
    """
    elapsed = _scripted(0.0, 0.0, 0.020, 0.020, 0.100, 0.100)
    with make_client(scorer=StubScorer(0.5, raises=True), elapsed=elapsed) as client:
        payload = client.post("/v1/decide", json=request_body()).json()

    assert payload["degraded"] is True
    assert payload["latency"]["features_ms"] == pytest.approx(20.0)
    assert payload["latency"]["inference_ms"] == pytest.approx(80.0)
    assert payload["latency"]["calibration_policy_ms"] == 0.0


def test_metrics_endpoint_exposes_the_slo_instruments() -> None:
    with make_client(scorer=StubScorer(0.01)) as client:
        client.post("/v1/decide", json=request_body())
        body = client.get("/metrics").text
    assert "fraudlens_decision_stage_seconds" in body
    assert "fraudlens_latency_budget_breaches_total" in body
    assert "fraudlens_decisions_total" in body
