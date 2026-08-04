"""The Tier 0/Tier 1 instruments the request path emits, and the ones it refuses to.

The refusals are the interesting half. This system's failure mode is silence -- labels are
30-90 days late, so a broken model looks like a working one for a quarter -- and a metric
that reads zero because nothing populated it is indistinguishable from a metric that reads
zero because everything is fine. Several tests below exist purely to assert that a number
we cannot honestly produce is *absent* rather than defaulted, which is the same choice the
ledger makes when it writes NULL instead of 0.0 for a degraded decision's score.
"""

from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

from tests.serving.conftest import RecordingWriter, StubScorer, make_client, request_body


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


def test_decision_value_is_recorded_in_dollars_not_only_in_counts() -> None:
    """GMV per arm, from the request path. Tier 0 business pulse and Tier 1 decline $.

    Counts alone cannot answer the question the business asks. Ten declines on $5 baskets
    and ten on $5,000 baskets are the same counter increment and a 1000x difference in
    what the policy just did.
    """
    before = _sample("fraudlens_decision_amount_usd_sum", {"action": "allow", "degraded": "false"})
    with make_client(scorer=StubScorer(0.001)) as client:
        body = request_body(amount=249.99)
        assert client.post("/v1/decide", json=body).json()["action"] == "allow"

    after = _sample("fraudlens_decision_amount_usd_sum", {"action": "allow", "degraded": "false"})
    # approx: the gauge is a running float sum, so the difference of two large partial sums
    # is not bit-exact. Asserting equality here would be testing IEEE 754, not the metric.
    assert after - before == pytest.approx(249.99)


def test_degraded_dollars_are_labelled_as_degraded() -> None:
    """ADR-0002 books the fail-safe arm's cost as "must be measured, not assumed".

    What makes that cost is dollars declined without a score, not decisions declined, so
    the `degraded` label is on the money and not only on the counter.
    """
    before = _sample("fraudlens_decision_amount_usd_sum", {"action": "deny", "degraded": "true"})
    with make_client() as client:  # no scorer: the load fails and every decision degrades
        # High amount on an unknown account is the one rung of the ladder that denies.
        body = request_body(amount=2400.0, days_since_first_seen=None)
        assert client.post("/v1/decide", json=body).json()["action"] == "deny"

    after = _sample("fraudlens_decision_amount_usd_sum", {"action": "deny", "degraded": "true"})
    assert after - before == pytest.approx(2400.0)


def test_a_degraded_decision_observes_no_probability() -> None:
    """The instrument that must stay empty during an outage.

    There is no calibrated probability on the fallback path. Observing 0.0 would drag the
    serving score distribution toward "confidently legitimate" at exactly the moment the
    model died -- the drift dashboard would go calm as the system broke. So the count does
    not move at all, and E7's PSI inherits an honest distribution rather than a padded one.
    """
    before = _sample("fraudlens_calibrated_probability_count")
    with make_client() as client:
        assert client.post("/v1/decide", json=request_body()).json()["degraded"] is True
    assert _sample("fraudlens_calibrated_probability_count") == before

    # And the scored path does move it, so the assertion above is about the degraded path
    # rather than about a metric that never records anything.
    with make_client(scorer=StubScorer(0.01)) as client:
        client.post("/v1/decide", json=request_body())
    assert _sample("fraudlens_calibrated_probability_count") == before + 1


def test_reason_codes_are_counted_by_code() -> None:
    """The closest thing the request path has to a per-feature null rate.

    `UNKNOWN_TENURE_PRICED_AS_MEDIAN` fires whenever `days_since_first_seen` is absent, and
    that field is the only input whose absence changes the *price* of the decision -- it
    selects the relationship-cost term. A jump in this series is the data-breakage alert.
    """
    label = {"code": "UNKNOWN_TENURE_PRICED_AS_MEDIAN"}
    before = _sample("fraudlens_reason_codes_total", label)
    with make_client(scorer=StubScorer(0.01)) as client:
        client.post("/v1/decide", json=request_body(days_since_first_seen=None))
    assert _sample("fraudlens_reason_codes_total", label) == before + 1


def test_a_rejected_request_is_counted_but_is_not_a_decision() -> None:
    """422 is the Tier 4 schema-violation rate, and it must not look like a decision.

    A malformed request produced no decision and wrote nothing to the ledger. Folding it
    into the decision counters would inflate throughput and, worse, would make a caller's
    integration bug read as a degradation of ours.
    """
    decisions_before = _sample(
        "fraudlens_decisions_total", {"action": "allow", "degraded": "false"}
    )
    label = {"route": "/v1/decide", "status": "422"}
    before = _sample("fraudlens_requests_total", label)

    with make_client(scorer=StubScorer(0.01)) as client:
        assert client.post("/v1/decide", json=request_body(amount="249.99")).status_code == 422

    assert _sample("fraudlens_requests_total", label) == before + 1
    assert (
        _sample("fraudlens_decisions_total", {"action": "allow", "degraded": "false"})
        == decisions_before
    )


def test_unmatched_paths_collapse_to_one_label() -> None:
    """Cardinality is a availability concern, not a tidiness one.

    Labelling by raw path lets an unauthenticated scanner walking /.env, /admin and a
    thousand friends mint a time series per probe and take the registry down with it.
    """
    before = _sample("fraudlens_requests_total", {"route": "unmatched", "status": "404"})
    with make_client(scorer=StubScorer(0.01)) as client:
        client.get("/.env")
        client.get("/admin/config")
    label = {"route": "unmatched", "status": "404"}
    assert _sample("fraudlens_requests_total", label) == before + 2


def test_a_failed_ledger_write_is_counted_not_only_logged() -> None:
    """An unpersisted decision is an auditability incident, not a request error.

    The customer got a correct decision; we simply cannot reconstruct it. "How many of
    your decisions can you not reproduce?" is a question a regulator asks and a log line
    cannot answer -- and the decision is still 200, so no error-rate panel would ever show
    it. This was previously logged and uncounted.
    """

    class ExplodingWriter(RecordingWriter):
        def record_decision(self, **kwargs: object) -> None:
            raise RuntimeError("ledger unreachable")

    before = _sample("fraudlens_ledger_write_failures_total")
    with make_client(scorer=StubScorer(0.01), writer=ExplodingWriter()) as client:
        assert client.post("/v1/decide", json=request_body()).status_code == 200
    assert _sample("fraudlens_ledger_write_failures_total") == before + 1


def test_model_load_is_reported_as_a_gauge_on_both_outcomes() -> None:
    """`fraudlens_model_loaded` is readiness in metric form, and zero is a real reading.

    A process serving fail-safe decisions is alive and answering; it is the gauge, not the
    liveness probe, that says the model is missing. Load time is recorded because a load
    slow enough to trip a start-up probe gets the pod killed and retried forever, which
    from outside is indistinguishable from a crash loop.
    """
    with make_client(scorer=StubScorer(0.01)):
        assert _sample("fraudlens_model_loaded") == 1.0
        assert _sample("fraudlens_model_load_seconds") >= 0.0

    with make_client():
        assert _sample("fraudlens_model_loaded") == 0.0
        # Zero, not the time-to-fail: there is no duration for work that did not complete.
        assert _sample("fraudlens_model_load_seconds") == 0.0


def test_the_exposition_carries_every_instrument_the_dashboards_query() -> None:
    """A cheap smoke test against the expensive failure: a panel reading "No data".

    tests/deploy/test_dashboards.py enforces the full correspondence between dashboards
    and exported names; this asserts the exposition endpoint actually serves them.
    """
    with make_client(scorer=StubScorer(0.01)) as client:
        client.post("/v1/decide", json=request_body())
        body = client.get("/metrics").text

    for name in (
        "fraudlens_decisions_total",
        "fraudlens_decision_amount_usd_bucket",
        "fraudlens_calibrated_probability_bucket",
        "fraudlens_reason_codes_total",
        "fraudlens_requests_total",
        "fraudlens_ledger_write_failures_total",
        "fraudlens_model_loaded",
        "fraudlens_model_load_seconds",
    ):
        assert name in body
