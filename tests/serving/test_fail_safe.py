"""The behaviour ADR-0002 exists to guarantee: on error we do not approve.

"Fail-safe means fail to the safe state, and for fraud the safe state is not 'approve'."
A silent default-approve during a model outage is the single worst failure this service
has, because it is invisible for 30-90 days -- until the chargebacks land.
"""

from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from fraudlens.serving.app import create_app
from fraudlens.serving.reasons import ReasonCode
from fraudlens.serving.runtime import utc_now
from tests.serving.conftest import RecordingWriter, StubScorer, make_client, request_body


@pytest.fixture(
    params=["model_never_loaded", "scoring_raises", "calibrator_out_of_range"],
    ids=lambda p: p,
)
def broken_client(request: pytest.FixtureRequest) -> TestClient:
    """Every way the scoring path can fail. All three must reach the same safe state."""
    if request.param == "model_never_loaded":
        return make_client(scorer=None)
    if request.param == "scoring_raises":
        return make_client(scorer=StubScorer(0.5, raises=True))
    # A calibrator returning 1.4 is broken, not confident. The cost model would price it
    # without complaint, which is exactly why it is caught rather than trusted.
    return make_client(scorer=StubScorer(1.4))


def test_degraded_decision_is_never_an_approve(broken_client: TestClient) -> None:
    """The load-bearing assertion of this whole service."""
    with broken_client as client:
        for amount in (5.0, 19.99, 100.0, 499.99, 500.0, 25_000.0):
            for tenure in (None, 0.0, 3.0, 45.0, 900.0):
                body = request_body(amount=amount, days_since_first_seen=tenure)
                payload = client.post("/v1/decide", json=body).json()
                assert payload["action"] != "allow", (amount, tenure, payload)
                assert payload["degraded"] is True


def test_degradation_is_marked_not_hidden(broken_client: TestClient) -> None:
    """A degraded decision must be excludable from analysis, so it must be identifiable."""
    with broken_client as client:
        payload = client.post("/v1/decide", json=request_body()).json()
    assert payload["degraded"] is True
    assert payload["degraded_reason"] in {
        ReasonCode.DEGRADED_MODEL_UNAVAILABLE.value,
        ReasonCode.DEGRADED_SCORING_ERROR.value,
    }
    # Null, not 0.0. A placeholder probability is how a fallback gets absorbed into a
    # calibration report and stops looking like an outage.
    assert payload["calibrated_probability"] is None


def test_degradation_reaches_the_ledger() -> None:
    """ADR-0002: marked degraded "in the ledger, so downstream analysis can exclude it"."""
    writer = RecordingWriter()
    with make_client(scorer=None, writer=writer) as client:
        client.post("/v1/decide", json=request_body())
    (row,) = writer.rows
    assert row["degraded"] is True
    assert row["degraded_reason"] == ReasonCode.DEGRADED_MODEL_UNAVAILABLE.value
    assert row["action"] != "allow"
    # Versioned even with no model: an unversioned decision cannot be replayed, and a
    # degraded decision is exactly the kind that gets disputed.
    assert row["model_version"]
    assert row["policy_version"]
    assert row["input_hash"]


def test_fallback_ladder_denies_only_the_documented_case() -> None:
    """Large basket on an account we have never seen denies; everything else challenges.

    Challenge is the default rung because it is friction the customer can clear. Denying
    everyone during an outage is safe for us and indefensible to a regulator.
    """
    with make_client(scorer=None) as client:
        big_new = client.post(
            "/v1/decide", json=request_body(amount=750.0, days_since_first_seen=2.0)
        ).json()
        big_established = client.post(
            "/v1/decide", json=request_body(amount=750.0, days_since_first_seen=400.0)
        ).json()
        small_new = client.post(
            "/v1/decide", json=request_body(amount=30.0, days_since_first_seen=0.0)
        ).json()
        big_unknown = client.post(
            "/v1/decide", json=request_body(amount=750.0, days_since_first_seen=None)
        ).json()

    assert big_new["action"] == "deny"
    # A missing tenure signal counts as new: we cannot show the account is established,
    # and in degraded mode the feature stage may be exactly what failed.
    assert big_unknown["action"] == "deny"
    assert big_established["action"] == "challenge"
    assert small_new["action"] == "challenge"


def test_a_malformed_request_is_not_a_degraded_decision() -> None:
    """422, not a fail-safe decline. Conflating the two hides caller bugs as outages."""
    with make_client(scorer=StubScorer(0.01)) as client:
        response = client.post("/v1/decide", json=request_body(amount=-1.0))
    assert response.status_code == 422
    assert "action" not in response.json()


def test_service_still_answers_while_reporting_itself_not_ready() -> None:
    """Readiness is a routing signal, not permission to answer.

    An instance with no model must keep serving fail-safe decisions -- refusing would turn
    a model outage into a checkout outage, which is the trade ADR-0002 declines to make.
    """
    with make_client(scorer=None) as client:
        assert client.get("/health/ready").status_code == 503
        assert client.post("/v1/decide", json=request_body()).status_code == 200


def test_a_failing_ledger_does_not_fail_the_decision() -> None:
    """The customer is waiting and the action is already chosen.

    The gap in the audit trail is an auditability incident and is logged as one, but
    turning it into a checkout failure trades priority #2 for priority #3 and gets neither.
    """

    class BrokenWriter:
        def record_decision(self, **kwargs: object) -> None:
            raise ConnectionError("ledger unreachable")

    with make_client(scorer=StubScorer(0.01), writer=BrokenWriter()) as client:
        response = client.post("/v1/decide", json=request_body())
    assert response.status_code == 200
    assert response.json()["action"] == "allow"


def test_an_unhandled_error_returns_503_rather_than_a_decision() -> None:
    """The last line of defence. `decide_transaction` already degrades safely, so reaching
    here means a bug in the service itself -- and §9a is explicit that a fraud system
    returning a default score is worse than one returning 503. The caller's timeout policy
    decides from there, rather than us inventing a decision from a state we do not
    understand. Simulated by breaking the injected clock, which runs outside the scored
    path exactly so that this branch is reachable in a test.
    """

    def broken_clock() -> dt.datetime:
        raise OSError("clock source unavailable")

    app = create_app(loader=lambda pin: StubScorer(0.01), clock=broken_clock, elapsed=lambda: 0.0)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/decide", json=request_body())
    assert response.status_code == 503
    assert "action" not in response.json()


def test_the_default_clock_is_timezone_aware_utc() -> None:
    """DTZ is on for this: a ledger timestamp with no offset cannot be audited across DST."""
    now = utc_now()
    assert now.tzinfo is not None
    assert now.utcoffset() == dt.timedelta(0)
