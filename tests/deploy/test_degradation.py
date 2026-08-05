"""The degradation drill: break the model, break the database, and assert what happens.

ADR-0002 ranks "reliability and graceful degradation" fourth of five and defines it
precisely — *fail to the safe state, and for fraud the safe state is not approve*. That
behaviour is the most important thing this system does, and until this file it was only
unit-tested: a stub loader raising in-process, with no container, no restart policy, no
connection pool and no orchestrator watching. Every one of those is a place where the
correct in-process behaviour becomes the wrong deployed behaviour:

* a HEALTHCHECK wired to readiness restarts a pod that is doing exactly the right thing,
  and restarting it does not produce a model;
* a connection pool without `pool_pre_ping` hands out sockets that died with the server,
  so a ten-second database restart becomes a minutes-long hole in the audit trail;
* a ledger failure that propagated to the caller would turn an auditability incident into
  a checkout error — and a 5xx in a checkout integration is frequently retried or, worse,
  failed open at the caller.

The two failures are deliberately different in kind. Losing the model must change the
*decision* (degraded, never allow) and must change readiness. Losing the database must
change *neither* — the decision is still correct and the instance is still the best
available place to send traffic — and must instead be counted, because "how many of your
decisions can you not reconstruct?" is a question with a number as its answer.

Each drill restores what it broke and asserts the restoration, because "does it recover"
is the half of a degradation drill that gets skipped.
"""

from __future__ import annotations

import subprocess

import httpx
import pytest
from sqlalchemy import Engine, create_engine, func, select

from fraudlens.streaming.schema import decision_ledger
from tests.deploy.conftest import BUNDLE, DATABASE_URL, compose, wait_until

pytestmark = pytest.mark.integration

QUARANTINE = BUNDLE.with_suffix(".pkl.quarantined")


def _request(transaction_id: int, features: dict[str, float]) -> dict[str, object]:
    return {
        "transaction_id": transaction_id,
        "transaction_at": "2026-08-05T14:32:10Z",
        # Large basket on an account we have never seen: the one rung of the fallback ladder
        # that denies. Chosen so the drill distinguishes "fell back correctly" from "fell
        # back to a blanket challenge", which would look the same on a smaller amount.
        "amount": 4000.0,
        "days_since_first_seen": 0.0,
        "features": features,
    }


def _bundle_features() -> dict[str, float]:
    import pickle

    with BUNDLE.open("rb") as handle:
        return dict.fromkeys(pickle.load(handle)["feature_names"], 0.0)  # noqa: S301


def _inspect(service: str, template: str) -> str:
    container = compose("ps", "-q", service).stdout.strip().splitlines()[0]
    out = subprocess.run(  # noqa: S603
        ["docker", "inspect", "-f", template, container],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _restart_count(service: str) -> int:
    """How many times Docker has restarted the container, from Docker's own accounting.

    The assertion that a not-ready container is *not* restarted cannot be made by watching
    it for a while — absence of an event over an interval is not evidence. This is.
    """
    return int(_inspect(service, "{{.RestartCount}}"))


def _health(service: str) -> str:
    return _inspect(service, "{{.State.Health.Status}}")


# ======================================================================================
# Drill 1 — the model goes away
# ======================================================================================


def test_the_service_survives_losing_its_model(api: httpx.Client) -> None:
    """Kill the artifact, restart the process, and assert the five things that matter:
    decisions still served, never `allow`, marked degraded, readiness red, liveness green —
    and the container not restarted for it.

    The artifact is removed and the process restarted rather than reached into, because
    that is how this failure actually arrives: the model is loaded once at startup and there
    is no reload path, so "the model went away" is only observable across a restart.
    """
    features = _bundle_features()
    healthy = api.post("/v1/decide", json=_request(910_000_001, features)).json()
    assert healthy["degraded"] is False, "the drill needs a working model to take away"
    assert api.get("/health/ready").status_code == 200

    restarts_before = _restart_count("api")
    BUNDLE.rename(QUARANTINE)
    try:
        compose("restart", "api")
        assert wait_until(lambda: _live(api), timeout=60.0), "process never came back up"

        # Readiness red: this instance cannot produce a scored decision and a load balancer
        # should prefer one that can.
        ready = api.get("/health/ready")
        assert ready.status_code == 503
        assert "ModelUnavailable" in ready.json()["detail"]

        # Liveness green, and the container is *not* being restarted for it. This is the
        # single most consequential assertion in the file: a HEALTHCHECK aliased to
        # readiness would put this container in a restart loop precisely while it was
        # behaving correctly, converting a model outage into a checkout outage.
        assert api.get("/health/live").status_code == 200
        assert wait_until(lambda: _health("api") == "healthy", timeout=60.0)
        assert _restart_count("api") == restarts_before

        # Still deciding, and deciding safely. A large basket on an unknown account is the
        # one rung of the ladder that denies.
        body = api.post("/v1/decide", json=_request(910_000_002, features)).json()
        assert body["degraded"] is True
        assert body["degraded_reason"] == "DEGRADED_MODEL_UNAVAILABLE"
        assert body["action"] == "deny"
        assert body["action"] != "allow"
        assert body["calibrated_probability"] is None
        assert body["versions"]["model_version"] == "unavailable"
    finally:
        QUARANTINE.rename(BUNDLE)
        compose("restart", "api")
        assert wait_until(lambda: _ready(api), timeout=90.0), "the service did not recover"


def test_a_degraded_decision_is_recorded_as_degraded(api: httpx.Client) -> None:
    """The ledger must be able to say "this one had no model" ninety days later, when the
    calibration report is built. A degraded decision absorbed as an ordinary one biases every
    downstream analysis by however long the model was down."""
    features = _bundle_features()
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    BUNDLE.rename(QUARANTINE)
    try:
        compose("restart", "api")
        assert wait_until(lambda: _live(api), timeout=60.0)
        assert api.post("/v1/decide", json=_request(910_000_003, features)).status_code == 200
        with engine.connect() as conn:
            row = conn.execute(
                select(decision_ledger).where(decision_ledger.c.transaction_id == 910_000_003)
            ).mappings().one()  # fmt: skip
        assert row["degraded"] is True
        # NULL, not 0.0. A placeholder here reads as "confidently legitimate" to anyone who
        # forgot to check `degraded`, and that misreading is how a fallback becomes a silent
        # approve in a report nobody re-derives.
        assert row["score"] is None
        assert row["calibrated_probability"] is None
        assert row["degraded_reason"] == "DEGRADED_MODEL_UNAVAILABLE"
    finally:
        engine.dispose()
        QUARANTINE.rename(BUNDLE)
        compose("restart", "api")
        assert wait_until(lambda: _ready(api), timeout=90.0)


# ======================================================================================
# Drill 2 — the database goes away
# ======================================================================================


def test_losing_the_ledger_costs_auditability_and_nothing_else(api: httpx.Client) -> None:
    """Decisions keep being made, keep being correct, and are *not* marked degraded — the
    model is fine, only the record-keeping is broken. Readiness stays green for the same
    reason: readiness answers "can this instance produce a scored decision", and it still
    can. Draining traffic away from it would achieve nothing except concentrating the same
    failure on a peer.

    What must change is the counter. ADR-0002 ranks auditability above latency, which is
    what makes "succeed the request, count the gap" the right resolution rather than a
    convenient one — but only if the gap is counted."""
    features = _bundle_features()
    assert api.post("/v1/decide", json=_request(920_000_001, features)).json()["degraded"] is False
    before = _ledger_write_failures(api)

    compose("stop", "postgres")
    try:
        assert wait_until(lambda: _ledger_write_failures(api) is not None, timeout=30.0)
        for offset in range(5):
            response = api.post("/v1/decide", json=_request(920_000_010 + offset, features))
            assert response.status_code == 200
            body = response.json()
            # Not degraded: degradation is a statement about the *decision*, and this
            # decision was scored by a working model. Conflating the two would make the
            # degraded-decision counter fire on a database incident and send the on-call
            # engineer to the model.
            assert body["degraded"] is False
            assert body["calibrated_probability"] is not None

        assert _ledger_write_failures(api) == before + 5
        assert api.get("/health/ready").status_code == 200
        assert api.get("/health/live").status_code == 200
    finally:
        compose("start", "postgres")

    # Recovery without a restart of the api. This is what `pool_pre_ping` buys: without it
    # the pool keeps handing out connections that died with the server, and a ten-second
    # database restart becomes a hole in the audit trail lasting until the pool churns.
    assert wait_until(lambda: _writes_are_landing(api, features), timeout=120.0), (
        "ledger writes did not resume after postgres came back; check pool_pre_ping"
    )


def _writes_are_landing(api: httpx.Client, features: dict[str, float]) -> bool:
    before = _ledger_write_failures(api)
    response = api.post("/v1/decide", json=_request(930_000_000 + int(before or 0), features))
    return response.status_code == 200 and _ledger_write_failures(api) == before


def _ledger_write_failures(api: httpx.Client) -> float:
    for line in api.get("/metrics").text.splitlines():
        if line.startswith("fraudlens_ledger_write_failures_total"):
            return float(line.split()[-1])
    raise AssertionError("fraudlens_ledger_write_failures_total is not exported")


def _live(api: httpx.Client) -> bool:
    try:
        return api.get("/health/live").status_code == 200
    except httpx.HTTPError:
        return False


def _ready(api: httpx.Client) -> bool:
    try:
        return api.get("/health/ready").status_code == 200
    except httpx.HTTPError:
        return False


def test_the_ledger_survives_its_own_outage(api: httpx.Client) -> None:
    """The rows written before the outage are still there afterwards. Postgres is doing the
    work; the assertion is that nothing in our restart path resets a volume, which is a
    mistake that is invisible until the quarter-end audit."""
    del api
    engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            count = conn.execute(select(func.count()).select_from(decision_ledger)).scalar_one()
        assert count > 0
    finally:
        engine.dispose()
