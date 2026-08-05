"""The HTTP surface of the investigation view: wiring, status codes, and what it refuses.

`test_investigation.py` owns the arithmetic. This file owns the contract a case-management
tool integrates against, including the two answers that are easy to conflate — "no ledger
here" and "no such transaction" — and the deliberate absence of any queue semantics.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fraudlens.serving.cases import CaseSource, case_source
from fraudlens.serving.investigation import LedgerDecision
from tests.serving.conftest import StubScorer, make_client

STAMP = dt.datetime(2026, 8, 5, 14, 32, 10, tzinfo=dt.timezone.utc)

DECISION = LedgerDecision(
    transaction_id=2987000,
    transaction_at=STAMP,
    decided_at=STAMP,
    calibrated_probability=0.90,
    action="deny",
    reason_codes=("SCORE_ABOVE_DENY_BOUNDARY", "NEW_ACCOUNT_TENURE"),
    model_version="champion-v7",
    policy_version="ev-argmax-3action-v1+rules-ladder-v1",
    feature_version="request-supplied-v1",
    config_hash="deadbeef",
    input_hash="cafebabe",
    degraded=False,
    degraded_reason=None,
)


def client_with_ledger(*rows: LedgerDecision) -> TestClient:
    """A wired app whose case source is an in-memory ledger. No database, no network."""
    by_id = {row.transaction_id: row for row in rows}

    def source() -> CaseSource:
        return by_id.get

    client = make_client(scorer=StubScorer(0.01))
    client.app.dependency_overrides[case_source] = source  # type: ignore[attr-defined]
    return client


def test_unconfigured_instance_says_so_rather_than_pretending_the_case_is_missing() -> None:
    """503, not 404. A dispute handler who reads 404 concludes we never decided the
    transaction, which is a materially different and wrong answer."""
    with make_client(scorer=StubScorer(0.01)) as client:
        response = client.get("/v1/cases/2987000")
    assert response.status_code == 503
    assert "no decision ledger is wired" in response.json()["detail"]


def test_unknown_transaction_is_a_404() -> None:
    with client_with_ledger(DECISION) as client:
        assert client.get("/v1/cases/999").status_code == 404


def test_the_full_view_of_a_recorded_decision() -> None:
    with client_with_ledger(DECISION) as client:
        body = client.get(
            "/v1/cases/2987000", params={"amount": 250.0, "days_since_first_seen": 20.0}
        ).json()

    decision: dict[str, Any] = body["decision"]
    assert decision["action"] == "deny"
    assert decision["calibrated_probability"] == 0.90
    assert decision["reason_codes"] == ["SCORE_ABOVE_DENY_BOUNDARY", "NEW_ACCOUNT_TENURE"]
    assert decision["degraded"] is False
    # The whole version set, on every case: which decisions get disputed is not known
    # when they are made.
    assert set(decision["versions"]) == {
        "model_version",
        "policy_version",
        "feature_version",
        "config_hash",
        "input_hash",
    }
    assert body["economics"]["tenure_bucket"] == "8-30d"
    assert body["amount_counterfactual"]["flips"][0]["action_at_or_above"] == "deny"
    assert body["unavailable"] == [
        {"field": "attribution.contributions", "reason": body["attribution"]["unavailable_reason"]}
    ]


def test_the_counterfactual_statement_carries_both_numbers() -> None:
    """The prose exists so an operator can paste one line into a case note.

    It has to name the amount, the score, and the threshold that *actually* moved. The
    last clause is the one with history: the statement used to quote the published
    `allow_to_deny` break-even next to a flip driven by the challenge/deny boundary, which
    is a true number in a place where it reads as the operative one.
    """
    with client_with_ledger(DECISION) as client:
        body = client.get(
            "/v1/cases/2987000", params={"amount": 250.0, "days_since_first_seen": 20.0}
        ).json()
    statement = body["amount_counterfactual"]["statement"]
    flip = body["amount_counterfactual"]["flips"][0]
    assert f"${flip['amount']:,.2f}" in statement
    assert f"{flip['boundary_at_observed_amount']:.3f}" in statement
    assert f"{flip['boundary_at_flip_amount']:.3f}" in statement
    assert "challenge/deny threshold" in statement
    assert "would have been challenge rather than deny" in statement
    # The solve is exact, so the boundary at the flip amount is the score itself. Asserted
    # because it is the cheapest available check that the closed form is not merely
    # self-consistent: it ties the algebra back to the recorded probability.
    assert flip["boundary_at_flip_amount"] == pytest.approx(
        body["decision"]["calibrated_probability"], abs=1e-6
    )
    # And the operative threshold at the observed amount must sit below the score, or the
    # ledger's `deny` and this endpoint's arithmetic disagree about the same transaction.
    assert flip["boundary_at_observed_amount"] < body["decision"]["calibrated_probability"]


def test_degraded_case_returns_a_null_probability_and_no_substitute() -> None:
    degraded = LedgerDecision(
        transaction_id=1,
        transaction_at=STAMP,
        decided_at=STAMP,
        calibrated_probability=None,
        action="challenge",
        reason_codes=("DEGRADED_MODEL_UNAVAILABLE", "FALLBACK_RULE_DEFAULT_CHALLENGE"),
        model_version="unavailable",
        policy_version="ev-argmax-3action-v1+rules-ladder-v1",
        feature_version="request-supplied-v1",
        config_hash="deadbeef",
        input_hash="cafebabe",
        degraded=True,
        degraded_reason="DEGRADED_MODEL_UNAVAILABLE",
    )
    with client_with_ledger(degraded) as client:
        body = client.get("/v1/cases/1", params={"amount": 250.0}).json()
    assert body["decision"]["calibrated_probability"] is None
    assert body["amount_counterfactual"] is None
    assert body["tenure_counterfactual"] is None


def test_a_historical_review_row_renders_rather_than_500s() -> None:
    """This deployment never emits `review` (findings §5), but the ledger spans policy
    versions and a viewer that cannot read an old row is useless exactly when someone is
    investigating a policy change."""
    reviewed = LedgerDecision(
        transaction_id=7,
        transaction_at=STAMP,
        decided_at=STAMP,
        calibrated_probability=0.5,
        action="review",
        reason_codes=("SCORE_ABOVE_CHALLENGE_BOUNDARY",),
        model_version="champion-v6",
        policy_version="ev-argmax-4action-v1",
        feature_version="request-supplied-v1",
        config_hash="deadbeef",
        input_hash="cafebabe",
        degraded=False,
        degraded_reason=None,
    )
    with client_with_ledger(reviewed) as client:
        body = client.get("/v1/cases/7", params={"amount": 250.0}).json()
    assert body["decision"]["action"] == "review"
    # And the recomputation under today's three-action policy shows what changed.
    assert body["economics"]["recomputed_action"] != "review"
    assert body["economics"]["matches_ledger_action"] is False


def test_there_is_no_queue_surface() -> None:
    """E6 refused to emit queue-depth metrics and this endpoint refuses queue semantics,
    for the same measured reason: value-of-review is positive for 5 of 92,427 rows, so
    there is no work to assign, claim or age against an SLA."""
    with client_with_ledger(DECISION) as client:
        paths = set(client.get("/openapi.json").json()["paths"])
    assert "/v1/cases/{transaction_id}" in paths
    banned = ("queue", "assign", "claim", "next", "worklist", "sla")
    assert not [p for p in paths for word in banned if word in p.lower()]


def test_the_case_route_does_not_disturb_the_decision_path() -> None:
    with client_with_ledger(DECISION) as client:
        response = client.post(
            "/v1/decide",
            json={
                "transaction_id": 2987001,
                "transaction_at": "2026-08-05T14:32:10Z",
                "amount": 249.99,
                "days_since_first_seen": 120.0,
                "features": {"C1": 1.0},
            },
        )
    assert response.status_code == 200
    assert response.json()["action"] == "allow"
