"""The request/response schema is a contract; a breaking change must fail CI.

These tests are deliberately literal. They enumerate field names and requiredness rather
than asserting "it round-trips", because the failure they exist to catch is someone
renaming or dropping a field that an integrator or the ledger depends on -- which a
round-trip test passes happily.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fraudlens.serving.contracts import DecideRequest, DecideResponse
from fraudlens.serving.decisioning import FEATURE_VERSION, POLICY_VERSION, SERVICE_VERSION
from tests.serving.conftest import (
    FROZEN_NOW,
    RecordingWriter,
    StubScorer,
    make_client,
    request_body,
)

# Frozen. Adding a field to either set is a contract change and must be a deliberate edit
# here; removing or renaming one breaks every integrator and every historical record.
REQUEST_FIELDS = {
    "transaction_id",
    "transaction_at",
    "amount",
    "days_since_first_seen",
    "features",
}
REQUIRED_REQUEST_FIELDS = {"transaction_id", "transaction_at", "amount", "features"}
RESPONSE_FIELDS = {
    "transaction_id",
    "action",
    "calibrated_probability",
    "reason_codes",
    "degraded",
    "degraded_reason",
    "decided_at",
    "versions",
    "latency",
}
VERSION_FIELDS = {
    "model_version",
    "policy_version",
    "feature_version",
    "config_hash",
    "service_version",
}
LATENCY_FIELDS = {
    "features_ms",
    "inference_ms",
    "calibration_policy_ms",
    "total_ms",
    "budget_breached",
}


def test_request_schema_is_frozen() -> None:
    assert set(DecideRequest.model_fields) == REQUEST_FIELDS
    required = {n for n, f in DecideRequest.model_fields.items() if f.is_required()}
    assert required == REQUIRED_REQUEST_FIELDS


def test_response_schema_is_frozen(client: TestClient) -> None:
    assert set(DecideResponse.model_fields) == RESPONSE_FIELDS
    payload = client.post("/v1/decide", json=request_body()).json()
    assert set(payload) == RESPONSE_FIELDS
    assert set(payload["versions"]) == VERSION_FIELDS
    assert set(payload["latency"]) == LATENCY_FIELDS


def test_the_action_vocabulary_is_the_three_action_policy() -> None:
    """`review` is not servable here: it is capacity-bound at 60 analyst slots/day and
    this path cannot check the queue inside a 150 ms budget. Emitting it would be an
    approve with extra steps."""
    action = DecideResponse.model_json_schema()["properties"]["action"]
    assert action["enum"] == ["allow", "challenge", "deny"]
    assert POLICY_VERSION == "ev-argmax-3action-v1+rules-ladder-v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amount", "249.99"),  # a string that looks numeric must not be coerced
        ("amount", 0.0),  # a zero-value basket is not a transaction
        ("amount", -5.0),
        ("transaction_id", -1),
        ("transaction_at", "2026-08-05T14:32:10"),  # naive: unauditable across a DST boundary
        ("days_since_first_seen", -3.0),
        ("features", {}),  # an empty vector cannot have produced a score
        ("features", {"C1": "1.0"}),
    ],
)
def test_invalid_input_is_rejected_at_the_boundary(
    client: TestClient, field: str, value: object
) -> None:
    """Validate on ingest (ADR-0002). The cheapest way to break decision correctness is to
    price a transaction from a field the caller did not mean."""
    assert client.post("/v1/decide", json=request_body(**{field: value})).status_code == 422


def test_unknown_fields_are_rejected_rather_than_ignored(client: TestClient) -> None:
    """Silently dropping `amount_cents` is how a $2.49 decision gets made on a $249 basket."""
    body = request_body()
    body["amount_cents"] = 24999
    assert client.post("/v1/decide", json=body).status_code == 422


def test_every_response_carries_the_full_version_set(client: TestClient) -> None:
    """ADR-0002 priority #2: which model, which policy, which config, for every decision --
    not only declines, because which decisions get disputed is unknown when they are made."""
    versions = client.post("/v1/decide", json=request_body()).json()["versions"]
    assert versions["model_version"] == "champion-v7"
    assert versions["policy_version"] == POLICY_VERSION
    assert versions["feature_version"] == FEATURE_VERSION
    assert versions["service_version"] == SERVICE_VERSION
    assert len(versions["config_hash"]) == 64  # sha256 hex


def test_the_model_version_pin_is_honoured() -> None:
    """Reproducibility: "latest" would mean replaying last Tuesday's traffic against this
    Tuesday's model and calling the difference drift."""
    with make_client(scorer=None, model_version_pin="champion-v5") as client:
        versions = client.post("/v1/decide", json=request_body()).json()["versions"]
    assert versions["model_version"] == "champion-v5"


def test_the_ledger_row_carries_every_field_needed_to_replay(client: TestClient) -> None:
    """The writer port's keyword names must stay 1:1 with the ledger's columns, so the
    adapter between them is a splat and never becomes a translation."""
    writer = RecordingWriter()
    with make_client(scorer=StubScorer(0.01), writer=writer) as wired:
        wired.post("/v1/decide", json=request_body())
    (row,) = writer.rows
    assert set(row) == {
        "transaction_id",
        "transaction_at",
        "decided_at",
        "score",
        "calibrated_probability",
        "action",
        "reason_codes",
        "model_version",
        "policy_version",
        "feature_version",
        "config_hash",
        "input_hash",
        "degraded",
        "degraded_reason",
    }
    assert row["decided_at"] == FROZEN_NOW
    assert row["decided_at"].tzinfo is not None


def test_the_openapi_document_is_usable_as_the_contract(client: TestClient) -> None:
    """ADR-0002 asks for well-documented APIs, not "read the source"."""
    schema = client.get("/openapi.json").json()
    assert set(schema["paths"]) >= {"/v1/decide", "/health/live", "/health/ready"}
    decide = schema["paths"]["/v1/decide"]["post"]
    assert {"200", "422"} <= set(decide["responses"])
    properties = schema["components"]["schemas"]["DecideRequest"]["properties"]
    # Every field documented: an integrator must not have to guess what D1 means.
    assert all("description" in prop for prop in properties.values())
