"""The audit trail must be append-only and idempotent, or it is not evidence."""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DatabaseError, IntegrityError

from fraudlens.streaming.ledger import DecisionLedger, DecisionRecord

from .conftest import EPOCH


def make_decision(transaction_id: int = 3485113, **overrides: object) -> DecisionRecord:
    fields: dict[str, object] = {
        "transaction_id": transaction_id,
        "transaction_at": EPOCH,
        "decided_at": EPOCH + dt.timedelta(milliseconds=42),
        "score": 0.31,
        "calibrated_probability": 0.0068,
        "action": "allow",
        "reason_codes": ("LOW_RISK_SCORE", "TENURE_ESTABLISHED"),
        "model_version": "champion-2026.08.05",
        "policy_version": "ev-argmax-v1",
        "feature_version": "feats-2026.08.05",
        "config_hash": "c0ffee" * 6,
        "input_hash": "deadbeef" * 4,
    }
    fields.update(overrides)
    return DecisionRecord(**fields)  # type: ignore[arg-type]


def test_decision_round_trips_with_full_lineage(engine: Engine) -> None:
    ledger = DecisionLedger(engine)
    original = make_decision()

    assert ledger.record(original) is True

    stored = ledger.get(original.transaction_id)
    assert stored == original
    # Timestamps come back timezone-aware even from SQLite, which has no
    # timezone-aware type. Auditability depends on this, not on the engine.
    assert stored is not None
    assert stored.decided_at.tzinfo is not None


def test_replaying_the_same_transaction_does_not_double_write(engine: Engine) -> None:
    """At-least-once delivery is safe only because this holds."""
    ledger = DecisionLedger(engine)
    decision = make_decision()

    assert ledger.record(decision) is True
    assert ledger.record(decision) is False
    assert ledger.record(decision) is False
    assert ledger.count() == 1


def test_a_conflicting_rewrite_is_ignored_not_applied(engine: Engine) -> None:
    """A duplicate must not become an update; history is not rewritable."""
    ledger = DecisionLedger(engine)
    ledger.record(make_decision(action="allow"))

    assert ledger.record(make_decision(action="deny", calibrated_probability=0.99)) is False

    stored = ledger.get(3485113)
    assert stored is not None
    assert stored.action == "allow"


def test_batch_writes_skip_duplicates(engine: Engine) -> None:
    ledger = DecisionLedger(engine)
    first = [make_decision(i) for i in range(100, 110)]
    assert ledger.record_many(first) == 10

    overlapping = [make_decision(i) for i in range(105, 120)]
    assert ledger.record_many(overlapping) == 10
    assert ledger.count() == 20


@pytest.mark.parametrize(
    "verb",
    ["UPDATE decision_ledger SET action = 'deny'", "DELETE FROM decision_ledger"],
)
def test_the_database_refuses_to_mutate_the_ledger(engine: Engine, verb: str) -> None:
    """Enforced by trigger, not by application discipline (ADR-0002)."""
    DecisionLedger(engine).record(make_decision())

    with pytest.raises(DatabaseError, match="append-only"), engine.begin() as conn:
        conn.execute(text(verb))

    stored = DecisionLedger(engine).get(3485113)
    assert stored is not None and stored.action == "allow"


def test_unknown_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown action"):
        make_decision(action="escalate")


def test_probability_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="outside"):
        make_decision(calibrated_probability=1.4)


def test_naive_timestamps_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        make_decision(decided_at=dt.datetime(2026, 1, 1))  # noqa: DTZ001 — the point of the test


def test_degraded_decision_must_state_why() -> None:
    with pytest.raises(ValueError, match="degraded"):
        make_decision(degraded=True)


def test_degraded_decision_records_its_fallback_reason(engine: Engine) -> None:
    """ADR-0002 §4: degraded decisions must be excludable downstream on evidence."""
    ledger = DecisionLedger(engine)
    ledger.record(make_decision(degraded=True, degraded_reason="model_unavailable:rules_fallback"))

    stored = ledger.get(3485113)
    assert stored is not None
    assert stored.degraded is True
    assert stored.degraded_reason == "model_unavailable:rules_fallback"


def test_check_constraints_exist_in_the_database_not_only_in_python(engine: Engine) -> None:
    """Bypassing the dataclass must still fail."""
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO decision_ledger (transaction_id, transaction_at, decided_at,"
                " score, calibrated_probability, action, reason_codes, model_version,"
                " policy_version, feature_version, config_hash, input_hash, degraded)"
                " VALUES (1, '2026-01-01', '2026-01-01', 0.5, 7.0, 'allow', '[]',"
                " 'm', 'p', 'f', 'c', 'i', false)"
            )
        )
