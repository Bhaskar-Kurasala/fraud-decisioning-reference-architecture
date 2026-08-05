"""Claims about shadow mode. The theme: shadow output cannot become a decision.

Not "is labelled as not a decision" — cannot become one. The tests below check that at the
level a query, a schema and a type each operate on, because each of those is a separate way
somebody could get it wrong six months from now.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from fraudlens.flywheel.shadow import (
    ShadowScore,
    ShadowScoreLog,
    create_shadow_table,
    shadow_scores,
)
from fraudlens.streaming.ledger import DecisionLedger, DecisionRecord
from fraudlens.streaming.migrate import migrate
from fraudlens.streaming.schema import decision_ledger

EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    migrate(eng)
    create_shadow_table(eng)
    try:
        yield eng
    finally:
        eng.dispose()


def _decision(transaction_id: int) -> DecisionRecord:
    return DecisionRecord(
        transaction_id=transaction_id,
        transaction_at=EPOCH,
        decided_at=EPOCH,
        score=0.4,
        calibrated_probability=0.04,
        action="allow",
        reason_codes=("LOW_RISK",),
        model_version="champion-v1",
        policy_version="p1",
        feature_version="f1",
        config_hash="c",
        input_hash="h",
    )


def _shadow(transaction_id: int, probability: float | None = 0.61) -> ShadowScore:
    return ShadowScore(
        transaction_id=transaction_id,
        challenger_version="challenger-v2",
        champion_version="champion-v1",
        scored_at=EPOCH,
        score=None if probability is None else 0.9,
        calibrated_probability=probability,
        failure_reason=None if probability is not None else "feature lookup timed out",
    )


def test_a_shadow_score_cannot_express_an_action() -> None:
    """The structural guarantee. There is no field to misread and no default to inherit."""
    fields = {f.name for f in dataclasses.fields(ShadowScore)}
    assert "action" not in fields
    assert not fields & {"action", "reason_codes", "decision", "degraded"}
    assert "action" not in shadow_scores.c


def test_shadow_scores_are_invisible_to_every_query_against_the_ledger(engine: Engine) -> None:
    """The query guarantee: a different table, so no filter can be forgotten."""
    ShadowScoreLog(engine).record_many([_shadow(i) for i in range(1, 51)])

    with engine.connect() as conn:
        assert conn.execute(select(decision_ledger)).mappings().all() == []
    assert DecisionLedger(engine).count() == 0


def test_a_shadow_score_does_not_displace_the_decision_on_the_same_transaction(
    engine: Engine,
) -> None:
    """Both exist, keyed the same way, and the ledger still reports one decision."""
    ledger = DecisionLedger(engine)
    ledger.record(_decision(1))
    ShadowScoreLog(engine).record_many([_shadow(1)])

    assert ledger.count() == 1
    recorded = ledger.get(1)
    assert recorded is not None
    assert recorded.model_version == "champion-v1"  # the champion decided, as it must
    assert recorded.calibrated_probability == 0.04  # not the challenger's 0.61


def test_a_failed_shadow_score_is_absent_not_zero(engine: Engine) -> None:
    """ADR-0003.

    A challenger that errors on the rows it finds hard is the most flattering possible
    bug: it scores the easy traffic, looks excellent, and promotes. Recording 0.0 would
    additionally make it look like a *confident* allow.
    """
    log = ShadowScoreLog(engine)
    log.record_many([_shadow(i) for i in range(1, 9)] + [_shadow(9, None), _shadow(10, None)])

    scored, attempted = log.coverage("challenger-v2")
    assert (scored, attempted) == (8, 10)
    failures = [s for s in log.scores_for("challenger-v2") if s.failure_reason is not None]
    assert all(s.calibrated_probability is None and s.score is None for s in failures)
    assert failures[0].failure_reason == "feature lookup timed out"


def test_a_value_without_a_reason_and_a_reason_without_a_value_are_both_rejected() -> None:
    with pytest.raises(ValueError, match="must say why"):
        ShadowScore(
            transaction_id=1,
            challenger_version="c2",
            champion_version="c1",
            scored_at=EPOCH,
            score=None,
            calibrated_probability=None,
        )
    with pytest.raises(ValueError, match="has no value"):
        ShadowScore(
            transaction_id=1,
            challenger_version="c2",
            champion_version="c1",
            scored_at=EPOCH,
            score=0.5,
            calibrated_probability=0.5,
            failure_reason="timed out",
        )


def test_the_database_enforces_the_pairing_too(engine: Engine) -> None:
    """Not just the dataclass. A backfill writing raw SQL is the population that forgets."""
    with pytest.raises(IntegrityError), engine.begin() as conn:
        conn.execute(
            shadow_scores.insert(),
            {
                "transaction_id": 1,
                "challenger_version": "c2",
                "champion_version": "c1",
                "scored_at": EPOCH,
                "score": 0.5,
                "calibrated_probability": None,
                "failure_reason": None,
            },
        )


def test_a_model_may_not_shadow_itself() -> None:
    """It produces a zero cost delta that reads as "no regression" rather than as a bug."""
    with pytest.raises(ValueError, match="cannot shadow itself"):
        ShadowScore(
            transaction_id=1,
            challenger_version="v1",
            champion_version="v1",
            scored_at=EPOCH,
            score=0.5,
            calibrated_probability=0.5,
        )


def test_two_challengers_can_shadow_the_same_traffic(engine: Engine) -> None:
    log = ShadowScoreLog(engine)
    second = dataclasses.replace(_shadow(1), challenger_version="challenger-v3")
    log.record_many([_shadow(1), second])
    assert log.coverage("challenger-v2") == (1, 1)
    assert log.coverage("challenger-v3") == (1, 1)


def test_failures_are_returned_rather_than_filtered_away(engine: Engine) -> None:
    """A caller that wants only usable rows must drop them itself, and thereby notice."""
    log = ShadowScoreLog(engine)
    log.record_many([_shadow(1), _shadow(2, None)])
    assert len(log.scores_for("challenger-v2")) == 2
