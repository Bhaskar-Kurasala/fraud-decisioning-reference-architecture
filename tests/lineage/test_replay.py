"""The §9a replay claim, asserted rather than assumed.

"A decision in the ledger can be replayed: same inputs + same recorded model and policy
versions must produce the same output." Every test here is one clause of that sentence,
including the clauses about where it stops being true.
"""

from __future__ import annotations

import datetime as dt
import time

import pytest
from sqlalchemy import Engine

from fraudlens.config import BusinessConstants
from fraudlens.lineage.replay import POLICY_VARIANTS, replay_decision
from fraudlens.serving import decisioning
from fraudlens.serving.latency import StageTimings
from fraudlens.serving.runtime import CalibratedScorer
from fraudlens.streaming.ledger import DecisionLedger, DecisionRecord
from tests.lineage.conftest import EPOCH, FixedScorer, request_for


def _decide_and_record(
    engine: Engine,
    scorer: CalibratedScorer | None,
    amount: float,
    days_since_first_seen: float | None = 3.0,
) -> tuple[DecisionRecord, dict[str, object]]:
    """Run the real serving decision path and write it to the real ledger.

    Deliberately not a hand-built `DecisionRecord`: a fabricated row would let the replay
    agree with a decision the service never made, which is the failure this whole epic
    exists to rule out.
    """
    request = request_for(amount, days_since_first_seen)
    outcome = decisioning.decide_transaction(request, scorer, StageTimings(time.perf_counter))
    record = DecisionRecord(
        transaction_id=request.transaction_id,
        transaction_at=request.transaction_at,
        decided_at=EPOCH + dt.timedelta(seconds=1),
        score=outcome.raw_score,
        calibrated_probability=outcome.calibrated_probability,
        action=outcome.action,
        reason_codes=outcome.reason_codes,
        model_version=scorer.version if scorer is not None else "none",
        policy_version=decisioning.POLICY_VERSION,
        feature_version=decisioning.FEATURE_VERSION,
        config_hash=decisioning.CONFIG_HASH,
        input_hash=decisioning.input_hash(request),
        degraded=outcome.degraded,
        degraded_reason=outcome.degraded_reason,
    )
    assert DecisionLedger(engine).record(record)
    return record, request.model_dump(mode="json")


def test_recorded_decision_replays_to_the_same_action(
    engine: Engine, scorer: CalibratedScorer
) -> None:
    """The claim itself: same input, same recorded versions, same action."""
    record, payload = _decide_and_record(engine, scorer, amount=249.99)
    stored = DecisionLedger(engine).get(record.transaction_id)
    assert stored is not None

    result = replay_decision(stored, payload)

    assert result.verified, result.verdict
    assert result.replayed_action == stored.action
    assert result.input_matches
    assert result.config_matches
    assert result.verdict == f"reproduced: {stored.action}"


# Chosen to land on each arm of the three-action policy, and the two 0.9 rows to
# straddle a boundary that moves with basket size: at p=0.9 a $250 basket is challenged
# and a $1,500 one is declined. A replay that ignored the amount would agree with the
# ledger on one of those and not the other.
@pytest.mark.parametrize(
    ("probability", "amount", "expected"),
    [
        (0.001, 249.99, "allow"),
        (0.55, 249.99, "challenge"),
        (0.9, 249.99, "challenge"),
        (0.9, 1500.0, "deny"),
        (0.99, 20.0, "deny"),
    ],
)
def test_replay_holds_on_every_arm_of_the_policy(
    engine: Engine, probability: float, amount: float, expected: str
) -> None:
    """One action reproducing proves little; the claim has to hold across the action space."""
    record, payload = _decide_and_record(engine, FixedScorer(probability), amount=amount)

    assert record.action == expected
    assert replay_decision(record, payload).verified


def test_a_different_input_is_detected_not_silently_replayed(
    engine: Engine, scorer: CalibratedScorer
) -> None:
    """The ledger holds `input_hash`, not the features — this is what that buys."""
    record, payload = _decide_and_record(engine, scorer, amount=249.99)
    tampered = {**payload, "amount": 1500.0}

    result = replay_decision(record, tampered)

    assert not result.input_matches
    assert not result.verified
    assert "does not hash to the recorded input_hash" in result.verdict


def test_a_changed_config_is_a_different_provenance_and_a_different_decision(
    engine: Engine, scorer: CalibratedScorer
) -> None:
    """A business-constant change must be visible in the hash *and* in the outcome.

    `a_abandon` is the share of good customers lost to a step-up challenge. At the
    audited 0.07 a $1,500 basket at p=0.55 is worth challenging; at 0.9 the challenge
    loses the customer almost as surely as a decline does while still letting 11% of
    fraud through, so the same probability on the same basket becomes a decline. The
    config is a decision input in exactly the sense the features are, which is why
    `config_hash` is on the ledger row: a retroactive change to it is detectable.
    """
    record, payload = _decide_and_record(engine, scorer, amount=1500.0)
    high_abandonment = BusinessConstants(a_abandon=0.9)

    result = replay_decision(record, payload, c=high_abandonment)

    assert not result.config_matches
    assert result.replayed_action != record.action
    assert "config_hash differs" in result.verdict


def test_replaying_under_the_recorded_config_still_matches(
    engine: Engine, scorer: CalibratedScorer
) -> None:
    """The counterpart: an explicit config equal to the recorded one is not a diff."""
    record, payload = _decide_and_record(engine, scorer, amount=1500.0)
    result = replay_decision(record, payload, c=BusinessConstants())
    assert result.config_matches
    assert result.verified


@pytest.mark.parametrize(
    ("amount", "days_since_first_seen", "expected"),
    [
        (249.99, 3.0, "challenge"),  # default rung
        (1500.0, 2.0, "deny"),  # big basket, account we have never seen
        (1500.0, None, "deny"),  # missing tenure counts as new, not as established
        (1500.0, 400.0, "challenge"),  # big basket, but a relationship to protect
    ],
)
def test_a_degraded_decision_replays_through_the_same_ladder(
    engine: Engine, amount: float, days_since_first_seen: float | None, expected: str
) -> None:
    """Every rung of the fail-safe ladder, re-derived from the ledger row.

    This is the clause of §9a that used to be false. The ladder lived in
    `serving.decisioning`, above this layer in the import contract, so a degraded row was
    the one category of decision nobody could later prove was made correctly — which is
    exactly backwards, because an outage produces the largest single block of them and
    that block is the one a regulator would ask about first.

    Moving the ladder to `policy` closed it. The point of the parametrisation is that all
    four rungs are covered by re-running the real ladder, not by a copy of it: a
    reimplementation here would agree with itself and prove nothing.
    """
    record, payload = _decide_and_record(engine, None, amount, days_since_first_seen)
    assert record.degraded

    result = replay_decision(record, payload)

    assert result.replayed_action == expected
    assert result.verified
    assert result.unreplayable_reason is None
    # Never `allow`, on any input — the invariant the whole fail-safe path exists for.
    assert result.replayed_action != "allow"


def test_a_degraded_replay_is_not_verified_on_a_different_input(engine: Engine) -> None:
    """Same guard as the scored path: agreement on the wrong input is a coincidence.

    The ladder has only two outcomes, so two unrelated transactions agree by chance about
    half the time. Without the hash check a degraded replay would read as proof roughly
    every other attempt.
    """
    record, payload = _decide_and_record(engine, None, amount=1500.0, days_since_first_seen=2.0)

    result = replay_decision(record, {**payload, "amount": 1501.0})

    assert result.replayed_action == "deny"  # the action still agrees
    assert not result.input_matches
    assert not result.verified  # but nothing has been verified


def test_an_unknown_policy_version_is_refused(engine: Engine, scorer: CalibratedScorer) -> None:
    """A row from a policy this build never had must not be replayed under today's."""
    record, payload = _decide_and_record(engine, scorer, amount=249.99)
    from_the_future = DecisionRecord(
        **{**record.as_row(), "reason_codes": record.reason_codes, "policy_version": "ev-v99"}
    )

    result = replay_decision(from_the_future, payload)

    assert result.replayed_action is None
    assert "unknown policy_version" in str(result.unreplayable_reason)


def test_serving_policy_version_is_registered() -> None:
    """The replay table must describe the policy the service actually runs.

    Without this the table is a second, unverified opinion about what production does:
    change `INCLUDE_REVIEW` in serving without bumping `POLICY_VERSION` and every replay
    would keep reporting "reproduced" against arithmetic the service no longer uses.
    """
    assert POLICY_VARIANTS[decisioning.POLICY_VERSION] == decisioning.INCLUDE_REVIEW
