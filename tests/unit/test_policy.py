"""Unit tests for expected value, the action argmax, and queue determinism."""

from __future__ import annotations

import numpy as np
import pytest

from fraudlens.config import BusinessConstants
from fraudlens.economics import (
    ALLOW,
    CHALLENGE,
    DENY,
    REVIEW,
    action_expected_values,
    annualise,
    realised_cost,
)
from fraudlens.policy import boundaries, decide, rank_for_review


def _costs(n: int, amount: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n, amount * 0.70 + 37.0), np.full(n, amount * 0.30 + 135.84)


def test_certain_fraud_is_denied_and_certain_good_is_allowed() -> None:
    fn, fp = _costs(2)
    actions = decide([1.0, 0.0], fn, fp, include_review=False)
    assert actions.tolist() == [DENY, ALLOW]


def test_review_is_never_chosen_when_it_is_excluded() -> None:
    fn, fp = _costs(101)
    p = np.linspace(0, 1, 101)
    assert REVIEW not in decide(p, fn, fp, include_review=False).tolist()


def test_review_variant_can_only_reduce_expected_cost() -> None:
    """More actions cannot make the argmax worse — a sanity check on the EV algebra.

    It is worth *realised* money that is another matter: on the test window the review
    arm is chosen 5 times and costs $583/yr more than not having it.
    """
    fn, fp = _costs(101)
    p = np.linspace(0, 1, 101)
    ev = action_expected_values(p, fn, fp)
    with_review = ev[decide(p, fn, fp, include_review=True), np.arange(101)]
    without = ev[decide(p, fn, fp, include_review=False), np.arange(101)]
    assert np.all(with_review >= without - 1e-12)


def test_ties_resolve_to_the_least_intrusive_action() -> None:
    """At an exact EV tie we do not touch the customer. 153 distinct isotonic values
    make exact ties reachable, so this is a decision, not a formality."""
    # A challenge that stops no fraud and loses every good customer is strictly
    # dominated, which isolates the allow/deny pair. With L == M, p = 0.5 is their exact
    # break-even: EV(allow) == EV(deny) == -50 in floating point, not merely close.
    dominated_challenge = BusinessConstants(f_pass=1.0, a_abandon=1.0)
    fn = np.array([100.0])
    fp = np.array([100.0])
    at_break_even = decide([0.5], fn, fp, include_review=False, c=dominated_challenge)
    assert at_break_even.tolist() == [ALLOW]


def test_boundaries_agree_with_the_argmax() -> None:
    """The operator-facing thresholds must describe the same policy the argmax runs."""
    fn, fp = _costs(1001)
    p = np.linspace(0, 1, 1001)
    b = boundaries(fn, fp)
    actions = decide(p, fn, fp, include_review=False)
    expected = np.where(
        p >= b["challenge_to_deny"],
        DENY,
        np.where(p >= b["allow_to_challenge"], CHALLENGE, ALLOW),
    )
    assert actions.tolist() == expected.tolist()


def test_realised_cost_matches_the_hand_written_arms() -> None:
    fn = np.array([100.0, 100.0, 100.0, 100.0])
    fp = np.array([200.0, 200.0, 200.0, 200.0])
    actions = np.array([ALLOW, CHALLENGE, REVIEW, DENY])
    fraud = realised_cost(actions, np.ones(4), fn, fp)
    good = realised_cost(actions, np.zeros(4), fn, fp)
    assert fraud.tolist() == pytest.approx([100.0, 0.11 * 100, 0.09 * 100 + 7.97, 0.0])
    assert good.tolist() == pytest.approx([0.0, 0.07 * 200, 0.09 * 200 + 7.97, 200.0])


def test_annualise_rejects_a_zero_length_window() -> None:
    assert annualise(1000.0, 32) == pytest.approx(1000.0 * 365 / 32)
    with pytest.raises(ValueError, match="days must be positive"):
        annualise(1000.0, 0)


def test_ranking_is_stable_across_repeated_runs_and_input_order() -> None:
    """The bug this guards: `np.argsort(-score)` over a step-function score picks an
    arbitrary subset of the tied block at the cut, which moved two published queue
    costs by $271 and $38. Ties here are dense on purpose."""
    rng = np.random.default_rng(0)
    ids = np.arange(1000, 2000)
    priority = np.round(rng.random(1000), 2)  # ~100 distinct values over 1,000 rows

    first = rank_for_review(priority, ids, 200)
    for _ in range(5):
        assert rank_for_review(priority, ids, 200).tolist() == first.tolist()

    # Same rows in a different order must yield the same *transactions*, since the
    # tie-break is on the identifier and not on position.
    shuffle = rng.permutation(1000)
    reordered = rank_for_review(priority[shuffle], ids[shuffle], 200)
    assert ids[shuffle][reordered].tolist() == ids[first].tolist()


def test_ranking_breaks_ties_by_ascending_transaction_id() -> None:
    ids = np.array([500, 100, 300])
    selected = rank_for_review(np.array([0.5, 0.5, 0.5]), ids, 3)
    assert ids[selected].tolist() == [100, 300, 500]


def test_ranking_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="must align"):
        rank_for_review(np.zeros(3), np.zeros(2), 1)
