"""What to do with a transaction: pick the action with the highest expected value.

There is no threshold in this module in the usual sense. The decision is an argmax over
per-transaction expected values, and the "threshold" is a consequence of it -- a
different number for every transaction, derived in `boundaries()` for the cases where
an operator or a rules engine needs to see one.

Measured on the 32-day test window (see tests/golden):
    allow / challenge / deny        $2,799,214/yr
    allow / challenge / review / deny  $2,799,797/yr
Adding the third action to a binary policy is worth ~68% of the total gain; adding the
fourth (analyst review) is worth *less than nothing* at $7.97 a case.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fraudlens.config import SETTINGS, BusinessConstants
from fraudlens.economics import (
    REVIEW,
    FloatArray,
    action_expected_values,
    break_even_probability,
)

IntArray = npt.NDArray[np.int8]


def decide(
    fraud_probability: npt.ArrayLike,
    fn_cost: npt.ArrayLike,
    fp_cost: npt.ArrayLike,
    *,
    include_review: bool,
    c: BusinessConstants = SETTINGS,
) -> IntArray:
    """Per-transaction EV argmax. Returns action indices (see `ACTION_NAMES`).

    `include_review` has no default on purpose. The two variants are different policies
    with different published costs ($2,799,797 with review, $2,799,214 without), and a
    silent default is how those two numbers got conflated in the first place. Review is
    also capacity-bound in production (60 analyst slots/day) whereas this argmax is not,
    so a caller enabling it owes the queue a capacity check -- see `policy.queue`.

    Ties resolve to the lowest action index, i.e. the least intrusive action. Exact EV
    ties are reachable: the isotonic score takes only 153 distinct values.
    """
    ev = action_expected_values(fraud_probability, fn_cost, fp_cost, c)
    if not include_review:
        # -inf rather than deleting the row, so the returned indices stay on the single
        # action scale. The research script deleted the row and remapped index 2 -> 3
        # afterwards, which is correct but silently wrong the moment anyone reorders
        # the action list.
        ev = ev.copy()
        ev[REVIEW] = -np.inf
    actions: IntArray = ev.argmax(axis=0).astype(np.int8)
    return actions


def boundaries(
    fn_cost: npt.ArrayLike,
    fp_cost: npt.ArrayLike,
    c: BusinessConstants = SETTINGS,
) -> dict[str, FloatArray]:
    """The fraud probabilities where the EV argmax switches action, per transaction.

    Equating the EV expressions pairwise. These are the operator-facing form of the
    policy: the same decision as `decide`, expressed as the number a reviewer or a
    rules engine can read off. They are a distribution, not a threshold -- the binary
    allow/deny boundary alone spans 2.3x across the test window.
    """
    fn = np.asarray(fn_cost, dtype=np.float64)
    fp = np.asarray(fp_cost, dtype=np.float64)
    return {
        "allow_to_challenge": c.a_abandon * fp / (fn * (1 - c.f_pass) + c.a_abandon * fp),
        "challenge_to_deny": (1 - c.a_abandon) * fp / (c.f_pass * fn + (1 - c.a_abandon) * fp),
        "allow_to_deny": break_even_probability(fn, fp),
    }
