"""Expected and realised cost of each action, given a calibrated fraud probability.

Expected value is what we decide on (we only have p); realised cost is what we are
billed (it needs the label, which arrives 30-90 days later). Both live here so they
cannot drift apart -- an EV formula and a P&L formula that disagree is a system that
optimises one thing and pays for another.

All four EVs are negative (they are costs); the best action is the argmax.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fraudlens.config import SETTINGS, BusinessConstants
from fraudlens.economics.costs import FloatArray

# Row order is the action index used everywhere in this package. It is deliberately
# ordered least- to most-intrusive so that `argmax`'s first-wins tie-break resolves an
# exact EV tie toward the action that touches the customer least.
ALLOW, CHALLENGE, REVIEW, DENY = 0, 1, 2, 3
ACTION_NAMES: tuple[str, ...] = ("allow", "challenge", "review", "deny")

# The findings are quoted per year from a 32-day test window. BUSINESS ASSUMPTION: the
# window is representative of the year -- it is not (no Black Friday, one adversary
# regime, 3,213 fraud events), so annualised figures carry sampling error that no
# confidence interval is attached to in the published tables.
DAYS_PER_YEAR = 365.0


def action_expected_values(
    fraud_probability: npt.ArrayLike,
    fn_cost: npt.ArrayLike,
    fp_cost: npt.ArrayLike,
    c: BusinessConstants = SETTINGS,
) -> FloatArray:
    """(4, n) matrix of expected values, one row per action, in ACTION_NAMES order."""
    p = np.asarray(fraud_probability, dtype=np.float64)
    fn = np.asarray(fn_cost, dtype=np.float64)
    fp = np.asarray(fp_cost, dtype=np.float64)
    return np.vstack(
        [
            -p * fn,
            # A step-up challenge stops most fraud (f_pass survives) but costs the
            # abandonment of a fraction of good customers who will not do it.
            -(p * c.f_pass * fn + (1 - p) * c.a_abandon * fp),
            # An analyst is wrong (1 - q_analyst) of the time either way, and costs
            # handling + delay on every case whatever the outcome.
            -((1 - c.q_analyst) * (p * fn + (1 - p) * fp) + c.c_review + c.d_delay),
            -(1 - p) * fp,
        ]
    )


def realised_cost(
    actions: npt.ArrayLike,
    is_fraud: npt.ArrayLike,
    fn_cost: npt.ArrayLike,
    fp_cost: npt.ArrayLike,
    c: BusinessConstants = SETTINGS,
) -> FloatArray:
    """Cost actually incurred per transaction once the label is known.

    Only the allow and deny arms are true realisations. The challenge and review arms
    substitute the expected outcome (f_pass, a_abandon, q_analyst) because no
    transaction in this dataset was ever challenged or reviewed -- those two rows are
    counterfactual, and every P&L quoted from them inherits that.
    """
    a = np.asarray(actions)
    y = np.asarray(is_fraud, dtype=np.float64)
    fn = np.asarray(fn_cost, dtype=np.float64)
    fp = np.asarray(fp_cost, dtype=np.float64)
    ev = action_expected_values(y, fn, fp, c)
    # Substituting the realised label for p in the EV expressions is exactly the
    # research formula for every arm: at y in {0,1} each EV collapses to the branch
    # that actually happened. Keeping one formula removes the chance of the decision
    # maths and the P&L maths drifting apart.
    cost: FloatArray = -np.take_along_axis(ev, a.reshape(1, -1).astype(np.intp), axis=0)[0]
    return cost


def annualise(total_cost: float, days: int) -> float:
    """Scale a window total to a year. `days` is distinct days observed, not span."""
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")
    return total_cost * DAYS_PER_YEAR / days
