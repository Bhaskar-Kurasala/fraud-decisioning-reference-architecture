"""Selecting which transactions a finite analyst team looks at.

Why this is not a one-line `argsort`: the calibrated isotonic score takes only **153
distinct values across 92,427 transactions** (isotonic regression is a step function).
At an analyst budget of 1,920 cases the score at the cut is 0.41463 and **120
transactions share it exactly**. `np.argsort(-score)` picks an arbitrary subset of that
tied block -- the choice depends on introsort partition order, which varies with input
order, dtype, memory layout and numpy version. That is how the published review-queue
costs came to disagree with the pipeline's own log by $271 and $38 (verification report
§3.1); the two rankings that reproduced exactly were the two with no ties.

An adverse decision that depends on array memory layout is not auditable. So the tie
break is explicit and part of the contract, not an accident of the sort algorithm.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def rank_for_review(
    priority: npt.ArrayLike,
    transaction_id: npt.ArrayLike,
    capacity: int,
) -> npt.NDArray[np.intp]:
    """Positions of the top-`capacity` transactions by priority, highest first.

    Ties are broken by ascending TransactionID, which is monotone in transaction time
    on this dataset: when two cases are equally worth reviewing, the earlier one is
    reviewed. Stable, reproducible across environments, and defensible to a customer.
    """
    p = np.asarray(priority, dtype=np.float64)
    tid = np.asarray(transaction_id)
    if p.shape != tid.shape:
        raise ValueError(f"priority {p.shape} and transaction_id {tid.shape} must align")
    if capacity < 0:
        raise ValueError(f"capacity must be non-negative, got {capacity}")
    # lexsort's last key is primary: -priority ascending == priority descending, with
    # transaction_id ascending underneath it.
    order: npt.NDArray[np.intp] = np.lexsort((tid, -p))
    return order[:capacity]
