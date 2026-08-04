"""The cost of being wrong, per transaction.

Two numbers decide everything downstream:

    L = amount * COGS + CB_FEE + OPS_DISPUTE      cost of approving fraud
    M = amount * MARGIN + relationship_cost(tenure)   cost of declining a good customer

L grows at the COGS rate (0.70) and M at the margin rate (0.30) plus a fixed
relationship term, so the break-even probability M/(L+M) *falls* as amount rises: we
must be 73% sure to decline a $20 order but only 37% sure on a $500 one. That inverts
the intuition most fraud teams operate on, and it is the reason a single global
threshold leaves $1.67M/yr on the table.

Pure functions over numpy arrays. No I/O, no state, no logging.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fraudlens.config import (
    SETTINGS,
    TENURE_EDGES,
    TENURE_LABELS,
    UNKNOWN_TENURE,
    BusinessConstants,
)

FloatArray = npt.NDArray[np.float64]
StrArray = npt.NDArray[np.str_]

# Wide enough for every label plus the sentinel; numpy fixed-width strings truncate
# silently, which would collapse two buckets into one and misprice them.
_LABEL_DTYPE = "<U16"


def tenure_bucket(days_since_first_seen: npt.ArrayLike) -> StrArray:
    """Bucket D1 (days since the card was first seen) into a tenure segment.

    Missing or out-of-range values return `UNKNOWN_TENURE` rather than raising: 41 of
    the 92,427 test-window transactions have no D1, and a fraud decision still has to
    be made for them. They are priced explicitly in `relationship_cost`.
    """
    d = np.asarray(days_since_first_seen, dtype=np.float64)
    # side="left" gives edges[i] < d <= edges[i+1], matching pd.cut's right-closed bins.
    idx = np.searchsorted(np.asarray(TENURE_EDGES[1:-1]), d, side="left")
    outside = np.isnan(d) | (d <= TENURE_EDGES[0]) | (d > TENURE_EDGES[-1])
    idx = np.where(outside, len(TENURE_LABELS), idx)
    table = np.asarray([*TENURE_LABELS, UNKNOWN_TENURE], dtype=_LABEL_DTYPE)
    return table[idx]


def relationship_cost(tenure: npt.ArrayLike, c: BusinessConstants = SETTINGS) -> FloatArray:
    """P(churn | declined) x residual LTV, per tenure bucket.

    This is the part of a false decline that is not the lost margin on the basket: the
    forward value of a customer who never comes back. It is the largest term in M for
    small baskets, and it is why the boundary is tenure-dependent at all.
    """
    priced = {label: c.p_churn_on_decline[label] * c.residual_ltv[label] for label in TENURE_LABELS}
    # Unknown tenure takes the median bucket cost. This is what `research/05_economics.py`
    # did (`ten.M_relationship.median()`) and it is preserved so the published figures
    # reproduce -- but it is a BUSINESS ASSUMPTION, not a technical default: it prices an
    # unidentified customer as an average one. The verification report §4.6 records this
    # path as inert; it is not (41 test rows), so a bucketing regression would be masked
    # here rather than raised. Monitor the unknown-tenure rate as a data-quality signal.
    priced[UNKNOWN_TENURE] = float(np.median(np.asarray(list(priced.values()))))

    labels = np.asarray(tenure, dtype=_LABEL_DTYPE)
    out = np.full(labels.shape, np.nan, dtype=np.float64)
    for label, value in priced.items():
        out[labels == label] = value
    if np.isnan(out).any():
        unknown = np.unique(labels[np.isnan(out)])
        raise ValueError(f"unpriceable tenure labels: {unknown.tolist()}")
    return out


def false_negative_cost(amount: npt.ArrayLike, c: BusinessConstants = SETTINGS) -> FloatArray:
    """L: what an undetected fraud costs. Goods shipped, plus fixed chargeback costs."""
    return np.asarray(amount, dtype=np.float64) * c.cogs + c.cb_fee + c.ops_dispute


def false_positive_cost(
    amount: npt.ArrayLike,
    tenure: npt.ArrayLike,
    c: BusinessConstants = SETTINGS,
) -> FloatArray:
    """M: what declining a good customer costs. Lost margin now, plus lost relationship."""
    return np.asarray(amount, dtype=np.float64) * c.margin + relationship_cost(tenure, c)


def break_even_probability(fn_cost: npt.ArrayLike, fp_cost: npt.ArrayLike) -> FloatArray:
    """The fraud probability at which allowing and denying cost the same: M / (L + M).

    Denying is worth it above this. It is a per-transaction quantity, not a threshold:
    on the test window it spans 0.369 ($500+ baskets) to 0.740 (1-7 day tenure).
    """
    fn = np.asarray(fn_cost, dtype=np.float64)
    fp = np.asarray(fp_cost, dtype=np.float64)
    return fp / (fn + fp)
