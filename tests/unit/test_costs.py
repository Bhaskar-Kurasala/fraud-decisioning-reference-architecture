"""Unit tests for the cost model. No data artifacts required."""

from __future__ import annotations

import numpy as np
import pytest

from fraudlens.config import SETTINGS, TENURE_LABELS, UNKNOWN_TENURE, BusinessConstants
from fraudlens.economics import (
    break_even_probability,
    false_negative_cost,
    false_positive_cost,
    relationship_cost,
    tenure_bucket,
)


def test_false_negative_cost_is_cogs_plus_fixed_fees() -> None:
    assert false_negative_cost(100.0) == pytest.approx(100 * 0.70 + 25.0 + 12.0)


def test_false_positive_cost_is_margin_plus_relationship() -> None:
    expected = 100 * 0.30 + 0.42 * 323.4342016540025
    assert false_positive_cost(100.0, "new(0d)") == pytest.approx(expected)


@pytest.mark.parametrize(
    ("d1", "label"),
    [
        (0.0, "new(0d)"),  # first-ever transaction, the highest-churn bucket
        (1.0, "1-7d"),
        (7.0, "1-7d"),  # right-closed: the boundary day belongs to the lower bucket
        (7.5, "8-30d"),
        (400.0, "181-400d"),
        (401.0, "400d+"),
        (np.nan, UNKNOWN_TENURE),
        (-5.0, UNKNOWN_TENURE),
    ],
)
def test_tenure_bucket_edges(d1: float, label: str) -> None:
    assert tenure_bucket([d1])[0] == label


def test_unknown_tenure_is_priced_at_the_median_bucket() -> None:
    # Not a defensive default: 41 test-window rows have no D1 and still get a decision.
    priced = [relationship_cost([label])[0] for label in TENURE_LABELS]
    assert relationship_cost([UNKNOWN_TENURE])[0] == pytest.approx(float(np.median(priced)))


def test_unpriceable_tenure_fails_loudly() -> None:
    # A silently defaulted relationship cost is a silently mispriced decline.
    with pytest.raises(ValueError, match="unpriceable"):
        relationship_cost(["9-99d"])


def test_break_even_falls_as_amount_rises() -> None:
    """The counter-intuitive core of the finding, asserted as a property."""
    amounts = np.array([10.0, 50.0, 200.0, 1000.0])
    fn = false_negative_cost(amounts)
    fp = false_positive_cost(amounts, np.full(amounts.shape, "new(0d)"))
    b = break_even_probability(fn, fp)
    assert np.all(np.diff(b) < 0)
    assert np.all((b > 0) & (b < 1))


def test_costs_are_monotone_in_amount() -> None:
    amounts = np.array([1.0, 10.0, 100.0, 10_000.0])
    assert np.all(np.diff(false_negative_cost(amounts)) > 0)
    tenures = np.full(amounts.shape, "400d+")
    assert np.all(np.diff(false_positive_cost(amounts, tenures)) > 0)


def test_constants_are_overridable_without_touching_the_defaults() -> None:
    """12-factor override for sensitivity analysis; defaults stay the audited ones."""
    sensitivity = BusinessConstants(cogs=0.60, margin=0.40)
    assert false_negative_cost(100.0, sensitivity) == pytest.approx(60 + 37)
    assert false_negative_cost(100.0) == pytest.approx(70 + 37)
    assert SETTINGS.cogs == 0.70


def test_margin_must_complement_cogs() -> None:
    with pytest.raises(ValueError, match=r"must equal 1\.0"):
        BusinessConstants(cogs=0.60)
