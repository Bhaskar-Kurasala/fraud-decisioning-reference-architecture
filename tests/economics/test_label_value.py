"""Tests for the label-acquisition value model.

These lock the two structural facts the finding rests on — the maturity cliff at the
clean-window close, and the direction of the analyst-noise penalty — plus the arithmetic
of the break-even. No data files, so they run in the default gate.
"""

from __future__ import annotations

import math

import pytest

from fraudlens.config import SETTINGS
from fraudlens.economics.label_value import (
    break_even_events_per_year,
    days_to_detect_rate_shift,
    days_to_maturity_floor,
    label_acquisition_value,
    maturity_fraction,
)

TEST_FRAUD_RATE = 0.0348  # measured on the 92,427-row test window


def test_maturity_is_a_cliff_not_a_curve() -> None:
    """The finding's whole premise: nothing is knowable until the clean window closes."""
    day_before = maturity_fraction(SETTINGS.clean_window_days - 1, TEST_FRAUD_RATE)
    day_of = maturity_fraction(SETTINGS.clean_window_days, TEST_FRAUD_RATE)
    assert day_before < TEST_FRAUD_RATE  # only disclosed fraud, and not all of it
    assert day_of > 0.99
    assert maturity_fraction(0.0, TEST_FRAUD_RATE) == 0.0


def test_maturity_floor_is_the_clean_window_for_any_usable_floor() -> None:
    assert days_to_maturity_floor(TEST_FRAUD_RATE) == SETTINGS.clean_window_days
    # Below the fraud rate the fraud-disclosure curve alone can satisfy the floor, and the
    # answer is a lognormal quantile rather than the wall. Not an operating regime — it is
    # here so the branch is exercised and the closed form is checked against its median.
    assert days_to_maturity_floor(TEST_FRAUD_RATE, floor=TEST_FRAUD_RATE / 2) == pytest.approx(
        SETTINGS.chargeback_median_days
    )


def test_maturity_floor_rejects_impossible_floors() -> None:
    with pytest.raises(ValueError, match="maturity floor"):
        days_to_maturity_floor(TEST_FRAUD_RATE, floor=0.0)


def test_analyst_noise_only_slows_detection() -> None:
    """Symmetric misclassification attenuates the shift; it never reverses it."""
    perfect = days_to_detect_rate_shift(0.756, 0.626, 5.0, analyst_agreement=1.0)
    published = days_to_detect_rate_shift(0.756, 0.626, 5.0, analyst_agreement=SETTINGS.q_analyst)
    poor = days_to_detect_rate_shift(0.756, 0.626, 5.0, analyst_agreement=0.80)
    assert perfect < published < poor


def test_detection_scales_inversely_with_capacity() -> None:
    ten = days_to_detect_rate_shift(0.78, 0.65, 10.0)
    twenty = days_to_detect_rate_shift(0.78, 0.65, 20.0)
    assert twenty == pytest.approx(ten / 2)


def test_no_shift_is_never_detected() -> None:
    assert days_to_detect_rate_shift(0.6, 0.6, 60.0) == math.inf


@pytest.mark.parametrize("agreement", [0.5, 0.4, 1.5])
def test_uninformative_analysts_are_refused(agreement: float) -> None:
    with pytest.raises(ValueError, match="analyst_agreement"):
        days_to_detect_rate_shift(0.6, 0.5, 60.0, analyst_agreement=agreement)


def test_zero_capacity_is_refused() -> None:
    with pytest.raises(ValueError, match="cases_per_day"):
        days_to_detect_rate_shift(0.6, 0.5, 0.0)


def test_value_is_zero_when_detection_is_slower_than_chargebacks() -> None:
    """Negative days saved must not become a credit."""
    assert label_acquisition_value(-30.0, 4_358_096.0, 1.0) == 0.0
    assert break_even_events_per_year(14_666.6, -30.0, 4_358_096.0) == math.inf


def test_break_even_matches_the_published_arithmetic() -> None:
    """5 slots/day: $14,667/yr buys 57.4 days against the measured $4.36M/yr severity."""
    saved = SETTINGS.clean_window_days - days_to_detect_rate_shift(0.7562, 0.7562 - 0.13, 5.0)
    assert saved == pytest.approx(57.4, abs=0.1)
    break_even = break_even_events_per_year(14_666.6, saved, 4_358_096.0)
    assert break_even == pytest.approx(0.021, abs=1e-3)
