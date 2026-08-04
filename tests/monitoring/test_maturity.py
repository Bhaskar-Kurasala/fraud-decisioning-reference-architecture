"""The maturity gate, driven by the real dispute-lag simulation from `streaming.labels`.

The load-bearing test here is `test_the_final_twenty_days_are_refused`: it uses the same
`DisputeLagModel` the label revealer uses, so the refusal is measured against the actual
label-latency behaviour of the system rather than against a hand-made counterexample.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from fraudlens.monitoring.maturity import (
    ImmatureWindowError,
    label_maturity,
    require_mature,
)
from fraudlens.streaming.labels import DisputeLagModel

EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _simulate(
    n: int, window_days: int, offset_days: int, fraud_rate: float = 0.035
) -> tuple[list[dt.datetime], list[dt.datetime | None]]:
    """Transactions over a window, with reveal times from the production lag model."""
    lag = DisputeLagModel()
    rng = np.random.default_rng(19)
    is_fraud = rng.random(n) < fraud_rate
    times = [
        EPOCH + dt.timedelta(days=offset_days + window_days * i / n, seconds=1) for i in range(n)
    ]
    revealed = [
        t + dt.timedelta(days=lag.lag_days(i, is_fraud=bool(f)))
        for i, (t, f) in enumerate(zip(times, is_fraud, strict=True))
    ]
    return times, list(revealed)


def test_a_fully_aged_window_is_mature_and_passes() -> None:
    """After the 90-day clean-window close, everything is knowable.

    This is the control: if the gate refused here it would refuse forever and no Tier 2
    metric would ever be computable.
    """
    times, revealed = _simulate(2_000, window_days=20, offset_days=0)
    as_of = EPOCH + dt.timedelta(days=200)
    maturity = label_maturity(
        times, revealed, as_of=as_of, window_start=EPOCH, window_end=EPOCH + dt.timedelta(days=20)
    )
    # Not 1.0: the dispute lag is lognormal, so a couple of the 2,000 land past day 200.
    # That long tail is the finding, not a fixture artifact — it is why the floor is 0.80
    # rather than 1.0.
    assert maturity.ratio > 0.99
    require_mature(maturity, 0.80)


def test_the_final_twenty_days_are_refused() -> None:
    """Findings §6: the last 20 days before `as_of` are unusable, and this is why.

    A clean transaction is only confirmed clean when the 90-day dispute window closes, so
    a 20-day-old block has essentially no matured labels at all. Any AUC or ECE computed
    over it is a statement about which disputes landed fastest.
    """
    as_of = EPOCH + dt.timedelta(days=182)
    times, revealed = _simulate(2_000, window_days=20, offset_days=162)
    maturity = label_maturity(
        times,
        revealed,
        as_of=as_of,
        window_start=EPOCH + dt.timedelta(days=162),
        window_end=as_of,
    )
    assert maturity.ratio < 0.05
    with pytest.raises(ImmatureWindowError) as raised:
        require_mature(maturity, 0.80)
    # The refusal has to name the direction of the bias, not merely decline. An operator
    # who is told "immature" reruns it; one who is told which way it is wrong does not.
    assert "fraud-enriched" in str(raised.value)
    assert "6.9%" in str(raised.value)


def test_the_revealed_subset_is_fraud_enriched_partway_through_maturation() -> None:
    """The signable direction of the bias, asserted rather than asserted-in-a-comment.

    Fraud discloses at a 34-day median; clean only at the 90-day close. So partway
    through, the rows you *can* see are disproportionately fraud and the observed rate
    reads high — the model looks like it under-predicts, and a challenger that predicts
    higher looks better calibrated than it is.
    """
    lag = DisputeLagModel()
    rng = np.random.default_rng(23)
    n = 20_000
    is_fraud = rng.random(n) < 0.035
    horizon = 45.0
    revealed_fraud = sum(
        1 for i, f in enumerate(is_fraud) if f and lag.lag_days(i, is_fraud=True) <= horizon
    )
    revealed_total = revealed_fraud + sum(
        1 for i, f in enumerate(is_fraud) if not f and lag.lag_days(i, is_fraud=False) <= horizon
    )
    assert revealed_total > 0
    observed_rate = revealed_fraud / revealed_total
    true_rate = float(is_fraud.mean())
    assert observed_rate > true_rate * 5


def test_the_maturity_curve_excludes_rows_too_recent_for_the_horizon() -> None:
    """t+90 must not read as 0% merely because the window is 30 days old.

    Counting rows that have not had 90 days to mature as "unlabelled at t+90" would make
    every long horizon collapse toward zero on any recent window, which would then refuse
    windows that are in fact fine.
    """
    times, revealed = _simulate(1_000, window_days=10, offset_days=0)
    as_of = EPOCH + dt.timedelta(days=30)
    maturity = label_maturity(
        times, revealed, as_of=as_of, window_start=EPOCH, window_end=EPOCH + dt.timedelta(days=10)
    )
    by_horizon = {p.horizon_days: p for p in maturity.curve}
    assert by_horizon[7].n_eligible == 1_000
    assert by_horizon[90].n_eligible == 0
    assert np.isnan(by_horizon[90].ratio)


def test_a_label_revealed_after_as_of_is_not_counted() -> None:
    """Belt and braces against the leak the whole gate exists to prevent.

    `LabelRevealer.matured` already filters on `revealed_at <= as_of`; a caller that
    queried the table directly and forgot would otherwise inflate maturity with labels
    from the future, making an immature window certify itself.
    """
    times = [EPOCH + dt.timedelta(days=1)] * 10
    revealed: list[dt.datetime | None] = [EPOCH + dt.timedelta(days=100)] * 10
    maturity = label_maturity(
        times,
        revealed,
        as_of=EPOCH + dt.timedelta(days=50),
        window_start=EPOCH,
        window_end=EPOCH + dt.timedelta(days=2),
    )
    assert maturity.n_revealed == 0
    assert maturity.ratio == 0.0


def test_require_mature_rejects_a_nonsense_floor() -> None:
    times, revealed = _simulate(100, window_days=5, offset_days=0)
    maturity = label_maturity(
        times, revealed, as_of=EPOCH, window_start=EPOCH, window_end=EPOCH + dt.timedelta(days=5)
    )
    with pytest.raises(ValueError, match="maturity floor"):
        require_mature(maturity, 1.5)
