"""Drift tests, including the hand-computable PSI claim the zero-bin policy rests on."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from fraudlens.monitoring.baseline import capture_baseline, categorical_reference
from fraudlens.monitoring.drift import (
    PSI_ALERT_THRESHOLD,
    categorical_drift,
    numeric_drift,
    population_stability_index,
)

EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def test_psi_reproduces_a_hand_computed_value() -> None:
    """A real claim, not a smoke test: 50/50 -> 25/75 is 0.274653 by hand.

        (0.25 - 0.50) * ln(0.25/0.50) = -0.25 * -0.693147 = 0.173287
        (0.75 - 0.50) * ln(0.75/0.50) =  0.25 *  0.405465 = 0.101366
                                                    total = 0.274653

    Locking the arithmetic matters more here than in most places: PSI has several
    published variants (some use log base 10, some average the two directions) that
    differ by a constant factor, and the 0.25 threshold in §5.2 is only meaningful for
    this one.
    """
    reference = np.array([0.5, 0.5])
    actual = np.array([0.25, 0.75])
    assert population_stability_index(reference, actual) == pytest.approx(0.2746530, abs=1e-7)


def test_psi_is_zero_against_itself() -> None:
    reference = np.array([0.1, 0.2, 0.3, 0.4])
    assert population_stability_index(reference, reference) == pytest.approx(0.0)


def test_psi_is_symmetric() -> None:
    """The (a-r)ln(a/r) form is symmetric; the asymmetric KL variant is not.

    Asserted because an implementation that quietly drifted to the KL form would still
    look plausible on every other test while changing what the 0.25 threshold means.
    """
    a = np.array([0.1, 0.2, 0.3, 0.4])
    b = np.array([0.4, 0.3, 0.2, 0.1])
    assert population_stability_index(a, b) == pytest.approx(population_stability_index(b, a))


def test_one_emptied_decile_alerts_and_matches_the_documented_floor() -> None:
    """The zero-bin decision, asserted as a number.

    With a 1e-4 floor, a single wholly emptied decile contributes
    (1e-4 - 0.1) * ln(1e-4 / 0.1) = 0.690085 on its own — 2.8x the alert threshold. This
    is the property the policy was chosen for: a tenth of traffic vanishing from a value
    range is a breakage and must page, not degrade quietly into an undefined number.
    """
    reference = np.full(10, 0.1)
    actual = np.array([0.0] + [0.1 + 0.1 / 9] * 9)
    expected_empty_bin_term = (1e-4 - 0.1) * math.log(1e-4 / 0.1)
    assert expected_empty_bin_term == pytest.approx(0.690085, abs=1e-6)

    psi = population_stability_index(reference, actual)
    assert psi > PSI_ALERT_THRESHOLD
    assert psi >= expected_empty_bin_term


def test_emptied_decile_alerts_across_three_orders_of_magnitude_of_floor() -> None:
    """The floor's magnitude moves the severity, not the decision.

    ln(1/eps) scaling: 0.456 at 1e-3, 0.690 at 1e-4, 1.151 at 1e-6. All clear 0.25, which
    is what makes an admittedly arbitrary constant defensible.
    """
    for eps, expected in ((1e-3, 0.455912), (1e-4, 0.690085), (1e-6, 1.151281)):
        term = (eps - 0.1) * math.log(eps / 0.1)
        assert term == pytest.approx(expected, abs=1e-5)
        assert term > PSI_ALERT_THRESHOLD


def test_numeric_drift_is_quiet_on_a_resample_of_the_baseline() -> None:
    """A stationary population must not alert, or the alert is worthless.

    Ties are the reason this is not obvious: the score distribution is spiky near zero,
    quantile edges collapse, and an implementation that assumed uniform decile mass would
    report drift on data drawn from the baseline's own generator.
    """
    rng = np.random.default_rng(3)
    train = rng.beta(0.35, 9.0, size=40_000)
    baseline = capture_baseline(
        captured_at=EPOCH,
        window_start=EPOCH - dt.timedelta(days=120),
        window_end=EPOCH,
        n_rows=train.size,
        numeric={"p": train},
    )
    fresh = rng.beta(0.35, 9.0, size=20_000)
    result = numeric_drift(baseline.numeric["p"], fresh)
    assert not result.alerting
    assert result.psi < 0.02
    assert result.ks is not None and result.ks < 0.05


def test_numeric_drift_alerts_on_a_shifted_score_distribution() -> None:
    """The E1 failure in Tier 0 form: a rebalanced score inflates every probability.

    That shift is exactly what a prior-shifted model does, and it is visible with no
    labels at all — which is the entire argument for the Tier 0 tier existing.
    """
    rng = np.random.default_rng(5)
    train = rng.beta(0.35, 9.0, size=40_000)
    baseline = capture_baseline(
        captured_at=EPOCH,
        window_start=EPOCH - dt.timedelta(days=120),
        window_end=EPOCH,
        n_rows=train.size,
        numeric={"p": train},
    )
    odds_ratio = 27.39  # (1 - pi) / pi at the 3.522% training base rate
    raw = rng.beta(0.35, 9.0, size=20_000)
    shifted = (raw * odds_ratio) / (raw * odds_ratio + (1 - raw))
    result = numeric_drift(baseline.numeric["p"], shifted)
    assert result.alerting
    assert result.psi > 1.0


def test_nulls_move_the_null_rate_not_the_psi() -> None:
    """An upstream field going missing must not masquerade as a distribution shift.

    Folding nulls into the lowest bin is the common shortcut, and it makes a data-pipeline
    outage indistinguishable from a population change on the dashboard — two incidents
    with entirely different runbooks.
    """
    rng = np.random.default_rng(7)
    train = rng.beta(0.35, 9.0, size=40_000)
    baseline = capture_baseline(
        captured_at=EPOCH,
        window_start=EPOCH - dt.timedelta(days=120),
        window_end=EPOCH,
        n_rows=train.size,
        numeric={"p": train},
    )
    fresh = rng.beta(0.35, 9.0, size=20_000)
    holed = fresh.copy()
    holed[:4_000] = np.nan
    intact = numeric_drift(baseline.numeric["p"], fresh)
    broken = numeric_drift(baseline.numeric["p"], holed)
    assert broken.null_rate == pytest.approx(0.2)
    assert intact.null_rate == 0.0
    assert broken.psi == pytest.approx(intact.psi, abs=0.05)


def test_unseen_categorical_levels_carry_mass_into_psi() -> None:
    """Entity churn must alert, not be silently dropped."""
    reference = categorical_reference("card4", ["visa"] * 700 + ["mastercard"] * 300)
    churned = ["visa"] * 400 + ["mastercard"] * 200 + ["newnetwork"] * 400
    result = categorical_drift(reference, churned)
    assert result.unseen_level_rate == pytest.approx(0.4)
    assert result.alerting
    # KS is a statement about a CDF and an unordered level set has none; reporting 0.0
    # would read as "no drift" rather than "not a question you can ask".
    assert result.ks is None
