"""Calibration tests, including the golden ECE claim against the published artifacts.

The most important assertion in this file is the *consistency* one: the reliability
curve has to re-derive the bin assignment because `models.metrics` exports only the
scalar, and re-derivation is exactly what produced the two corrected numbers in
`docs/findings/fit-balanced-empirical-result.md`. So the duplication is pinned rather
than trusted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fraudlens.models.metrics import expected_calibration_error
from fraudlens.monitoring.calibration import calibration_report, reliability_curve

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


def _synthetic(n: int = 20_000, seed: int = 31) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.beta(0.35, 9.0, size=n)
    y = (rng.random(n) < p).astype(np.int_)
    return y, p


def test_the_reliability_curve_and_the_shared_ece_cannot_disagree() -> None:
    """The mass-weighted gaps of the curve must sum to `expected_calibration_error`.

    This is the guard on the one piece of duplicated binning in the package. If the
    curve's edges ever drift from the shared ECE's — right-closed instead of right-open,
    equal-count instead of quantile-edge, a different bin count — this fails, instead of
    a dashboard reporting a calibration figure that the promotion gate disagrees with.
    """
    y, p = _synthetic()
    curve = reliability_curve(y, p)
    total = sum(bin_.n for bin_ in curve)
    recomputed = sum(bin_.n / total * abs(bin_.gap) for bin_ in curve)
    assert recomputed == pytest.approx(expected_calibration_error(y, p), abs=1e-12)


def test_a_well_calibrated_score_reports_slope_one_and_intercept_zero() -> None:
    """Labels drawn from the score itself are calibrated by construction.

    Without this control, a slope near 1 on real data proves nothing — it could equally
    be a bug that always returns 1.
    """
    y, p = _synthetic(n=60_000)
    report = calibration_report(y, p)
    assert report.ece < 0.01
    assert report.slope == pytest.approx(1.0, abs=0.12)
    assert report.intercept == pytest.approx(0.0, abs=0.25)
    assert report.rate_ratio == pytest.approx(1.0, abs=0.1)


def test_a_prior_shifted_score_is_caught_by_ece_and_by_the_intercept() -> None:
    """The E1 failure: inflate every probability by a constant odds factor.

    AUC is unchanged — the transform is strictly monotone — so this is invisible to any
    ranking metric, which is the whole reason ECE is in the alerting tier.
    """
    y, p = _synthetic(n=40_000)
    odds_ratio = 27.39
    shifted = (p * odds_ratio) / (p * odds_ratio + (1 - p))
    report = calibration_report(y, shifted)
    assert report.ece > 0.15
    # Predicting far more fraud than occurs: the ratio of actual to predicted collapses.
    assert report.rate_ratio < 0.2
    assert report.intercept < -1.0


def test_calibration_refuses_a_single_class_window() -> None:
    """A window with no observed fraud is an immature-label symptom, not a calibrated one.

    Returning NaN here would put a gap on the dashboard that reads as an exporter glitch
    rather than as "the monitoring pipeline is reading data it should have refused".
    """
    p = np.linspace(0.01, 0.5, 500)
    y = np.zeros(500, dtype=np.int_)
    with pytest.raises(ValueError, match="single-class window"):
        calibration_report(y, p)


def test_empty_bins_are_omitted_rather_than_plotted_on_the_floor() -> None:
    """Ties collapse quantile edges everywhere at a 3.5% base rate.

    An emitted empty bin would plot at observed_rate 0.0 and read as a catastrophic
    over-prediction that never happened.
    """
    tied = np.concatenate([np.full(9_000, 0.01), np.linspace(0.02, 0.9, 1_000)])
    y = (np.arange(10_000) % 50 == 0).astype(np.int_)
    curve = reliability_curve(y, tied, n_bins=20)
    assert all(bin_.n > 0 for bin_ in curve)
    assert len(curve) < 20


@pytest.mark.golden
def test_the_refit_rebalanced_score_reproduces_the_published_ece() -> None:
    """Golden: `data/p_te_bal_fitted.npy` must still give ECE 0.1389.

    That figure is the measured half of findings §2 — the model 0.0021 AUC below champion
    that costs $4.36M/yr more. If this moves, either the artifact changed or the shared
    ECE definition did, and both invalidate the published penalty.
    """
    scored = DATA / "scored_test.parquet"
    balanced = DATA / "p_te_bal_fitted.npy"
    if not (scored.exists() and balanced.exists()):
        pytest.skip("data/ missing — regenerate with `bash run_all.sh`")

    import pandas as pd

    y = pd.read_parquet(scored, columns=["isFraud"])["isFraud"].to_numpy().astype(np.int_)
    p = np.load(balanced).astype(np.float64)
    report = calibration_report(y, p)
    assert report.ece == pytest.approx(0.1389, abs=5e-5)
    # Direction, not just magnitude: the rebalanced score predicts ~23% fraud on a 3.5%
    # population, so it is systematically high and the intercept must be strongly
    # negative. An ECE that matched with the wrong sign would be a different defect.
    assert report.intercept < -1.0
    assert report.rate_ratio < 0.35


@pytest.mark.golden
def test_the_champion_score_reproduces_the_published_ece() -> None:
    """Golden: the isotonic champion must still give ECE 0.0027."""
    scored = DATA / "scored_test.parquet"
    if not scored.exists():
        pytest.skip("data/ missing — regenerate with `bash run_all.sh`")

    import pandas as pd

    frame = pd.read_parquet(scored, columns=["isFraud", "p"])
    report = calibration_report(
        frame["isFraud"].to_numpy().astype(np.int_), frame["p"].to_numpy().astype(np.float64)
    )
    assert report.ece == pytest.approx(0.0027, abs=5e-5)
