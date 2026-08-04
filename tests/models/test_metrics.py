from __future__ import annotations

import numpy as np
import pytest

from fraudlens.models.metrics import (
    ModelMetrics,
    calibration_line,
    evaluate_scores,
    expected_calibration_error,
)


def _calibrated_sample(n: int = 20_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Scores that are true probabilities by construction."""
    rng = np.random.default_rng(seed)
    p = rng.beta(1.0, 20.0, size=n)
    y = (rng.random(n) < p).astype(np.int_)
    return y, p


def test_ece_near_zero_for_a_perfectly_calibrated_score() -> None:
    y, p = _calibrated_sample()
    assert expected_calibration_error(y, p) < 0.01


def test_ece_detects_a_prior_shifted_score() -> None:
    """The E1 failure mode in miniature: a strictly monotone odds inflation
    leaves ranking untouched and destroys calibration."""
    y, p = _calibrated_sample()
    odds = p / (1 - p)
    shifted = (odds * 20.0) / (odds * 20.0 + 1.0)
    assert expected_calibration_error(y, shifted) > 10 * expected_calibration_error(y, p)


def test_calibration_line_is_identity_for_a_calibrated_score() -> None:
    y, p = _calibrated_sample()
    slope, intercept = calibration_line(y, p)
    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.3)


def test_calibration_intercept_moves_with_a_prior_shift() -> None:
    y, p = _calibrated_sample()
    odds = p / (1 - p)
    shifted = (odds * 20.0) / (odds * 20.0 + 1.0)
    _, intercept = calibration_line(y, shifted)
    assert intercept < -2.0


def test_evaluate_scores_reports_injected_cost() -> None:
    y, p = _calibrated_sample(n=2_000)
    cost = np.full(len(y), 1.25)
    metrics = evaluate_scores(y, p, cost)
    assert isinstance(metrics, ModelMetrics)
    assert metrics.n == 2_000
    assert metrics.expected_cost_per_txn == pytest.approx(1.25)
    assert 0.0 <= metrics.auc <= 1.0
    assert set(metrics.as_mlflow_metrics()) == {
        "n",
        "auc",
        "pr_auc",
        "ece",
        "calibration_slope",
        "calibration_intercept",
        "brier",
        "expected_cost_per_txn",
    }


def test_evaluate_scores_rejects_misaligned_inputs() -> None:
    y, p = _calibrated_sample(n=100)
    with pytest.raises(ValueError, match="same length"):
        evaluate_scores(y, p, np.ones(99))


def test_evaluate_scores_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError, match="empty window"):
        evaluate_scores(np.array([], dtype=np.int_), np.array([]), np.array([]))
