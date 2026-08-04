"""Evaluation metrics for a fraud score, with expected cost as the primary one.

Discrimination metrics (AUC, PR-AUC) answer "does the score rank fraud above
non-fraud". They are necessary and badly insufficient: E1 measured a model
0.0021 AUC below champion that costs $4.36M/yr more, because a four-action EV
policy consumes the *probability*, not the ranking. Every metric here except
AUC and PR-AUC is sensitive to that difference; expected cost is the one the
gate actually decides on.

Expected cost arrives as a per-transaction array supplied by the caller. This
layer sits below `economics`/`policy` in the dependency order and so cannot
compute it — which is the right shape anyway, because it lets the gate be
exercised against any cost definition without a policy simulator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

# Probabilities are clipped before the logit so a score of exactly 0 or 1 cannot
# produce an infinite covariate. 1e-6 is well below any threshold the policy
# layer uses, so the clip can never change a decision.
_LOGIT_CLIP = 1e-6

DEFAULT_ECE_BINS = 20


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """One model's performance on one evaluation window."""

    n: int
    auc: float
    pr_auc: float
    ece: float
    calibration_slope: float
    calibration_intercept: float
    brier: float
    expected_cost_per_txn: float

    def as_mlflow_metrics(self) -> dict[str, float]:
        """Flatten for `log_metric`. `n` is included: a metric without its
        sample size cannot be compared across windows."""
        return {key: float(value) for key, value in asdict(self).items()}


def expected_calibration_error(
    y_true: IntArray,
    y_prob: FloatArray,
    n_bins: int = DEFAULT_ECE_BINS,
) -> float:
    """Quantile-binned ECE.

    Quantile bins rather than equal-width bins because at a 3.5% base rate
    almost every transaction lands in the bottom equal-width bin, so equal-width
    ECE is dominated by one bucket and reports ~0 for a badly miscalibrated
    model. Quantile binning puts equal *mass* in each bin, which is what makes
    the 0.0027 vs 0.1389 separation visible.

    The binning matches `research/03b_calibrate.py` exactly — quantile edges with
    the outer bounds forced open, assigned right-open via `digitize`. That is not
    fussiness: ECE is sensitive to tie handling at this base rate, and E2 traced
    a published-number discrepancy to a reimplementation that binned
    right-closed instead. This function is intended to become the single
    definition the research scripts, the tests and the service all share.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    n = len(y_prob)
    edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -1.0, 2.0
    bin_index = np.digitize(y_prob, edges[1:-1])

    error = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            # Ties can collapse quantile edges; an empty bin contributes no mass
            # and is skipped rather than counted as perfectly calibrated.
            continue
        error += (count / n) * abs(float(y_true[mask].mean()) - float(y_prob[mask].mean()))
    return error


def calibration_line(y_true: IntArray, y_prob: FloatArray) -> tuple[float, float]:
    """Cox calibration: regress the outcome on the logit of the score.

    Slope 1 / intercept 0 is perfect. Slope < 1 means the score is overconfident
    (spread too wide); a non-zero intercept means calibration-in-the-large is
    off — the model is systematically high or low. A prior-shifted or
    class-rebalanced score shows up as a large positive intercept, which is
    precisely the failure ECE also catches, reported here in a form a model
    validator can read directly.
    """
    logit = np.log(np.clip(y_prob, _LOGIT_CLIP, 1 - _LOGIT_CLIP))
    logit -= np.log1p(-np.clip(y_prob, _LOGIT_CLIP, 1 - _LOGIT_CLIP))
    # penalty=None: any shrinkage would bias the slope towards zero and make a
    # miscalibrated model look better calibrated than it is.
    fitted = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    fitted.fit(logit.reshape(-1, 1), y_true)
    return float(fitted.coef_[0][0]), float(fitted.intercept_[0])


def evaluate_scores(
    y_true: IntArray,
    y_prob: FloatArray,
    realised_cost: FloatArray,
    n_bins: int = DEFAULT_ECE_BINS,
) -> ModelMetrics:
    """Compute the full metric set for one score on one window.

    `realised_cost` is the per-transaction cost the policy layer incurred when
    this score was routed through it, given the true label. It is injected
    rather than computed here because the cost functions live one layer up.
    """
    if not (len(y_true) == len(y_prob) == len(realised_cost)):
        raise ValueError("y_true, y_prob and realised_cost must be the same length")
    if len(y_true) == 0:
        raise ValueError("cannot evaluate an empty window")

    slope, intercept = calibration_line(y_true, y_prob)
    return ModelMetrics(
        n=len(y_true),
        auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        ece=expected_calibration_error(y_true, y_prob, n_bins),
        calibration_slope=slope,
        calibration_intercept=intercept,
        brier=float(brier_score_loss(y_true, y_prob)),
        expected_cost_per_txn=float(realised_cost.mean()),
    )
