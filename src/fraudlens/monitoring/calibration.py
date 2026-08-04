"""Tier 2 calibration: ECE, the reliability curve, and the slope/intercept behind them.

Worth $4.36M/yr per findings §2 — a model 0.0021 AUC below champion, which no review
process would block, costs that much more once its probabilities are consumed by an EV
policy instead of merely ranked.

**ECE is not defined here.** It is `models.metrics.expected_calibration_error`, called.
That is load-bearing: this project has already had to correct two numbers that were
re-derived from a prose description rather than from the source (see
`docs/findings/fit-balanced-empirical-result.md` — 0.0035 against a published 0.0027,
caused by binning right-closed instead of right-open). A second ECE in the repo is the
same failure waiting to happen, so there is one.

The reliability curve does have to re-derive the bin assignment, because
`models.metrics` exposes the scalar and not the binning that produced it. That
duplication is the exact risk just described, so it is converted into a checked
invariant instead of a hope: `tests/monitoring/test_calibration.py` asserts that the
mass-weighted gaps of this curve sum to `expected_calibration_error` to machine
precision. If the two ever disagree the test fails rather than a dashboard lying.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fraudlens.models.metrics import (
    DEFAULT_ECE_BINS,
    calibration_line,
    expected_calibration_error,
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One point on the reliability curve."""

    lower: float
    upper: float
    n: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        return self.observed_rate - self.mean_predicted


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Everything §5 Tier 2 asks for about calibration on one window."""

    n: int
    ece: float
    slope: float
    intercept: float
    predicted_fraud_rate: float
    observed_fraud_rate: float
    bins: tuple[ReliabilityBin, ...]

    @property
    def rate_ratio(self) -> float:
        """Actual over predicted fraud rate — base-rate drift in one number.

        Reported as a ratio rather than a difference because the base rate is 3.5%: a
        0.01 absolute gap is a rounding error at 50% prevalence and a 29% error here.
        """
        if self.predicted_fraud_rate == 0.0:
            return float("nan")
        return self.observed_fraud_rate / self.predicted_fraud_rate


def reliability_curve(
    y_true: IntArray, y_prob: FloatArray, n_bins: int = DEFAULT_ECE_BINS
) -> tuple[ReliabilityBin, ...]:
    """Quantile-binned reliability curve, binned identically to the ECE it accompanies.

    Empty bins are omitted rather than emitted with a zero observed rate. A collapsed
    quantile edge (ties, which are everywhere at this base rate) would otherwise plot as
    a point sitting on the floor, which reads as a catastrophic over-prediction that
    never happened.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must be the same length")
    edges = np.quantile(y_prob, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = -1.0, 2.0
    bin_index = np.digitize(y_prob, edges[1:-1])

    curve: list[ReliabilityBin] = []
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            continue
        curve.append(
            ReliabilityBin(
                lower=float(edges[b]),
                upper=float(edges[b + 1]),
                n=count,
                mean_predicted=float(y_prob[mask].mean()),
                observed_rate=float(y_true[mask].mean()),
            )
        )
    return tuple(curve)


def calibration_report(
    y_true: IntArray, y_prob: FloatArray, n_bins: int = DEFAULT_ECE_BINS
) -> CalibrationReport:
    """The full calibration picture for one window.

    Raises on a single-class window rather than returning a slope of NaN. In this system
    a window with no observed fraud almost always means the labels have not matured, not
    that fraud stopped; `maturity.require_mature` is the check that should have caught it
    upstream, and a NaN quietly written to a gauge would look like an exporter glitch
    instead of a monitoring pipeline reading immature data.
    """
    if len(y_true) == 0:
        raise ValueError("cannot assess calibration on an empty window")
    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "calibration slope is undefined on a single-class window; "
            "this usually means the labels for this window have not matured"
        )
    slope, intercept = calibration_line(y_true, y_prob)
    return CalibrationReport(
        n=len(y_true),
        ece=expected_calibration_error(y_true, y_prob, n_bins),
        slope=slope,
        intercept=intercept,
        predicted_fraud_rate=float(y_prob.mean()),
        observed_fraud_rate=float(y_true.mean()),
        bins=reliability_curve(y_true, y_prob, n_bins),
    )
