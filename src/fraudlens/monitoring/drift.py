"""Tier 0 drift: what moved in the inputs, measured before any label exists to prove it.

These are the only numbers available on day zero. ADR-0002 defines the error budget on
them for that reason — a conventional budget burns on 5xx while the model quietly costs
millions, because the loss that proves the model broke arrives 30-90 days after the
traffic that broke it.

Four measurements, each answering a different failure:

- **PSI** — the distribution moved. Alerts at 0.25 (§5.2).
- **KS** — the same question without binning, as a cross-check that a PSI move is not an
  artifact of where the decile edges happened to fall.
- **Null rate** — the leading cause of silent failure. An upstream field that stops being
  populated does not raise; it imputes, and the model scores confidently on a constant.
- **Unseen categorical level rate** — entity churn, and the one that a numeric-only drift
  suite misses entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fraudlens.monitoring.baseline import (
    CategoricalReference,
    NumericReference,
    bin_proportions,
)

FloatArray = npt.NDArray[np.float64]

# §5.2. The industry convention (<0.1 stable, 0.1-0.25 watch, >0.25 act) rather than a
# figure derived here; it is a page-the-human threshold, and inventing our own would mean
# nobody on call has intuition for it.
PSI_ALERT_THRESHOLD = 0.25

# --- The zero-bin policy -----------------------------------------------------------
#
# PSI is sum((a - r) * ln(a / r)), which is undefined the moment any bin empties. Every
# implementation has to choose, and the choice moves the number past the 0.25 threshold,
# so it is a business decision rather than a numerical detail.
#
# Chosen: floor both proportions at 1e-4 (one in ten thousand). Consequence, computed:
# one wholly emptied decile contributes (1e-4 - 0.1) * ln(1e-4 / 0.1) = 0.690 on its own
# — 2.8x the alert threshold. So a single vanished decile pages, which is the intent: a
# tenth of traffic disappearing from a value range is a breakage, not drift.
#
# The magnitude of the floor is admittedly arbitrary, but its effect on the *decision* is
# not. The contribution scales as ln(1/eps): at 1e-3 an emptied decile gives 0.456, at
# 1e-6 it gives 1.151. Every value across three orders of magnitude clears 0.25, so the
# alert fires either way and only the reported severity moves. That is the property that
# makes the choice defensible; the number is a signal, not an estimate.
#
# Rejected — merging empty bins into their neighbours. It is the more "correct" fix
# numerically and it is operationally backwards: the worse the shift, the more bins
# merge, the fewer terms the sum has, and the smaller the reported PSI. It suppresses
# precisely the event the alert exists for.
#
# Rejected — refusing to report when a bin empties. Sound in a research notebook,
# indefensible on call: it turns the most severe drift into a missing datapoint, and a
# gap on a Grafana panel reads as "the exporter is down", not "act now".
#
# Note the floor breaks the sum-to-one of both vectors by at most n_bins * eps = 1e-3.
# That is below the resolution of any decision taken on this number.
_PROPORTION_FLOOR = 1e-4

# The bucket unseen categorical levels are collected into so they carry mass through the
# PSI sum. Its reference proportion is zero by construction, so it is floored, and any
# meaningful arrival of new levels alerts.
UNSEEN_LEVEL = "\x00unseen"


@dataclass(frozen=True, slots=True)
class DriftResult:
    """One feature's Tier 0 picture on one window."""

    feature: str
    n: int
    psi: float
    # None for categoricals: KS is a statement about a CDF, and an unordered level set
    # has no CDF. Reporting 0.0 there would look like "no drift" rather than "not a
    # question you can ask".
    ks: float | None
    null_rate: float
    baseline_null_rate: float
    unseen_level_rate: float

    @property
    def alerting(self) -> bool:
        return self.psi > PSI_ALERT_THRESHOLD


def population_stability_index(reference: FloatArray, actual: FloatArray) -> float:
    """PSI between two proportion vectors, under the floor policy documented above."""
    if reference.shape != actual.shape:
        raise ValueError(f"proportion vectors differ in shape: {reference.shape} vs {actual.shape}")
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("proportions must be a non-empty 1-D vector")
    if np.any(reference < 0.0) or np.any(actual < 0.0):
        raise ValueError("proportions must be non-negative")
    ref = np.clip(reference, _PROPORTION_FLOOR, None)
    act = np.clip(actual, _PROPORTION_FLOOR, None)
    return float(np.sum((act - ref) * np.log(act / ref)))


def ks_statistic(reference: NumericReference, values: FloatArray) -> float:
    """Two-sample KS against the baseline's quantile grid.

    An indicator, not a test. The reference CDF is the stored 101-point grid, so the
    statistic is resolved to about 0.01 and no p-value is derivable from it. That is a
    deliberate trade: keeping the baseline a few kilobytes of summary rather than a copy
    of the training window is worth more than two decimal places on a number whose only
    use is to corroborate a PSI move.
    """
    finite = np.sort(values[~np.isnan(values)])
    if finite.size == 0:
        return 0.0
    grid = np.asarray(reference.quantiles, dtype=np.float64)
    levels = np.linspace(0.0, 1.0, grid.size)
    reference_cdf = np.interp(finite, grid, levels)
    upper = np.arange(1, finite.size + 1, dtype=np.float64) / finite.size
    lower = upper - 1.0 / finite.size
    return float(max(np.max(np.abs(upper - reference_cdf)), np.max(np.abs(lower - reference_cdf))))


def numeric_drift(reference: NumericReference, values: FloatArray) -> DriftResult:
    """Drift for one numeric feature against its baseline reference."""
    actual = bin_proportions(values, reference.bin_edges)
    return DriftResult(
        feature=reference.name,
        n=int(values.size),
        psi=population_stability_index(np.asarray(reference.proportions, dtype=np.float64), actual),
        ks=ks_statistic(reference, values),
        null_rate=float(np.isnan(values).mean()) if values.size else 0.0,
        baseline_null_rate=reference.null_rate,
        unseen_level_rate=0.0,
    )


def categorical_drift(reference: CategoricalReference, values: Sequence[str | None]) -> DriftResult:
    """Drift for one categorical feature, with unseen levels pooled rather than dropped.

    Dropping unseen levels is the common implementation and it is the wrong one here:
    entity churn — a new acquirer, a new device family, a renamed product code — shows up
    as levels that were not in training, and silently discarding them makes the remaining
    distribution look stable at exactly the moment the population changed underneath it.
    """
    present = [v for v in values if v is not None]
    total = len(present)
    counts: dict[str, int] = {}
    for value in present:
        counts[value] = counts.get(value, 0) + 1
    levels = (*sorted(reference.proportions), UNSEEN_LEVEL)
    unseen = sum(count for level, count in counts.items() if level not in reference.proportions)
    ref_vector = np.array([reference.proportions.get(lv, 0.0) for lv in levels], dtype=np.float64)
    act_vector = np.array(
        [
            (unseen if lv == UNSEEN_LEVEL else counts.get(lv, 0)) / total if total else 0.0
            for lv in levels
        ],
        dtype=np.float64,
    )
    return DriftResult(
        feature=reference.name,
        n=len(values),
        psi=population_stability_index(ref_vector, act_vector),
        ks=None,
        null_rate=1.0 - total / len(values) if values else 0.0,
        baseline_null_rate=reference.null_rate,
        unseen_level_rate=unseen / total if total else 0.0,
    )
