"""The individual promotion checks: what each one measures, and its threshold.

Separated from `gate` so that the ordering argument (which lives there) is not
buried inside the arithmetic. Each function is total — it always returns a
`CheckResult` rather than raising or returning a bare bool — because a failing
check is a normal outcome whose *reason* is the thing an auditor reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import numpy.typing as npt

from fraudlens.models.metrics import ModelMetrics
from fraudlens.models.sequential import DEFAULT_TUNING_HORIZON, paired_cost_delta

FloatArray = npt.NDArray[np.float64]


class CheckStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    reason: str
    detail: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GateThresholds:
    """Every number here is a business decision, not a tuning knob."""

    # 80% of the window's chargebacks must have had time to land. The hazard
    # curve has a 34-day median and a ~97-day tail; 80% is reached around day 60,
    # which is the earliest point at which the remaining tail cannot plausibly
    # reverse a cost comparison of the size we ship.
    min_label_maturity: float = 0.80
    label_maturity_days: int = 90

    # Absolute ECE regression allowed against the incumbent. The champion sits
    # at 0.0027; 0.01 leaves room for genuine sampling noise between windows
    # while sitting 14x below the 0.1389 that E1 showed costs $4.36M/yr.
    max_ece_regression: float = 0.01

    # The cost test must show the challenger *cheaper*, not merely not-worse:
    # promotion carries its own migration and revalidation cost, so a tie is a
    # reason to stay put.
    cost_alpha: float = 0.05
    cost_tuning_horizon: int = DEFAULT_TUNING_HORIZON

    # A segment may not get more expensive by more than 5% of its own incumbent
    # cost. Relative rather than absolute because segment costs span two orders
    # of magnitude (a <$25 order versus a $500+ one), and an absolute dollar
    # tolerance would be vacuous for one and unpassable for the other.
    segment_tolerance_fraction: float = 0.05
    # ...floored in dollars so a near-zero-cost segment cannot fail on noise.
    segment_tolerance_floor: float = 0.02
    # Below this count a segment mean is noise; excluded rather than allowed to
    # veto, and the exclusion is reported.
    segment_min_size: int = 500


@dataclass(frozen=True, slots=True)
class GateInputs:
    """Everything the gate needs, all of it supplied by the caller.

    `champion_cost` / `challenger_cost` are per-transaction realised costs from
    replaying both models through the same policy over the same transactions in
    the same order. They are injected because the cost functions live in the
    `economics` layer, which sits above this one.
    """

    champion: ModelMetrics
    challenger: ModelMetrics
    champion_cost: FloatArray
    challenger_cost: FloatArray
    label_maturity: float
    # e.g. {"amount_band": array_of_band_labels, "tenure": array_of_bucket_labels}
    segments: Mapping[str, Sequence[str]] = field(default_factory=dict)


def label_maturity_fraction(
    transaction_times: Sequence[datetime],
    as_of: datetime,
    maturity_days: int,
) -> float:
    """Fraction of the window whose outcome observation period has closed.

    Timezone-aware datetimes are required. A decision ledger with ambiguous
    timestamps cannot be reconciled across a DST boundary, and a maturity
    calculation that is an hour wrong at the boundary silently changes which
    transactions count as matured.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    if not transaction_times:
        raise ValueError("cannot compute maturity over an empty window")
    cutoff = as_of - timedelta(days=maturity_days)
    matured = 0
    for moment in transaction_times:
        if moment.tzinfo is None:
            raise ValueError("transaction times must be timezone-aware")
        if moment <= cutoff:
            matured += 1
    return matured / len(transaction_times)


def check_label_maturity(inputs: GateInputs, thresholds: GateThresholds) -> CheckResult:
    observed = inputs.label_maturity
    detail = {"label_maturity": observed, "threshold": thresholds.min_label_maturity}
    if observed < thresholds.min_label_maturity:
        return CheckResult(
            name="label_maturity",
            status=CheckStatus.FAILED,
            reason=(
                f"only {observed:.1%} of labels have matured, below the "
                f"{thresholds.min_label_maturity:.0%} required; a cost comparison on "
                "immature labels understates fraud and favours the more permissive model"
            ),
            detail=detail,
        )
    return CheckResult(
        name="label_maturity",
        status=CheckStatus.PASSED,
        reason=f"{observed:.1%} of labels matured over {thresholds.label_maturity_days}d",
        detail=detail,
    )


def check_calibration(inputs: GateInputs, thresholds: GateThresholds) -> CheckResult:
    regression = inputs.challenger.ece - inputs.champion.ece
    detail = {
        "champion_ece": inputs.champion.ece,
        "challenger_ece": inputs.challenger.ece,
        "ece_regression": regression,
        "tolerance": thresholds.max_ece_regression,
        "challenger_calibration_slope": inputs.challenger.calibration_slope,
        "challenger_calibration_intercept": inputs.challenger.calibration_intercept,
    }
    if regression > thresholds.max_ece_regression:
        return CheckResult(
            name="calibration",
            status=CheckStatus.FAILED,
            reason=(
                f"ECE regressed {inputs.champion.ece:.4f} -> {inputs.challenger.ece:.4f} "
                f"(+{regression:.4f}, tolerance {thresholds.max_ece_regression:.4f}); the "
                "policy layer consumes the probability, so this is a cost defect even when "
                f"ranking is unchanged (AUC {inputs.champion.auc:.4f} -> "
                f"{inputs.challenger.auc:.4f})"
            ),
            detail=detail,
        )
    return CheckResult(
        name="calibration",
        status=CheckStatus.PASSED,
        reason=f"ECE {inputs.champion.ece:.4f} -> {inputs.challenger.ece:.4f}",
        detail=detail,
    )


def check_expected_cost(inputs: GateInputs, thresholds: GateThresholds) -> CheckResult:
    delta = paired_cost_delta(
        inputs.champion_cost,
        inputs.challenger_cost,
        alpha=thresholds.cost_alpha,
        tuning_horizon=thresholds.cost_tuning_horizon,
    )
    detail = {
        "mean_cost_delta_per_txn": delta.mean_delta,
        "ci_lower": delta.ci_lower,
        "ci_upper": delta.ci_upper,
        "alpha": delta.alpha,
        "n_paired": float(delta.n),
    }
    if delta.challenger_is_cheaper:
        return CheckResult(
            name="expected_cost",
            status=CheckStatus.PASSED,
            reason=(
                f"challenger saves ${-delta.mean_delta:.4f}/txn "
                f"(95% confidence sequence [{delta.ci_lower:.4f}, {delta.ci_upper:.4f}])"
            ),
            detail=detail,
        )
    verb = "costs more" if delta.mean_delta > 0 else "is not certifiably cheaper"
    return CheckResult(
        name="expected_cost",
        status=CheckStatus.FAILED,
        reason=(
            f"challenger {verb}: {delta.mean_delta:+.4f} $/txn, confidence sequence "
            f"[{delta.ci_lower:.4f}, {delta.ci_upper:.4f}] does not exclude zero from above"
        ),
        detail=detail,
    )


def _segment_regressions(
    inputs: GateInputs, thresholds: GateThresholds
) -> tuple[dict[str, float], list[str]]:
    """Per-segment cost deltas, and the human-readable list of breaches."""
    detail: dict[str, float] = {}
    breaches: list[str] = []
    for dimension, labels in inputs.segments.items():
        levels = np.asarray(labels, dtype=object)
        if len(levels) != len(inputs.champion_cost):
            raise ValueError(f"segment '{dimension}' is not aligned with the cost arrays")
        for level in np.unique(levels):
            mask = levels == level
            count = int(mask.sum())
            if count < thresholds.segment_min_size:
                continue
            champion_mean = float(inputs.champion_cost[mask].mean())
            delta = float(inputs.challenger_cost[mask].mean()) - champion_mean
            key = f"{dimension}={level}"
            detail[f"segment_delta.{key}"] = delta
            tolerance = max(
                thresholds.segment_tolerance_fraction * abs(champion_mean),
                thresholds.segment_tolerance_floor,
            )
            if delta > tolerance:
                breaches.append(f"{key} +${delta:.4f}/txn (tolerance ${tolerance:.4f}, n={count})")
    return detail, breaches


def check_segments(inputs: GateInputs, thresholds: GateThresholds) -> CheckResult:
    if not inputs.segments:
        return CheckResult(
            name="segment_guard",
            status=CheckStatus.SKIPPED,
            reason="no segment definitions supplied",
        )
    detail, breaches = _segment_regressions(inputs, thresholds)
    if breaches:
        return CheckResult(
            name="segment_guard",
            status=CheckStatus.FAILED,
            reason=(
                f"{len(breaches)} segment(s) regressed beyond tolerance: "
                + "; ".join(sorted(breaches)[:5])
            ),
            detail=detail,
        )
    return CheckResult(
        name="segment_guard",
        status=CheckStatus.PASSED,
        reason=f"no segment regressed beyond tolerance across {len(inputs.segments)} dimension(s)",
        detail=detail,
    )
