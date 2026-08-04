from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from fraudlens.models.checks import (
    CheckStatus,
    GateInputs,
    GateThresholds,
    label_maturity_fraction,
)
from fraudlens.models.gate import evaluate_promotion
from fraudlens.models.metrics import ModelMetrics

_RNG = np.random.default_rng(11)
_N = 20_000


def _metrics(ece: float = 0.004, cost: float = 2.5) -> ModelMetrics:
    return ModelMetrics(
        n=_N,
        auc=0.90,
        pr_auc=0.52,
        ece=ece,
        calibration_slope=1.0,
        calibration_intercept=0.0,
        brier=0.022,
        expected_cost_per_txn=cost,
    )


def _costs(saving: float) -> tuple[np.ndarray, np.ndarray]:
    champion = _RNG.gamma(2.0, 1.0, size=_N)
    return champion, champion - saving


def _inputs(**overrides: object) -> GateInputs:
    champion_cost, challenger_cost = _costs(0.20)
    base = {
        "champion": _metrics(),
        "challenger": _metrics(ece=0.005, cost=2.3),
        "champion_cost": champion_cost,
        "challenger_cost": challenger_cost,
        "label_maturity": 0.95,
        "segments": {"amount_band": np.where(np.arange(_N) % 2 == 0, "low", "high")},
    }
    base.update(overrides)
    return GateInputs(**base)  # type: ignore[arg-type]


def _by_name(decision: object, name: str) -> object:
    return next(c for c in decision.checks if c.name == name)  # type: ignore[attr-defined]


def test_a_genuinely_better_challenger_is_promoted() -> None:
    decision = evaluate_promotion(_inputs())
    assert decision.promote, decision.summary()
    assert all(c.status is CheckStatus.PASSED for c in decision.checks)


def test_immature_labels_block_and_skip_everything_downstream() -> None:
    decision = evaluate_promotion(_inputs(label_maturity=0.42))
    assert not decision.promote
    assert _by_name(decision, "label_maturity").status is CheckStatus.FAILED
    for name in ("calibration", "expected_cost", "segment_guard"):
        assert _by_name(decision, name).status is CheckStatus.SKIPPED


def test_calibration_regression_blocks_even_when_cost_looks_better() -> None:
    """The E1 shape, isolated: cheaper on this window, badly miscalibrated."""
    decision = evaluate_promotion(_inputs(challenger=_metrics(ece=0.14, cost=2.3)))
    assert not decision.promote
    assert _by_name(decision, "calibration").status is CheckStatus.FAILED
    # The cost check still runs and still reports; a failing gate must show the
    # whole picture, not stop at the first objection.
    assert _by_name(decision, "expected_cost").status is CheckStatus.PASSED


def test_a_tie_on_cost_is_not_enough_to_promote() -> None:
    champion_cost, _ = _costs(0.0)
    decision = evaluate_promotion(
        _inputs(champion_cost=champion_cost, challenger_cost=champion_cost.copy())
    )
    assert not decision.promote
    assert _by_name(decision, "expected_cost").status is CheckStatus.FAILED


def test_segment_guard_blocks_an_aggregate_win_that_hides_a_regression() -> None:
    champion_cost = _RNG.gamma(2.0, 1.0, size=_N)
    band = np.where(np.arange(_N) % 2 == 0, "low", "high")
    challenger_cost = champion_cost - 0.60
    # One segment gets materially worse while the aggregate still improves.
    challenger_cost[band == "high"] = champion_cost[band == "high"] + 0.40
    decision = evaluate_promotion(
        _inputs(
            champion_cost=champion_cost,
            challenger_cost=challenger_cost,
            segments={"amount_band": band},
        )
    )
    assert not decision.promote
    assert _by_name(decision, "expected_cost").status is CheckStatus.PASSED
    assert _by_name(decision, "segment_guard").status is CheckStatus.FAILED


def test_segment_guard_is_skipped_and_reported_when_no_segments_given() -> None:
    decision = evaluate_promotion(_inputs(segments={}))
    assert decision.promote
    assert _by_name(decision, "segment_guard").status is CheckStatus.SKIPPED


def test_small_segments_cannot_veto_on_noise() -> None:
    champion_cost = _RNG.gamma(2.0, 1.0, size=_N)
    band = np.array(["bulk"] * _N, dtype=object)
    band[:50] = "rare"
    challenger_cost = champion_cost - 0.30
    challenger_cost[band == "rare"] = champion_cost[band == "rare"] + 50.0
    decision = evaluate_promotion(
        _inputs(
            champion_cost=champion_cost,
            challenger_cost=challenger_cost,
            segments={"amount_band": band},
        ),
        GateThresholds(segment_min_size=500),
    )
    assert _by_name(decision, "segment_guard").status is CheckStatus.PASSED


def test_misaligned_segments_fail_loudly() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        evaluate_promotion(_inputs(segments={"amount_band": np.array(["a", "b"])}))


def test_label_maturity_fraction_counts_closed_windows() -> None:
    as_of = datetime(2026, 8, 5, tzinfo=UTC)
    times = [as_of - timedelta(days=d) for d in (10, 100, 120, 200)]
    assert label_maturity_fraction(times, as_of, 90) == 0.75


def test_label_maturity_rejects_naive_datetimes() -> None:
    as_of = datetime(2026, 8, 5, tzinfo=UTC)
    with pytest.raises(ValueError, match="timezone-aware"):
        label_maturity_fraction([datetime(2026, 1, 1)], as_of, 90)  # noqa: DTZ001
