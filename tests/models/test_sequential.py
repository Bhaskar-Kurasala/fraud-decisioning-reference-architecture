from __future__ import annotations

import math

import numpy as np
import pytest

from fraudlens.models.sequential import paired_cost_delta


def test_no_difference_leaves_zero_inside_the_interval() -> None:
    rng = np.random.default_rng(1)
    champion = rng.gamma(2.0, 3.0, size=50_000)
    challenger = champion + rng.normal(0.0, 1.0, size=50_000)
    result = paired_cost_delta(champion, challenger)
    assert result.ci_lower < 0.0 < result.ci_upper
    assert not result.challenger_is_cheaper


def test_a_large_saving_is_certified() -> None:
    rng = np.random.default_rng(2)
    champion = rng.gamma(2.0, 3.0, size=50_000)
    challenger = champion - 0.5
    result = paired_cost_delta(champion, challenger)
    assert result.mean_delta == pytest.approx(-0.5)
    assert result.challenger_is_cheaper


def test_the_confidence_sequence_is_wider_than_a_one_shot_interval() -> None:
    """The price of being allowed to look every time labels mature further.

    If this ever stops holding, the boundary has been replaced by a
    fixed-horizon interval and repeated evaluation is silently inflating the
    false-promotion rate.
    """
    rng = np.random.default_rng(3)
    champion = rng.gamma(2.0, 3.0, size=20_000)
    challenger = champion + rng.normal(0.0, 2.0, size=20_000)
    result = paired_cost_delta(champion, challenger, tuning_horizon=20_000)
    sequence_half_width = (result.ci_upper - result.ci_lower) / 2
    fixed_horizon_half_width = 1.96 * result.std_delta / math.sqrt(result.n)
    assert sequence_half_width > fixed_horizon_half_width


def test_pairing_is_enforced() -> None:
    with pytest.raises(ValueError, match="paired"):
        paired_cost_delta(np.ones(10), np.ones(9))


def test_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError, match="two paired"):
        paired_cost_delta(np.ones(1), np.ones(1))
    with pytest.raises(ValueError, match="alpha"):
        paired_cost_delta(np.ones(10), np.ones(10), alpha=1.5)
