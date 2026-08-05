"""The guarantee the promotion gate rests on, simulated rather than asserted by citation.

**The claim.** `paired_cost_delta` returns the Robbins normal-mixture confidence sequence
for the mean paired cost difference. Its guarantee is *time-uniform*:

    P( there exists any n >= 2 such that the true mean falls outside CI_n )  <=  alpha

— one alpha for the whole, unbounded sequence of looks, not one alpha per look. The
practical consequence is the reason it is here: the gate is re-run every time more
chargebacks land, which is continuously, and it may stop and promote the first time the
interval clears zero. No alpha-spending schedule, no pre-registered horizon, no penalty for
looking. A fixed-horizon 95% interval offers no such thing — it controls error at one
pre-chosen n, and re-reading it at every arrival inflates the false-promotion rate without
bound.

**Where the guarantee is weaker than the citation.** Howard et al. (2021) §3 state the
boundary for a known (or sub-Gaussian-bounded) scale parameter. `paired_cost_delta` plugs
in the sample standard deviation, so coverage is asymptotic rather than exact and finite-n
coverage depends on the tail of the cost distribution. That distribution is heavy — a
single high-amount chargeback dominates a window — so this is not a pedantic caveat. It is
tested on a heavy-tailed null below for that reason, and it is the honest limit of what can
be claimed: a nonparametric, exactly-valid version needs an empirical-Bernstein or
betting-style boundary, which this is not.

**What would make this test worth nothing.** If the boundary were quietly replaced by
`1.96 * s / sqrt(n)`, `test_a_fixed_horizon_interval_fails_the_same_simulation` is what
notices. It runs the identical peeking simulation through the t-interval and shows it
breaking, so the anytime-valid test cannot be passed by a fixed-horizon test wearing the
name.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from fraudlens.models.sequential import _mixture_half_width, paired_cost_delta

ALPHA = 0.05
PATHS = 400
HORIZON = 20_000
CHUNK = 100

# 60 looks, log-spaced. Log rather than linear because the boundary is loosest early — a
# linear grid would concentrate the looks where the sequence is already tight and would
# flatter it.
PEEKS = np.unique(np.geomspace(30, HORIZON, 60).astype(int))


def _running_mean_and_std(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running mean and sample std of each path, evaluated at every peek."""
    csum = np.cumsum(paths, axis=1)[:, PEEKS - 1]
    csq = np.cumsum(paths**2, axis=1)[:, PEEKS - 1]
    n = PEEKS.astype(np.float64)
    mean = csum / n
    var = np.maximum((csq - n * mean**2) / (n - 1.0), 0.0)
    return mean, np.sqrt(var)


def _ever_excludes_zero(draw, *, sequential: bool) -> float:  # type: ignore[no-untyped-def]
    """Fraction of null paths where the interval excludes zero at *any* look."""
    n = PEEKS.astype(np.float64)
    excluded = 0
    for _ in range(0, PATHS, CHUNK):
        paths = draw(CHUNK, HORIZON)
        mean, std = _running_mean_and_std(paths)
        if sequential:
            radius = np.array(
                [
                    [
                        _mixture_half_width(s, int(k), ALPHA, HORIZON)
                        for s, k in zip(row, PEEKS, strict=True)
                    ]
                    for row in std
                ]
            )
        else:
            radius = 1.96 * std / np.sqrt(n)
        excluded += int(np.any(np.abs(mean) > radius, axis=1).sum())
    return excluded / PATHS


def _gaussian_null(rng: np.random.Generator):  # type: ignore[no-untyped-def]
    def draw(rows: int, cols: int) -> np.ndarray:
        return np.asarray(rng.normal(0.0, 1.0, size=(rows, cols)))

    return draw


def _heavy_tailed_null(rng: np.random.Generator):  # type: ignore[no-untyped-def]
    """Two gammas differenced: skewed, fat, and zero-mean. The shape a cost delta has."""

    def draw(rows: int, cols: int) -> np.ndarray:
        left = rng.gamma(0.4, 60.0, size=(rows, cols))
        right = rng.gamma(0.4, 60.0, size=(rows, cols))
        return np.asarray(left - right)

    return draw


@pytest.mark.slow
def test_the_sequence_holds_alpha_under_continuous_peeking() -> None:
    """The claim, on a well-behaved null: 60 looks, one alpha for all of them."""
    rng = np.random.default_rng(7)
    rate = _ever_excludes_zero(_gaussian_null(rng), sequential=True)
    assert rate <= ALPHA, f"time-uniform coverage violated: {rate:.3f} > {ALPHA}"


@pytest.mark.slow
def test_the_sequence_holds_alpha_on_a_heavy_tailed_cost_delta() -> None:
    """The case that matters: the plug-in variance is where this could fail in practice."""
    rng = np.random.default_rng(11)
    rate = _ever_excludes_zero(_heavy_tailed_null(rng), sequential=True)
    assert rate <= ALPHA, f"time-uniform coverage violated on a heavy tail: {rate:.3f}"


@pytest.mark.slow
def test_a_fixed_horizon_interval_fails_the_same_simulation() -> None:
    """The control. Without it, "anytime-valid" is an unfalsifiable label on a t-test.

    The same null, the same looks, read through a 95% t-interval: the false-rejection rate
    is several times alpha. That gap is the entire reason §4.2 specifies a sequential test,
    and it is what the gate would be paying if it peeked at a fixed-horizon interval —
    which it necessarily would, because labels arrive continuously.
    """
    rng = np.random.default_rng(7)
    rate = _ever_excludes_zero(_gaussian_null(rng), sequential=False)
    assert rate > 3 * ALPHA, (
        f"a fixed-horizon interval peeked at {len(PEEKS)} times should over-reject badly; "
        f"got {rate:.3f}. If this passes, the boundary under test may not be sequential."
    )


def test_the_boundary_is_not_a_rescaled_t_interval() -> None:
    """Structural check, no simulation: the width has the wrong shape to be one.

    A fixed-horizon half-width is 1.96 * s / sqrt(n), so `half_width * sqrt(n) / s` is the
    constant 1.96 at every n. For the mixture boundary it is not constant, and two separate
    things are true about it:

    - It exceeds 1.96 everywhere — the sequence is never narrower than the one-shot
      interval, at any sample size. That is the price of unlimited looking.
    - Past the tuning horizon it *grows*, carrying the sqrt(log n) term that time-uniformity
      requires. Below the horizon it is instead dominated by the mixture's small-sample
      term and is wider still, which is why the comparison is made above the horizon.
    """
    scaled = {
        n: _mixture_half_width(1.0, n, ALPHA, HORIZON) * math.sqrt(n)
        for n in (2_000, 20_000, 200_000, 2_000_000, 20_000_000)
    }
    assert all(value > 1.96 for value in scaled.values()), scaled

    above_horizon = [scaled[n] for n in (200_000, 2_000_000, 20_000_000)]
    assert all(later > earlier for earlier, later in pairwise(above_horizon)), above_horizon


def _e1_shaped_window(rng: np.random.Generator, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """92,427 paired costs — the published test window — with a per-transaction shift.

    The delta carries its own dispersion rather than being a constant offset: two models
    disagree on *which* transactions they misprice, not by a fixed amount on every one. A
    constant offset has zero variance and would be certified at any effect size, which
    makes it a test of nothing.
    """
    champion = rng.gamma(0.6, 90.0, size=92_427)
    challenger = champion + shift + rng.normal(0.0, 60.0, size=92_427)
    return champion, challenger


def test_it_certifies_a_regression_the_size_e1_measured() -> None:
    """$4,358,096/yr over 92,427 transactions in a 32-day window is ~$4.14/txn."""
    champion, challenger = _e1_shaped_window(np.random.default_rng(3), 4.14)
    result = paired_cost_delta(champion, challenger)
    assert result.ci_lower > 0.0  # certifiably *worse*: a gate failure, not a near miss


def test_it_refuses_to_certify_a_saving_too_small_for_the_window() -> None:
    """The counterpart, and the more important one operationally.

    Findings §2 puts the calibrated-vs-raw penalty at $71,545/yr, about $0.07/txn. A gate
    that certified that from a single window would be certifying noise, and the confidence
    sequence declining to is the behaviour §4.8's power calculation predicts. The honest
    reading is that improvements of that size are not shippable on one window's evidence —
    not that the test is broken.

    Note the sample mean comes out the *wrong sign* on this seed: a true $0.07/txn saving
    inside $60 of per-transaction dispersion is not merely uncertain, it is invisible. Any
    gate deciding on the point estimate alone would flip on the seed.
    """
    champion, challenger = _e1_shaped_window(np.random.default_rng(3), -0.07)
    result = paired_cost_delta(champion, challenger)
    assert not result.challenger_is_cheaper
    assert result.ci_lower < 0.0 < result.ci_upper
    assert (result.ci_upper - result.ci_lower) > 10 * 0.07
