"""Paired, anytime-valid inference on per-transaction cost difference.

Why not a t-test on AUC, and why not a fixed-horizon test at all:

The blueprint's §4.8 power calculation settles it. At a 0.14% approved-population
fraud rate and 800k auths/arm/day, detecting a 5% relative fraud reduction needs
~4.5M per arm (5.6 days); detecting 1% needs ~112M per arm — 140 days, longer
than both the retrain cadence and the adversary's adaptation cycle. A
fixed-horizon experiment sized for the effects we actually ship cannot conclude
before the world it is measuring has changed. Offline counterfactual replay is
therefore the primary evidence, and the test below is what reads it.

Two consequences shape this module:

1. **Paired, on the same transactions.** Both models are replayed over the same
   historical window, so the difference is taken per transaction. This removes
   all between-period variance — the dominant term, since fraud rate and mix
   move far more day-to-day than the models differ.
2. **A confidence sequence, not a confidence interval.** Replay is re-run every
   time labels mature further, and each look at a fixed-horizon interval inflates
   the type-I error. The Robbins normal-mixture boundary below is valid at every
   sample size simultaneously, so the gate may be re-evaluated as often as
   labels arrive without any alpha-spending bookkeeping. It costs roughly 1.9x
   the width of a one-shot 95% interval, which is the honest price of looking
   whenever you like.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# The mixture boundary is tightest near this horizon and remains valid
# everywhere else. 100k transactions is roughly the 32-day replay window the
# findings were produced on, so the boundary is sharp where it is actually read.
DEFAULT_TUNING_HORIZON = 100_000


@dataclass(frozen=True, slots=True)
class PairedCostDelta:
    """Per-transaction cost difference, challenger minus champion.

    Negative means the challenger is cheaper. `ci_upper < 0` is the evidence the
    gate requires: the challenger saves money at the stated confidence.
    """

    n: int
    mean_delta: float
    std_delta: float
    ci_lower: float
    ci_upper: float
    alpha: float
    tuning_horizon: int

    @property
    def challenger_is_cheaper(self) -> bool:
        return self.ci_upper < 0.0


def _mixture_half_width(std: float, n: int, alpha: float, tuning_horizon: int) -> float:
    """Robbins' normal-mixture confidence-sequence radius.

    radius = s * sqrt( 2(n*rho^2 + 1) / (n^2 * rho^2) * ln( sqrt(n*rho^2 + 1) / alpha ) )

    with rho^2 = 1 / tuning_horizon. See Howard et al. (2021), "Time-uniform,
    nonparametric, nonasymptotic confidence sequences", §3.
    """
    rho_sq = 1.0 / tuning_horizon
    term = n * rho_sq + 1.0
    return std * math.sqrt((2.0 * term / (n * n * rho_sq)) * math.log(math.sqrt(term) / alpha))


def paired_cost_delta(
    champion_cost: FloatArray,
    challenger_cost: FloatArray,
    alpha: float = 0.05,
    tuning_horizon: int = DEFAULT_TUNING_HORIZON,
) -> PairedCostDelta:
    """Confidence sequence on the mean per-transaction cost difference.

    Costs are realised (label-aware) costs from replaying both models through
    the same policy over the same transactions, in the same order.
    """
    if len(champion_cost) != len(challenger_cost):
        raise ValueError("costs must be paired on the same transactions")
    n = len(champion_cost)
    if n < 2:
        raise ValueError("need at least two paired observations")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    delta = challenger_cost - champion_cost
    mean = float(delta.mean())
    # Sample std: the cost distribution is heavy-tailed (a single high-amount
    # chargeback dominates), so the boundary is wide by construction. That is
    # the correct behaviour — it is why a small aggregate improvement genuinely
    # cannot be certified from a short window.
    std = float(delta.std(ddof=1))
    radius = _mixture_half_width(std, n, alpha, tuning_horizon)
    return PairedCostDelta(
        n=n,
        mean_delta=mean,
        std_delta=std,
        ci_lower=mean - radius,
        ci_upper=mean + radius,
        alpha=alpha,
        tuning_horizon=tuning_horizon,
    )
