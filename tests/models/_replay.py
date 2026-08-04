"""Test-only counterfactual replay: score -> action -> realised cost.

This duplicates the four-action EV argmax from `research/05_economics.py` on
purpose and temporarily. Its production home is `fraudlens.economics` /
`fraudlens.policy`, which sit *above* `fraudlens.models` in the dependency order
and are being built concurrently. The gate takes realised per-transaction costs
as an input precisely so it never has to import them; this module is what
supplies those costs to the regression test, and it should be deleted in favour
of the real policy simulator once that lands.

Business constants are loaded from the repository's `config.py` rather than
restated here — a copied constant is a constant that will drift.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int_]

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def load_root_config() -> ModuleType:
    """Import the repository's root `config.py` without mutating `sys.path`."""
    spec = importlib.util.spec_from_file_location("fraudlens_root_config", REPO_ROOT / "config.py")
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("could not load config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class CostInputs:
    """Per-transaction FN/FP costs and the intervention parameters."""

    loss_fn: FloatArray  # L: Amount*COGS + chargeback fee + dispute ops
    loss_fp: FloatArray  # M: Amount*MARGIN + relationship cost
    f_pass: float
    a_abandon: float
    q_analyst: float
    c_review: float
    d_delay: float

    @classmethod
    def from_frame(cls, loss_fn: FloatArray, loss_fp: FloatArray, cfg: ModuleType) -> CostInputs:
        return cls(
            loss_fn=loss_fn,
            loss_fp=loss_fp,
            f_pass=cfg.F_PASS,
            a_abandon=cfg.A_ABANDON,
            q_analyst=cfg.Q_ANALYST,
            c_review=cfg.C_REVIEW,
            d_delay=cfg.D_DELAY,
        )


def ev_argmax_actions(p: FloatArray, cost: CostInputs) -> IntArray:
    """Action per transaction: allow / challenge / review / deny, by EV argmax."""
    fn, fp = cost.loss_fn, cost.loss_fp
    expected_values = np.vstack(
        [
            -p * fn,
            -(p * cost.f_pass * fn + (1 - p) * cost.a_abandon * fp),
            -((1 - cost.q_analyst) * (p * fn + (1 - p) * fp) + cost.c_review + cost.d_delay),
            -(1 - p) * fp,
        ]
    )
    return np.asarray(expected_values.argmax(axis=0), dtype=np.int_)


def realised_cost(actions: IntArray, y: IntArray, cost: CostInputs) -> FloatArray:
    """Cost actually incurred, given the true label."""
    fn, fp = cost.loss_fn, cost.loss_fp
    out = np.zeros(len(y), dtype=np.float64)

    allow = actions == 0
    out[allow] = y[allow] * fn[allow]

    chal = actions == 1
    out[chal] = y[chal] * cost.f_pass * fn[chal] + (1 - y[chal]) * cost.a_abandon * fp[chal]

    review = actions == 2
    out[review] = (
        (1 - cost.q_analyst) * (y[review] * fn[review] + (1 - y[review]) * fp[review])
        + cost.c_review
        + cost.d_delay
    )

    deny = actions == 3
    out[deny] = (1 - y[deny]) * fp[deny]
    return out


def replay(p: FloatArray, y: IntArray, cost: CostInputs) -> FloatArray:
    """Per-transaction realised cost of routing `p` through the EV policy."""
    return realised_cost(ev_argmax_actions(p, cost), y, cost)
