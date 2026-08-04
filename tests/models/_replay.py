"""Counterfactual replay for the gate tests: score -> action -> realised cost.

Routes a score through the *production* policy and cost model. This module exists
only to shape the gate's inputs: `models` sits below `economics` in the import
contract, so the gate takes per-transaction costs as arguments and never imports
the policy itself. Tests are not bound by that layering, so the fixture can use
the real definitions.

An earlier version reimplemented the EV argmax and the realised-cost arithmetic
here, because `economics/` and `policy/` were being written concurrently. That
duplication is gone. Reimplementing the policy from its description is what
produced two separate corrections in this project already (see
docs/findings/fit-balanced-empirical-result.md); a second copy living in the test
suite would be the same mistake with a longer fuse, because it would let the
gate's regression test keep passing against arithmetic the service no longer
uses — the test would defend a policy nobody runs.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt

from fraudlens.economics.expected_value import realised_cost
from fraudlens.policy.decisions import decide

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
    """Per-transaction false-negative and false-positive costs.

    The intervention parameters (f_pass, a_abandon, q_analyst, c_review, d_delay)
    are no longer carried here — they live in `fraudlens.config` and reach the
    calculation through the production functions, so there is one place they can
    be wrong. `tests/golden/test_config_parity.py` asserts those match the
    research values.
    """

    loss_fn: FloatArray  # L: Amount*COGS + chargeback fee + dispute ops
    loss_fp: FloatArray  # M: Amount*MARGIN + relationship cost

    @classmethod
    def from_frame(
        cls, loss_fn: FloatArray, loss_fp: FloatArray, cfg: ModuleType | None = None
    ) -> CostInputs:
        """`cfg` is accepted and ignored; kept so callers need not change."""
        del cfg
        return cls(loss_fn=loss_fn, loss_fp=loss_fp)


def replay(p: FloatArray, y: IntArray, cost: CostInputs) -> FloatArray:
    """Per-transaction realised cost of routing `p` through the EV policy.

    Four-action (review enabled) — the policy the published findings table
    reports. The three-action variant costs $583/yr less on this window, which is
    the measured value of offering analyst review, so the choice is stated rather
    than defaulted (`decide` has no default for `include_review` for that reason).
    """
    actions = decide(p, cost.loss_fn, cost.loss_fp, include_review=True)
    return realised_cost(actions, y, cost.loss_fn, cost.loss_fp)
