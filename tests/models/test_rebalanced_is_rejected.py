"""The regression test this epic exists for.

E1 measured a class-rebalanced refit of the champion at AUC 0.9029 against the
champion's 0.9045 — a 0.0021 gap that no reasonable review process would block —
which costs $4.37M/yr more, because its ECE is 0.1389 against 0.0035 and it
challenges 52% of traffic instead of 7%.

A registry that gates on discrimination promotes that model. This test asserts
that ours refuses it, and asserts *which* checks refuse it and why. It turns a
one-off finding into an enforced invariant: if someone later relaxes the
calibration tolerance or swaps the cost test for a t-test on AUC, this fails.

Inputs are the real artifacts: `data/p_te_bal_fitted.npy` (the genuine
`class_weight='balanced'` refit, not the analytic prior shift) and
`data/econ_test.parquet` (the champion score plus per-transaction L and M).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.models.checks import CheckStatus, GateInputs, GateThresholds
from fraudlens.models.gate import evaluate_promotion
from fraudlens.models.metrics import evaluate_scores
from fraudlens.models.sequential import paired_cost_delta
from tests.models._replay import DATA_DIR, CostInputs, load_root_config, replay

ECON = DATA_DIR / "econ_test.parquet"
REBALANCED = DATA_DIR / "p_te_bal_fitted.npy"

pytestmark = [
    pytest.mark.golden,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (ECON.exists() and REBALANCED.exists()),
        reason="requires data/ artifacts (regenerate via run_all.sh)",
    ),
]

TRADING_DAYS_PER_YEAR = 365


@pytest.fixture(scope="module")
def replayed() -> dict[str, object]:
    cfg = load_root_config()
    frame = pd.read_parquet(ECON)
    y = frame["isFraud"].to_numpy(dtype=np.int_)
    cost = CostInputs.from_frame(
        frame["L"].to_numpy(dtype=np.float64), frame["M"].to_numpy(dtype=np.float64), cfg
    )
    champion_score = frame["p"].to_numpy(dtype=np.float64)
    challenger_score = np.load(REBALANCED).astype(np.float64)

    return {
        "y": y,
        "days": int(frame["day"].nunique()),
        "champion_score": champion_score,
        "challenger_score": challenger_score,
        "champion_cost": replay(champion_score, y, cost),
        "challenger_cost": replay(challenger_score, y, cost),
        "amount_band": frame["amt_band"].astype(str).to_numpy(),
        "tenure": frame["tenure"].astype(str).to_numpy(),
    }


@pytest.fixture(scope="module")
def gate_inputs(replayed: dict[str, object]) -> GateInputs:
    y = replayed["y"]
    return GateInputs(
        champion=evaluate_scores(y, replayed["champion_score"], replayed["champion_cost"]),
        challenger=evaluate_scores(y, replayed["challenger_score"], replayed["challenger_cost"]),
        champion_cost=replayed["champion_cost"],
        challenger_cost=replayed["challenger_cost"],
        # The test window is historical; its labels are fully matured, so this
        # check must not be what does the rejecting.
        label_maturity=1.0,
        segments={
            "amount_band": replayed["amount_band"],
            "tenure": replayed["tenure"],
        },
    )


def test_e1_numbers_still_reproduce(gate_inputs: GateInputs, replayed: dict[str, object]) -> None:
    """Guard the premise. If these move, the rejection below proves nothing."""
    champion, challenger = gate_inputs.champion, gate_inputs.challenger
    assert champion.auc == pytest.approx(0.9045, abs=5e-4)
    assert challenger.auc == pytest.approx(0.9029, abs=5e-4)
    assert champion.ece == pytest.approx(0.0035, abs=5e-4)
    assert challenger.ece == pytest.approx(0.1389, abs=5e-3)

    days = replayed["days"]
    annual = lambda c: float(c.sum()) * TRADING_DAYS_PER_YEAR / days  # noqa: E731
    penalty = annual(gate_inputs.challenger_cost) - annual(gate_inputs.champion_cost)
    assert penalty == pytest.approx(4_369_611, rel=0.02)


def test_auc_alone_would_have_promoted_it(gate_inputs: GateInputs) -> None:
    """The counterfactual that makes the gate load-bearing.

    A conventional gate — "promote unless AUC regresses by more than 0.005" —
    accepts this model. That is not a strawman; it is the industry default.
    """
    auc_regression = gate_inputs.champion.auc - gate_inputs.challenger.auc
    assert auc_regression < 0.005


def test_the_gate_rejects_the_rebalanced_model(gate_inputs: GateInputs) -> None:
    decision = evaluate_promotion(gate_inputs)
    assert not decision.promote, decision.summary()

    by_name = {c.name: c for c in decision.checks}
    assert by_name["label_maturity"].status is CheckStatus.PASSED
    # Calibration is the check that catches it, and it must be the calibration
    # check specifically: that is what makes the recorded rejection reason name
    # the real defect instead of a downstream symptom.
    assert by_name["calibration"].status is CheckStatus.FAILED
    assert "ECE regressed" in by_name["calibration"].reason
    # ...and the cost test independently agrees, on the same transactions.
    assert by_name["expected_cost"].status is CheckStatus.FAILED
    assert by_name["segment_guard"].status is CheckStatus.FAILED


def test_the_cost_evidence_is_paired_and_unambiguous(gate_inputs: GateInputs) -> None:
    delta = paired_cost_delta(gate_inputs.champion_cost, gate_inputs.challenger_cost)
    assert delta.mean_delta > 4.0
    # Even the *lower* end of an anytime-valid confidence sequence says the
    # rebalanced model is more expensive. There is no window size at which this
    # becomes a close call.
    assert delta.ci_lower > 0.0
    assert not delta.challenger_is_cheaper


def test_every_amount_band_and_tenure_bucket_regresses(gate_inputs: GateInputs) -> None:
    """The harm is not concentrated; it is universal. A segment guard alone
    would also have caught it, which is the redundancy the ordering buys."""
    decision = evaluate_promotion(gate_inputs)
    segment = next(c for c in decision.checks if c.name == "segment_guard")
    deltas = {k: v for k, v in segment.detail.items() if k.startswith("segment_delta.")}
    assert len(deltas) >= 10
    assert all(v > 0 for v in deltas.values()), deltas


def test_relaxing_the_calibration_tolerance_does_not_let_it_through(
    gate_inputs: GateInputs,
) -> None:
    """Defence in depth: even with the calibration guard effectively disabled,
    the cost test and the segment guard still block it."""
    permissive = GateThresholds(max_ece_regression=1.0)
    decision = evaluate_promotion(gate_inputs, permissive)
    assert not decision.promote
    by_name = {c.name: c for c in decision.checks}
    assert by_name["calibration"].status is CheckStatus.PASSED
    assert by_name["expected_cost"].status is CheckStatus.FAILED
