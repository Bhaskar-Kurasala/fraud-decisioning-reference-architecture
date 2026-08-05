"""Claims about promotion. Each test name is the claim.

Two of these are the epic's reason for existing: a challenger that wins on AUC and loses on
cost must be refused, and an immature window must not be evaluated at all.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from fraudlens.flywheel.promotion import evaluate_challenger, record_outcome
from fraudlens.models.checks import CheckStatus, GateThresholds
from fraudlens.models.registry import Stage
from fraudlens.models.tracking import local_tracking_uri
from tests.flywheel.conftest import (
    EPOCH,
    bin_calibrated,
    blur,
    build_window,
    odds_shift,
    truth,
)


def _status(outcome, name: str) -> CheckStatus:  # type: ignore[no-untyped-def]
    assert outcome.decision is not None
    return next(c for c in outcome.decision.checks if c.name == name).status


def test_a_challenger_that_wins_on_auc_and_loses_on_cost_is_refused(
    rng: np.random.Generator,
) -> None:
    """E1's finding, as an executable guard.

    The challenger here ranks *better* than the champion — it sees the unblurred latent —
    and it is 20x prior-shifted, so it challenges most of the book. A gate keyed on
    discrimination promotes it. On the published numbers that decision cost $4,358,096/yr.
    """
    z, y = truth(rng)
    champion = bin_calibrated(blur(rng, z, 0.6), y)
    challenger = odds_shift(bin_calibrated(z, y), 20.0)

    window = build_window(
        rng,
        champion_probability=champion,
        challenger_probability=challenger,
        is_fraud=y,
        matured_fraction=0.95,
    )
    outcome = evaluate_challenger(window, as_of=EPOCH, include_review=False)

    assert outcome.decision is not None
    assert outcome.champion is not None and outcome.challenger is not None
    assert not outcome.promoted

    # The premise, asserted rather than narrated: it really does win the metric a
    # conventional gate reads, on both discrimination measures.
    assert outcome.challenger.auc > outcome.champion.auc
    assert outcome.challenger.pr_auc > outcome.champion.pr_auc
    # ...and it really is more expensive to run.
    assert outcome.challenger.expected_cost_per_txn > outcome.champion.expected_cost_per_txn

    # Refused on calibration *and* on cost — the first names the defect, the second
    # prices it. Both are reported; neither short-circuits the other.
    assert _status(outcome, "calibration") is CheckStatus.FAILED
    assert _status(outcome, "expected_cost") is CheckStatus.FAILED
    cost = next(c for c in outcome.decision.checks if c.name == "expected_cost")
    assert cost.detail["mean_cost_delta_per_txn"] > 0.0
    # The confidence sequence, not a point estimate, is what refuses it.
    assert cost.detail["ci_lower"] > 0.0


def test_the_auc_gap_is_the_one_nobody_would_block(rng: np.random.Generator) -> None:
    """Separate claim, separately asserted: the discrimination difference is invisible.

    Findings §2 measured 0.0021. The point is not the exact figure — it is that the
    challenger is not distinguishable from the champion on the metric most teams gate with,
    while being refused by this gate.
    """
    z, y = truth(rng)
    champion = bin_calibrated(z, y)
    challenger = odds_shift(champion, 20.0)  # strictly monotone: AUC is unchanged exactly

    window = build_window(
        rng,
        champion_probability=champion,
        challenger_probability=challenger,
        is_fraud=y,
        matured_fraction=0.95,
    )
    outcome = evaluate_challenger(window, as_of=EPOCH, include_review=False)
    assert outcome.decision is not None
    calibration = next(c for c in outcome.decision.checks if c.name == "calibration")
    assert "AUC" in calibration.reason  # the rejection states the AUC it was not decided on
    assert not outcome.promoted


def test_a_genuinely_cheaper_challenger_is_promoted(rng: np.random.Generator) -> None:
    """Without this the gate could be "always block" and every other test would pass."""
    z, y = truth(rng)
    calibrated = bin_calibrated(z, y)
    window = build_window(
        rng,
        # Incumbent is the miscalibrated one; the challenger fixes it at identical ranking.
        champion_probability=odds_shift(calibrated, 20.0),
        challenger_probability=calibrated,
        is_fraud=y,
        matured_fraction=0.95,
    )
    outcome = evaluate_challenger(window, as_of=EPOCH, include_review=False)
    assert outcome.promoted, outcome.summary()
    assert all(c.status is not CheckStatus.FAILED for c in outcome.decision.checks)  # type: ignore[union-attr]


def test_an_immature_window_is_refused_before_any_cost_is_computed(
    rng: np.random.Generator,
) -> None:
    """The most expensive mistake available in this system.

    `decision is None` is the assertion that matters: not "the gate blocked it" but "the
    gate was never asked", because there was nothing honest to ask it with.
    """
    z, y = truth(rng)
    calibrated = bin_calibrated(z, y)
    window = build_window(
        rng,
        # A challenger that would sail through on a matured window.
        champion_probability=odds_shift(calibrated, 20.0),
        challenger_probability=calibrated,
        is_fraud=y,
        matured_fraction=0.30,
    )
    outcome = evaluate_challenger(window, as_of=EPOCH, include_review=False)

    assert outcome.decision is None
    assert not outcome.promoted
    assert outcome.refusal is not None
    assert "6.9%" in outcome.refusal  # findings §6, quoted where the refusal is read
    assert "30.0% of 20000 labels matured" in outcome.summary()


def test_the_maturity_floor_the_gate_uses_is_the_gate_s_own(rng: np.random.Generator) -> None:
    """Not `maturity.DEFAULT_MATURITY_FLOOR`. A reporting default must not decide money."""
    z, y = truth(rng)
    calibrated = bin_calibrated(z, y)
    window = build_window(
        rng,
        champion_probability=odds_shift(calibrated, 20.0),
        challenger_probability=calibrated,
        is_fraud=y,
        matured_fraction=0.70,
    )
    assert evaluate_challenger(window, as_of=EPOCH, include_review=False).decision is None

    relaxed = GateThresholds(min_label_maturity=0.60)
    assert (
        evaluate_challenger(window, as_of=EPOCH, include_review=False, thresholds=relaxed).decision
        is not None
    )


def test_labels_are_never_read_from_the_future(rng: np.random.Generator) -> None:
    """A reveal time after `as_of` counts as unrevealed, so a careless query cannot certify.

    The reveal times in this fixture are t+34d against transactions 120 days before
    `as_of`; moving `as_of` back before them must make the window immature rather than
    leaving it certified.
    """
    z, y = truth(rng)
    calibrated = bin_calibrated(z, y)
    window = build_window(
        rng,
        champion_probability=odds_shift(calibrated, 20.0),
        challenger_probability=calibrated,
        is_fraud=y,
        matured_fraction=1.0,
    )
    early = EPOCH - dt.timedelta(days=100)
    outcome = evaluate_challenger(window, as_of=early, include_review=False)
    assert outcome.decision is None


def test_an_unrevealed_row_may_not_carry_a_label(rng: np.random.Generator) -> None:
    """Because "unrevealed" and "not fraud" are the $4.36M confusion, structurally."""
    z, y = truth(rng)
    calibrated = bin_calibrated(z, y)
    window = build_window(
        rng,
        champion_probability=calibrated,
        challenger_probability=calibrated,
        is_fraud=y,
        matured_fraction=0.9,
    )
    with pytest.raises(ValueError, match="present or absent together"):
        type(window)(
            transaction_times=window.transaction_times,
            revealed_times=[None] * len(window.is_fraud),
            is_fraud=window.is_fraud,
            amount=window.amount,
            tenure_days=window.tenure_days,
            champion_probability=window.champion_probability,
            challenger_probability=window.challenger_probability,
        )


def test_a_refusal_is_recorded_as_a_refusal_not_as_a_blocked_gate(
    rng: np.random.Generator, client
) -> None:
    """§5.2's promotion-gate-failure alert must distinguish "worse" from "cannot yet tell".

    They need different people. "Worse" is a modelling conversation; "cannot yet tell" is
    the normal state a challenger sits in for the ninety days between training and knowing,
    and paging on it would train everyone to ignore the alert.
    """
    z, y = truth(rng)
    calibrated = bin_calibrated(z, y)
    window = build_window(
        rng,
        champion_probability=odds_shift(calibrated, 20.0),
        challenger_probability=calibrated,
        is_fraud=y,
        matured_fraction=0.30,
    )
    outcome = evaluate_challenger(window, as_of=EPOCH, include_review=False)
    client.create_registered_model("fraudlens-score")
    client.create_model_version(name="fraudlens-score", source="fixture://no-artifact")

    record = record_outcome(client, "fraudlens-score", 1, outcome, "mlops@example.com", EPOCH)
    assert not record.promoted
    assert record.to_stage is Stage.STAGING
    assert record.justification.startswith("REFUSED")
    # The incumbent is untouched: nothing has been demonstrated about the challenger.
    stages = {int(v.version) for v in client.search_model_versions("name='fraudlens-score'")}
    assert stages == {1}


@pytest.fixture
def client(tmp_path: Path):
    uri = local_tracking_uri(tmp_path / "mlruns")
    yield MlflowClient(tracking_uri=uri, registry_uri=uri)
