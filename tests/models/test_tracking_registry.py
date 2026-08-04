from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fraudlens.models.checks import GateInputs
from fraudlens.models.gate import evaluate_promotion
from fraudlens.models.metrics import ModelMetrics
from fraudlens.models.provenance import Provenance
from fraudlens.models.registry import (
    Stage,
    apply_gate_decision,
    current_champion,
    register_model,
    transition_stage,
)
from fraudlens.models.tracking import (
    TrainingRun,
    build_signature,
    ensure_experiment,
    log_training_run,
)

NOW = datetime(2026, 8, 5, 12, 0, tzinfo=UTC)


def _run() -> TrainingRun:
    return TrainingRun(
        run_name="champion-2026-08-05",
        params={"max_iter": 400, "learning_rate": 0.06},
        metrics=ModelMetrics(
            n=92_427,
            auc=0.9045,
            pr_auc=0.5271,
            ece=0.0035,
            calibration_slope=0.98,
            calibration_intercept=-0.02,
            brier=0.02201,
            expected_cost_per_txn=2.6557,
        ),
        provenance=Provenance(
            git_sha="a" * 40,
            git_dirty=False,
            config_hash="b" * 64,
            data_checksum="c" * 64,
        ),
        signature=build_signature(("TransactionAmt", "D1", "card1")),
    )


def test_a_logged_run_carries_its_full_provenance(client: Any) -> None:
    experiment_id = ensure_experiment(client, "fraudlens-test")
    run_id = log_training_run(client, experiment_id, _run(), NOW)

    logged = client.get_run(run_id)
    assert logged.data.params["max_iter"] == "400"
    assert logged.data.metrics["ece"] == pytest.approx(0.0035)
    assert logged.data.metrics["expected_cost_per_txn"] == pytest.approx(2.6557)
    assert logged.data.tags["fraudlens.git_sha"] == "a" * 40
    assert logged.data.tags["fraudlens.git_dirty"] == "false"
    assert logged.data.tags["fraudlens.config_hash"] == "b" * 64
    assert logged.data.tags["fraudlens.data_checksum"] == "c" * 64
    assert "TransactionAmt" in logged.data.tags["fraudlens.model_signature"]
    assert "fraud_probability" in logged.data.tags["fraudlens.model_output"]


def test_ensure_experiment_is_idempotent(client: Any) -> None:
    first = ensure_experiment(client, "fraudlens-test")
    assert ensure_experiment(client, "fraudlens-test") == first


def test_logging_rejects_a_naive_clock(client: Any) -> None:
    experiment_id = ensure_experiment(client, "fraudlens-test")
    with pytest.raises(ValueError, match="timezone-aware"):
        log_training_run(client, experiment_id, _run(), datetime(2026, 8, 5))  # noqa: DTZ001


def test_signature_requires_inputs() -> None:
    with pytest.raises(ValueError, match="no inputs"):
        build_signature(())


def _register(client: Any, tmp_path: Path, name: str = "fraud-scorer") -> int:
    experiment_id = ensure_experiment(client, "fraudlens-test")
    run_id = log_training_run(client, experiment_id, _run(), NOW)
    return register_model(client, name, source=str(tmp_path / "model"), run_id=run_id)


def test_champion_lookup_is_empty_before_any_promotion(client: Any, tmp_path: Path) -> None:
    _register(client, tmp_path)
    assert current_champion(client, "fraud-scorer") is None


def test_promotion_moves_the_champion_and_archives_the_incumbent(
    client: Any, tmp_path: Path
) -> None:
    first = _register(client, tmp_path)
    transition_stage(
        client, "fraud-scorer", first, Stage.PRODUCTION, "initial baseline", "kbhaskar36", NOW
    )
    assert int(current_champion(client, "fraud-scorer").version) == first

    second = _register(client, tmp_path)
    decision = evaluate_promotion(_winning_inputs())
    record = apply_gate_decision(client, "fraud-scorer", second, decision, "kbhaskar36", NOW)

    assert record.promoted
    assert record.to_stage is Stage.PRODUCTION
    assert int(current_champion(client, "fraud-scorer").version) == second
    assert client.get_model_version("fraud-scorer", str(first)).current_stage == "Archived"


def test_a_blocked_challenger_is_recorded_and_leaves_the_champion_alone(
    client: Any, tmp_path: Path
) -> None:
    """Rejections are evidence. They are written, not suppressed."""
    first = _register(client, tmp_path)
    transition_stage(
        client, "fraud-scorer", first, Stage.PRODUCTION, "initial baseline", "kbhaskar36", NOW
    )
    second = _register(client, tmp_path)

    decision = evaluate_promotion(_losing_inputs())
    record = apply_gate_decision(client, "fraud-scorer", second, decision, "kbhaskar36", NOW)

    assert not record.promoted
    assert record.to_stage is Stage.STAGING
    assert int(current_champion(client, "fraud-scorer").version) == first

    tags = client.get_model_version("fraud-scorer", str(second)).tags
    assert tags["fraudlens.promotion.decision"] == "blocked"
    assert tags["fraudlens.promotion.actor"] == "kbhaskar36"
    assert tags["fraudlens.promotion.decided_at"] == NOW.isoformat()
    checks = json.loads(tags["fraudlens.promotion.checks"])
    assert [c["name"] for c in checks] == [
        "label_maturity",
        "calibration",
        "expected_cost",
        "segment_guard",
    ]
    assert any(c["status"] == "failed" for c in checks)


def test_a_transition_without_justification_is_refused(client: Any, tmp_path: Path) -> None:
    version = _register(client, tmp_path)
    with pytest.raises(ValueError, match="justification"):
        transition_stage(client, "fraud-scorer", version, Stage.PRODUCTION, "  ", "who", NOW)


def _metrics(ece: float, cost: float) -> ModelMetrics:
    return ModelMetrics(
        n=10_000,
        auc=0.90,
        pr_auc=0.52,
        ece=ece,
        calibration_slope=1.0,
        calibration_intercept=0.0,
        brier=0.022,
        expected_cost_per_txn=cost,
    )


def _winning_inputs() -> GateInputs:
    champion_cost = np.random.default_rng(5).gamma(2.0, 1.0, size=10_000)
    return GateInputs(
        champion=_metrics(0.0035, 2.0),
        challenger=_metrics(0.0040, 1.7),
        champion_cost=champion_cost,
        challenger_cost=champion_cost - 0.3,
        label_maturity=0.95,
    )


def _losing_inputs() -> GateInputs:
    champion_cost = np.random.default_rng(5).gamma(2.0, 1.0, size=10_000)
    return GateInputs(
        champion=_metrics(0.0035, 2.0),
        challenger=_metrics(0.1389, 4.0),
        champion_cost=champion_cost,
        challenger_cost=champion_cost + 0.3,
        label_maturity=0.95,
        segments={"amount_band": ["low"] * 5_000 + ["high"] * 5_000},
    )
