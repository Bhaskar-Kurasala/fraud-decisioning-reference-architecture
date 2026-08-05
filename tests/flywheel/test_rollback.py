"""Claims about rollback. The theme: undo is not available, so accounting must be.

The value of a rollback in an append-only system is the sentence "these N decisions were
made under the model we rolled back, and here is what they cost". Every test below is about
being able to say it, or about not being able to say something stronger than the evidence
supports.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mlflow.tracking import MlflowClient
from sqlalchemy import Engine, create_engine, delete
from sqlalchemy.exc import DatabaseError
from sqlalchemy.pool import StaticPool

from fraudlens.flywheel.rollback import decisions_by_model, quantify, roll_back
from fraudlens.models.registry import Stage
from fraudlens.models.tracking import local_tracking_uri
from fraudlens.streaming.ledger import DecisionLedger, DecisionRecord
from fraudlens.streaming.migrate import migrate
from fraudlens.streaming.schema import decision_ledger

EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
BAD = "challenger-v3"
GOOD = "champion-v2"
N_BAD = 400


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    migrate(eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def populated(engine: Engine) -> Engine:
    """400 decisions under the bad model, 100 under the good one that preceded it."""
    ledger = DecisionLedger(engine)
    ledger.record_many(
        [_decision(i, GOOD, "allow", EPOCH - dt.timedelta(days=40 - i // 10)) for i in range(100)]
        + [
            _decision(
                100 + i,
                BAD,
                "deny" if i % 4 == 0 else "allow",
                EPOCH - dt.timedelta(days=30) + dt.timedelta(hours=i),
            )
            for i in range(N_BAD)
        ]
    )
    return engine


def _decision(transaction_id: int, model: str, action: str, at: dt.datetime) -> DecisionRecord:
    return DecisionRecord(
        transaction_id=transaction_id,
        transaction_at=at,
        decided_at=at,
        score=0.7,
        calibrated_probability=0.55,
        action=action,
        reason_codes=("HIGH_RISK",),
        model_version=model,
        policy_version="p1",
        feature_version="f1",
        config_hash="c",
        input_hash="h",
    )


def _population(n: int, matured: int) -> dict[str, Any]:
    rng = np.random.default_rng(4)
    return {
        "is_fraud": [bool(i % 9 == 0) if i < matured else None for i in range(n)],
        "amount": np.asarray(rng.lognormal(4.2, 1.0, size=n), dtype=np.float64),
        "tenure_days": np.asarray(rng.integers(0, 500, size=n), dtype=np.float64),
    }


def test_the_affected_population_is_identifiable_by_model_version(populated: Engine) -> None:
    """`model_version` is a required non-defaulted ledger field precisely for this query."""
    affected = decisions_by_model(populated, BAD)
    assert len(affected) == N_BAD
    assert {d.transaction_id for d in affected}.isdisjoint(range(100))
    # Time-ordered, so the exposure window is read off the ends.
    assert affected == sorted(affected, key=lambda d: d.transaction_at)


def test_the_population_is_priced_and_the_action_mix_reported(populated: Engine) -> None:
    affected = decisions_by_model(populated, BAD)
    exposure = quantify(
        affected, model_version=BAD, include_review=False, **_population(N_BAD, N_BAD)
    )
    assert exposure.n_decisions == N_BAD
    assert exposure.n_priced == N_BAD
    assert exposure.action_counts == {"allow": 300, "deny": 100}
    assert exposure.realised_cost_total > 0.0
    assert exposure.annualised_cost > exposure.realised_cost_total  # window is under a year


def test_unmatured_rows_are_excluded_from_the_price_not_counted_as_clean(
    populated: Engine,
) -> None:
    """At the moment a rollback is considered, most of the population is unmatured.

    Pricing those as non-fraud would value the incident at the friction cost alone — the
    term a bad model is *cheap* on — so the count and the price are reported separately
    and the gap between them is visible.
    """
    affected = decisions_by_model(populated, BAD)
    partial = quantify(affected, model_version=BAD, include_review=False, **_population(N_BAD, 100))
    full = quantify(affected, model_version=BAD, include_review=False, **_population(N_BAD, N_BAD))

    assert partial.n_decisions == full.n_decisions == N_BAD
    assert partial.n_priced == 100
    assert partial.realised_cost_total < full.realised_cost_total


def test_the_counterfactual_is_absent_rather_than_zero_when_it_cannot_be_computed(
    populated: Engine,
) -> None:
    """There is no champion estimator on disk. The number people most want is the one
    there is least honest basis for, so it is None and says so (ADR-0003)."""
    affected = decisions_by_model(populated, BAD)
    exposure = quantify(
        affected, model_version=BAD, include_review=False, **_population(N_BAD, N_BAD)
    )
    assert exposure.counterfactual_cost_total is None
    assert exposure.counterfactual_delta is None


def test_the_counterfactual_is_computed_when_restored_scores_are_supplied(
    populated: Engine,
) -> None:
    """The intended supplier is `shadow`: a demoted champion left scoring in shadow mode
    already holds exactly these probabilities."""
    affected = decisions_by_model(populated, BAD)
    exposure = quantify(
        affected,
        model_version=BAD,
        include_review=False,
        restored_probability=np.full(N_BAD, 0.01),  # a champion that would have allowed
        **_population(N_BAD, N_BAD),
    )
    assert exposure.counterfactual_cost_total is not None
    assert exposure.counterfactual_delta == pytest.approx(
        exposure.realised_cost_total - exposure.counterfactual_cost_total
    )


def test_a_rollback_does_not_touch_the_ledger(populated: Engine, client: Any) -> None:
    """The point of the epic. The decisions stand and remain attributable to the bad model."""
    before = decisions_by_model(populated, BAD)
    exposure = quantify(
        before, model_version=BAD, include_review=False, **_population(N_BAD, N_BAD)
    )
    _register(client, "fraudlens-score", versions=2)
    roll_back(
        client,
        "fraudlens-score",
        bad_version=2,
        restore_version=1,
        exposure=exposure,
        reason="challenger regressed on new-account approval",
        actor="oncall@example.com",
        now=EPOCH,
    )
    assert decisions_by_model(populated, BAD) == before


def test_the_ledger_refuses_the_undo_a_rollback_is_not(populated: Engine) -> None:
    """Belt and braces: even a deliberate attempt to erase the population is rejected.

    This is why rollback had to be defined as a pin revert plus an accounting, rather than
    implemented as the obvious thing.
    """
    with pytest.raises(DatabaseError, match="append-only"), populated.begin() as conn:
        conn.execute(delete(decision_ledger).where(decision_ledger.c.model_version == BAD))


def test_the_exposure_is_written_into_the_registry_justification(
    populated: Engine, client: Any
) -> None:
    """A rollback recorded as "reverted, model was bad" stops being answerable in six
    months. The registry is where a model-risk review looks, so the number goes there."""
    affected = decisions_by_model(populated, BAD)
    exposure = quantify(
        affected, model_version=BAD, include_review=False, **_population(N_BAD, 100)
    )
    _register(client, "fraudlens-score", versions=2)
    archived, restored = roll_back(
        client,
        "fraudlens-score",
        bad_version=2,
        restore_version=1,
        exposure=exposure,
        reason="challenger regressed on new-account approval",
        actor="oncall@example.com",
        now=EPOCH,
    )
    assert archived.to_stage is Stage.ARCHIVED
    assert restored.to_stage is Stage.PRODUCTION
    assert f"{N_BAD} decisions taken under version 2" in archived.justification
    assert "100 matured" in archived.justification
    assert "counterfactual not computed" in archived.justification
    assert "The ledger is unchanged" in archived.justification

    versions = client.search_model_versions("name='fraudlens-score'")
    stages = {int(v.version): v.current_stage for v in versions}
    assert stages == {1: Stage.PRODUCTION.value, 2: Stage.ARCHIVED.value}


def test_rolling_back_to_the_version_being_rolled_back_is_refused(
    populated: Engine, client: Any
) -> None:
    affected = decisions_by_model(populated, BAD)
    exposure = quantify(
        affected, model_version=BAD, include_review=False, **_population(N_BAD, N_BAD)
    )
    with pytest.raises(ValueError, match="no-op"):
        roll_back(
            client,
            "fraudlens-score",
            bad_version=2,
            restore_version=2,
            exposure=exposure,
            reason="x",
            actor="a",
            now=EPOCH,
        )


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Any]:
    uri = local_tracking_uri(tmp_path / "mlruns")
    yield MlflowClient(tracking_uri=uri, registry_uri=uri)


def _register(client: Any, name: str, *, versions: int) -> None:
    client.create_registered_model(name)
    for _ in range(versions):
        client.create_model_version(name=name, source="fixture://no-artifact")
