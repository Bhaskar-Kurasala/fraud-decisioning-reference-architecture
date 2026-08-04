"""Fixtures for the lineage tests.

The replay claim in §9a is end-to-end by nature — it asserts that a decision written by
the *serving* path re-derives from its own ledger row — so these fixtures wire the real
serving decision function to a real SQLite ledger. Tests are not bound by the import
contract that keeps `lineage` from importing `serving`, and that is exactly what lets the
suite check the two agree.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from fraudlens.models.tracking import local_tracking_uri
from fraudlens.serving.contracts import DecideRequest
from fraudlens.serving.runtime import CalibratedScorer
from fraudlens.streaming.migrate import migrate

EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


@pytest.fixture
def client(tmp_path: Path) -> Any:
    """A real MLflow client against a throwaway local SQLite store.

    Real rather than mocked for the same reason `tests/models` uses one: the card is read
    out of MLflow's actual response objects, and a mock would assert only that we call
    the methods we think we call.
    """
    uri = local_tracking_uri(tmp_path / "mlruns")
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


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


class FixedScorer:
    """A scorer that returns a set probability, so the test controls the policy input.

    Not a mock of the model: the point of these tests is the *policy* replay, and pinning
    the probability is how a divergence gets attributed to the cost model or the config
    rather than to score noise. The model arm is not replayable at all (see
    `lineage.replay`), so simulating it would test nothing.
    """

    def __init__(self, probability: float) -> None:
        self._probability = probability

    @property
    def version(self) -> str:
        return "test-scorer-v1"

    def score(self, features: Mapping[str, float]) -> float:
        del features
        return self._probability

    def calibrate(self, raw_score: float) -> float:
        return raw_score


@pytest.fixture
def scorer() -> CalibratedScorer:
    # 0.55 sits between the allow/challenge and challenge/deny boundaries for the amounts
    # used below, so a config change can move the action in either direction — a
    # probability parked at 0.001 would replay identically under any config and the test
    # would pass while proving nothing.
    return FixedScorer(0.55)


def request_for(amount: float, days_since_first_seen: float | None = 3.0) -> DecideRequest:
    return DecideRequest(
        transaction_id=2987000,
        transaction_at=EPOCH,
        amount=amount,
        days_since_first_seen=days_since_first_seen,
        features={"C1": 1.0, "C13": 24.0},
    )
