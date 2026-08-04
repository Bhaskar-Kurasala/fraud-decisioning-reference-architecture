"""Fixtures for the scoring API.

Every seam the service exposes is stubbed here, which is the whole reason they exist: no
test in this directory needs a network, an MLflow server, a database or a sleep.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.trace import TracerProvider

from fraudlens.serving.app import create_app
from fraudlens.serving.runtime import CalibratedScorer, ModelUnavailableError

FROZEN_NOW = dt.datetime(2026, 8, 5, 14, 32, 10, tzinfo=dt.timezone.utc)


class StubScorer:
    """A scorer whose output the test dictates. Satisfies `CalibratedScorer` structurally."""

    def __init__(self, probability: float, *, version: str = "champion-v7", raises: bool = False):
        self._probability = probability
        self._version = version
        self._raises = raises

    @property
    def version(self) -> str:
        return self._version

    def score(self, features: Mapping[str, float]) -> float:
        if self._raises:
            raise RuntimeError("inference backend exploded")
        return self._probability

    def calibrate(self, raw_score: float) -> float:
        return self._probability


class RecordingWriter:
    """Captures what would have been persisted, so the ledger contract can be asserted."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record_decision(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


def make_client(
    *,
    scorer: CalibratedScorer | None = None,
    probability: float = 0.02,
    elapsed: Callable[[], float] | None = None,
    writer: Any = None,
    model_version_pin: str | None = None,
    tracer_provider: TracerProvider | None = None,
) -> TestClient:
    """A wired app. `scorer=None` with no probability override means the load fails."""

    def loader(pin: str | None) -> CalibratedScorer:
        if scorer is None:
            raise ModelUnavailableError("no artifact configured")
        return scorer

    app = create_app(
        loader=loader,
        model_version_pin=model_version_pin,
        clock=lambda: FROZEN_NOW,
        elapsed=elapsed if elapsed is not None else (lambda: 0.0),
        writer=writer,
        # Defaults to None, which is the production default too: no provider means the
        # OTel API hands back a non-recording tracer, so every test in this directory
        # exercises the exact configuration a unit test would accidentally get wrong --
        # one that exports nothing and opens nothing.
        tracer_provider=tracer_provider,
    )
    return TestClient(app)


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Healthy service, scorer returns a low probability (an allow)."""
    with make_client(scorer=StubScorer(0.01)) as c:
        yield c


def request_body(**overrides: Any) -> dict[str, Any]:
    """A valid `POST /v1/decide` body."""
    body: dict[str, Any] = {
        "transaction_id": 2987000,
        "transaction_at": "2026-08-05T14:32:10Z",
        "amount": 249.99,
        "days_since_first_seen": 120.0,
        "features": {"C1": 1.0, "C13": 24.0},
    }
    body.update(overrides)
    return body
