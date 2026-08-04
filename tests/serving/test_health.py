"""Liveness and readiness have different semantics and must not be aliased.

Aliasing them is how a model outage becomes a checkout outage: the orchestrator restarts
a pod that is answering correctly in degraded mode, the replacement also has no model, and
the loop continues while the fail-safe path -- which was working -- is taken offline.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from fraudlens.serving.app import create_app
from fraudlens.serving.runtime import CalibratedScorer
from tests.serving.conftest import FROZEN_NOW, StubScorer, make_client, request_body


def test_live_is_true_even_with_no_model() -> None:
    """Liveness never consults the model. Restarting does not produce one."""
    with make_client(scorer=None) as client:
        response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "live"


def test_ready_is_false_with_no_model_and_says_why() -> None:
    with make_client(scorer=None) as client:
        response = client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert "no artifact configured" in (response.json()["detail"] or "")


def test_ready_is_true_once_the_model_is_loaded(client: TestClient) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "detail": None}


def test_the_model_is_loaded_once_at_startup_not_per_request() -> None:
    """Per-request loading would spend the whole 25 ms inference budget on deserialisation
    and would let two concurrent requests be decided by different artifacts."""
    pins: list[str | None] = []

    def counting_loader(pin: str | None) -> CalibratedScorer:
        pins.append(pin)
        return StubScorer(0.01)

    app = create_app(
        loader=counting_loader,
        model_version_pin="champion-v7",
        clock=lambda: FROZEN_NOW,
        elapsed=lambda: 0.0,
    )
    with TestClient(app) as client:
        for _ in range(5):
            client.get("/health/ready")
            client.post("/v1/decide", json=request_body())

    assert pins == ["champion-v7"]
