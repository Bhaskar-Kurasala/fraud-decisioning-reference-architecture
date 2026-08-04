"""Fixtures for the model-layer tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mlflow.tracking import MlflowClient

from fraudlens.models.tracking import local_tracking_uri

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[Any]:
    """A real MLflow client against a throwaway local SQLite store.

    Real, not mocked: the point of the tracking module is that it agrees with
    MLflow's actual API surface, and a mock would assert only that we call the
    methods we think we call.
    """
    uri = local_tracking_uri(tmp_path / "mlruns")
    yield MlflowClient(tracking_uri=uri, registry_uri=uri)
