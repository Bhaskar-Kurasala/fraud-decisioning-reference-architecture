"""MLflow experiment tracking for training runs.

The MLflow client is injected rather than constructed here so that tests exercise
the real logging path against a temporary local store, with no server and no
global `mlflow.set_tracking_uri` side effect leaking between tests.

What gets logged is fixed by §9a, not by convenience: params, the full metric
set, the model signature, and the provenance triple (git SHA, config hash, data
checksum). A run missing any of those cannot be revalidated later, and a model
that cannot be revalidated cannot be defended when a decline is disputed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from fraudlens.models.metrics import ModelMetrics
from fraudlens.models.provenance import Provenance

# The scored probability, named so the serving contract and the registry agree.
SCORE_OUTPUT_NAME = "fraud_probability"


def local_tracking_uri(root: Path) -> str:
    """Tracking + registry URI for a serverless local store.

    SQLite rather than the `./mlruns` file store: MLflow 3.15 puts the
    filesystem backend in maintenance mode and refuses it without an opt-out
    environment variable, and the file store has never supported the model
    registry we depend on. SQLite needs no server, so tests and a laptop run use
    the same code path CI does.
    """
    root.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{root / 'mlflow.db'}"


def build_signature(feature_names: tuple[str, ...]) -> ModelSignature:
    """Input schema plus the single probability output.

    All features are declared `double`. The scorer receives an already-encoded
    numeric matrix, so a wider type vocabulary here would describe the feature
    pipeline's inputs rather than the model's, and would silently drift from it.
    """
    if not feature_names:
        raise ValueError("a signature with no inputs cannot validate anything")
    inputs = Schema([ColSpec("double", name) for name in feature_names])
    outputs = Schema([ColSpec("double", SCORE_OUTPUT_NAME)])
    return ModelSignature(inputs=inputs, outputs=outputs)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """One training run's complete record."""

    run_name: str
    params: Mapping[str, Any]
    metrics: ModelMetrics
    provenance: Provenance
    signature: ModelSignature


def ensure_experiment(client: Any, name: str) -> str:
    """Return the experiment id, creating it if absent."""
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    return str(client.create_experiment(name))


def log_training_run(
    client: Any,
    experiment_id: str,
    run: TrainingRun,
    now: datetime,
) -> str:
    """Log a completed training run and return its run id.

    `now` is injected so a replayed or backfilled run records the time it
    represents rather than the time the backfill happened to execute.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    tags = {
        "mlflow.runName": run.run_name,
        "fraudlens.logged_at": now.isoformat(),
        # The signature is stored on the run, not only alongside a model
        # artifact, so a training-serving skew investigation can read the
        # expected input schema without materialising the model.
        "fraudlens.model_signature": run.signature.to_dict()["inputs"],
        "fraudlens.model_output": run.signature.to_dict()["outputs"],
        **run.provenance.as_tags(),
    }
    created = client.create_run(
        experiment_id=experiment_id,
        start_time=int(now.timestamp() * 1000),
        tags=tags,
    )
    run_id = str(created.info.run_id)

    for key, value in run.params.items():
        client.log_param(run_id, key, str(value))
    for key, value in run.metrics.as_mlflow_metrics().items():
        client.log_metric(run_id, key, value)

    client.set_terminated(run_id, "FINISHED")
    return run_id
