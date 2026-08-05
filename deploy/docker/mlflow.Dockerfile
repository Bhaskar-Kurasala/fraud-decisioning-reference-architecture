# The MLflow server: the shared tracking + registry backend.
#
# This closes docs/lineage.md gap 8. Today `scripts/regenerate_model_card.py` writes into a
# local `mlruns/mlflow.db`, so the model card names a run id that points into a store
# nobody else has and two people regenerating the same card get different run ids for the
# same numbers. A server makes the run pointer resolvable by someone other than the person
# who created it, which is the claim the card's provenance line already makes.
#
# Its own image rather than `ghcr.io/mlflow/mlflow`: the version has to be the one the
# `tracking` extra resolves to, because the registry client in the composition root and the
# server have to agree about the schema, and an upstream tag floats independently of our
# lockfile.

FROM python:3.10-slim-bookworm@sha256:a2e667fa3c71dd34bc4a619e165743a0d4575829b6a4970f8096223f057040b7

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

# Exact pin, matching what `uv sync --extra tracking` resolves to.
RUN pip install --no-cache-dir "mlflow==3.15.1"

RUN useradd --system --create-home --shell /usr/sbin/nologin --uid 10002 mlflow \
    && mkdir -p /mlflow/artifacts && chown -R mlflow:mlflow /mlflow

USER mlflow
WORKDIR /mlflow
EXPOSE 5000

# SQLite on a volume, not Postgres. What gap 8 asks for is a *shared* backend, and the
# sharing comes from the server being reachable over HTTP — not from the storage engine.
# `models.tracking.local_tracking_uri` already chose SQLite for a documented reason (MLflow
# 3 put the `./mlruns` file store into maintenance mode and the file store never supported
# the registry), so using the same engine here keeps one code path rather than two. The
# cost, stated: SQLite serialises writers, which is fine for one training client and would
# not be for a fleet — and a fleet is exactly what ADR-0002 declines to build.
#
# `--serve-artifacts` so clients fetch through the server rather than needing the artifact
# volume mounted themselves. The scoring container downloads its bundle over HTTP for that
# reason: it must not need a shared filesystem with the training host, which in a real
# deployment is in a different trust zone.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", \
     "--serve-artifacts"]
