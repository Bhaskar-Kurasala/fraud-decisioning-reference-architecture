# The scoring container: serving + streaming, and deliberately not tracking.
#
# pyproject splits the extras by deployment unit because "the scoring container has no
# reason to carry MLflow's dependency tree, and shipping it would widen the attack surface
# of the only service on the checkout path". An image that installs `--all-extras` makes
# that split decorative, so this one installs exactly two extras and the composition root
# imports MLflow lazily to match. Streaming *is* here: auditability is ADR-0002's priority
# #2 and an image that decides but cannot record is not the production unit.
#
# Measured consequence of the split, on this machine: 984 MB with [serving,streaming] and
# 1.27 GB with tracking added — 286 MB of MLflow's server-side tree (alembic, gunicorn,
# graphene, its own docker client) sitting inside the checkout path and never called there.
#
# 984 MB is still not a small image, and the reason is worth writing down rather than
# hiding: pandas and pyarrow are *base* dependencies of the package, so the scoring
# container carries them even though the request path never reads a parquet file. Fixing
# that means moving them into an extra, which changes pyproject and would ripple through
# the research and streaming units — a separate decision from this one, recorded rather
# than quietly worked around here.

# Pinned by digest, not by tag. `python:3.10-slim-bookworm` is a moving target; a rebuild
# six months from now must produce the same base or the "regenerable by a single command"
# claim in §9a is false for the image.
FROM python:3.10-slim-bookworm@sha256:a2e667fa3c71dd34bc4a619e165743a0d4575829b6a4970f8096223f057040b7 AS builder

# Wheels only. If a dependency ever needs compiling, this fails here rather than silently
# pulling gcc into a layer we then have to remember to drop — and a build toolchain in the
# runtime image is a remote-code-execution primitive sitting next to the payment path.
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ONLY_BINARY=:all:

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir ".[serving,streaming]"

# ---------------------------------------------------------------------------------------

FROM python:3.10-slim-bookworm@sha256:a2e667fa3c71dd34bc4a619e165743a0d4575829b6a4970f8096223f057040b7

# Not root. The service parses attacker-influenced JSON on the checkout path and unpickles
# a model artifact at startup; both are code paths where a container escape starts with
# "and it was running as uid 0". No shell for the account either — nothing in the runtime
# image needs to log in as it.
RUN useradd --system --create-home --shell /usr/sbin/nologin --uid 10001 fraudlens

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER fraudlens
EXPOSE 8000

# Liveness, never readiness. A container whose model failed to load is answering correct
# fail-safe decisions; restarting it does not produce a model, and wiring HEALTHCHECK to
# /health/ready would put it in a restart loop precisely while it was doing the right
# thing. Readiness is a load-balancer question and is asked over HTTP by whatever routes
# traffic — Compose has no such concept, Kubernetes does, and deploy/k8s wires it there.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]

# One worker. The §4.3 budget is a tail-latency budget, and multiple workers behind a
# single process manager would each load their own copy of the artifact — same memory
# multiplied, and two concurrent requests decided by two separately-deserialised objects,
# which is unauditable. Horizontal scaling is replicas (ADR-0002: the service is stateless
# so that it can be), not workers.
CMD ["uvicorn", "fraudlens.serving.composition:build_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
