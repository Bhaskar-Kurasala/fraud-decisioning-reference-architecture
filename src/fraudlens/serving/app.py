"""The FastAPI application: routes, lifespan, and the OpenAPI contract.

`create_app` takes every dependency explicitly. There is no module-level singleton that
reaches for a model server, so a test constructs a fully wired app with a stub scorer, a
frozen clock and a scripted elapsed-time source, and needs no network, no MLflow and no
`sleep` (§9a).

The OpenAPI document is intended to be the integration contract -- ADR-0002 asks for
well-documented APIs rather than "read the source" -- so responses, status codes and
field semantics are declared here rather than left to be inferred.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraudlens.serving.contracts import (
    DecideRequest,
    DecideResponse,
    HealthResponse,
    LatencyReport,
    VersionSet,
)
from fraudlens.serving.decisioning import (
    CONFIG_HASH,
    FEATURE_VERSION,
    POLICY_VERSION,
    SERVICE_VERSION,
    Outcome,
    decide_transaction,
    input_hash,
)
from fraudlens.serving.latency import (
    CALIBRATION_POLICY,
    FEATURES,
    INFERENCE,
    TOTAL,
    StageTimings,
)
from fraudlens.serving.metrics import DECISIONS
from fraudlens.serving.runtime import (
    Clock,
    DecisionWriter,
    Elapsed,
    ModelLoader,
    ServiceState,
    utc_now,
)

logger = logging.getLogger(__name__)

_DESCRIPTION = """
Synchronous fraud decisioning for the checkout path.

`POST /v1/decide` returns one of three actions -- `allow`, `challenge` (step-up
authentication, not a decline) or `deny` -- chosen by expected-value argmax over a
per-transaction cost model. The decision boundary is **not** a fixed threshold: it moves
with basket amount and account tenure, because the cost of an undetected fraud grows at
the cost-of-goods rate while the cost of a false decline grows at the margin rate plus a
fixed relationship cost. Concretely, we must be ~73% sure to decline a $20 order and only
~37% sure on a $500 one.

**Degradation.** If the model is unavailable the service still answers, on a documented
rule ladder, with `degraded: true` and `calibrated_probability: null`. It never returns
`allow` on that path. A degraded decision is safe to act on and must be excluded from
calibration and performance analysis.

**Reason codes** are stable machine identifiers, never display text. Rendering and
localisation belong to the caller. At least one is always present.
"""


def create_app(
    *,
    loader: ModelLoader,
    model_version_pin: str | None = None,
    clock: Clock = utc_now,
    elapsed: Elapsed = time.perf_counter,
    writer: DecisionWriter | None = None,
) -> FastAPI:
    """Build the service. `loader` runs once, in the lifespan startup hook."""
    state = ServiceState(loader, model_version_pin=model_version_pin)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load once per process. A failure here is recorded, not raised: the process must
        # come up so it can serve fail-safe decisions and report itself not-ready.
        state.load()
        yield

    app = FastAPI(
        title="FraudLens scoring API",
        version=SERVICE_VERSION,
        description=_DESCRIPTION,
        lifespan=lifespan,
    )
    _register_routes(app, state, clock=clock, elapsed=elapsed, writer=writer)
    return app


def _register_routes(
    app: FastAPI,
    state: ServiceState,
    *,
    clock: Clock,
    elapsed: Elapsed,
    writer: DecisionWriter | None,
) -> None:
    @app.post(
        "/v1/decide",
        response_model=DecideResponse,
        summary="Decide one transaction",
        response_description=(
            "The action to take, the calibrated probability behind it (null if degraded), "
            "stable reason codes, the exact version set used, and per-stage timings."
        ),
        responses={
            200: {"description": "A decision. May be degraded; check the `degraded` field."},
            422: {
                "description": "The request did not validate. No decision was made and "
                "nothing was written to the ledger -- a malformed request is a caller "
                "error, not a degraded decision, and must not be conflated with one."
            },
        },
    )
    def decide_endpoint(request: DecideRequest) -> DecideResponse:
        timings = StageTimings(elapsed)
        started = elapsed()
        outcome = decide_transaction(request, state.scorer, timings)
        decided_at = clock()
        timings.record(TOTAL, (elapsed() - started) * 1000.0)

        DECISIONS.labels(action=outcome.action, degraded=str(outcome.degraded).lower()).inc()
        versions = VersionSet(
            model_version=state.model_version,
            policy_version=POLICY_VERSION,
            feature_version=FEATURE_VERSION,
            config_hash=CONFIG_HASH,
            service_version=SERVICE_VERSION,
        )
        _persist(writer, request, outcome, decided_at, versions)
        return DecideResponse(
            transaction_id=request.transaction_id,
            action=outcome.action,  # type: ignore[arg-type]  # narrowed by the fallback ladder
            calibrated_probability=outcome.calibrated_probability,
            reason_codes=list(outcome.reason_codes),
            degraded=outcome.degraded,
            degraded_reason=outcome.degraded_reason,
            decided_at=decided_at,
            versions=versions,
            latency=_latency_report(timings),
        )

    @app.get(
        "/health/live",
        response_model=HealthResponse,
        summary="Liveness: is the process able to answer at all",
        description=(
            "Never consults the model. A pod serving correct fail-safe decisions is alive; "
            "restarting it does not produce a model and would turn a model degradation "
            "into a checkout outage. Aliasing this to readiness is the classic way to do "
            "exactly that."
        ),
    )
    def live() -> HealthResponse:
        return HealthResponse(status="live")

    @app.get(
        "/health/ready",
        response_model=HealthResponse,
        summary="Readiness: is the model loaded and can we serve a scored decision",
        description=(
            "503 when the model failed to load. The instance keeps answering `/v1/decide` "
            "in degraded mode while this is red -- readiness governs whether a load "
            "balancer should prefer this instance, not whether it is allowed to answer."
        ),
        responses={503: {"description": "Model not loaded. Decisions will be degraded."}},
    )
    def ready(response: Response) -> HealthResponse:
        if state.ready:
            return HealthResponse(status="ready")
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="not_ready", detail=state.failure or "model not loaded")

    @app.get("/metrics", summary="Prometheus exposition", include_in_schema=False)
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    _register_failsafe_handler(app)


def _register_failsafe_handler(app: FastAPI) -> None:
    """Last line of defence: an unhandled error must not look like a success upstream.

    `decide_transaction` already degrades safely, so reaching this means the failure was
    in the framework rather than in scoring -- serialisation, or a bug in this module. A
    503 is the honest answer there (§9a: "a fraud system that silently returns a default
    score is worse than one that returns 503"); the caller's own timeout policy then
    decides, rather than us inventing a decision from a state we do not understand.
    """

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Logged with the traceback: this branch means a bug in this module, and a 503 with
        # no stack behind it is how such a bug survives a quarter.
        logger.exception("unhandled error in scoring service")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": f"scoring service error: {type(exc).__name__}"},
        )


def _persist(
    writer: DecisionWriter | None,
    request: DecideRequest,
    outcome: Outcome,
    decided_at: dt.datetime,
    versions: VersionSet,
) -> None:
    """Append the decision to the audit trail, if one is wired.

    Failing to persist must not fail the decision: the customer is waiting and the action
    has already been chosen. The write is logged loudly instead, because a gap in an
    append-only ledger is an auditability incident even when the decision was correct.

    A degraded decision writes NULL for score and probability rather than a stand-in
    value. The ledger enforces the pairing, so "probability IS NULL" is a guarantee that
    the decision was made without a model — not a convention a later query has to
    remember.
    """
    if writer is None:
        return
    try:
        writer.record_decision(
            transaction_id=request.transaction_id,
            transaction_at=request.transaction_at,
            decided_at=decided_at,
            score=outcome.raw_score,
            calibrated_probability=outcome.calibrated_probability,
            action=outcome.action,
            reason_codes=outcome.reason_codes,
            model_version=versions.model_version,
            policy_version=versions.policy_version,
            feature_version=versions.feature_version,
            config_hash=versions.config_hash,
            input_hash=input_hash(request),
            degraded=outcome.degraded,
            degraded_reason=outcome.degraded_reason,
        )
    except Exception:
        logger.exception("ledger write failed for transaction %s", request.transaction_id)


def _latency_report(timings: StageTimings) -> LatencyReport:
    ms = timings.milliseconds
    return LatencyReport(
        # Zero when the stage never ran, which is the degraded case. Read alongside
        # `degraded`: a 0.0 inference time means no inference happened, not a fast one.
        features_ms=ms.get(FEATURES, 0.0),
        inference_ms=ms.get(INFERENCE, 0.0),
        calibration_policy_ms=ms.get(CALIBRATION_POLICY, 0.0),
        total_ms=ms.get(TOTAL, 0.0),
        budget_breached=timings.breached,
    )
