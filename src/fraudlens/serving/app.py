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

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from opentelemetry.trace import Tracer, TracerProvider
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraudlens.serving.audit import persist
from fraudlens.serving.case_view import register_case_view
from fraudlens.serving.cases import register_case_routes
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
    decide_transaction,
)
from fraudlens.serving.latency import (
    CALIBRATION_POLICY,
    FEATURES,
    INFERENCE,
    TOTAL,
    StageTimings,
)
from fraudlens.serving.metrics import (
    MODEL_LOAD_SECONDS,
    MODEL_LOADED,
    observe_decision,
)
from fraudlens.serving.middleware import register_failsafe_handler, register_request_counter
from fraudlens.serving.runtime import (
    Clock,
    DecisionWriter,
    Elapsed,
    ModelLoader,
    ServiceState,
    utc_now,
)
from fraudlens.serving.tracing import record_decision_trace, tracer_for

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
    tracer_provider: TracerProvider | None = None,
) -> FastAPI:
    """Build the service. `loader` runs once, in the lifespan startup hook.

    `tracer_provider` defaults to None, which means no spans and no exporter. Tracing is
    a parameter for the same reason the loader and the clock are: importing this module
    must never open a connection, and an env-var-driven exporter would make that property
    depend on the environment a test happens to run in.
    """
    state = ServiceState(loader, model_version_pin=model_version_pin)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Load once per process. A failure here is recorded, not raised: the process must
        # come up so it can serve fail-safe decisions and report itself not-ready.
        # `time.perf_counter`, deliberately not the injected `elapsed`. That seam exists to
        # let a test script the *request* stage timings; drawing two values from it at
        # startup would silently shift every scripted sequence by two steps and the latency
        # tests would then be asserting on the wrong stage. Startup is not in the §4.3
        # budget, so it does not need the injected clock.
        started = time.perf_counter()
        state.load()
        # Timed because a load slow enough to trip a start-up probe gets the pod killed and
        # retried forever, which from outside is indistinguishable from a crash loop. Zero
        # on failure: reporting the time-to-fail as a load time would flatter it.
        MODEL_LOAD_SECONDS.set(time.perf_counter() - started if state.ready else 0.0)
        MODEL_LOADED.set(1 if state.ready else 0)
        yield

    app = FastAPI(
        title="FraudLens scoring API",
        version=SERVICE_VERSION,
        description=_DESCRIPTION,
        lifespan=lifespan,
    )
    _register_routes(
        app,
        state,
        clock=clock,
        elapsed=elapsed,
        writer=writer,
        tracer=tracer_for(tracer_provider),
    )
    return app


def _register_routes(
    app: FastAPI,
    state: ServiceState,
    *,
    clock: Clock,
    elapsed: Elapsed,
    writer: DecisionWriter | None,
    tracer: Tracer,
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
        with tracer.start_as_current_span(TOTAL) as span:
            # Wall-clock anchor for the reconstructed stage spans. Separate from
            # `elapsed`, which is monotonic and has no epoch a tracing backend can plot.
            anchor_ns = time.time_ns()
            outcome = decide_transaction(request, state.scorer, timings)
            decided_at = clock()
            timings.record(TOTAL, (elapsed() - started) * 1000.0)
            versions = VersionSet(
                model_version=state.model_version,
                policy_version=POLICY_VERSION,
                feature_version=FEATURE_VERSION,
                config_hash=CONFIG_HASH,
                service_version=SERVICE_VERSION,
            )
            observe_decision(
                action=outcome.action,
                degraded=outcome.degraded,
                amount=request.amount,
                calibrated_probability=outcome.calibrated_probability,
                reason_codes=outcome.reason_codes,
            )
            record_decision_trace(
                tracer, span, request=request, outcome=outcome, versions=versions,
                timings=timings, anchor_ns=anchor_ns,
            )  # fmt: skip
        persist(writer, request, outcome, decided_at, versions)
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

    register_request_counter(app)
    register_failsafe_handler(app)
    # The investigation view (E12). Takes no state from here on purpose: it reads the
    # ledger through its own dependency and is off the §4.3 budget, so it must not add an
    # argument to the constructor of the path that has one.
    register_case_routes(app)
    # The same case, rendered for a human, sharing the same ledger dependency. Registered
    # beside the JSON route rather than served as a static file from a sidecar: an
    # air-gapped estate should not need a second container to read one decision.
    register_case_view(app)


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
