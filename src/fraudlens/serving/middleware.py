"""What wraps every response, regardless of which route produced it.

Two concerns live here, and they are the same concern seen from either side: what we
record about a response, and what we return when no route was able to produce one. Both
apply to every request and neither belongs to any endpoint, which is why they were the
part of `app.py` that made it read as three modules stapled together.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from fraudlens.serving.metrics import REQUESTS

logger = logging.getLogger(__name__)


def register_request_counter(app: FastAPI) -> None:
    """Count every response by route template and status. Tier 3 throughput and errors.

    `fraudlens_decisions_total` counts decisions, which is a different population: a 422
    produced no decision, and that difference is the point. The 422 rate on `/v1/decide`
    is the Tier 4 schema-violation rate, since `contracts` validates strictly and forbids
    unknown fields; the 503 rate is the fail-safe handler, meaning a bug in this module.
    """

    @app.middleware("http")
    async def count(request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        # Route template, never `request.url.path`: a scanner walking /.env, /admin and a
        # thousand friends would otherwise mint a time series per probe and take the
        # registry down with it. Unmatched paths collapse to one label for that reason.
        route = getattr(request.scope.get("route"), "path", "unmatched")
        REQUESTS.labels(route=route, status=str(response.status_code)).inc()
        return response


def register_failsafe_handler(app: FastAPI) -> None:
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
