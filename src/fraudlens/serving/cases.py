"""Route registration for the investigation view. Off the checkout path.

Registered from `app.create_app` the same way the request counter and the fail-safe
handler are, and for the same reason: it applies to the application rather than to any
one handler, so it does not belong inline in `_register_routes`.

**Why the ledger reader is a FastAPI dependency and not a `create_app` argument.** The
scoring service is constructed by `composition.build_app` from the environment, and a
deployment that serves decisions without a database is real — the load test and the k8s
smoke both run it. Threading a reader through `create_app` would put a second optional
port on the constructor of the hot path for the benefit of a route that is not on it.
`app.dependency_overrides[case_source] = ...` is the framework's own seam and keeps the
wiring where the wiring already lives. Unconfigured, the route answers 503 with the
reason, not 404: "this deployment has no ledger reader" and "we have no record of that
transaction" are different answers and a dispute handler must not confuse them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path, Query, status

from fraudlens.serving.case_contracts import CaseResponse, LedgerDecision
from fraudlens.serving.investigation import build_case

# Returns the recorded decision, or None if this transaction was never decided by us.
# A callable rather than a class: it has no state and one method, matching `ModelLoader`.
CaseSource = Callable[[int], LedgerDecision | None]


def case_source() -> CaseSource | None:
    """The unconfigured default. Overridden per deployment; see the module docstring."""
    return None


_NOT_CONFIGURED = (
    "no decision ledger is wired into this instance, so no case can be read. This is a "
    "deployment configuration, not a missing transaction."
)


def register_case_routes(app: FastAPI) -> None:
    """Mount `GET /v1/cases/{transaction_id}`."""

    @app.get(
        "/v1/cases/{transaction_id}",
        response_model=CaseResponse,
        summary="The investigation view of one recorded decision",
        description=(
            "Everything reconstructable about one decision: the action, the reason codes, "
            "the full version set, whether it was degraded, the calibrated probability "
            "(null when it was), the economic boundaries that applied to this specific "
            "transaction, and the closed-form counterfactual over basket amount and "
            "account tenure.\n\n"
            "**Not on the 150 ms path.** §4.3 budgets `POST /v1/decide`; this endpoint is "
            "read-only, off the checkout path, and has no latency SLO.\n\n"
            "**Not a review queue.** There is no assignment, no claiming and no SLA "
            "timer, because there is no queue: findings §5 measured the value of human "
            "review as positive for 5 of 92,427 transactions, so this deployment runs the "
            "three-action policy and there is no work to distribute. The consumers are "
            "dispute handlers, adverse-action responders, on-call engineers and challenger "
            "promotion sign-off. See `docs/analyst-integration.md`.\n\n"
            "`amount` and `days_since_first_seen` are query parameters because the ledger "
            "stores neither -- it holds `input_hash`, not the request (docs/lineage.md "
            "§3). Supply them from the order record to unlock the economics and the "
            "counterfactual; omit them and those blocks are null with a stated reason."
        ),
        responses={
            404: {"description": "No decision was ever recorded for this transaction id."},
            503: {"description": "This instance has no ledger reader configured."},
        },
    )
    def read_case(
        transaction_id: Annotated[int, Path(ge=0, description="The ledger key.")],
        amount: Annotated[
            float | None,
            Query(gt=0.0, description="Basket value, from the order record. Prices both arms."),
        ] = None,
        days_since_first_seen: Annotated[
            float | None,
            Query(
                ge=0.0,
                description="IEEE-CIS D1. Omit when unknown -- it is priced as the median "
                "bucket and the response says so, which is different from claiming 0 days.",
            ),
        ] = None,
        source: Annotated[CaseSource | None, Depends(case_source)] = None,
    ) -> CaseResponse:
        if source is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, _NOT_CONFIGURED)
        record = source(transaction_id)
        if record is None:
            # 404 rather than an empty case. A decision that was made but never reached
            # the ledger is invisible here by construction, which is what
            # `fraudlens_ledger_write_failures_total` exists to make countable.
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"no decision recorded for transaction {transaction_id}",
            )
        return build_case(record, amount=amount, days_since_first_seen=days_since_first_seen)
