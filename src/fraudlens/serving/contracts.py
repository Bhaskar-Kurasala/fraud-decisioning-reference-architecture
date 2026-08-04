"""The HTTP contract for `POST /v1/decide`.

This is the trust boundary. ADR-0002 puts correctness of the decision economics first,
and the cheapest way to violate it is to price a transaction from a field the caller did
not mean: a string `"200"` silently floated, an unknown key ignored, a naive timestamp
that lands in the wrong day when the ledger is read across a DST boundary. So validation
is strict and total -- `strict=True` (no coercion), `extra="forbid"` (unknown fields are
a 422, not a shrug), `frozen=True` (a validated request cannot be edited downstream into
something that was never validated).

These models are also the OpenAPI document. ADR-0002 asks for well-documented APIs rather
than "read the source", so every field carries a description and an example; the generated
schema is intended to be sufficient to integrate against without opening this file.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field

from fraudlens.economics import ACTION_NAMES

# The service's action space. Narrower than `ACTION_NAMES` because this deployment runs
# the three-action policy (see `decisioning.POLICY_VERSION`); `review` is reachable only
# from the offline queue, which can check analyst capacity and this path cannot.
Action = Literal["allow", "challenge", "deny"]

# Fails at import if `economics` ever renames or drops an action, rather than at the first
# request that would have returned a name no client knows. The action vocabulary lives in
# `economics.ACTION_NAMES`; this is a subset assertion, not a second copy of it.
if not set(get_args(Action)).issubset(ACTION_NAMES):
    raise RuntimeError(f"serving action space {get_args(Action)} not a subset of {ACTION_NAMES}")

_STRICT = ConfigDict(strict=True, extra="forbid", frozen=True)


def _widen_json_integer(value: Any) -> Any:
    """Accept a JSON integer where a decimal is expected. Nothing else.

    Strict mode is what stops `"249.99"` being read as a number, and that protection is
    the point -- a string amount is a real money bug. But strict mode in Python-validation
    mode (which is what FastAPI runs) also rejects a plain `1` for a float field, and
    integer-valued inputs are the norm for both basket amounts and count features. So the
    one widening we allow is int -> float, before validation; `bool` is excluded because
    it is an `int` subclass in Python and a boolean amount is a bug, not a number.
    """
    return float(value) if type(value) is int else value


Number = Annotated[float, BeforeValidator(_widen_json_integer)]


class DecideRequest(BaseModel):
    """One transaction to decide on, at the moment of checkout."""

    model_config = _STRICT

    transaction_id: int = Field(
        ge=0,
        description="Caller's identifier for this transaction. Becomes the ledger key, "
        "so it must be unique and stable across retries.",
        examples=[2987000],
    )
    # `strict=False` on this field alone: an ISO-8601 string is the wire format, and strict
    # Python-mode validation would reject it. `AwareDatetime` still rejects a naive one,
    # which is the property that actually matters (DTZ is on for the same reason).
    transaction_at: Annotated[AwareDatetime, Field(strict=False)] = Field(
        description="When the transaction occurred. Must carry an offset -- a decision "
        "ledger with ambiguous timestamps cannot be audited across a DST boundary.",
        examples=["2026-08-05T14:32:10Z"],
    )
    amount: Number = Field(
        gt=0.0,
        description="Basket value in the merchant's settlement currency. Drives both "
        "cost arms, so the break-even probability moves with it.",
        examples=[249.99],
    )
    days_since_first_seen: Number | None = Field(
        default=None,
        ge=0.0,
        description="Days since this card was first seen (IEEE-CIS D1). Null when the "
        "signal is unavailable; the customer is then priced as an average one and the "
        "response says so via UNKNOWN_TENURE_PRICED_AS_MEDIAN.",
        examples=[3.0],
    )
    features: dict[str, Number] = Field(
        min_length=1,
        description="The model's input vector, already assembled by the caller. There is "
        "no feature store in this deployment (see FEATURE_VERSION), so the caller owns "
        "assembly and the service owns everything downstream of it.",
        examples=[{"C1": 1.0, "C13": 24.0, "V257": 0.0}],
    )


class VersionSet(BaseModel):
    """Everything needed to reproduce this decision. ADR-0002 priority #2.

    Returned on every response, not only on declines: a dispute is opened against a
    decision, and which decisions get disputed is not known when they are made.
    """

    model_config = _STRICT

    model_version: str = Field(description="Identifier of the loaded scoring artifact.")
    policy_version: str = Field(description="Which EV policy variant ran (three- or four-action).")
    feature_version: str = Field(description="Contract version of the supplied feature vector.")
    config_hash: str = Field(description="SHA-256 of the business constants the costs came from.")
    service_version: str = Field(description="Version of this service.")


class LatencyReport(BaseModel):
    """Per-stage wall time against the §4.3 budget, in milliseconds."""

    model_config = _STRICT

    features_ms: float = Field(description="Economic feature assembly. Budget 60 ms.")
    inference_ms: float = Field(description="Model score. Budget 25 ms.")
    calibration_policy_ms: float = Field(description="Calibration and EV argmax. Budget 15 ms.")
    total_ms: float = Field(description="End to end inside the handler. Budget 150 ms.")
    budget_breached: bool = Field(
        description="True if any stage exceeded its allocation. Also counted as an SLO "
        "metric; returned here so a caller can correlate a slow checkout without a join."
    )


class DecideResponse(BaseModel):
    """The decision, why it was reached, and what produced it."""

    model_config = _STRICT

    transaction_id: int
    action: Action = Field(
        description="What to do with the transaction. `challenge` is a step-up, not a decline."
    )
    calibrated_probability: Annotated[float | None, Field(ge=0.0, le=1.0)] = Field(
        description="Calibrated P(fraud), or null when no model produced it. Calibrated, "
        "not a raw score: an uncalibrated probability fed to the cost model is worth "
        "$4.36M/yr of avoidable loss (docs/findings/fit-balanced-empirical-result.md). "
        "Null rather than a placeholder on the degraded path -- a 0.0 there would be "
        "indistinguishable from a confident 'not fraud' to anyone who forgot to check "
        "`degraded`, and that misreading is how a fallback becomes a silent approve.",
    )
    reason_codes: Annotated[list[str], Field(min_length=1)] = Field(
        description="Stable machine identifiers, never display text. At least one is "
        "always present, including on approvals and on the fallback path.",
        examples=[["SCORE_ABOVE_DENY_BOUNDARY", "HIGH_AMOUNT_BAND", "NEW_ACCOUNT_TENURE"]],
    )
    degraded: bool = Field(
        description="True when the model path did not produce this decision. A degraded "
        "decision is a valid instruction to act on but must be excluded from calibration "
        "and performance analysis, which is why it is marked rather than hidden."
    )
    degraded_reason: str | None = Field(
        default=None,
        description="Cause of degradation, or null. Set together with `degraded`.",
    )
    decided_at: AwareDatetime = Field(description="When this service made the decision.")
    versions: VersionSet
    latency: LatencyReport


class HealthResponse(BaseModel):
    """Body of both health endpoints. Their *semantics* differ, not their shape."""

    model_config = _STRICT

    status: Literal["live", "ready", "not_ready"]
    detail: str | None = Field(default=None, description="Why not ready, when not ready.")
