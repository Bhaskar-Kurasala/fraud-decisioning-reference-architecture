"""The HTTP contract for `GET /v1/cases/{transaction_id}` — the investigation view.

Deliberately separate from `contracts.py`. That module is the checkout-path trust
boundary and is validated strictly because a coerced string amount there is a money bug;
this one is an outbound read model assembled from rows we already wrote, so `strict=True`
would buy nothing and would reject the `numpy.float64` the cost model hands back. What is
shared between the two is a rule, not a type: **absence is a field state, not a value**
(ADR-0003). Every optional block below is null when it could not be computed honestly,
and every null is paired with an entry in `unavailable` saying why.

The consumers are named in `docs/analyst-integration.md`: dispute handlers, adverse-action
responders, on-call engineers, and whoever signs off a challenger promotion. There is no
queue here — no assignment, no claiming, no SLA timer. Findings §5 measured the value of
human review at positive for 5 of 92,427 transactions, so this deployment runs the
three-action policy and there is no work to distribute.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal, get_args

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from fraudlens.economics import ACTION_NAMES

# Wider than `contracts.Action`, which is the *emitting* contract for this deployment's
# three-action policy. This is the *reading* contract over an append-only ledger that
# already spans policy versions, so it must render a historical `review` row rather than
# fail on it. A case viewer that 500s on an action the current policy no longer emits is
# broken exactly when someone is investigating a policy change.
CaseAction = Literal["allow", "challenge", "review", "deny"]

if set(get_args(CaseAction)) != set(ACTION_NAMES):
    raise RuntimeError(f"case action space {get_args(CaseAction)} != {ACTION_NAMES}")

_READ_MODEL = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True)
class LedgerDecision:
    """The input side of this contract: one ledger row, as the case view needs it.

    A plain dataclass rather than `streaming.ledger.DecisionRecord`. That type lives in a
    module which imports SQLAlchemy at scope, and pyproject splits the extras by
    deployment unit so the scoring image can be built without it — same reasoning as the
    `DecisionWriter` port, in the read direction. The field names match the ledger's
    columns one to one so the adapter stays a splat and never becomes a translation.
    """

    transaction_id: int
    transaction_at: dt.datetime
    decided_at: dt.datetime
    calibrated_probability: float | None
    action: str
    reason_codes: tuple[str, ...]
    model_version: str
    policy_version: str
    feature_version: str
    config_hash: str
    input_hash: str
    degraded: bool
    degraded_reason: str | None


class Absence(BaseModel):
    """One thing this response could not tell you, and why.

    ADR-0003 §4: an empty field is ambiguous between "no data", "not applicable" and "not
    built", and on a screen ambiguity reads as calm. A dispute handler looking at a null
    counterfactual needs to know whether the transaction was degraded or whether they
    simply forgot to pass `amount`, and those two lead to different next actions.
    """

    model_config = _READ_MODEL

    field: str = Field(description="Dotted path of the field that is null.")
    reason: str = Field(description="Why it could not be computed. Not an error.")


class LedgerVersionSet(BaseModel):
    """Exactly the version columns the ledger holds. Not `contracts.VersionSet`.

    That one carries `service_version`, which the ledger does not record — so reusing it
    here would force this endpoint to fill the field with the version of whichever process
    happens to be serving the read. That is a fabricated observation about a decision made
    months ago, which is the defect ADR-0003 exists to stop.
    """

    model_config = _READ_MODEL

    model_version: str
    policy_version: str = Field(
        description="Names both the EV variant and the fallback ladder, so a replay knows "
        "which arithmetic produced this row."
    )
    feature_version: str
    config_hash: str = Field(description="SHA-256 of the business constants in force then.")
    input_hash: str = Field(
        description="Digest of the request, not a copy of it. The features are not in the "
        "ledger on purpose (docs/lineage.md §3): an append-only copy of customer-linked "
        "signals has no deletion path. This proves *which* input was used; it cannot show it."
    )


class DecisionFacts(BaseModel):
    """What the ledger recorded. Read straight through, nothing derived."""

    model_config = _READ_MODEL

    action: CaseAction
    calibrated_probability: float | None = Field(
        description="Calibrated P(fraud), or null when the decision was degraded. Null is "
        "a guarantee here, not a convention: `ck_decision_score_presence` pairs it with "
        "the degraded flag in the database."
    )
    reason_codes: list[str] = Field(
        description="Stable machine identifiers as written at decision time. Rendering is "
        "the caller's job; the audit trail must not need translating to be read later."
    )
    degraded: bool
    degraded_reason: str | None
    transaction_at: AwareDatetime
    decided_at: AwareDatetime
    versions: LedgerVersionSet


class EconomicContext(BaseModel):
    """The boundaries that applied to *this* transaction, recomputed from its own costs.

    There is no threshold in this system. `policy.boundaries` returns a different number
    for every transaction, and these are that transaction's.
    """

    model_config = _READ_MODEL

    amount: float
    tenure_bucket: str = Field(description="D1 bucket, or `unknown` when D1 was absent.")
    false_negative_cost: float = Field(description="L: amount x COGS + chargeback + ops.")
    false_positive_cost: float = Field(description="M: amount x margin + relationship cost.")
    allow_to_challenge: float = Field(description="Operative lower boundary under this policy.")
    challenge_to_deny: float = Field(description="Operative upper boundary under this policy.")
    allow_to_deny: float = Field(
        description="The binary allow/deny break-even, M/(L+M). This is the number the "
        "published anchors in findings §3 quote (0.740 at 1-7d tenure, 0.369 at $500+), "
        "and it is NOT operative in this three-action deployment -- a step-up challenge "
        "stops 89% of fraud for a 7% abandonment cost, which pushes the real deny boundary "
        "far above it. Reported because every external write-up of this policy cites it."
    )
    recomputed_action: CaseAction | None = Field(
        description="What today's constants and policy would decide on the recorded "
        "probability. Null when the decision was degraded, since the argmax needs a "
        "probability and there is not one."
    )
    matches_ledger_action: bool | None = Field(
        description="False means the recorded decision is not reproducible from today's "
        "configuration. That is a finding, not a bug: check `config_hash` against the "
        "current one before answering a dispute with a recomputed number."
    )
    config_hash_matches_current: bool = Field(
        description="Whether the constants that priced this decision are the ones running "
        "now. When false, everything above is a *counterfactual under today's constants*."
    )


class ActionFlip(BaseModel):
    """An exact amount at which this transaction's action changes, holding the score."""

    model_config = _READ_MODEL

    amount: float = Field(description="Solved, not sampled. The action differs either side.")
    boundary: str = Field(description="Which boundary crossing produces the flip.")
    # Deliberately *this* boundary at both amounts, not the published `allow_to_deny`.
    # Under the three-action policy the published binary break-even is not the boundary
    # anything crosses, so quoting it beside a flip invites the reader to check the score
    # against a threshold that had no part in the decision.
    boundary_at_observed_amount: float = Field(
        description="The flipping boundary evaluated at the amount actually transacted. "
        "The recorded score sits on one side of this."
    )
    boundary_at_flip_amount: float = Field(
        description="The same boundary at the flip amount. Equals the recorded "
        "probability by construction -- a free check that the closed-form solve is right."
    )
    action_below: CaseAction
    action_at_or_above: CaseAction


class AmountCounterfactual(BaseModel):
    """Would a smaller basket have been allowed? Solved in closed form.

    The boundary is an explicit rational function of amount, so there is nothing to
    perturb or sample: setting it equal to the recorded probability and solving for amount
    gives the exact point where the action changes. A sampled perturbation would answer a
    weaker question more slowly and with an arbitrary grid.
    """

    model_config = _READ_MODEL

    observed_amount: float
    observed_action: CaseAction
    # No `observed_allow_to_deny` here: all three boundaries at the observed amount are
    # already in `economics`, which is present under exactly the same condition (an
    # `amount` was supplied). A second copy is a second thing that can disagree.
    flips: list[ActionFlip] = Field(
        description="Ordered by amount. Empty means no positive basket value changes the "
        "action at this score -- which is the common case, and is itself the answer."
    )
    statement: str = Field(
        description="The above in one sentence, for an operator's notes. Operator-facing, "
        "never customer-facing: the numbers are the contract, this is a convenience."
    )


class TenureOutcome(BaseModel):
    """What this transaction would have been, had the account been in this bucket."""

    model_config = _READ_MODEL

    tenure_bucket: str
    allow_to_deny: float
    challenge_to_deny: float
    action: CaseAction
    observed: bool = Field(description="True for the bucket the transaction was actually in.")


class TenureCounterfactual(BaseModel):
    """Tenure is categorical, so this enumerates rather than solves. Seven buckets plus
    `unknown`; enumeration is exact and cheaper than any search over them."""

    model_config = _READ_MODEL

    observed_tenure: str
    observed_action: CaseAction
    by_tenure: list[TenureOutcome]
    statement: str


class Attribution(BaseModel):
    """What of the score we can honestly name, and what we refuse to narrate.

    A fabricated explanation on an adverse-action notice is a compliance problem, not a
    quality one, which is why the refusal is a field rather than a docstring.
    """

    model_config = _READ_MODEL

    contributions: list[str] | None = Field(
        description="Per-feature contributions. Null in this deployment -- see "
        "`unavailable_reason`. Null and not an empty list: an empty list reads as 'we "
        "checked and nothing contributed', which is a much stronger claim than the truth."
    )
    unavailable_reason: str | None
    documented_features: list[str] = Field(
        description="The only features in the vector whose meaning is published by the "
        "data vendor. An attribution, when one becomes computable, is reported over these "
        "and no others."
    )
    refusal: str = Field(
        description="What this endpoint will not emit, and why. Stable text: it is "
        "intended to be quotable in a dispute response."
    )


class CaseResponse(BaseModel):
    """One recorded decision, everything derivable from it, and what is not derivable."""

    model_config = _READ_MODEL

    transaction_id: int
    decision: DecisionFacts
    economics: EconomicContext | None
    amount_counterfactual: AmountCounterfactual | None
    tenure_counterfactual: TenureCounterfactual | None
    attribution: Attribution
    unavailable: list[Absence] = Field(
        description="One entry per null block above. Empty when everything was computable."
    )
