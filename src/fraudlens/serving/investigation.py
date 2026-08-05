"""Assemble the investigation view of one recorded decision, and solve its counterfactual.

**This module is off the §4.3 latency path and must never be imported into it.** §4.3
budgets `POST /v1/decide` at p99 150 ms; it budgets nothing here, because nothing here
blocks a checkout. The counterfactual below evaluates the policy nine times and the cost
model a dozen — trivial for a human reading one case, indefensible inside the decision.
`tests/serving/test_investigation.py` asserts that `decisioning` and `contracts` do not
import this module, so the separation is enforced rather than remembered.

**The counterfactual is the point of this endpoint.** The decision boundary is not a
learned artifact — it is an explicit rational function of amount and tenure
(`policy.boundaries`), so there is nothing to perturb or sample. Set the boundary equal to
the recorded probability, solve for amount, and you have the exact point at which the
action changes: one division, and it answers what a dispute handler actually asks.

The solve is then verified against `policy.decide`, which is not belt and braces. Under
the three-action policy the binary `allow_to_deny` break-even — the number findings §3
publishes and every write-up of this system quotes — is *not* operative, because a step-up
challenge stops 89% of fraud for a 7% abandonment cost and pushes the real deny boundary
well above it. Solving it yields a correct amount at which nothing happens; the oracle is
what drops it.

What this module cannot do is explain the *score*. See `REFUSAL`.
"""

from __future__ import annotations

from typing import Final, cast

from fraudlens.config import SETTINGS, TENURE_LABELS, UNKNOWN_TENURE
from fraudlens.economics import (
    ACTION_NAMES,
    false_negative_cost,
    false_positive_cost,
    relationship_cost,
    tenure_bucket,
)
from fraudlens.policy import boundaries, decide
from fraudlens.serving.case_contracts import (
    Absence,
    ActionFlip,
    AmountCounterfactual,
    Attribution,
    CaseAction,
    CaseResponse,
    DecisionFacts,
    EconomicContext,
    LedgerDecision,
    LedgerVersionSet,
    TenureCounterfactual,
    TenureOutcome,
)
from fraudlens.serving.decisioning import CONFIG_HASH, INCLUDE_REVIEW

# Every boundary in `policy.boundaries` has the shape N/(N+D) with N = alpha*M and
# D = beta*L. Carrying the two coefficients rather than three separately-solved formulas
# leaves one derivation to check instead of three.
_BOUNDARY_COEFFICIENTS: Final[dict[str, tuple[float, float]]] = {
    "allow_to_challenge": (SETTINGS.a_abandon, 1.0 - SETTINGS.f_pass),
    "challenge_to_deny": (1.0 - SETTINGS.a_abandon, SETTINGS.f_pass),
    "allow_to_deny": (1.0, 1.0),
}

# The fixed term of L. Not amount-dependent, which is the whole reason the break-even
# moves with amount at all.
_FIXED_FN_COST: Final = SETTINGS.cb_fee + SETTINGS.ops_dispute

# The only columns in the IEEE-CIS vector whose meaning the data vendor published.
# `C1`-`C14` count something Vesta never documented and `V1`-`V339` are undocumented
# engineered features; naming them in an explanation would be inventing the explanation.
DOCUMENTED_FEATURES: Final[tuple[str, ...]] = (
    "TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "D1",
)  # fmt: skip

ATTRIBUTION_UNAVAILABLE: Final = (
    "no scoring artifact is registered in this deployment (the research scripts persist "
    "scores, not estimators, and the MLflow Production pointer is dangling), and the "
    "ledger stores input_hash rather than the feature values (docs/lineage.md §3). A "
    "faithful per-feature contribution needs both. This field stays null until it does "
    "not; it will never be a zero."
)

REFUSAL: Final = (
    "This service will not narrate the anonymized features. Most of the model's input is "
    "undocumented by the vendor -- C1-C14 count something Vesta never specified, V1-V339 "
    "are engineered features with no published meaning -- so 'V258 contributed +0.31' is a "
    "number dressed as a reason, and 'the amount was 5x the customer's median' is not "
    "supportable at all: there is no customer history in the feature vector. A portion of "
    "this score is attributable to features nobody can explain, and saying so is the "
    "honest answer to a dispute. The defensible part of the decision is the economics, "
    "which is fully reconstructable and is what this response leads with."
)


def build_case(
    record: LedgerDecision,
    *,
    amount: float | None,
    days_since_first_seen: float | None,
) -> CaseResponse:
    """The whole view. `amount` is optional because the ledger does not hold it.

    Not an oversight either: the ledger records the decision, and the basket value lives on
    the order record the dispute desk already has open. Passing it in unlocks the economics
    and the counterfactual; omitting it leaves them null with a stated reason, which beats
    this service guessing an amount and then pricing a dispute from the guess.
    """
    absences: list[Absence] = []
    economics = tenure = None
    if amount is None:
        _absent(absences, "economics", "no `amount` supplied; the ledger does not store it")
    else:
        # NaN, never 0.0: D1 = 0 is a real value (the card's first ever transaction), so a
        # zero would price a customer we know nothing about as the riskiest possible one.
        d1 = float("nan") if days_since_first_seen is None else days_since_first_seen
        tenure = str(tenure_bucket([d1])[0])
        economics = _economics(record, amount, tenure)

    amount_cf = tenure_cf = None
    probability = record.calibrated_probability
    if probability is None:
        _absent(
            absences,
            "counterfactuals",
            "the decision was degraded, so no probability was recorded; asking what a "
            "different amount would have decided requires a score there was not one of",
        )
    elif amount is not None and tenure is not None:
        amount_cf = _amount_counterfactual(probability, amount, tenure)
        tenure_cf = _tenure_counterfactual(probability, amount, tenure)
    else:
        _absent(absences, "counterfactuals", "no `amount` supplied to vary against")

    _absent(absences, "attribution.contributions", ATTRIBUTION_UNAVAILABLE)
    return CaseResponse(
        transaction_id=record.transaction_id,
        decision=_facts(record),
        economics=economics,
        amount_counterfactual=amount_cf,
        tenure_counterfactual=tenure_cf,
        attribution=Attribution(
            contributions=None,
            unavailable_reason=ATTRIBUTION_UNAVAILABLE,
            documented_features=list(DOCUMENTED_FEATURES),
            refusal=REFUSAL,
        ),
        unavailable=absences,
    )


def _absent(into: list[Absence], field: str, reason: str) -> None:
    into.append(Absence(field=field, reason=reason))


def _facts(record: LedgerDecision) -> DecisionFacts:
    return DecisionFacts(
        action=cast(CaseAction, record.action),
        calibrated_probability=record.calibrated_probability,
        reason_codes=list(record.reason_codes),
        degraded=record.degraded,
        degraded_reason=record.degraded_reason,
        transaction_at=record.transaction_at,
        decided_at=record.decided_at,
        versions=LedgerVersionSet(
            model_version=record.model_version,
            policy_version=record.policy_version,
            feature_version=record.feature_version,
            config_hash=record.config_hash,
            input_hash=record.input_hash,
        ),
    )


def _boundaries(amount: float, tenure: str) -> dict[str, float]:
    fn = false_negative_cost([amount])
    fp = false_positive_cost([amount], [tenure])
    return {name: float(value[0]) for name, value in boundaries(fn, fp).items()}


def _action(probability: float, amount: float, tenure: str) -> CaseAction:
    """The action, from `policy.decide` itself. Never re-derived from a description of it:
    two corrections on this project came from someone reimplementing the policy in prose,
    costing $71,545/yr and $583/yr."""
    fn = false_negative_cost([amount])
    fp = false_positive_cost([amount], [tenure])
    index = int(decide(probability, fn, fp, include_review=INCLUDE_REVIEW)[0])
    return cast(CaseAction, ACTION_NAMES[index])


def _economics(record: LedgerDecision, amount: float, tenure: str) -> EconomicContext:
    bounds = _boundaries(amount, tenure)
    probability = record.calibrated_probability
    recomputed = None if probability is None else _action(probability, amount, tenure)
    return EconomicContext(
        amount=amount,
        tenure_bucket=tenure,
        false_negative_cost=float(false_negative_cost([amount])[0]),
        false_positive_cost=float(false_positive_cost([amount], [tenure])[0]),
        allow_to_challenge=bounds["allow_to_challenge"],
        challenge_to_deny=bounds["challenge_to_deny"],
        allow_to_deny=bounds["allow_to_deny"],
        recomputed_action=recomputed,
        matches_ledger_action=None if recomputed is None else recomputed == record.action,
        config_hash_matches_current=record.config_hash == CONFIG_HASH,
    )


def _flip_amount(boundary: str, probability: float, relationship: float) -> float | None:
    """Solve `boundary(amount) == probability` for amount. Closed form, one division.

    With N = alpha*(margin*a + R) and D = beta*(cogs*a + k), the boundary N/(N+D) equals p
    exactly when p*D = (1-p)*N, which is linear in a. A zero denominator means the
    boundary is flat in amount at this probability and no amount reaches it.
    """
    alpha, beta = _BOUNDARY_COEFFICIENTS[boundary]
    numerator = (1.0 - probability) * alpha * relationship - probability * beta * _FIXED_FN_COST
    denominator = probability * beta * SETTINGS.cogs - (1.0 - probability) * alpha * SETTINGS.margin
    if abs(denominator) < 1e-12:
        return None
    amount = numerator / denominator
    # A non-positive solve is a real answer, not a failure: it means no basket a customer
    # could have placed would have crossed this boundary at this score.
    return amount if amount > 0.0 else None


def _amount_counterfactual(probability: float, amount: float, tenure: str) -> AmountCounterfactual:
    observed = _action(probability, amount, tenure)
    relationship = float(relationship_cost([tenure])[0])
    flips: list[ActionFlip] = []
    for boundary in _BOUNDARY_COEFFICIENTS:
        solved = _flip_amount(boundary, probability, relationship)
        if solved is None:
            continue
        # The oracle. A solved amount is only a flip if `policy.decide` disagrees across
        # it; `allow_to_deny` in particular solves cleanly and flips nothing under the
        # three-action policy. Probed multiplicatively so the epsilon scales with the
        # basket rather than vanishing on a $5,000 one.
        below = _action(probability, solved * 0.999, tenure)
        above = _action(probability, solved * 1.001, tenure)
        if below != above:
            flips.append(
                ActionFlip(
                    amount=solved,
                    boundary=boundary,
                    boundary_at_observed_amount=_boundaries(amount, tenure)[boundary],
                    boundary_at_flip_amount=_boundaries(solved, tenure)[boundary],
                    action_below=below,
                    action_at_or_above=above,
                )
            )
    flips.sort(key=lambda flip: flip.amount)
    return AmountCounterfactual(
        observed_amount=amount,
        observed_action=observed,
        flips=flips,
        statement=_amount_statement(probability, amount, observed, tenure, flips),
    )


_BOUNDARY_PROSE: Final[dict[str, str]] = {
    "allow_to_challenge": "allow/challenge",
    "challenge_to_deny": "challenge/deny",
    "allow_to_deny": "allow/deny",
}


def _amount_statement(
    probability: float, amount: float, observed: CaseAction, tenure: str, flips: list[ActionFlip]
) -> str:
    """One sentence a dispute handler can put in their notes, quoting only live numbers.

    The threshold named here is the one the flip actually crosses. An earlier draft quoted
    the published `allow_to_deny` break-even alongside every flip, which reads plausibly
    and is wrong in the specific way this project keeps finding: at p=0.90 it printed
    "the break-even would have been 0.516 rather than 0.489" beside a decision driven by a
    challenge/deny boundary at 0.890. Both quoted numbers are below the score, so the
    sentence invites the reader to conclude the transaction was over the line by a mile
    when it cleared the operative threshold by 0.01.
    """
    if not flips:
        return (
            f"At this score no basket value changes the outcome: the action is {observed} "
            f"at every positive amount for an account in the {tenure} bucket."
        )
    parts = [
        f"the {_BOUNDARY_PROSE[f.boundary]} threshold is "
        f"{f.boundary_at_observed_amount:.3f} here and would have been "
        f"{f.boundary_at_flip_amount:.3f} at ${f.amount:,.2f}, so below that amount this "
        f"transaction would have been {f.action_below} rather than {f.action_at_or_above}"
        for f in flips
    ]
    return (
        f"Observed {observed} at ${amount:,.2f} on a score of {probability:.3f} "
        f"for an account in the {tenure} bucket; " + "; ".join(parts) + "."
    )


def _tenure_counterfactual(probability: float, amount: float, tenure: str) -> TenureCounterfactual:
    """Enumerate the buckets. Tenure is categorical, so there is nothing to solve — and
    eight exact evaluations beat any search over eight values."""
    outcomes = [
        TenureOutcome(
            tenure_bucket=bucket,
            allow_to_deny=_boundaries(amount, bucket)["allow_to_deny"],
            challenge_to_deny=_boundaries(amount, bucket)["challenge_to_deny"],
            action=_action(probability, amount, bucket),
            observed=bucket == tenure,
        )
        for bucket in (*TENURE_LABELS, UNKNOWN_TENURE)
    ]
    observed = _action(probability, amount, tenure)
    others = [o for o in outcomes if o.action != observed]
    statement = f"Observed {observed} at {tenure} tenure. " + (
        "No tenure bucket changes it."
        if not others
        else "The same score and basket would have been "
        + "; ".join(f"{o.action} at {o.tenure_bucket}" for o in others)
        + "."
    )
    return TenureCounterfactual(
        observed_tenure=tenure,
        observed_action=observed,
        by_tenure=outcomes,
        statement=statement,
    )
