"""Stable machine identifiers explaining why a decision was reached.

These exist so an adverse decision can be defended to the customer who disputes it and
to the regulator who asks how it was made (ADR-0002 priority #2, SR 11-7 / EU AI Act
adverse-action posture). Three properties follow from that and are enforced by test:

1. They are **identifiers, never display text**. The string `NEW_ACCOUNT_TENURE` is a
   join key into whatever renders customer-facing copy; changing the copy must not
   change the audit trail, and the audit trail must not need translating to be read in
   six months. i18n is explicitly out of scope for this reason (ADR-0002).
2. Every decision carries **at least one**, including approvals. An unexplained approve
   is as unauditable as an unexplained decline, and the fail-safe path is exactly where
   an empty list would otherwise slip through.
3. They **annotate** the decision, they do not make it. The action comes from
   `policy.decide`; these codes describe the inputs that put it there. If a code and
   the action ever disagree, the code is wrong -- never the other way round.
"""

from __future__ import annotations

from enum import Enum

from fraudlens.config import UNKNOWN_TENURE
from fraudlens.policy import fallback

# Amount bands, for explanation only on the scored path: the decision boundary there is
# per-transaction and continuous (`policy.boundaries`) and these are the coarse buckets a
# human reads. The split points are where the break-even inversion is most visible, 0.740
# at the low end against 0.369 for $500+ baskets.
#
# The high band takes its value from `policy.fallback` rather than restating 500.0. An
# earlier version of this comment claimed the bands "carry no money -- moving them changes
# what a dispute letter says, not what we decide", and that was wrong: the same threshold
# is the deny rung of the degraded ladder, where it decides. Importing it keeps the
# dispute letter's notion of "high amount" identical to the one the ladder acted on, which
# is the only version of this that survives a customer asking why.
_HIGH_AMOUNT = fallback.FALLBACK_HIGH_AMOUNT
_LOW_AMOUNT = 20.0

# Tenure buckets whose relationship cost dominates the false-positive term, which is why
# a decline here needs the strongest evidence. Values must exist in `config.TENURE_LABELS`.
_NEW_ACCOUNT_TENURES = frozenset({"new(0d)", "1-7d"})
_ESTABLISHED_TENURES = frozenset({"181-400d", "400d+"})


class ReasonCode(str, Enum):
    """The closed taxonomy. Adding a member is a contract change; renaming one breaks
    every historical decision record that cites it, so members are append-only."""

    # --- where the calibrated probability sat relative to this transaction's own
    # break-even boundaries. Exactly one of these is always emitted.
    SCORE_ABOVE_DENY_BOUNDARY = "SCORE_ABOVE_DENY_BOUNDARY"
    SCORE_ABOVE_CHALLENGE_BOUNDARY = "SCORE_ABOVE_CHALLENGE_BOUNDARY"
    SCORE_BELOW_ALL_BOUNDARIES = "SCORE_BELOW_ALL_BOUNDARIES"

    # --- the economic context that placed those boundaries where they were.
    HIGH_AMOUNT_BAND = "HIGH_AMOUNT_BAND"
    LOW_AMOUNT_BAND = "LOW_AMOUNT_BAND"
    NEW_ACCOUNT_TENURE = "NEW_ACCOUNT_TENURE"
    ESTABLISHED_ACCOUNT_TENURE = "ESTABLISHED_ACCOUNT_TENURE"
    # Surfaced rather than swallowed: an unidentified customer is priced as an average
    # one (`economics.costs.relationship_cost`), which is an assumption a disputing
    # customer is entitled to know was applied to them.
    UNKNOWN_TENURE_PRICED_AS_MEDIAN = "UNKNOWN_TENURE_PRICED_AS_MEDIAN"

    # --- degradation. Present iff `degraded` is true on the response and in the ledger.
    DEGRADED_MODEL_UNAVAILABLE = "DEGRADED_MODEL_UNAVAILABLE"
    DEGRADED_SCORING_ERROR = "DEGRADED_SCORING_ERROR"

    # --- which rung of the fallback ladder fired. Values come from `policy.fallback`,
    # which owns the ladder: the rung that fired and the code that reports it must be the
    # same string, and two literals that agree today are two literals that can diverge.
    FALLBACK_RULE_HIGH_AMOUNT_NEW_ACCOUNT = fallback.RULE_HIGH_AMOUNT_NEW_ACCOUNT
    FALLBACK_RULE_DEFAULT_CHALLENGE = fallback.RULE_DEFAULT_CHALLENGE


def score_band_reason(
    probability: float,
    *,
    challenge_boundary: float,
    deny_boundary: float,
) -> ReasonCode:
    """Which side of this transaction's own boundaries the calibrated score fell.

    Boundaries are supplied by `policy.boundaries`, never recomputed here: the whole
    point of the reason code is that it cites the same arithmetic the decision used.
    """
    if probability >= deny_boundary:
        return ReasonCode.SCORE_ABOVE_DENY_BOUNDARY
    if probability >= challenge_boundary:
        return ReasonCode.SCORE_ABOVE_CHALLENGE_BOUNDARY
    return ReasonCode.SCORE_BELOW_ALL_BOUNDARIES


def context_reasons(amount: float, tenure: str) -> list[ReasonCode]:
    """The amount and tenure facts a reviewer needs to read the decision.

    May legitimately be empty (a mid-band amount on a mid-tenure account is unremarkable);
    the caller is what guarantees the response is never reason-less.
    """
    codes: list[ReasonCode] = []
    if amount >= _HIGH_AMOUNT:
        codes.append(ReasonCode.HIGH_AMOUNT_BAND)
    elif amount < _LOW_AMOUNT:
        codes.append(ReasonCode.LOW_AMOUNT_BAND)
    if tenure == UNKNOWN_TENURE:
        codes.append(ReasonCode.UNKNOWN_TENURE_PRICED_AS_MEDIAN)
    elif tenure in _NEW_ACCOUNT_TENURES:
        codes.append(ReasonCode.NEW_ACCOUNT_TENURE)
    elif tenure in _ESTABLISHED_TENURES:
        codes.append(ReasonCode.ESTABLISHED_ACCOUNT_TENURE)
    return codes
