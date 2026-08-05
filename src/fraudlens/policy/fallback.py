"""The rule ladder used when there is no model to decide with.

This is a policy, not a serving detail, and it lives here for a reason that only became
visible once decisions had to be replayed. The ladder originally sat in
`serving.decisioning`, which is the top layer — so `lineage.replay`, which sits below it,
could not reconstruct a degraded decision at all. That is precisely backwards: an outage
produces the largest single block of unusual decisions in the system's history, and that
block was the one block nobody could later prove was made correctly. Recorded as gap 2 in
docs/lineage.md before it was closed here.

**The ladder.** ADR-0002 priority #4: fail to the safe state, and for fraud the safe state
is not approve.

    high amount AND new/unknown account  ->  deny
    everything else                      ->  challenge

and never `allow`, on any input, ever.

Two decisions are load-bearing.

First, it does **not** call `decide`. The EV argmax needs a calibrated probability and in
degraded mode there is not one; feeding it an assumed prior would launder a guess through
the cost model and produce a decision that looks economically derived and is not. An
honest rule ladder is auditable, a fabricated input is not.

Second, the default rung is `challenge`, not `deny`. A challenge is friction the customer
can clear themselves; a decline is an irreversible adverse action, and taking one against
every customer on no evidence is not defensible to a regulator however safe it is for us.
Denying is reserved for the one case where an undetected fraud is most expensive and we
have the least reason to believe the account: a large basket on an account we have never
seen.
"""

from __future__ import annotations

from typing import Final, NamedTuple

# Versioned like the EV policy is, and for the same reason: a decision that cannot name
# the policy that produced it cannot be replayed. Bump this when either rung changes.
FALLBACK_VERSION: Final = "rules-ladder-v1"

# Rule identifiers. Plain strings rather than an enum because the reason-code taxonomy is
# a serving-layer contract that cannot be imported from here; `serving.reasons` takes its
# enum values from these constants so there is one definition rather than two that agree
# until someone edits one.
RULE_HIGH_AMOUNT_NEW_ACCOUNT: Final = "FALLBACK_RULE_HIGH_AMOUNT_NEW_ACCOUNT"
RULE_DEFAULT_CHALLENGE: Final = "FALLBACK_RULE_DEFAULT_CHALLENGE"

# BUSINESS ASSUMPTION, and this one carries money. On the scored path the decision
# boundary is continuous and per-transaction (`policy.boundaries`); here it is a step, and
# moving it moves the line between "we challenge you" and "we decline you" during an
# outage. It is set at $500 because that is where the published break-even is lowest
# (0.369 against 0.731 under $25) — the amount band where an undetected fraud is most
# expensive relative to a false decline, and therefore the only band where declining on no
# evidence is defensible.
FALLBACK_HIGH_AMOUNT: Final = 500.0

# An account is "new" to the ladder for its first week. Deliberately coarser than the
# tenure buckets the cost model uses: those price a risk, this one answers whether we have
# enough history to give the customer the benefit of the doubt with no score in hand.
FALLBACK_NEW_ACCOUNT_DAYS: Final = 7.0


class FallbackDecision(NamedTuple):
    """The action and the rung that produced it."""

    action: str
    rule: str


def is_high_amount_new_account(amount: float, days_since_first_seen: float | None) -> bool:
    """The one escalation in the ladder: big basket, account we have never seen.

    Uses the raw request fields rather than the bucketed tenure on purpose — in degraded
    mode the feature stage may be exactly what failed, so the fallback must not depend on
    it. A missing tenure signal counts as new: we cannot show the account is established,
    and on this path absence of evidence is not evidence of a relationship.
    """
    if amount < FALLBACK_HIGH_AMOUNT:
        return False
    return days_since_first_seen is None or days_since_first_seen <= FALLBACK_NEW_ACCOUNT_DAYS


def decide_without_model(amount: float, days_since_first_seen: float | None) -> FallbackDecision:
    """Choose an action with no score available. Never returns `allow`."""
    if is_high_amount_new_account(amount, days_since_first_seen):
        return FallbackDecision("deny", RULE_HIGH_AMOUNT_NEW_ACCOUNT)
    return FallbackDecision("challenge", RULE_DEFAULT_CHALLENGE)
