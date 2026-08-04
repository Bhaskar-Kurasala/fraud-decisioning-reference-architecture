"""Re-derive a ledger row's action from the versions the row itself records.

§9a: "A decision in the ledger can be replayed: same inputs + same recorded model and
policy versions must produce the same output. This is asserted by a test, not assumed."
This module is the executable half of that sentence; `tests/lineage/test_replay.py` is
the assertion.

**What is reconstructed, and what is not.** The ledger deliberately stores `input_hash`
rather than the feature vector (see `serving.decisioning.input_hash`): the features are
the caller's data and copying them into an append-only audit table would create a second
permanent store of customer-linked signals with no deletion path. The consequence is
exact and must not be glossed:

- The **policy arm is replayed**. Given the recorded calibrated probability and the
  original request, the cost model and the EV argmax are re-run and the action is
  compared. A config change, a cost-function change or a policy-version change shows up
  here as a diff.
- The **model arm is not replayed**. Features -> raw score -> calibrated probability is
  not recomputed, because the features are not in the ledger and the artifact named by
  `model_version` may no longer be loadable. `model_version` is therefore *evidence*
  ("this artifact scored it"), not something this function verifies.
- Whether the caller supplied the *right* input is checked by hash, not assumed. That is
  the whole reason `input_hash` is on the row: a replay that disagrees on the action can
  be attributed either to a differing input or to a genuine divergence, and those two
  have completely different consequences in a dispute.

So the honest claim is: *same recorded probability + same recorded input + same recorded
config and policy version => same action, and any deviation is localised.* That is
weaker than "the whole decision replays" and it is what the schema can support. See
docs/lineage.md for the gaps and what closing each would cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from fraudlens.config import SETTINGS, BusinessConstants
from fraudlens.economics import (
    ACTION_NAMES,
    false_negative_cost,
    false_positive_cost,
    tenure_bucket,
)
from fraudlens.models.provenance import config_hash
from fraudlens.policy import decide
from fraudlens.streaming.ledger import DecisionRecord

# policy_version -> whether that policy offered the analyst-review arm.
#
# A lookup rather than a default, because the two variants are different policies with
# different published costs ($2,799,214 without review, $2,799,797 with) and replaying a
# row under the wrong one silently produces a plausible action. An unknown version is
# refused: a ledger row written by a policy this build has never heard of must not be
# quietly replayed under today's policy and reported as "matched".
#
# `tests/lineage/test_replay.py::test_serving_policy_version_is_registered` fails if
# serving's POLICY_VERSION drifts from this table, which is what stops the table from
# becoming a stale second opinion about what the service runs.
POLICY_VARIANTS: Mapping[str, bool] = MappingProxyType({"ev-argmax-3action-v1": False})


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """The diff between the recorded action and the re-derived one."""

    transaction_id: int
    recorded_action: str
    # None when the row could not be replayed at all; `unreplayable_reason` says why.
    replayed_action: str | None
    input_matches: bool
    config_matches: bool
    unreplayable_reason: str | None

    @property
    def matched(self) -> bool:
        """The two actions agree. Necessary for a verification, not sufficient for one."""
        return self.replayed_action is not None and self.replayed_action == self.recorded_action

    @property
    def verified(self) -> bool:
        """The recorded decision was re-derived from the input it was actually made on.

        Separate from `matched` because agreement on a *different* input is not evidence
        of anything: with only three actions, two unrelated transactions agree by chance
        far too often to treat a bare action match as a verification. This is the property
        an audit should assert on.
        """
        return self.matched and self.input_matches

    @property
    def verdict(self) -> str:
        """One sentence an auditor can act on.

        The failure modes are not interchangeable and must not collapse into "replay
        failed": a differing input is the caller's mistake, a differing config is a
        change-management finding, and a divergence on identical inputs and config is a
        defect in the decision path. The input check is reported *before* the action
        comparison for the reason `verified` exists — an action that agrees on the wrong
        input is a coincidence being read as proof.
        """
        if self.unreplayable_reason is not None:
            return f"not replayable: {self.unreplayable_reason}"
        if not self.input_matches:
            return (
                f"unverified: the supplied input does not hash to the recorded input_hash, "
                f"so replaying it (recorded {self.recorded_action}, replayed "
                f"{self.replayed_action}) says nothing about the recorded decision"
            )
        if self.matched:
            return f"reproduced: {self.recorded_action}"
        if not self.config_matches:
            return (
                f"recorded {self.recorded_action}, replayed {self.replayed_action}; the "
                "business config has changed since the decision (config_hash differs), "
                "which moves every cost and therefore every boundary"
            )
        return (
            f"DIVERGENCE: recorded {self.recorded_action}, replayed {self.replayed_action} "
            "on the same input and the same config — the decision path no longer agrees "
            "with its own audit trail"
        )


def replay_decision(
    record: DecisionRecord,
    payload: Mapping[str, Any],
    *,
    c: BusinessConstants = SETTINGS,
) -> ReplayResult:
    """Re-derive `record`'s action from `payload` and the versions on the row.

    `payload` is the original `/v1/decide` request body as JSON-compatible types — the
    same shape `serving.decisioning.input_hash` digested. It is supplied by the caller
    because the ledger does not hold it; supplying the wrong one is detected rather than
    trusted.

    `c` is the business config to replay under. Passing today's config answers "would we
    decide this the same way now"; passing the config that was current at decision time
    answers "was the decision correct when it was made". Both are legitimate questions
    and `config_matches` says which one was asked.
    """
    input_matches = config_hash(payload) == record.input_hash
    config_matches = config_hash(c.model_dump()) == record.config_hash

    reason = _unreplayable_reason(record, payload)
    if reason is not None:
        return _refused(record, reason, input_matches, config_matches)

    probability = record.calibrated_probability
    if probability is None:
        # Unreachable through the ledger, whose CHECK constraint makes a null probability
        # equivalent to `degraded` — already refused above. Kept as a branch rather than
        # a cast so that relaxing that constraint produces a refusal here instead of the
        # policy being run on a fabricated score.
        return _refused(
            record, "no calibrated probability on the row", input_matches, config_matches
        )

    include_review = POLICY_VARIANTS[record.policy_version]
    # NaN, not 0.0, for an absent D1 — mirrors the serving path exactly. 0.0 is a real
    # value (the card's first ever transaction) and would replay an unknown-tenure
    # customer as the riskiest possible one.
    raw_d1 = payload.get("days_since_first_seen")
    d1 = np.nan if raw_d1 is None else float(raw_d1)
    amount = [float(payload["amount"])]
    tenure = tenure_bucket([d1])
    fn = false_negative_cost(amount, c)
    fp = false_positive_cost(amount, tenure, c)
    action_index = int(decide(probability, fn, fp, include_review=include_review, c=c)[0])
    return ReplayResult(
        transaction_id=record.transaction_id,
        recorded_action=record.action,
        replayed_action=ACTION_NAMES[action_index],
        input_matches=input_matches,
        config_matches=config_matches,
        unreplayable_reason=None,
    )


def _refused(
    record: DecisionRecord,
    reason: str,
    input_matches: bool,
    config_matches: bool,
) -> ReplayResult:
    """A result carrying no replayed action. The hash checks still stand and are reported:
    knowing the input and the config match is useful even when the action cannot be
    re-derived, because it narrows what an investigator has to look at."""
    return ReplayResult(
        transaction_id=record.transaction_id,
        recorded_action=record.action,
        replayed_action=None,
        input_matches=input_matches,
        config_matches=config_matches,
        unreplayable_reason=reason,
    )


def _unreplayable_reason(record: DecisionRecord, payload: Mapping[str, Any]) -> str | None:
    """Why this row cannot be re-derived here, or None if it can.

    Each branch is a known limit of the audit trail rather than a defensive check, and
    each is listed in docs/lineage.md with the cost of closing it.
    """
    if record.degraded:
        # The fail-safe rule ladder lives in `serving.decisioning`, above this layer in
        # the import contract, so it cannot be called from here — and reimplementing it
        # from its prose is exactly the mistake that has already cost this project two
        # corrections. Closing this means moving the ladder down to `policy`.
        return (
            "degraded decision: produced by the serving fail-safe ladder, which sits above "
            "the lineage layer and has no importable definition here"
        )
    if record.policy_version not in POLICY_VARIANTS:
        return (
            f"unknown policy_version {record.policy_version!r}; this build cannot say what "
            f"that policy did (known: {sorted(POLICY_VARIANTS)})"
        )
    if "amount" not in payload:
        return "supplied payload has no `amount`; it is not a /v1/decide request body"
    return None
