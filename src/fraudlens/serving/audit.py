"""The adapter from a served decision to the append-only ledger.

`runtime.DecisionWriter` says of itself that the adapter turning its keyword arguments
into a ledger row "belongs to whatever composes the two". This is that composition. It is
a field-for-field splat by design — the port's argument names match the ledger's columns
one to one specifically so this never becomes a translation, because a translation layer
between a decision and its audit record is a place where the two can disagree.

The failure policy is the reason this is a module rather than four lines inline: a failed
ledger write is an *auditability* incident, not a request error, and those are handled in
opposite directions. The customer already has a correct decision and must not be made to
wait or retry for our record-keeping; but a gap in an append-only ledger is exactly the
row a dispute asks about ninety days later, so it has to be countable rather than merely
logged. ADR-0002 ranks auditability second of five, above latency, which is what makes
"succeed the request, count the gap" the right resolution rather than a convenient one.
"""

from __future__ import annotations

import datetime as dt
import logging

from fraudlens.serving.contracts import DecideRequest, VersionSet
from fraudlens.serving.decisioning import Outcome, input_hash
from fraudlens.serving.metrics import LEDGER_WRITE_FAILURES
from fraudlens.serving.runtime import DecisionWriter

logger = logging.getLogger(__name__)


def persist(
    writer: DecisionWriter | None,
    request: DecideRequest,
    outcome: Outcome,
    decided_at: dt.datetime,
    versions: VersionSet,
) -> None:
    """Append the decision to the audit trail, if one is wired.

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
        # Counted as well as logged. "How many of your decisions can you not reconstruct?"
        # has to be answerable from a number rather than by grepping a quarter of logs.
        LEDGER_WRITE_FAILURES.inc()
        logger.exception("ledger write failed for transaction %s", request.transaction_id)
