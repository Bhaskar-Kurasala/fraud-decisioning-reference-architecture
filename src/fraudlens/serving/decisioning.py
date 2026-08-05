"""Features -> score -> calibrate -> policy, and the fail-safe path when that breaks.

Every number here comes from a lower layer. The costs are `economics.costs`, the action
is `policy.decide`, the boundaries quoted in the reason codes are `policy.boundaries`.
Nothing is re-derived from a description of itself: two corrections have already been
needed on this project because someone reimplemented the policy from its prose, and the
resulting decisions were wrong by $71,545/yr and by $583/yr respectively.

**The fallback ladder** (ADR-0002 priority #4: "fail to the safe state, and for fraud the
safe state is not approve"). When the model is unavailable or scoring raises:

    high amount AND new/unknown account  ->  deny
    everything else                      ->  challenge

and never `allow`, on any input, ever. Two decisions are load-bearing in that ladder.

First, it does **not** call `policy.decide`. The EV argmax needs a calibrated probability,
and in degraded mode there is not one; feeding it an assumed prior would launder a guess
through the cost model and produce a decision that looks economically derived and is not.
An honest rule ladder is auditable, a fabricated input is not.

Second, the default rung is `challenge`, not `deny`. A challenge is friction the customer
can clear themselves; a decline is an irreversible adverse action, and taking one against
every customer on no evidence is not defensible to a regulator however safe it is for us.
Denying is reserved for the one case where an undetected fraud is most expensive and we
have the least reason to believe the account: a large basket on an account we have never
seen. ADR-0002 records the cost of this arm as "false declines during degradation; must be
measured, not assumed", which is what `fraudlens_degraded_decisions_total` is for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from fraudlens.config import SETTINGS
from fraudlens.economics import (
    ACTION_NAMES,
    false_negative_cost,
    false_positive_cost,
    tenure_bucket,
)
from fraudlens.models.provenance import config_hash
from fraudlens.policy import FALLBACK_VERSION, boundaries, decide, decide_without_model
from fraudlens.serving.contracts import DecideRequest
from fraudlens.serving.latency import CALIBRATION_POLICY, FEATURES, INFERENCE, StageTimings
from fraudlens.serving.metrics import DEGRADATIONS
from fraudlens.serving.reasons import (
    ReasonCode,
    context_reasons,
    score_band_reason,
)
from fraudlens.serving.runtime import CalibratedScorer

logger = logging.getLogger(__name__)

# This deployment runs the THREE-action policy: allow / challenge / deny, review excluded.
#
# Stated explicitly because `decide` has no default for `include_review` and the two
# variants are different policies with different published costs ($2,799,214 without
# review, $2,799,797 with). The choice is not about which is cheaper on paper -- it is
# that the review arm is capacity-bound at 60 analyst slots/day and this path cannot check
# the queue inside a 150 ms synchronous budget. A four-action service would emit `review`
# for cases no analyst will ever see, which is an approve with extra steps. On the test
# window the review arm is chosen 5 times and costs $583/yr MORE than not having it, so
# excluding it here also happens to be the cheaper of the two. `policy.queue.rank_for_review`
# serves the offline path that can honour a capacity constraint.
INCLUDE_REVIEW: Final = False

# Names BOTH policies this deployment can decide under, because it can decide under either
# and the ledger has one column to say which. A scored decision came from the EV argmax; a
# degraded one came from the rule ladder; a replay months later has to know which
# arithmetic to re-run, and "ev-argmax-3action-v1" alone would silently claim the argmax
# priced a decision no model ever saw.
POLICY_VERSION: Final = f"ev-argmax-3action-v1+{FALLBACK_VERSION}"

# There is no feature store in this deployment; the caller assembles the vector and this
# string versions that contract. It is not a pipeline version and must not be read as one.
FEATURE_VERSION: Final = "request-supplied-v1"
SERVICE_VERSION: Final = "0.1.0"

# Hashed once at import: the constants are frozen for the process lifetime (config is read
# at import, 12-factor), so recomputing per request would spend budget to get the same
# digest. Reuses the models layer's canonical-JSON hash rather than defining a second one.
CONFIG_HASH: Final = config_hash(SETTINGS.model_dump())


def input_hash(request: DecideRequest) -> str:
    """Digest of the exact request this decision was made from.

    A dispute months later asks "what did you know when you declined me". The feature
    values themselves live with the caller; this pins which ones they were, so a replay
    that produces a different action can be distinguished from one fed different input.
    Reuses the models layer's canonical-JSON SHA-256 rather than adding a second hash --
    it is generic over mappings and the name is the only thing about it that says config.
    """
    return config_hash(request.model_dump(mode="json"))


@dataclass(frozen=True)
class Outcome:
    """One decision plus everything the ledger and the response need from it."""

    action: str
    # Both None on the degraded path. The rule ladder decides without a model,
    # so there is no score to report and none to persist.
    raw_score: float | None
    calibrated_probability: float | None
    reason_codes: tuple[str, ...]
    degraded: bool
    degraded_reason: str | None


def decide_transaction(
    request: DecideRequest,
    scorer: CalibratedScorer | None,
    timings: StageTimings,
) -> Outcome:
    """Decide one transaction, degrading to the documented rule ladder on any failure.

    The `except Exception` is deliberate and is the one place in this codebase where a
    bare-ish catch is correct: §9a says errors that lose money are handled explicitly, and
    every way the scoring path can fail has the same correct response -- do not approve.
    Narrowing it would mean a novel exception type reaches the framework's error handler
    and returns a 500, and a 500 in a checkout integration is frequently retried or, worse,
    fails open at the caller.
    """
    if scorer is None:
        return _fallback(request, ReasonCode.DEGRADED_MODEL_UNAVAILABLE)
    try:
        return _scored(request, scorer, timings)
    except Exception:
        logger.exception("scoring failed for transaction %s, falling back", request.transaction_id)
        return _fallback(request, ReasonCode.DEGRADED_SCORING_ERROR)


def _scored(
    request: DecideRequest,
    scorer: CalibratedScorer,
    timings: StageTimings,
) -> Outcome:
    """The model path. Stages match the §4.3 budget lines one for one."""
    with timings.stage(FEATURES):
        # NaN, not 0.0, for a missing signal: 0.0 is a real D1 (the card's first ever
        # transaction) and would price a customer we know nothing about as the riskiest
        # possible one. `tenure_bucket` maps NaN to the unknown bucket, which is priced
        # explicitly and surfaced as a reason code.
        d1 = np.nan if request.days_since_first_seen is None else request.days_since_first_seen
        tenure = tenure_bucket([d1])
        fn = false_negative_cost([request.amount])
        fp = false_positive_cost([request.amount], tenure)

    with timings.stage(INFERENCE):
        raw = float(scorer.score(request.features))

    with timings.stage(CALIBRATION_POLICY):
        probability = float(scorer.calibrate(raw))
        if not 0.0 <= probability <= 1.0:
            # A calibrator that returns something outside [0, 1] is broken, and the cost
            # model would happily price it. Treated as a scoring failure so the request
            # takes the fail-safe path rather than producing a plausible-looking decision.
            raise ValueError(f"calibrated probability {probability} outside [0, 1]")
        action_index = int(decide(probability, fn, fp, include_review=INCLUDE_REVIEW)[0])
        codes = _explain(probability, request, str(tenure[0]), fn, fp)

    return Outcome(
        action=ACTION_NAMES[action_index],
        raw_score=raw,
        calibrated_probability=probability,
        reason_codes=codes,
        degraded=False,
        degraded_reason=None,
    )


def _explain(
    probability: float,
    request: DecideRequest,
    tenure: str,
    fn: npt.NDArray[np.float64],
    fp: npt.NDArray[np.float64],
) -> tuple[str, ...]:
    """Reason codes for a scored decision. Always at least the score band."""
    bounds = boundaries(fn, fp)
    band = score_band_reason(
        probability,
        challenge_boundary=float(bounds["allow_to_challenge"][0]),
        deny_boundary=float(bounds["challenge_to_deny"][0]),
    )
    return tuple(code.value for code in [band, *context_reasons(request.amount, tenure)])


def _fallback(request: DecideRequest, cause: ReasonCode) -> Outcome:
    """Route to the rule ladder and record why we are on it.

    The ladder itself is `policy.decide_without_model`. It moved there so that
    `lineage.replay`, which sits below serving in the import contract, can reconstruct a
    degraded decision — an outage produces the largest block of unusual decisions in the
    system's history, and while the ladder lived here that block was the one block nobody
    could later prove was decided correctly.
    """
    DEGRADATIONS.labels(reason=cause.value).inc()
    action, rule = decide_without_model(request.amount, request.days_since_first_seen)
    return Outcome(
        action=action,
        # Null rather than a number, for both. There is no score; publishing a
        # placeholder is how a degraded decision gets silently absorbed into a
        # calibration report — and 0.0 is the worst placeholder available, since it
        # reads as "confidently legitimate".
        raw_score=None,
        calibrated_probability=None,
        reason_codes=(cause.value, rule),
        degraded=True,
        degraded_reason=cause.value,
    )
