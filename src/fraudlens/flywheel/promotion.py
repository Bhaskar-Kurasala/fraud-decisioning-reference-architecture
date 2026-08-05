"""Wiring the promotion gate to ledger data, and refusing before it can lie.

The gate itself lives in `models.gate` and is not reimplemented here. What this module
does is the part the gate deliberately does not: turn a window of real decisions, real
revealed labels and real challenger observations into the arrays the gate consumes, and
refuse to hand it anything it would evaluate honestly and report misleadingly.

Two things happen here that could not happen inside `models`:

**`require_mature` is called before any cost is computed.** `monitoring` sits above
`models` in the layer order, so the gate can only be *told* a maturity ratio — it cannot
go and check. This module can, and does, and the check comes first. When it fails, no
cost array is built at all. That is stronger than computing the delta and marking it
untrustworthy: a number that exists gets read, gets pasted into a review deck, and loses
its caveat somewhere in between. Promotion on an immature window is the single most
expensive mistake available in this system, because the bias runs one way — unrevealed
rows read as clean, so the more permissive challenger always looks cheaper.

**The comparison runs on revealed rows only.** The alternative, scoring the whole window
with unrevealed rows treated as non-fraud, is findings §6's error: the observed fraud rate
falls to 6.9% of true. So the cost arrays cover the matured subset, and the maturity ratio
is what bounds how much of the window that subset misses. At the 0.80 floor, up to a fifth
of the window is excluded, and the residual bias from that fifth is the risk the threshold
was chosen to accept — not a risk that has been eliminated.

The comparison is paired on the same transactions and read through an anytime-valid
confidence sequence (`models.sequential`), because labels trickle in continuously and this
window will be re-evaluated every time more of them land. A fixed-horizon interval re-read
on every arrival has no type-I error guarantee at all.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import numpy.typing as npt

from fraudlens.economics import (
    false_negative_cost,
    false_positive_cost,
    realised_cost,
    tenure_bucket,
)
from fraudlens.models.checks import GateInputs, GateThresholds
from fraudlens.models.gate import GateDecision, evaluate_promotion
from fraudlens.models.metrics import ModelMetrics, evaluate_scores
from fraudlens.models.registry import PromotionRecord, Stage, apply_gate_decision, transition_stage
from fraudlens.monitoring.maturity import ImmatureWindowError, LabelMaturity, label_maturity
from fraudlens.monitoring.maturity import require_mature as require_mature_window
from fraudlens.policy import decide

FloatArray = npt.NDArray[np.float64]

# The bands findings §4 publishes the break-even distribution over (0.731 at <$25 down to
# 0.369 at $500+). Reused verbatim so a segment regression reported by the gate can be read
# against a number that has already been published, rather than against a band nobody has
# seen before.
AMOUNT_BAND_EDGES: tuple[float, ...] = (25.0, 50.0, 100.0, 250.0, 500.0)


@dataclass(frozen=True)
class PromotionWindow:
    """One window of paired champion/challenger observations over the same transactions.

    `is_fraud[i]` is None exactly when the label has not been revealed. Carrying the
    unknown as None rather than False is the whole point: False here is the $4.36M
    mistake, because it makes every allowed transaction look free.
    """

    transaction_times: Sequence[dt.datetime]
    revealed_times: Sequence[dt.datetime | None]
    is_fraud: Sequence[bool | None]
    amount: FloatArray
    tenure_days: FloatArray
    champion_probability: FloatArray
    challenger_probability: FloatArray

    def __post_init__(self) -> None:
        lengths = {
            len(self.transaction_times),
            len(self.revealed_times),
            len(self.is_fraud),
            len(self.amount),
            len(self.tenure_days),
            len(self.champion_probability),
            len(self.challenger_probability),
        }
        if len(lengths) != 1:
            raise ValueError(f"window columns are not aligned; lengths {sorted(lengths)}")
        if not lengths or lengths == {0}:
            raise ValueError("cannot evaluate a promotion over an empty window")
        for revealed_at, label in zip(self.revealed_times, self.is_fraud, strict=True):
            # A label without a reveal time cannot be shown to have arrived legitimately,
            # and a reveal time without a label is a row the maturity count would treat as
            # matured while the cost computation drops it. Either way the two statistics
            # would disagree about the same row.
            if (revealed_at is None) != (label is None):
                raise ValueError("is_fraud and revealed_times must be present or absent together")

    @property
    def revealed(self) -> npt.NDArray[np.bool_]:
        return np.asarray([label is not None for label in self.is_fraud], dtype=bool)


@dataclass(frozen=True, slots=True)
class PromotionOutcome:
    """Either a gate verdict or a refusal to reach one. Never both, never neither.

    `decision is None` means the window was never evaluated, which is a different claim
    from "the gate blocked it" and is recorded as such. Collapsing the two would make the
    promotion-gate-failure alert (§5.2) fire identically for "the challenger is worse" and
    "we cannot yet tell", and those need different people.
    """

    maturity: LabelMaturity
    decision: GateDecision | None
    refusal: str | None
    # Both None on a refusal, and None rather than a zeroed `ModelMetrics`: a refused
    # window has no measured AUC, and a Tier 2 panel reading this must render a gap.
    champion: ModelMetrics | None = None
    challenger: ModelMetrics | None = None

    @property
    def promoted(self) -> bool:
        return self.decision is not None and self.decision.promote

    def summary(self) -> str:
        stamp = self.maturity.stamp()
        if self.decision is None:
            return f"REFUSED\n  {self.refusal}\n  {stamp}"
        return f"{self.decision.summary()}\n  {stamp}"


def _amount_band(amount: FloatArray) -> list[str]:
    index = np.searchsorted(np.asarray(AMOUNT_BAND_EDGES), amount, side="right")
    labels = [
        f"<${AMOUNT_BAND_EDGES[0]:.0f}",
        *(f"${lo:.0f}-{hi:.0f}" for lo, hi in pairwise(AMOUNT_BAND_EDGES)),
        f"${AMOUNT_BAND_EDGES[-1]:.0f}+",
    ]
    return [labels[i] for i in index]


def _costs(
    probability: FloatArray,
    is_fraud: npt.NDArray[np.int_],
    fn_cost: FloatArray,
    fp_cost: FloatArray,
    *,
    include_review: bool,
) -> FloatArray:
    """Per-transaction realised cost of routing one model's probabilities through policy.

    Both models are run through the *same* policy and the same cost functions, so the
    delta isolates the score. A challenger evaluated under a policy retuned to suit it is
    a different experiment, and one whose result cannot be attributed.
    """
    actions = decide(probability, fn_cost, fp_cost, include_review=include_review)
    return realised_cost(actions, is_fraud, fn_cost, fp_cost)


def evaluate_challenger(
    window: PromotionWindow,
    *,
    as_of: dt.datetime,
    include_review: bool,
    thresholds: GateThresholds | None = None,
) -> PromotionOutcome:
    """Measure maturity, refuse or proceed, and return the gate's full verdict.

    `include_review` has no default, for the reason `policy.decide` gives: the two policy
    variants have different published costs and a silent default is how they got conflated
    the first time. A promotion decided under one and shipped under the other is worse.
    """
    thresholds = thresholds or GateThresholds()
    maturity = label_maturity(
        window.transaction_times,
        window.revealed_times,
        as_of=as_of,
        window_start=min(window.transaction_times),
        window_end=max(window.transaction_times),
    )
    try:
        # The gate's floor, passed explicitly. `maturity.DEFAULT_MATURITY_FLOOR` is a
        # reporting decision and is free to diverge from the promotion decision; only one
        # of the two may decide whether money moves.
        require_mature_window(maturity, thresholds.min_label_maturity)
    except ImmatureWindowError as refused:
        return PromotionOutcome(maturity=maturity, decision=None, refusal=str(refused))

    revealed = window.revealed
    labels = np.asarray([bool(f) for f in window.is_fraud if f is not None], dtype=np.int_)
    amount = window.amount[revealed]
    tenure = tenure_bucket(window.tenure_days[revealed])
    fn_cost = false_negative_cost(amount)
    fp_cost = false_positive_cost(amount, tenure)

    champion_cost = _costs(
        window.champion_probability[revealed],
        labels,
        fn_cost,
        fp_cost,
        include_review=include_review,
    )
    challenger_cost = _costs(
        window.challenger_probability[revealed],
        labels,
        fn_cost,
        fp_cost,
        include_review=include_review,
    )

    champion = evaluate_scores(labels, window.champion_probability[revealed], champion_cost)
    challenger = evaluate_scores(labels, window.challenger_probability[revealed], challenger_cost)
    inputs = GateInputs(
        champion=champion,
        challenger=challenger,
        champion_cost=champion_cost,
        challenger_cost=challenger_cost,
        label_maturity=maturity.ratio,
        segments={
            "amount_band": _amount_band(amount),
            "tenure": [str(t) for t in tenure],
        },
    )
    return PromotionOutcome(
        maturity=maturity,
        decision=evaluate_promotion(inputs, thresholds),
        refusal=None,
        champion=champion,
        challenger=challenger,
    )


def record_outcome(
    client: Any,
    name: str,
    version: int,
    outcome: PromotionOutcome,
    actor: str,
    now: dt.datetime,
) -> PromotionRecord:
    """Write the verdict to the registry, promotion or not.

    A refusal is written too, and it is written as a *refusal* rather than as a blocked
    gate decision. The registry is the record a model-risk review reads, and "we could not
    yet tell" is the answer that most often needs defending — it is the state a challenger
    sits in for the ninety days between training and knowing.
    """
    if outcome.decision is not None:
        return apply_gate_decision(client, name, version, outcome.decision, actor, now)

    justification = outcome.summary()
    transition_stage(
        client,
        name,
        version,
        Stage.STAGING,
        justification=justification,
        actor=actor,
        now=now,
        # Emphatically not archiving the incumbent. Nothing has been demonstrated about
        # the challenger; the champion keeps deciding.
        archive_existing=False,
    )
    return PromotionRecord(
        model_name=name,
        version=version,
        promoted=False,
        to_stage=Stage.STAGING,
        actor=actor,
        decided_at=now,
        justification=justification,
    )
