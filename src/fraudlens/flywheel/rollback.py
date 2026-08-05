"""Rollback: what it can mean when the decisions have already been made.

A promoted model does not merely exist — it has declined people. Those declines are
adverse actions against identifiable customers, and the ledger that records them is
append-only by database trigger (`migrations/003`). So the word "rollback" has to be
demoted from what it means everywhere else:

- It is **not** an undo. Nothing in the ledger changes. The rows the bad model wrote stay
  exactly as written, still carrying its `model_version`, and a rollback that quietly
  rewrote them would destroy the only evidence that the incident happened.
- It is **not** a replay. The transactions are gone. The customer who abandoned after a
  challenge did not come back because a registry pin moved.

What it is, is two things:

1. **A pin revert.** The registry stops pointing new traffic at the bad version. That is
   the entire prospective effect, and it takes effect at the next model load — every
   decision already taken stands.
2. **An identified, priced population.** `model_version` is a required field on every
   ledger row precisely so this query is possible: these 40,000 decisions were taken under
   the model we rolled back, here is their action mix, and here is what they cost. That
   number is what a rollback is *for*. Reverting the pin without it leaves the
   organisation knowing it made a mistake and not knowing how large.

**On the counterfactual.** The obviously-desired number is "what would the champion have
decided instead" — but computing it requires re-scoring the affected transactions with the
restored model, and this repository has no champion estimator on disk (the research scripts
persisted scores, not fitted models). Rather than fabricate it, `restored_probability` is an
optional input: supply it and the counterfactual delta is computed, omit it and the field is
`None`, never `0.0` (ADR-0003). The intended supplier is `shadow` — a demoted champion that
keeps scoring in shadow mode has exactly these probabilities already, which is the practical
argument for leaving the previous champion in shadow after a promotion rather than turning
it off.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sqlalchemy import Engine, select

from fraudlens.economics import (
    ACTION_NAMES,
    annualise,
    false_negative_cost,
    false_positive_cost,
    realised_cost,
    tenure_bucket,
)
from fraudlens.models.registry import PromotionRecord, Stage, transition_stage
from fraudlens.policy import decide
from fraudlens.streaming.schema import decision_ledger

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AffectedDecision:
    """One decision taken under the rolled-back model, as the ledger recorded it."""

    transaction_id: int
    transaction_at: dt.datetime
    action: str
    calibrated_probability: float | None


@dataclass(frozen=True, slots=True)
class Exposure:
    """The size and price of the population a rolled-back model decided.

    `realised_cost_total` covers only the transactions whose labels have matured;
    `n_decisions` covers all of them. Reporting the cost of the matured subset against the
    count of the whole population would understate the unit cost, and understating it is
    the direction that makes the incident look smaller.
    """

    model_version: str
    window_start: dt.datetime
    window_end: dt.datetime
    n_decisions: int
    n_priced: int
    action_counts: dict[str, int]
    realised_cost_total: float
    annualised_cost: float
    # None when no restored-model probabilities were supplied. See module docstring: this
    # is the number people most want and the one there is least honest basis for.
    counterfactual_cost_total: float | None
    counterfactual_delta: float | None


def decisions_by_model(
    engine: Engine,
    model_version: str,
    *,
    window_start: dt.datetime | None = None,
    window_end: dt.datetime | None = None,
) -> list[AffectedDecision]:
    """Every ledger row decided by one model version, in time order.

    Queries the ledger table directly rather than through `DecisionLedger`, which reads by
    primary key. This is the population query, and it is the one that makes `model_version`
    a required non-defaulted field on every row worth having.
    """
    stmt = select(
        decision_ledger.c.transaction_id,
        decision_ledger.c.transaction_at,
        decision_ledger.c.action,
        decision_ledger.c.calibrated_probability,
    ).where(decision_ledger.c.model_version == model_version)
    if window_start is not None:
        stmt = stmt.where(decision_ledger.c.transaction_at >= window_start)
    if window_end is not None:
        stmt = stmt.where(decision_ledger.c.transaction_at <= window_end)
    with engine.connect() as conn:
        rows = conn.execute(stmt.order_by(decision_ledger.c.transaction_at)).mappings().all()
    return [AffectedDecision(**dict(row)) for row in rows]


def quantify(
    decisions: Sequence[AffectedDecision],
    *,
    model_version: str,
    is_fraud: Sequence[bool | None],
    amount: FloatArray,
    tenure_days: FloatArray,
    include_review: bool,
    restored_probability: FloatArray | None = None,
) -> Exposure:
    """Price the affected population, and the counterfactual only if it is available.

    Rows without a matured label are excluded from the cost, not counted as clean. At the
    point a rollback is being considered the model has usually been live for weeks, so most
    of the population is unmatured — treating those as non-fraud would price the incident at
    close to the friction cost alone, which is the term the bad model was *cheap* on.
    """
    if not decisions:
        raise ValueError("cannot quantify an empty population")
    if not (len(decisions) == len(is_fraud) == len(amount) == len(tenure_days)):
        raise ValueError("decisions, labels, amount and tenure must be aligned")

    action_counts: dict[str, int] = {}
    for record in decisions:
        action_counts[record.action] = action_counts.get(record.action, 0) + 1

    priced = np.asarray([label is not None for label in is_fraud], dtype=bool)
    labels = np.asarray([bool(f) for f in is_fraud if f is not None], dtype=np.int_)
    fn_cost = false_negative_cost(amount[priced])
    fp_cost = false_positive_cost(amount[priced], tenure_bucket(tenure_days[priced]))

    # The action taken is read from the ledger, not recomputed. What the incident cost is
    # what actually happened, and recomputing it from the stored probability would silently
    # substitute today's policy version for the one that was live at the time.
    actions_taken = np.asarray(
        [_action_index(d.action) for d, keep in zip(decisions, priced, strict=True) if keep],
        dtype=np.intp,
    )
    incurred = realised_cost(actions_taken, labels, fn_cost, fp_cost)
    total = float(incurred.sum())

    span_days = max((decisions[-1].transaction_at - decisions[0].transaction_at).days, 1)
    counterfactual: float | None = None
    delta: float | None = None
    if restored_probability is not None:
        if len(restored_probability) != len(decisions):
            raise ValueError("restored_probability must cover every affected decision")
        restored_actions = decide(
            restored_probability[priced], fn_cost, fp_cost, include_review=include_review
        )
        counterfactual = float(realised_cost(restored_actions, labels, fn_cost, fp_cost).sum())
        delta = total - counterfactual

    return Exposure(
        model_version=model_version,
        window_start=decisions[0].transaction_at,
        window_end=decisions[-1].transaction_at,
        n_decisions=len(decisions),
        n_priced=int(priced.sum()),
        action_counts=action_counts,
        realised_cost_total=total,
        annualised_cost=annualise(total, span_days),
        counterfactual_cost_total=counterfactual,
        counterfactual_delta=delta,
    )


# The ledger stores the action as a word; the EV matrix indexes it by position. Derived
# from `ACTION_NAMES` rather than written out, because a mapping written out by hand is a
# second definition of the action order and the failure mode of the two disagreeing is a
# population priced under the wrong arm with no error raised anywhere.
_ACTION_INDEX: dict[str, int] = {name: index for index, name in enumerate(ACTION_NAMES)}


def _action_index(action: str) -> int:
    try:
        return _ACTION_INDEX[action]
    except KeyError:
        raise ValueError(f"ledger holds an action this pricing cannot value: {action!r}") from None


def roll_back(
    client: Any,
    name: str,
    *,
    bad_version: int,
    restore_version: int,
    exposure: Exposure,
    reason: str,
    actor: str,
    now: dt.datetime,
) -> tuple[PromotionRecord, PromotionRecord]:
    """Repin production to `restore_version` and archive `bad_version`.

    The exposure is required, not optional, and it is written into both justifications.
    A rollback recorded as "reverted, model was bad" is the version of this event that
    stops being answerable six months later when someone asks how much it cost; the
    registry is where a model-risk review looks, so the number goes there.
    """
    if bad_version == restore_version:
        raise ValueError("rolling back to the version being rolled back is a no-op")
    priced = (
        f"counterfactual delta ${exposure.counterfactual_delta:,.0f}"
        if exposure.counterfactual_delta is not None
        # Stated as unavailable rather than omitted. An absent line reads as zero.
        else "counterfactual not computed (no restored-model scores for this population)"
    )
    justification = (
        f"{reason}\n"
        f"{exposure.n_decisions} decisions taken under version {bad_version} between "
        f"{exposure.window_start.isoformat()} and {exposure.window_end.isoformat()}; "
        f"{exposure.n_priced} matured and cost ${exposure.realised_cost_total:,.0f} "
        f"(${exposure.annualised_cost:,.0f}/yr annualised); {priced}. "
        "The ledger is unchanged: these decisions stand and remain attributable."
    )
    transition_stage(
        client,
        name,
        bad_version,
        Stage.ARCHIVED,
        justification=justification,
        actor=actor,
        now=now,
    )
    transition_stage(
        client,
        name,
        restore_version,
        Stage.PRODUCTION,
        justification=justification,
        actor=actor,
        now=now,
        # Already archived above, explicitly and with a reason. Letting the promotion
        # archive it as a side effect would leave the bad version tagged with the
        # restoration's justification instead of its own.
        archive_existing=False,
    )
    return (
        PromotionRecord(
            model_name=name,
            version=bad_version,
            promoted=False,
            to_stage=Stage.ARCHIVED,
            actor=actor,
            decided_at=now,
            justification=justification,
        ),
        PromotionRecord(
            model_name=name,
            version=restore_version,
            promoted=True,
            to_stage=Stage.PRODUCTION,
            actor=actor,
            decided_at=now,
            justification=justification,
        ),
    )
