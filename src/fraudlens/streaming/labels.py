"""Delayed-label revealer — makes label latency a measured property.

The dataset is a static 182-day snapshot in which every label is present from
the first row. That is not what operating a fraud model feels like: a
chargeback arrives 30-90 days after the transaction, so on any given day the
recent past looks fraud-free. The findings measured the consequence — 38.7% of
training-window fraud had not yet been disputed, and the observed fraud rate
collapses to 6.9% of the true rate in the final 20-day block. A model trained
on that learns "recent = safe".

This module reproduces that shape by withholding each label until its simulated
dispute lands. The invariant it exists to hold is negative: **no label is
readable before its reveal time.** That is enforced structurally — a row only
enters ``revealed_labels`` at the moment it is revealed, so there is no pending
state in the database for a careless join to read early — and again on the read
path, so a backfill cannot reopen the hole.

---

**Label provenance (E12c).** Not every label is a chargeback. A human
adjudication is also a label — an analyst looked at a transaction and called
it — and it arrives ~90 days earlier than the chargeback it may or may not
prevent. The two are different random variables: a chargeback is an *outcome*
(bank reversed the charge), a human call is an *opinion* correct at the
measured ``q_analyst`` = 0.91 and drawn from a censored, non-random sample.

They cannot share ``revealed_labels`` (which keys on ``transaction_id`` alone,
so only one label per transaction can exist) and they should not: mixing them
silently corrupts both training and the promotion gate. Human adjudications
live in ``human_adjudications``, and the default training/promotion path reads
``revealed_labels`` and cannot see them. Including human labels is an explicit
opt-in through :func:`effective_labels`, which tags every label with its
origin so the caller can filter.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import NormalDist

from sqlalchemy import Engine, select

from fraudlens.config.settings import SETTINGS
from fraudlens.streaming.schema import (
    human_adjudications,
    insert_ignoring_duplicates,
    revealed_labels,
)

_NORMAL = NormalDist()


def _utc_now() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)


@dataclass(frozen=True)
class DisputeLagModel:
    """How long after a transaction its true label becomes knowable.

    Lognormal for fraud, matching the research simulation: chargebacks cluster
    around a month with a long tail of late disputes, and a constant delay
    would understate exactly the tail that hurts retraining.

    The draw is a hash of the transaction id rather than a sequential RNG. That
    makes the lag a pure function of the transaction: it survives a producer
    restart, does not depend on the order events happened to be processed in,
    and is reproducible from the seed alone (spec §9a). The cost is that the
    numbers are not bit-identical to the seeded numpy stream in research/06,
    only distributionally identical.
    """

    median_days: float = SETTINGS.chargeback_median_days
    log_sigma: float = SETTINGS.chargeback_log_sigma
    clean_window_days: float = SETTINGS.clean_window_days
    seed: int = SETTINGS.label_sim_seed

    def lag_days(self, transaction_id: int, *, is_fraud: bool) -> float:
        if not is_fraud:
            return self.clean_window_days
        digest = hashlib.blake2b(f"{self.seed}:{transaction_id}".encode(), digest_size=8).digest()
        # +0.5 keeps the quantile strictly inside (0, 1); inv_cdf(0) is -inf.
        uniform = (int.from_bytes(digest, "big") + 0.5) / 2.0**64
        return math.exp(math.log(self.median_days) + self.log_sigma * _NORMAL.inv_cdf(uniform))


@dataclass(frozen=True)
class PendingLabel:
    """A label that exists but is not yet knowable."""

    transaction_id: int
    transaction_at: dt.datetime
    is_fraud: bool
    revealed_at: dt.datetime
    dispute_lag_days: float


@dataclass(frozen=True)
class RevealedLabel:
    """A label that has landed and may be used."""

    transaction_id: int
    is_fraud: bool
    transaction_at: dt.datetime
    revealed_at: dt.datetime
    recorded_at: dt.datetime
    dispute_lag_days: float
    label_source: str


class LabelRevealer:
    """Schedules labels, writes them when due, and refuses to read them early."""

    def __init__(
        self,
        engine: Engine,
        *,
        lag_model: DisputeLagModel | None = None,
        now: Callable[[], dt.datetime] = _utc_now,
        label_source: str = "chargeback_simulation",
    ) -> None:
        self._engine = engine
        self._lag = lag_model or DisputeLagModel()
        # Injected so the 182-day maturation can be tested in milliseconds
        # instead of waited out.
        self._now = now
        self._label_source = label_source
        self._stmt = insert_ignoring_duplicates(engine, revealed_labels)

    def schedule(
        self, transaction_id: int, transaction_at: dt.datetime, *, is_fraud: bool
    ) -> PendingLabel:
        if transaction_at.tzinfo is None:
            raise ValueError("transaction_at must be timezone-aware")
        lag = self._lag.lag_days(transaction_id, is_fraud=is_fraud)
        return PendingLabel(
            transaction_id=transaction_id,
            transaction_at=transaction_at,
            is_fraud=is_fraud,
            revealed_at=transaction_at + dt.timedelta(days=lag),
            dispute_lag_days=lag,
        )

    def reveal_due(self, pending: Iterable[PendingLabel], as_of: dt.datetime) -> list[PendingLabel]:
        """Write every label due at ``as_of``; return those still pending.

        Returning the remainder rather than a count keeps the due/not-due test
        in one place. A caller that reimplemented the comparison to prune its
        own queue would be the obvious way to leak a label early.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        due: list[PendingLabel] = []
        still_pending: list[PendingLabel] = []
        for label in pending:
            (due if label.revealed_at <= as_of else still_pending).append(label)
        if due:
            recorded_at = self._now()
            rows = [
                {
                    "transaction_id": label.transaction_id,
                    "is_fraud": label.is_fraud,
                    "transaction_at": label.transaction_at,
                    "revealed_at": label.revealed_at,
                    "recorded_at": recorded_at,
                    "dispute_lag_days": label.dispute_lag_days,
                    "label_source": self._label_source,
                }
                for label in due
            ]
            with self._engine.begin() as conn:
                conn.execute(self._stmt, rows)
        return still_pending

    def matured(self, as_of: dt.datetime) -> list[RevealedLabel]:
        """Labels knowable at ``as_of``.

        The ``revealed_at <= as_of`` filter is redundant with the write path and
        kept anyway. Every label-dependent metric in Tier 2 and the promotion
        gate reads through here, and a leak would silently inflate model
        performance for a quarter before anyone could notice.
        """
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        stmt = (
            select(revealed_labels)
            .where(revealed_labels.c.revealed_at <= as_of)
            .order_by(revealed_labels.c.revealed_at)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [RevealedLabel(**dict(row)) for row in rows]


# ---------------------------------------------------------------------------
# Label provenance — human adjudications and reconciliation (E12c)
# ---------------------------------------------------------------------------


class LabelOrigin(str, Enum):
    """What produced this label. Determines where it may be consumed.

    ``CHARGEBACK`` labels are outcomes and may be used anywhere — training,
    promotion, monitoring. ``HUMAN`` labels are opinions: useful as a
    control-chart detector (E12b) but excluded from training and promotion
    because the queue's selection bias (censored above the decision boundary)
    makes them a biased training sample.
    """

    CHARGEBACK = "chargeback"
    HUMAN = "human"


@dataclass(frozen=True)
class HumanAdjudication:
    """One analyst's call on one transaction. An opinion, not an outcome.

    No ``confidence`` field. Analysts do not self-assess accuracy; the system-
    wide ``q_analyst`` governs how the label is used, and a per-call confidence
    would be uncalibrated and load-bearing — exactly the kind of number that
    gets trusted because it exists.
    """

    transaction_id: int
    is_fraud: bool
    adjudicated_at: dt.datetime
    adjudicator: str


@dataclass(frozen=True)
class EffectiveLabel:
    """The label a consumer should use, after provenance is resolved.

    ``origin`` tells you whether this is ground truth (chargeback) or an
    opinion (human). ``disagrees_with_human`` is set when the chargeback
    contradicted a human call on the same transaction — that disagreement is
    the control-chart signal E12b priced, and it is surfaced rather than
    buried because an analyst disagreeing with reality is exactly the event a
    monitoring team needs to see.
    """

    transaction_id: int
    is_fraud: bool
    origin: LabelOrigin
    disagrees_with_human: bool = False


def record_adjudication(engine: Engine, adjudication: HumanAdjudication) -> bool:
    """Write one human adjudication. Returns True if it was new.

    Idempotent on (transaction_id, adjudicator): the same person re-submitting
    the same transaction is a no-op, not a correction. A second opinion is a
    different adjudicator and therefore a new row.
    """
    if adjudication.adjudicated_at.tzinfo is None:
        raise ValueError("adjudicated_at must be timezone-aware")
    # RETURNING, same pattern as the ledger: ON CONFLICT DO NOTHING gives no
    # row-count signal on either engine, so the insert's returned rows are the
    # idempotency signal.
    stmt = insert_ignoring_duplicates(engine, human_adjudications).returning(
        human_adjudications.c.transaction_id
    )
    with engine.begin() as conn:
        returned = conn.execute(
            stmt,
            {
                "transaction_id": adjudication.transaction_id,
                "is_fraud": adjudication.is_fraud,
                "adjudicated_at": adjudication.adjudicated_at,
                "adjudicator": adjudication.adjudicator,
            },
        ).fetchall()
    return len(returned) == 1


def read_adjudications(engine: Engine, as_of: dt.datetime) -> list[HumanAdjudication]:
    """Human adjudications recorded at or before ``as_of``.

    No maturity gate here: a human label is available the moment it is written.
    The gate matters for chargebacks because the dispute lag makes recent rows
    misleading; a human call has no lag.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    stmt = (
        select(human_adjudications)
        .where(human_adjudications.c.adjudicated_at <= as_of)
        .order_by(human_adjudications.c.adjudicated_at)
    )
    with engine.connect() as conn:
        rows = conn.execute(stmt).mappings().all()
    return [HumanAdjudication(**dict(row)) for row in rows]


def reconcile(
    human: HumanAdjudication | None,
    chargeback: RevealedLabel | None,
) -> EffectiveLabel | None:
    """Resolve which label wins for one transaction.

    **Chargeback wins when both exist.** The chargeback is an outcome; the
    human label was an early guess. When they disagree, the disagreement is
    recorded — it is the control-chart signal, not something to hide.

    **When only the human label exists**, it is returned with ``origin=HUMAN``.
    Callers that feed training or promotion must filter these out; the type
    exists so the filter is on a tag, not on a convention. The separate table
    makes the default path (reading ``revealed_labels``) structurally unable
    to see them at all.

    **The prevented-loss objection.** A human "fraud" call followed by no
    chargeback is ambiguous: the analyst may have been right and the decline
    prevented the loss, or wrong and the customer never disputed. Treating
    that case as "not fraud" (because no chargeback) is conservative — it
    biases toward the outcome rather than toward the opinion. The alternative
    (treating every human "fraud" call without a chargeback as a prevented
    loss) would inflate the fraud rate by the analyst false-positive rate,
    which is the larger error. The ambiguity is real and is the reason human
    labels are kept out of training: the training set cannot resolve it, and
    pretending it does not exist is worse than admitting it.
    """
    if chargeback is not None:
        return EffectiveLabel(
            transaction_id=chargeback.transaction_id,
            is_fraud=chargeback.is_fraud,
            origin=LabelOrigin.CHARGEBACK,
            disagrees_with_human=(human is not None and human.is_fraud != chargeback.is_fraud),
        )
    if human is not None:
        return EffectiveLabel(
            transaction_id=human.transaction_id,
            is_fraud=human.is_fraud,
            origin=LabelOrigin.HUMAN,
        )
    return None


def effective_labels(
    chargebacks: Sequence[RevealedLabel],
    *,
    humans: Sequence[HumanAdjudication] | None = None,
    include_human: bool = False,
) -> list[EffectiveLabel]:
    """Labels for a set of transactions, provenance-tagged.

    The training and promotion paths call this with ``include_human=False``
    (the default), which returns chargeback labels only — human labels are
    structurally invisible. The monitoring path (the control-chart detector
    from E12b) calls with ``include_human=True`` and gets both, with each
    label tagged so the consumer can still tell them apart.

    The ``include_human`` flag is a keyword-only bool rather than a subtler
    enum because the choice is binary and the default must be the safe one.
    A caller who passes ``humans=...`` without ``include_human=True`` gets
    chargeback-only — the human labels are available for disagreement
    detection but do not enter the label set themselves.
    """
    chargeback_by_id = {c.transaction_id: c for c in chargebacks}
    if humans:
        # Latest adjudication per transaction wins when there are multiple
        # (second opinions). Stable sort means the last one in time is kept.
        human_by_id: dict[int, HumanAdjudication] = {}
        for h in sorted(humans, key=lambda a: a.adjudicated_at):
            human_by_id[h.transaction_id] = h
    else:
        human_by_id = {}

    result: list[EffectiveLabel] = []
    # Chargeback labels first — these are always included.
    for tid, cb in chargeback_by_id.items():
        effective = reconcile(human_by_id.get(tid), cb)
        if effective is not None:
            result.append(effective)
    # Human-only labels — included only on explicit opt-in.
    if include_human:
        for tid, human in human_by_id.items():
            if tid not in chargeback_by_id:
                effective = reconcile(human, None)
                if effective is not None:
                    result.append(effective)
    return result
