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
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import NormalDist

from sqlalchemy import Engine, select

from fraudlens.streaming.schema import insert_ignoring_duplicates, revealed_labels

# Mirrors CHARGEBACK_MEDIAN_DAYS / CHARGEBACK_LOG_SIGMA / SEED_LABEL_SIM in the
# root config.py, which is the provenance of the published 38.7% figure. They
# are restated rather than imported because config.py is the research entry
# point, not a library; they move to fraudlens.config when E2 lands it.
CHARGEBACK_MEDIAN_DAYS = 34.0
CHARGEBACK_LOG_SIGMA = 0.85
DEFAULT_LAG_SEED = 7

# BUSINESS ASSUMPTION. A legitimate transaction is never confirmed by a
# positive event; you learn it was good only when the dispute window closes
# without a chargeback. Card network windows run to 120 days; 90 is the
# operating convention. This number decides how long a transaction is
# unusable for training, so it belongs in a review cycle, not in a constant.
DEFAULT_CLEAN_WINDOW_DAYS = 90.0

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

    median_days: float = CHARGEBACK_MEDIAN_DAYS
    log_sigma: float = CHARGEBACK_LOG_SIGMA
    clean_window_days: float = DEFAULT_CLEAN_WINDOW_DAYS
    seed: int = DEFAULT_LAG_SEED

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
