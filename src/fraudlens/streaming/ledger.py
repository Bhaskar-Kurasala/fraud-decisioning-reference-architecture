"""Append-only decision ledger — the audit trail (ADR-0002, priority #2).

The ledger accepts an already-computed decision. It does not import the policy
or model layers and does not know how an action was chosen. That is deliberate:
the audit trail must be writable from the serving path, from a replay driver,
and from a backfill, and none of those should have to agree on how scoring is
wired. It also means this module can be tested without a model.

Mutability is prevented by the database (see ``migrations/003``), not here.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Engine, func, select

from fraudlens.streaming.schema import ACTIONS, decision_ledger, insert_ignoring_duplicates


@dataclass(frozen=True)
class DecisionRecord:
    """One fully reconstructable decision.

    Every version field is required rather than defaulted. A default would let
    an unversioned decision into the ledger, and an unversioned decision cannot
    be replayed — which is the property spec §9a asserts by test.
    """

    transaction_id: int
    transaction_at: dt.datetime
    decided_at: dt.datetime
    # None when the decision was made without a model (see `degraded`). Not 0.0:
    # a fabricated score is indistinguishable from a real near-zero one, and it
    # biases every distribution built from this column toward zero in proportion
    # to how long the model was down.
    score: float | None
    calibrated_probability: float | None
    action: str
    reason_codes: tuple[str, ...]
    model_version: str
    policy_version: str
    feature_version: str
    config_hash: str
    input_hash: str
    degraded: bool = False
    degraded_reason: str | None = None

    def __post_init__(self) -> None:
        # Fail loudly at the boundary. The DB CHECK constraints are the real
        # control; these raise a comprehensible error before a 400-row batch is
        # rejected wholesale by the server with a constraint name.
        if self.action not in ACTIONS:
            raise ValueError(f"unknown action {self.action!r}; expected one of {ACTIONS}")
        probability = self.calibrated_probability
        if probability is not None and not 0.0 <= probability <= 1.0:
            raise ValueError(f"calibrated_probability {probability} outside [0, 1]")
        # A scored decision must carry its score; a degraded one must not invent
        # one. Enforced in the DB too — this is the comprehensible error.
        scored = probability is not None and self.score is not None
        unscored = probability is None and self.score is None
        if self.degraded and not unscored:
            raise ValueError(
                "a degraded decision has no model score; leave score and probability None"
            )
        if not self.degraded and not scored:
            raise ValueError(
                "a non-degraded decision must carry both score and calibrated_probability"
            )
        for name in ("transaction_at", "decided_at"):
            stamp: dt.datetime = getattr(self, name)
            if stamp.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.degraded != (self.degraded_reason is not None):
            raise ValueError("degraded and degraded_reason must be set together or not at all")

    def as_row(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "transaction_at": self.transaction_at,
            "decided_at": self.decided_at,
            "score": self.score,
            "calibrated_probability": self.calibrated_probability,
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "model_version": self.model_version,
            "policy_version": self.policy_version,
            "feature_version": self.feature_version,
            "config_hash": self.config_hash,
            "input_hash": self.input_hash,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }


class DecisionLedger:
    """Idempotent, append-only writer and reader for ``decision_ledger``."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        # RETURNING, not rowcount. `ON CONFLICT DO NOTHING` gives psycopg3 a rowcount of
        # -1 for a fresh insert *and* for a conflict, so `bool(rowcount)` reports True in
        # both cases — the duplicate signal is not merely unreliable on Postgres, it is
        # inverted-broken, and it happens to work on SQLite, which is where the default
        # suite runs. Found by the E10 degradation drill, which was the first thing to
        # exercise this against a real Postgres.
        #
        # With RETURNING the answer comes from the database rather than from a driver
        # convention: a conflicting row is not inserted, so it returns nothing. One
        # behaviour, both engines, no dialect branch to keep in sync.
        self._stmt = insert_ignoring_duplicates(engine, decision_ledger).returning(
            decision_ledger.c.transaction_id
        )

    def record(self, decision: DecisionRecord) -> bool:
        """Write one decision. Returns True if it was new, False if a duplicate.

        The producer delivers at-least-once (a checkpoint is committed after the
        write, so a crash between the two replays the event). This return value
        is what makes that safe *and* observable: a silently-swallowed duplicate
        and a genuinely new decision look different to the caller.
        """
        with self._engine.begin() as conn:
            returned = conn.execute(self._stmt, decision.as_row()).fetchall()
        return len(returned) == 1

    def record_many(self, decisions: Sequence[DecisionRecord]) -> int:
        """Write a batch, skipping duplicates. Returns the number newly written.

        Counts the rows the insert actually returned. The previous implementation took
        the difference of two `count()` calls around the write, which needed the ledger
        to be otherwise idle to be correct — under the concurrent writers this is built
        for, a second producer's rows would be attributed to this batch.
        """
        if not decisions:
            return 0
        rows = [d.as_row() for d in decisions]
        with self._engine.begin() as conn:
            return len(conn.execute(self._stmt, rows).fetchall())

    def get(self, transaction_id: int) -> DecisionRecord | None:
        stmt = select(decision_ledger).where(decision_ledger.c.transaction_id == transaction_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).mappings().one_or_none()
        if row is None:
            return None
        data = dict(row)
        data["reason_codes"] = tuple(data["reason_codes"])
        return DecisionRecord(**data)

    def count(self) -> int:
        with self._engine.connect() as conn:
            return int(conn.execute(select(func.count()).select_from(decision_ledger)).scalar_one())
