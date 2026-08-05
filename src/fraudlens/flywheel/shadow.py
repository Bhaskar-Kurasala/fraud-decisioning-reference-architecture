"""Shadow mode: the challenger scores live traffic and decides none of it.

The requirement is easy to state and easy to get wrong: a shadow score must be impossible
to mistake for a decision. Not "clearly labelled" — impossible. Labelling is a convention,
and conventions survive until the first analyst who writes their own SQL, which is the
population an audit trail exists to serve (ADR-0003 rejects exactly this class of fix).

So the separation here is structural, at three levels:

1. **A shadow score has no action.** `ShadowScore` has no `action` field and
   `shadow_scores` has no `action` column. There is no value to misread, no default to
   inherit, and no query that can compute an approve rate from this table. A challenger
   observation is a probability and nothing else; turning it into an action is the
   policy layer's job and the policy layer is never called on this path.
2. **A shadow score is not in the ledger.** Different table, so `SELECT * FROM
   decision_ledger` cannot return one however it is filtered. The alternative — a
   `shadow BOOLEAN` column on `decision_ledger` — was rejected: every existing consumer
   of that table (drift, calibration, P&L, the maturity gate) would silently start
   double-counting the day the column was added, and each would need the same `WHERE
   shadow = false` remembered independently.
3. **A shadow score that failed to compute is absent, not zero.** A challenger that
   errors on the rows it finds hard is the most flattering possible bug: it scores the
   easy traffic, looks excellent, and promotes. `score`/`calibrated_probability` are
   nullable and paired with `failure_reason` under a CHECK, exactly as the ledger pairs
   its null score with `degraded`.

Not append-only-triggered, unlike the ledger: a shadow score is an observation about a
decision, not the record of one, and nothing adverse to a customer rests on it. The
composite key (transaction_id, challenger_version) is what makes re-scoring idempotent
and lets two challengers shadow the same traffic at once.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Engine,
    Float,
    Index,
    MetaData,
    Table,
    Text,
    select,
)

from fraudlens.streaming.schema import UtcDateTime

# Its own MetaData rather than `streaming.schema.metadata`. The streaming migrations are
# the source of truth for the audit trail and are applied by a documented command; the
# flywheel's own table has no business being created as a side effect of migrating the
# ledger, and keeping the two apart means a shadow-table change can never alter the
# schema that decisions are written to.
shadow_metadata = MetaData()

shadow_scores = Table(
    "shadow_scores",
    shadow_metadata,
    Column("transaction_id", BigInteger, primary_key=True, autoincrement=False),
    Column("challenger_version", Text, primary_key=True),
    # The model that actually decided this transaction, recorded alongside. Without it a
    # paired comparison months later has to infer the champion from a registry history
    # that may have moved on twice since.
    Column("champion_version", Text, nullable=False),
    Column("scored_at", UtcDateTime, nullable=False),
    Column("score", Float, nullable=True),
    Column("calibrated_probability", Float, nullable=True),
    Column("failure_reason", Text, nullable=True),
    # NOTE: there is deliberately no `action` column here. See the module docstring.
    CheckConstraint(
        "calibrated_probability IS NULL"
        " OR (calibrated_probability >= 0 AND calibrated_probability <= 1)",
        name="ck_shadow_probability_range",
    ),
    CheckConstraint(
        "(failure_reason IS NULL AND score IS NOT NULL AND calibrated_probability IS NOT NULL)"
        " OR (failure_reason IS NOT NULL AND score IS NULL AND calibrated_probability IS NULL)",
        name="ck_shadow_score_presence",
    ),
    Index("ix_shadow_scores_challenger", "challenger_version"),
)


@dataclass(frozen=True)
class ShadowScore:
    """What a challenger thought about a transaction it did not decide.

    There is no `action` here and there will not be one. If a caller needs the action a
    challenger *would* have taken, it computes it from `calibrated_probability` through
    the policy layer at analysis time — which keeps the counterfactual visibly
    counterfactual instead of storing it next to the real one.
    """

    transaction_id: int
    challenger_version: str
    champion_version: str
    scored_at: dt.datetime
    score: float | None
    calibrated_probability: float | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.scored_at.tzinfo is None:
            raise ValueError("scored_at must be timezone-aware")
        if self.challenger_version == self.champion_version:
            # A challenger shadowing itself produces a zero cost delta that passes every
            # gate check except the cost one, and reads as "no regression" rather than as
            # the misconfiguration it is.
            raise ValueError("a model cannot shadow itself; challenger and champion are the same")
        probability = self.calibrated_probability
        if probability is not None and not 0.0 <= probability <= 1.0:
            raise ValueError(f"calibrated_probability {probability} outside [0, 1]")
        scored = probability is not None and self.score is not None
        unscored = probability is None and self.score is None
        if self.failure_reason is not None and not unscored:
            raise ValueError("a failed shadow score has no value; leave score and probability None")
        if self.failure_reason is None and not scored:
            raise ValueError(
                "a shadow score with no value must say why it has none; set failure_reason"
            )

    def as_row(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "challenger_version": self.challenger_version,
            "champion_version": self.champion_version,
            "scored_at": self.scored_at,
            "score": self.score,
            "calibrated_probability": self.calibrated_probability,
            "failure_reason": self.failure_reason,
        }


def create_shadow_table(engine: Engine) -> None:
    """Create `shadow_scores` if absent. Separate from `streaming.migrate` by design."""
    shadow_metadata.create_all(engine)


class ShadowScoreLog:
    """Writer and reader for challenger observations.

    Named a log rather than a ledger: the ledger is the append-only audit record of
    adverse actions taken against customers, and giving this the same noun would invite
    the confusion the whole module exists to prevent.
    """

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record_many(self, scores: Sequence[ShadowScore]) -> int:
        if not scores:
            return 0
        with self._engine.begin() as conn:
            conn.execute(shadow_scores.insert(), [s.as_row() for s in scores])
        return len(scores)

    def scores_for(self, challenger_version: str) -> list[ShadowScore]:
        """Every observation from one challenger, failures included.

        Failures are returned rather than filtered. A caller that wants only the usable
        rows must drop them explicitly, which is the moment it notices that 12% of the
        window is missing — the number that decides whether the comparison is worth
        running at all.
        """
        stmt = (
            select(shadow_scores)
            .where(shadow_scores.c.challenger_version == challenger_version)
            .order_by(shadow_scores.c.transaction_id)
        )
        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [ShadowScore(**dict(row)) for row in rows]

    def coverage(self, challenger_version: str) -> tuple[int, int]:
        """(rows scored successfully, rows attempted) for one challenger.

        The gate's cost comparison is only as trustworthy as this ratio, and a challenger
        that silently failed on the hard tail of the distribution would otherwise present
        a clean comparison over a population that excludes precisely the transactions
        fraud lives in.
        """
        observations = self.scores_for(challenger_version)
        return sum(1 for s in observations if s.failure_reason is None), len(observations)
