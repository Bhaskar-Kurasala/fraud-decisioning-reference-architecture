"""Physical schema for the decision ledger and the delayed-label table.

The numbered SQL files under ``migrations/`` are the source of truth for
Postgres. These SQLAlchemy Core tables mirror them so the same write and read
paths can be exercised on SQLite in the default test suite, without a live
database. ``tests/streaming/test_schema_parity.py`` fails if the two drift,
which is what makes the duplication safe rather than a second source of truth.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    MetaData,
    Table,
    Text,
    TypeDecorator,
)
from sqlalchemy.engine import Engine
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.sql.dml import Insert
from sqlalchemy.types import JSON

# The four actions the policy layer can take. Duplicated as a DB CHECK rather
# than trusted from application code: ADR-0002 ranks auditability second, and an
# audit trail that can contain an action the policy never defined is not one.
ACTIONS: tuple[str, ...] = ("allow", "challenge", "review", "deny")


class UtcDateTime(TypeDecorator[dt.datetime]):
    """Timestamp column that rejects naive datetimes and always reads back UTC.

    A decline is an adverse action against an identifiable customer and has to
    be reconstructable years later. "2026-11-01 01:30 local" is two distinct
    instants on a DST fall-back night, so a naive timestamp makes the ordering
    of a disputed decision unprovable. SQLite has no timezone-aware type at all,
    so the guarantee has to live in the column type rather than in the database.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: dt.datetime | None, dialect: Dialect) -> Any:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("ledger timestamps must be timezone-aware; got a naive datetime")
        return value.astimezone(dt.timezone.utc)

    def process_result_value(self, value: Any, dialect: Dialect) -> dt.datetime | None:
        if value is None:
            return None
        stamped: dt.datetime = value
        if stamped.tzinfo is None:
            # SQLite discards the offset on write; everything is stored as UTC.
            return stamped.replace(tzinfo=dt.timezone.utc)
        return stamped.astimezone(dt.timezone.utc)


metadata = MetaData()

_ACTION_LIST = ", ".join(f"'{a}'" for a in ACTIONS)

decision_ledger = Table(
    "decision_ledger",
    metadata,
    # transaction_id is the primary key, not a surrogate. That single choice is
    # what makes writes idempotent: a replayed transaction collides instead of
    # duplicating. Consequence, stated deliberately: this ledger records the
    # one checkout-path decision per transaction. A later re-decision (appeal,
    # manual override) is a different event and belongs in a different table.
    Column("transaction_id", BigInteger, primary_key=True, autoincrement=False),
    Column("transaction_at", UtcDateTime, nullable=False),
    Column("decided_at", UtcDateTime, nullable=False),
    # Nullable on purpose: a degraded decision comes from the rule ladder with
    # no model behind it. Writing 0.0 would be a fabricated observation that
    # drags every drift and calibration metric toward zero in proportion to the
    # outage — the dashboards would look calmest exactly when the model was most
    # broken.
    Column("score", Float, nullable=True),
    Column("calibrated_probability", Float, nullable=True),
    Column("action", Text, nullable=False),
    Column("reason_codes", JSON, nullable=False),
    Column("model_version", Text, nullable=False),
    Column("policy_version", Text, nullable=False),
    Column("feature_version", Text, nullable=False),
    Column("config_hash", Text, nullable=False),
    Column("input_hash", Text, nullable=False),
    Column("degraded", Boolean, nullable=False),
    Column("degraded_reason", Text, nullable=True),
    CheckConstraint(
        "calibrated_probability IS NULL"
        " OR (calibrated_probability >= 0 AND calibrated_probability <= 1)",
        name="ck_decision_probability_range",
    ),
    CheckConstraint(f"action IN ({_ACTION_LIST})", name="ck_decision_action"),
    # Keeps "probability IS NULL means degraded" a guarantee rather than a
    # convention, so the filter every downstream metric depends on is
    # enforceable instead of remembered.
    CheckConstraint(
        "(degraded = false AND calibrated_probability IS NOT NULL AND score IS NOT NULL)"
        " OR (degraded = true AND calibrated_probability IS NULL AND score IS NULL)",
        name="ck_decision_score_presence",
    ),
    # ADR-0002 §4: a degraded decision must be excludable from downstream
    # analysis rather than silently absorbed. A row flagged degraded with no
    # stated reason cannot be excluded on evidence, so the pair is constrained.
    CheckConstraint(
        "(degraded = false AND degraded_reason IS NULL)"
        " OR (degraded = true AND degraded_reason IS NOT NULL)",
        name="ck_decision_degraded_reason",
    ),
    Index("ix_decision_ledger_transaction_at", "transaction_at"),
)

revealed_labels = Table(
    "revealed_labels",
    metadata,
    Column(
        "transaction_id",
        BigInteger,
        ForeignKey("decision_ledger.transaction_id"),
        primary_key=True,
        autoincrement=False,
    ),
    Column("is_fraud", Boolean, nullable=False),
    Column("transaction_at", UtcDateTime, nullable=False),
    # revealed_at is simulated time: when the dispute would have landed.
    # recorded_at is wall-clock: when this process wrote the row. Both are kept
    # because reconciling them is how you prove the revealer was not run early.
    Column("revealed_at", UtcDateTime, nullable=False),
    Column("recorded_at", UtcDateTime, nullable=False),
    Column("dispute_lag_days", Float, nullable=False),
    Column("label_source", Text, nullable=False),
    CheckConstraint("dispute_lag_days >= 0", name="ck_label_lag_non_negative"),
    CheckConstraint("revealed_at >= transaction_at", name="ck_label_reveal_after_transaction"),
    Index("ix_revealed_labels_revealed_at", "revealed_at"),
)

# Human adjudications — an opinion, not an outcome. Separate from
# ``revealed_labels`` for the same reason ``shadow_scores`` is separate from
# ``decision_ledger``: mixing two different kinds of observation in one table
# means every consumer has to remember to filter, and the one that forgets
# silently corrupts the training set (E12c). ``revealed_labels`` has
# ``transaction_id`` as a sole primary key, so the two cannot share a table
# in any case — a human label arrives ~90 days before its chargeback.
#
# Composite key on (transaction_id, adjudicator): the same person does not
# adjudicate the same transaction twice, but a second opinion is a real thing
# and is a new row, not an edit.
human_adjudications = Table(
    "human_adjudications",
    metadata,
    Column(
        "transaction_id",
        BigInteger,
        ForeignKey("decision_ledger.transaction_id"),
        primary_key=True,
        autoincrement=False,
    ),
    Column("is_fraud", Boolean, nullable=False),
    Column("adjudicated_at", UtcDateTime, nullable=False),
    Column("adjudicator", Text, primary_key=True),
    Index("ix_human_adjudications_adjudicated_at", "adjudicated_at"),
)

_TABLES: tuple[str, ...] = ("decision_ledger", "revealed_labels", "human_adjudications")


def insert_ignoring_duplicates(engine: Engine, table: Table) -> Insert:
    """``INSERT ... ON CONFLICT DO NOTHING`` for whichever engine we are on.

    DO NOTHING rather than DO UPDATE: an upsert is an UPDATE, which the
    append-only trigger rejects — rightly, since re-emitting a transaction must
    be a no-op, not a rewrite of history. The conflict target is the table's
    primary key columns, so this works for both single-column (``transaction_id``)
    and composite (``transaction_id, adjudicator``) keys.
    """
    pk = [c.name for c in table.primary_key.columns]
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        return pg_insert(table).on_conflict_do_nothing(index_elements=pk)
    if engine.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        return sqlite_insert(table).on_conflict_do_nothing(index_elements=pk)
    raise ValueError(f"unsupported dialect {engine.dialect.name!r}")


def sqlite_append_only_ddl() -> tuple[str, ...]:
    """Triggers that make both tables append-only under SQLite.

    The Postgres equivalent lives in ``migrations/003_append_only_guards.sql``.
    Two dialects, two statements — this is not an abstraction over a single
    implementation, it is the same invariant expressed in the only two engines
    this system runs on. It exists so the append-only guarantee is asserted by
    the default test suite rather than only in an integration run nobody blocks
    a merge on.
    """
    statements: list[str] = []
    for table in _TABLES:
        for verb in ("UPDATE", "DELETE"):
            statements.append(
                f"CREATE TRIGGER {table}_no_{verb.lower()} BEFORE {verb} ON {table} "
                f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END"
            )
    return tuple(statements)
