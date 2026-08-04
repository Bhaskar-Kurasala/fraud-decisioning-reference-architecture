"""The same invariants against the real engine.

SQLite proves the logic; only Postgres proves the plpgsql triggers, the REVOKE,
and TIMESTAMPTZ round-tripping. Marked ``integration`` because it needs the
compose stack — run with::

    docker compose -f deploy/compose/postgres.yml --profile core up -d
    FRAUDLENS_DATABASE_URL=postgresql+psycopg://fraudlens:fraudlens@localhost/fraudlens \\
        uv run pytest -m integration
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DatabaseError

from fraudlens.streaming.labels import LabelRevealer
from fraudlens.streaming.ledger import DecisionLedger
from fraudlens.streaming.migrate import migrate

from .conftest import EPOCH
from .test_ledger import make_decision

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("FRAUDLENS_DATABASE_URL", "")


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    if not DATABASE_URL:
        pytest.skip("FRAUDLENS_DATABASE_URL not set")
    eng = create_engine(DATABASE_URL)
    with eng.begin() as conn:
        # DROP is allowed; UPDATE and DELETE are not. Rebuilding the schema is a
        # reviewable DDL change, which is exactly the distinction ADR-0002 wants.
        conn.execute(text("DROP TABLE IF EXISTS revealed_labels, decision_ledger CASCADE"))
    migrate(eng)
    try:
        yield eng
    finally:
        eng.dispose()


def test_migrations_are_idempotent(pg_engine: Engine) -> None:
    migrate(pg_engine)


def test_postgres_trigger_blocks_update_and_delete(pg_engine: Engine) -> None:
    DecisionLedger(pg_engine).record(make_decision())
    for statement in ("UPDATE decision_ledger SET action = 'deny'", "DELETE FROM decision_ledger"):
        with pytest.raises(DatabaseError, match="append-only"), pg_engine.begin() as conn:
            conn.execute(text(statement))
    assert DecisionLedger(pg_engine).count() == 1


def test_postgres_idempotent_write(pg_engine: Engine) -> None:
    ledger = DecisionLedger(pg_engine)
    assert ledger.record(make_decision()) is True
    assert ledger.record(make_decision()) is False


def test_timestamptz_round_trips_across_an_offset(pg_engine: Engine) -> None:
    ledger = DecisionLedger(pg_engine)
    offset = dt.timezone(dt.timedelta(hours=-5))
    decided = EPOCH.astimezone(offset)
    ledger.record(make_decision(4242, decided_at=decided))

    stored = ledger.get(4242)
    assert stored is not None
    assert stored.decided_at == decided


def test_labels_reveal_on_schedule_against_postgres(pg_engine: Engine) -> None:
    DecisionLedger(pg_engine).record(make_decision(77))
    revealer = LabelRevealer(pg_engine, now=lambda: EPOCH)
    label = revealer.schedule(77, EPOCH, is_fraud=True)

    assert revealer.reveal_due([label], label.revealed_at - dt.timedelta(seconds=1)) == [label]
    assert revealer.matured(EPOCH + dt.timedelta(days=365)) == []

    revealer.reveal_due([label], label.revealed_at)
    assert len(revealer.matured(label.revealed_at)) == 1
