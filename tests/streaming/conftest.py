"""SQLite fixtures so the ledger invariants are asserted in the default suite.

A test that only runs when Postgres happens to be up is a test nobody blocks a
merge on. The append-only and idempotency guarantees are the point of this
module, so they are exercised against SQLite here and re-exercised against the
real engine under the ``integration`` marker.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from fraudlens.streaming.migrate import migrate

# An arbitrary but fixed replay epoch. Deliberately not "now": a test whose
# expectations move with the wall clock is a test that fails on a Tuesday.
EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


@pytest.fixture
def engine() -> Iterator[Engine]:
    # StaticPool keeps every connection pointed at the same in-memory database;
    # the ledger opens a fresh connection per write.
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    migrate(eng)
    try:
        yield eng
    finally:
        eng.dispose()
