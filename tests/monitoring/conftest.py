"""Fixtures for the monitoring suite.

The ledger fixtures mirror `tests/streaming/conftest.py` rather than importing it:
pytest does not share conftest across sibling packages, and a cross-package import of a
conftest is the kind of thing that works until someone runs one directory.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import numpy as np
import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.pool import StaticPool

from fraudlens.monitoring.baseline import Baseline, capture_baseline
from fraudlens.monitoring.report import SCORE_FEATURE
from fraudlens.streaming.migrate import migrate

# Fixed replay epoch. A test whose expectations move with the wall clock fails on a
# Tuesday for reasons nobody can reconstruct on a Wednesday.
EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    migrate(eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture
def score_baseline() -> Baseline:
    """A baseline over a beta-shaped score distribution roughly like the champion's."""
    rng = np.random.default_rng(11)
    scores = rng.beta(0.35, 9.0, size=20_000)
    return capture_baseline(
        captured_at=EPOCH,
        window_start=EPOCH - dt.timedelta(days=120),
        window_end=EPOCH,
        n_rows=scores.size,
        numeric={SCORE_FEATURE: scores},
    )
