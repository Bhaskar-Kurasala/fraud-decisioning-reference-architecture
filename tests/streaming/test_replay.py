"""Replay ordering, pacing, and resume. A restart from zero corrupts the run."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from fraudlens.streaming.replay import ReplayCheckpoint, ReplayEvent, ReplayProducer

from .conftest import EPOCH


@pytest.fixture
def source(tmp_path: Path) -> Path:
    # Deliberately shuffled on disk: the producer's job is to impose time order,
    # so a fixture that is already sorted would not test anything.
    frame = pd.DataFrame(
        {
            "TransactionID": [30, 10, 21, 11, 20, 31],
            "day": [151, 150, 151, 150, 151, 152],
            "isFraud": [0, 0, 1, 0, 0, 0],
            "TransactionAmt": [58.95, 39.0, 226.0, 12.5, 99.0, 400.0],
            "ProductCD": list("WWWCWW"),
        }
    )
    path = tmp_path / "scored_test.parquet"
    frame.to_parquet(path)
    return path


def collect(producer: ReplayProducer) -> list[ReplayEvent]:
    return list(producer.stream())


def build(source: Path, **kwargs: object) -> ReplayProducer:
    params: dict[str, object] = {
        "epoch": EPOCH,
        "real_seconds_per_replay_day": 0.0,
        # An injected clock: the 32-day test window must be replayable in
        # milliseconds, and a test that sleeps is a test that gets deleted.
        "sleep": lambda _: None,
        "monotonic": lambda: 0.0,
    }
    params.update(kwargs)
    return ReplayProducer(source, **params)  # type: ignore[arg-type]


def test_emits_in_time_order(source: Path) -> None:
    events = collect(build(source))
    assert [e.transaction_id for e in events] == [10, 11, 20, 21, 30, 31]
    assert [e.day for e in events] == [150, 150, 151, 151, 151, 152]


def test_transaction_time_is_timezone_aware_and_monotonic(source: Path) -> None:
    events = collect(build(source))
    stamps = [e.transaction_at for e in events]
    assert all(s.tzinfo is not None for s in stamps)
    assert stamps == sorted(stamps)
    assert stamps[0] == EPOCH + dt.timedelta(days=150)


def test_payload_carries_the_source_columns_untouched(source: Path) -> None:
    first = collect(build(source))[0]
    assert first.payload["TransactionAmt"] == 39.0
    assert first.payload["ProductCD"] == "W"
    # Internal pacing column must not leak into the decisioning layer.
    assert "_frac" not in first.payload


def test_resume_continues_past_the_checkpoint(source: Path, tmp_path: Path) -> None:
    checkpoint = ReplayCheckpoint(tmp_path / "replay.json")
    producer = build(source, checkpoint=checkpoint)

    for event in producer.stream():
        producer.commit(event)
        if event.transaction_id == 20:
            break

    resumed = collect(build(source, checkpoint=checkpoint))
    assert [e.transaction_id for e in resumed] == [21, 30, 31]


def test_an_exhausted_replay_resumes_to_nothing(source: Path, tmp_path: Path) -> None:
    checkpoint = ReplayCheckpoint(tmp_path / "replay.json")
    producer = build(source, checkpoint=checkpoint)
    for event in producer.stream():
        producer.commit(event)

    assert collect(build(source, checkpoint=checkpoint)) == []


def test_uncommitted_events_are_replayed_not_lost(source: Path, tmp_path: Path) -> None:
    """At-least-once: a crash after emitting but before committing re-emits.

    This is why the ledger writes are idempotent rather than the checkpoint
    being written first.
    """
    checkpoint = ReplayCheckpoint(tmp_path / "replay.json")
    producer = build(source, checkpoint=checkpoint)
    emitted = []
    for event in producer.stream():
        emitted.append(event.transaction_id)
        if event.transaction_id == 11:
            break  # consumer crashes here, before commit
        producer.commit(event)

    resumed = [e.transaction_id for e in build(source, checkpoint=checkpoint).stream()]
    assert resumed[0] == 11


def test_speed_multiplier_paces_by_replay_day(source: Path) -> None:
    """1 replay-day = N real-seconds, measured without waiting for any of it."""
    slept: list[float] = []
    clock = {"t": 0.0}

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["t"] += seconds

    producer = build(
        source,
        real_seconds_per_replay_day=10.0,
        sleep=sleep,
        monotonic=lambda: clock["t"],
    )
    list(producer.stream())

    # Days 150, 150, 151, 151, 151, 152 -> two full-day gaps plus within-day
    # spreading, so total elapsed is two replay-days of pacing.
    assert clock["t"] == pytest.approx(20.0, abs=10.0 / 3 + 1e-6)
    assert all(s >= 0 for s in slept)


def test_missing_required_column_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"TransactionID": [1], "day": [150]}).to_parquet(path)
    with pytest.raises(ValueError, match="isFraud"):
        collect(build(path))


def test_naive_epoch_is_rejected(source: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build(source, epoch=dt.datetime(2026, 1, 1))  # noqa: DTZ001
