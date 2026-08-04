"""Replay producer — turns a static snapshot back into a stream.

The dataset is 182 days that already happened. Scoring it as a batch would
measure nothing about operating a model: drift, label latency and decay are
properties of *time passing*, and a batch has none. So the test window is
re-emitted in its original order, at a configurable speed, and the rest of the
system experiences it as arriving traffic.

Restarting from the beginning would corrupt the experiment — the ledger would
see the same day twice and every rate metric computed over it would be wrong.
Hence the checkpoint. Delivery is at-least-once: the checkpoint is committed
*after* the consumer has handled an event, so a crash in between replays it.
That is deliberate and is why the ledger writes are idempotent; the alternative
(checkpoint first) is at-most-once, which silently drops decisions.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

# TransactionDT is not carried in the scored artifacts, only the derived `day`.
# Within a day we order by TransactionID, which IEEE-CIS assigns in timestamp
# order — an assumption about the dataset, recorded here because it is the one
# thing that would silently reorder the stream if it were false.
_REQUIRED_COLUMNS = ("TransactionID", "isFraud", "day")

SECONDS_PER_DAY = 86_400.0


@dataclass(frozen=True)
class ReplayEvent:
    """One transaction, presented as if it had just happened."""

    transaction_id: int
    day: int
    transaction_at: dt.datetime
    is_fraud: bool
    # The remaining source columns, untouched. The producer does not know what
    # the decisioning layer needs and does not decide for it.
    payload: Mapping[str, Any]


class ReplayCheckpoint:
    """Last committed stream position, as ``(day, transaction_id)``.

    A file rather than a table: the producer has no other reason to hold a
    database connection, and a checkpoint that shares a transaction with the
    ledger would couple restart semantics to ledger availability.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def position(self) -> tuple[int, int] | None:
        if not self._path.exists():
            return None
        state = json.loads(self._path.read_text(encoding="utf-8"))
        return int(state["day"]), int(state["transaction_id"])

    def commit(self, event: ReplayEvent) -> None:
        # Write-then-rename: a crash mid-write must not leave a truncated
        # checkpoint, which would resume from an arbitrary point rather than
        # from the beginning.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(
            json.dumps({"day": event.day, "transaction_id": event.transaction_id}),
            encoding="utf-8",
        )
        tmp.replace(self._path)


def _load_rows(source: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(source)
    missing = sorted(set(_REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}")
    frame = frame.sort_values(["day", "TransactionID"], kind="mergesort").reset_index(drop=True)
    # Position within the day, so a day's volume is spread across its replay
    # slot instead of arriving as an instantaneous burst at midnight.
    frame["_frac"] = frame.groupby("day").cumcount() / frame.groupby("day")[
        "TransactionID"
    ].transform("size")
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


class ReplayProducer:
    """Emits transactions in time order at ``real_seconds_per_replay_day``."""

    def __init__(
        self,
        source: Path,
        *,
        epoch: dt.datetime,
        real_seconds_per_replay_day: float,
        checkpoint: ReplayCheckpoint | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if epoch.tzinfo is None:
            raise ValueError("epoch must be timezone-aware")
        if real_seconds_per_replay_day < 0:
            raise ValueError("real_seconds_per_replay_day must be non-negative")
        self._source = source
        self._epoch = epoch
        self._pace = real_seconds_per_replay_day
        self._checkpoint = checkpoint
        self._sleep = sleep
        self._monotonic = monotonic

    def commit(self, event: ReplayEvent) -> None:
        """Record that ``event`` has been fully handled downstream."""
        if self._checkpoint is not None:
            self._checkpoint.commit(event)

    def _to_event(self, row: dict[str, Any]) -> ReplayEvent:
        day = int(row["day"])
        frac = float(row["_frac"])
        payload = {k: v for k, v in row.items() if k != "_frac"}
        return ReplayEvent(
            transaction_id=int(row["TransactionID"]),
            day=day,
            transaction_at=self._epoch + dt.timedelta(days=day, seconds=SECONDS_PER_DAY * frac),
            is_fraud=bool(row["isFraud"]),
            payload=payload,
        )

    def stream(self) -> Iterator[ReplayEvent]:
        """Yield events in order, resuming past the checkpoint if one exists."""
        rows = _load_rows(self._source)
        resume_from = self._checkpoint.position() if self._checkpoint else None
        if resume_from is not None:
            rows = [r for r in rows if (int(r["day"]), int(r["TransactionID"])) > resume_from]
        if not rows:
            return
        origin = float(rows[0]["day"]) + float(rows[0]["_frac"])
        started = self._monotonic()
        for row in rows:
            offset_days = float(row["day"]) + float(row["_frac"]) - origin
            delay = started + offset_days * self._pace - self._monotonic()
            if delay > 0:
                self._sleep(delay)
            yield self._to_event(row)
