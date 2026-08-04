"""Label-latency behaviour, and the one invariant that must never break."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy import Engine

from fraudlens.streaming.labels import DisputeLagModel, LabelRevealer, PendingLabel

from .conftest import EPOCH


def build_pending(revealer: LabelRevealer, count: int = 500) -> list[PendingLabel]:
    return [
        revealer.schedule(
            transaction_id=3_485_000 + i,
            transaction_at=EPOCH + dt.timedelta(days=i % 32),
            is_fraud=i % 29 == 0,
        )
        for i in range(count)
    ]


def test_no_label_is_readable_before_its_reveal_time(engine: Engine) -> None:
    """THE invariant. A leaked label inflates every model metric downstream.

    Swept across the whole maturation horizon rather than spot-checked: the
    failure mode is one boundary being wrong, not the mechanism being absent.
    """
    revealer = LabelRevealer(engine, now=lambda: EPOCH)
    pending = build_pending(revealer)
    schedule = {label.transaction_id: label.revealed_at for label in pending}

    for offset_days in range(0, 400, 7):
        as_of = EPOCH + dt.timedelta(days=offset_days)
        pending = revealer.reveal_due(pending, as_of)

        # Nothing in the table, and nothing returned by the read path, may have
        # a reveal time in the future.
        matured = revealer.matured(as_of)
        assert all(label.revealed_at <= as_of for label in matured), (
            f"leaked a label at as_of={as_of.isoformat()}"
        )
        # And every id we got back really was due, per the schedule computed
        # independently of the database.
        assert all(schedule[label.transaction_id] <= as_of for label in matured)
        # Nothing still queued is already due.
        assert all(label.revealed_at > as_of for label in pending)


def test_a_future_label_is_absent_not_merely_filtered(engine: Engine) -> None:
    """There is no pending row in the database for a careless join to find."""
    revealer = LabelRevealer(engine, now=lambda: EPOCH)
    label = revealer.schedule(1, EPOCH, is_fraud=True)
    day_before = label.revealed_at - dt.timedelta(seconds=1)

    remaining = revealer.reveal_due([label], day_before)

    assert remaining == [label]
    with engine.connect() as conn:
        from sqlalchemy import func, select

        from fraudlens.streaming.schema import revealed_labels

        assert conn.execute(select(func.count()).select_from(revealed_labels)).scalar_one() == 0


def test_revealing_twice_does_not_double_write(engine: Engine) -> None:
    revealer = LabelRevealer(engine, now=lambda: EPOCH)
    pending = build_pending(revealer, count=50)
    late = EPOCH + dt.timedelta(days=500)

    revealer.reveal_due(pending, late)
    revealer.reveal_due(pending, late)

    assert len(revealer.matured(late)) == 50


def test_lag_is_a_pure_function_of_the_transaction(engine: Engine) -> None:
    """A producer restart must not redraw the lag, or the experiment shifts."""
    model = DisputeLagModel()
    first = [model.lag_days(i, is_fraud=True) for i in range(200)]
    second = [DisputeLagModel().lag_days(i, is_fraud=True) for i in range(200)]

    assert first == second
    # And a different seed gives a different world, so the seed is load-bearing.
    assert first != [DisputeLagModel(seed=8).lag_days(i, is_fraud=True) for i in range(200)]


def test_dispute_lag_is_distributed_not_constant() -> None:
    """Chargebacks mature over 30-90 days; a constant delay hides the tail."""
    model = DisputeLagModel()
    lags = sorted(model.lag_days(i, is_fraud=True) for i in range(20_000))

    median = lags[len(lags) // 2]
    assert 32.0 < median < 36.0, median
    # The long tail is the part that hurts retraining: some disputes land after
    # the retrain window has already closed.
    assert lags[0] < 10.0
    assert lags[-1] > 180.0


def test_naive_datetimes_are_rejected(engine: Engine) -> None:
    revealer = LabelRevealer(engine)
    with pytest.raises(ValueError, match="timezone-aware"):
        revealer.schedule(1, dt.datetime(2026, 1, 1), is_fraud=False)  # noqa: DTZ001
    with pytest.raises(ValueError, match="timezone-aware"):
        revealer.matured(dt.datetime(2026, 1, 1))  # noqa: DTZ001


X_PARQUET = Path("data/X.parquet")


@pytest.mark.skipif(not X_PARQUET.exists(), reason="data/ is git-ignored; run 00_download_data.sh")
def test_reproduces_the_measured_label_latency_shape() -> None:
    """Findings §7: 38.7% of training-window fraud undisputed at train time,
    and the observed rate collapsing to ~6.9% of true in the final block.

    Reproduced distributionally, not bit-for-bit: the lag here is hashed per
    transaction rather than drawn from a sequential seeded stream, because a
    replay must survive a restart. research/06 remains the provenance of the
    published digits.
    """
    import pandas as pd

    train = pd.read_parquet(X_PARQUET, columns=["isFraud", "split", "day"])
    train = train[train.split == "train"]
    model = DisputeLagModel()
    train_end_day = 119

    lag = [
        model.lag_days(i, is_fraud=bool(f)) for i, f in enumerate(train.isFraud.to_numpy(), start=1)
    ]
    train = train.assign(arrived=(train.day.to_numpy() + lag) <= train_end_day)
    # The bug being reproduced: an undisputed transaction is coded legitimate.
    observed = (train.isFraud.to_numpy() == 1) & train.arrived.to_numpy()

    invisible = 1 - observed.sum() / train.isFraud.sum()
    assert 0.34 < invisible < 0.44, f"{invisible:.3%} invisible (published: 38.7%)"

    block = train.day.to_numpy() // 20
    final = block == block.max()
    ratio = observed[final].mean() / train.isFraud.to_numpy()[final].mean()
    assert ratio < 0.12, f"final-block observed/true ratio {ratio:.4f} (published: 0.069)"
