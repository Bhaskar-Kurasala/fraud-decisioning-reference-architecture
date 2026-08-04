"""The periodic job end to end: ledger in, gauges out, Tier 2 refused when it must be."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from prometheus_client import REGISTRY
from sqlalchemy import Engine

from fraudlens.monitoring.baseline import Baseline
from fraudlens.monitoring.maturity import ImmatureWindowError, label_maturity
from fraudlens.monitoring.report import (
    SCORE_FEATURE,
    emit,
    matured_performance,
    run_report,
)
from fraudlens.streaming.labels import DisputeLagModel
from fraudlens.streaming.ledger import DecisionLedger, DecisionRecord

EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _seed_ledger(
    engine: Engine, n: int, *, window_days: int, offset_days: int, shift: bool = False
) -> None:
    """Write `n` decisions plus their scheduled labels, using the production lag model."""
    from fraudlens.streaming.labels import LabelRevealer

    rng = np.random.default_rng(41)
    ledger = DecisionLedger(engine)
    revealer = LabelRevealer(engine, lag_model=DisputeLagModel())
    probabilities = rng.beta(0.35, 9.0, size=n)
    if shift:
        probabilities = (probabilities * 27.39) / (probabilities * 27.39 + (1 - probabilities))
    is_fraud = rng.random(n) < probabilities

    records = []
    pending = []
    for i in range(n):
        at = EPOCH + dt.timedelta(days=offset_days + window_days * i / n, seconds=1)
        records.append(
            DecisionRecord(
                transaction_id=offset_days * 1_000_000 + i,
                transaction_at=at,
                decided_at=at,
                score=float(probabilities[i]),
                calibrated_probability=float(probabilities[i]),
                action="allow" if probabilities[i] < 0.5 else "deny",
                reason_codes=("TEST",),
                model_version="m1",
                policy_version="p1",
                feature_version="f1",
                config_hash="c1",
                input_hash=f"h{i}",
            )
        )
        pending.append(
            revealer.schedule(records[-1].transaction_id, at, is_fraud=bool(is_fraud[i]))
        )
    ledger.record_many(records)
    revealer.reveal_due(pending, EPOCH + dt.timedelta(days=10_000))


def test_a_matured_window_produces_tier2(engine: Engine, score_baseline: Baseline) -> None:
    _seed_ledger(engine, 4_000, window_days=20, offset_days=0)
    report = run_report(
        engine,
        score_baseline,
        window_start=EPOCH,
        window_end=EPOCH + dt.timedelta(days=20),
        as_of=EPOCH + dt.timedelta(days=200),
    )
    assert report.n_decisions == 4_000
    assert report.tier2_refusal is None
    assert report.tier2 is not None
    assert 0.0 <= report.tier2.auc <= 1.0
    # Every Tier 2 number carries its maturity stamp (§5). A figure without one is not
    # interpretable, and an uninterpretable number on an operator's screen gets acted on.
    assert "100.0% of 4000 labels matured" in report.tier2.maturity_stamp
    assert report.score_drift is not None and not report.score_drift.alerting


def test_the_final_window_refuses_tier2_but_still_reports_tier0(
    engine: Engine, score_baseline: Baseline
) -> None:
    """The organising constraint, end to end.

    Tier 0 is computed and published; Tier 2 is refused. That asymmetry is the reason the
    tiers exist — the leading indicators carry the load while the labels mature.
    """
    as_of = EPOCH + dt.timedelta(days=182)
    _seed_ledger(engine, 4_000, window_days=20, offset_days=162)
    report = run_report(
        engine,
        score_baseline,
        window_start=EPOCH + dt.timedelta(days=162),
        window_end=as_of,
        as_of=as_of,
    )
    assert report.tier2 is None
    assert report.tier2_refusal is not None
    assert report.maturity.ratio < 0.05
    # Tier 0 still answers, with no labels involved at all.
    assert report.score_drift is not None
    assert sum(report.action_rates.values()) == pytest.approx(1.0)
    assert not np.isnan(report.mean_predicted_probability)


def test_a_tier2_refusal_does_not_blank_the_label_free_gauges(
    engine: Engine, score_baseline: Baseline
) -> None:
    """The tier split has to survive contact with the emitter, not just the report.

    Mean and p95 predicted probability are §5 Tier 0 — they need no labels. An earlier
    version of `emit` set the mean to NaN alongside the Tier 2 block because both were
    read off the same `CalibrationReport`, which meant the leading indicators went dark
    for the same 30-90 days the lagging ones did. That is the failure the tiers exist to
    prevent, so it is asserted rather than assumed.
    """
    _seed_ledger(engine, 3_000, window_days=20, offset_days=0)
    report = run_report(
        engine,
        score_baseline,
        window_start=EPOCH,
        window_end=EPOCH + dt.timedelta(days=20),
        as_of=EPOCH + dt.timedelta(days=5),
    )
    emit(report)

    assert report.tier2 is None  # the window is immature, as intended
    assert np.isnan(REGISTRY.get_sample_value("fraudlens_observed_fraud_rate"))
    mean = REGISTRY.get_sample_value("fraudlens_predicted_fraud_rate")
    p95 = REGISTRY.get_sample_value("fraudlens_predicted_probability_p95")
    assert mean is not None and not np.isnan(mean)
    assert p95 is not None and p95 >= mean


def test_a_shifted_score_alerts_on_psi_with_no_labels_at_all(
    engine: Engine, score_baseline: Baseline
) -> None:
    """The $4.36M failure, detected on day zero rather than in a quarter."""
    _seed_ledger(engine, 4_000, window_days=20, offset_days=0, shift=True)
    report = run_report(
        engine,
        score_baseline,
        window_start=EPOCH,
        window_end=EPOCH + dt.timedelta(days=20),
        as_of=EPOCH + dt.timedelta(days=5),
    )
    assert report.score_drift is not None
    assert report.score_drift.alerting
    assert report.tier2 is None


def test_refused_tier2_gauges_are_nan_not_stale(engine: Engine, score_baseline: Baseline) -> None:
    """A gauge left unset keeps its last value forever.

    A refused run would otherwise leave yesterday's reassuring ECE on the dashboard
    indefinitely, so the panel looks healthiest exactly when it stopped measuring.
    """
    _seed_ledger(engine, 3_000, window_days=20, offset_days=0)
    good = run_report(
        engine,
        score_baseline,
        window_start=EPOCH,
        window_end=EPOCH + dt.timedelta(days=20),
        as_of=EPOCH + dt.timedelta(days=200),
    )
    emit(good)
    assert not np.isnan(REGISTRY.get_sample_value("fraudlens_calibration_ece") or float("nan"))

    stale = run_report(
        engine,
        score_baseline,
        window_start=EPOCH,
        window_end=EPOCH + dt.timedelta(days=20),
        as_of=EPOCH + dt.timedelta(days=5),
    )
    emit(stale)
    assert np.isnan(REGISTRY.get_sample_value("fraudlens_calibration_ece"))
    assert (
        REGISTRY.get_sample_value(
            "fraudlens_monitoring_baseline_info", {"baseline_id": good.baseline_id}
        )
        == 1.0
    )


def test_matured_performance_refuses_rather_than_warns() -> None:
    """Requirement stated as a test: below the floor it raises, it does not return."""
    times = [EPOCH + dt.timedelta(days=1)] * 100
    revealed: list[dt.datetime | None] = [EPOCH + dt.timedelta(days=2)] * 10 + [None] * 90
    maturity = label_maturity(
        times,
        revealed,
        as_of=EPOCH + dt.timedelta(days=3),
        window_start=EPOCH,
        window_end=EPOCH + dt.timedelta(days=2),
    )
    y = np.array([0, 1] * 5, dtype=np.int_)
    p = np.linspace(0.01, 0.9, 10)
    with pytest.raises(ImmatureWindowError):
        matured_performance(y, p, maturity, floor=0.80)


def test_a_baseline_without_a_score_reference_is_rejected(
    engine: Engine, score_baseline: Baseline
) -> None:
    """Score-distribution PSI is the earliest warning available; it cannot be optional."""
    stripped = Baseline(
        baseline_id=score_baseline.baseline_id,
        captured_at=score_baseline.captured_at,
        window_start=score_baseline.window_start,
        window_end=score_baseline.window_end,
        n_rows=score_baseline.n_rows,
        numeric={},
        categorical={},
    )
    assert SCORE_FEATURE not in stripped.numeric
    with pytest.raises(ValueError, match="cannot be skipped"):
        run_report(
            engine,
            stripped,
            window_start=EPOCH,
            window_end=EPOCH + dt.timedelta(days=20),
            as_of=EPOCH + dt.timedelta(days=200),
        )
