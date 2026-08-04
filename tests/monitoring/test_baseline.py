"""The baseline is an artifact: same data must hash the same, edits must be caught."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from fraudlens.monitoring.baseline import Baseline, capture_baseline, load_baseline

EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _capture(values: np.ndarray, captured_at: dt.datetime = EPOCH) -> Baseline:
    return capture_baseline(
        captured_at=captured_at,
        window_start=EPOCH - dt.timedelta(days=120),
        window_end=EPOCH,
        n_rows=values.size,
        numeric={"p": values},
        categorical={"card4": ["visa"] * 7 + ["mastercard"] * 3},
    )


def test_identical_data_hashes_identically_even_at_a_different_capture_time() -> None:
    """`captured_at` is recorded but excluded from the hash.

    Otherwise re-running the capture job produces a new id for an unchanged reference
    distribution, and the id stops being able to answer the only question it exists for:
    did the yardstick move, or did the traffic?
    """
    values = np.linspace(0.0, 1.0, 5_000)
    first = _capture(values)
    second = _capture(values, captured_at=EPOCH + dt.timedelta(days=30))
    assert first.baseline_id == second.baseline_id


def test_a_changed_distribution_changes_the_id() -> None:
    a = _capture(np.linspace(0.0, 1.0, 5_000))
    b = _capture(np.linspace(0.0, 0.9, 5_000))
    assert a.baseline_id != b.baseline_id


def test_baseline_round_trips_through_json(tmp_path: Path) -> None:
    original = _capture(np.linspace(0.0, 1.0, 5_000))
    path = tmp_path / "baseline.json"
    original.save(path)
    loaded = load_baseline(path)
    assert loaded == original
    # The forced-open outer edges must survive the trip, or out-of-range traffic starts
    # landing in a bin that does not exist.
    assert loaded.numeric["p"].bin_edges[0] == -np.inf
    assert loaded.numeric["p"].bin_edges[-1] == np.inf


def test_an_edited_baseline_file_is_rejected(tmp_path: Path) -> None:
    """Editing the JSON until the alert stops firing is the realistic failure mode.

    It is undetectable unless somebody recomputes the hash, so loading recomputes it.
    """
    path = tmp_path / "baseline.json"
    _capture(np.linspace(0.0, 1.0, 5_000)).save(path)
    path.write_text(path.read_text(encoding="utf-8").replace('"n_rows": 5000', '"n_rows": 9999'))
    with pytest.raises(ValueError, match="does not match its recorded id"):
        load_baseline(path)


def test_capture_refuses_a_naive_window() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        capture_baseline(
            captured_at=EPOCH,
            window_start=dt.datetime(2025, 1, 1),  # noqa: DTZ001 — the point of the test
            window_end=EPOCH,
            n_rows=10,
            numeric={"p": np.linspace(0.0, 1.0, 10)},
        )


def test_measured_bin_proportions_are_not_assumed_uniform() -> None:
    """A heavily tied feature collapses quantile edges; the masses are then not 0.1 each.

    Storing measured proportions rather than assuming uniform deciles is what stops a
    stationary tied feature from reporting drift against itself.
    """
    tied = np.array([0.0] * 8_000 + list(np.linspace(0.1, 1.0, 2_000)))
    baseline = capture_baseline(
        captured_at=EPOCH,
        window_start=EPOCH - dt.timedelta(days=120),
        window_end=EPOCH,
        n_rows=tied.size,
        numeric={"tied": tied},
    )
    proportions = baseline.numeric["tied"].proportions
    assert max(proportions) > 0.5
    assert sum(proportions) == pytest.approx(1.0)
