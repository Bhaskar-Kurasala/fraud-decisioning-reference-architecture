"""Synthetic promotion windows.

Driven from scores rather than from estimators, deliberately. There is no champion
artifact on disk in this repository — the research scripts persisted scores, not fitted
models, and the registry's Production version 2 points at an artifact that is not there.
Every claim the gate makes is a claim about *scores and labels*, so a fixture that supplies
those exercises exactly what the gate does and nothing it does not.

The two instruments below are the ones findings §2 used:

- `bin_calibrated` is isotonic calibration in its simplest honest form — replace a score by
  the observed fraud rate of its rank bin. It produces a well-calibrated score whose
  discrimination is whatever the underlying ranking had.
- `odds_shift` is the closed-form prior shift. Strictly monotone, so it leaves AUC exactly
  unchanged while destroying calibration — which is the transform that isolates the thing
  a ranking metric cannot see.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import numpy.typing as npt
import pytest

from fraudlens.flywheel.promotion import PromotionWindow

# Fixed epoch, not `now`: a test whose expectations move with the wall clock fails on a
# Tuesday, and the whole subject here is time-dependent.
EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# Matches the dataset: ~3.5% base rate, ~$135 mean ticket.
BASE_RATE_LOGIT = -4.0
N_DEFAULT = 20_000
_CALIBRATION_BINS = 50


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260805)


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def odds_shift(probability: FloatArray, factor: float) -> FloatArray:
    odds = probability / (1.0 - probability) * factor
    return np.asarray(odds / (1.0 + odds), dtype=np.float64)


def bin_calibrated(score: FloatArray, y: BoolArray, n_bins: int = _CALIBRATION_BINS) -> FloatArray:
    """Observed fraud rate of each rank bin — a calibrated probability from any ranking."""
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.int64)
    ranks[order] = np.arange(len(score))
    bins = (ranks * n_bins) // len(score)
    out = np.empty(len(score), dtype=np.float64)
    for b in range(n_bins):
        mask = bins == b
        out[mask] = float(y[mask].mean())
    return np.clip(out, 1e-4, 1.0 - 1e-4)


def truth(rng: np.random.Generator, n: int = N_DEFAULT) -> tuple[FloatArray, BoolArray]:
    """A latent risk variable and the fraud outcomes it generated."""
    z = np.asarray(rng.normal(0.0, 1.0, size=n), dtype=np.float64)
    y = rng.random(n) < _sigmoid(1.6 * z + BASE_RATE_LOGIT)
    return z, np.asarray(y, dtype=bool)


def blur(rng: np.random.Generator, z: FloatArray, noise: float) -> FloatArray:
    """A strictly worse ranking of the same population."""
    return np.asarray(z + rng.normal(0.0, noise, size=len(z)), dtype=np.float64)


def build_window(
    rng: np.random.Generator,
    *,
    champion_probability: FloatArray,
    challenger_probability: FloatArray,
    is_fraud: BoolArray,
    matured_fraction: float,
    as_of: dt.datetime = EPOCH,
) -> PromotionWindow:
    """A window whose first `matured_fraction` of rows have revealed labels."""
    n = len(is_fraud)
    amount = np.asarray(rng.lognormal(4.2, 1.1, size=n), dtype=np.float64)
    tenure_days = np.asarray(rng.integers(0, 500, size=n), dtype=np.float64)
    transaction_times = [as_of - dt.timedelta(days=120) + dt.timedelta(minutes=i) for i in range(n)]
    n_matured = int(n * matured_fraction)
    revealed_times: list[dt.datetime | None] = [
        transaction_times[i] + dt.timedelta(days=34) if i < n_matured else None for i in range(n)
    ]
    labels: list[bool | None] = [bool(is_fraud[i]) if i < n_matured else None for i in range(n)]
    return PromotionWindow(
        transaction_times=transaction_times,
        revealed_times=revealed_times,
        is_fraud=labels,
        amount=amount,
        tenure_days=tenure_days,
        champion_probability=champion_probability,
        challenger_probability=challenger_probability,
    )
