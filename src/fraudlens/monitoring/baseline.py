"""The training-window reference distribution, captured as a versioned artifact.

"PSI 0.31 versus baseline" is a meaningless sentence unless *which* baseline is
recoverable. A reference distribution that can be silently regenerated turns every
drift alert into an unfalsifiable claim: the number moved, and nobody can say whether
the traffic changed or the yardstick did. So a baseline is treated exactly like a model
artifact — content-hashed, serialisable, and carrying the window it was measured over.

The hash is `models.provenance.config_hash`, not a second hashing function. Two hash
definitions in one repo is two answers to "did this change", and the point of a hash is
that there is one.

What is stored is a *summary*, never the rows: a quantile grid, bin edges, bin
proportions and a null rate per feature. That is enough to bin new traffic identically
and to approximate the reference CDF, and it means a baseline is a few kilobytes of JSON
that can live in git rather than a data extract that cannot.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from fraudlens.models.provenance import config_hash

FloatArray = npt.NDArray[np.float64]

# Deciles. Ten bins is the PSI convention the 0.25 threshold in spec §5.2 was chosen
# against; changing it changes the number without changing the traffic, so it is fixed
# here rather than made configurable (§9a: no configuration for a value that never
# changes). The consequence is stated plainly: a baseline captured with different bins
# is not comparable to one captured with these, which is why the bin edges travel inside
# the artifact and are re-used rather than recomputed at scoring time.
DEFAULT_BINS = 10

# 101-point grid: the reference CDF is resolved to 0.01. That is ample for a drift alarm
# and deliberately insufficient for a hypothesis test — the KS statistic computed off it
# is an indicator, not a p-value, and `drift.ks_statistic` says so.
QUANTILE_LEVELS: FloatArray = np.linspace(0.0, 1.0, 101, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class NumericReference:
    """Reference summary for one numeric feature."""

    name: str
    quantiles: tuple[float, ...]
    bin_edges: tuple[float, ...]
    # Measured rather than assumed uniform. Heavily tied features (TransactionAmt at
    # round values, D1 at 0) collapse quantile edges, so real decile masses are not 0.1
    # and a PSI that assumed they were would report drift on a stationary population.
    proportions: tuple[float, ...]
    null_rate: float


@dataclass(frozen=True, slots=True)
class CategoricalReference:
    """Reference summary for one categorical feature."""

    name: str
    proportions: Mapping[str, float]
    null_rate: float


@dataclass(frozen=True, slots=True)
class Baseline:
    """A hashed, window-stamped reference distribution."""

    baseline_id: str
    captured_at: dt.datetime
    window_start: dt.datetime
    window_end: dt.datetime
    n_rows: int
    numeric: Mapping[str, NumericReference]
    categorical: Mapping[str, CategoricalReference]

    def as_json(self) -> str:
        return json.dumps(
            _payload(self)
            | {"baseline_id": self.baseline_id, "captured_at": self.captured_at.isoformat()},
            indent=2,
        )

    def save(self, path: Path) -> None:
        path.write_text(self.as_json(), encoding="utf-8")


def bin_proportions(values: FloatArray, edges: Sequence[float]) -> FloatArray:
    """Mass per bin under a fixed set of edges, nulls excluded.

    Right-open assignment via `digitize` on the interior edges, matching
    `models.metrics.expected_calibration_error`. Not a style choice: E2 traced a
    published-number discrepancy to a reimplementation that binned right-closed, and
    having two binning conventions in one codebase is how that happens again.

    Nulls are dropped rather than binned. A missing value is not a small value, and
    folding it into the lowest bin would report a null-rate spike as a distribution
    shift — the null rate is measured separately and alerted on separately (§5.2).
    """
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        return np.zeros(len(edges) - 1, dtype=np.float64)
    index = np.digitize(finite, np.asarray(edges[1:-1], dtype=np.float64))
    counts = np.bincount(index, minlength=len(edges) - 1).astype(np.float64)
    proportions: FloatArray = counts / finite.size
    return proportions


def numeric_reference(
    name: str, values: FloatArray, n_bins: int = DEFAULT_BINS
) -> NumericReference:
    """Summarise one numeric feature over the training window."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    finite = values[~np.isnan(values)]
    if finite.size == 0:
        raise ValueError(f"feature {name!r} is entirely null in the baseline window")
    quantiles = np.quantile(finite, QUANTILE_LEVELS)
    edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    # Outer edges forced open so a future value below the training minimum or above its
    # maximum lands in an existing bin instead of an eleventh one. Out-of-range traffic
    # is real (a new price point, a new merchant) and must show up as mass moving to an
    # edge bin, which PSI sees, rather than as an array-shape mismatch.
    edges[0], edges[-1] = -np.inf, np.inf
    return NumericReference(
        name=name,
        quantiles=tuple(float(q) for q in quantiles),
        bin_edges=tuple(float(e) for e in edges),
        proportions=tuple(float(p) for p in bin_proportions(values, [float(e) for e in edges])),
        null_rate=float(np.isnan(values).mean()),
    )


def categorical_reference(name: str, values: Sequence[str | None]) -> CategoricalReference:
    """Summarise one categorical feature over the training window."""
    if not values:
        raise ValueError(f"feature {name!r} has no rows in the baseline window")
    present = [v for v in values if v is not None]
    counts: dict[str, int] = {}
    for value in present:
        counts[value] = counts.get(value, 0) + 1
    total = len(present) or 1
    return CategoricalReference(
        name=name,
        proportions={level: count / total for level, count in sorted(counts.items())},
        null_rate=1.0 - len(present) / len(values),
    )


def capture_baseline(
    *,
    captured_at: dt.datetime,
    window_start: dt.datetime,
    window_end: dt.datetime,
    n_rows: int,
    numeric: Mapping[str, FloatArray],
    categorical: Mapping[str, Sequence[str | None]] | None = None,
    n_bins: int = DEFAULT_BINS,
) -> Baseline:
    """Capture and hash a reference distribution.

    `captured_at` is required rather than defaulted to now(): a wall-clock default puts
    the moment of capture into an artifact that is supposed to be reproducible, and then
    two identical captures differ. For the same reason `captured_at` is recorded but
    deliberately *excluded from the hash* — the id identifies the reference distribution,
    so re-capturing the same window over the same data yields the same id and proves the
    baseline did not change, which is exactly the question the id exists to answer.
    """
    for name, stamp in (("window_start", window_start), ("window_end", window_end)):
        if stamp.tzinfo is None:
            raise ValueError(f"{name} must be timezone-aware")
    if window_end <= window_start:
        raise ValueError("baseline window_end must be after window_start")
    baseline = Baseline(
        baseline_id="",
        captured_at=captured_at,
        window_start=window_start,
        window_end=window_end,
        n_rows=n_rows,
        numeric={n: numeric_reference(n, v, n_bins) for n, v in numeric.items()},
        categorical={n: categorical_reference(n, v) for n, v in (categorical or {}).items()},
    )
    return _replace_id(baseline, config_hash(_payload(baseline)))


def load_baseline(path: Path) -> Baseline:
    """Read a baseline and re-verify its id.

    The re-hash is the whole point of persisting the id. An edited JSON file is the
    realistic way a baseline gets "adjusted" until the alert stops firing, and that
    edit is undetectable unless somebody recomputes.
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    stored_id = str(data.pop("baseline_id"))
    baseline = Baseline(
        baseline_id=stored_id,
        captured_at=dt.datetime.fromisoformat(data["captured_at"]),
        window_start=dt.datetime.fromisoformat(data["window_start"]),
        window_end=dt.datetime.fromisoformat(data["window_end"]),
        n_rows=int(data["n_rows"]),
        numeric={n: NumericReference(**_numeric_from_json(r)) for n, r in data["numeric"].items()},
        categorical={n: CategoricalReference(**r) for n, r in data["categorical"].items()},
    )
    recomputed = config_hash(_payload(baseline))
    if recomputed != stored_id:
        raise ValueError(
            f"baseline at {path} does not match its recorded id "
            f"({stored_id[:12]} != {recomputed[:12]}); it was edited or truncated"
        )
    return baseline


def _numeric_from_json(row: Mapping[str, Any]) -> dict[str, Any]:
    # The forced-open outer edges are +/-inf, which strict JSON cannot express. Python's
    # json emits and parses the `Infinity` extension, so the artifact round-trips here
    # but is not portable to a strict reader — worth knowing before anyone points a JS
    # tool at these files.
    return {
        "name": row["name"],
        "quantiles": tuple(float(q) for q in row["quantiles"]),
        "bin_edges": tuple(float(e) for e in row["bin_edges"]),
        "proportions": tuple(float(p) for p in row["proportions"]),
        "null_rate": float(row["null_rate"]),
    }


def _payload(baseline: Baseline) -> dict[str, Any]:
    """The hashed content: the distribution and the window it describes, nothing else.

    `captured_at` is deliberately absent — see `capture_baseline`. It is serialised
    alongside the hash rather than inside it.
    """
    return {
        "window_start": baseline.window_start.isoformat(),
        "window_end": baseline.window_end.isoformat(),
        "n_rows": baseline.n_rows,
        "numeric": {
            name: {
                "name": ref.name,
                "quantiles": list(ref.quantiles),
                "bin_edges": list(ref.bin_edges),
                "proportions": list(ref.proportions),
                "null_rate": ref.null_rate,
            }
            for name, ref in sorted(baseline.numeric.items())
        },
        "categorical": {
            name: {
                "name": ref.name,
                "proportions": dict(sorted(ref.proportions.items())),
                "null_rate": ref.null_rate,
            }
            for name, ref in sorted(baseline.categorical.items())
        },
    }


def _replace_id(baseline: Baseline, baseline_id: str) -> Baseline:
    return Baseline(
        baseline_id=baseline_id,
        captured_at=baseline.captured_at,
        window_start=baseline.window_start,
        window_end=baseline.window_end,
        n_rows=baseline.n_rows,
        numeric=baseline.numeric,
        categorical=baseline.categorical,
    )
