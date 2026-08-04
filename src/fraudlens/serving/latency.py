"""Per-stage timing against the design spec §4.3 budget.

The SLO is `POST /v1/decide` p99 <= 150 ms end to end, allocated: feature assembly 60 ms,
model inference 25 ms, calibration + policy 15 ms, ledger write (async) 0 ms, overhead
50 ms. Per-stage rather than end-to-end only, because "the request took 400 ms" is not an
actionable page -- the allocation is what tells you whether to look at the feature store
or at the model.

The elapsed-time source is injected. `time.perf_counter` in production; a test drives a
scripted sequence, so a budget-breach test asserts on the counter rather than on a sleep
(§9a: tests need no network and no sleeping).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Final

from fraudlens.serving.metrics import BUDGET_BREACHES, STAGE_SECONDS

FEATURES: Final = "features"
INFERENCE: Final = "inference"
CALIBRATION_POLICY: Final = "calibration_policy"
TOTAL: Final = "total"

# Milliseconds, straight from §4.3. `total` is the SLO; the three stages are the
# allocation inside it and their sum (100 ms) deliberately leaves the 50 ms of overhead
# the spec budgets for deserialisation, validation and the response write.
BUDGET_MS: Final[dict[str, float]] = {
    FEATURES: 60.0,
    INFERENCE: 25.0,
    CALIBRATION_POLICY: 15.0,
    TOTAL: 150.0,
}


class StageTimings:
    """Accumulates per-stage elapsed time for one request and reports budget breaches.

    One instance per request; not thread-safe and does not need to be, since a request is
    handled by one task.
    """

    def __init__(self, elapsed: Callable[[], float]) -> None:
        self._elapsed = elapsed
        self._ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time a stage, recording it even if the body raises.

        Recording on the failure path is the point: the fail-safe branch must still show
        how long the model took to not answer, otherwise a slow-failing dependency looks
        free in the histogram.
        """
        started = self._elapsed()
        try:
            yield
        finally:
            self.record(name, (self._elapsed() - started) * 1000.0)

    def record(self, name: str, milliseconds: float) -> None:
        """Record a stage duration, emit it, and count it if it broke its allocation."""
        self._ms[name] = milliseconds
        STAGE_SECONDS.labels(stage=name).observe(milliseconds / 1000.0)
        budget = BUDGET_MS.get(name)
        if budget is not None and milliseconds > budget:
            BUDGET_BREACHES.labels(stage=name).inc()

    @property
    def milliseconds(self) -> dict[str, float]:
        """Stage name -> elapsed milliseconds, in the order the stages ran."""
        return dict(self._ms)

    @property
    def breached(self) -> bool:
        """Whether any recorded stage, including the total, exceeded its allocation.

        Exposed on the response as well as counted, so a caller correlating a slow
        checkout with a specific decision does not have to join against our metrics.
        """
        return any(ms > BUDGET_MS[name] for name, ms in self._ms.items() if name in BUDGET_MS)
