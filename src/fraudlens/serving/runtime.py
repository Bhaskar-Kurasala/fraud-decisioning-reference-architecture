"""Process state and the four seams the service is wired through.

Everything that would otherwise make a test need a network, an MLflow server or a
`sleep` is a parameter here: the model loader, the wall clock, the elapsed-time source
and the audit writer. That is §9a's injected-clock/injected-loader rule, and it is also
what makes the fail-safe path testable at all -- you cannot assert that a service
degrades safely if the only way to break the model is to break a real one.

The scorer is loaded **once**, at startup, into `ServiceState`. Per-request loading would
spend the entire 25 ms inference budget on deserialisation and would let two concurrent
requests be decided by different artifacts, which is unauditable.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """The scoring artifact could not be loaded or is not usable.

    Raised by a loader, caught at startup. It is not fatal: the process stays up and
    serves the documented fallback with `/health/ready` failing, because an outage of the
    model is not an outage of fraud control (ADR-0002 priority #4).
    """


@runtime_checkable
class CalibratedScorer(Protocol):
    """The loaded artifact: the discriminator and the calibration fitted on top of it.

    Two methods rather than one because §4.3 budgets them separately (inference 25 ms,
    calibration + policy 15 ms) and because they are two objects in the artifact -- the
    gradient-boosted model and the isotonic regression. Collapsing them would make the
    stage that decays first invisible in the histogram.

    A Protocol rather than a base class: the production implementation is an sklearn
    pipeline owned by the models layer and the test implementation is a stub, so this is
    a port with two real implementations, not a speculative abstraction (§9a).
    """

    @property
    def version(self) -> str:
        """Identifier of this artifact. Goes in the ledger and the response."""

    def score(self, features: Mapping[str, float]) -> float:
        """Raw discriminator output. Monotone in fraud risk, not a probability."""

    def calibrate(self, raw_score: float) -> float:
        """Map the raw score to a calibrated P(fraud) in [0, 1]."""


class DecisionWriter(Protocol):
    """Port to the audit trail.

    Deliberately not `streaming.ledger.DecisionLedger`. Two reasons, both structural:
    the scoring container has no reason to carry SQLAlchemy and psycopg (pyproject splits
    the extras by deployment unit for exactly this), and the serving layer must be
    testable and deployable without a database. The adapter that turns these keyword
    arguments into a `DecisionRecord` is a field-for-field splat and belongs to whatever
    composes the two -- the names below match the ledger's columns 1:1 so that it stays
    a splat and never becomes a translation.

    Writes are best-effort from the caller's perspective: §4.3 budgets the ledger write
    at 0 ms because it is asynchronous, and a decision that was made but not yet persisted
    must still reach the customer.
    """

    def record_decision(
        self,
        *,
        transaction_id: int,
        transaction_at: dt.datetime,
        decided_at: dt.datetime,
        # None on the degraded path — there was no model, so there is no score.
        # A stand-in value here would be a fabricated observation.
        score: float | None,
        calibrated_probability: float | None,
        action: str,
        reason_codes: tuple[str, ...],
        model_version: str,
        policy_version: str,
        feature_version: str,
        config_hash: str,
        input_hash: str,
        degraded: bool,
        degraded_reason: str | None,
    ) -> None:
        """Append one fully-versioned decision to the audit trail."""


# Returns the artifact for a pinned version, or raises `ModelUnavailableError`. A callable
# rather than a class: it has no state and one method, and a class would be a wrapper
# layer that only forwards a call (§9a).
ModelLoader = Callable[[str | None], CalibratedScorer]

# Wall clock, for timestamps that go in the ledger. Must return tz-aware UTC.
Clock = Callable[[], dt.datetime]

# Monotonic elapsed-time source in seconds, for the latency budget. Separate from `Clock`
# because a wall clock can step backwards over an NTP correction and a negative stage
# duration would silently corrupt the SLO metric.
Elapsed = Callable[[], float]


def utc_now() -> dt.datetime:
    """Default clock. UTC and tz-aware; DTZ is on precisely to force this."""
    return dt.datetime.now(tz=dt.timezone.utc)


class ServiceState:
    """The scorer held for the process lifetime, plus why it is missing when it is.

    Mutable by design and mutated in exactly one place (`load`, at startup). Readiness is
    derived from it rather than tracked separately, so the flag and the reality cannot
    disagree.
    """

    def __init__(self, loader: ModelLoader, *, model_version_pin: str | None = None) -> None:
        self._loader = loader
        # The pin is what makes a decision reproducible: "latest" would mean a replay of
        # last Tuesday's traffic silently used this Tuesday's model.
        self._pin = model_version_pin
        self._scorer: CalibratedScorer | None = None
        self._failure: str | None = None

    def load(self) -> None:
        """Load the artifact once. Records the failure instead of raising.

        Raising here would take the process down, which converts a model problem into a
        checkout outage -- the opposite of the fail-safe posture. The service comes up
        not-ready and every decision it makes is marked degraded until a load succeeds.
        """
        try:
            self._scorer = self._loader(self._pin)
            self._failure = None
        except Exception as exc:
            self._scorer = None
            self._failure = f"{type(exc).__name__}: {exc}"
            logger.error("model load failed, serving degraded: %s", self._failure)

    @property
    def scorer(self) -> CalibratedScorer | None:
        """The loaded artifact, or None if it never loaded."""
        return self._scorer

    @property
    def ready(self) -> bool:
        """Whether the model is loaded and the service can serve a scored decision.

        Distinct from liveness on purpose. A not-ready instance is still answering, still
        fail-safe, and must not be restarted -- restarting it does not produce a model.
        """
        return self._scorer is not None

    @property
    def failure(self) -> str | None:
        """The load failure, for `/health/ready` to report. None when ready."""
        return self._failure

    @property
    def model_version(self) -> str:
        """Version of the loaded artifact, or the pin, or an explicit unknown marker.

        Never empty: an unversioned decision cannot be replayed, and the degraded path
        still produces decisions that a dispute can be raised against.
        """
        if self._scorer is not None:
            return self._scorer.version
        return self._pin or "unavailable"
