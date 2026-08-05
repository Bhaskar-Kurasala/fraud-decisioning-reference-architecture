"""When to retrain: four Tier 0 signals, none of which needs a label.

**Why performance is not in this list.** The obvious trigger is "AUC or realised cost
regressed". It is unusable at this label latency. Findings §6 measured the observed fraud
rate collapsing to 6.9% of the true rate over the final 20 days of a window, because
chargebacks land on a 34-day median with a ~97-day tail. A performance number computed on a
recent window is therefore not a noisy estimate of performance — it is a biased one, biased
toward "everything is fine", and the bias shrinks only as the window ages out of usefulness.
A trigger keyed on it fires roughly a quarter after the traffic that broke the model, which
is a quarter of losses collected before the retrain starts.

The four signals below are all computable on the day the traffic arrives:

1. **Score PSI** vs the training baseline — the earliest possible warning, and the one that
   moves for a genuinely new population rather than for a new *label* population.
2. **Feature drift breadth** — how many inputs moved, not just whether one did. A single
   drifting feature is usually an upstream release; several at once is the population.
3. **Action-mix shift** — the policy consumes a probability, so a calibration break shows
   up as the mix moving long before it shows up as a loss. This is the signal that would
   have caught E1's rebalanced model on day one: it challenges 52% of traffic against the
   champion's 6.8%, which is visible in the queue depth that afternoon.
4. **Staleness** — no signal at all, just age. Included because the other three are
   *detectors*, and a detector that has not fired is not evidence of health when the thing
   it detects can be gradual.

**Every signal reports whether it could be computed.** A signal whose input was missing is
`UNAVAILABLE`, never `QUIET`. ADR-0003: a drift job that stopped running would otherwise
render as "no drift", and the trigger would be quietest exactly when it was blindest.

**Every fired signal names what moved.** A trigger that says "retrain" without saying which
input shifted is one nobody trusts by the third false positive, and an untrusted trigger
gets muted rather than investigated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from fraudlens.monitoring.drift import PSI_ALERT_THRESHOLD, DriftResult


class SignalStatus(str, Enum):
    FIRED = "fired"
    QUIET = "quiet"
    # The input was not supplied. Distinct from QUIET on purpose — see module docstring.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class Signal:
    """One Tier 0 indicator, its reading, and what a human should do about it."""

    name: str
    status: SignalStatus
    # None when UNAVAILABLE. Not 0.0: a fabricated reading of zero is exactly the
    # "quietest when blindest" failure ADR-0003 is about.
    value: float | None
    threshold: float
    explanation: str


@dataclass(frozen=True, slots=True)
class TriggerThresholds:
    """Business decisions about when a retrain is worth its revalidation cost."""

    # §5.2's alert level, reused rather than re-derived: on-call already has intuition
    # for 0.25 and a second, differently-tuned PSI threshold in the same system is how
    # two dashboards start disagreeing about whether drift is happening.
    score_psi: float = PSI_ALERT_THRESHOLD

    # One drifting feature is usually an upstream schema change and is fixed upstream.
    # Three simultaneously is the population, which is the thing a retrain addresses.
    min_drifting_features: int = 3

    # Total-variation distance over the four-action mix. The champion challenges 6.8% of
    # traffic; a 5-point TV move is most of that arm reallocated, which is already
    # visible in analyst queue depth and approved GMV. Below that, day-to-day mix noise
    # dominates and the trigger would fire on Tuesdays.
    action_mix_shift: float = 0.05

    # The dispute window closes at 90 days. Past that the model has never seen a full
    # quarter of labels that now exist, which is a retrain worth doing on its own terms
    # even when nothing has drifted.
    max_model_age_days: int = 90


@dataclass(frozen=True, slots=True)
class TriggerObservation:
    """Everything the trigger reads. Optional fields mean "not measured", not "zero"."""

    baseline_action_mix: Mapping[str, float]
    score_psi: float | None = None
    feature_drift: Sequence[DriftResult] | None = None
    action_mix: Mapping[str, float] | None = None
    days_since_retrain: int | None = None


@dataclass(frozen=True, slots=True)
class TriggerVerdict:
    signals: tuple[Signal, ...]

    @property
    def fired(self) -> bool:
        return any(s.status is SignalStatus.FIRED for s in self.signals)

    @property
    def firing(self) -> tuple[Signal, ...]:
        return tuple(s for s in self.signals if s.status is SignalStatus.FIRED)

    @property
    def unavailable(self) -> tuple[Signal, ...]:
        return tuple(s for s in self.signals if s.status is SignalStatus.UNAVAILABLE)

    def explain(self) -> str:
        """The message an operator is paged with.

        Blind signals are listed even when something else fired, because "PSI is over
        threshold and by the way the feature drift job has not reported in two days"
        is a materially different situation from "PSI is over threshold".
        """
        head = "RETRAIN" if self.fired else "NO RETRAIN"
        lines = [f"{head} ({len(self.firing)} of {len(self.signals)} signals firing)"]
        lines.extend(f"  [{s.status.value}] {s.name}: {s.explanation}" for s in self.signals)
        return "\n".join(lines)


def _unavailable(name: str, threshold: float, why: str) -> Signal:
    return Signal(
        name=name,
        status=SignalStatus.UNAVAILABLE,
        value=None,
        threshold=threshold,
        explanation=f"not measured ({why}); this signal is blind, not quiet",
    )


def _score_drift(observation: TriggerObservation, thresholds: TriggerThresholds) -> Signal:
    psi = observation.score_psi
    if psi is None:
        return _unavailable("score_psi", thresholds.score_psi, "no scoring window supplied")
    fired = psi > thresholds.score_psi
    return Signal(
        name="score_psi",
        status=SignalStatus.FIRED if fired else SignalStatus.QUIET,
        value=psi,
        threshold=thresholds.score_psi,
        explanation=(
            f"score distribution PSI {psi:.3f} vs the training baseline "
            f"({'over' if fired else 'under'} {thresholds.score_psi:.2f}); "
            "the population being scored has moved, whatever the labels eventually say"
        ),
    )


def _feature_drift(observation: TriggerObservation, thresholds: TriggerThresholds) -> Signal:
    threshold = float(thresholds.min_drifting_features)
    results = observation.feature_drift
    if results is None:
        return _unavailable("feature_drift", threshold, "the per-feature drift job did not report")
    drifting = sorted((r for r in results if r.alerting), key=lambda r: r.psi, reverse=True)
    fired = len(drifting) >= thresholds.min_drifting_features
    named = ", ".join(f"{r.feature} (PSI {r.psi:.2f})" for r in drifting[:5])
    return Signal(
        name="feature_drift",
        status=SignalStatus.FIRED if fired else SignalStatus.QUIET,
        value=float(len(drifting)),
        threshold=threshold,
        explanation=(
            f"{len(drifting)} of {len(results)} features over PSI {PSI_ALERT_THRESHOLD:.2f}"
            + (f": {named}" if drifting else "")
        ),
    )


def _total_variation(actual: Mapping[str, float], baseline: Mapping[str, float]) -> float:
    """Half the L1 distance between two action mixes.

    TV rather than a per-action delta because the mix is a simplex: an arm cannot grow
    without another shrinking, so per-action alerts fire in pairs and double-count one
    event. TV is also directly readable — 0.05 is "5% of traffic changed hands".
    """
    keys = set(actual) | set(baseline)
    return 0.5 * sum(abs(actual.get(k, 0.0) - baseline.get(k, 0.0)) for k in keys)


def _action_mix(observation: TriggerObservation, thresholds: TriggerThresholds) -> Signal:
    actual = observation.action_mix
    if actual is None:
        return _unavailable(
            "action_mix", thresholds.action_mix_shift, "no decision window supplied"
        )
    shift = _total_variation(actual, observation.baseline_action_mix)
    fired = shift > thresholds.action_mix_shift
    moved = sorted(
        ((k, actual.get(k, 0.0) - observation.baseline_action_mix.get(k, 0.0)) for k in actual),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    detail = ", ".join(f"{name} {delta:+.1%}" for name, delta in moved[:3])
    return Signal(
        name="action_mix",
        status=SignalStatus.FIRED if fired else SignalStatus.QUIET,
        value=shift,
        threshold=thresholds.action_mix_shift,
        explanation=(
            f"{shift:.1%} of traffic changed action vs the baseline mix ({detail}); "
            "the policy consumes the probability, so a calibration break moves this "
            "months before it moves realised loss"
        ),
    )


def _staleness(observation: TriggerObservation, thresholds: TriggerThresholds) -> Signal:
    threshold = float(thresholds.max_model_age_days)
    age = observation.days_since_retrain
    if age is None:
        return _unavailable("staleness", threshold, "the registry did not report a training date")
    fired = age > thresholds.max_model_age_days
    return Signal(
        name="staleness",
        status=SignalStatus.FIRED if fired else SignalStatus.QUIET,
        value=float(age),
        threshold=threshold,
        explanation=(
            f"champion trained {age}d ago; the 90-day dispute window has closed on "
            f"{max(age - 90, 0)}d of traffic it has never been fitted to"
        ),
    )


def evaluate_trigger(
    observation: TriggerObservation,
    thresholds: TriggerThresholds | None = None,
) -> TriggerVerdict:
    """Read all four signals and return every one of them, fired or not.

    All four are always evaluated — there is no short-circuit on the first firing signal.
    The retrain is decided by one signal but the *diagnosis* needs the other three: score
    PSI over threshold with a flat action mix is a scoring-population change, and with the
    mix moving too it is a calibration break. Those have different fixes, and only one of
    them is a retrain.
    """
    thresholds = thresholds or TriggerThresholds()
    return TriggerVerdict(
        signals=(
            _score_drift(observation, thresholds),
            _feature_drift(observation, thresholds),
            _action_mix(observation, thresholds),
            _staleness(observation, thresholds),
        )
    )
