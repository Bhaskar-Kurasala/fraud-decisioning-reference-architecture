"""Claims about the retrain trigger.

The load-bearing one is the last: a signal whose input never arrived must not read as a
signal that did not fire.
"""

from __future__ import annotations

from fraudlens.flywheel.trigger import (
    SignalStatus,
    TriggerObservation,
    TriggerThresholds,
    evaluate_trigger,
)
from fraudlens.monitoring.drift import DriftResult

BASELINE_MIX = {"allow": 0.928, "challenge": 0.068, "review": 0.0005, "deny": 0.0035}


def _drift(feature: str, psi: float) -> DriftResult:
    return DriftResult(
        feature=feature,
        n=10_000,
        psi=psi,
        ks=0.02,
        null_rate=0.0,
        baseline_null_rate=0.0,
        unseen_level_rate=0.0,
    )


def _quiet() -> TriggerObservation:
    return TriggerObservation(
        baseline_action_mix=BASELINE_MIX,
        score_psi=0.03,
        feature_drift=[_drift("C1", 0.01), _drift("D15", 0.04)],
        action_mix=dict(BASELINE_MIX),
        days_since_retrain=14,
    )


def test_a_healthy_window_does_not_fire() -> None:
    verdict = evaluate_trigger(_quiet())
    assert not verdict.fired
    assert not verdict.unavailable
    assert all(s.status is SignalStatus.QUIET for s in verdict.signals)


def test_score_drift_fires_and_names_itself() -> None:
    observation = TriggerObservation(**{**vars_of(_quiet()), "score_psi": 0.41})
    verdict = evaluate_trigger(observation)
    assert verdict.fired
    assert [s.name for s in verdict.firing] == ["score_psi"]
    assert "0.410" in verdict.firing[0].explanation


def test_feature_drift_needs_breadth_not_a_single_feature() -> None:
    """One drifting feature is an upstream release; three is the population."""
    one = TriggerObservation(
        **{**vars_of(_quiet()), "feature_drift": [_drift("card1", 0.9), _drift("C1", 0.01)]}
    )
    assert not evaluate_trigger(one).fired

    three = TriggerObservation(
        **{
            **vars_of(_quiet()),
            "feature_drift": [_drift("card1", 0.9), _drift("addr1", 0.4), _drift("C13", 0.3)],
        }
    )
    verdict = evaluate_trigger(three)
    assert verdict.fired
    # Actionability: the operator is told which features moved, worst first.
    explanation = verdict.firing[0].explanation
    assert explanation.index("card1") < explanation.index("addr1") < explanation.index("C13")


def test_the_rebalanced_challenger_is_caught_by_action_mix_on_day_one() -> None:
    """E1's model challenges 52% of traffic against the champion's 6.8%.

    That is a Tier 0 observation — no label required — which is the whole argument for
    triggering on mix rather than waiting a quarter for the loss to land.
    """
    observation = TriggerObservation(
        **{
            **vars_of(_quiet()),
            "action_mix": {"allow": 0.465, "challenge": 0.522, "review": 0.0, "deny": 0.0127},
        }
    )
    verdict = evaluate_trigger(observation)
    fired = {s.name for s in verdict.firing}
    assert "action_mix" in fired
    mix = next(s for s in verdict.signals if s.name == "action_mix")
    assert mix.value is not None and mix.value > 0.45


def test_staleness_fires_with_no_drift_at_all() -> None:
    observation = TriggerObservation(**{**vars_of(_quiet()), "days_since_retrain": 200})
    verdict = evaluate_trigger(observation)
    assert [s.name for s in verdict.firing] == ["staleness"]
    assert "110d" in verdict.firing[0].explanation


def test_a_missing_input_is_unavailable_and_never_quiet() -> None:
    """ADR-0003. A drift job that stopped reporting must not render as "no drift"."""
    observation = TriggerObservation(baseline_action_mix=BASELINE_MIX)
    verdict = evaluate_trigger(observation)
    assert not verdict.fired
    assert len(verdict.unavailable) == 4
    assert all(s.value is None for s in verdict.unavailable)
    assert all(s.status is not SignalStatus.QUIET for s in verdict.signals)
    # And the operator is told, rather than being shown four silent green signals.
    assert verdict.explain().count("[unavailable]") == 4


def test_blindness_is_reported_alongside_a_firing_signal() -> None:
    """ "PSI is high" and "PSI is high and we are blind elsewhere" need different responses."""
    observation = TriggerObservation(
        baseline_action_mix=BASELINE_MIX, score_psi=0.4, days_since_retrain=3
    )
    verdict = evaluate_trigger(observation)
    assert verdict.fired
    assert {s.name for s in verdict.unavailable} == {"feature_drift", "action_mix"}
    assert "[unavailable] feature_drift" in verdict.explain()


def test_thresholds_are_the_business_numbers_the_spec_names() -> None:
    thresholds = TriggerThresholds()
    assert thresholds.score_psi == 0.25  # §5.2 alert level
    assert thresholds.max_model_age_days == 90  # dispute-window close


def vars_of(observation: TriggerObservation) -> dict[str, object]:
    return {
        "baseline_action_mix": observation.baseline_action_mix,
        "score_psi": observation.score_psi,
        "feature_drift": observation.feature_drift,
        "action_mix": observation.action_mix,
        "days_since_retrain": observation.days_since_retrain,
    }
