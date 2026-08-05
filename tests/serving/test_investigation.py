"""The case view's arithmetic, and the two refusals it is built around.

Split from `test_cases.py` deliberately: this file exercises `build_case` directly, with
no HTTP and no app, because the counterfactual is arithmetic and asserting on it through a
JSON round-trip would make a wrong number look like a serialisation bug.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from fraudlens.economics import (
    ACTION_NAMES,
    false_negative_cost,
    false_positive_cost,
    relationship_cost,
)
from fraudlens.policy import boundaries, decide
from fraudlens.serving.investigation import (
    DOCUMENTED_FEATURES,
    LedgerDecision,
    build_case,
)

STAMP = dt.datetime(2026, 8, 5, 14, 32, 10, tzinfo=dt.timezone.utc)


def record(**overrides: object) -> LedgerDecision:
    """A scored `deny` on a $250 basket, 20-day-old account."""
    fields: dict[str, object] = {
        "transaction_id": 2987000,
        "transaction_at": STAMP,
        "decided_at": STAMP,
        "calibrated_probability": 0.90,
        "action": "deny",
        "reason_codes": ("SCORE_ABOVE_DENY_BOUNDARY",),
        "model_version": "champion-v7",
        "policy_version": "ev-argmax-3action-v1+rules-ladder-v1",
        "feature_version": "request-supplied-v1",
        "config_hash": "deadbeef",
        "input_hash": "cafebabe",
        "degraded": False,
        "degraded_reason": None,
    }
    fields.update(overrides)
    return LedgerDecision(**fields)  # type: ignore[arg-type]


def test_boundaries_are_this_transactions_own_not_a_global_threshold() -> None:
    case = build_case(record(), amount=250.0, days_since_first_seen=20.0)
    assert case.economics is not None
    expected = boundaries(false_negative_cost([250.0]), false_positive_cost([250.0], ["8-30d"]))
    assert case.economics.tenure_bucket == "8-30d"
    assert case.economics.allow_to_challenge == pytest.approx(expected["allow_to_challenge"][0])
    assert case.economics.challenge_to_deny == pytest.approx(expected["challenge_to_deny"][0])
    assert case.economics.allow_to_deny == pytest.approx(expected["allow_to_deny"][0])


def test_published_anchors_reproduce_from_the_same_arithmetic() -> None:
    """Findings §3 publishes the break-even by tenure at the bucket's median basket. The
    case view must land on those numbers or one of the two is wrong."""
    # (D1 in the bucket, bucket label, median M from findings §3, published boundary)
    anchors = [
        (3.0, "1-7d", 223.0, 0.740),
        (0.0, "new(0d)", 158.0, 0.642),
        (500.0, "400d+", 58.0, 0.379),
    ]
    for d1, tenure, median_m, published in anchors:
        # Invert M = amount * margin + relationship_cost(tenure) to recover the basket the
        # published median corresponds to, then let the case view price it forward.
        amount = (median_m - float(relationship_cost([tenure])[0])) / 0.30
        case = build_case(record(), amount=amount, days_since_first_seen=d1)
        assert case.economics is not None
        assert case.economics.tenure_bucket == tenure
        assert case.economics.allow_to_deny == pytest.approx(published, abs=0.002)


def test_amount_flip_is_exact_and_the_policy_agrees_across_it() -> None:
    """The closed form is only worth having if it is exact. Assert the solved amount puts
    the boundary *at* the recorded probability, and that `policy.decide` flips there."""
    case = build_case(record(), amount=250.0, days_since_first_seen=20.0)
    assert case.amount_counterfactual is not None
    flips = case.amount_counterfactual.flips
    assert [f.boundary for f in flips] == ["challenge_to_deny"]
    flip = flips[0]
    at_flip = boundaries(
        false_negative_cost([flip.amount]), false_positive_cost([flip.amount], ["8-30d"])
    )["challenge_to_deny"][0]
    assert at_flip == pytest.approx(0.90, abs=1e-9)
    assert (flip.action_below, flip.action_at_or_above) == ("challenge", "deny")
    for probe, expected in ((flip.amount * 0.99, "challenge"), (flip.amount * 1.01, "deny")):
        index = int(
            decide(
                0.90,
                false_negative_cost([probe]),
                false_positive_cost([probe], ["8-30d"]),
                include_review=False,
            )[0]
        )
        assert ACTION_NAMES[index] == expected


def test_the_binary_break_even_solves_cleanly_and_is_still_not_reported() -> None:
    """`allow_to_deny` is the number findings §3 publishes, and under the three-action
    policy it is not operative: a step-up stops 89% of fraud, so the real deny boundary
    sits far above it. The solve succeeds and the oracle must drop it -- this is the
    reason the oracle exists at all."""
    case = build_case(record(calibrated_probability=0.55), amount=250.0, days_since_first_seen=20.0)
    assert case.amount_counterfactual is not None
    assert case.amount_counterfactual.flips == []
    assert case.amount_counterfactual.observed_action == "challenge"
    assert "no basket value changes the outcome" in case.amount_counterfactual.statement


def test_tenure_counterfactual_enumerates_every_bucket_including_unknown() -> None:
    case = build_case(record(), amount=250.0, days_since_first_seen=20.0)
    assert case.tenure_counterfactual is not None
    buckets = [o.tenure_bucket for o in case.tenure_counterfactual.by_tenure]
    assert buckets[-1] == "unknown"
    assert len(buckets) == 8
    assert sum(o.observed for o in case.tenure_counterfactual.by_tenure) == 1
    # The boundary falls as tenure lengthens, so the action can only get more severe.
    denies = [o.tenure_bucket for o in case.tenure_counterfactual.by_tenure if o.action == "deny"]
    assert "400d+" in denies


def test_a_degraded_decision_gets_no_counterfactual_and_says_why() -> None:
    """ADR-0003. There is no probability, so there is no question to answer -- and
    substituting one would launder a guess through the cost model."""
    case = build_case(
        record(
            calibrated_probability=None,
            action="challenge",
            degraded=True,
            degraded_reason="DEGRADED_MODEL_UNAVAILABLE",
        ),
        amount=250.0,
        days_since_first_seen=20.0,
    )
    assert case.amount_counterfactual is None
    assert case.tenure_counterfactual is None
    assert case.economics is not None, "boundaries need no probability; they stay computable"
    assert case.economics.recomputed_action is None
    assert case.economics.matches_ledger_action is None
    assert any(a.field == "counterfactuals" and "degraded" in a.reason for a in case.unavailable)


def test_missing_amount_leaves_economics_absent_rather_than_assumed() -> None:
    case = build_case(record(), amount=None, days_since_first_seen=20.0)
    assert case.economics is None
    assert case.amount_counterfactual is None
    fields = {a.field for a in case.unavailable}
    assert {"economics", "counterfactuals"} <= fields


def test_missing_tenure_is_priced_as_unknown_not_as_a_new_account() -> None:
    case = build_case(record(), amount=250.0, days_since_first_seen=None)
    assert case.economics is not None
    assert case.economics.tenure_bucket == "unknown"


def test_attribution_refuses_the_anonymized_features_and_returns_null_not_zero() -> None:
    case = build_case(record(), amount=250.0, days_since_first_seen=20.0)
    assert case.attribution.contributions is None
    assert case.attribution.unavailable_reason is not None
    assert case.attribution.documented_features == list(DOCUMENTED_FEATURES)
    refusal = case.attribution.refusal
    assert "V1-V339" in refusal and "C1-C14" in refusal
    assert any(a.field == "attribution.contributions" for a in case.unavailable)


def test_a_stale_config_hash_is_reported_rather_than_silently_recomputed() -> None:
    """The recomputed action is a counterfactual under *today's* constants whenever the
    hashes differ, and answering a dispute with it unqualified would be wrong."""
    case = build_case(record(config_hash="not-the-current-one"), amount=250.0,
                      days_since_first_seen=20.0)  # fmt: skip
    assert case.economics is not None
    assert case.economics.config_hash_matches_current is False
    assert case.economics.matches_ledger_action is True


def test_the_decision_path_does_not_import_the_case_view() -> None:
    """§4.3 budgets the decision, not this. The separation is only real if the hot path
    cannot reach it, so assert on the imports rather than on a comment."""
    serving = Path(__file__).resolve().parents[2] / "src" / "fraudlens" / "serving"
    for module in ("decisioning.py", "contracts.py", "audit.py", "reasons.py"):
        source = (serving / module).read_text()
        assert "investigation" not in source, module
        assert "case_contracts" not in source, module
