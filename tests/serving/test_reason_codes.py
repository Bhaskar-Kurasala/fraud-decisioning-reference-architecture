"""Reason codes: always present, always machine identifiers, always consistent with the action.

An adverse decision that cannot be explained is not defensible to the customer who
disputes it or the regulator who audits it (ADR-0002 priority #2).
"""

from __future__ import annotations

import pytest

from fraudlens.serving.reasons import ReasonCode
from tests.serving.conftest import StubScorer, make_client, request_body

_TAXONOMY = {code.value for code in ReasonCode}


@pytest.mark.parametrize("probability", [0.0, 0.001, 0.2, 0.37, 0.5, 0.74, 0.99, 1.0])
@pytest.mark.parametrize("amount", [5.0, 100.0, 999.0])
@pytest.mark.parametrize("tenure", [None, 0.0, 45.0, 900.0])
def test_every_decision_carries_at_least_one_reason_code(
    probability: float, amount: float, tenure: float | None
) -> None:
    """Including approvals. An unexplained approve is as unauditable as an unexplained decline."""
    with make_client(scorer=StubScorer(probability)) as client:
        payload = client.post(
            "/v1/decide", json=request_body(amount=amount, days_since_first_seen=tenure)
        ).json()
    assert len(payload["reason_codes"]) >= 1
    assert set(payload["reason_codes"]) <= _TAXONOMY


def test_reason_codes_are_identifiers_not_display_text() -> None:
    """SCREAMING_SNAKE, ASCII, no spaces. i18n belongs to whatever renders them (ADR-0002)."""
    for code in ReasonCode:
        assert code.value == code.name
        assert code.value.isascii()
        assert code.value.replace("_", "").isalnum()
        assert code.value.upper() == code.value


def test_a_decline_cites_the_band_that_produced_it() -> None:
    """The code must agree with the action; if they disagree the code is the bug."""
    # Certain fraud on a large basket: the EV argmax denies at any plausible boundary.
    with make_client(scorer=StubScorer(1.0)) as client:
        payload = client.post("/v1/decide", json=request_body(amount=800.0)).json()
    assert payload["action"] == "deny"
    assert ReasonCode.SCORE_ABOVE_DENY_BOUNDARY.value in payload["reason_codes"]
    assert ReasonCode.HIGH_AMOUNT_BAND.value in payload["reason_codes"]


def test_an_approve_says_why_it_was_below_every_boundary() -> None:
    with make_client(scorer=StubScorer(0.0)) as client:
        payload = client.post("/v1/decide", json=request_body()).json()
    assert payload["action"] == "allow"
    assert ReasonCode.SCORE_BELOW_ALL_BOUNDARIES.value in payload["reason_codes"]


def test_an_unidentified_customer_is_told_they_were_priced_as_average() -> None:
    """The 41-of-92,427 unknown-tenure path is a business assumption applied to a person,
    so it is surfaced rather than buried in the cost function."""
    with make_client(scorer=StubScorer(0.5)) as client:
        payload = client.post("/v1/decide", json=request_body(days_since_first_seen=None)).json()
    assert ReasonCode.UNKNOWN_TENURE_PRICED_AS_MEDIAN.value in payload["reason_codes"]


def test_degraded_decisions_name_both_the_cause_and_the_rule() -> None:
    """ "Why was I declined" must be answerable during an outage too."""
    with make_client(scorer=None) as client:
        payload = client.post(
            "/v1/decide", json=request_body(amount=900.0, days_since_first_seen=1.0)
        ).json()
    assert payload["reason_codes"] == [
        ReasonCode.DEGRADED_MODEL_UNAVAILABLE.value,
        ReasonCode.FALLBACK_RULE_HIGH_AMOUNT_NEW_ACCOUNT.value,
    ]
