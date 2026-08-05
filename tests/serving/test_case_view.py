"""The HTML page: that it renders, that it refuses, and that it is not a queue.

Driven through `TestClient` in-process. No uvicorn, no browser and no selenium -- §9a
rules those out, and the page is server-rendered precisely so that a string assertion is
the whole test rather than a proxy for one.

`test_investigation.py` owns the arithmetic and `test_cases.py` owns the JSON contract.
What is asserted here is only what the *page* adds: that absence is visible, that the
refusal is present at full length, and that nothing on it implies pending work.
"""

from __future__ import annotations

import re
from html import escape

from fraudlens.serving.investigation import REFUSAL, LedgerDecision
from tests.serving.test_cases import DECISION, STAMP, client_with_ledger

DEGRADED = LedgerDecision(
    transaction_id=1,
    transaction_at=STAMP,
    decided_at=STAMP,
    calibrated_probability=None,
    action="challenge",
    reason_codes=("DEGRADED_MODEL_UNAVAILABLE", "FALLBACK_RULE_DEFAULT_CHALLENGE"),
    model_version="unavailable",
    policy_version="ev-argmax-3action-v1+rules-ladder-v1",
    feature_version="request-supplied-v1",
    config_hash="deadbeef",
    input_hash="cafebabe",
    degraded=True,
    degraded_reason="DEGRADED_MODEL_UNAVAILABLE",
)

PRICED = {"amount": 250.0, "days_since_first_seen": 20.0}


def test_the_page_renders_a_recorded_decision_with_its_economics() -> None:
    with client_with_ledger(DECISION) as client:
        response = client.get("/cases/2987000", params=PRICED)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    assert "Decision record for transaction 2987000" in page
    assert "0.9000" in page  # the recorded probability, not a rounded stand-in
    assert "SCORE_ABOVE_DENY_BOUNDARY" in page
    assert "champion-v7" in page  # the version set a replay needs
    assert "8-30d" in page
    # The score's position against the boundary that actually decided, spelled out. The
    # published binary break-even is on the page too, labelled as not operative, because
    # people ask about it -- but it must never be the number attached to the outcome.
    assert "at or above challenge/deny" in page
    assert "NOT operative" in page
    assert "would have been challenge rather than deny" in page  # the amount counterfactual
    assert "The same score and basket would have been" in page  # the tenure counterfactual


def test_a_degraded_decision_shows_no_probability_and_no_substitute_for_one() -> None:
    """The failure this guards is a template filling a null with 0.0000, which reads as
    'confidently legitimate' to anyone who did not check the degraded flag (ADR-0003)."""
    with client_with_ledger(DEGRADED) as client:
        page = client.get("/cases/1", params=PRICED).text
    assert "0.0000" not in page
    assert "no model scored this transaction" in page
    assert "no score to place against the boundaries" in page
    assert "no score to recompute from" in page
    # And the counterfactual is stated as absent rather than left as an empty block.
    assert "What would have changed the outcome" in page
    assert "a degraded decision has no score to vary against" in page


def test_absence_is_marked_rather_than_blank() -> None:
    """A missing number must look different from a zero, on a screen more than anywhere."""
    with client_with_ledger(DEGRADED) as client:
        page = client.get("/cases/1", params=PRICED).text
    assert page.count('class="absent"') >= 3
    # No cell is left empty: an empty cell is the blank that ADR-0003 says reads as calm.
    assert "<td></td>" not in page


def test_omitting_the_amount_says_so_rather_than_showing_an_empty_section() -> None:
    with client_with_ledger(DECISION) as client:
        page = client.get("/cases/2987000").text
    assert "no basket amount was supplied with this lookup" in page
    assert "will not price a dispute from a guess" in page
    # And the response's own `unavailable` list is rendered, not just the inline notice.
    assert "Not computed on this page, and why" in page
    assert "no `amount` supplied" in page


def test_the_refusal_is_present_in_full_and_is_not_small_print() -> None:
    with client_with_ledger(DECISION) as client:
        page = client.get("/cases/2987000", params=PRICED).text
    # Escaped, not raw: the refusal contains an apostrophe, and a page that interpolated
    # ledger-derived text unescaped would be an injection hole on the one screen a
    # compliance reviewer reads.
    assert escape(REFUSAL) in page
    assert "Per-feature contributions" in page
    assert "TransactionAmt" in page  # the documented features it *would* be reported over
    # Its own block at body size or larger. The rejected shape was a grey footnote, which
    # is how an empty 'top factors' table ends up filled in from intuition.
    assert '<div class="refusal">' in page
    assert ".refusal p { font-size: 1.05rem;" in page


def test_the_page_is_self_contained() -> None:
    """No CDN, no bundler, no framework: an air-gapped payments estate cannot fetch either,
    and a page that renders naked during an incident is worse than no page."""
    with client_with_ledger(DECISION) as client:
        page = client.get("/cases/2987000", params=PRICED).text
    assert "http://" not in page and "https://" not in page
    assert "<script" not in page
    assert "<link" not in page


def test_the_page_exposes_no_queue_shaped_route_or_control() -> None:
    """ADR-0004: value of review is positive for 5 of 92,427 transactions, so there is no
    work to distribute. A control on this page would assert a queue the measurement says
    should not exist -- and a queue built one convenience at a time is still a queue."""
    with client_with_ledger(DECISION) as client:
        page = client.get("/cases/2987000", params=PRICED).text
        paths = set(client.get("/openapi.json").json()["paths"])
    assert "/cases/{transaction_id}" in paths
    banned = ("queue", "assign", "claim", "worklist", "sla", "inbox", "approve", "reject")
    assert not [p for p in paths for word in banned if word in p.lower()]
    assert not re.findall("|".join(rf"\b{word}\w*" for word in banned), page, flags=re.I)
    # Nothing to submit, nothing to click, nowhere to go next.
    for control in ("<form", "<button", "<input", "<a ", "href="):
        assert control not in page.lower()


def test_an_unknown_transaction_is_a_page_not_a_json_error_body() -> None:
    with client_with_ledger(DECISION) as client:
        response = client.get("/cases/999")
    assert response.status_code == 404
    assert "No decision recorded for transaction 999" in response.text


def test_an_unconfigured_instance_says_so_rather_than_pretending_the_case_is_missing() -> None:
    """503, not 404 -- the same distinction the JSON route draws, for the same reason: a
    handler who reads 404 tells a customer we never decided their order."""
    from tests.serving.conftest import StubScorer, make_client

    with make_client(scorer=StubScorer(0.01)) as client:
        response = client.get("/cases/2987000")
    assert response.status_code == 503
    assert "No decision ledger is configured" in response.text
