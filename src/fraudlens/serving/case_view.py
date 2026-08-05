"""One HTML page for one recorded decision. A lookup, not a work queue.

**Server-rendered, not a static page that fetches the JSON.** The rejected alternative was
a static `case.html` calling `GET /v1/cases/{id}` from the browser, which is the more
fashionable shape and is wrong here twice. It would reimplement ADR-0003 in JavaScript --
every `null` in that response has to become a visible absence rather than an empty cell,
and a second implementation of that rule is a second place for it to rot. And it could
only be tested through a browser, which §9a rules out; rendered here, the page is a string
a `TestClient` can assert on in-process. The cost is that this module knows about HTML,
which is the smaller price.

**No build step, no framework, no CDN.** Inline CSS, no JavaScript at all. A payments
estate is air-gapped often enough that a page which needs to reach a CDN is a page that
renders naked during an incident, and a dispute desk should not need a bundler to read one
decision.

**Off the §4.3 path**, exactly like the JSON endpoint it renders: it reads the ledger, it
never blocks a checkout, it has no latency budget.

**This is not a review queue** (ADR-0004). There is no list, no next-case link, no
assignment, no claiming, no timer, and no control that decides anything -- the decision on
this page was already taken by the policy. Every one of those would visually assert a queue
that findings §5 measured should not exist (value of review positive for 5 of 92,427).
`tests/serving/test_case_view.py` asserts the page contains no form, no button and no
queue-shaped vocabulary.
"""

from __future__ import annotations

from html import escape
from typing import Annotated

from fastapi import Depends, FastAPI, Path, Query
from fastapi.responses import HTMLResponse

from fraudlens.serving.case_contracts import CaseResponse, EconomicContext
from fraudlens.serving.cases import CaseSource, case_source
from fraudlens.serving.investigation import build_case

_STYLE = """
body { font: 15px/1.55 system-ui, sans-serif; margin: 0 auto; max-width: 54rem;
       padding: 2rem 1rem; color: #111; background: #fff; }
h1 { font-size: 1.35rem; margin: 0 0 .2rem; }
h2 { font-size: 1rem; letter-spacing: .06em; text-transform: uppercase;
     margin: 2rem 0 .5rem; }
p.sub { color: #555; margin: 0 0 1.5rem; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid #ddd; padding: .35rem .5rem; text-align: left;
         vertical-align: top; }
th { width: 17rem; font-weight: 600; }
.action { font-size: 1.25rem; font-weight: 700; }
.absent { background: #fdf3d0; border: 1px dashed #8a6d00; border-radius: 3px;
          color: #6b4e00; padding: 0 .3rem; font-style: italic; }
.refusal { border: 2px solid #8a1c1c; border-radius: 4px; padding: 1rem 1.2rem;
           margin-top: 2rem; }
.refusal h2 { margin-top: 0; color: #8a1c1c; }
.refusal p { font-size: 1.05rem; margin: .6rem 0 0; }
.note { color: #555; font-size: .9rem; }
"""

# The one-line framing at the top of the page. It says what the page is *for* because a
# screen showing a fraud decision is read by default as something awaiting a human.
_SUBTITLE = (
    "A read-only record of one decision that has already been taken and cannot be changed "
    "here. Nothing on this page is pending, and there is no action for you to take on it."
)


def _absent(text: str) -> str:
    """A missing value must not look like a zero -- ADR-0003, sharpened for a screen.

    A blank cell reads as 'nothing here'; this reads as 'not computed, and here is why'.
    """
    return f'<span class="absent">{escape(text)}</span>'


def _rows(pairs: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><th>{escape(k)}</th><td>{v}</td></tr>" for k, v in pairs)
    return f"<table>{body}</table>"


def _yes_no(value: bool | None) -> str:
    if value is None:
        return _absent("not computed")
    return "yes" if value else "no"


def _decision_section(case: CaseResponse) -> str:
    facts = case.decision
    probability = (
        f"{facts.calibrated_probability:.4f}"
        if facts.calibrated_probability is not None
        else _absent("no model scored this transaction -- it was decided by the rule ladder")
    )
    versions = facts.versions
    return "<h2>The decision, as recorded</h2>" + _rows(
        [
            ("Action", f'<span class="action">{escape(facts.action)}</span>'),
            ("Calibrated P(fraud)", probability),
            ("Reason codes", escape(", ".join(facts.reason_codes))),
            ("Degraded", _yes_no(facts.degraded)),
            ("Degraded reason", escape(facts.degraded_reason or "") or _absent("not degraded")),
            ("Transacted at", escape(facts.transaction_at.isoformat())),
            ("Decided at", escape(facts.decided_at.isoformat())),
            ("Model version", escape(versions.model_version)),
            ("Policy version", escape(versions.policy_version)),
            ("Feature version", escape(versions.feature_version)),
            ("Config hash", escape(versions.config_hash)),
            (
                "Input hash",
                escape(versions.input_hash)
                + '<div class="note">A digest of the request, not a copy of it. This proves '
                "which input was used; the feature values are deliberately not stored.</div>",
            ),
        ]
    )


def _position(probability: float | None, economics: EconomicContext) -> str:
    """Where the score sits against the two boundaries that actually decided."""
    if probability is None:
        return _absent("no score to place against the boundaries")
    if probability < economics.allow_to_challenge:
        return f"below allow/challenge ({economics.allow_to_challenge:.4f}) &rarr; allow"
    if probability < economics.challenge_to_deny:
        return (
            f"at or above allow/challenge ({economics.allow_to_challenge:.4f}) and below "
            f"challenge/deny ({economics.challenge_to_deny:.4f}) &rarr; challenge"
        )
    return f"at or above challenge/deny ({economics.challenge_to_deny:.4f}) &rarr; deny"


def _economics_section(case: CaseResponse) -> str:
    economics = case.economics
    if economics is None:
        return "<h2>The economics of this transaction</h2>" + _absent(
            "not computed: no basket amount was supplied with this lookup, and the ledger "
            "does not store one. Add ?amount=... (and days_since_first_seen if you know it) "
            "from the order record. This service will not price a dispute from a guess."
        )
    return "<h2>The economics of this transaction</h2>" + _rows(
        [
            ("Basket amount", f"${economics.amount:,.2f}"),
            ("Tenure bucket", escape(economics.tenure_bucket)),
            ("Cost of a missed fraud (L)", f"${economics.false_negative_cost:,.2f}"),
            ("Cost of a false decline (M)", f"${economics.false_positive_cost:,.2f}"),
            ("allow &rarr; challenge (operative)", f"{economics.allow_to_challenge:.4f}"),
            ("challenge &rarr; deny (operative)", f"{economics.challenge_to_deny:.4f}"),
            (
                "allow &rarr; deny (NOT operative)",
                f"{economics.allow_to_deny:.4f}"
                + '<div class="note">The binary break-even M/(L+M) that every published '
                "description of this policy quotes. Under the three-action policy nothing "
                "crosses it. Do not read it as the decline threshold.</div>",
            ),
            ("Where this score sits", _position(case.decision.calibrated_probability, economics)),
            (
                "Recomputed under today's constants",
                escape(economics.recomputed_action or "") or _absent("no score to recompute from"),
            ),
            ("Matches the recorded action", _yes_no(economics.matches_ledger_action)),
            ("Priced by the constants running now", _yes_no(economics.config_hash_matches_current)),
        ]
    )


def _counterfactual_section(case: CaseResponse) -> str:
    amount_cf, tenure_cf = case.amount_counterfactual, case.tenure_counterfactual
    if amount_cf is None or tenure_cf is None:
        return "<h2>What would have changed the outcome</h2>" + _absent(
            "not computed. See the list at the foot of this page for which input was "
            "missing; a degraded decision has no score to vary against, which is a "
            "different situation from a lookup that omitted the amount."
        )
    flips = (
        "".join(
            f"<tr><th>${flip.amount:,.2f} ({escape(flip.boundary)})</th><td>below: "
            f"{escape(flip.action_below)}; at or above: {escape(flip.action_at_or_above)}</td></tr>"
            for flip in amount_cf.flips
        )
        or "<tr><th>No flip amount exists</th><td>no positive basket value changes this.</td></tr>"
    )
    buckets = "".join(
        f"<tr><th>{escape(outcome.tenure_bucket)}"
        f"{' (observed)' if outcome.observed else ''}</th>"
        f"<td>{escape(outcome.action)}</td></tr>"
        for outcome in tenure_cf.by_tenure
    )
    return (
        "<h2>What would have changed the outcome</h2>"
        f"<p>{escape(amount_cf.statement)}</p>"
        f"<table>{flips}</table>"
        f"<p>{escape(tenure_cf.statement)}</p>"
        f"<table>{buckets}</table>"
    )


def _attribution_section(case: CaseResponse) -> str:
    """The refusal, at body size and in its own bordered block.

    Deliberately not a footnote. A fabricated per-feature explanation on an adverse-action
    notice is a compliance problem, and the rejected alternative -- a small grey caveat
    under an empty 'top factors' table -- is how that gets written anyway: the reader sees
    an empty table and fills it in from intuition.
    """
    attribution = case.attribution
    return (
        '<div class="refusal"><h2>Why this score is not explained here</h2>'
        f"<p>{escape(attribution.refusal)}</p>"
        f"<p>Per-feature contributions: {_absent('not computed')}<br>"
        f'<span class="note">{escape(attribution.unavailable_reason or "")}</span></p>'
        "<p>If an attribution ever becomes computable it is reported over these features "
        f"and no others: <b>{escape(', '.join(attribution.documented_features))}</b>.</p></div>"
    )


def _unavailable_section(case: CaseResponse) -> str:
    rows = "".join(
        f"<tr><th>{escape(item.field)}</th><td>{escape(item.reason)}</td></tr>"
        for item in case.unavailable
    )
    return f"<h2>Not computed on this page, and why</h2><table>{rows}</table>"


def render_case(case: CaseResponse) -> str:
    """The whole page. One decision, and an explicit account of what is missing from it."""
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Decision record {case.transaction_id}</title><style>{_STYLE}</style></head>"
        f"<body><h1>Decision record for transaction {case.transaction_id}</h1>"
        f'<p class="sub">{escape(_SUBTITLE)}</p>'
        + _decision_section(case)
        + _economics_section(case)
        + _counterfactual_section(case)
        + _attribution_section(case)
        + _unavailable_section(case)
        + "</body></html>"
    )


def _notice(title: str, body: str) -> str:
    return (
        f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>{escape(title)}"
        f"</title><style>{_STYLE}</style></head><body><h1>{escape(title)}</h1>"
        f"<p>{escape(body)}</p></body></html>"
    )


def register_case_view(app: FastAPI) -> None:
    """Mount `GET /cases/{transaction_id}` -- the HTML view of `/v1/cases/{transaction_id}`.

    Shares `case_source` with the JSON route so one dependency override wires both, and
    calls `build_case` directly rather than making an HTTP request to our own API: an
    in-process call cannot half-fail, and the alternative would give this page a network
    dependency on the pod it is already running in.
    """

    @app.get(
        "/cases/{transaction_id}",
        response_class=HTMLResponse,
        summary="The investigation view of one recorded decision, as a page",
        description=(
            "The same content as `GET /v1/cases/{transaction_id}`, rendered for a human. "
            "One transaction, looked up by id. There is no list, no search and no work "
            "assignment here, because there is no queue (ADR-0004)."
        ),
    )
    def case_page(
        transaction_id: Annotated[int, Path(ge=0, description="The ledger key.")],
        amount: Annotated[float | None, Query(gt=0.0)] = None,
        days_since_first_seen: Annotated[float | None, Query(ge=0.0)] = None,
        source: Annotated[CaseSource | None, Depends(case_source)] = None,
    ) -> HTMLResponse:
        # 503 and 404 are rendered as pages, not as JSON error bodies: whoever reaches this
        # URL is reading a screen, and "we have no record of this transaction" is an answer
        # they may have to repeat to a customer. Same distinction the JSON route draws.
        if source is None:
            return HTMLResponse(
                _notice(
                    "No decision ledger is configured",
                    "This instance has no ledger reader wired in, so no case can be read. "
                    "That is a deployment configuration, not a missing transaction.",
                ),
                status_code=503,
            )
        record = source(transaction_id)
        if record is None:
            return HTMLResponse(
                _notice(
                    f"No decision recorded for transaction {transaction_id}",
                    "We hold no decision for that id. That can mean we never decided it, or "
                    "that the decision was made and the ledger write failed -- the two are "
                    "not distinguishable from here.",
                ),
                status_code=404,
            )
        case = build_case(record, amount=amount, days_since_first_seen=days_since_first_seen)
        return HTMLResponse(render_case(case))
