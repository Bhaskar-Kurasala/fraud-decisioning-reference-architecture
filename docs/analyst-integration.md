# Consuming the investigation API

`GET /v1/cases/{transaction_id}` is the read side of the decision ledger. It exists to
answer, about one transaction, the four questions this system is actually asked:

- **A customer disputes a decline.** What did we decide, on what evidence, under which
  model and constants, and what would have had to be different for the answer to change?
- **An adverse-action response is due.** What can we say honestly, and what must we not
  say? (§ *The adverse-action path*.)
- **On-call is working an incident** from `docs/runbooks/`. Was this decision degraded,
  and does it still reproduce under the constants running now?
- **Someone is signing off a challenger promotion** and wants to look at the decisions
  that moved, one at a time, rather than at an aggregate.

`OPS_DISPUTE` is a term in our own cost function. These people exist; the endpoint is
built for them.

---

## 1. What this API deliberately does not do

**It is not a review queue.** There is no assignment, no claiming, no SLA timer, no
next-case endpoint, no work distribution. That is measured, not an omission:

> `fraud-decisioning-findings.md` §5 — value-of-review is positive for **5 of 92,427**
> transactions at $7.97/case. All four queue-ranking rules lose money (−$14,783 to
> −$28,116/yr realised), and ranking by uncertainty × exposure — the rule a case-management
> tool would suggest — is the *worst* of the four.

So this deployment runs the three-action policy (`serving.decisioning.INCLUDE_REVIEW =
False`) and there is no queue, because there is no work worth distributing. E6 refused to
emit queue-depth metrics for the same reason. If `f_pass` (the rate at which fraudsters
defeat the step-up challenge) is ever re-measured above ≈0.20, findings §5's sensitivity
grid flips this conclusion and a queue becomes a legitimate design question — *then*, with
the measurement in hand.

`tests/serving/test_cases.py::test_there_is_no_queue_surface` asserts the OpenAPI document
contains no route whose path mentions a queue, an assignment, a claim or an SLA. It is a
guard against this drifting back in.

**It is not on the checkout path.** §4.3 budgets `POST /v1/decide` at p99 ≤ 150 ms. It
budgets nothing here. `serving.investigation` must never be imported by the decision path
and a test asserts that it is not. Do not put a case lookup in a checkout flow.

**It does not show feature values.** The ledger stores `input_hash`, a digest of the
request, not a copy of it — `docs/lineage.md` §3 explains the trade: an append-only copy of
customer-linked signals has no deletion path, and building a store that makes erasure
structurally impossible is worse than declining to build the erasure workflow. The
consequence is exact and you have to design around it: the case view can prove *which*
input was used; it cannot show you that input.

---

## 2. Wiring it up

The route is registered by `app.create_app` but the ledger reader is not, because a
deployment that serves decisions without a database is real (the load test and the k8s
smoke both run one). Unconfigured, the endpoint answers **503 with a stated reason** —
never 404, because "this instance has no ledger" and "we have no record of that
transaction" are different answers and confusing them would have a dispute handler tell a
customer we never decided their order.

Provide the reader with FastAPI's dependency override:

```python
from fraudlens.serving.cases import case_source
from fraudlens.serving.case_contracts import LedgerDecision

def read_one(transaction_id: int) -> LedgerDecision | None:
    row = ledger.get(transaction_id)          # streaming.ledger.DecisionLedger
    return None if row is None else LedgerDecision(
        transaction_id=row.transaction_id,
        transaction_at=row.transaction_at,
        decided_at=row.decided_at,
        calibrated_probability=row.calibrated_probability,
        action=row.action,
        reason_codes=row.reason_codes,
        model_version=row.model_version,
        policy_version=row.policy_version,
        feature_version=row.feature_version,
        config_hash=row.config_hash,
        input_hash=row.input_hash,
        degraded=row.degraded,
        degraded_reason=row.degraded_reason,
    )

app.dependency_overrides[case_source] = lambda: read_one
```

The splat is written out rather than swallowed by `**kwargs` for the same reason
`composition.LedgerWriter` writes its arguments out: with `**kwargs` a renamed ledger
column type-checks forever and fails one row at a time in production.

### Query parameters, and why they are parameters

```
GET /v1/cases/2987000?amount=250.00&days_since_first_seen=20
```

`amount` and `days_since_first_seen` are supplied by *you*, from the order record, because
the ledger holds neither. Omit them and the `economics` and counterfactual blocks come back
`null` with an entry in `unavailable` saying which input was missing. The service will not
assume an amount: pricing a dispute from a guessed basket value is exactly the class of
defect ADR-0003 exists to stop.

Omitting `days_since_first_seen` is *not* the same as sending `0`. Zero is a real D1 — the
card's first ever transaction, the riskiest tenure bucket there is. Omitted means unknown,
which is priced as the median bucket and reported as `tenure_bucket: "unknown"`.

---

## 3. Reading the response

### The decision, verbatim

`decision` is read straight out of the ledger. `calibrated_probability` is `null` when
`degraded` is true, and that null is a database guarantee, not a convention: the
`ck_decision_score_presence` CHECK constraint pairs them. Do not substitute a zero
downstream — a 0.0 there reads as "confidently legitimate" to anyone who did not check the
flag.

`versions` carries the five identifiers a replay needs. `policy_version` names *both* the
EV variant and the fallback ladder, because a degraded decision was made by the ladder and
a replay has to know which arithmetic to re-run.

### The boundaries that applied to *this* transaction

There is no threshold in this system. `policy.boundaries` returns a different number for
every transaction, and `economics` is that transaction's:

| Field | Meaning |
|---|---|
| `allow_to_challenge` | Below this, allow. Operative. |
| `challenge_to_deny` | At or above this, deny. Operative. |
| `allow_to_deny` | The binary break-even M/(L+M). **Not operative here.** |

`allow_to_deny` is the number every published description of this policy quotes (findings
§3: 0.740 at 1–7d tenure, 0.369 at $500+ baskets), and under the three-action policy it
does not decide anything: a step-up challenge stops 89% of fraud for a 7% good-customer
abandonment cost, which pushes the real deny boundary far above it — typically 0.81–0.92
rather than 0.34–0.57. It is reported because you will be asked about it, and it is labelled
so nobody quotes it as the decline threshold.

### Reproducibility, and when it fails

`config_hash_matches_current` is false when the business constants that priced this
decision are not the ones running now. When that happens, everything in `economics` and in
the counterfactuals is *a counterfactual under today's constants* — `matches_ledger_action`
being true tells you today's configuration would reach the same action, which is useful and
is not the same claim as "this is what we did". Say the right one of those in a dispute
response.

---

## 4. The counterfactual

The decision boundary is an explicit rational function of amount and tenure, so we do not
sample it — we solve it. `amount_counterfactual.flips` gives the **exact** basket value at
which the action changes, holding the recorded probability fixed, verified against
`policy.decide` before it is reported.

Worked example: transaction 2987000, calibrated p = 0.90, $250 basket, 20-day-old account.

```
economics.tenure_bucket      8-30d
economics.allow_to_challenge 0.0699
economics.challenge_to_deny  0.8898     <- p = 0.90 is above it, hence deny
economics.allow_to_deny      0.4885     <- the published-form number; not operative

amount_counterfactual.flips[0]
  amount                       197.94
  boundary                     challenge_to_deny
  boundary_at_observed_amount  0.8898   <- the threshold p=0.900 actually cleared
  boundary_at_flip_amount      0.9000   <- equals p by construction; a free check
  action_below                 challenge
  action_at_or_above           deny
```

> Observed deny at $250.00 on a score of 0.900 for an account in the 8-30d bucket; the
> challenge/deny threshold is 0.890 here and would have been 0.900 at $197.94, so below
> that amount this transaction would have been challenge rather than deny.

Both quoted thresholds are the boundary that actually moved. An earlier version of this
sentence quoted `allow_to_deny` — 0.516 against 0.489 — which is arithmetically correct
and reads as though a 0.900 score cleared the line by four hundred basis points, when in
fact it cleared the operative threshold by one. Getting that wrong on a dispute letter is
the same class of error as ADR-0003: a true number in a place where it will be read as a
different one.

That is a precise, defensible answer to "why me, and what would have been different" — and
it is $197.94, not "somewhere between $150 and $250", because it was solved rather than
probed.

An **empty** `flips` list is a real answer, not a failure: it means no positive basket
value changes the outcome at this score. It is the common case, and the `statement` field
says so in words.

`tenure_counterfactual` enumerates all seven tenure buckets plus `unknown` — tenure is
categorical, so there is nothing to solve. On the same case it produces the result that
surprises people:

> Observed deny at 8-30d tenure. The same score and basket would have been challenge at
> 1-7d.

A week-old account gets *more* benefit of the doubt than a five-year-old one, because
`relationship_cost` is highest for a brand-new customer (P(churn | declined) = 0.38 against
a $541 residual LTV at 1–7d, versus 0.09 against $367 at 400d+). This inverts what most
fraud teams expect and it is the correct consequence of the cost model. Expect to explain
it. It is in the response precisely so you can.

---

## 5. Attribution, and what we refuse to say

`attribution.contributions` is **`null`**, and it will stay null until this deployment has
both a registered scoring artifact and the feature vector for the transaction. It is not an
empty list — an empty list reads as "we checked and nothing contributed", which is a far
stronger claim than the truth — and it is never a zero.

The refusal is a field on the response (`attribution.refusal`), stable and quotable,
because it is the part a compliance reviewer will ask about:

- Of the feature vector, only a handful of columns have any published meaning:
  `TransactionAmt`, `ProductCD`, `card4`/`card6` (brand and type),
  `P_emaildomain`/`R_emaildomain`, and `D1`. Those are in
  `attribution.documented_features` and are the only ones an attribution would ever be
  reported over.
- `C1`–`C14` count something Vesta never documented. `V1`–`V339` are undocumented
  engineered features. **We will not emit "V258 contributed +0.31" dressed as an
  explanation**, and we will not invent behavioural language such as "5× the customer's
  median" — there is no customer history in the feature vector to support it.
- A portion of the score is therefore attributable to features nobody can explain. Saying
  so is more useful to a dispute handler than a fabricated narrative, and a fabricated
  explanation on an adverse-action notice is a compliance problem, not a quality one.

What *is* fully explainable is the economics, and it is the whole rest of this response.
Lead with it.

---

## 6. The adverse-action path

When a declined customer asks why, the defensible answer is assembled from this endpoint in
this order, and nothing else:

1. **The action and when it was taken** — `decision.action`, `decision.decided_at`.
2. **The reason codes** — `decision.reason_codes`. These are stable machine identifiers,
   never display text; rendering and localisation are yours (ADR-0002 rules i18n out of
   scope here for exactly this reason). `SCORE_ABOVE_DENY_BOUNDARY` joins to your copy,
   and changing the copy must not change the audit trail.
3. **The economic context, in plain terms** — the amount band and the tenure band that
   placed the boundary where it was (`HIGH_AMOUNT_BAND`, `NEW_ACCOUNT_TENURE`,
   `UNKNOWN_TENURE_PRICED_AS_MEDIAN`). `UNKNOWN_TENURE_PRICED_AS_MEDIAN` in particular
   means we priced an unidentified customer as an average one, which is an assumption the
   customer is entitled to know was applied to them.
4. **What would have changed the outcome** — the counterfactual, quantified.
5. **The honest limit** — a portion of the score comes from features the vendor does not
   document, and we say so rather than inventing a reason.

Two rules for the response you send:

- **Never quote the calibrated probability to the customer.** It is a model output on a
  scale that means nothing outside this system, and quoting it invites a debate about a
  number rather than about the decision.
- **A degraded decline is a different letter.** `decision.degraded` true means no model
  scored this transaction: it was decided by the documented rule ladder
  (`policy.fallback`), the only rung that declines being a ≥$500 basket on an account under
  eight days old. There is no score, no boundary comparison and no counterfactual for it —
  the response returns `null` for all three with the reason stated — and the answer to
  "what would have been different" is the rung, not a threshold. Do not let a template fill
  those nulls in.

---

## 7. Integrating a case-management tool (Sift, Forter, internal)

These tools expect to be fed a queue. Do not build the adapter that fabricates one.

The supported integration is *lookup*: your tool holds the case (it already has the order,
the customer and the dispute), and calls this endpoint with the transaction id plus the
amount and D1 it already knows, to enrich that case with the decision, the version set and
the counterfactual. It is a read, per case, on demand.

- **Cardinality:** one transaction per call. There is no list endpoint and no search. If
  you need a population — "every deny last Tuesday" — query the ledger directly; that is an
  analytics question and it does not belong on the serving container.
- **Latency:** no SLO. This is off the §4.3 budget by design. Do not place it in a
  synchronous customer-facing flow and then discover it has no budget.
- **404 vs 503:** 404 means we hold no decision for that id — which can legitimately mean
  the decision was made but the ledger write failed, a hole that
  `fraudlens_ledger_write_failures_total` exists to make countable. 503 means this instance
  has no ledger reader configured. Alert on the second; investigate the first.
- **Caching:** the ledger is append-only and a `transaction_id` is decided once, so the
  `decision` block is immutable and cacheable indefinitely. The derived blocks are not:
  they are recomputed from the constants running now, which is the point of
  `config_hash_matches_current`.
