# ADR-0004: No analyst review queue, and what the case API is for instead

**Status:** Accepted
**Date:** 2026-08-05
**Refs:** findings §3, §5, §7; ADR-0002 (auditability ranked #2); ADR-0003

---

## Context

Every reference architecture for fraud decisioning has a review queue in it. A
transaction is flagged, it enters a prioritised queue, an analyst opens a 360°
case view, decides, and the decision becomes a training label. It is what the
commercial platforms sell and what most teams build. Reviewers of this system
have twice asked where ours is.

We measured it, and at this merchant it loses money.

**Finding §5.** Value-of-review is positive for **5 of 92,427** transactions, at
$7.97 per case. Every capacity-allocation rule we tried is worse than having no
queue at all:

| Ranking rule | Annual cost | Realised value of review |
|---|---|---|
| by score (the case-management default) | $2,997,659 | −$16,678 |
| by uncertainty p(1−p) | $3,003,436 | −$17,041 |
| by uncertainty × exposure | $3,106,035 | −$28,116 |
| by value-of-review | $2,980,763 | −$14,783 |
| **no review at all** | **$2,799,797** | — |

Ranking by uncertainty × exposure is the *worst* of the four, which is worth
sitting with: it fills the queue with large, genuinely ambiguous transactions,
and that is correct advice under a different cost structure. The rule is not
stupid. It is right somewhere else.

**Why it loses here.** The third action already exists and it is cheap. A
step-up challenge recovers roughly 68% of the gain that separates the best
global threshold from the four-action policy, at near-zero marginal cost and no
capacity ceiling. An analyst is a fourth action that costs $7.97, adds latency,
and is capped at 60 slots/day — competing against something that is nearly free
and unbounded. There is very little left for a human to add.

**The number the programme actually hinges on** is `f`, the rate at which
fraudsters defeat the step-up challenge. It is a vendor performance figure, not
a modelling one:

| f ↓ / analyst cost → | $2.00 | $4.00 | $7.97 | $15.00 |
|---|---|---|---|---|
| 0.05 | 0.00% | 0.00% | 0.00% | 0.00% |
| 0.20 | 2.16% | 1.40% | 0.50% | 0.17% |
| 0.50 | 4.88% | 4.20% | 3.07% | 1.60% |

Below f ≈ 0.20, do not staff a review team for payments at all.

## Decision

**This deployment runs the three-action policy — allow, challenge, deny — and
ships no review queue.** `serving.decisioning.INCLUDE_REVIEW = False`, stated
explicitly rather than defaulted, and `policy.decide` deliberately has no default
for that parameter so the choice cannot be made by omission.

Three consequences are deliberate and should not be read as omissions:

1. **No queue metrics.** E6 declined to emit analyst queue depth or value per
   analyst-hour. An empty queue-depth panel implies a queue that is keeping up,
   which is a stronger and more misleading claim than showing nothing (ADR-0003).
2. **`policy.queue.rank_for_review` still exists**, for the offline path that can
   honour a capacity constraint. It is not wired to serving and must not be — the
   request path cannot check a queue inside a 150 ms budget, and a service that
   emits `review` for cases no analyst will ever see has invented an approve with
   extra steps.
3. **The four-action policy remains the published headline** ($2,799,797). This
   deployment's three-action variant costs $583/yr *less* on the test window,
   because the review arm is selected 5 times. Both numbers are real; they are
   different policies and the ledger records which one decided.

**What we build instead: an investigation API, not a review queue.** The humans
in this system are real, they are simply not reviewers:

| Who | What they need | Why they exist |
|---|---|---|
| Dispute handlers | The full case behind a chargeback | `OPS_DISPUTE` is a term in our own cost function |
| Adverse-action responders | Why this customer was declined | ADR-0002 ranks auditability #2, above latency |
| On-call engineers | The decisions behind an alert | docs/runbooks/ sends them somewhere |
| Promotion approvers | Champion vs challenger on real decisions | The runbooks name who decides |

`GET /v1/cases/{transaction_id}` serves those four. It has **no latency budget**
— it is explicitly off the §4.3 path — and it deliberately implements no queue
semantics: no assignment, no claiming, no SLA timers, no work distribution.
Building those would be building the queue this ADR declines, one convenience at
a time.

## Explainability, and what we refuse to fabricate

The obvious ask is SHAP. It does not survive contact with this dataset.

Of the 228 feature columns, only a handful are semantically nameable:
`TransactionAmt`, `ProductCD`, `card4`/`card6` (brand and type),
`P_emaildomain`/`R_emaildomain`, and `D1` (days since first seen). `C1`–`C14`
count something Vesta never documented. `V1`–`V339` are undocumented engineered
features. A TreeSHAP attribution over those yields *"V258 contributed +0.31"* —
a number with a label attached, not an explanation.

Worse, the canonical analyst-facing phrasing — *"amount is 5× this customer's
median"* — **is not derivable here at all.** There is no customer history in the
feature vector; there is `D1`, and that is a tenure, not a baseline. Generating
that sentence would mean inventing a comparison we never computed. On an
adverse-action notice that is not a quality problem, it is a compliance problem.

**Decision: attribute over the nameable features, refuse to narrate the
anonymized ones, and report honestly how much of the score is unattributable.**
That admission is more useful to a dispute handler than a fluent fabrication,
because it tells them what they can and cannot put in a letter.

**What we have that is better than SHAP for this system:** the decision boundary
is an explicit economic function of amount and tenure, so the counterfactual is
*solvable* rather than approximable. We can state the exact amount at which the
action would flip — *"at $200 rather than $2,000 the break-even would have been
0.731 rather than 0.369, and this transaction would have been allowed"*. A
perturbation method would sample its way toward an answer we can compute in
closed form. This is a genuine advantage of deriving the policy from economics
rather than fitting a threshold, and it only exists because the boundary is not a
black box.

## Consequences

**Accepted.**
- We have no human-in-the-loop label source, so every label arrives via
  chargeback at 34–90 days. That is the constraint driving the entire monitoring
  and flywheel design, and declining the queue means declining the one lever that
  would shorten it. `docs/findings/review-as-label-acquisition.md` prices that
  lever explicitly rather than leaving it as an intuition.
- If `f` rises above ~0.20, or analyst cost falls, this decision flips. It is
  keyed to two measurable numbers, and both belong to a vendor rather than to us.
  **Neither is currently monitored.** That is a real gap: an ADR keyed to
  quantities nobody watches decays into a belief.

**Rejected alternatives.**
- *Ship the queue anyway because everyone has one.* This is the decision the
  whole project argues against — it optimises for looking like a reference
  architecture rather than for the merchant's P&L, and we can put a number on
  what it would cost.
- *Ship a queue but leave it unstaffed.* Worse than both options. The service
  would emit `review` for cases nobody sees, which is an approve wearing a
  different label, and it would be recorded in the ledger as a considered
  decision.
- *Build the case view as a queue-shaped API "for future flexibility".* §9a
  prohibits speculative extension points. If `f` moves, the queue can be built
  then, by someone who knows what it needs to do.

## Review trigger

Revisit when any of these becomes true, and note that the first two are the ones
we are not currently measuring:

- measured `f` (challenge defeat rate) exceeds 0.20
- fully-loaded analyst cost per case falls below ~$4.00
- the merchant's mix shifts materially toward high-amount baskets, where the
  break-even is lowest (0.369 at $500+) and review has the most room
- a regulator requires human review of automated adverse decisions, which makes
  this a compliance requirement rather than an economic one — in which case the
  cost is the price of operating, and the queue gets built
