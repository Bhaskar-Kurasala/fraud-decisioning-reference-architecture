# Architecture

**What this is.** The system that productionises the research findings in
`fraud-decisioning-findings.md` — a fraud-decisioning service for an IEEE-CIS /
Vesta merchant, built around an economics-driven policy rather than a learned
threshold. This document is the spine: it ties the findings to the code, names
the decisions and the rejected alternatives, and is the one thing to read to
understand the system end to end.

**Date:** 2026-08-05 · **56 source files, 62 test files, 515 tests, 4 ADRs**

---

## 1. The problem the findings describe

The research produced seven findings. Three of them shaped the architecture
more than the rest, and the architecture exists to keep them true:

| Finding | Number | What it forced |
|---|---|---|
| The intermediate actions (challenge) are worth 3× the threshold tuning | §1: $583/yr | A three-action policy, not a binary classifier |
| Miscalibration costs $4.4M/yr at an AUC difference nobody would block | §2: $4,358,096/yr | Isotonic calibration on a held-out window, checked not assumed |
| The label-latency bug: observed fraud collapses to 6.9% of true | §6: 38.7% | Every label-dependent metric maturity-gated; the flywheel split into Tier 0 triggers and Tier 2 promotions |

Finding §5 — review does not pay at this merchant — shaped the system by its
absence: there is no review queue, and ADR-0004 records why. Finding §3 — the
break-even is a distribution, not a number — is what makes the investigation
API's counterfactual solvable in closed form.

---

## 2. The layered architecture

The import contract is the spine. It is enforced at build time by
import-linter, not by review:

```
serving      ← the request path; nothing above it may be reachable
flywheel     ← retrain trigger, shadow, promotion, rollback
monitoring   ← drift, maturity, Tier 0–2 metrics
lineage      ← replay, model card
streaming    ← ledger, labels, provenance — the audit trail
policy       ← the decision boundary and the fail-safe ladder
economics    ← the cost model: L, M, relationship_cost
models       ← scorer, calibration, registry, gate, checks
features     ← tenure buckets, amount bands
config       ← business constants, settings
```

**The ordering is the load-bearing part.** Two consequences that are not
accidental:

- **Economics is below policy.** The cost model can be tested and reasoned
  about in isolation, without importing the thing that consumes it. A cycle
  here — economics depending on policy — would mean the cost functions can no
  longer be verified independently of the decision rule that uses them, and
  they are the part that has to be exactly right.

- **The flywheel is below serving and above monitoring.** A retraining decision
  must never be reachable from the request path. Putting the flywheel between
  serving and monitoring makes that a build failure rather than a 3 a.m.
  discovery that a promotion ran inside a 150 ms budget. The flywheel reads
  monitoring's drift signals to decide *when* to retrain and the registry to
  decide *what* to promote — both below it in the contract.

---

## 3. The decision path

```
POST /v1/decide
  │
  ├── input_hash(request) ──────────────────────── digest for the ledger
  ├── CalibratedScorer.score(features) ─────────── raw score → isotonic → p
  │
  ├── if scorer is unavailable:
  │     policy.fallback.decide_without_model() ── the fail-safe ladder
  │     (deny high-amount new accounts; challenge everything else; never allow)
  │
  ├── policy.decide(p, L, M, include_review=False)
  │     argmax over {allow, challenge, deny} of expected value
  │     L = Amount × COGS + CB_FEE + OPS_DISPUTE       (cost of a missed fraud)
  │     M = Amount × MARGIN + relationship_cost(tenure) (cost of a false decline)
  │
  ├── reasons.attach(action, score, tenure) ────── reason codes for the letter
  │
  └── DecisionLedger.record(decision) ───────────── append-only, idempotent
        │
        └── 150 ms p99 budget (§4.3), measured by open-loop load generation
```

**Why the boundary is explicit, not learned.** The decision boundary is
`M / (L + M)` — the point where the expected cost of allowing equals the
expected cost of declining. It is a rational function of amount and tenure, not
a fitted threshold. This costs something (the whole cost model had to be
explicit) and buys something that no other fraud system I have worked on has:
the counterfactual is *solvable*. The investigation API can state the exact
amount at which a deny would have become a challenge — `$197.94`, not
"somewhere between $150 and $250" — because setting the boundary equal to the
recorded probability and solving for amount is one division.

**Why there are three actions, not two.** Finding §1 measured the challenge
arm — a step-up authentication that recovers ~68% of the gain from the
four-action policy at near-zero marginal cost. A binary allow/deny classifier
throws that away. `INCLUDE_REVIEW = False` is stated explicitly (not defaulted)
and `policy.decide` has no default for that parameter, so the choice cannot be
made by omission.

---

## 4. The fail-safe path

When the scorer is unavailable — model crash, dependency failure, cold start —
the system does not decline everything (lost revenue) and does not allow
everything (lost money). It runs a rule ladder:

1. **High-amount new account** (amount > $500, tenure < 7d) → **deny**
2. **Everything else** → **challenge**

Never **allow**, on any input. The ladder lives in `policy.fallback`, not in
`serving`, because the layer ordering has to allow `lineage.replay` to call it —
an outage produces the largest block of unusual decisions, and that block is the
one a regulator would ask about first. Moving the ladder to `policy` closed
lineance gap 2 and made degraded decisions replayable.

---

## 5. The flywheel

```
Tier 0 (label-free, day 0)              Tier 2 (label-dependent, day 90+)
┌──────────────────────┐                ┌──────────────────────────┐
│ score PSI            │                │ require_mature(≥ 80%)    │
│ feature drift breadth│   retrain →    │ confidence sequence      │
│ action-mix TV shift  │ ──────────→    │ (anytime-valid, not t-test)│
│ staleness (90d max)  │   shadow →     │ promotion gate on cost   │
└──────────────────────┘   promote      └──────────────────────────┘
        │                                                    │
        └── trigger.evaluate_trigger() ──┘         promotion.evaluate_window()
```

**The split is the whole design.** Tier 0 signals fire the retrain on the day
traffic arrives. Tier 2 signals gate the promotion on matured labels. The
alternative — triggering on performance — is unimplementable: §6 measured
observed fraud collapsing to 6.9% of true over a 20-day block, so a performance
number on a recent window isn't noisy, it's *biased toward "fine."* A trigger
keyed on it fires a quarter after the traffic that broke the model.

**The confidence sequence, not the t-test.** Labels trickle in continuously, so
the promotion window is re-evaluated every time more chargebacks mature. A
fixed-horizon 95% interval re-read that way has no type-I error guarantee —
simulation showed it certifies a challenger identical to the champion **49% of
the time**. The anytime-valid confidence sequence (Robbins normal-mixture)
holds 0.25% under the same simulation. The cost: 1.5–4× larger effect to
certify.

**Shadow scoring is structurally not a decision.** A challenger scoring in
shadow writes to `shadow_scores`, a separate table — not a `shadow BOOLEAN` on
`decision_ledger`. Mixing them would mean every consumer of the ledger silently
double-counts, each needing the same filter remembered. A shadow score cannot
be mistaken for a decision because it is not in the place decisions are.

---

## 6. The analyst contract

This deployment has no review queue (ADR-0004). Value-of-review is positive on
**5 of 92,427** transactions. The four humans in this system are dispute
handlers, adverse-action responders, on-call engineers, and promotion
approvers — people investigating one decision they already have the ID of.

**`GET /v1/cases/{transaction_id}`** serves them: the full recorded decision,
the economics, the closed-form counterfactual, and an honest attribution
refusal. The refusal is the most important part: of 228 feature columns, only
seven are vendor-documented (`C1–C14` count something Vesta never specified,
`V1–V339` are undocumented engineered features). A TreeSHAP attribution yields
"V258 contributed +0.31" — a number with a label, not an explanation. The
response reports honestly how much of the score is unattributable, which is a
worse-sounding answer that a dispute handler can actually act on.

**`GET /cases/{transaction_id}`** renders the same content as HTML —
server-rendered, no JavaScript, no CDN, no build step. An air-gapped payments
estate cannot fetch either, and a page that renders naked during an incident is
worse than no page.

---

## 7. Label provenance

A chargeback is an *outcome*. A human adjudication is an *opinion* at q=0.91
from a censored sample. They cannot share a table (E12c):

| | Chargeback | Human adjudication |
|---|---|---|
| Table | `revealed_labels` | `human_adjudications` |
| Arrives | 34-day median, 90-day tail | same day |
| Error rate | 0% (ground truth) | 9% (q_analyst) |
| Sampling | population-wide | above the decision boundary |
| Default for training | **yes** | no (opt-in only) |

The default training/promotion path reads `revealed_labels` and structurally
cannot see human labels. Chargeback wins on reconciliation; disagreements are
recorded as the control-chart signal.

---

## 8. Quality attributes (ADR-0002)

The ranking, and what it bought:

1. **Reliability** — the fail-safe ladder (never `allow` on any input); the
   `RETURNING`-based idempotency that works on both engines; the append-only
   ledger enforced by trigger, not by convention.
2. **Auditability** — every decision is replayable: same input + same recorded
   versions → same action, asserted by test on every arm of the policy,
   including the degraded path. The ledger stores `input_hash`, not the
   features — an undeletable copy of customer signals has no deletion path.
3. **Latency** — 150 ms p99 budget, measured by open-loop load generation
   (§4.3). The investigation API is explicitly off this path.
4. **Consistency** — the promotion gate refuses an immature window *before*
   computing any cost, because a number that exists gets pasted into a deck
   and loses its caveat. `require_mature` raises rather than warns.
5. **Observability** — five metric tiers ordered by *observability latency*
   (not subject): Tier 0 fires day 0, Tier 2 waits for labels. Absence never
   renders as a value (ADR-0003) — a gauge that could not be computed is `NaN`,
   not yesterday's number.

---

## 9. What this system does not do

Each refusal is measured, not accidental:

- **No review queue.** §5 prices it: review loses $14,783/yr against no queue at
  all. The queue pays as an *instrument* (5–20 slots/day, not 60) at a break-
  even that depends on an unmeasured hazard rate (E12b).
- **No per-feature attribution.** No champion artifact exists (research scripts
  persist scores, not estimators), and the ledger stores `input_hash` not
  features. An attribution would be fabricated, and on an adverse-action notice
  that is a compliance problem, not a quality one.
- **No performance-based retrain trigger.** §6: observed fraud is 6.9% of true
  on recent windows. The trigger would fire a quarter late.
- **No speculative extension points.** §9a: zero interfaces with zero
  implementations. A null field with a stated reason can be filled in later
  without a seam.

---

## 10. The numbers, as one table

| Quantity | Value | Where it lives |
|---|---|---|
| Annual cost under the recommended policy | $2,799,797 | findings §7 |
| Cost of miscalibration (the thing calibration prevents) | $4,358,096/yr | findings §2 |
| Value of the challenge arm vs binary threshold | 3× (§1) | findings §1 |
| Chargeback median lag | 34 days | `config.settings` |
| Clean-window close (step maturity) | 90 days | `config.settings` |
| Maturity floor (refuse below) | 80% | `monitoring.maturity` |
| Score PSI alert threshold | 0.25 | `monitoring.drift` |
| Analyst agreement rate | 0.91 | `config.settings` |
| Label-acquisition break-even (5 slots/day) | 0.021 events/yr | E12b |
| Peeking false-positive rate (fixed-horizon) | 49.25% | E8 |
| Peeking false-positive rate (confidence sequence) | 0.25% | E8 |

---

## 11. Open gaps

Summarised from `docs/lineage.md`; see there for detail and cost-to-close.

- **Gap 1:** `model_version` names an artifact but does not pin its contents.
- **Gap 3:** `config_hash` proves *different*, never *what*.
- **Gap 4:** `policy_version` is a free string (partially closed by replay table).
- **Gap 5:** Append-only by trigger, not sealed (hash-chain needed for legal evidence).
- **Gap 6:** Research pipeline emits no manifest.
- **Gap 7:** One published number cannot be regenerated (the $1,999 LTV).
- **Gap 8:** `mlruns/` is local; the model card's run pointer is not shareable.
- **Gap 9:** Model card limitations are code, not a live reference.

Closed: gap 2 (degraded replay), gap 10 (label provenance).
