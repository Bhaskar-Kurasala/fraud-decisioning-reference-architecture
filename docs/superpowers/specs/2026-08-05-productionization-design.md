# Productionizing the Fraud Decisioning System — Design Spec

**Date:** 2026-08-05
**Status:** Approved
**Author:** Architecture working session

---

## 1. Context

An existing research pipeline (`01..06_*.py`) trains a fraud model on IEEE-CIS
data and derives a cost-optimal decisioning policy. Its output is
`fraud-decisioning-findings.md`: a set of measured claims about calibration,
threshold economics, review capacity, and label latency.

The pipeline has been run. `data/` holds all artifacts; `outputs/` holds all
seven stage logs (2026-08-05 00:17–00:18). The directory is **not** a git
repository and has no commit history.

This spec covers converting that research artifact into a production system:
serving, monitoring, observability, deployment, a data flywheel, and lineage.

### 1.1 Purpose

The research established *what the right decision policy is*. It did not
establish that we can **operate** one. Those are different problems, and the
second is where fraud programmes usually fail.

The findings are worth $2.4M/yr against a do-nothing baseline. That number is
currently trapped in a set of scripts that ran once, on a laptop, against a
static file. To collect it we need a system that scores live traffic within the
checkout latency budget, keeps working when the fraud population shifts under
us, tells us it has stopped working *before* the losses land, and can prove
after the fact why any individual customer was declined.

Each of those is an unsolved problem in the current state:

| Research established | Operating requires |
|---|---|
| A policy worth $2.4M/yr on historical data | Evidence it still holds on traffic we have not seen |
| Calibration is worth $6.6M/yr | Detection when calibration decays, before the loss appears |
| Optimal thresholds vary by amount and tenure | Those thresholds applied per-transaction, in-line, under 150ms |
| The model scored AUC 0.9045 once | Knowing whether today's model still does, given labels arrive 30–90 days late |
| A policy derived by hand | A defensible basis for changing it, and for reverting |

The investigation this spec plans is therefore: *what does it actually take to
run this thing, and what breaks first?* The system must genuinely work — the
reasoning about why it is built this way is recorded as it is discovered.

### 1.2 Environment (measured 2026-08-05)

| | |
|---|---|
| CPU / RAM | 10 cores, 11 GB (4.1 GB available), 8 GB swap |
| Disk | 777 GB free |
| Docker | 29.5.2, Compose v5.1.4, daemon running |
| Kubernetes | none installed (no kind/minikube/kubectl/helm) |
| Python | 3.10.12; `uv` present; sklearn 1.7.2, pandas 2.3.3, numpy 2.2.6 |
| Absent | mlflow, evidently |
| Present | fastapi 0.136.3 |

Two consequences:

1. The README's "OOMs on 3 GB" limitation for `FIT_BALANCED=1` no longer binds.
   The empirical class-rebalanced refit is runnable and **will be run**, closing
   a stated limitation with a measurement.
2. The deployable demo is **Docker Compose native**. Kubernetes manifests are
   authored and validated statically (`kubeconform`) but not applied. This is
   stated plainly rather than implied otherwise.

---

## 2. Goals and non-goals

### Goals

- Verify every numeric claim in `fraud-decisioning-findings.md` against
  regenerated artifacts; report discrepancies honestly.
- Extract research logic into a tested package with **one** source of truth for
  features, economics, and policy.
- Serve decisions over HTTP with reason codes and a latency SLO.
- Replay 182 days as a stream; log decisions; reveal labels on a delay.
- Monitor across five metric tiers, with real-time leading indicators carrying
  the load while labels mature.
- Close the loop: drift/decay triggers retrain → challenger → promotion gate.
- Record lineage such that any decision is reproducible from its inputs.
- Produce a commit history that reads as a learning progression.

### Non-goals

- Real cloud deployment, real payment traffic, real PII.
- Beating the Kaggle leaderboard. The out-of-time AUC 0.9045 is deliberately
  below leaderboard scores because leaderboard entries leak future information
  through combined train+test entity aggregates.
- Fairness analysis. IEEE-CIS carries no protected attributes; a credible
  fairness study needs the Bank Account Fraud (NeurIPS 2022) dataset. Stated as
  a limitation, not faked.

---

## 3. Repository structure

Approach: **extract and preserve**. Research scripts move to `research/`
unmodified initially (they are the audit trail for the findings), then are
refactored to import from the extracted package so research and production
cannot drift.

```
.
├── src/fraudlens/
│   ├── config/          settings (pydantic-settings), business constants
│   ├── features/        feature builders — shared research ↔ serving
│   ├── models/          training, calibration, MLflow registry adapters
│   ├── economics/       cost model, LTV, expected value
│   ├── policy/          thresholds, four-action tiering, capacity assignment
│   ├── serving/         FastAPI app, schemas, reason codes
│   ├── streaming/       replay producer, decision ledger, label revealer
│   ├── monitoring/      drift, calibration, performance, exporters
│   ├── flywheel/        retrain trigger, challenger, promotion gate
│   └── lineage/         provenance, model cards, audit trail
├── research/            01..06 preserved — provenance of the findings
├── tests/               unit, integration, contract, golden-value
├── deploy/
│   ├── compose/         docker-compose stack
│   ├── grafana/         provisioned dashboards (as code)
│   ├── prometheus/      scrape config, alert rules
│   ├── k8s/             manifests (validated, not applied)
│   └── runbooks/        one per alert
├── docs/
│   ├── adr/             architecture decision records
│   ├── architecture/    C4 diagrams, metric catalog
│   └── findings/        verification report, updated findings
└── data/, outputs/      existing artifacts (git-ignored)
```

**Rationale for rejected alternatives.** A *sidecar* (leave research alone,
build production beside it) produces two definitions of every feature and cost
— the drift is invisible until it is expensive. A *full rewrite* destroys the
provenance linking the findings to the code that produced them.

---

## 4. Runtime architecture

```
replay producer ──► scoring API ──► decision ledger (Postgres)
 (182 days,          (FastAPI)          │
  time-ordered)          │              ▼
                         │       label revealer (configurable delay)
                         │              │
                         ▼              ▼
                  Prometheus ◄─ monitoring workers (drift, calibration)
                         │              │
                         ▼              ▼
                     Grafana       flywheel (retrain → challenger → gate)
                                          │
                                          ▼
                                   MLflow (runs, registry, promotion)
```

### 4.1 Components

| Component | Responsibility | Interface |
|---|---|---|
| Replay producer | Emit test-window transactions in time order at configurable rate | HTTP → scoring API |
| Scoring API | Features → score → calibrate → policy → decision + reason codes | `POST /v1/decide` |
| Decision ledger | Append-only record of every decision with full lineage | Postgres |
| Label revealer | Reveal true label after configurable maturation delay | Postgres writer |
| Monitoring workers | Compute drift, calibration, performance on schedule | Prometheus exporters |
| Flywheel | Detect decay → retrain → evaluate challenger → gate → promote | MLflow registry |
| MLflow | Experiment tracking, model registry, stage transitions | Server + backing store |

### 4.2 The promotion gate

The gate is the architecturally load-bearing piece. A challenger promotes only
if it beats the champion on **expected cost computed on matured labels** — not
on AUC.

The findings already prove why AUC is the wrong gate: a class-rebalanced score
with **identical AUC (0.9050) and identical PR-AUC** to the champion costs
**$6.6M/yr more** when routed through an EV policy, because its ECE is 0.2009
instead of 0.0027. Any gate keyed on ranking metrics would promote that model.

The gate therefore checks, in order:

1. **Label maturity** — refuse to evaluate below a maturity threshold.
2. **Calibration** — ECE must not regress beyond tolerance.
3. **Cost delta** — sequential test on per-transaction cost difference, paired
   on the same transactions, with confidence interval.
4. **Segment guard** — no segment (amount band, tenure) may regress beyond
   tolerance even if aggregate improves.

Sequential testing rather than a fixed-horizon t-test because, as the blueprint
argues, a 1% fraud improvement is not detectable by naive A/B at realistic
volumes. Failing the gate is a normal outcome and is logged, not suppressed.

### 4.3 Latency budget

`POST /v1/decide` p99 ≤ 150 ms end to end, allocated: feature assembly 60 ms,
model inference 25 ms, calibration + policy 15 ms, ledger write (async) 0 ms,
overhead 50 ms. Breaches counted as an SLO metric, not merely logged.

---

## 5. Metrics catalog

Organised by **observability latency**, not by subject. The organising
constraint:

> Chargebacks mature over 30–90 days. The question "is the model still working?"
> is unanswerable in real time. A dashboard showing only AUC is blind for a
> month.

Tier 0 exists because it is all you have on day zero.

### Tier 0 — Real-time leading indicators (no labels required)

| Metric | Rationale |
|---|---|
| Score distribution PSI vs training baseline | Earliest possible warning |
| Approve / review / decline rate | Policy shift visible before loss is |
| Mean, p95 predicted probability | Prediction drift |
| Per-feature drift (PSI/KS), top-10 drifting | Localises the moved input |
| Unseen categorical level rate | Entity churn |
| Null / missing rate per feature vs training | Leading cause of silent failure |
| Training–serving skew (offline vs online score, same row) | Catches pipeline bugs |
| GMV approved / declined per hour | Business pulse |
| Analyst queue depth and wait time | Capacity breach before SLA breach |

### Tier 1 — Business P&L (objective function; default landing dashboard)

| Metric | Note |
|---|---|
| **Net value saved ($/day, annualised)** | Headline |
| Realized vs expected total cost | Calibration expressed in dollars |
| Fraud loss $ and chargeback count (matured) | Lagging truth |
| False-decline cost $ | Marked assumption-dependent on the panel itself |
| Cost per decision | Unit economics |
| Approval rate by tenure segment | New-customer approval is the growth metric |
| Value per analyst-hour | Does review earn its cost |
| Decline $ / approve $ by amount band | Where policy binds |

### Tier 2 — Model health (label-dependent, lagging)

Every panel stamped with label maturity.

| Metric | Note |
|---|---|
| **Label maturity curve** (% labeled at t+7/30/60/90d) | Gates every panel in this tier |
| AUC / PR-AUC, rolling, matured labels only | PR-AUC primary at 3.5% base rate |
| **ECE + reliability curve** | Worth $6.6M/yr per findings §2 |
| Calibration slope and intercept | Direction, not just magnitude |
| Actual vs predicted fraud rate | Base-rate drift |
| Performance by segment | Aggregate AUC hides segment collapse |
| Champion vs challenger cost delta + CI | Gate input |
| Days since retrain | Staleness |
| Decile lift | Analyst-legible |

### Tier 3 — System and serving

| Metric | Target |
|---|---|
| p50/p95/**p99** latency, budget breach count | p99 ≤ 150 ms |
| Throughput, error rate, 5xx | |
| Feature lookup latency, cache hit rate | Usually the real cost |
| Model load time, memory, restarts | |
| End-to-end trace spans | OpenTelemetry |

### Tier 4 — Data quality and governance

| Metric |
|---|
| Schema violation rate |
| Duplicate transaction rate |
| Late / out-of-order event rate |
| % decisions with complete lineage (model + feature + policy version + input hash) |
| Reason-code coverage |
| **Assumption freshness** — days since `P(churn\|declined)` reviewed, with owner |

### 5.1 Dashboards

Four, provisioned as code in `deploy/grafana/`:

- `00-executive` — Tier 1. **Default landing.**
- `01-model-health` — Tier 2.
- `02-serving-slo` — Tier 3.
- `03-data-quality` — Tier 4 + Tier 0 drift detail.

### 5.2 Alerts

Only these page; everything else is dashboard-only. Each has a runbook.

| Alert | Condition |
|---|---|
| Score drift | PSI > 0.25 vs baseline |
| Data breakage | Null rate breach vs training baseline |
| Latency SLO | p99 > 150 ms sustained |
| Decline anomaly | Decline rate ±3σ from trailing window |
| Calibration decay | ECE > threshold on matured labels |
| Training–serving skew | Any detected mismatch |
| Promotion gate failure | Challenger rejected |

---

## 6. Verification (Epic E1)

Every claim below is re-derived from regenerated artifacts and diffed. The
verification report records each as **REPRODUCED**, **DISCREPANT** (with
magnitude and cause), or **UNVERIFIABLE** (with reason).

### Dataset-level

- 590,540 transactions, 182 days, $79.7M GMV, 3.50% fraud, $3.08M at risk
- Split: train 0–119 (414,542, 3.52%), calib 120–149 (83,571, 3.41%),
  test 150–181 (92,427, 3.48%)
- Test window GMV $12.7M; annualisation factor 365/32

### Model

- Out-of-time AUC **0.9045**, PR-AUC **0.527**

### Findings §1 — policy ladder (annual cost, saved, fraud-$ caught, FP count)

| Policy | Cost | Saved |
|---|---|---|
| P0 approve everything | $5,248,067 | — |
| P1 best global threshold (p ≥ 0.698) | $4,469,896 | $778,171 (14.8%) |
| P2 decline top 1% | $4,483,938 | $764,130 (14.6%) |
| P3 per-transaction EV | $4,339,611 | $908,457 (17.3%) |
| P4 four-action EV argmax | $2,799,797 | $2,448,270 (46.7%) |

Plus: P1→P2 gap $14k/yr; P1→P4 gap $1.67M/yr; third action ≈68% of the gain.

### Findings §2 — miscalibration

| Score | AUC | ECE | Annual cost | Penalty |
|---|---|---|---|---|
| Calibrated isotonic | 0.9045 | 0.0027 | $2,799,797 | — |
| Raw uncalibrated | 0.9050 | 0.0054 | $2,871,342 | $71,545 |
| Class-rebalanced | 0.9050 | 0.2009 | $9,424,667 | **$6,624,870** |

Rebalanced predicts 23.6% fraud on a 3.5% population; challenges 71% of traffic.
**This will be re-derived empirically via `FIT_BALANCED=1`** (now feasible at
11 GB) rather than analytically, and the two compared.

### Findings §3–§7

- Break-even by tenure: new 0.642, 1–7d 0.740, 31–90d 0.534, 400d+ 0.379
- Break-even by amount: <$25 → 0.731, $500+ → 0.369
- Median L/M = 0.80; FN cost median $84.95, FP cost median $144.00
- Value-of-review positive for **5 of 92,427** transactions at $7.97/case;
  all four queue rankings worse than the no-review policy
- Label latency: 38.7% of training-window fraud not yet disputed; observed rate
  collapses to 6.9% of true rate in the final 20 days
- Five-term P&L: fraud $1,979,064 (70.7%), friction $820,278 (29.3%),
  ops $455, infra $843, total $2,800,640
- LTV survivorship: $323 → $1,999 under naive conditioning; P(repeat) 44.5%
  across 151,928 watchable customers

### Handling discrepancies

If a number does not reproduce, the discrepancy is investigated, the cause
documented, and the findings doc corrected **in a commit that cites the
evidence**. The findings will not be silently regenerated to match. A corrected
finding with a documented cause is a stronger artifact than a suspiciously
perfect one.

---

## 7. Testing strategy

| Layer | Content |
|---|---|
| Unit | Feature builders, cost functions, EV math, policy selection, calibration |
| **Golden-value** | Locks verified findings numbers. Fails if refactoring changes any published result. |
| Property | Cost monotonicity in amount; EV argmax consistency; break-even bounds in [0,1] |
| Contract | API request/response schemas, versioned; breaking changes fail CI |
| Integration | Compose stack up → replay → decisions land → labels reveal → metrics move |
| Leakage regression | Asserts no train-window fit sees calibration or test rows |
| Load | Latency budget under target RPS |

Golden-value tests are the mechanism that prevents research/production drift.
Coverage target 85% on `src/fraudlens/`, enforced in CI.

---

## 8. Delivery plan — twelve epics

| Epic | Content | Depends on |
|---|---|---|
| E0 | Foundation: git init, `uv`/pyproject, ruff, mypy, pytest, pre-commit, CI, ADR-0001 | — |
| E1 | **Verification**: reproduce findings, diff all claims, golden tests, report | E0 |
| E2 | Core library extraction: features, economics, policy + unit tests | E0, E1 |
| E3 | MLflow: tracking, params/metrics, signatures, registry, promotion | E2 |
| E4 | Serving API: FastAPI, reason codes, contract tests, latency budget | E2 |
| E5 | Streaming: replay producer, decision ledger, delayed-label revealer | E2 |
| E6 | Observability: OTel, Prometheus, four Grafana dashboards, alert rules | E4, E5 |
| E7 | Drift and calibration monitoring (PSI, ECE, delayed-label performance) | E5 |
| E8 | Flywheel: trigger, shadow, challenger, promotion gate, rollback | E3, E7 |
| E9 | Lineage: provenance, model cards, audit trail, reproducibility manifest | E3, E5 |
| E10 | Deployment: compose stack, healthchecks, k8s manifests, load test, chaos | E6 |
| E11 | Synthesis: architecture doc, C4 diagrams, ADR index, operating handbook, what we would do differently | all |

E3/E4/E5 are mutually independent and parallelisable, as are E7/E9.

---

## 9. Commit conventions

The history is the lab notebook. Someone picking this system up in six months —
or the same engineer after forgetting the details — needs to know why each
choice was made and what was tried first. That is what makes the system
maintainable; the fact that the reasoning is also legible to a reader is a
consequence, not the goal.

1. **One commit = one reasoning step.** Not one file.
2. **Bodies carry reasoning**: problem, decision, rejected alternative, evidence.
3. **Failures stay in history.** Discrepancies, reverts, and performance
   regressions are committed then fixed — the sequence is the evidence of
   engineering judgement.
4. **ADRs for contested decisions only** (~10). No ADR for "we use pytest".
5. **Conventional Commits** (`feat|fix|perf|refactor|test|docs|chore|build|ci`)
   with `Epic:` and `Refs: ADR-NNNN` trailers.
6. **Ordering tells the story.** Tests land with or before implementation;
   instrumentation lands before the thing it measures is tuned; the measurement
   motivating a change is committed before the change.

Illustrative body:

```
feat(economics): make decision threshold amount-dependent

A single global threshold assumes every transaction has the same cost
asymmetry. It does not: L = Amount*0.70 + 37 grows with amount while
M = Amount*0.30 + relationship_cost(tenure) grows more slowly, so the
break-even probability falls as amount rises.

Rejected: tuning one global threshold on the test set — it optimises a
metric nobody is paid on.

Measured: per-transaction boundary recovers $908,457/yr vs $778,171 for
the best global threshold on the test window.

Epic: E2
Refs: ADR-0004
```

---

## 9a. Engineering standards

These are binding on every epic. They exist because the failure mode of a
twelve-epic plan executed in parallel is a large volume of plausible code that
nobody can maintain.

### Volume is not progress

Every module must justify its existence against the alternative of not writing
it. Specifically prohibited:

- Abstractions with one implementation. No `BaseScorer` with a single subclass,
  no strategy pattern for a strategy that never varies, no plugin registry for
  two things.
- Configuration for values that have never changed and have no reason to.
- Wrapper layers that only forward calls.
- Speculative extension points. If a second implementation arrives, the
  abstraction can be extracted then, with knowledge of what actually varies.

The economics module is the crown jewel and should be small: the cost functions
are four lines of arithmetic that must be exactly right, not a framework.

**Test for every file: can its purpose be stated in one sentence, and is that
sentence load-bearing?** If not, it is deleted or merged.

### Maintainability

- Every public function typed; `mypy --strict` on `src/fraudlens/`.
- Modules under ~300 lines. Beyond that is a signal of mixed responsibility.
- Dependencies flow one direction: `config → features → models → economics →
  policy → serving`. No cycles; enforced by an import-linter check in CI.
- Errors that lose money or data are handled explicitly. Everything else fails
  loudly rather than degrading silently — a fraud system that silently returns
  a default score is worse than one that returns 503.

### Reproducibility

Non-negotiable, because the entire value of the findings rests on it.

- Every artifact records the git SHA, config hash, and input data checksum that
  produced it.
- All seeds fixed and recorded; `SEED` already exists in `config.py`.
- Dependencies pinned with a lockfile (`uv.lock`), committed.
- Any published number regenerable by a single documented command.
- A decision in the ledger can be replayed: same inputs + same recorded model
  and policy versions must produce the same output. This is asserted by a test,
  not assumed.

### Commentary standard

Comments explain **why**, never what. The bar: a comment earns its place if it
records a decision, a constraint, a business consequence, or a non-obvious
failure mode. Specifically valuable here — noting where a line encodes a
business assumption rather than a technical fact, since those are the lines that
need review when the business changes.

```python
# Break-even falls as amount rises: L grows at the COGS rate (0.70) while M
# grows at the margin rate (0.30) plus a fixed relationship cost. Consequence:
# we must be 73% sure to decline a $20 order but only 37% sure on a $500 one,
# which inverts the intuition most fraud teams operate on.
```

Not:

```python
# calculate the break-even threshold
```

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Findings do not reproduce | Investigate, document cause, correct openly (§6) |
| Memory pressure (4.1 GB available of 11 GB) | Chunked processing; `FIT_BALANCED` run in isolation |
| Scope sprawl across 12 epics | Each epic independently valuable and separately committed |
| Compose stack too heavy to run | Profiles: `core` (API + Postgres) vs `full` (+ MLflow, Prometheus, Grafana) |
| `P(churn\|declined)` unmeasurable | Versioned assumption artifact with owner, review date, sensitivity analysis, and a dashboard freshness metric |

---

## 11. The soft assumption

`P(churn|declined)` has no data support in IEEE-CIS. It is tenure-graded from
published false-decline research and it drives the entire false-positive cost
term.

Production tooling cannot fix an unmeasurable input. It is therefore treated the
way a risk organisation treats a number it cannot measure: a **first-class,
versioned assumption artifact** with a named owner, a review date, a documented
provenance, a sensitivity analysis showing the decision boundaries it moves, and
a Tier-4 dashboard metric tracking its staleness.

Making the softest number the most visible one is the point.
