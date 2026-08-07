# The AI/ML Architect's Blueprint for Fraud & Abuse Decisioning
### The Model Portfolio, the Economics, and What to Deploy When
*A companion volume to "Real-Time Fraud and Abuse Detection — A Break/Fix System Design" — written from the seat of the ML architect, where the model choices actually get made.*

---

## What this document adds

The source document is the best of the three in this series. It already does what most system-design material never does: it states the objective function as **total cost, not AUC**; it separates *how risky is this* (model) from *what do we do about it* (policy) from *why did we do it* (audit); it draws governance rather than appending it; and Appendix C's "what to introduce when" is the correct antidote to architecture-by-inventory.

But there is a hole in the middle of it, and it is exactly the hole you asked me to fill.

> **In 102 pages, the phrase "GBDT ensemble" appears once — inside a latency budget diagram — and "logistic regression" appears once, in a governance aside. The document builds a magnificent chassis and never opens the engine.**

Every V-chapter says *the model* as though there were one, as though its class were obvious, and as though the choice had no business consequences. In production fraud, none of that is true. A mature platform runs **eight to fourteen models simultaneously**, of at least five different classes, on three different latency paths, retrained on four different cadences, with different regulators caring about different ones. Choosing among them, sequencing them against business triggers, and knowing when each stops earning its place is the ML architect's actual job.

This document therefore covers:

1. **The economics of each fraud class**, computed — and the discovery that the FN:FP ratio spans from **1:8 to 355:1** across classes, which is why one model cannot serve them.
2. **The model portfolio** — fourteen model classes, what each is genuinely for, and the maturity ladder that ties each to a *business trigger* rather than a fashion.
3. **The problems that are modelling problems, not architecture problems** — delayed and censored labels, selection bias from your own declines, extreme imbalance, calibration, threshold optimisation under analyst capacity, adversarial probing.
4. **What to deploy when**, from a production and P&L point of view — including the three cases where the right answer is *buy a third-party signal instead of building a model*.
5. **Drift, retraining cadence and champion–challenger statistics** — including the power analysis that shows why you cannot A/B test a 1% improvement in fraud, and what to do instead.

Where the source document is authoritative I do not repeat it. I extend it.

---

# PART 0 — The ML Architect's Pre-Model Worksheet

Fourteen questions. The source document's Chapter 1 answers the system ones. These are the ones that determine what you train.

| # | Question | What it determines |
|---|---|---|
| 1 | **Which of the six fraud classes am I modelling?** | Each has a different loss ratio, label latency and adversary. Answer this or everything downstream is incoherent |
| 2 | **What does a false negative cost, and a false positive?** | The ratio sets the entire operating regime — see §2.1, where it ranges 1:8 to 355:1 |
| 3 | **When does a label arrive, and does it arrive at all?** | 45–120 days for chargebacks, never for declines. Determines whether supervised learning is even available |
| 4 | **What fraction of my training population was selected by my own model?** | Selection bias. Determines whether you need reject inference, and how much your offline metrics lie |
| 5 | **What is the base rate, and how many positives per segment per retrain window?** | Below ~500 positives per segment you cannot fit a segment model, however much you want one |
| 6 | **Is the decision inline, near-real-time, or batch?** | Sets the *maximum model complexity*, not just the serving stack. 18 ms of inference budget forbids most of the menu |
| 7 | **Must the score be a calibrated probability, or is a rank enough?** | If it feeds an EV equation — and it should — ranking models are unusable without calibration |
| 8 | **Must the decision be explained to a customer or a regulator?** | May eliminate whole model classes, or force a surrogate. Non-negotiable in some jurisdictions |
| 9 | **Is the signal in the row, the sequence, the graph, or the document?** | This — not model fashion — determines whether you need GBDT, a sequence model, a GNN, or a vision model |
| 10 | **How fast does the adversary adapt relative to my retrain cadence?** | If the adversary is faster, you are always fitting a distribution that has already moved. Changes the whole strategy |
| 11 | **How many analysts do I have, and what does one review cost?** | `review` capacity is a hard constraint that sets a threshold. It is arithmetic, not preference |
| 12 | **What is my per-decision cost budget?** | $0.0008 forbids a large ensemble on every event, and forces the cascade |
| 13 | **Can I buy this signal instead of learning it?** | Consortium and device-intelligence data often beat any model improvement. §3.6 |
| 14 | **What must I be able to reproduce, and for how long?** | Seven years of decision reproducibility constrains model artefacts, feature logging and even library versions |

> **The single question most often skipped is #4.** Your training data contains only transactions you approved. The model that generated that selection is baked into your labels. Every offline metric you compute is conditioned on your own past behaviour, and the effect is not small — it is typically the largest source of error in a mature fraud model's offline/online gap. It is treated in §4.2.

---

# PART 1 — The Use Case, Extended

I keep the source document's reference company so the two texts compose: a **marketplace with an embedded payments platform**, 40M MAU, ~$40B annualised GMV, 120M scored events/day, 1,400 eps baseline / 8,000 peak, 80 ms p99 decision budget, ≤$0.0008 per decision.

What the source document does not supply — and what the modelling work requires — are the **volumes, base rates and unit economics per fraud class.** Here they are, as a discovery output.

## 1.1 Event and volume decomposition

| Event type | Volume/day | Decision point | Latency class |
|---|---|---|---|
| Payment authorisation | 1.6M | Inline | ≤80 ms p99 |
| Login | 18M | Inline | ≤50 ms p99 |
| Signup | 0.4M | Inline | ≤120 ms (tolerant) |
| Listing / message / review | 12M | Inline + async | ≤80 ms |
| Payout request | 0.09M | Near-real-time | seconds |
| Platform / bot events | 88M | Inline, very cheap | ≤10 ms |
| **Total** | **~120M** | | |

## 1.2 The economics per fraud class — the table the source document implies but never computes

| Class | Attempted rate | Loss per successful event | Cost of one false positive | **FN : FP** | **Operating regime** |
|---|---|---|---|---|---|
| **Payment fraud** | 1.4% of auths | **$210** (goods + $25 chargeback fee + ops) | **$89** = $11 contribution margin + 0.23 × $340 residual LTV of a churned buyer | **2.4 : 1** | **Near-symmetric → a calibration-and-threshold problem** |
| **Account takeover** | 0.06% of logins | **$640** (drained balance, remediation, support, churn) | **$1.80** (step-up challenge: 4% abandon × $11 + friction) | **355 : 1** | **Recall-dominant → challenge liberally** |
| **Promo / bonus abuse** | 3.1% of signups | **$25** (credit farmed) | **$204** = 0.6 × $340 LTV of a blocked genuine new user | **1 : 8** | **Precision-dominant → the exact opposite** |
| **Seller collusion / bust-out** | 0.02% of sellers | **$18,400** (aggregate before detection) | **existential for the seller**; modelled at $9,200 (churn + reputational + dispute) | **2 : 1** | Symmetric but **enormous stakes both ways → review-dominant** |
| **AML / money movement** | unknown by construction | **regulatory**; internal risk price $4.1M per material finding | de-banking a legitimate user: $340 LTV + complaint + potential discrimination exposure | **effectively ∞ : 1** *within a capacity constraint* | **Coverage-complete, conservative, explainable** |
| **Bots / scraping** | 6% of platform events | $0.004 (infra + inventory distortion) | $0.02 (throttled power user) | **1 : 5** | **Cheap-per-event → optimise cost, not accuracy** |

> ### The insight the whole document turns on
> **The false-negative-to-false-positive ratio spans nearly four orders of magnitude across six classes that share one platform, one feature store and one team.**
>
> Account takeover wants you to challenge on the faintest suspicion. Promo abuse wants you to be nearly certain before you block. Payment fraud sits almost exactly in the middle, which is precisely why it is a *threshold* problem rather than a *recall* problem — and why fraud practitioners who came from AML or from bot defence get payments wrong in opposite directions.
>
> **You cannot express this with one model, one threshold, or one metric.** The source document's Chapter 0 says the six classes "do not share a serving path." The stronger statement is: **they do not share a loss function, so they cannot share a model, a threshold, an evaluation metric, or a definition of "good."** They share a *feature platform* and a *policy engine*. That is the correct boundary, and it is worth saying in the first two minutes of any interview.

## 1.3 Analyst capacity — the constraint that sets the `review` band

| | Figure |
|---|---|
| Fraud analysts | 180, loaded cost **$58/hr** |
| Productive hours/day | 7.5 |
| Payment case handling time | 7 min → **64 cases/analyst/day** |
| **Total capacity** | **11,520 case-slots/day** |
| Allocation | 6,200 payment + ATO · 2,800 seller/collusion · 1,900 AML · 620 promo |
| **Cost per reviewed case** | **$6.77** |
| Analyst accuracy (payment) | 91% agreement with eventual chargeback outcome |
| Analyst decision value | Converts a P(fraud)=0.35 uncertain case into a 0.91-accurate decision |

**This is the constraint that makes `review` a real action rather than a wish.** At 1.6M auths/day, a review band of even 1% would generate 16,000 cases against 6,200 slots. The review band is therefore **not chosen; it is solved for** (§2.4).

## 1.4 Label arrival — the property that shapes every modelling decision

| Class | Label source | Median arrival | 90th percentile | Censoring |
|---|---|---|---|---|
| Payment fraud | Chargeback (network dispute) | **34 days** | **97 days** | Declines never labelled; ~19% of true fraud never disputed |
| ATO | Victim report + forensic review | 6 days | 41 days | Silent takeovers never reported |
| Promo abuse | Downstream behaviour heuristics | 21 days | 60 days | **Often no hard label at all** |
| Seller collusion | Investigation outcome | 74 days | 210 days | Only what analysts looked at |
| AML | SAR filed / regulator feedback | 90+ days | never | **The label is an opinion** |
| Bots | Immediate signals + honeypots | minutes | hours | Best-labelled class by far |

> **Two structural consequences that no amount of architecture fixes:**
> 1. **At any moment your most recent 90 days of data are partially unlabelled.** A model trained naively on "everything up to today" learns that recent transactions are safe — because their chargebacks haven't arrived yet. This single bug has shipped in production at more than one major payments company. §4.1.
> 2. **You only observe outcomes for what you approved.** Your labels are the output of your own policy. §4.2.

---

# PART 2 — The Economics, Computed
### *The source document writes the EV equations. This section solves them.*

## 2.1 From equations to numbers

The source document gives the four actions and their expected values. Instantiate them for payment fraud with the §1.2 constants:

```
Let p = P(fraud | features), L = $210 (fraud loss), M = $89 (cost of a wrong decline)
Challenge:  fraud pass-through rate f = 0.11 (fraudsters defeat step-up 11% of the time)
            good-customer abandon rate a = 0.07
Review:     analyst accuracy q = 0.91, cost c = $6.77, delay cost d = $1.20

EV(allow)     = −p·210
EV(deny)      = −(1−p)·89
EV(challenge) = −p·0.11·210 − (1−p)·0.07·89      = −23.1p − 6.23(1−p)
EV(review)    = −[p·(1−q)·210 + (1−p)·(1−q)·89] − 6.77 − 1.20
              = −(18.9p + 8.01(1−p)) − 7.97
```

**Take the argmax at each p** — and note that the answer is a *partition of the probability line into four regions*, not a threshold:

| p range | Best action | EV at the boundary | Why |
|---|---|---|---|
| p < **0.031** | **ALLOW** | −$6.5 | Challenge friction exceeds expected fraud loss |
| 0.031 – **0.28** | **CHALLENGE** | | Cheap, and it converts most fraud without touching good customers |
| 0.28 – **0.71** | **REVIEW** | | Analyst accuracy is worth $6.77 + $1.20 only in the genuinely uncertain middle |
| p > **0.71** | **DENY** | −$25.8 | Expected loss exceeds the cost of a wrong decline |

> ### This is the single most important table an ML architect can produce in this domain, and here is why.
>
> **First, it proves the model must be calibrated.** Every boundary above is a probability. A model with 0.94 AUC and a score that is not a probability cannot be plugged into this at all — you would be comparing an arbitrary ranking to a dollar figure. AUC is invariant to any monotone transform of the score; **the entire economics of this system lives in exactly the information AUC discards.** That is the argument for §4.7, and it is why "we improved AUC by 1.2 points" can accompany a *worse* P&L.
>
> **Second, it shows where the money is.** A binary allow/deny system is forced to pick one boundary somewhere in [0.031, 0.71] and eat either friction or losses across the whole band. The intermediate actions recover most of the available value — at these constants, roughly **$41M/yr of the total opportunity** sits in the challenge and review bands. The source document says "challenge and review are architecture, not modelling." Correct, and now quantified.
>
> **Third, the boundaries move with the constants, and the constants are business inputs.** `M` depends on contribution margin and churn — a growth team's data. `q` and `c` depend on analyst staffing — an operations team's data. `f` depends on your step-up vendor. **Your decision boundaries are owned by four different departments**, which is exactly why the source document's V20 five-term dashboard matters.

## 2.2 The boundaries are not constant — segment them, and derive by how much

`M`, the cost of a wrong decline, is not $89 for everyone. Recompute per segment:

| Segment | Contribution margin | P(churn on decline) | Residual LTV | **M** | Deny boundary |
|---|---|---|---|---|---|
| New buyer, first order | $6 | 0.61 | $410 | **$256** | **p > 0.85** |
| Tenured buyer (>2 yr, 40+ orders) | $14 | 0.09 | $780 | **$84** | p > 0.71 |
| Guest checkout, low value | $4 | 0.31 | $95 | **$33** | **p > 0.49** |
| High-value order (>$600) | $71 | 0.18 | $780 | **$211** | p > 0.83 |
| Seller payout | $0 | 0.55 | $9,200 | **$5,060** | **p > 0.98** |

> **The deny boundary varies by a factor of two across ordinary segments and by a factor of twenty when a seller payout is involved.** A single global threshold is therefore leaving money on the table in both directions simultaneously — over-declining new buyers whose churn cost is highest, and under-declining guest checkouts where it barely matters.
>
> And note what this does *not* require: it does not require a segment-specific *model*. It requires **one calibrated model and a segment-aware policy layer**. That is the source document's V9 separation, and this table is the proof of why it earns its complexity. **Segmentation belongs in the policy, not the model, whenever the segmentation is about *cost* rather than about *behaviour*.** That single rule saves teams from a proliferation of unmaintainable segment models.

## 2.3 What calibration is worth, in dollars

Teams argue about calibration as a technical nicety. Price it.

Suppose the model is systematically over-confident in the 0.2–0.4 band by a factor of 1.6 — a very ordinary miscalibration after a class-rebalanced training run that nobody corrected. Then transactions whose true `p` is 0.19 are scored 0.30 and routed to **REVIEW** instead of **CHALLENGE**.

```
Volume in the affected band:            ~2.1% of 1.6M auths/day  =  33,600/day
Misrouted to review instead of challenge
Cost delta per event: EV(review) − EV(challenge) at p=0.19  ≈  −$11.4 vs −$9.5  =  $1.9
Direct cost:  33,600 × $1.9 × 365                            =  $23.3M/yr
```

And that is only the *direct* cost. The 33,600/day also **exhausts the 6,200-slot analyst capacity by 5.4×**, which means the genuinely uncertain 0.28–0.71 cases queue behind them and are auto-actioned by the overflow policy. **The second-order cost is larger than the first.**

> **Calibration is not a modelling refinement. At this scale a single unaddressed miscalibration is a nine-figure-adjacent operating error, and it is completely invisible to AUC, to precision, to recall, and to every dashboard a typical team runs.** Monitor Expected Calibration Error per segment per day, alert on it, and gate releases on it. §4.7.

## 2.4 Solving for the review band from analyst capacity

The review band in §2.1 was derived from EV alone and yields:

```
Volume with 0.28 < p < 0.71  ≈  1.9% of auths  =  30,400 cases/day
Available payment/ATO capacity                 =   6,200 slots/day
```

**Five times over capacity.** So the EV-optimal band is not feasible, and the correct formulation is a **capacity-constrained allocation**:

```
maximise   Σ_{i ∈ reviewed} [ EV(review_i) − EV(best automated action_i) ]
subject to |reviewed| ≤ 6,200
```

Sort by **value of review** = `EV(review) − max(EV(allow), EV(challenge), EV(deny))`, and take the top 6,200. That quantity peaks near p ≈ 0.5 (maximum uncertainty) and is scaled by the transaction's exposure. Which gives you:

> **Route to review by `uncertainty × exposure`, not by score.** A $40 order at p=0.5 and a $4,000 order at p=0.5 are equally uncertain and hundred-fold different in value-of-review. Ranking the review queue by score alone — which is what almost every case-management system does out of the box — wastes the scarcest resource in the whole operation.

Practical result at these numbers: the feasible review band becomes roughly `0.36 < p < 0.63` **weighted by exposure band**, with everything else routed to challenge or deny. And this immediately gives operations a lever they can price: **the 6,201st analyst is worth the marginal value-of-review at the cut point** — about $19/case at current staffing, against a $6.77 cost. **You are capacity-constrained, not judgement-constrained, and hiring is positive-EV until that marginal value falls to $6.77.** That is an argument an operations VP can act on, produced by an ML architect from a model's score distribution.

## 2.5 What a label is worth — and therefore how big the exploration holdout should be

The source document says the exploration holdout should be introduced "immediately — from the first model." Correct. But *how big*? Derive it, because this is one of the few places where you deliberately lose money to buy information, and it needs a defensible number.

**The cost of a holdout:** you approve transactions you would have declined.
```
Holdout at score band s:  cost = n·P(fraud|s)·$210 − n·(1−P(fraud|s))·(recovered margin)
```

**The benefit:** unbiased labels in the region where your model is most uncertain and most consequential — which is the only way to correct the selection bias of §4.2 and the only way to know whether `P(fraud|s)` is still what you think it is.

**Size it from statistical power, not from instinct:**
```
Requirement: detect a 3pp shift in P(fraud|s) within a score decile, 80% power, α=0.05
At P≈0.45 in the top declined decile: n ≈ 2 × 7.85 × 0.45 × 0.55 / 0.03²  ≈  4,320 per decile per window
Across the 3 deciles that matter, per quarter                              ≈  13,000 approvals
As a fraction of declines (≈24,000/day)                                    ≈  0.6% of declines
Direct cost: 13,000 × 0.42 avg P(fraud) × $210 − recovered margin           ≈  $1.02M/yr
```

> **$1.02M/yr, derived, defensible, and cheap against a $44M annual fraud loss and a $23M calibration exposure.** Present it that way and it gets approved. Present it as "we'd like to let some fraud through for science" and it never does.
>
> And two refinements that matter: **stratify the holdout across score bands and segments** (a uniform random holdout wastes most of its budget on the obvious), and **cap exposure per holdout transaction** so the tail is bounded — you want information, not a $9,000 lesson.

## 2.6 Retraining cadence, derived

```
Measured decay:  offline-replay AUC falls ~0.004/week in steady state;
                 P&L impact ≈ $71k/week of accumulated loss at week n
Cost of a retrain cycle: compute ($2.4k) + validation engineering (0.6 wk)
                         + independent model validation (2.1 wk elapsed under MRM)
                         + release risk
Total loaded cost per cycle ≈ $58k, elapsed ≈ 3 weeks
```

Optimal scheduled cadence lands near **every 4 weeks** — the point where accumulated decay cost equals cycle cost. But:

> **Adversarial decay is punctuated, not smooth.** A new attack technique does not degrade you at $71k/week; it degrades you at $400k/week starting on a Tuesday. **A purely scheduled cadence is therefore wrong on its own.** The correct design is *scheduled cadence + event-triggered retrain*, where the trigger is a drift signal (§5) and there is a **pre-approved fast path through model risk management for a like-for-like refit** — same features, same architecture, new data. Negotiating that fast path with the MRM function is one of the highest-leverage things an ML architect does in a regulated environment, and it is a political task, not a technical one.

## 2.7 The five-term P&L, made concrete

The source document's V20 names the five terms and observes that nobody owns the sum. Here is what the dashboard actually contains at this scale:

| Term | Current annual | Owner | Moves when |
|---|---|---|---|
| Fraud losses | **$44.0M** | Fraud | Threshold ↓, model ↑, attack ↑ |
| Friction cost (declined good + abandoned challenges) | **$61.3M** | Growth / Payments | Threshold ↑, challenge rate ↑ |
| Operations (analyst) | **$16.4M** | Operations | Review band ↑, handling time ↑ |
| Infrastructure | **$28.6M** ($0.00065 × 120M × 365) | Platform | Cascade efficiency, model size |
| Regulatory / remediation provision | **$9.0M** | Compliance | Coverage gaps, fairness findings |
| **Total** | **$159.3M** | *nobody* | |

> **Friction cost is larger than fraud losses, and it is invisible on the fraud team's dashboard.** That is the entire organisational pathology in one line. The fraud team is measured on $44M and controls a lever that moves $61M on someone else's P&L. Every over-tightening incident in the history of this industry comes from that asymmetry.
>
> **The architect's move:** make the friction term *measurable* — which requires the exploration holdout (§2.5) to estimate the counterfactual approval outcome — and then insist that **the threshold-setting authority sits with whoever owns the sum.** This is a design decision about *who can change a number*, and it belongs on the architecture diagram exactly as much as the feature store does.

---

# PART 3 — The Model Portfolio
### *Fourteen classes, what each is genuinely for, and the business trigger that earns it a place*

## 3.0 The governing principle

> **Signal shape determines model class. Latency budget determines where it runs. Regulation determines what it may be. Business trigger determines when it arrives.**
>
> Not: "GBDT is the industry standard." Not: "we should try deep learning."

And a second principle that saves more money than any other statement in this document:

> **In tabular fraud, gains come from features, not from learners.** A well-tuned gradient-boosted tree on good features beats a novel architecture on the same features, reliably and repeatedly. The published wins attributed to deep learning in fraud almost always came from *new inputs* — sequences, graphs, text, device telemetry — not from a better function approximator on the same 350 columns.
>
> **Therefore: use representation learning to manufacture features offline; keep the inline decision model a boosted tree.** This one pattern gives you the accuracy of deep models, the latency of trees, the explainability the regulator wants, and an ops burden your team can carry. It is how the best fraud shops actually run, and very few documents say it plainly.

## 3.1 The model class menu

| # | Class | What it is genuinely for | Inline-viable? | Explainable? | Data need | Earns its place when |
|---|---|---|---|---|---|---|
| 1 | **Expert rules** | Encoding known-certain patterns; instant response to a live attack; regulatory hard-stops | Yes, <1 ms | Perfectly | None | **Always.** Rules never leave; they become the fast path and the guardrail |
| 2 | **Logistic regression (WOE/binned)** | The regulated baseline; adverse-action reason codes fall out naturally | Yes, <1 ms | Perfectly | ~2k positives | Regulated decisions; small data; a benchmark you must always keep |
| 3 | **GAM / Explainable Boosting Machine** | Non-linear per-feature shape with additive decomposition; **monotonic constraints** | Yes, ~2 ms | Very | ~10k positives | You need non-linearity *and* a defensible per-feature story. Underused |
| 4 | **GBDT — XGBoost / LightGBM / CatBoost** | **The workhorse of tabular fraud** | Yes, 8–18 ms | Via SHAP/surrogates | ~20k positives | The moment feature interactions matter more than a single-feature story |
| 5 | **Random forest** | Robustness, variance reduction, quick baselines | Yes | Weakly | ~20k | Rarely optimal. Useful as a stability check on GBDT |
| 6 | **Stacked ensembles** | Squeezing the last 1–2 points | Marginal | Poorly | Large | **Usually not worth it inline.** Cost, latency and MRM burden exceed the gain |
| 7 | **Isolation Forest / LOF / one-class SVM** | **Novelty detection where no labels exist yet** | Yes, cheap | Weakly | Unlabelled | New attack surfaces, new markets, promo abuse at launch, cold-start products |
| 8 | **Autoencoder / VAE reconstruction error** | Unsupervised anomaly over high-dimensional behaviour | Offline → feature | No | Large unlabelled | When the anomaly lives in the joint distribution of many weak signals |
| 9 | **Clustering (HDBSCAN, connected components)** | **Ring and mule-network discovery**; bulk account farms | Offline | Structurally | Unlabelled | You have confirmed organised rings that per-event features cannot catch |
| 10 | **Graph algorithms (PageRank, label propagation, community detection)** | Turning relational structure into **tabular features** for the GBDT | Precomputed | Yes (path is the evidence) | Entity graph | The source doc's V7 — the correct default for graph signal |
| 11 | **Graph neural networks (GraphSAGE, GAT)** | Learned relational representations where hand-designed graph features plateau | **Rarely inline** — precompute embeddings | Poorly | Large labelled graph | Only after graph features plateau *and* you can serve embeddings. §3.4 |
| 12 | **Sequence models (GRU / Transformer over event sequences)** | Behavioural signal in *order and timing* — session dynamics, ATO, bust-out trajectories | Offline → embedding | Poorly | Long histories | When the signal is "how this unfolded," not "what this is". §3.3 |
| 13 | **Deep tabular (TabNet, FT-Transformer, NODE)** | — | Marginal | No | Very large | **Almost never.** Has not reliably beaten tuned GBDT on tabular. Treat claims sceptically |
| 14 | **LLMs** | Investigation narrative, rule drafting, evidence summarisation, analyst assist | **Never in the decision path** | N/A | — | The source document's V21. Assistance and documentation, not decisioning |

**Supporting techniques that are not model classes but decide outcomes:**

| Technique | For |
|---|---|
| **PU learning / reject inference** | Learning when negatives are unreliable and declines are unlabelled (§4.2) |
| **Survival analysis on label maturation** | Projecting final chargeback rate from partial observation (§4.1) |
| **Isotonic / Platt / beta calibration** | Turning a score into a probability (§4.7) |
| **Conformal prediction** | Distribution-free uncertainty bands — genuinely useful for routing to `review` |
| **Contextual bandits** | Optimising *treatment* (which challenge type) rather than the risk score |
| **Two-tower / embedding retrieval** | "Is this entity similar to known confirmed fraud?" as a feature |
| **RuleFit / tree-extracted rules** | Converting a model's learned interactions into deployable, auditable rules |

## 3.2 The maturity ladder — what to deploy when, tied to a *business trigger*

This is the model-layer analogue of the source document's Appendix C, and it follows the same discipline: **each row is a trigger, not a stage to be reached on schedule.**

| Stage | **Business trigger (not a milestone)** | What you deploy | Why *this* and not more | What it costs you |
|---|---|---|---|---|
| **M0** | Losses are visible and rules can express them | **Expert rules + velocity counters** | Zero label requirement; hours to change; perfectly auditable | Rule sprawl; analysts become the model |
| **M1** | Rule false-positive cost exceeds fraud cost, *or* you observe interactions rules cannot express | **Logistic regression on ~40 engineered features** | ~2k positives is enough; reason codes are native; MRM approves it easily; establishes the calibration discipline early | Misses interactions; needs manual feature work |
| **M2** | LR under-fits measurably — a GBDT beats it by >3pp PR-AUC in offline replay on a *held-out future window* | **GBDT, single model, calibrated** | The workhorse. Handles interactions, missingness, mixed types | SHAP infrastructure; MRM burden; overfitting to a moving adversary |
| **M3** | Analysts report attacks that "look normal per transaction" | **Streaming velocity + entity aggregate features** into the same GBDT | Feature work, not model work — **cheapest large gain available** | Stream infrastructure (the source doc's V6) |
| **M4** | Confirmed organised rings that per-entity features demonstrably miss | **Entity graph → precomputed graph features** into GBDT | Path evidence is explainable and citable; 10 ms budget respected | Entity resolution is now a system you own (V7) |
| **M5** | A new surface launches, or attacks appear that have no labels yet | **Unsupervised novelty layer running in parallel**, output as a *feature* and an *analyst queue* — never as an autonomous decision | Buys you detection during the label vacuum | High false-positive rate; must not be given deny authority |
| **M6** | Per-decision cost is a material budget line **and** the score distribution is genuinely bimodal | **Model cascade**: cheap model on all traffic, expensive model on the uncertain middle | The source doc's V16 — often 60–75% cost reduction | New attack surface: adversaries probe the cheap tier. §4.12 |
| **M7** | Session/behavioural signal is demonstrably present — ATO, bust-out, bot sophistication rising | **Sequence model trained offline → embedding served as features** | Keeps inline path fast and tree-based | A second training pipeline, a second drift surface |
| **M8** | Hand-designed graph features plateau **and** ring topology is evolving faster than you can hand-engineer | **GNN → node embeddings, refreshed nightly, served as features** | Learned structure without inline traversal | Real ops burden; poor explainability; **defer as long as possible** |
| **M9** | Segments differ *behaviourally* (not merely in cost) and each has ≥500 positives per retrain window | **Segment models** (e.g. card-present vs CNP vs payouts) | Genuine distributional difference | Model count multiplies; MRM inventory grows; **most teams do this far too early** |
| **M10** | Treatment effectiveness varies and you can measure it | **Contextual bandit on treatment choice** (which challenge, which step-up) | Optimises the *action*, which is where remaining value is | Exploration cost; interaction with the policy layer |

> **The two rows people get wrong.**
> **M9 (segment models) is almost always premature.** Teams split by segment because it feels rigorous. §2.2 showed that cost-based segmentation belongs in the *policy*, and behaviour-based segmentation needs ≥500 positives per segment per retrain window — at a 0.14% base rate that is 357,000 transactions per segment per window. Most proposed segments do not clear it, and a segment model fitted on 90 positives is worse than the global model with a segment feature. **Add the segment as a feature first; split only when the global model with that feature is measurably worse than a dedicated one on a held-out future window.**
>
> **M8 (GNN) is the most over-reached row in the table.** Hand-designed graph features (shared-device counts, component size, k-hop distinct-card counts, community risk score) capture most of the available relational signal at a fraction of the cost and with path-level explainability a regulator accepts. Reach for a GNN only when you can show a plateau *and* you have somewhere to serve embeddings *and* you have an answer for "explain this to the customer."

## 3.3 Where sequence models genuinely earn their place

The signal in fraud is sometimes in the *row* and sometimes in the *trajectory*. Sequence models are for the second case, and the test is concrete: **can a human analyst tell fraud from legitimate by looking at a single event, or do they need to see the sequence?**

| Fraud class | Signal location | Verdict |
|---|---|---|
| Payment fraud (stolen card) | Mostly the row + velocity aggregates | GBDT + aggregates. Sequence adds ~1–2pp — real but not the first investment |
| **Account takeover** | **The trajectory** — login → change email → change payout → drain | **Sequence model earns its place clearly.** The individual events are all legitimate-looking |
| **Bust-out / seller collusion** | **The trajectory over weeks** — reputation building then sudden behaviour change | **Yes.** This is a change-point problem in disguise |
| Promo abuse | Signup burst patterns + graph | Graph + aggregates usually sufficient |
| Bots | Timing micro-structure, inter-event intervals | Yes, and cheaply — often a small model on inter-arrival statistics |

**The production pattern that works:** train a GRU or a small causal Transformer over the user's last *k* events offline; extract a fixed-length embedding; **write the embedding to the online feature store on a 5-minute cadence; serve it as 32–64 additional columns to the inline GBDT.** You get the sequence signal inside the 18 ms inference budget because the sequence model never runs inline.

**The cost you accept:** the embedding is stale by up to five minutes, which is fine for ATO trajectory (which unfolds over minutes to hours) and useless for card-testing (which unfolds in seconds — that is what the ≤2 s velocity features are for). **Match the feature's refresh cadence to the phenomenon's timescale**, which is exactly the source document's freshness tiering applied to learned features.

## 3.4 Graph: features first, embeddings later, GNN reluctantly

| Approach | What you get | Latency | Explainability | When |
|---|---|---|---|---|
| **Hand-designed graph features** — component size, shared-device card count, k-hop distinct entities, community fraud rate, days-since-component-first-seen | 70–85% of available relational signal | Precomputed, 10 ms lookup | **Path is the evidence** — an analyst can see it | **Start here. Often finish here.** |
| **Graph embeddings (node2vec / metapath2vec)** | Latent structural position | Nightly precompute | Poor | When topology matters beyond counts |
| **GNN (GraphSAGE / GAT)** | Learned neighbourhood aggregation, inductive to new nodes | Nightly embedding refresh | Very poor | Genuine plateau + ops capacity |
| **Live multi-hop traversal at decision time** | Freshest possible | **Blows the budget** | Yes | **Never inline.** Analyst tooling only |

> **The explainability argument is decisive in a regulated setting and is usually left out of this comparison.** When a customer disputes a decline, "your account is two hops from a device that touched fourteen cards flagged for fraud in the last 30 days" is a defensible statement with an evidentiary path. "Your GraphSAGE embedding was in a high-risk region of latent space" is not a statement you can put in an adverse-action notice. **If any part of your decision must be explained, graph features beat graph embeddings for reasons that have nothing to do with accuracy.**

## 3.5 The unsupervised layer — necessary, and dangerous if misused

Every mature fraud platform runs unsupervised detection. Almost every one that gets it wrong makes the same mistake: **giving it decision authority.**

**What it is for:**
1. **The label vacuum.** A new market, a new product, a new attack. Labels arrive in 34–97 days; the attack does damage today.
2. **Novelty as a feature.** Reconstruction error or isolation score fed into the supervised model is a cheap, powerful column that captures "unlike anything I've seen" — a thing supervised models are structurally bad at.
3. **Analyst work generation.** Anomaly clusters make excellent investigation queues, which generate labels, which feed the supervised model. **This is the loop that converts unsupervised detection into supervised accuracy.**

**What it is not for:** autonomous deny decisions. At the base rates here, an isolation forest tuned to catch 60% of fraud flags 4–8% of traffic. At 1.6M auths/day that is 64,000 wrongful challenges daily, costing more in friction than the fraud it catches. **Precision, not recall, is the binding constraint on any autonomous action, and unsupervised methods do not have it.**

> **The correct wiring, stated once:** unsupervised layer → (a) a numeric feature into the supervised model, (b) a ranked analyst queue, (c) an alerting signal on *cluster emergence*. Never → the policy engine.

## 3.6 Build vs buy — the three signals you should not try to learn

This is absent from most ML system-design material and it is one of the highest-ROI judgements an architect makes.

| Signal | Why you cannot learn it internally | Typical cost | Verdict |
|---|---|---|---|
| **Consortium / network fraud data** — has this card, device or email been seen in confirmed fraud *at other merchants* | You only see your own traffic. A card tested at 30 merchants and used at yours looks clean to you and obvious to a consortium | $0.008–0.03/query | **Buy.** Often the single largest available lift, and no amount of modelling substitutes for data you do not have |
| **Device intelligence / fingerprinting** | Requires SDK-level telemetry, emulator and farm detection, and an adversarial arms race that is a full product in itself | $0.01–0.04/query | **Buy** unless device signal is your core differentiator |
| **Identity / KYC verification, document authenticity** | Vision models on identity documents, plus issuing-authority data you cannot access | $0.30–2.00/check | **Buy**, and call it only in the high-risk band |

**And here is where it becomes an architecture decision rather than a procurement one.** At $0.02/query against a $0.0008 all-in per-decision budget, you **cannot** call a consortium API on every event — it would be 25× your entire budget. So:

> **Third-party enrichment is a cascade tier.** Call it only for the uncertain middle. At these numbers: call the consortium on the ~9% of events whose cheap-tier score lands in the ambiguous band → 144,000 calls/day × $0.02 = **$2,880/day = $1.05M/yr**, against a measured fraud-loss reduction typically in the $6–11M range. **A 6–10× return on a line item that a naive design would have declared unaffordable.**
>
> This is the same shape as the source document's V16 model cascade, and it is worth generalising: **the cascade is not only about model cost. It is the mechanism by which any expensive signal — a big model, a paid API, a human — becomes affordable by being applied selectively.** State it that way and the cascade stops being an optimisation and becomes an architectural pattern.

## 3.7 The portfolio view — because you will run twelve models, not one

A mature version of this platform runs, simultaneously:

| # | Model | Class | Path | Retrain | Owner |
|---|---|---|---|---|---|
| 1 | Payment risk — champion | GBDT | Inline | 4 wks | Fraud DS |
| 2 | Payment risk — challenger | GBDT | Shadow | 4 wks | Fraud DS |
| 3 | Payment risk — cheap tier | LR / shallow GBDT | Inline, all traffic | 8 wks | Fraud DS |
| 4 | ATO risk | GBDT + sequence embedding | Inline (login) | 4 wks | Account Security |
| 5 | Sequence encoder | GRU/Transformer | Offline → features | 8 wks | Fraud DS |
| 6 | Promo abuse | GBDT | Inline (signup) | 6 wks | Growth Integrity |
| 7 | Bot / automation | Small GBDT + timing features | Inline, ultra-cheap | 2 wks | Platform Abuse |
| 8 | Graph feature pipeline | Graph algorithms | Nightly → features | 12 wks | Fraud Platform |
| 9 | Novelty detector | Isolation Forest / AE | Parallel → feature + queue | 12 wks | Fraud DS |
| 10 | Seller / collusion risk | GBDT on graph + sequence | Near-real-time | 8 wks | Seller Risk |
| 11 | AML monitoring | Rules + GBDT triage | Batch | 26 wks (MRM-heavy) | Financial Crime |
| 12 | Calibrator(s) | Isotonic per segment | Inline | **weekly** | Fraud DS |

> **Three consequences that only become visible at the portfolio level, and that no single-model discussion surfaces:**
>
> **(a) The calibrator retrains 4× more often than the model.** Calibration drifts faster than discrimination — the ranking stays good while the probabilities go stale. Recalibrating without refitting is cheap, fast, low-risk, and usually skips heavy MRM as a parameter update rather than a model change. **It is the highest-frequency, lowest-cost lever in the portfolio, and most teams do not have it as a separate artefact.**
>
> **(b) Twelve models means twelve MRM inventory entries, twelve drift dashboards, twelve retrain calendars and twelve owners.** Model count is an *operating cost*, not a capability. Every proposed thirteenth model should be asked to displace one.
>
> **(c) Stagger the retrain calendar deliberately.** If models 1, 4, 6 and 10 all refit on the same features in the same week, a feature-pipeline bug propagates to your entire estate simultaneously and you have no unaffected control. **Stagger it so that at any moment something is running on last month's features and can act as a canary.**

---

# PART 4 — Modelling Stage Deep Dives
### *Internals → strategy menu → how an architect selects → our choice → evals → drift signals*

---

## 4.1 Label Engineering — the hardest problem, and the one nobody interviews on

**The decision this stage makes:** *"What am I actually predicting, and do I know the answer yet?"*

### Internals
Three distinct pathologies, routinely conflated:

**(a) Delay.** Chargebacks arrive at a median of 34 days, p90 of 97. Train on data up to today and the last three months look artificially clean. The model learns *recency implies safety* — a feature it will happily exploit, because in your training data it is true.

**(b) Censoring.** ~19% of true payment fraud is never disputed (small amounts, unengaged cardholders, expired dispute windows). Those are labelled negative. Your "negatives" are a mixture of genuine negatives and undisputed fraud.

**(c) Bias.** Declined transactions produce no outcome at all. Your labelled population is the population *you chose to approve*. §4.2.

### Strategy menu

| Strategy | Mechanics | Wins | Fails |
|---|---|---|---|
| **Maturity cut-off** | Train only on data older than the p95 label window (97 days) | Simple, correct, universally the right starting point | **Throws away your three most recent months — against an adversary who changes monthly, that is the most valuable data you have** |
| **Survival / hazard model on label arrival** | Model chargeback timing; reweight partially-observed windows by the probability that the label is still coming | **Recovers the recent window** — the highest-value modelling technique in this whole section | A second model to maintain and validate |
| **Inverse-probability weighting by observation window** | Weight each row by 1/P(label observed by now) | Cheap approximation of the above | Needs the hazard estimate anyway |
| **Proxy labels** — analyst disposition, victim report, early-warning network alerts | Available in days, not months | Fast feedback; enables weekly iteration | **Proxy ≠ outcome.** Analyst disposition is itself a model's output |
| **Multi-label / multi-task** | Predict chargeback AND analyst-flag AND early alert jointly | Uses all signals; the auxiliary tasks regularise | More complex; task weighting is another hyperparameter |
| **PU learning (positive-unlabelled)** | Treat negatives as unlabelled; estimate label frequency | Correct for censoring | Requires an estimate of the positive prior |
| **Label smoothing on the uncertain tail** | Soft labels on undisputed-but-suspicious | Reduces confident wrongness | Hurts calibration if applied carelessly |

### How the architect selects

1. **Fit the label-arrival hazard curve first.** It is a day of work and it tells you everything: your maturity cut-off, your safe evaluation window, and how much recent data you can recover with reweighting. At our numbers: 62% of chargebacks by day 34, 90% by day 97, 97% by day 140.
2. **Then decide how much recent data you need.** With a monthly-adapting adversary, a 97-day cut-off means training on a world three months out of date. **That is the argument for the hazard model** — it converts a 97-day blind spot into a reweighted 30-day one.
3. **Then decide your target.** "Chargeback within 120 days" and "analyst-confirmed fraud" and "any adverse outcome" are three different targets with three different base rates and three different business meanings. **Write down which one your threshold economics in §2.1 assumed** — because `L = $210` was derived from chargebacks, so the target must be chargebacks or the boundaries are wrong.

**Our choice:** primary target = chargeback-within-120-days; **hazard-reweighted training window extending to T−30 days**; auxiliary tasks for analyst disposition and early-warning alerts in a multi-task GBDT (shared features, separate objectives, weighted 0.7/0.2/0.1); PU correction on the undisputed tail; and a **strict rule that evaluation always happens on a fully matured window** even though training does not.

### Evals & drift

| Eval | Gate |
|---|---|
| Label maturity leakage test | Train with/without reweighting; the "days-since-transaction" feature must have near-zero importance |
| Hazard model calibration | Predicted vs actual label arrival by cohort, monthly |
| Proxy–outcome agreement | Analyst disposition vs eventual chargeback: track; ours is 0.91 |
| Evaluation window purity | 100% of eval data past p95 maturity |

| Drift signal | Diagnosis | Action |
|---|---|---|
| **Label arrival curve shifting** | Issuer dispute behaviour changed, or a network rule changed | **Refit the hazard model before retraining the risk model** — otherwise the reweighting is now wrong and silently corrupts training |
| Proxy–outcome agreement falling | Analysts drifting, or attack mix changed | Sample and distinguish; both are real and have opposite fixes |
| Undisputed-fraud estimate rising | Attackers moving to sub-dispute-threshold amounts | **A business signal**: your loss is under-measured, not just your labels |

> **This section is where the largest silent errors in production fraud models live**, and it is almost never asked about in interviews. Being able to say *"first I'd fit the label-arrival hazard curve, because it determines my training window, my evaluation window and how much of my recent data I can recover"* immediately separates you from candidates who say "we train on labelled fraud."

---

## 4.2 Selection Bias and Reject Inference

**The decision this stage makes:** *"What would have happened to the transactions I declined?"*

### Internals
You approve ~98.5% and decline ~1.5%. Only approvals generate outcomes. So `P(fraud | features, approved)` is what you learn, and `P(fraud | features)` is what you need. The gap widens every retrain cycle, because each model's declines shape the next model's training data. **This is a feedback loop that compounds, and it makes your model progressively blind in exactly the region where it is most confident.**

The symptom is characteristic and widely misdiagnosed: **offline metrics improve steadily while online performance stagnates.** Teams usually blame the feature pipeline.

### Strategy menu

| Strategy | Mechanics | Wins | Fails |
|---|---|---|---|
| **Ignore it** | Train on approvals only | Honest simplicity | Compounding blindness; the offline/online gap you cannot explain |
| **Exploration holdout** | Approve a stratified random sample of would-be declines | **Unbiased ground truth. The only true fix** | Costs real money (§2.5: ~$1.02M/yr) |
| **Reject inference — augmentation** | Score rejects with the current model, add as weighted pseudo-labels | Cheap | **Circular — it re-learns the current model's beliefs.** Use only with a holdout to anchor it |
| **Reject inference — parcelling** | Bin rejects by score, assign labels at the bad rate observed in the nearest accepted bin | Standard in credit risk | Assumes MAR within bins |
| **Heckman-style two-stage** | Model the selection process explicitly | Principled | Strong distributional assumptions; fragile |
| **Weighting by inverse propensity of acceptance** | Reweight accepted rows | Well-founded when propensity is known — **and you know it, because you set it** | High variance in low-propensity regions |
| **Use downstream signals on declines** | Retry behaviour, other-merchant outcomes via consortium, later confirmed fraud on the same entity | Free labels for *some* rejects | Partial coverage |

### How the architect selects
**The exploration holdout is not optional if you want your offline numbers to mean anything**, and §2.5 shows it costs about 2.3% of annual fraud losses to buy. Everything else on the menu is a way of *extending* the holdout's information to the rest of the reject population — none of them substitutes for it.

**Our choice:** stratified exploration holdout sized by power (§2.5) with a per-transaction exposure cap; inverse-propensity weighting anchored on the holdout; consortium-derived outcomes for declines where available; and a standing rule that **every offline metric is reported twice — on the accepted population and on the holdout-corrected population — with the gap itself tracked as a health metric.**

> **That gap is the single best early-warning indicator of a model that is confidently wrong.** When accepted-population AUC and holdout-corrected AUC diverge, you are optimising against your own past decisions. It should be on the front page of the model dashboard, and I have never seen it there.

---

## 4.3 Class Imbalance — mostly a distraction, with two exceptions

**The decision this stage makes:** *"Do I need to do anything special about a 0.14% base rate?"* Usually: less than you think.

### Strategy menu

| Strategy | Verdict for fraud |
|---|---|
| **Do nothing; use the natural distribution** | **Usually correct with GBDT.** Trees handle imbalance far better than the folklore suggests, and it preserves calibration |
| `scale_pos_weight` / class weights | Reasonable; **destroys calibration** — you must recalibrate afterwards, and teams forget |
| **Random undersampling of negatives** | Useful purely for *training-time economics* at 1.6M/day. Recalibrate with the known sampling rate |
| **SMOTE / synthetic oversampling** | **No.** Interpolating between fraud events in tabular space manufactures transactions that could not occur, and the adversarial setting makes the manufactured region meaningless |
| **Focal loss** | Marginal for tabular; borrowed from dense object detection where the imbalance is structurally different |
| **Two-stage: cheap recall filter, then precise model** | **Yes** — this is the cascade (§3.6), and it addresses imbalance as a *compute* problem, which is what it actually is |
| **Anomaly detection because "fraud is rare"** | Rare ≠ anomalous. Sophisticated fraud is designed to look normal. §3.5 |

**The two exceptions where imbalance genuinely bites:**
1. **Segment models (M9).** A global 0.14% is workable; a segment with 40 positives is not. This is the real constraint behind the §3.2 warning.
2. **Evaluation variance.** With 2,240 daily positives, daily metrics are noisy enough to trigger false drift alarms. Use rolling windows and confidence intervals, and see §4.10 for the power maths.

> **Say this in an interview:** *"Imbalance is mostly a compute and evaluation problem, not a learning problem — GBDTs handle 0.1% base rates fine. What I'd actually worry about is that any rebalancing destroys calibration, and calibration is where all the money is."* That inverts the expected answer and is correct.

---

## 4.4 Feature Engineering — where the gains actually are

**The decision this stage makes:** *"What does the model get to see?"* — and this, not the learner, determines performance.

### The feature families, by fraud class

| Family | Examples | Freshness | Class it serves |
|---|---|---|---|
| **Raw transaction** | Amount, currency, MCC, hour, channel, BIN attributes | Instant | All |
| **Entity profile** | Account age, order count, historical dispute rate, tenure | ≤5 min | All |
| **Velocity / counting** | Distinct cards per device in 600 s; auths per card in 60 s; declines-then-approve patterns | **≤2 s** | **Payment (card testing)** |
| **Deviation-from-self** | Amount vs the entity's own p95; hour vs their usual; new-geo flag | ≤5 min | Payment, ATO |
| **Graph / relational** | Component size, shared-device card count, k-hop distinct entities, community risk | ≤15 min | Collusion, promo, rings |
| **Sequence embedding** | Last-k-event encoder output | ≤5 min | **ATO, bust-out** |
| **Device / network** | Fingerprint reuse, emulator score, proxy/VPN, ASN reputation | Instant | ATO, bots, promo |
| **Third-party / consortium** | Cross-merchant fraud flags, identity confidence | Per-call | **Payment — the largest single lift** |
| **Novelty** | Isolation score, reconstruction error | ≤5 min | Emerging attacks |
| **Text / content** | Listing text, message content, review embeddings | Minutes | Seller fraud, content abuse |

### The three feature-design rules that matter most

1. **Ratios and deviations beat absolutes.** "Amount is $840" is weakly predictive. "Amount is 6.2× this account's 90-day p95, at an hour they have never transacted, from a country they have never used" is a signature. **Encode deviation-from-self explicitly** — trees can approximate it from raw features only with many splits and much more data.
2. **Design the counting window to match the attack's timescale.** Card testing is seconds. ATO is hours. Bust-out is weeks. **A single 24-hour window serves none of them well.** Ship a small grid — 60 s, 600 s, 1 h, 24 h, 7 d, 30 d — per key entity, and let the model choose. This is the concrete justification for the source document's freshness tiering.
3. **Every feature needs a null-and-late policy declared at design time.** In an 80 ms budget with a 66,000 reads/s store, some lookups will be slow or missing. **"Missing" and "zero" are different, and conflating them is a live incident** — a missing velocity counter that defaults to zero looks like a brand-new safe entity, which is precisely the fraudster's profile. Use explicit missingness (GBDTs handle it natively), never zero-fill, and **feed a `features_degraded` count into the policy layer** so the decision can be more conservative when it is less informed.

> **Rule 3 is the one that shows production scars.** The failure mode is: a Redis shard degrades, velocity features silently default to zero, every fraudster looks like a clean new user, and losses spike for forty minutes before anyone connects the outage to the fraud rate. Declare the null policy per feature, monitor the degraded-feature rate, and make the policy layer respond to it.

---

## 4.5 Model Architecture Selection and the Cascade

**The decision this stage makes:** *"How much compute does this event deserve?"*

### The cascade, designed properly

| Tier | Applies to | Model | Cost/event | Latency | Purpose |
|---|---|---|---|---|---|
| **T0 — rules** | 100% | Expert rules, hard-stops, allow-lists | ~$0.000001 | <1 ms | Certainty in both directions |
| **T1 — cheap** | ~99.4% remaining | Shallow GBDT, 60 features, no third-party calls | ~$0.00004 | 3 ms | Resolve the obvious majority |
| **T2 — full** | ~9% (the uncertain middle) | Full GBDT, 350 features, graph + sequence embeddings | ~$0.0009 | 16 ms | The real decision |
| **T3 — enriched** | ~1.2% | T2 + consortium + device intelligence + identity | ~$0.03 | 60 ms | Where the money is |
| **T4 — human** | ~0.39% (6,200/day) | Analyst | $6.77 | minutes–hours | Genuine ambiguity × high exposure |

```
Blended cost/decision  =  0.994×0.00004 + 0.09×0.0009 + 0.012×0.03 + 0.0039×6.77/1  ≈  ...
Excluding human review (an operations line, not infrastructure):     ≈ $0.00048
Against the $0.0008 budget                                            ✓ 40% headroom
Uniform T2-on-everything would cost:                                  $0.0009  ✗ over budget
Uniform T3-on-everything:                                             $0.03    ✗ 37× over
```

> **The cascade is what makes the expensive tiers affordable at all**, and note that the same structure carries model compute, paid data, *and* human attention. That generalisation is the useful part: **T0→T4 is one continuum of increasing cost per decision, and designing it is the same exercise at every tier.**

### The three cascade design rules

1. **Escalate on uncertainty, not on score.** Send to T2 the events where T1 is *unsure*, not the events T1 scores high. A confidently-high T1 score needs no second opinion; a middling one does. Escalation criterion: T1 score in the ambiguous band **or** T1's own uncertainty estimate is high **or** exposure is large.
2. **The cheap tier must never be able to deny alone.** It may allow and it may escalate. A wrongful deny from a 60-feature model has the same customer cost as one from the full model, without the evidence to defend it. **T1 has allow-and-escalate authority only.** This one rule prevents most cascade incidents.
3. **The cheap tier is an attack surface** (source doc V14). Adversaries who learn that low-value, simple-looking transactions never reach T2 will craft to stay in T1. Mitigations: randomised escalation of a small share (~0.5%) regardless of score; **never expose which tier decided**; and monitor the T1-resolved population's realised fraud rate as a first-class metric — a rise there is the signature of tier-gaming.

---

## 4.6 Calibration

**The decision this stage makes:** *"Is 0.4 actually 40%?"* — and §2.3 priced getting this wrong at $23M.

### Strategy menu

| Method | Mechanics | Wins | Fails |
|---|---|---|---|
| **None** | Raw model output | Fine only if the model is natively calibrated (well-tuned GBDT with logloss often nearly is) | Any rebalancing, any class weighting, any ensembling breaks it |
| **Platt scaling** | Logistic fit on scores | Simple, low-variance, needs little data | Assumes a sigmoid shape; poor on the tails, which is where your thresholds are |
| **Isotonic regression** | Monotone step fit | **Flexible, non-parametric, the usual production choice** | Needs more data; steps at the extremes; must be monotone-guarded |
| **Beta calibration** | 3-parameter, principled for probabilities | Better tail behaviour than Platt | Less familiar |
| **Per-segment calibration** | Separate calibrators by segment | **Necessary — miscalibration is segment-specific** | Data hungry; needs a fallback for thin segments |
| **Temperature scaling** | Single parameter | Cheap for neural nets | Too rigid for GBDT ensembles |
| **Conformal prediction** | Distribution-free coverage bands | **Excellent for `review` routing** — gives an honest uncertainty interval | Bands, not points; needs exchangeability, which an adversary violates |

### How the architect selects
**Always calibrate on a held-out, temporally-later, fully-matured window** — never on training folds and never on a random split, because calibration drifts with time and a random split hides exactly that. Calibrate **per segment** where volume allows (≥5,000 positives), with a pooled fallback.

**Our choice:** isotonic regression per segment, fitted weekly on the trailing matured window, deployed as a **separately versioned artefact** from the model.

> **Separating the calibrator from the model is one of the highest-value, least-known operational moves in this domain.** It means:
> - You can fix drifting probabilities **weekly** without retraining or re-validating the model.
> - Under model risk management, a calibrator refit on the same architecture is usually a **parameter update**, not a model change — a materially lighter approval path.
> - When your P&L moves, you can diagnose *discrimination* (has the ranking degraded?) separately from *calibration* (has the probability scale shifted?). **They have completely different causes and completely different fixes**, and a single "the model got worse" alert conflates them.

### Evals & drift

| Metric | Gate |
|---|---|
| **Expected Calibration Error**, per segment, daily | < 0.02 |
| Reliability diagram | Visually monotone; no systematic band bias |
| **Slope/intercept of observed vs predicted** | Slope ∈ [0.92, 1.08] |
| Brier score decomposition | Track refinement and calibration terms **separately** |
| Drift: ECE rising with AUC flat | **Calibration drift, not concept drift → recalibrate, do not retrain** |
| Drift: AUC falling with ECE flat | **Discrimination loss → retrain, recalibration won't help** |

> Those last two rows are the cheapest, most useful diagnostic pair in the entire fraud stack, and they cost nothing to compute.

---

## 4.7 Evaluation — the metrics that matter, and why AUC lies

| Metric | Verdict |
|---|---|
| **ROC-AUC** | **Nearly useless here.** At a 0.14% base rate it is dominated by the vast negative class and is invariant to the monotone transforms that carry all the economics |
| **PR-AUC / average precision** | Better; sensitive to the positive class. Use as a model-comparison metric, not as a decision metric |
| **Recall @ fixed FPR** | Good, operationally meaningful — "what fraction of fraud do we catch at a 0.5% decline rate" |
| **Precision @ k** | Meaningful when k is your analyst capacity — this is the **review queue quality** metric |
| **$-weighted recall** | `Σ(loss of caught fraud) / Σ(loss of all fraud)`. **The metric closest to the business** |
| **$-saved at fixed friction** | The best single number: dollars of fraud prevented at a fixed decline rate |
| **Expected Calibration Error** | Mandatory (§4.6) |
| **Net EV per 1,000 decisions** | Simulate the full four-action policy from §2.1 on a held-out window. **The metric that should gate releases** |
| **Loss-capture curve** | Cumulative $-fraud caught vs cumulative $-friction incurred — the fraud analogue of a triage curve, and the artefact to show a sponsor |

> **The release gate should be net EV per 1,000 decisions on a matured, holdout-corrected, temporally-later window — not AUC.** This single change in what a model must beat aligns the data science team's incentive with the P&L, and it is a decision an architect can simply make.

**And the evaluation discipline that catches the most bugs:**

| Rule | Why |
|---|---|
| **Always evaluate on a temporally-later window** | Random splits leak future information through entity-level correlation. A card's later transactions in train and earlier in test is a common and devastating leak |
| **Split by entity, not by row** | Same reason |
| **Report on accepted *and* holdout-corrected populations** | §4.2 |
| **Report per segment and per attack type** | Aggregate improvement routinely hides a regression on the class currently costing you money |
| **Simulate the policy, not just the score** | A better score with the same thresholds can produce a worse P&L via the review-capacity interaction |

---

## 4.8 Champion–Challenger and the Statistics of Proving Anything

**The decision this stage makes:** *"Is the new model actually better, and how long must I wait to know?"*

### The power calculation that governs everything

```
Approved-population fraud rate p = 0.14%.  Daily auths 1.6M → 800k per arm at 50/50.
n per arm = 2(z_{α/2}+z_β)² p(1−p) / δ²

Detect a 5% relative reduction (0.140% → 0.133%, δ = 7×10⁻⁵):
  n ≈ 4.48M per arm  →  5.6 days of traffic
Detect a 1% relative reduction (δ = 1.4×10⁻⁵):
  n ≈ 112M per arm   →  140 days of traffic
```

> **You can detect a 5% improvement in under a week. You cannot detect a 1% improvement at all — 140 days per arm exceeds both your retrain cadence and the adversary's adaptation cycle, so the world you are measuring changes before the experiment concludes.**
>
> **And it is worse than it looks**, because of §1.4: the outcome you are measuring arrives 34–97 days after the transaction. A 5.6-day exposure window still needs ~100 days of label maturation before you can read it.

### What you do about it — four techniques, in order of value

1. **Offline counterfactual replay is your primary evidence, not your secondary.** Replay both models over the same historical matured window with the same policy simulation. You get orders of magnitude more statistical power because you are not splitting traffic, and you control for period effects exactly. **Online tests then confirm rather than discover.**
2. **Read early proxies through a maturation model.** Use the §4.1 hazard curve to project the final chargeback rate from day-14 partial observation, with an honest confidence band. This turns a 100-day wait into a 21-day decision with quantified uncertainty.
3. **Measure at the score level where you can.** Score distribution shift and rank-correlation between champion and challenger on live traffic are observable *immediately* and are excellent leading indicators. They will not prove the P&L, but they will tell you within an hour that something is badly wrong.
4. **Variance reduction (CUPED) using pre-period entity behaviour.** Typically buys 20–40% variance reduction, which is a 1.4–1.7× reduction in required duration — helpful, not transformative.

**And the design rule that follows:**

> **Do not run champion–challenger to chase 1% improvements; you cannot resolve them.** Run it to (a) confirm that a change validated offline does not have an unexpected online effect, (b) catch feature-pipeline skew between training and serving, and (c) protect against regressions. **Ship on offline evidence; use online tests as a safety net.** That is the opposite of the received wisdom from web A/B testing, and it is correct here because the base rate is 0.14% rather than 4%.

---

## 4.9 Adversarial Robustness — the modelling angle

The source document's V14 covers the oracle problem architecturally. Here is what it means for the model itself.

| Threat | Modelling response |
|---|---|
| **Probing** — attacker submits transactions to map your boundary | Do not return granular reasons to untrusted actors; add small stochastic jitter at decision boundaries so the boundary is not crisply learnable; rate-limit *per entity cluster*, not per account |
| **Feature manipulation** — attacker controls inputs (device string, email, address format, amount) | **Weight features by manipulation cost.** A feature the adversary can change for free (user agent) should carry less influence than one that costs them money (card BIN, verified identity, account age). Encode this explicitly during feature selection — most teams never do |
| **Mimicry** — crafting transactions to look like the legitimate population | Deviation-from-self features are harder to mimic than population-level features, because the attacker doesn't know the victim's baseline |
| **Poisoning** — injecting mislabelled data via the analyst or dispute loop | Analyst decisions are training labels; a compromised or manipulated analyst poisons the model. Sample-level provenance + reviewer agreement monitoring |
| **Cascade gaming** (§4.5) | Randomised escalation; monitor T1-resolved realised fraud rate |
| **Model extraction** | Rate limits, score quantisation in any external-facing response, no confidence exposure |

> **The idea worth carrying out of this section: features have a *cost of manipulation*, and it belongs in feature selection alongside importance.** A feature with high importance and zero manipulation cost is a liability — the model will lean on it and the adversary will flip it. Ranking candidate features by `importance ÷ manipulation cost` produces a measurably more durable model, and I have rarely seen a team do it.

---

## 4.10 Explainability, Adverse Action and Fairness

**The decision this stage makes:** *"Can I tell the customer, the analyst and the regulator why?"* — three different audiences needing three different artefacts.

| Audience | Artefact | Requirement |
|---|---|---|
| **Customer** | Adverse-action reason codes | A small, fixed, human-meaningful vocabulary. **Not SHAP values.** Regulated in some jurisdictions and for some decision types |
| **Analyst** | Evidence panel — top contributing features with values, entity graph path, similar past cases | Actionable, not just attributive |
| **Model validator / regulator** | Global behaviour, monotonicity, stability, fairness testing, documented limitations | Reproducible for years |

**Strategy menu for explanation:**

| Method | Wins | Fails |
|---|---|---|
| **Natively interpretable model (LR, EBM)** | Explanation is the model | Accuracy cost, sometimes real, sometimes assumed |
| **SHAP (TreeSHAP)** | Exact for trees, fast, per-decision | Attributions ≠ reasons; hard to map to a fixed code vocabulary; can be unstable under correlated features |
| **Reason-code mapping layer** | Maps feature contributions to a **fixed business vocabulary** of ~25 codes | Requires curation — and this is the right investment |
| **Counterfactual explanations** | "Had the amount been under $X…" — intuitive | Can leak the boundary to an adversary. **Do not expose externally** |
| **Surrogate model** | A simple model approximating the complex one | Fidelity gaps; a surrogate that disagrees with the decision is worse than nothing |

**Our choice:** TreeSHAP computed **asynchronously after the response is sent** (it does not fit the 80 ms budget and does not need to), persisted with the decision record; a curated 25-code reason vocabulary mapped from SHAP groups; a monotonically-constrained GBDT so that direction-of-effect is guaranteed and defensible.

> **Monotonic constraints deserve more attention than they get.** Constraining "higher velocity ⇒ not lower risk" costs typically 0–1pp of PR-AUC and buys: a model that cannot be gamed by pushing a feature in an unexpected direction, an explanation that never contradicts intuition, a validator conversation that is dramatically easier, and robustness to distribution shift in the tails. **For regulated fraud decisioning it is very often a net positive**, and the accuracy cost is usually assumed rather than measured.

**Fairness:** test disparate impact across protected and proxy attributes on the *decision*, not just the score; note that geography, device type and payment instrument are all proxies. The honest architectural position: **you cannot fix fairness in the model alone**, because the label itself (`chargeback filed`) carries the biases of who disputes. Document it, measure it per release, and treat a fairness regression as a release blocker.

---

## 4.11 GenAI in the Loop — where it belongs and where it does not

The source document's V21 handles the regulated-narrative use case. The ML architect's summary:

| Use | Verdict |
|---|---|
| **Analyst investigation assistant** — summarise evidence, surface similar cases, draft the case narrative | **Yes.** Measured 30–40% handling-time reduction, which converts directly into review capacity (§2.4) and therefore into money |
| **SAR / regulatory narrative drafting** | **Yes, with mandatory human authorship and sign-off.** The analyst owns the filing |
| **Rule drafting from analyst description** | **Yes** — generate a candidate rule, then backtest it against history before anyone deploys it |
| **Feature ideation from attack write-ups** | Yes, cheap and useful |
| **Scoring transactions** | **No.** Latency, cost, calibration, explainability, reproducibility and adversarial robustness all fail |
| **Deciding** | **No.** The policy layer is deterministic by design |

> **The business framing that gets this funded:** at 11,520 case-slots/day and $6.77/case, a 35% handling-time reduction is worth **either $10.0M/yr of cost** or, better, **4,000 additional case-slots/day** — which §2.4 valued at roughly $19 each at the margin. **The investigation assistant is worth more as capacity than as savings**, and framing it that way changes which team funds it and how it is measured.

---

# PART 5 — The Drift and Retraining Control Plane

## 5.1 Five kinds of drift, and why they need different responses

| Type | Definition | Signature | Response | Speed needed |
|---|---|---|---|---|
| **Data / infrastructure drift** | Feature values change because a pipeline changed | Feature nulls, distribution jumps, schema mismatch | **Fix the pipeline.** Never retrain on this | Minutes |
| **Population drift** | Traffic mix changes (new market, campaign, seasonality) | Feature distributions shift, fraud rate stable per segment | Often nothing; maybe recalibrate per segment | Days |
| **Calibration drift** | Ranking holds, probabilities stale | **ECE ↑, AUC flat** | **Recalibrate weekly. Do not retrain** | Weekly |
| **Concept drift** | P(fraud \| features) genuinely changed | **AUC ↓, ECE may be flat** | Retrain | Weeks |
| **Adversarial drift** | An intelligent opponent found a gap | **Sharp, localised, segment-specific**; often a *fall* in flagged volume before a rise in losses | **Rules first (hours), model second (weeks)** | Hours |

> **The distinction that saves the most money: calibration drift is 4× more common than concept drift and 20× cheaper to fix.** Teams that only have one "model is degrading" alert retrain on calibration drift — a three-week, MRM-gated, high-risk cycle to fix something a weekly isotonic refit would have handled. **Separate the two signals and you separate the two responses.**

> **The distinction that saves the most money the other way: adversarial drift moves faster than any retrain cycle.** By the time you have retrained, revalidated and released, the attack has run for three weeks. **The correct response to adversarial drift is a rule, deployed in hours, by the fraud strategy team, without an engineering deploy** — which is exactly why the source document's V1 externalised rule engine never becomes obsolete. Rules are not a primitive precursor to ML; they are the **fast-response tier of a permanent portfolio**, and the model is the slow-response tier.

## 5.2 The differential diagnosis table

| AUC / PR | ECE | Feature nulls | Score dist. | Flagged volume | Realised fraud rate | ⇒ Diagnosis | First action |
|---|---|---|---|---|---|---|---|
| ↓ | — | ↑ | shifted | — | ↑ | **Feature pipeline broken** | Fix pipeline. **Do not retrain** |
| — | ↑ | — | — | ↑↓ | — | **Calibration drift** | Recalibrate (weekly job) |
| ↓ | — | — | — | — | ↑ | **Concept drift** | Retrain |
| ↓ **in one segment** | — | — | segment-shifted | **↓ in that segment** | **↑** | **Adversarial — attack found a gap** | **Rule now, model later** |
| — | — | — | — | ↑ | ↓ | Population drift (new benign cohort) | Segment-recalibrate; probably nothing |
| — | — | — | — | — | — | but **offline/online gap widening** | **Selection bias compounding** (§4.2) | Check holdout-corrected metrics |
| — | — | — | — | — | — | but **T1-resolved fraud rate ↑** | **Cascade gaming** (§4.5) | Randomise escalation; investigate |
| ↑ | — | — | — | ↓ | ↓ | *Everything improved* | **Investigate as hard as a regression** — usually a label pipeline break making fraud look absent |

> **The last row again, because it is the one that costs the most.** A label pipeline that stops ingesting chargebacks makes every metric improve. Fraud "falls." The model "gets better." Nobody investigates a good week. **Alert on unexplained improvement.**

## 5.3 The retraining trigger matrix

| Component | Leading signal | Confirming signal | Trigger | Action | Cost / elapsed | Risk |
|---|---|---|---|---|---|---|
| **Rules** | Analyst reports; attack telemetry | Segment fraud rate ↑ | any credible attack | Deploy rule | **Hours** | Low — reversible, auditable |
| **Calibrator** | ECE > 0.02 | Slope outside [0.92, 1.08] | weekly, or on breach | Isotonic refit | **1 day**, parameter update under MRM | Low |
| **Cheap tier (T1)** | T1-resolved realised fraud ↑ | Escalation rate shift | quarterly or on breach | Refit | 1 wk | Low — cannot deny alone |
| **Main GBDT** | PR-AUC ↓ >3pp on matured replay | Holdout-corrected gap stable (rules out §4.2) | 4 wks scheduled, or on breach | Refit, same features | **3 wks incl. MRM** | Medium |
| **Feature set** | New attack not expressible in current features | Analyst hypothesis validated offline | on demand | Add features → full retrain | 4–6 wks | Medium |
| **Sequence encoder** | Downstream feature importance ↓ | Session behaviour distribution shift | 8 wks | Refit encoder, regenerate embeddings | 2 wks | Medium |
| **Graph pipeline** | Component-size distribution shift; entity-resolution precision ↓ | New data source onboarded | 12 wks or on breach | Re-tune resolution + recompute | 2 wks | Medium — **ER errors propagate everywhere** |
| **Novelty detector** | Alert volume drifting; cluster stability ↓ | — | 12 wks | Refit | 1 wk | Low |
| **Hazard / label model** | Label arrival curve shifted | Issuer or network rule change | on breach | Refit **before** any risk-model retrain | 1 wk | **High if skipped — corrupts training weights** |
| **Thresholds / policy** | P&L term ratio shifts; capacity change | Business input change (margin, LTV, staffing) | monthly review | Re-solve §2.1 | **Hours, no deploy** | Low, and the **highest-frequency lever you have** |

**The ordering rule, again, because it is universal:** **diagnose in ascending order of remediation cost.** Pipeline (minutes) → threshold (hours) → rule (hours) → calibrator (1 day) → model refit (3 weeks) → feature engineering (6 weeks). The instinct to retrain first is expensive and usually wrong.

## 5.4 The monitoring surface

**Model health (per model, daily):** PR-AUC and $-weighted recall on matured windows · ECE and reliability slope per segment · score distribution PSI vs training · feature-level PSI and null rate · SHAP importance stability (a top-5 feature's importance moving >30% in a month is a real alarm) · **holdout-corrected vs accepted-population metric gap.**

**Business health (daily, one dashboard, one owner — §2.7):** the five P&L terms · decline rate and challenge rate by segment · **realised fraud rate on the exploration holdout** (the only unbiased number you have) · review queue depth and age vs the 6,200 capacity · analyst agreement rate · loss-capture curve position.

**Adversarial health (hourly):** flagged-volume anomalies **by segment** · new-entity-cluster emergence rate · T1-resolved realised fraud rate · repeat-probe patterns per entity cluster · velocity-feature degraded rate.

> **The single most valuable panel: realised fraud rate on the exploration holdout, by score decile.** It is the only measurement in the entire system uncontaminated by your own decisions. When your model says a decile is 45% fraud and the holdout says 31%, you have found the calibration problem that §2.3 priced at $23M — and no other instrument in the stack could have told you.

---

# PART 6 — The Interview Playbook

## 6.1 The opening (extending the source document's Chapter 0 move)

> *"Fraud is at least six problems with different loss functions. I'll take card-not-present payment fraud plus account takeover as the inline spine, because those set the hardest constraint.*
>
> *Before architecture, one number: the false-negative to false-positive ratio. For payment fraud it's about 2.4 to 1 — a $210 loss versus an $89 cost of wrongly declining a good customer once you count churned lifetime value. For account takeover it's about 355 to 1, because a step-up challenge costs almost nothing. And for promotion abuse it inverts entirely — roughly 1 to 8, because blocking a genuine new user costs more than the credit they'd have farmed.*
>
> ***Those three regimes cannot share a model, a threshold, or an evaluation metric.*** *They share a feature platform and a policy engine. That's the boundary I'd draw first.*
>
> *Second: because those numbers go into an expected-value equation, the model must emit a calibrated probability, not a rank. That immediately makes AUC the wrong headline metric — it's invariant to exactly the monotone transformations that carry all the economics. I'd gate releases on net expected value per thousand decisions, simulated on a matured, later window.*
>
> *Third: the review action is capacity-constrained, not preference-constrained. A hundred and eighty analysts gives about 11,500 case-slots a day, and the EV-optimal review band would generate 30,000. So the review boundary is solved, not chosen — and you rank the queue by uncertainty times exposure, not by score.*
>
> *For the model itself: a calibrated gradient-boosted tree on the inline path, because tabular gains come from features, not learners. Sequence and graph signal get learned offline and served as precomputed feature columns, so the inline path stays inside eighteen milliseconds and stays explainable. Now — the failure sequence…"*

## 6.2 Fifteen sentences that signal a principal ML architect

1. "The FN:FP ratio spans four orders of magnitude across six classes. They can't share a model."
2. "AUC is invariant to monotone transforms — and the entire economics lives in exactly that information."
3. "Segmentation by cost belongs in the policy. Segmentation by behaviour belongs in the model. Most teams get this backwards."
4. "The review band is solved from analyst capacity, and the queue is ranked by uncertainty × exposure, not by score."
5. "Gains come from features, not learners. I'd spend on sequence and graph signal before I'd spend on a new architecture."
6. "Learn representations offline; serve them as columns; keep the inline model a tree."
7. "First I'd fit the label-arrival hazard curve, because it sets the training window, the evaluation window, and how much recent data I can recover."
8. "My negatives include undisputed fraud, and my training set only contains transactions I approved. Both are correctable and neither is free."
9. "The exploration holdout is sized by statistical power. About a million a year, against a forty-four-million loss line."
10. "The calibrator is a separately versioned artefact that retrains four times more often than the model — and usually clears model risk as a parameter update."
11. "ECE up with AUC flat means recalibrate. AUC down with ECE flat means retrain. Two signals, two costs, two responses."
12. "I can detect a 5% improvement in six days and a 1% improvement never. So offline counterfactual replay is my primary evidence and the online test is a safety net."
13. "Rules aren't a primitive precursor to ML. They're the hours-response tier of a permanent portfolio; the model is the weeks-response tier."
14. "Features have a cost of manipulation, and it belongs in feature selection next to importance."
15. "Friction cost is larger than fraud loss and sits on someone else's dashboard. That asymmetry explains every over-tightening incident in this industry."

## 6.3 Trap questions

| Question | Weak answer | Principal answer |
|---|---|---|
| "Which model would you use?" | "XGBoost." | "Calibrated GBDT inline, because the signal is tabular and I have 18 ms. But the interesting question is what feeds it — sequence embeddings for ATO trajectory, precomputed graph features for rings, and a bought consortium signal in the uncertain band, which is usually the largest single lift available and no amount of modelling substitutes for data I don't have." |
| "How do you handle class imbalance?" | "SMOTE." | "Mostly by not over-reacting to it — GBDTs handle 0.14% fine. What I'd guard is that any rebalancing destroys calibration, and calibration is where all the money is. SMOTE specifically I'd avoid: interpolating between fraud events manufactures transactions that can't occur, in an adversarial setting where that region is meaningless." |
| "Would you use deep learning?" | "Yes, for better accuracy." | "For tabular, no — deep tabular hasn't reliably beaten tuned GBDT, and the published fraud wins came from new inputs, not better learners. For sequences and graphs, yes, but trained offline to produce embeddings served as feature columns. That gets the accuracy, keeps the latency, and keeps an explanation I can put in an adverse-action notice." |
| "How do you know the model is working?" | "We monitor AUC." | "Net expected value per thousand decisions on a matured, holdout-corrected, temporally-later window — plus the realised fraud rate on the exploration holdout by score decile, which is the only unbiased measurement in the system." |
| "How often do you retrain?" | "Monthly." | "Scheduled every four weeks, where decay cost equals cycle cost. But adversarial decay is punctuated, not smooth, so the real design is scheduled plus event-triggered, with a pre-agreed fast path through model risk for a like-for-like refit. Negotiating that fast path is a political task and it's worth doing before you need it." |
| "How do you deal with delayed labels?" | "We wait 90 days." | "That throws away my three most valuable months against a monthly-adapting adversary. I'd fit a hazard model on label arrival and reweight partially-observed windows by the probability the label is still coming — recovering training data to T−30 while still evaluating only on fully matured windows." |
| "Your AUC went from 0.94 to 0.96 — ship it?" | "Yes." | "Not on that evidence. I'd want net EV on the simulated four-action policy, per segment, on a matured window, plus the calibration check — because a better-ranking model with unchanged thresholds can produce a worse P&L through the review-capacity interaction alone." |

---

# PART 7 — Portability

| Domain | The six-class analogue | Label latency | FN:FP | Distinctive modelling problem |
|---|---|---|---|---|
| **Credit underwriting** | Origination vs behavioural vs collections | 6–24 months | ~4:1 | **Reject inference is a regulated discipline here**, with decades of methodology. Borrow it wholesale |
| **Insurance claims / SIU** | Opportunistic vs organised vs provider | Months | ~3:1 | Exposure is *known exactly* at decision time — the cleanest `V_i` you will ever get for EV ranking |
| **Ads click / impression fraud** | Bots vs domain spoofing vs incentivised | Hours–days | ~1:3 | Volume is 1000× higher and unit value 1000× lower → **cost per decision dominates everything** |
| **Content abuse / trust & safety** | Spam vs harassment vs CSAM vs misinformation | Immediate–never | Varies by 6 orders of magnitude across classes | Text and image models dominate; **the CSAM class has an effectively infinite FN cost and its own legal regime** |
| **Healthcare claims fraud** | Upcoding vs phantom billing vs kickback networks | Months–years | High FN | **Graph and peer-comparison signal dominate** — outlier-relative-to-peer-provider is the core feature family |

**Two transfers worth stating:** the **EV-argmax over four actions** (§2.1) transfers unchanged to any domain with an intermediate action available — and most have one and don't use it. And the **capacity-constrained review band with `uncertainty × exposure` ranking** (§2.4) transfers to every human-in-the-loop risk operation in existence.

---

# PART 8 — Templates

## 8.1 Model Card — the fraud-specific fields most templates omit

```
MODEL: <name>  VERSION: <semver>  CLASS: <fraud class>  OWNER: <person>
PURPOSE: predicts P(<exact target definition>) within <label window>

ECONOMICS
  FN cost $___    FP cost $___    ratio ___    → operating regime: recall | precision | balanced
  Feeds decision boundaries at: allow<___ challenge<___ review<___ deny≥___
  Boundaries vary by segment?  Y/N  → see policy version ___

LABELS
  Target definition (exact):     ___
  Arrival: median ___d  p90 ___d  Censoring rate ___%
  Hazard model version:          ___     Reweighting applied to: T−___ days
  Proxy labels used:             ___     Proxy–outcome agreement: ___

SELECTION BIAS
  % of training population selected by a prior model: ___%
  Exploration holdout: ___% of declines, ___ labels/quarter
  Accepted-population metric ___ vs holdout-corrected ___  → GAP: ___  ⚠ track this

CALIBRATION
  Method ___  Segments ___  Refit cadence ___  Artefact version ___ (SEPARATE from model)
  Current ECE by segment: ___

ADVERSARIAL
  Top-5 features by importance, with MANIPULATION COST:
    1. ___ (importance ___, cost to adversary: free | $ | hard)
  Monotonic constraints applied: ___
  Cascade tier: ___   Can this tier DENY alone?  Y/N

EVALUATION
  Gate metric: net EV per 1,000 decisions on matured holdout-corrected later window
  Per-segment results: ___   Per-attack-type results: ___
  Fairness test results: ___

OPERATIONS
  Retrain: scheduled ___  triggered by ___   MRM path: full | fast-path
  Inference budget: ___ ms of ___ ms   Cost/decision: $___
  Degradation behaviour if features missing: ___
```

## 8.2 The "should we add a model?" gate

```
PROPOSED MODEL: ___
1. Which of the six classes, and what is its FN:FP ratio?
2. What business trigger (from §3.2) has fired?  Evidence: ___
3. What does the CURRENT model with this signal as a FEATURE achieve?   ← try this first, always
4. Positives available per retrain window: ___   (< 500 → do not build a separate model)
5. Which existing model does this displace?   (if none, justify +1 to the MRM inventory)
6. Inference budget available: ___ ms   Cost: $___
7. Explainability requirement: ___   Can this class satisfy it?
8. Who owns retraining, drift monitoring and the MRM entry for the next 3 years?
9. Expected net EV improvement, and can it be detected? (§4.8 power calculation)
10. Revisit trigger if it underperforms: ___
```

## 8.3 The one-page derivation to reproduce from memory

```
1.  Which fraud class? → its adversary, label latency, decision point.
2.  Price FN and FP.  → operating regime.
3.  Instantiate EV(allow/challenge/review/deny). → the four-region partition. → CALIBRATION IS MANDATORY.
4.  Vary the FP cost by segment. → segment boundaries live in POLICY, not the model.
5.  Count analyst capacity. → solve the feasible review band; rank by uncertainty × exposure.
6.  Fit the label-arrival hazard curve. → training window, eval window, reweighting.
7.  Size the exploration holdout from statistical power. → the only unbiased data you will ever have.
8.  Where is the signal — row, sequence, graph, document? → model class.
9.  What is the inference budget? → where each class runs (inline vs offline-to-features).
10. What must be explained? → may veto a class outright; consider monotonic constraints.
11. Build the cascade: rules → cheap → full → enriched → human. Escalate on UNCERTAINTY.
12. Separate the calibrator from the model. Refit it 4× more often.
13. Gate releases on net EV, not AUC. Evaluate on matured, later, holdout-corrected windows.
14. Run the power calculation before promising an A/B result.
15. Diagnose drift in ascending order of remediation cost. Rules in hours; models in weeks.
```

---

## Closing thesis

The source document proves that a fraud platform is an **architecture** problem: latency budgets, feature stores, point-in-time correctness, governance drawn rather than appended.

This document argues that inside that architecture sits a **portfolio** problem. Not "the model" — twelve models, of five classes, on three latency paths, with four retrain cadences, each earning its place against a business trigger and each costing an operating burden that must be justified against a P&L nobody owns.

The move from ML engineer to ML architect in this domain has a precise shape: it is the move **from optimising a metric to solving a decision under cost, capacity and adversarial pressure**. The metric is a rank; the decision is a partition of a probability line into four regions whose boundaries are owned by four departments, constrained by an analyst roster, drifting under an intelligent opponent, and defensible to a regulator seven years later.

> **Price the errors before you choose the model. Calibrate, because the economics live in the probability. Solve the boundaries; don't pick them. Spend on features, not learners. Learn offline, serve as columns. Separate the calibrator from the model, and the rule from the model, because they answer in hours, days and weeks respectively — and the adversary does not wait for your retrain cycle.**

---
---

# ANNEX A — The AWS Reference Implementation

## A.0 The decision that shapes everything: do not call a model endpoint inline

Before any diagram. The budget from the source document's Figure 1.1 allocates **18 ms to model inference** inside an **80 ms p99**, at **8,000 eps peak**.

```
Managed model endpoint (HTTP/gRPC hop):
  network RTT within VPC        1.5 – 3 ms
  TLS + serialisation           1 – 2 ms
  endpoint queueing at 8k eps   2 – 6 ms (and the tail is the problem, not the mean)
  actual GBDT inference         2 – 4 ms
                                ───────────
  realistic p99                 12 – 20 ms  ⚠ consumes the entire allocation, sometimes exceeds it

In-process inference (booster loaded in the decision service):
  actual GBDT inference         2 – 4 ms
  p99 with GC headroom          6 – 9 ms   ✓ fits with room for the 18 ms reserve
```

> **Therefore: the inline model is compiled into the decision service, not called over the network.** Export the trained booster (native format, ONNX, or a compiled predictor such as Treelite), version it as an immutable artefact, ship it as a sidecar or embedded asset, and hot-swap it behind a version pointer. **SageMaker endpoints are for the T3 enriched tier, for batch scoring, and for the async explanation path — not for the 99% of traffic that must answer in 18 ms.**
>
> This is the most commonly-fumbled decision in AWS fraud architectures, because the platform's default path pushes you toward a managed endpoint. Saying *"a managed endpoint is a network hop, and at eight thousand events per second inside an eighteen-millisecond budget the hop is the whole budget"* is a strong signal that you have operated one of these.

**What you give up and how to get it back:** managed endpoints give you canary routing, autoscaling and capture-for-monitoring for free. In-process serving means you rebuild those: **model version routing via a config flag (AppConfig), champion/challenger by request hash, and asynchronous inference-capture to Kinesis** — which you need anyway, because feature logging (§ source doc V5) is already asynchronous.

## A.1 The architecture

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║   FRAUD DECISIONING — AWS REFERENCE ARCHITECTURE     80 ms p99 · 8,000 eps peak · $0.0008/decision            ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

███ INLINE DECISION PATH  ───────────────────────────────────── everything here is inside the 80 ms budget

 ┌──────────────┐   ┌───────────────────────────────────────────────────────────────────────────────────┐
 │ Payment /    │──▶│  DECISION SERVICE  — ECS Fargate or EKS, 3 AZ, no cold starts, connection-pooled   │
 │ auth service │   ├───────────────────────────────────────────────────────────────────────────────────┤
 └──────────────┘   │ ① parse + authz              6 ms                                                  │
                    │ ② FEATURE FETCH  14 ms  ─── batched multi-key GET, ONE round trip ──┐              │
                    │ ③ graph features 10 ms  ─── precomputed, cached ───────────────┐    │              │
                    │ ④ MODEL         18 ms  ─── booster IN-PROCESS (see A.0) ──┐    │    │              │
                    │      T0 rules → T1 cheap → T2 full → [T3 async escalate]  │    │    │              │
                    │ ⑤ CALIBRATOR     ~0 ms  ─── isotonic lookup, separate artefact version              │
                    │ ⑥ POLICY         8 ms  ─── EV argmax over 4 actions, segment boundaries, hot-reload │
                    │ ⑦ reason codes   6 ms  ·  18 ms RESERVE (do not spend)                              │
                    └───────────────────────────────────────────────────────────────│────│────│──────────┘
                              │ decision + reason set                                │    │    │
                              ▼                                                      ▼    ▼    ▼
    ┄┄▶ ASYNC (after response): feature-vector log · SHAP · audit write · event emit  │    │    │
                                                                                      │    │    │
  ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲   ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲   ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 │ ElastiCache / MemoryDB    │  │ DynamoDB — entity profiles   │  │ ElastiCache — graph features    │
 │ VELOCITY COUNTERS ≤2 s    │  │ + SageMaker Feature Store    │  │ (nightly Neptune → cache)       │
 │ ~60 GB, 24 h TTL          │  │ ONLINE store, ≤5 min          │  │ ≤15 min                         │
 │ 66,000 reads/s at peak    │  │ ~415 GB                       │  │                                 │
  ╲_________________________╱   ╲____________________________╱   ╲_______________________________╱

███ STREAMING FEATURE PLANE  ┄┄▶ ─────────────────────────────── the ≤2 s freshness tier lives or dies here

 ┌────────────┐   ╔══════════════════╗   ┌────────────────────────────────┐   ┌─────────────────────────┐
 │ all events │┄─▶║ Kinesis / MSK    ║┄─▶│ Managed Service for Apache     │┄─▶│ velocity counters →     │
 │ 120M/day   │   ║ partitioned by   ║   │ Flink — windowed aggregates,   │   │ MemoryDB                │
 └────────────┘   ║ ENTITY KEY (not  ║   │ watermarks, allowed lateness,  │   │ profiles → DynamoDB     │
                  ║ random) so a     ║   │ **exactly-once to the sink**   │   └─────────────────────────┘
                  ║ single entity's  ║   └────────────────────────────────┘
                  ║ events are ordered║   ⚠ Partition by entity or your 60-second card-velocity counter
                  ╚══════════════════╝     is computed from out-of-order events and is quietly wrong.

███ OFFLINE / TRAINING PLANE  ═══▶ ──────────────────────────────────────────────────────────────────────

 ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲   ┌─────────────────────────────────────────────────────────┐
│ S3 lakehouse (Iceberg)                     │  │ SageMaker Feature Store — OFFLINE                        │
│ raw events 13 mo · features 25 mo          │  │ **point-in-time joins** = the anti-leakage mechanism      │
│ decisions + served feature vectors 7 yr    │  │ time-travel to reconstruct any historical training set    │
│ (Object Lock, WORM)                        │  └─────────────────────────────────────────────────────────┘
 ╲_________________________________________╱
 ┌─────────────────────┐ ┌──────────────────────┐ ┌────────────────────┐ ┌──────────────────────────────┐
 │ LABEL PIPELINE      │ │ TRAINING             │ │ EVALUATION         │ │ REGISTRY & GOVERNANCE        │
 │ chargebacks (34/97d)│ │ SageMaker Training   │ │ counterfactual     │ │ SageMaker Model Registry     │
 │ analyst dispositions│ │ Step Functions       │ │ REPLAY of the full │ │ + approval status = the MRM  │
 │ HAZARD MODEL on     │ │ orchestrates:        │ │ 4-action policy on │ │   GATE the serving layer     │
 │ label arrival (§4.1)│ │ label → features →   │ │ a matured, later,  │ │   HONOURS at deploy time     │
 │ EXPLORATION HOLDOUT │ │ train → eval → gate  │ │ holdout-corrected  │ │ Clarify: bias + SHAP         │
 │ (§2.5) tagged       │ │ → register           │ │ window             │ │ Model Cards; lineage         │
 └─────────────────────┘ └──────────────────────┘ └────────────────────┘ └──────────────────────────────┘

███ GRAPH & CASE  ─────────────────────────────────────────────────────────────────────────────────────
 Neptune (entity graph, nightly component/community/PageRank → feature cache) · Neptune Analytics for
 batch algorithms · analyst case management + traversal UI reads Neptune directly (NOT in the inline path)

███ MONITORING  ───────────────────────────────────────────────────────────────────────────────────────
 CloudWatch EMF (per-model, per-segment: PR-AUC, ECE, PSI, null rate, degraded-feature rate, cost/decision)
 SageMaker Model Monitor (data quality + drift) · Clarify (bias drift) · **COMPOSITE ALARMS = §5.2 pairs**
 QuickSight: the five-term P&L dashboard (§2.7) — one page, one owner
```

## A.2 Service selection

| Component | AWS choice | Why | Rejected |
|---|---|---|---|
| **Inline inference** | **In-process booster** in the decision service | A.0 — the network hop is the budget | SageMaker real-time endpoint inline: 12–20 ms p99 at 8k eps |
| Enriched tier (T3) | SageMaker async/real-time endpoint + third-party APIs | Only ~1.2% of traffic; latency tolerance is higher | — |
| **Velocity store (≤2 s)** | **MemoryDB** (durable) or ElastiCache | 66k reads/s, sub-ms, in-memory; the §1.2 working set fits | DynamoDB alone: single-digit ms is fine for profiles, marginal for the 14 ms 8-key batch |
| Profile / entity features | **SageMaker Feature Store** online (+ DynamoDB) | **Offline/online parity and point-in-time joins are the whole reason to use a feature store** — not "everyone has one" | Homegrown: you will re-implement point-in-time correctness badly |
| **Stream processing** | **Managed Service for Apache Flink** | Event-time windows, watermarks, allowed lateness, exactly-once sinks. Kinesis Analytics SQL is not expressive enough for the velocity grid | Micro-batch (Spark Streaming): the ≤2 s tier is unreachable |
| Event bus | **Kinesis or MSK, partitioned by entity key** | Ordering per entity is a correctness requirement, not a performance one | Random partitioning: silently wrong velocity counters |
| Graph | **Neptune** (+ Neptune Analytics) | Best managed property graph on any cloud; path evidence for explanations | Live traversal inline: blows the 10 ms allocation |
| Training orchestration | **Step Functions** + SageMaker Training | Eval gate and registry approval become explicit states | Ad hoc notebooks: unreproducible under MRM |
| **Model governance** | **SageMaker Model Registry approval status enforced at deploy** | The source doc's V17 gate — *"a control that is not drawn is not built"*. The serving layer must refuse an unapproved artefact | Approval in a wiki |
| Explainability | **Clarify** (SHAP) + a curated reason-code mapping | Async, post-response, persisted with the decision | Inline SHAP: does not fit and does not need to |
| Policy layer | **AppConfig** + DynamoDB | Hot-reload thresholds without a deploy; versioned and audited — the source doc's FR-3 | Config in the image |
| Decision record | DynamoDB → S3 with **Object Lock**, 7 yr | Immutable, reproducible | Logs |

## A.3 The three AWS traps in this domain

1. **Partitioning the event stream by anything other than entity key.** Velocity is a per-entity windowed aggregate; if a card's events land on three shards, your 60-second counter is computed from an unordered subset. It will look fine in tests and be wrong under load.
2. **Feature-store online/offline skew.** The store gives you parity *only if training reads the offline store through point-in-time joins and serving reads the online store with the same transformation code*. Two implementations of "distinct cards in 600 s" — one in Flink, one in Spark for training — is the classic training/serving skew, and it is invisible until the online/offline metric gap opens. **One transformation definition, two execution engines, and a daily parity test on sampled entities.**
3. **Model Monitor as your only drift signal.** It watches data quality and distribution. It does not know your ECE, your $-weighted recall, your holdout-corrected gap, or your P&L. **Emit those as custom EMF metrics and build composite alarms from §5.2.**

---

# ANNEX B — The Azure Reference Implementation

## B.0 Where Azure is strong here

Two genuine advantages for a *regulated* fraud platform:

**(a) The Responsible AI tooling is the closest fit to model risk management of any cloud.** The Responsible AI dashboard bundles error analysis, fairness assessment across cohorts, interpretability, counterfactuals and causal analysis into one artefact, and the **Responsible AI scorecard is a PDF you can hand to a model validator.** For the source document's V17 and V18 chapters — MRM approval, fairness, adverse action — this is a materially shorter path than assembling equivalents.

**(b) Purview gives lineage and classification across the estate**, which matters for V19's retention-versus-erasure conflict: knowing which features, models and decision records derive from a subject's data is the prerequisite for answering an erasure request at all.

**The weakness, stated plainly:** graph. Cosmos DB for Apache Gremlin or PostgreSQL with Apache AGE, neither of which matches Neptune operationally. For the entity graph the pragmatic answer is often a **nightly materialised component/community table** computed in Spark rather than a live graph database — which, per §3.4, is what you serve inline anyway.

## B.1 The architecture (deltas from Annex A)

```
INLINE   Payment service → Decision Service (Container Apps / AKS, zone-redundant)
         ① parse+authz → ② feature fetch (Azure Managed Redis, ≤2 s velocity; Cosmos DB, ≤5 min profile)
         → ③ graph features (Redis, nightly) → ④ MODEL IN-PROCESS (same A.0 argument; Azure ML
           online endpoints are a network hop) → ⑤ calibrator → ⑥ POLICY (App Configuration, hot-reload)
         ┄▶ async: feature log, SHAP, audit → ADLS (immutable blob, 7 yr)

STREAM   Event Hubs (partitioned by ENTITY KEY) → Stream Analytics for simple windows,
         or Azure Databricks Structured Streaming / Flink for the full velocity grid
         → Azure Managed Redis (velocity) + Cosmos DB (profiles)

OFFLINE  ADLS Gen2 + Delta/Parquet · Azure ML MANAGED FEATURE STORE (point-in-time joins)
         Azure ML pipelines: label → hazard model → features → train → evaluate → gate → register
         Azure ML Model Registry + **approval gate honoured by the deploy pipeline**
         RESPONSIBLE AI DASHBOARD + SCORECARD → the MRM artefact (B.0)

GRAPH    Nightly Spark/GraphFrames component + community computation → materialised table → Redis
         (Cosmos Gremlin only if interactive analyst traversal justifies it)

MONITOR  Azure Monitor + App Insights · Azure ML data drift + model monitoring
         KQL log alerts = the §5.2 signal pairs · Power BI: the five-term P&L dashboard
```

## B.2 Selection notes

| Component | Choice | Note |
|---|---|---|
| Inline inference | **In-process** | Same A.0 reasoning. Azure ML managed online endpoints are excellent for T3, batch and async explanation |
| Velocity store | **Azure Managed Redis** | Sub-ms, the working set fits |
| Profiles | **Cosmos DB** | Single-digit-ms, global distribution, per-tenant partitioning |
| Feature store | **Azure ML managed feature store** | Point-in-time joins; **the reason to adopt it, not the registry UI** |
| Streaming | Stream Analytics for simple windows; **Databricks/Flink for the velocity grid** | Stream Analytics runs out of expressiveness at the multi-window grid |
| Governance | **Azure ML Registry + Responsible AI scorecard + Purview** | B.0 — the strongest MRM story of the three |
| Policy | **App Configuration** | Hot-reload with versioning and audit; feature flags for kill switches and fail-open |
| Fail-open decision | App Configuration flag + Front Door health | The source doc's V13 is a *business* decision; make the toggle explicit, audited and rehearsed |

---

# ANNEX C — The Databricks Implementation

## C.0 What Databricks uniquely buys a fraud platform

| Requirement | Databricks primitive |
|---|---|
| **Point-in-time correct training sets** (source doc V5) | **Feature Engineering client point-in-time joins** — leakage prevention as an API, not a discipline |
| **The three freshness tiers** (≤2 s / ≤5 min / ≤15 min) | <cite index="103-1">Lakebase offers three online serving paths: Managed Publish for minute-level updates, Declarative Window for consistent windowed aggregates, and Direct Stream writing directly to the serving layer when you need seconds-fresh operational state.</cite> **These map one-to-one onto the source document's freshness tiering** |
| **Online feature serving** | <cite index="98-1">Databricks Online Feature Stores, powered by Lakebase, serve feature data to real-time applications and model serving endpoints with low latency</cite>, <cite index="101-1">with governance and lineage maintained back to the offline feature tables</cite> |
| **Label maturation and time travel** | Delta time travel — reconstruct the exact training set as of any date, which is the V5 reproducibility requirement |
| **Feature drift monitoring** | Lakehouse Monitoring — <cite index="97-1">Databricks Feature Store relies on Lakehouse Monitoring for feature data quality tracking</cite> |
| **Model registry + lineage** | MLflow in Unity Catalog: model → features → source tables, automatically |
| **The five-term P&L dashboard** | AI/BI + Genie over the same lakehouse the decisions land in — §2.7 becomes a query, not an integration project |
| **Graph** | GraphFrames batch → materialised component/community table (per §3.4, this is what you serve anyway) |

> <cite index="103-1">The Lakebase blog frames the hard part correctly: the engineering work is the **routing** — every feature requires a deliberate choice weighing the cost of real-time processing against the performance impact of data lag.</cite> That is precisely §4.4's rule 2 (match the window to the attack timescale) expressed as a platform decision, and it is the right frame.

## C.1 The honest limitation

**The 18 ms / 8,000 eps inline path is not a Databricks strength.** Model Serving is a network hop with the same arithmetic as A.0. Feature Serving Endpoints are excellent for applications with a tens-of-milliseconds budget; they are not built to sit inside a payment authorisation's 18 ms inference slice at 8,000 events per second.

**Therefore the recommended topology is the split you have now seen three times, for the same reason each time:**

| Plane | Where |
|---|---|
| **Inline decision path** (rules → cheap → full model → calibrator → policy) | **Your own service, model artefact in-process**, reading Redis/Lakebase-published features |
| Feature computation, publication and freshness routing | **Databricks** — Structured Streaming / DLT → Online Feature Store |
| Labels, hazard model, training, point-in-time joins, evaluation, replay | **Databricks** |
| Registry, lineage, drift monitoring, MRM evidence | **Databricks (MLflow + UC + Lakehouse Monitoring)** |
| Case management, analyst tooling, P&L dashboard | **Databricks (Apps + AI/BI)** |

> **The sentence to say:** *"I'd run the entire model lifecycle on Databricks — point-in-time joins, label maturation with time travel, training, replay evaluation, registry, lineage and drift are genuinely best-in-class there — and I'd serve the model artefact in-process in my own decision service, because eighteen milliseconds at eight thousand events per second doesn't survive a network hop on any platform."*

## C.2 The Databricks-specific wins worth naming

1. **Point-in-time joins remove the single most common training bug in fraud.** Label leakage through as-of-now feature values is endemic; making the correct thing the default API is worth more than it sounds.
2. **Time travel makes the §4.1 hazard reweighting tractable.** You can reconstruct "what did we know on day *d*" exactly, which is what a maturation model needs for fitting and validation.
3. **Traces, features, labels, decisions and outcomes in one catalog** means the §5.4 monitoring surface — including the holdout-corrected metric gap and the five-term P&L — is a set of SQL views, not an integration programme.
4. **Unity Catalog governs features, models and functions together**, which materially shortens the MRM inventory exercise: "which models use this feature, and which decisions did they produce" is a lineage query.

---

# ANNEX D — Cross-Platform and How to Choose

## D.1 Equivalence

| Component | AWS | Azure | Databricks |
|---|---|---|---|
| **Inline inference** | **In-process booster** | **In-process booster** | **In-process booster** |
| Managed endpoint (T3/batch/async) | SageMaker endpoints | Azure ML online endpoints | Model Serving |
| Velocity store (≤2 s) | MemoryDB / ElastiCache | Azure Managed Redis | **Lakebase Direct Stream** |
| Profile store (≤5 min) | DynamoDB + SM Feature Store | Cosmos DB | **Lakebase Managed Publish** |
| Windowed aggregates | Flink (MSK/Kinesis) | Stream Analytics / Databricks | **Declarative Window** |
| Point-in-time joins | SageMaker Feature Store offline | Azure ML feature store | **Feature Engineering client** |
| Graph | **Neptune** | Gremlin / AGE *(weak)* | GraphFrames batch |
| Training orchestration | Step Functions + SageMaker | Azure ML pipelines | Lakeflow Jobs |
| Registry + deploy gate | SageMaker Model Registry | Azure ML Registry | **MLflow in Unity Catalog** |
| Explainability / fairness | Clarify | **Responsible AI dashboard + scorecard** | MLflow + custom |
| Drift monitoring | Model Monitor + custom EMF | Azure ML monitors | **Lakehouse Monitoring** |
| Policy hot-reload | AppConfig | App Configuration | External |
| Signal-pair alerting | Composite alarms | KQL log alerts | SQL alerts |
| 7-yr immutable decisions | S3 Object Lock | Immutable blob | UC Volumes + policy |

## D.2 Honest scorecard for this workload

| | Strongest | Weakest |
|---|---|---|
| **AWS** | **Neptune is the best managed graph**; Flink for the ≤2 s tier; the widest primitive set for building the inline path exactly as specified | Governance assembled across many services; Model Monitor doesn't know your economics |
| **Azure** | **The best MRM and fairness artefact story** (Responsible AI scorecard) — which for a regulated fraud programme is a large, real advantage; Purview lineage for the V19 retention conflict | Graph; Stream Analytics runs out of expressiveness for the velocity grid |
| **Databricks** | **The entire model lifecycle** — point-in-time joins, time travel for label maturation, replay evaluation, lineage, drift, and the P&L dashboard over the same data | The inline 18 ms / 8k eps path is not its job |

## D.3 How to answer "which platform?"

> *"Three questions decide it.*
>
> ***One: where does the inline path live?*** *On every platform the answer is the same — the model artefact runs in-process in your decision service, because eighteen milliseconds at eight thousand events per second doesn't survive a network hop. So the platform choice is really about everything except the hot path.*
>
> ***Two: how regulated is this decision?*** *If a validator, a regulator and an adverse-action notice are in scope, Azure's Responsible AI scorecard is the shortest path to the artefact those people actually want, and Purview answers the retention-versus-erasure question. That's worth more than a benchmark.*
>
> ***Three: is relational structure central?*** *If organised rings are a primary loss channel and analysts need interactive traversal, Neptune is the strongest managed graph and pulls me toward AWS. If graph features are enough — which per my earlier point they usually are — that constraint relaxes and Databricks' lifecycle advantages dominate.*
>
> *My default is Databricks for the model lifecycle and feature platform, an in-process serving path in my own service, and whichever cloud's graph and governance story matches the regulatory posture. And I'd note that the parts that actually determine whether this system makes money — pricing the errors, calibrating, solving the boundaries, sizing the exploration holdout, separating the calibrator from the model, diagnosing drift in ascending order of cost — are identical everywhere."*

## D.4 Currency discipline
Write the **capability**, not the product. Re-verify quarterly against release notes and GA/preview matrices. **For this workload specifically, re-check latency and throughput characteristics of any managed serving or feature-store tier every quarter** — those numbers move, and they are the ones that decide the architecture. And re-run the economics, not the feature list: does this capability move `L`, `M`, the review capacity, the cost per decision, or the retrain elapsed time? If none of the five, skip it.
