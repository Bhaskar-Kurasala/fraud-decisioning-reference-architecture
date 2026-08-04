# Fraud Decisioning on Real Data — Findings

Built on IEEE-CIS Fraud Detection (Vesta Corporation e-commerce transactions).
590,540 transactions · 182 days · $79.7M GMV · 3.50% fraud · $3.08M at risk.

Split is out-of-time: train days 0–119, calibrate 120–149, test 150–181.
All P&L below is measured on the test window (92,427 txns, $12.7M GMV) against
real labels, annualised at 365/32.

Model: HistGradientBoosting, 225 features, leak-free entity aggregates.
**Out-of-time AUC 0.9045, PR-AUC 0.527.**

---

## 1. The intermediate actions are worth 3x the threshold tuning

| Policy | Annual cost | Saved vs do-nothing | Fraud caught | Good declined |
|---|---|---|---|---|
| P0 approve everything | $5,248,067 | — | 0% | 0 |
| P1 best global threshold (p ≥ 0.698) | $4,469,896 | $778,171 (14.8%) | 30.9% | 187 |
| P2 decline top 1% by score | $4,483,938 | $764,130 (14.6%) | 27.9% | 131 |
| P3 per-transaction EV threshold | $4,339,611 | $908,457 (17.3%) | 33.0% | 241 |
| **P4 four-action EV argmax** | **$2,799,797** | **$2,448,270 (46.7%)** | **64.2%** | **11** |

P4 catches twice the fraud of the best binary policy while hard-declining
**11 good customers instead of 187** — a 17x reduction in the worst customer
outcome, achieved simultaneously with doubling fraud capture.

The gap between P1 and P2 is $14k/yr. The gap between P1 and P4 is $1.67M/yr.
Threshold tuning and score-ranking are ~1% of the available value; having a
third action is ~68% of it. Same model, same score, same AUC throughout.

---

## 2. Miscalibration costs $6.6M/yr at identical AUC

Three scorings of the same model on the same test set, routed through the same
policy:

| Score | AUC | ECE | Annual cost | Penalty | Allow | Challenge | Deny |
|---|---|---|---|---|---|---|---|
| Calibrated (isotonic) | 0.9045 | 0.0027 | $2,799,797 | — | 92.8% | 6.8% | 0.35% |
| Raw uncalibrated | 0.9050 | 0.0054 | $2,871,342 | $71,545 | 94.7% | 4.9% | 0.49% |
| Class-rebalanced, uncorrected | 0.9050 | 0.2009 | $9,424,667 | **$6,624,870** | 26.8% | 71.0% | 1.96% |

The rebalanced score has **identical AUC and identical PR-AUC** to the champion
— the ranking is unchanged to four decimals. Its probabilities are inflated 27x
(it predicts 23.6% fraud on a 3.5% population). Routed through an EV policy it
challenges 71% of all traffic and costs more than doing nothing at all.

No ranking metric can see this. AUC is invariant to monotone transforms of the
score, and rebalancing is exactly a monotone transform. Every dashboard a
typical team runs would show this model as fine.

---

## 3. The boundary is a distribution, not a number

Per-transaction costs: `L = Amount×0.70 + $25 + $12`, `M = Amount×0.30 + relationship_cost(tenure)`.

| Percentile | allow→challenge | challenge→deny | allow→deny (binary) |
|---|---|---|---|
| p1 | 0.037 | 0.805 | 0.328 |
| p50 | 0.089 | 0.913 | 0.554 |
| p99 | 0.227 | 0.969 | 0.789 |

**By tenure — the deny boundary spans 2x:**

| Tenure | Fraud rate | Median M | Boundary |
|---|---|---|---|
| 1–7d | 8.4% | $223 | **0.740** |
| new (0d) | 4.7% | $158 | 0.642 |
| 31–90d | 2.3% | $103 | 0.534 |
| 400d+ | 1.1% | $58 | **0.379** |

**By amount — and here it inverts:**

| Amount | Fraud rate | Median L | Median M | Boundary |
|---|---|---|---|---|
| < $25 | **8.0%** | $49 | $139 | **0.731** |
| $50–100 | 2.7% | $78 | $145 | 0.608 |
| $250–500 | 4.9% | $272 | $223 | 0.437 |
| $500+ | 5.2% | $613 | $372 | **0.369** |

You must be **73% sure** to decline a sub-$25 transaction and only **37% sure**
to decline a $500+ one — the opposite of what most fraud teams do. The reason:
`M` is dominated by a fixed relationship cost that doesn't shrink with basket
size, while `L` scales with it. On a $20 order the customer relationship is
worth more than the goods, several times over.

Note also that fraud rate is highest at *both* ends of the amount distribution
(8.0% under $25, 5.2% over $500, 2.7% in the middle). Exposure and risk are
non-monotonically related — you cannot use amount as a risk proxy.

---

## 4. Median L/M = 0.80 — this is a precision problem, not a recall problem

| | Median | Mean | p99 |
|---|---|---|---|
| FN cost (L) | $84.95 | $133.39 | $912.79 |
| FP cost (M) | $144.00 | $143.55 | $503.69 |

On the median transaction a false positive costs **more** than a false negative.
p10 of L/M is 0.40; p90 is 1.67.

This is a payment-fraud dataset, and the blueprint's assumed FN:FP of 2.4:1 is
inverted here. The driver is basket size: at a $69 median order the goods are
worth less than the customer. A team arriving from an ATO or AML background,
where recall dominates, would systematically over-decline this book.

---

## 5. Human review does not pay at this merchant

Value-of-review is positive for **5 of 92,427 transactions** at $7.97/case.
Every capacity allocation rule loses money, because the thing being allocated
has negative value:

| Ranking rule | Annual cost | Fraud in queue | Realised VoR |
|---|---|---|---|
| by score (case-mgmt default) | $2,997,388 | 69.1% | −$16,678 |
| by uncertainty p(1−p) | $3,003,398 | 38.8% | −$17,041 |
| by uncertainty × exposure | $3,106,035 | 19.1% | −$28,116 |
| by value-of-review | $2,980,763 | 58.2% | −$14,783 |

All four are worse than the $2,799,797 no-review policy. Ranking by
uncertainty × exposure is the *worst* of the four — it fills the queue with
large, genuinely ambiguous transactions, which is correct advice under a
different cost structure and wrong here.

**When review starts to pay** (share of volume routed to a human):

| f ↓ / cost → | $2.00 | $4.00 | $7.97 | $15.00 |
|---|---|---|---|---|
| 0.05 | 0.00% | 0.00% | 0.00% | 0.00% |
| 0.11 | 0.11% | 0.03% | 0.01% | 0.00% |
| 0.20 | 2.16% | 1.40% | 0.50% | 0.17% |
| 0.35 | 4.40% | 3.29% | 2.00% | 0.81% |
| 0.50 | 4.88% | 4.20% | 3.07% | 1.60% |

The whole review programme hinges on `f`, the rate at which fraudsters defeat
your step-up challenge — a vendor performance number, not a modelling number.
Below f ≈ 0.20 you should not staff a review team for payments at all.

---

## 6. The label-latency bug, simulated

Chargeback arrival modelled lognormal, median 34 days. Training on day 119 with
"everything labelled so far":

- 38.7% of true fraud in the training window has not yet been disputed
- Observed rate collapses to **6.9% of true rate** in the most recent 20 days

| Block (20d) | True rate | Observed rate | Ratio |
|---|---|---|---|
| 0 | 2.64% | 2.43% | 0.92 |
| 2 | 4.14% | 3.29% | 0.80 |
| 4 | 3.51% | 1.47% | 0.42 |
| 5 | 4.32% | 0.30% | **0.07** |

A model fitted on this learns that recent transactions are safe, and any feature
correlated with recency becomes a spurious safety signal. The fix is a label
maturity cutoff — train only on data old enough for labels to have arrived —
which costs you your freshest weeks and is the real reason fraud models retrain
on a lag.

---

## 7. Five-term P&L under the recommended policy

| Term | Annual | Share |
|---|---|---|
| Fraud losses | $1,979,064 | 70.7% |
| Friction (declines + abandons) | $820,278 | 29.3% |
| Operations (analyst) | $455 | 0.0% |
| Infrastructure | $843 | 0.0% |
| **Total** | **$2,800,640** | |

Friction/fraud = 0.41x. The blueprint asserted friction exceeds fraud losses;
here it doesn't — because the EV policy is already precision-heavy by
construction (11 hard declines in 32 days). Under the best *binary* policy the
same ratio is far higher. The blueprint's claim is a property of
threshold-based systems, not of fraud systems generally.

---

## Method notes and limitations

- LTV derived from observed spend, corrected for survivorship: conditioning on
  ≥2 transactions inflated new-customer LTV from $323 to $1,999. P(repeat)
  measured directly at 44.5% across 151,928 watchable customers.
- `p_churn|declined` is the one parameter with no data support here; it is
  tenure-graded from published false-decline research and should be treated as
  the softest input in the model.
- Challenge and review outcomes are counterfactual — realised P&L uses true
  labels for the loss, expected values for the intervention.
- No exploration holdout exists in this dataset, so selection bias from the
  original issuer's declines is uncorrected and unmeasurable. Real deployments
  need one.
