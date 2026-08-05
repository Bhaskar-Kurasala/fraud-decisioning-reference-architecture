# Review priced as a label source: the term §5 left out, and what it is worth

**Date:** 2026-08-05
**Epic:** E12
**Status:** Complete — §5 qualified, not overturned. The recommendation changes; the arithmetic does not.

Reproduce: `uv run python scripts/analyse_review_label_value.py`

## Why this was run

`fraud-decisioning-findings.md` §5 concludes that human review does not pay: value-of-review
is positive on **5 of 92,427** transactions at $7.97/case, all four queue-ranking rules cost
more than the $2,799,797 no-review policy, and the whole programme turns on `f`, the rate
fraudsters defeat a step-up challenge.

That prices review as a one-shot decision improvement — does a human profitably change
*this* decision. It is a complete accounting of that question and it omits a second output.
An analyst adjudication is a **label today**. A chargeback is a label after a 34-day median
dispute, and a good transaction is only confirmed good when the 90-day window closes
silently. Label latency is the organising constraint of this whole system: §6 measured the
observed fraud rate collapsing to 6.9% of true in the final 20 days, and
`monitoring.maturity` refuses every label-dependent metric on an immature window rather
than reporting it with a caveat.

So review might pay as a *label-acquisition mechanism* even though it loses as a *decision
mechanism*. Nobody had priced that. This note does.

## Method

The §5 queue is reproduced from `data/econ_test.parquet` through the extracted
`fraudlens.economics` cost model, with one deliberate deviation: a deterministic tie-break
on `TransactionID`. §5's own footnote records that `argsort` over the 153-valued isotonic
score made two of its four rows irreproducible run-to-run. The 60-slot value-of-review
queue reproduces the published $2,980,763 exactly.

The value of an early label is not a dollar amount, it is a shortened exposure. The causal
chain assumed here, stated so it can be argued with:

> earlier labels → earlier detection of decay → fewer days operating a degraded model →
> less accumulated excess cost

Each arrow is an assumption. The third has a measured anchor —
[`fit-balanced-empirical-result.md`](fit-balanced-empirical-result.md) puts the cost of
operating one model that passed an AUC gate and should not have at **$4,358,096/yr**. The
second is modelled as a 3σ fixed-sample test on the reviewed cohort's fraud rate, which is
deliberately cruder than the CUSUM anyone would actually run: it *overstates* detection
time, so it understates the labels' value, which is the direction to err in when the
conclusion under test is "buy the labels". The first is where the analysis nearly dies; see
§"Two ways this result fails".

## 1. The queue is a purchase of same-day labels

| slots/day | labels/yr | cost vs no-review | $/label | queue fraud rate | min p in queue |
|---|---|---|---|---|---|
| 5 | 1,825 | $14,667 | $8.04 | 0.756 | 0.375 |
| 10 | 3,650 | $27,988 | $7.67 | 0.784 | 0.375 |
| 20 | 7,300 | $54,785 | $7.50 | 0.805 | 0.375 |
| 30 | 10,950 | $85,557 | $7.81 | 0.787 | 0.375 |
| **60** (§5) | **21,900** | **$180,966** | **$8.26** | **0.582** | **0.047** |
| 120 | 43,800 | $386,072 | $8.81 | 0.397 | 0.026 |

Read the third column as a purchase price rather than a loss. The 60-slot programme costs
$180,966/yr against no review, of which $174,543 is simply $7.97 × 21,900 cases — the
analyst's *decisions* are close to value-neutral in aggregate (−$6,423/yr), which is the
same fact §5 reports from the other direction. What you are buying for $8.26 is a label
that arrives 90 days before its chargeback would.

## 2. The deadline is 90 days, and it is a cliff

| age | 7d | 30d | 60d | 89d | 90d | 120d |
|---|---|---|---|---|---|---|
| labels knowable | 0.11% | 1.53% | 2.60% | 3.03% | **99.56%** | 99.76% |

Under the dispute-lag model in `streaming/labels.py`, fraud discloses lognormally but a
clean transaction is confirmed only by the dispute window closing. At a 3.5% fraud rate the
clean step is 96% of the mass, so maturity is not a curve that creeps up to the 80% floor —
it sits near 3% for three months and then jumps. **A label-dependent metric is not
computable on day 89 and is fully computable on day 90.** That single fact is what makes an
analyst label worth anything at all: it is not competing against a partially-matured
chargeback signal, it is competing against nothing.

## 3. Labels as training data: worth nothing, and this is the important negative

The obvious version of this idea — analyst labels feed the retraining loop — does not
survive contact with the numbers, for two independent reasons.

**Volume.** The training window holds 414,542 transactions and 14,600 fraud events. A
60-slot queue produces 21,900 labels a year, a 5-slot queue 1,825. Adding 2% more rows to a
training set does not measurably move a gradient-boosted model.

**Selection.** The queue samples the top of the value-of-review ranking. At 5 slots/day the
lowest-scored case in the queue has p = 0.375; at 60 slots/day, p = 0.047. The median
allow→challenge boundary is **0.089** and the median allow→deny boundary is **0.554**. The
5-slot queue therefore contains *no* transactions from the region where the allow/challenge
decision is actually made, and the 60-slot queue only 8.7%. These labels describe a part of
the score space the model already handles confidently. They are a censored sample, drawn
precisely where the decision is easiest.

This is the selection-bias objection, and against the training-data mechanism it is fatal.
No sensitivity analysis rescues it; the mechanism is priced at zero and excluded from
everything below.

## 4. Labels as a monitoring signal: this one survives, on a weaker requirement

The reason the same objection does not kill the monitoring mechanism is a distinction worth
stating plainly, because it is the load-bearing step in this note:

> An **estimator** needs an unbiased sample. A **detector** needs a stationary one.

The reviewed cohort's fraud rate is a badly biased estimate of anything — it is 58–80%
against a 3.5% population rate. But the selection rule (rank by value-of-review, take the
top *k*) is a fixed function of the score and the cost model. Hold that rule fixed and the
statistic is stable under the null. A move in it is a real move, even though its level means
nothing. This is a control chart, not a measurement, and control charts are allowed to be
biased.

Analyst error enters as symmetric misclassification: an observed rate of `(1−q) + (2q−1)r`,
so at q = 0.91 the chart sees 82% of any true shift. That slows detection — at 5 slots/day,
detecting a 13-point precision drop takes 32.6 days at q = 0.91 against 19.6 at q = 1.00,
and 67.0 days at q = 0.80 — but it does not bias the direction. A noisy label still works as
a detector.

**Does the cohort actually move when the model breaks?** Measured, not assumed, against the
one decay event in this repository:

| score | queue fraud rate | PSI vs champion |
|---|---|---|
| champion (isotonic) | 0.5823 | 0.0000 |
| raw uncalibrated | 0.5589 | 0.3820 (alerts) |
| **rebalanced, empirically refit** | **0.1958** | **2.9542 (alerts)** |

The $4.36M/yr failure moves the reviewed cohort's fraud rate from 58% to 20%. At 60 slots
that is detectable within a day. The signal is real and it is enormous.

## 5. Break-even

Fixing severity at the measured $4,358,096/yr, and taking the 90-day wall as the
counterfactual detection time:

**Days to detect a drop of x in the reviewed cohort's fraud rate:**

| slots/day | −0.05 | −0.13 | −0.25 |
|---|---|---|---|
| 5 | 220.4 | 32.6 | 8.8 |
| 10 | 104.7 | 15.5 | 4.2 |
| 20 | 50.2 | 7.4 | 2.0 |
| 30 | 34.8 | 5.1 | 1.4 |
| 60 | 21.9 | 3.2 | 0.9 |
| 120 | 10.8 | 1.6 | 0.4 |

Each row uses *its own* cohort's fraud rate as the baseline (0.756 at 5 slots, 0.582 at 60),
because a shorter queue takes only the top of the ranking and is therefore purer. Two
consequences, and the second is a real weakness in this table: rows are not a clean 1/*k*
scaling of one another, and applying the same *absolute* 13-point drop to a cohort at 0.756
and to one at 0.582 flatters the short queues, whose baseline sits nearer a variance-minimising
extreme. A proportional shift would narrow the gap between the 5- and 60-slot rows. It would
not close it — the cost axis moves 12×, the detection axis less than 10× — but read the
recommendation below as "small queues are enough", not as a precise ratio.

**Decay events per year at which the queue pays for itself as an instrument:**

| slots/day | $/yr | −0.05 | −0.13 | −0.25 |
|---|---|---|---|---|
| 5 | $14,667 | never | **0.021** | 0.015 |
| 10 | $27,988 | never | 0.031 | 0.027 |
| 20 | $54,785 | 0.115 | 0.056 | 0.052 |
| 30 | $85,557 | 0.130 | 0.084 | 0.081 |
| **60** (§5) | $180,966 | 0.223 | **0.175** | 0.170 |
| 120 | $386,072 | 0.408 | 0.366 | 0.361 |

"never" means detection is slower than the chargeback wall, so the labels buy nothing at
all — the small queues are powerful per case but blind to small shifts.

**The break-even condition, in the form an operator can act on:**

> A review queue pays for itself purely as a decay sensor when the expected annual cost of
> model decay that (a) manifests as a precision shift in the region the queue samples and
> (b) is invisible to unsupervised drift exceeds **$761,325/yr at 60 slots/day**, or
> **$93,266/yr at 5 slots/day**.

Equivalently, at the measured severity: one such event every 5.7 years justifies the
published 60-slot programme, and one every 48 years justifies a 5-slot one.

The wall is the other lever, and it is a convention rather than a measurement. At 5
slots/day the break-even moves to 0.099 events/yr if you treat a transaction as clean at 45
days rather than 90 — a factor of 4.6. Any team that shortens its clean window shortens the
value of this programme proportionally.

## Two ways this result fails, and how much of it they take

Both clauses in that break-even condition are doing real work. They are the argument
against, and they are strong enough that the headline number should not be quoted without
them.

**(a) For the decay we have measured, a free signal already fires.** PSI on the score against
the refit rebalanced model is **2.9542** — 12× the 0.25 alert threshold — available on day
zero, at zero cost, with no labels. The reviewed cohort's 58%→20% collapse is a genuine
detection, but it is not an *earlier* one, so its marginal value over the monitoring already
in the repository is zero. The only decay modes where the analyst label is the earliest
signal are those with a stable score and feature distribution and a changed score→outcome
relation. That class is real (a marketing campaign delivering legitimate customers with
fraud-like signatures is the canonical case), but this dataset contains no instance of it,
so its frequency and severity are entirely unmeasured. What the label does add over PSI is
denomination: PSI says something moved, the cohort's precision says what it costs. That is
worth something and this note does not attempt to price it.

**(b) For the decay the free signal misses, the queue misses it too.** If an adversary finds
a blind spot — fraud that the model scores low — the score distribution does not move, so
PSI is silent by construction. It is also invisible to the queue, and for the same reason
the training-data mechanism died: the queue does not sample there. Of 80,071 currently-good
transactions below the 90th percentile of score, the 60-slot queue samples **3**.

| new fraud injected in that region | population rate | queue sees | random audit @60/day detects in |
|---|---|---|---|
| +0.5% (400 cases) | 3.48% → 3.91% | nothing | 269 days |
| +1.0% (800 cases) | 3.48% → 4.34% | nothing | 67 days |
| +2.0% (1,601 cases) | 3.48% → 5.21% | nothing | 17 days |

And no affordable audit rescues it either: a *random* 60-case/day audit — unbiased, and the
textbook answer — needs 269 days to see a half-percent uplift, because at a 3.5% base rate
random sampling is hopelessly underpowered. It loses to simply waiting 90 days for the
chargebacks. This is the honest reason fraud teams do not run random audits, and it is worth
recording alongside the reason they should not over-run queues.

So the label-acquisition term is large, real, and narrow. It is not a general argument for
staffing review.

## Verdict on §5

**Qualified, not overturned.** §5's arithmetic is untouched and reproduces exactly: as a
decision mechanism review loses money at this merchant, the queue-ranking comparison stands,
and the `f`-sensitivity conclusion stands — below f ≈ 0.20 there is no case for review as a
decision channel, and nothing here changes that.

What changes is the recommendation §5 implies. "Do not staff a review team for payments at
all" is too strong, because it prices only one of the two things a review team produces. The
defensible version:

> Do not staff review as a **decision channel**. Staff it as an **instrument**, and size it
> for detection power rather than for coverage.

Concretely, at this merchant: **5–20 slots/day, not 60.** Twenty slots costs $54,785/yr
against $180,966, produces a *higher*-precision cohort (80.5% fraud vs 58.2%, because it
takes only the top of the value-of-review ranking), and detects a 13-point precision drop in
7.4 days against a 90-day wall. The 60-slot configuration buys 3× the labels and 3.2-day
detection instead of 7.4-day detection, when the deadline it is racing is 90 days. That is
$126,000/yr for four days of headroom nobody needs.

## Limitations

- **The hazard rate is the whole result and it is not in this dataset.** One 32-day test
  window, zero observed decay events. Every break-even above is a threshold the operator has
  to compare against a number this analysis cannot supply.
- **Severity is anchored on a single measured event**, and that event is a deliberately
  broken model rather than an observed production decay. $4,358,096/yr is the right order of
  magnitude for "a model that passed an AUC gate and should not have"; it is not a
  distribution.
- **The retraining loop is assumed to convert detection into value linearly and
  instantly.** It does not: detection has to be followed by a rollback or a retrain, and any
  delay there comes straight off `days_saved`. At 5 slots/day the entire margin is 57 days,
  so a two-week response process consumes a quarter of it.
- **`q_analyst = 0.91` is assumed, not measured**, and there is no way to measure it here
  without the chargebacks whose latency is the point. An analyst calibration study needs its
  own matured window.
- **Reviewed transactions are challenged or declined, so their outcome is
  counterfactual** — the same limitation §7 flags. This analysis is partly insulated from
  it, because the label being bought is the analyst's adjudication of whether the
  transaction *was* fraud, not the outcome of the action taken. But that adjudication has
  never been checked against a realised chargeback in this dataset.
- **The 3σ fixed-sample detector is crude**, and conservative in the direction of the
  conclusion. A CUSUM would detect sooner and make the queue look better than it does here.
- **The 90-day clean window is an operating convention**, not a measurement. It is the
  single largest lever on the result after the hazard rate.
