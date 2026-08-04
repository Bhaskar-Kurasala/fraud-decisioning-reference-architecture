# Verification Report — `fraud-decisioning-findings.md`

Independent recomputation of every claim in design-spec §6 against the persisted
artifacts in `data/`, cross-checked against `outputs/*.log`.

**Method.** No pipeline stage was re-run. All numbers below were recomputed from
`data/meta.parquet`, `data/X.parquet` (column subsets only), `data/econ_test.parquet`,
`data/scored_test.parquet`, `data/tenure_econ.csv` and `data/policies.csv`, using the
constants imported from `config.py` and the identical estimator/EV/realised-cost code
paths from `research/05_economics.py` and `research/06_full.py`. sklearn 1.7.2,
pandas 2.3.3, numpy 2.2.6.

**Repository note.** During this investigation the research stages were moved from the
repo root to `research/` (staged git renames, `run_all.sh` updated, ADR-0001). File
contents are unchanged; paths in this report use the new locations.

---

## 1. Summary table

| # | Claim | Status | Published | Measured | Delta |
|---|---|---|---|---|---|
| 1 | Transactions | REPRODUCED | 590,540 | 590,540 | 0 |
| 1 | Days | REPRODUCED | 182 | 182 (day 0–181) | 0 |
| 1 | GMV | REPRODUCED | $79.7M | $79,738,949 | 0 |
| 1 | Fraud rate | REPRODUCED | 3.50% | 3.4990% | 0 |
| 1 | "$3.08M at risk" | REPRODUCED (framing caveat) | $3.08M | $3,083,845 fraud GMV | 0 — but economic FN cost is $2.92M, see §3.1 |
| 2 | Train size / rate | REPRODUCED | 414,542 / 3.52% | 414,542 / 3.5220% | 0 |
| 2 | Calib size / rate | REPRODUCED | 83,571 / 3.41% | 83,571 / 3.4103% | 0 |
| 2 | Test size / rate | REPRODUCED | 92,427 / 3.48% | 92,427 / 3.4763% | 0 |
| 2 | Test GMV | REPRODUCED | $12.7M | $12,726,595 | 0 |
| 2 | Annualisation | REPRODUCED | 365/32 | test window = 32 distinct days; `consts.json` DAYS=32 | 0 |
| 3 | Out-of-time AUC | REPRODUCED | 0.9045 | 0.9045 (isotonic) | 0 |
| 3 | PR-AUC | REPRODUCED | 0.527 | 0.5271 (isotonic) | 0 |
| 4 | P0 annual | REPRODUCED | $5,248,067 | $5,248,067 | $0 |
| 4 | P1 (p≥0.698) | REPRODUCED | $4,469,896 / saved $778,171 (14.8%) | $4,469,896 / $778,171 (14.8%); thr 0.6981 | $0 |
| 4 | P2 | REPRODUCED | $4,483,938 / saved $764,130 (14.6%) | identical | $0 |
| 4 | P3 | REPRODUCED | $4,339,611 / saved $908,457 (17.3%) | identical | $0 |
| 4 | P4 | REPRODUCED | $2,799,797 / saved $2,448,270 (46.7%) | identical | $0 |
| 4 | P1→P2 gap | REPRODUCED | ~$14k/yr | $14,041 | 0 |
| 4 | P1→P4 gap | REPRODUCED | $1.67M/yr | $1,670,099 | 0 |
| 4 | Third action share of gain | REPRODUCED | ~68% | 68.22% | 0 |
| 5 | Isotonic AUC/ECE/cost | REPRODUCED | 0.9045 / 0.0027 / $2,799,797 | 0.9045 / 0.00273 / $2,799,797 | $0 |
| 5 | Raw AUC/ECE/cost/penalty | REPRODUCED | 0.9050 / 0.0054 / $2,871,342 / $71,545 | 0.9050 / 0.00538 / $2,871,342 / $71,545 | $0 |
| 5 | Rebalanced AUC/ECE/cost/penalty | REPRODUCED — **but analytic, not empirical** | 0.9050 / 0.2009 / $9,424,667 / $6,624,870 | identical to the cent | $0 |
| 5 | Rebalanced mean p | REPRODUCED | 23.6% | 23.57% | 0 |
| 5 | Rebalanced challenge share | REPRODUCED | 71% | 71.01% | 0 |
| 6 | Break-even new(0d) | REPRODUCED | 0.642 | 0.6417 | 0 |
| 6 | Break-even 1–7d | REPRODUCED | 0.740 | 0.7403 | 0 |
| 6 | Break-even 31–90d | REPRODUCED | 0.534 | 0.5341 | 0 |
| 6 | Break-even 400d+ | REPRODUCED | 0.379 | 0.3787 | 0 |
| 7 | Break-even <$25 | REPRODUCED | 0.731 | 0.7307 | 0 |
| 7 | Break-even $50–100 | REPRODUCED | 0.608 | 0.6078 | 0 |
| 7 | Break-even $250–500 | REPRODUCED | 0.437 | 0.4371 | 0 |
| 7 | Break-even $500+ | REPRODUCED | 0.369 | 0.3686 | 0 |
| 8 | Median L/M | REPRODUCED | 0.80 | 0.8049 | 0 |
| 8 | Median FN cost L | REPRODUCED | $84.95 | $84.95 | $0 |
| 8 | Median FP cost M | REPRODUCED | $144.00 | $144.0036 | $0 |
| 9 | VoR positive count | REPRODUCED | 5 of 92,427 at $7.97/case | 5 of 92,427; C+D = 6.77+1.20 = $7.97 | 0 |
| 9 | Queue: by score | **DISCREPANT** | $2,997,388 | $2,997,659 | **+$271** |
| 9 | Queue: by uncertainty | **DISCREPANT** | $3,003,398 | $3,003,436 | **+$38** |
| 9 | Queue: by uncertainty×exposure | REPRODUCED | $3,106,035 | $3,106,035 | $0 |
| 9 | Queue: by value-of-review | REPRODUCED | $2,980,763 | $2,980,763 | $0 |
| 9 | All four worse than $2,799,797 | REPRODUCED | yes | yes (worst margin +$180,966) | — |
| 10 | Fraud invisible at train time | REPRODUCED | 38.7% | 38.7% (8,955 of 14,600 arrived) | 0 |
| 10 | Final-block observed/true ratio | REPRODUCED | 6.9% | 0.0685 (block 5, days 100–119) | 0 |
| 11 | Fraud losses | REPRODUCED | $1,979,064 (70.7%) | $1,979,064 (70.7%) | $0 |
| 11 | Friction | REPRODUCED | $820,278 (29.3%) | $820,278 (29.3%) | $0 |
| 11 | Ops | REPRODUCED | $455 | $455 | $0 |
| 11 | Infra | REPRODUCED | $843 | $843 | $0 |
| 11 | Total | REPRODUCED | $2,800,640 | $2,800,640 | $0 |
| 12 | Corrected new-customer LTV | REPRODUCED | $323 | $323.43 | $0 |
| 12 | Naive LTV | **UNVERIFIABLE** | $1,999 | no artifact; best reconstruction $1,984.16 | see §3.2 |
| 12 | P(repeat) | REPRODUCED | 44.5% | 44.47% | 0 |
| 12 | Watchable customers | REPRODUCED | 151,928 | 151,928 | 0 |

**Score: 47 of 50 checked quantities reproduced exactly. 2 discrepant (both small,
both in the review-queue table). 1 unverifiable.**

---

## 2. Provenance: analytic vs empirically measured

This is the single most important framing point in the report.

### 2.1 The $6,624,870 miscalibration penalty is **ANALYTIC**, not empirical

Established from code, not from the README's assertion:

- `research/03_model.py` fits **one** model (`class_weight=None`, the champion). It
  saves `p_ca_raw.npy` and `p_te_raw.npy`. The class-balanced refit is gated behind
  `if os.environ.get("FIT_BALANCED") == "1":` and would write `data/p_te_bal_fitted.npy`.
- **`data/p_te_bal_fitted.npy` does not exist.** The gate was never taken in the
  published run (`outputs/03_model.log` shows exactly one `[champion] 400 iters, 43s`
  line and no `[class-balanced]` line).
- `research/03b_calibrate.py` never reads any fitted-balanced file. It constructs
  `p_bal` in closed form by prior shift:
  `pi = y[train].mean(); r = (1-pi)/pi; p_bal = p_raw*r / (p_raw*r + 1 - p_raw)`.
- Verified numerically: recomputing that closed form from `p_raw` in
  `data/econ_test.parquet` reproduces the stored `p_bal` with
  **max absolute difference = 0.0** (bit-identical). π = 0.03522, r = 27.393×.

Consequences that follow *by construction*, not by measurement:

- Rebalanced AUC and PR-AUC are *necessarily* identical to raw (0.9050 / 0.5391) —
  a prior shift is a strictly monotone map, so the "identical AUC" headline is a
  mathematical tautology here, not an empirical coincidence. The findings text
  ("the ranking is unchanged to four decimals") slightly overstates this as a
  discovery.
- ECE 0.2009, mean p 23.57%, 71.01% challenge rate and the $6,624,870 penalty are all
  downstream of the closed form. They are *correct arithmetic under a model of what a
  balanced fit does*, not observations of a balanced fit.
- An actual `class_weight='balanced'` HistGBM would additionally change tree structure
  (split gains are reweighted), so its ranking would **not** be identical to the
  champion and its ECE would not be exactly 0.2009. The empirical penalty could differ
  in either direction.

The README (limitation 5) and the code comment both disclose this. The **findings doc
itself does not** — §2 presents the rebalanced row alongside two empirically measured
rows in the same table with no marker. That is the main honesty gap found in this
review. The design spec §6 already schedules the empirical re-derivation via
`FIT_BALANCED=1`; per instruction it was not run here.

### 2.2 Other non-empirical inputs

| Quantity | Provenance |
|---|---|
| `P_CHURN_ON_DECLINE` (0.42 … 0.09 by tenure) | **Assumed.** No data support in IEEE-CIS. Feeds `M_relationship`, therefore feeds every break-even boundary (§6, §7), every M-dependent cost, and the entire P4 result. Disclosed in the findings method notes. |
| `F_PASS = 0.11`, `A_ABANDON = 0.07`, `Q_ANALYST = 0.91` | **Assumed** (vendor/industry numbers). `F_PASS` alone drives the P4 vs P1 gap; the findings' own sensitivity grid shows the review conclusion flips above f ≈ 0.20. |
| `C_REVIEW = 6.77`, `D_DELAY = 1.20`, `COST_PER_DECISION = 0.0008` | **Assumed** unit costs. |
| Challenge and review realised outcomes | **Counterfactual.** `realised()` uses the true label for the allow/deny arms but expected values (F, Aa, Q) for the intervention arms. No transaction in this dataset was actually challenged or reviewed. |
| Label-latency lag distribution | **Simulated** — lognormal(median 34d, σ 0.85), seed 7. IEEE-CIS labels are final; §6 demonstrates a failure mode, it does not measure one. |
| Everything else | **Empirically measured** from the artifacts: counts, GMV, fraud rates, AUC/PR-AUC/ECE, all L/M distributions, all policy costs, P(repeat), annualised GMV per repeater. |

---

## 3. Discrepancy analysis

### 3.1 Claim 9 — review-queue costs for "by score" and "by uncertainty"

| Ranking | Findings doc | `outputs/06_full.log` | My recomputation |
|---|---|---|---|
| by score | $2,997,388 | **$2,997,659** | **$2,997,659** |
| by uncertainty p(1−p) | $3,003,398 | **$3,003,436** | **$3,003,436** |
| by uncertainty × exposure | $3,106,035 | $3,106,035 | $3,106,035 |
| by value-of-review | $2,980,763 | $2,980,763 | $2,980,763 |

Secondary figures show the same pattern: fraud-in-queue published 69.1%/38.8% vs
log-and-measured 69.0%/38.9%; realised VoR published −$16,678/−$17,041 vs
log-and-measured −$16,676/−$17,045.

**My recomputation agrees with the stage log exactly. The findings doc disagrees with
its own pipeline output.** This is not a reproduction failure — it is a transcription
or stale-run drift in the published document.

**Root cause hypothesis — tie-breaking under the isotonic step function.** Measured:
the calibrated score `p` takes only **153 distinct values across 92,427 transactions**
(isotonic regression is a step function; the pool-adjacent-violators solution collapses
long runs to a single level). The analyst budget is 1,920 cases. The value sitting at
rank 1,920 in the score ranking is 0.41463, and **120 transactions share that exact
value**. For `p(1−p)` the rank-1,920 boundary value has **362 ties**.

`np.argsort(-rank)[:budget]` therefore selects an essentially arbitrary subset of the
tied block, determined by the introsort/quicksort partition order — which is sensitive
to input array order, dtype, numpy version and even array memory layout. Swapping which
of the 120 tied cases enter the queue changes realised cost by a few hundred dollars on
a $3M annualised base ($271 = 0.009%).

The two rankings that reproduce exactly are precisely the two that have **no ties**:
`uncertainty × exposure` has 19,914 distinct values (continuous `TransactionAmt`
multiplier), and `value-of-review` is a continuous function of L and M. That is a clean
confirming signal for the tie-break hypothesis.

**Severity: cosmetic.** Both deltas are <0.01%, neither changes sign, ordering, or the
conclusion that all four rankings lose to the no-review policy. **Fixability:** replace
`np.argsort(-rank)` with a deterministic stable sort plus an explicit tie-break key
(e.g. `np.lexsort((TransactionID, -rank))`), and the number becomes reproducible across
environments. This should be done before the value is put under a regression test.

### 3.2 Claim 12 — the $1,999 naive LTV is unverifiable

**Why it cannot be checked:** the pipeline contains only `research/04b_ltv.py`, the
*corrected* estimator. There is no `04_ltv.py` (the naive version the findings doc
compares against), and there is no `outputs/04_ltv.log`. `run_all.sh` runs
`01, 02, 03, 03b, 04b, 05, 06` — the naive stage is not in the pipeline. The $1,999
figure appears in `fraud-decisioning-findings.md` and in the design spec, and in no
code, log, or data artifact. It is therefore **not regenerable by `run_all.sh`**,
contrary to that script's stated reproducibility guarantee.

The other half of the claim reproduces cleanly: corrected new-customer residual LTV
= **$323.43** (p_repeat 0.3829 × ann_gmv_if_repeat $3,312.28 × MARGIN 0.30 ×
DISCOUNT 0.85), matching `data/tenure_econ.csv` and `outputs/04b_ltv.log`.

I attempted to reconstruct $1,999 from the persisted panel. Candidate naive estimators
for the new(0d) bucket, all × MARGIN × DISCOUNT:

| Reconstruction | Value |
|---|---|
| median annualised GMV over repeaters with obs≥7 (i.e. drop only the P(repeat) haircut) | $844.63 |
| **mean** annualised GMV over repeaters with obs≥7 | **$1,984.16** |
| median over repeaters, no obs≥7 filter | $4,529.65 |
| median over all n≥2 uids, no 60-day watchability filter | $5,627.85 |
| median over all n≥2 uids, obs≥7 | $930.75 |

The closest is **$1,984.16** — mean rather than median annualised spend over
new-customer repeaters — which is 0.7% from the published $1,999. That is consistent
with the naive stage having used a mean where 04b uses a median, but I cannot confirm
it: the difference is within the range that a slightly different `uid` construction or
`obs_days` floor would produce, and the original code no longer exists.

**Recommendation:** either restore the naive estimator as a stage (or a test fixture)
so the $323 → $1,999 contrast is regenerable, or restate the claim using a figure that
is (e.g. "$323 vs $845 if you drop only the P(repeat) haircut" — same qualitative point,
2.6× instead of 6.2×, and fully reproducible from `04b_ltv.py`'s own intermediates).

### 3.3 Claim 1 — "$3.08M at risk" is a framing choice, not an error

$3,083,845 is the **gross GMV of fraudulent transactions**, and reproduces exactly.
But the economics used everywhere downstream define the cost of an undetected fraud as
`L = Amount × 0.70 + $25 + $12`, which totals **$2,923,222** over the same
transactions — 5.2% lower, because the merchant loses cost-of-goods, not retail price,
partly offset by fixed chargeback fees. The headline number and the P&L are on
different bases. Not wrong, but a reader carrying "$3.08M at risk" into the P&L tables
will not be able to reconcile it. Worth a footnote.

---

## 4. Leakage and methodology concerns

Read from `research/02_features.py`, `03_model.py`, `03b_calibrate.py`,
`05_economics.py`, `06_full.py`. Ordered by materiality.

### 4.1 P1's threshold is selected on the test set (in-sample tuning) — MATERIAL

`05_economics.py` sweeps 300 candidate thresholds over the **test window itself** and
keeps the one minimising realised test-set cost:

```python
for t in np.unique(np.quantile(p, np.linspace(0.90,0.9999,300))):
    act = np.where(p>=t, 3, 0)
    c = realised(act,y,L,M).sum()
    if best is None or c<best[1]: best=(t,c,act)
```

`p ≥ 0.698` is the *oracle-optimal* global threshold for these 92,427 rows with their
true labels revealed. A production team could not have picked it in advance. The same
applies, more mildly, to P3 (per-transaction EV threshold, no free parameter) and P4
(EV argmax, no free parameter) — those two are honest out-of-sample decisions.

**Direction of the bias:** it *flatters P1*. The headline "third action is 68% of the
value, threshold tuning is ~1%" is therefore **conservative** — a fairly-selected
threshold (swept on calib, applied to test) would perform no better than $4,469,896 and
almost certainly worse, widening the P1→P4 gap. The conclusion survives; the specific
$778,171 does not deserve the precision it is quoted with.

The same critique applies to the whole policy comparison: there is no holdout for
*policy selection*. P0–P4 are all scored on the one 32-day window that also fixed the
isotonic map's evaluation and the calibration story. No confidence intervals are
attached to any dollar figure; the 32-day window contains 3,213 fraud events, so the
sampling error on a $2.8M annualised number is not negligible.

### 4.2 Feature construction — clean, with one caveat

Genuinely leak-free, verified by reading:

- Frequency encodings (`*_cnt`) are fit on `df.loc[trmask]` value counts only.
- Entity amount aggregates (`amt_over_*`, `amt_z_*`) take mean/std from the train mask
  only, then map onto all rows.
- Velocity features (`*_txn_idx`, `*_secs_prev`, `*_amt_prev`, `*_amt_cummean`) are
  computed after `sort_values('TransactionDT')` using `cumcount`, `shift(1)` and
  `(cumsum - self)` — strictly backward-looking, no future information.
- High-cardinality categorical capping uses the top 120 levels **from train only**.
- Isotonic is fit on the calib window (days 120–149) and applied to test (150–181) —
  the correct temporal ordering, no calibration leakage.

**Caveat:** `MODEL` sets `early_stopping=True, validation_fraction=0.12`. sklearn's
internal validation split is **random, not temporal** — so the model's stopping point
(400 iters) is chosen against a randomly-interleaved 12% of days 0–119. This is not
label leakage into test, but it is a mild optimism in the stopping criterion relative to
a strictly out-of-time early-stopping split. Low materiality; worth making explicit if
the model config becomes a production artifact.

### 4.3 `D1` is both a model feature and the economics driver

`D1` (days since first card activity) is in the 225-feature matrix *and* is what buckets
each transaction into a tenure segment, which sets `M_relationship`, which sets `M`,
which sets every decision boundary. Not leakage, but it creates a coupling: any drift
or corruption in `D1` moves the score and the cost function in the same direction
simultaneously, so the two error sources cannot be disentangled in monitoring. The
production metrics catalog should track `D1` distribution as a Tier-4 data-quality
signal specifically for this reason.

### 4.4 Uncorrected selection bias — disclosed, but load-bearing

The findings note it: IEEE-CIS contains only transactions the original issuer approved
(or at least attempted), with no exploration holdout. Every quantity in this report is
conditioned on that selection. `M` — the cost of declining a good customer — is
estimated on a population from which the riskiest applicants were already removed. The
break-even boundaries in §6/§7 inherit this. This is correctly flagged as unmeasurable
here, and it is the strongest argument in the document for building an exploration
holdout into the production system.

### 4.5 Tenure buckets on the LTV panel use `D1first`, transactions use `D1`

`04b_ltv.py` buckets a uid by `D1` at its *first observed* transaction; `05_economics.py`
buckets each transaction by its *own* `D1`. A uid therefore contributes its
`M_relationship` from one bucket while its later transactions are priced in another.
Defensible (the relationship value is set at acquisition), but it is an unstated
modelling choice, and it means the `n` column in `outputs/04b_ltv.log` (train-window
transaction counts) and the `n_uid` column describe different populations in the same
table.

### 4.6 Minor

- `05_economics.py` fills missing tenure mappings with `ten.M_relationship.median()`.
  Measured: no test-window rows hit that path (all seven buckets are populated), so it
  is inert here — but it is a silent fallback that would mask a future bucketing bug.
- The review-queue remap `a = np.where(a==2, 3, a)` after `e[[0,1,3]].argmax(0)` is
  correct (index 2 of the 3-row subset is action "deny"), but is fragile to anyone
  editing the action list. Verified correct as written.
- `DAYS` is derived as `d.day.nunique()` rather than `max-min+1`. Identical here (32),
  but would silently under-annualise if a day had zero transactions.

---

## 5. Bottom line

The pipeline is unusually honest and the arithmetic is sound. 47 of 50 checked
quantities reproduce to the cent from the persisted artifacts, including every headline
number: the $2,448,270 policy gain, the $6,624,870 miscalibration penalty, the 0.9045
AUC, the entire five-term P&L, both break-even ladders, and the label-latency collapse.

Three things need attention before these numbers are treated as production baselines:

1. **The $6.6M miscalibration row is analytic and the findings doc does not say so.**
   It is disclosed in the README and in a code comment, but it sits unmarked in a table
   next to two empirical rows. Mark it, or replace it with the `FIT_BALANCED=1`
   measurement the spec already plans.
2. **Two review-queue costs in the findings doc contradict the pipeline's own log**
   ($271 and $38 of drift, caused by argsort tie-breaking across the 153-level isotonic
   step function). Fix the tie-break determinism, then re-transcribe.
3. **The $1,999 naive LTV has no generating code and no log.** `run_all.sh` cannot
   regenerate it, which breaks the stated reproducibility guarantee for that one figure.

And one methodological note that should travel with the headline claim: **P1's optimal
threshold was tuned on the test set**, so the "intermediate actions are worth 3× the
threshold tuning" finding is understated rather than overstated. That is the right
direction for a claim to be wrong in, but it should be said out loud.
