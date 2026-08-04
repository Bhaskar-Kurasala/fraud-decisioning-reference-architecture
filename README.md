# Fraud Decisioning: model → calibration → economics → policy

An end-to-end reproduction of cost-sensitive fraud decisioning on real payment
data. The point is not the model. The point is what happens to the model's
score after it is produced: calibration, per-transaction expected value,
capacity constraints, and the policy layer that turns a probability into an
action.

Headline result: on this data the **choice of action set is worth 3x more than
any threshold tuning**, and an **uncalibrated-but-equally-accurate score costs
$6.6M/yr** while showing an identical AUC.

See `fraud-decisioning-findings.md` for the full write-up.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

`run_all.sh` downloads the data (~680MB), verifies checksums, and runs all
seven stages, logging each to `outputs/`. Roughly 15 minutes on 1 core / 3GB.

To change the business assumptions, edit `config.py` and rerun stages 5–6 only:

```bash
python3 05_economics.py && python3 06_full.py
```

---

## Data

**IEEE-CIS Fraud Detection** — real Vesta Corporation e-commerce transactions.

| | |
|---|---|
| Source | `huggingface.co/datasets/aliceczr/ieee-fraud-detection` |
| Original | Kaggle `ieee-fraud-detection` (requires competition acceptance) |
| Files | `train_transaction.csv` (683MB), `train_identity.csv` (26MB) |
| MD5 | `58b4038d8715f5e11007b826bef00ce7`, `8487db5001c8bad139f3318d5d3db416` |
| Rows | 590,540 transactions over 182 days |
| Labels | 20,663 fraud (3.50%) |
| GMV | $79,738,949 |

Chosen over the more common ULB credit-card dataset because the economics need
real transaction amounts, a long time axis, and entity identifiers. ULB has
PCA-anonymised features and two days of history — you cannot build a business
layer on it.

---

## Pipeline

| Stage | Script | Produces | Runtime |
|---|---|---|---|
| 1 | `01_profile.py` | base rates, amount distribution, temporal drift, segments | ~1 min |
| 2 | `02_features.py` | `X.parquet`, `meta.parquet`, `feats.json` — 225 features, out-of-time split | ~4 min |
| 3 | `03_model.py` | `p_ca_raw.npy`, `p_te_raw.npy` — champion GBDT | ~3 min |
| 3b | `03b_calibrate.py` | `scored_test.parquet` — isotonic calibration + reliability | ~10 s |
| 4 | `04b_ltv.py` | `tenure_econ.csv` — survivorship-corrected LTV per tenure | ~2 min |
| 5 | `05_economics.py` | `econ_test.parquet`, `policies.csv` — EV, policies, boundaries | ~20 s |
| 6 | `06_full.py` | miscalibration cost, capacity, sensitivity, label latency, P&L | ~30 s |

Stages 5 and 6 are the analysis; 1–4 exist to feed them.

### Split

Strictly out-of-time. Day index = `(TransactionDT − min) // 86400`.

```
train  day   0–119   414,542 txns   3.52% fraud
calib  day 120–149    83,571 txns   3.41% fraud   <- isotonic fitted here only
test   day 150–181    92,427 txns   3.48% fraud   <- every reported number
```

The calibration slice is separate from test. Fitting isotonic on test would
make the calibration results meaningless.

### Leakage controls

- Frequency encodings and entity amount-aggregates fitted on the **train window
  only**, then mapped forward. Unseen levels → 0 / NaN.
- Velocity features (`*_secs_prev`, `*_amt_prev`, `*_amt_cummean`) built with
  `shift`/`cumsum` on time-sorted data — backward-looking by construction.
- Customer proxy `uid = card1_card2_addr1_(D1−day)` is the standard IEEE-CIS
  construction: same physical card + address + first-seen date.

Out-of-time **AUC 0.9045, PR-AUC 0.527**. This is deliberately lower than the
~0.94 on the Kaggle leaderboard, where entity aggregates are computed over the
combined train+test set. That leaks future information and is not available at
decision time.

---

## Reproducibility

**Determinism.** Seeds in `config.py`: `SEED=42` (model), `SEED_LABEL_SIM=7`
(chargeback-lag simulation). HistGradientBoosting with a fixed `random_state`
is deterministic for a given thread count; results were validated single-core.
Multi-core runs may differ in the last decimal of AUC via floating-point
reduction order. All economics downstream of the saved `.npy` scores are
exactly deterministic.

**Constants.** Every business assumption lives in `config.py` and nowhere else.
No script hardcodes a cost. If a number appears in the report it traces to
`config.py` or to the data.

**Environment.** `requirements.txt` is pinned; `ENVIRONMENT.txt` records the
validated platform (Python 3.12.3, x86_64, scikit-learn 1.8.0).

**Validation.** Refactoring stages 2–6 to read from `config.py` reproduced every
figure bit-identically ($2,799,797 P4 annual cost, $6,624,870 miscalibration
penalty, 0.328–0.789 boundary spread).

---

## Business layer

Costs are per transaction, not global:

```
L_i = Amount_i × COGS + CB_FEE + OPS_DISPUTE          # false negative
M_i = Amount_i × MARGIN + relationship_cost(tenure_i)  # false positive
```

`relationship_cost = P(churn|declined) × residual_LTV`, both per tenure bucket.

**`residual_LTV` is measured**, not assumed. Derived in `04b_ltv.py` from
observed spend, corrected for survivorship: conditioning on customers with ≥2
transactions inflated new-customer LTV from $323 to $1,999. The fix measures
`P(repeat)` directly (44.5% across 151,928 customers observable for ≥60 days)
and takes `LTV = P(repeat) × value_if_repeat × discount`.

**`P(churn|declined)` is the one input with no data support here.** It is
tenure-graded from published false-decline research. It is the softest number
in the model and the first thing to challenge.

---

## Known limitations

1. **No exploration holdout.** This data contains only transactions the original
   issuer approved and observed. Selection bias from their declines is
   uncorrected and, with this dataset, unmeasurable. A real deployment needs a
   stratified holdout to estimate the counterfactual.
2. **Challenge and review outcomes are counterfactual.** Realised P&L uses true
   labels for the loss and expected values for the intervention. `F_PASS`,
   `A_ABANDON` and `Q_ANALYST` are vendor/ops parameters, not fitted — hence
   the sensitivity grid in stage 6.
3. **`P(churn|declined)` is assumed** (see above).
4. **Label latency is simulated, not observed.** IEEE-CIS labels are final. The
   stage-6 experiment injects a lognormal chargeback lag to demonstrate the
   failure mode; it does not measure it in this data.
5. **The class-rebalanced comparison is analytic by default.** Training at a
   50/50 effective prior multiplies the odds by `(1−π)/π`; this closed form is
   exactly what an uncorrected `class_weight='balanced'` fit produces, and it is
   exactly reproducible. To refit empirically instead:
   `FIT_BALANCED=1 python3 03_model.py` — needs ~6GB, OOMs on 3GB.
6. **Single merchant, single 182-day window, one adversary regime.** The
   direction of the findings should generalise; the magnitudes should not be
   transplanted.
7. **No fairness or disparate-impact analysis.** IEEE-CIS carries no protected
   attributes. For that dimension use the Bank Account Fraud (NeurIPS 2022)
   dataset, which ships protected attributes deliberately.

---

## Files

```
config.py               all constants, paths, seeds
requirements.txt        pinned dependencies
ENVIRONMENT.txt         validated platform
00_download_data.sh     data acquisition + checksum verification
run_all.sh              full pipeline
01..06_*.py             stages
fraud-decisioning-findings.md   the write-up
```
