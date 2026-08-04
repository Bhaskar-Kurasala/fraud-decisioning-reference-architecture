# Closing the FIT_BALANCED gap: the miscalibration finding, measured

**Date:** 2026-08-05
**Epic:** E1
**Status:** Complete — finding survives, its framing does not

## Why this was run

`fraud-decisioning-findings.md` §2 claims miscalibration costs **$6,624,870/yr at
identical AUC**. The rhetorical force of that claim rests entirely on the word
*identical*: two models that rank equally well, one of which costs $6.6M more.

The verification pass established that the "class-rebalanced" row was never a
model. `research/03_model.py` fits only the champion; the balanced variant sits
behind `FIT_BALANCED=1`, which the README records as OOMing at 3 GB and which
the stage log confirms was never taken. `03b_calibrate.py` instead constructs
the score by closed-form prior shift:

```python
r = (1 - pi) / pi                      # pi = 0.03522 train base rate, r = 27.39x
p_bal = (p_te_raw * r) / (p_te_raw * r + (1 - p_te_raw))
```

That transform is **strictly monotone**. So "identical AUC" was not a finding —
it was arithmetic. A monotone map cannot change the ranking, and AUC depends only
on ranking. The claim's most striking feature was a tautology.

The current machine has 11 GB. The refit is now feasible, so the question is
settled by measurement rather than argument.

## Method

`FIT_BALANCED=1 python3 research/03_model.py` — a genuine
`HistGradientBoostingClassifier(class_weight='balanced')` fit on the same 225
features and the same train window, 400 iterations, 51s. Scored on the same test
window. Costed through the same four-action EV argmax policy and the same
per-transaction `L`/`M` from `data/econ_test.parquet`.

## Result

| Score | AUC | PR-AUC | ECE | Annual cost | Allow | Challenge | Deny |
|---|---|---|---|---|---|---|---|
| Champion (isotonic) | 0.9045 | 0.5271 | 0.0035 | $2,799,214 | 92.8% | 6.8% | 0.35% |
| Raw uncalibrated | 0.9050 | 0.5391 | 0.0054 | $2,871,342 | 94.6% | 4.9% | 0.49% |
| Rebalanced — **analytic** | 0.9050 | 0.5391 | 0.2009 | $9,434,296 | 26.8% | 71.2% | 1.97% |
| Rebalanced — **empirically refit** | **0.9029** | **0.5167** | **0.1389** | **$7,168,825** | 46.5% | 52.2% | 1.27% |

Agreement between the two rebalanced scores: Spearman 0.907, max absolute
probability difference 0.655.

## What changed and what did not

**The headline number is overstated by 38%.** The measured miscalibration
penalty is **$4,369,611/yr** ($7,168,825 − $2,799,214), not $6,624,870. The
analytic construction exaggerated it because a pure prior shift inflates every
probability by the same odds factor, whereas a real rebalanced fit also
*restructures the trees* — it partially re-learns the problem and lands closer to
the truth than the naive transform implies.

**"At identical AUC" is false as measured.** The real refit does not rank
identically: AUC falls 0.9050 → 0.9029 and PR-AUC 0.5391 → 0.5167. Rebalancing
costs a little discrimination as well as a lot of calibration.

**The finding itself survives, and is arguably stronger.** A model whose AUC
drops by 0.002 — a difference no reasonable review process would block, and one
most teams would call noise — costs **$4.37M/yr more** once routed through an
EV policy, because its ECE is 40× worse. It challenges 52% of all traffic
instead of 7%.

That is the operationally important claim, and it is now measured rather than
constructed:

> A model can be indistinguishable from the champion on the metric you gate
> promotion with, and still be catastrophically more expensive to operate.

## Consequences for the production design

1. **The promotion gate must check calibration, not just discrimination.** A
   gate keyed on AUC accepts a model at ΔAUC = −0.002 that costs $4.37M/yr. This
   is no longer a hypothetical justification for the gate ordering in the design
   spec (§4.2) — it is a measured counterexample.
2. **ECE belongs in the alerting tier, not the reporting tier.** The gap between
   ECE 0.0035 and 0.1389 is invisible to every ranking metric and worth more
   than the entire fraud-loss reduction the project set out to capture.
3. **The findings doc needs correcting** — both the number and the "identical
   AUC" framing. Handled in a separate commit so the correction is legible.

## Provenance

Artifact: `data/p_te_bal_fitted.npy` (regenerable via
`PYTHONPATH=. FIT_BALANCED=1 python3 research/03_model.py`).
Log: `outputs/03_model_fitbalanced.log`.
Costing reproduced against `data/econ_test.parquet` using the constants in
`config.py`; the champion figure recomputed here ($2,799,214) sits $583 below the
published $2,799,797, within the tie-breaking noise documented in the
verification report §3.1.
