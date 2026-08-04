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

Costed through the published **four-action** EV argmax (allow / challenge /
review / deny), which is the policy the findings table reports:

| Score | AUC | PR-AUC | ECE | Annual cost | Penalty | Reviewed |
|---|---|---|---|---|---|---|
| Champion (isotonic) | 0.9045 | 0.5271 | 0.0027 | $2,799,797 | — | 5 |
| Raw uncalibrated | 0.9050 | 0.5391 | 0.0054 | $2,871,342 | $71,545 | 0 |
| Rebalanced — **analytic** | 0.9050 | 0.5391 | 0.2009 | $9,424,667 | $6,624,870 | 171 |
| Rebalanced — **empirically refit** | **0.9029** | **0.5167** | **0.1389** | **$7,157,893** | **$4,358,096** | 128 |

Agreement between the two rebalanced scores: Spearman 0.907, max absolute
probability difference 0.655.

The same comparison under the three-action policy (review disallowed) gives
$2,799,214 / $2,871,342 / $9,434,296 / $7,168,825 — a penalty of $4,369,611.
The choice of policy moves the penalty by ~$11k, which is immaterial to the
conclusion but matters for reproducibility, so both are recorded and both are
locked by golden tests.

## What changed and what did not

**The headline number is overstated by 34%.** The measured miscalibration
penalty is **$4,358,096/yr** ($7,157,893 − $2,799,797), not $6,624,870. The
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

## Correction to an earlier version of this note

An earlier revision reported the champion at $2,799,214 with ECE 0.0035 and
attributed the $583 gap to the published $2,799,797 as "tie-breaking noise". That
was wrong, and E2 caught it during extraction.

The two figures are two different policies, not two measurements of one:

- $2,799,214 — three-action argmax (allow / challenge / deny)
- $2,799,797 — four-action argmax, which is what the findings table reports

The $583 is the measured cost of *offering* analyst review. It is not noise; it
is a real economic quantity, and it clears its $7.97/case cost on exactly 5 of
92,427 transactions. The four-action argmax involves no ranking and therefore no
ties, so the tie-break explanation could not have applied.

The ECE discrepancy had the same cause — a different binning scheme. The
research uses quantile *edges* with the outer bounds forced to [-1, 2];
recomputing that way gives 0.0027, matching publication.

Both errors came from reimplementing the policy and the metric from their
description rather than from the source. That is precisely the drift ADR-0001
exists to prevent, and it argues for the extracted library becoming the single
definition used by the research scripts, the tests, and the service alike.

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
