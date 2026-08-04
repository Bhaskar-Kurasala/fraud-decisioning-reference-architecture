# Runbook: FraudLensCalibrationDecay

**Alert:** `fraudlens_ece > 0.02` while 30-day label maturity ≥ 0.80, for 1h · **Tier 2** ·
dashboard `01 Model health`

## Symptom

Expected calibration error on matured labels has passed 0.02. The model's probabilities no
longer mean what they say, and the EV policy consumes probabilities — not rankings.

This is the most expensive failure this system has, and the least visible. The measured
case: a class-rebalanced refit scored **AUC 0.9029 against the champion's 0.9045** — a gap
of 0.0021 that no review process blocks and most teams would call noise — with ECE 0.1389
instead of 0.0027. Routed through the same policy it challenged **52.2% of traffic instead
of 6.8%** and cost **$4.36M/yr** more. Full derivation in
`docs/findings/fit-balanced-empirical-result.md`.

The 0.02 threshold sits deliberately between the two measured points: ~7× the champion, so
well outside normal variation, and ~7× below the known-catastrophic model, so there is room
to act before the money is gone.

## The maturity gate is part of the alert

The rule will not fire below 80% matured labels at 30 days. An ECE computed on 20% of the
labels is a statement about *early-disputing fraud*, not about the model — and the measured
shape of that bias is severe: the observed fraud rate collapses to **6.9% of the true rate**
in the final 20 days of any window. Without the gate this alert would page on the shape of
the dispute pipeline. §4.2 applies the same refusal to the promotion gate.

If you are reading this because someone asked "why didn't calibration alert sooner" — that
is why, and it is a deliberate trade, not a gap.

## Likely cause

1. **Population shift the model has not been refitted for.** Check `fraudlens_score_psi`
   and the score-drift runbook; drift usually precedes calibration decay by weeks.
2. **Base-rate shift.** Compare `fraudlens_actual_fraud_rate` against
   `fraudlens_predicted_fraud_rate`. A parallel offset is a prior shift, which is the
   cheapest form of this to fix — recalibrate, do not retrain.
3. **The calibrator was fitted on the wrong slice.** Isotonic regression on days 120–149;
   if a training change let the calibration split leak into training, calibration is
   optimistic by construction. The leakage regression test exists for this.
4. **A promotion that passed the gate on a stale maturity reading.**

Use slope and intercept, not just ECE. Slope below 1 means over-confidence, which the EV
argmax converts directly into over-declining; that tells you which direction the money is
leaking before you decide what to do.

## First diagnostic query

```promql
fraudlens_ece
fraudlens_calibration_slope
fraudlens_calibration_intercept

# Is it a prior shift (parallel) or a shape problem?
fraudlens_actual_fraud_rate
fraudlens_predicted_fraud_rate

# Confirm the evaluation is honest before acting on it.
fraudlens_label_maturity_ratio{horizon_days="30"}

# Did the serving-side distribution move first? It usually did.
fraudlens_score_psi
```

## Remediation

- **Recalibrate before retraining.** If discrimination is intact (AUC/PR-AUC steady) and
  only calibration decayed, refitting the isotonic layer on a recent matured slice is
  cheaper, faster and lower-risk than a new model. It is also the intervention with the
  largest measured payoff in this system's history.
- If discrimination has decayed too, trigger the flywheel and let the promotion gate
  decide on cost, not on AUC.
- **Never disable calibration to "get back to raw scores".** Raw uncalibrated costs
  $71,545/yr more than isotonic, and that is the *good* case.
- Any candidate must clear the §4.2 gate: maturity, then ECE tolerance, then paired cost
  delta with a CI, then the segment guard.

## Who decides

The model owner decides recalibrate-vs-retrain. **Promotion is the gate's decision, not a
person's** — that is the point of having one. The fraud risk owner is informed because the
interval between decay and remediation carries a quantifiable expected loss, and someone
has to own carrying it.
