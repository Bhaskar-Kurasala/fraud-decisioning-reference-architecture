# Runbook: FraudLensPromotionGateFailure

**Alert:** `increase(fraudlens_promotion_gate_failures_total[6h]) > 0` · **Tier 2** ·
dashboard `01 Model health`

## Symptom

A challenger model was evaluated against the champion and **rejected**.

Read this first: §4.2 states plainly that *failing the gate is a normal outcome and is
logged, not suppressed*. This alert is not "something broke". It fires because the
alternative — a gate that quietly rejects every challenger while the champion ages past
the population it was fitted on — is invisible for a quarter, and a system whose retrain
loop has silently stopped working looks exactly like one that never needed to retrain.

**The action is to read the gate report. It is never to promote.**

## Which check failed, and what each means

The gate checks in order, and the first failure is the diagnosis:

1. **Label maturity** — refused to evaluate at all. Not a model verdict. The challenger is
   untested, and forcing an evaluation would measure early-disputing fraud rather than the
   model (the observed fraud rate collapses to 6.9% of true in the final 20 days of a
   window). Wait, or widen the evaluation window backwards.
2. **Calibration** — ECE regressed beyond tolerance. The most likely rejection and the one
   most worth respecting: the measured case is a challenger 0.0021 AUC below champion that
   would cost **$4.36M/yr** through calibration alone. A challenger that looks better on
   every ranking metric and fails here is the gate doing precisely the job it exists for.
3. **Cost delta** — the paired sequential test on per-transaction cost did not clear its
   confidence interval. Either the challenger is not better, or there is not yet enough
   evidence. These are different situations and the CI distinguishes them: a delta near
   zero with a wide interval means keep collecting; a delta clearly positive means stop.
4. **Segment guard** — aggregate improved but a segment regressed beyond tolerance. Usually
   new accounts or the low-amount band, because those carry the highest break-even (0.642
   new, 0.731 under $25) and are the easiest to quietly sacrifice for an aggregate win.
   New-customer approval is the growth metric; this guard is protecting it.

## First diagnostic query

```promql
increase(fraudlens_promotion_gate_failures_total[6h])

# What the gate was looking at.
fraudlens_challenger_cost_delta_usd
fraudlens_challenger_cost_delta_ci_low
fraudlens_challenger_cost_delta_ci_high
fraudlens_ece
fraudlens_label_maturity_ratio{horizon_days="30"}

# The reason this alert exists at all: is the champion aging while nothing promotes?
fraudlens_days_since_retrain
```

The gate report in MLflow carries the per-check verdicts. Start there, not here.

## Remediation

- **Single failure:** record it and move on. This is the system working.
- **Repeated failures on the same check:** that is the real signal. Persistent calibration
  rejections point at the training recipe, not at the gate. Persistent maturity refusals
  mean the retrain cadence is faster than the label pipeline can support, which is a
  scheduling problem.
- **`days_since_retrain` climbing while failures accumulate:** the loop is stuck. This is
  the situation the alert was written for, and it needs an owner, not another retrain.
- **Never lower a gate threshold to let a challenger through.** The thresholds are the only
  thing standing between this system and a model that is indistinguishable from champion on
  the metric you gate with and catastrophically more expensive to run. If a threshold is
  genuinely wrong, changing it is an ADR with evidence, not an operational tweak.

## Who decides

The gate decides promotion. No human overrides it in-flight — that is the entire design.
The model owner investigates repeated failures and owns changes to the training recipe.
Changing a gate threshold or accepting a challenger that failed requires the fraud risk
owner and a written record, because both mean accepting a measured expected loss.
