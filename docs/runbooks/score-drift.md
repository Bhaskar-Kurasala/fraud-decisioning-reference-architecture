# Runbook: FraudLensScoreDrift

**Alert:** `fraudlens_score_psi > 0.25` for 30m · **Tier 0** · dashboard `03 Data quality`

## Symptom

The distribution of calibrated scores the model is producing has moved materially away
from the training baseline. PSI above 0.25 is the conventional "the population is not the
one you fitted on" line, and it is the value `monitoring.drift.PSI_ALERT_THRESHOLD` uses.

This is the earliest warning the system has. It fires with **no labels involved**, which
is the entire reason it exists: the chargebacks that would confirm or refute it settle
over 30–90 days, and by then the loss has landed.

## Likely cause, in the order they actually occur

1. **An upstream feature pipeline changed.** A join that started returning nulls, a
   currency or unit change, a new merchant category flowing in. Cheapest to confirm and
   by far the most common.
2. **Traffic mix shifted.** A marketing push, a new geography, a BIN range, a partner
   integration going live. The model is fine; it is being asked a different question.
3. **The population genuinely moved.** Adversaries adapt. This is the case the flywheel
   exists for, and it is the *least* likely of the three on any given page.
4. **A model was promoted.** Check `fraudlens_model_loaded` and the version attribute on
   a recent trace before anything else — a drift alert 20 minutes after a deploy is a
   deploy, not drift.

## First diagnostic query

```promql
# Which features moved, not just that something did.
topk(10, fraudlens_feature_psi)

# Is the decision mix moving with it? If PSI is up and the mix is flat, the drift is in
# a region of the score space the boundary does not care about.
fraudlens:decisions:rate5m / scalar(sum(fraudlens:decisions:rate5m))
```

Then, on `01 Model health`, compare the serving p50/p95 predicted probability against the
week before. A shift in the mean with a stable shape is a base-rate change; a change in
shape at a stable mean is usually a feature.

## Remediation

- **Do not retrain reflexively.** A retrain on drifted-but-mislabelled input bakes the
  breakage in, and the promotion gate cannot catch it because the gate needs matured
  labels which do not exist yet.
- If a specific feature dominates `fraudlens_feature_psi`, treat this as a data incident
  and page the owning pipeline. Fixing the input is the fix.
- If drift is broad and no feature dominates, the population moved. Trigger the flywheel
  (E8) to train a challenger and let the gate decide on matured labels. Interim risk is
  carried, deliberately, rather than by promoting on a hunch.
- Only if the decline rate is also anomalous is any immediate policy action justified —
  and then see the decline-anomaly runbook, not this one.

## Who decides

The on-call engineer diagnoses and can page a pipeline owner. **Changing the policy or
promoting a model is not an on-call decision** — it is the fraud risk owner's, because
both move money and both are subject to the promotion gate in §4.2. If drift is severe
enough that someone wants to act before the gate can run, that is an explicit,
time-boxed, documented risk acceptance by the risk owner, not a config change.
