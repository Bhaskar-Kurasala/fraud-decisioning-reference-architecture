# Runbook: FraudLensTrainingServingSkew

**Alert:** `increase(fraudlens_training_serving_skew_total[1h]) > 0` · **Tier 0** ·
dashboard `03 Data quality`

## Symptom

The offline pipeline and the online service produced **different scores for the same
transaction row**. Not "slightly different" — the check is exact within tolerance, and any
mismatch fires.

This is the only alert here with a threshold of zero and no `for` clause. That is not
severity inflation. Skew is not a degree-of-badness measurement: if the two paths disagree
on one row they can disagree on any row, and every offline number the organisation holds —
the $2.4M/yr policy value, the AUC, the ECE, the promotion gate's cost delta — is
conditional on them agreeing. A skew event does not mean the model is worse. It means we
do not know what the model is.

## Likely cause

1. **Two definitions of a feature.** The failure mode the repository structure exists to
   prevent: research and serving must import the *same* builders, never reimplement them.
   Two corrections have already been needed on this project from exactly this — someone
   reimplemented logic from its prose description and the decisions were wrong by
   $71,545/yr and $583/yr respectively.
2. **Version drift between the paths.** Different artifact, different `config_hash`,
   different library version. The ledger records all three per decision, so this is
   answerable rather than arguable.
3. **Type or precision divergence.** float32 vs float64, a NaN handled as 0.0 on one side
   and as the unknown bucket on the other. `decisioning` is explicit that missing tenure
   is NaN and never 0.0, because 0.0 is a real value (a card's first transaction) — a path
   that substitutes 0.0 prices the customer as the riskiest possible one.
4. **Non-determinism.** Unseeded randomness, or dict/set ordering leaking into a feature.

## First diagnostic query

```promql
increase(fraudlens_training_serving_skew_total[1h])
```

Then leave Prometheus. This one is diagnosed in the ledger, not on a dashboard: take the
skewed `transaction_id`, pull its row, and compare `model_version`, `policy_version`,
`feature_version`, `config_hash` and `input_hash` between the two paths. `input_hash` is
the discriminator — if it matches and the scores differ, the divergence is in the model or
the runtime; if it differs, the two paths were fed different input and the feature layer is
where to look. §9a requires that a ledgered decision replays to the same output; the replay
test is the reproduction case.

## Remediation

- **Freeze promotions immediately.** The gate's inputs are computed offline. Promoting on
  a cost delta measured through a skewed path is worse than not promoting at all.
- Reproduce with the replay path on the exact `transaction_id`, then bisect: input hash,
  then feature values, then artifact version, then library versions.
- Fix by **deleting the duplicate definition**, not by reconciling the two. A tolerance
  that makes the check pass is how this returns in six months.
- Add the reproducing row to the golden-value tests. Skew that was found once and is not
  locked by a test will be found again.

## Who decides

The on-call engineer freezes promotions — that is a safe, reversible default and does not
need approval. The model owner owns the fix. **Unfreezing requires the skew count to
return to zero and the reproducing case to be in the test suite**; it is not a judgement
call about whether the discrepancy "looked small".
