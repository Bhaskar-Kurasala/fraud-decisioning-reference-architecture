# Model card — fraudlens-fraud-score v2

Generated from the MLflow run by `scripts/regenerate_model_card.py`. Do not edit by hand: the next regeneration overwrites it, which is the point — a card that can be edited independently of the run it describes will disagree with it.

- Registry stage: `Production`
- Source run: `620633e3291c41719334c9048d13213e`
- Card generated: 2026-08-04T20:53:28.639910+00:00

## What this model is for

Scores one card-not-present transaction at checkout. The score is not the decision: it feeds a per-transaction expected-value argmax over allow / challenge / deny, so the model's *calibration* is what the money depends on, not its ranking. A well-ranking, badly-calibrated score is the specific failure this system was built after measuring — 0.0021 AUC below champion, $4.36M/yr more expensive.

Out of scope: any use where the output is read as a ranking, a risk band, or a standalone probability presented to a customer. The number is only meaningful alongside the cost model it was calibrated for.

## Training and evaluation

| Field | Value |
|---|---|
| Estimator | HistGradientBoostingClassifier |
| Training window (day index) | 0-119 |
| Calibration window (day index) | 120-149 |
| Evaluation window (day index) | 150-181 (out-of-time) |
| Calibration | isotonic, fit on the calibration window, clipped |
| Policy the cost metric was computed under | EV argmax, review arm off |
| Seed | 42 |

## Measured performance

| Metric | Value |
|---|---|
| Expected cost per transaction ($) | 2.65518 |
| Expected calibration error | 0.00273141 |
| Calibration slope (1.0 is perfect) | 0.931807 |
| Calibration intercept (0.0 is perfect) | -0.176804 |
| Brier score | 0.0220098 |
| ROC-AUC | 0.904489 |
| PR-AUC | 0.527087 |
| Transactions evaluated | 92427 |

Measured on the evaluation window named above. Expected cost is the metric the promotion gate decides on; the discrimination metrics are reported because they are conventional, not because they are sufficient.

## Provenance

| Pin | Value |
|---|---|
| Git SHA (training run) | 1e94d75bf94f5066a567972e0740e53eeb5c69df |
| Tree dirty at training | true |
| Config hash (training run) | e54735d01629a246b967356b5575fed394fd8930e9c07b575695fa8acf5da6fe |
| Data checksum (training run) | 797f845e5ab1b336ce9f14d3887842775def6b69c3eb90bac99e83435f6f43eb |
| Git SHA (card generation) | 1e94d75bf94f5066a567972e0740e53eeb5c69df |
| Config hash (card generation) | e54735d01629a246b967356b5575fed394fd8930e9c07b575695fa8acf5da6fe |
| Data checksum (card generation) | 797f845e5ab1b336ce9f14d3887842775def6b69c3eb90bac99e83435f6f43eb |
| Dependency lock digest | 70612094d8d081cc399f5b8822c62bef95d2651d24ab026df4cb6eb5e2860d67 |
| Python | 3.12.12 |
| Seeds | label_sim=7, model_fit=42 |

## Reproducibility

Regenerate with:

```
uv run --extra tracking --extra streaming python scripts/regenerate_model_card.py
```

**This artifact is not exactly reproducible.**

- Produced from a modified working tree. Commit 1e94d75bf94f5066a567972e0740e53eeb5c69df does NOT identify the code that ran, so this artifact cannot be regenerated from the repository alone.

## Known limitations

- **P1's threshold was selected on the test set.** `research/05_economics.py` sweeps 300 candidate thresholds against realised test-window cost with true labels revealed. That is an oracle choice no production team could make in advance. The bias flatters the tuned-threshold baseline, so the headline gap between it and the EV policy is conservative rather than overstated — but the specific $778,171 does not deserve the precision it is quoted with.
- **No confidence intervals are attached to any dollar figure.** The evaluation window contains 3,213 fraud events; sampling error on a $2.8M annualised number is not negligible, and there is no holdout for *policy* selection.
- **The false-positive cost rests on an unmeasurable input.** `P(churn | declined)` has no data support in IEEE-CIS; it is tenure-graded from published false-decline research. It drives the entire false-positive term and therefore every decision boundary.
- **The survivorship correction on LTV should be read as 'roughly 6x', not as a measured quantity.** Conditioning on >=2 transactions inflated new-customer LTV from $323.43 to about $2,000. The corrected $323.43 reproduces exactly; the naive figure was produced by a script no longer in the repository and cannot be regenerated, so it is not a measurement.
- **Challenge and review outcomes are counterfactual.** No transaction in IEEE-CIS was ever actually challenged, so the pass-through and abandonment rates in the cost model are vendor figures, not observations from this data.
- **Early stopping used a random, not temporal, validation split** (`validation_fraction=0.12`). Not label leakage into the evaluation window, but a mild optimism in the stopping point relative to a strictly out-of-time split.

## Replay and dispute handling

A decision made by this model can be re-derived from its ledger row with `fraudlens.lineage.replay.replay_decision`, which re-runs the cost model and the policy argmax against the recorded probability. The model arm is *not* re-run: the ledger stores a hash of the request, not the feature vector. What that does and does not prove is set out in `docs/lineage.md`.
