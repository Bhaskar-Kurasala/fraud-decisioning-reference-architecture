"""A model card rendered from the MLflow run, not typed by hand.

A hand-written card is accurate on the day it is written and wrong from the next
retraining onward, because nothing fails when the model changes and the card does not.
So every number here is read from the run that produced the model: metrics from
`log_metric`, windows and calibration method from `log_param`, code/config/data
provenance from the tags `models.provenance` wrote. If the training driver did not log
something, the card says "not recorded" rather than inventing it — a blank in a model
card is a finding, and papering over it is how a card becomes decorative.

Two blocks are *not* read from the run, deliberately:

- **Business framing.** The card is read by people who do not build models; without the
  decision economics it feeds, "ECE 0.0027" is not actionable. This text changes when the
  system's purpose changes, not when the model is retrained, so it belongs in code
  beside the renderer rather than in a per-run parameter.
- **Standing limitations.** These are properties of the study that produced the system
  (docs/findings/verification-report.md), not of any one artifact, and they are quoted at
  the confidence the findings state them at — no firmer. "Roughly 6x" stays "roughly 6x".
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fraudlens.lineage.manifest import Manifest

NOT_RECORDED = "_not recorded in the run_"

# Params the card wants, and the label each gets. Absence is rendered, not hidden: this
# tuple doubles as the spec the training driver has to satisfy, and a card full of
# "not recorded" is the visible form of a training driver that logs too little.
PARAM_LABELS: tuple[tuple[str, str], ...] = (
    ("model_type", "Estimator"),
    ("train_days", "Training window (day index)"),
    ("calib_days", "Calibration window (day index)"),
    ("test_days", "Evaluation window (day index)"),
    ("calibration_method", "Calibration"),
    ("policy", "Policy the cost metric was computed under"),
    ("seed", "Seed"),
)

# Metrics, in the order a model validator reads them: what the model costs first, how
# well it is calibrated second, and discrimination last. That ordering is E1's finding
# expressed as layout — a model 0.0021 AUC below champion cost $4.36M/yr more, so a card
# that opens with AUC trains the reader to look at the wrong number.
METRIC_LABELS: tuple[tuple[str, str], ...] = (
    ("expected_cost_per_txn", "Expected cost per transaction ($)"),
    ("ece", "Expected calibration error"),
    ("calibration_slope", "Calibration slope (1.0 is perfect)"),
    ("calibration_intercept", "Calibration intercept (0.0 is perfect)"),
    ("brier", "Brier score"),
    ("auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("n", "Transactions evaluated"),
)

_INTENDED_USE = (
    "Scores one card-not-present transaction at checkout. The score is not the decision: "
    "it feeds a per-transaction expected-value argmax over allow / challenge / deny, so "
    "the model's *calibration* is what the money depends on, not its ranking. A "
    "well-ranking, badly-calibrated score is the specific failure this system was built "
    "after measuring — 0.0021 AUC below champion, $4.36M/yr more expensive.",
    "Out of scope: any use where the output is read as a ranking, a risk band, or a "
    "standalone probability presented to a customer. The number is only meaningful "
    "alongside the cost model it was calibrated for.",
)

_STANDING_LIMITATIONS = (
    "**P1's threshold was selected on the test set.** `research/05_economics.py` sweeps "
    "300 candidate thresholds against realised test-window cost with true labels "
    "revealed. That is an oracle choice no production team could make in advance. The "
    "bias flatters the tuned-threshold baseline, so the headline gap between it and the "
    "EV policy is conservative rather than overstated — but the specific $778,171 does "
    "not deserve the precision it is quoted with.",
    "**No confidence intervals are attached to any dollar figure.** The evaluation "
    "window contains 3,213 fraud events; sampling error on a $2.8M annualised number is "
    "not negligible, and there is no holdout for *policy* selection.",
    "**The false-positive cost rests on an unmeasurable input.** `P(churn | declined)` "
    "has no data support in IEEE-CIS; it is tenure-graded from published false-decline "
    "research. It drives the entire false-positive term and therefore every decision "
    "boundary.",
    "**The survivorship correction on LTV should be read as 'roughly 6x', not as a "
    "measured quantity.** Conditioning on >=2 transactions inflated new-customer LTV "
    "from $323.43 to about $2,000. The corrected $323.43 reproduces exactly; the naive "
    "figure was produced by a script no longer in the repository and cannot be "
    "regenerated, so it is not a measurement.",
    "**Challenge and review outcomes are counterfactual.** No transaction in IEEE-CIS "
    "was ever actually challenged, so the pass-through and abandonment rates in the cost "
    "model are vendor figures, not observations from this data.",
    "**Early stopping used a random, not temporal, validation split** "
    "(`validation_fraction=0.12`). Not label leakage into the evaluation window, but a "
    "mild optimism in the stopping point relative to a strictly out-of-time split.",
)


@dataclass(frozen=True, slots=True)
class CardSource:
    """One registered model version, flattened out of MLflow."""

    model_name: str
    version: str
    stage: str
    run_id: str
    params: Mapping[str, str]
    metrics: Mapping[str, float]
    tags: Mapping[str, str]


def read_run(client: Any, model_name: str, version: int) -> CardSource:
    """Pull the registered version and the run behind it.

    The client is injected, matching `models.tracking`: the card renderer is exercised
    against a real local store in tests rather than against a mock that would only assert
    we call the methods we think we call.
    """
    model_version = client.get_model_version(model_name, str(version))
    run_id = str(model_version.run_id)
    if not run_id:
        # A version with no run behind it has no metrics, no params and no provenance
        # tags. Rendering a card for it would produce a document that looks complete and
        # says nothing.
        raise ValueError(f"{model_name} v{version} has no source run; nothing to render")
    run = client.get_run(run_id)
    return CardSource(
        model_name=model_name,
        version=str(version),
        stage=str(model_version.current_stage),
        run_id=run_id,
        params=dict(run.data.params),
        metrics=dict(run.data.metrics),
        tags=dict(run.data.tags),
    )


def render(source: CardSource, manifest: Manifest, generated_at: dt.datetime) -> str:
    """Render the card as Markdown.

    `generated_at` is injected rather than read from the clock so regenerating the card
    from an unchanged run produces an unchanged file — a card whose only diff is a
    timestamp trains reviewers to skip the diff.
    """
    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    lines: list[str] = [
        f"# Model card — {source.model_name} v{source.version}",
        "",
        "Generated from the MLflow run by `scripts/regenerate_model_card.py`. Do not edit "
        "by hand: the next regeneration overwrites it, which is the point — a card that "
        "can be edited independently of the run it describes will disagree with it.",
        "",
        f"- Registry stage: `{source.stage}`",
        f"- Source run: `{source.run_id}`",
        f"- Card generated: {generated_at.isoformat()}",
        "",
        "## What this model is for",
        "",
    ]
    lines.extend(_paragraphs(_INTENDED_USE))
    lines.extend(["## Training and evaluation", ""])
    lines.extend(_table("Field", "Value", _param_rows(source)))
    lines.extend(["## Measured performance", ""])
    lines.extend(_table("Metric", "Value", _metric_rows(source)))
    lines.extend(
        [
            "Measured on the evaluation window named above. Expected cost is the metric "
            "the promotion gate decides on; the discrimination metrics are reported "
            "because they are conventional, not because they are sufficient.",
            "",
            "## Provenance",
            "",
        ]
    )
    lines.extend(_table("Pin", "Value", _provenance_rows(source, manifest)))
    lines.extend(_reproducibility_block(manifest))
    lines.extend(["## Known limitations", ""])
    lines.extend(f"- {item}" for item in _STANDING_LIMITATIONS)
    lines.extend(
        [
            "",
            "## Replay and dispute handling",
            "",
            "A decision made by this model can be re-derived from its ledger row with "
            "`fraudlens.lineage.replay.replay_decision`, which re-runs the cost model and "
            "the policy argmax against the recorded probability. The model arm is *not* "
            "re-run: the ledger stores a hash of the request, not the feature vector. "
            "What that does and does not prove is set out in `docs/lineage.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def _paragraphs(items: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for item in items:
        out.extend([item, ""])
    return out


def _table(left: str, right: str, rows: list[tuple[str, str]]) -> list[str]:
    lines = [f"| {left} | {right} |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.append("")
    return lines


def _param_rows(source: CardSource) -> list[tuple[str, str]]:
    return [(label, source.params.get(key, NOT_RECORDED)) for key, label in PARAM_LABELS]


def _metric_rows(source: CardSource) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, label in METRIC_LABELS:
        value = source.metrics.get(key)
        # 6 significant places: ECE separations that matter here are at the third decimal
        # (0.0027 vs 0.1389) and rounding to two would erase the finding the gate exists
        # to catch.
        rows.append((label, NOT_RECORDED if value is None else f"{value:.6g}"))
    return rows


def _provenance_rows(source: CardSource, manifest: Manifest) -> list[tuple[str, str]]:
    """Run tags first, manifest second, and both shown when they disagree.

    The run tags say what the *training* run saw; the manifest says what the machine
    rendering the card sees. Showing only one hides the case where a card is regenerated
    from a different checkout than the model was trained on, which is the common way a
    card starts lying without anyone editing it.
    """
    return [
        ("Git SHA (training run)", source.tags.get("fraudlens.git_sha", NOT_RECORDED)),
        ("Tree dirty at training", source.tags.get("fraudlens.git_dirty", NOT_RECORDED)),
        ("Config hash (training run)", source.tags.get("fraudlens.config_hash", NOT_RECORDED)),
        ("Data checksum (training run)", source.tags.get("fraudlens.data_checksum", NOT_RECORDED)),
        ("Git SHA (card generation)", manifest.git_sha),
        ("Config hash (card generation)", manifest.config_hash),
        ("Data checksum (card generation)", manifest.data_checksum or NOT_RECORDED),
        ("Dependency lock digest", manifest.lock_digest or NOT_RECORDED),
        ("Python", manifest.python_version),
        ("Seeds", ", ".join(f"{k}={v}" for k, v in sorted(manifest.seeds.items())) or NOT_RECORDED),
    ]


def _reproducibility_block(manifest: Manifest) -> list[str]:
    lines = [
        "## Reproducibility",
        "",
        f"Regenerate with:\n\n```\n{manifest.regeneration_command}\n```",
        "",
    ]
    if manifest.reproducible:
        lines.extend(
            [
                "Every pin above was present when this card was generated: the tree was "
                "clean, the input data was checksummed and the dependency lock was read.",
                "",
            ]
        )
        return lines
    lines.extend(["**This artifact is not exactly reproducible.**", ""])
    lines.extend(f"- {caveat}" for caveat in manifest.caveats)
    lines.append("")
    return lines
