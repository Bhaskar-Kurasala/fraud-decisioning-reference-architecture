"""The card must be a function of the run, including where the run is silent.

Two failure modes are being ruled out. First, a card that quietly omits a metric the run
does not carry — the reader then cannot tell "not measured" from "not shown". Second, a
card that renders a clean reproducibility section for an artifact built from a dirty
tree, which is the exact case the manifest exists to catch and the exact case a
hand-written card gets wrong.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from fraudlens.config import SETTINGS
from fraudlens.lineage.manifest import Manifest, build_manifest
from fraudlens.lineage.model_card import NOT_RECORDED, CardSource, read_run, render
from fraudlens.models.metrics import ModelMetrics
from fraudlens.models.provenance import Provenance
from fraudlens.models.registry import register_model
from fraudlens.models.tracking import TrainingRun, build_signature, ensure_experiment
from fraudlens.models.tracking import log_training_run as log_run

NOW = dt.datetime(2026, 8, 5, tzinfo=dt.timezone.utc)
MODEL_NAME = "fraudlens-fraud-score"

# The champion's published figures (fraud-decisioning-findings.md §2). Real numbers
# rather than 1.0s so a rendering bug that swaps two rows is visible to a reader of the
# test rather than only to an assertion.
METRICS = ModelMetrics(
    n=92427,
    auc=0.9045,
    pr_auc=0.5271,
    ece=0.0027,
    calibration_slope=0.9318,
    calibration_intercept=-0.1768,
    brier=0.0220,
    expected_cost_per_txn=2.6552,
)


@pytest.fixture
def source(client: Any) -> CardSource:
    """A registered model version backed by a real run in a throwaway MLflow store."""
    run = TrainingRun(
        run_name="champion-isotonic",
        params={
            "model_type": "HistGradientBoostingClassifier",
            "train_days": "0-119",
            "calib_days": "120-149",
            "test_days": "150-181 (out-of-time)",
            "calibration_method": "isotonic, fit on the calibration window, clipped",
            "policy": "EV argmax, review arm off",
            # `seed` is deliberately not logged: the "not recorded" path is a real state
            # of this repository's training driver, not a hypothetical one.
        },
        metrics=METRICS,
        provenance=Provenance(
            git_sha="a" * 40,
            git_dirty=False,
            config_hash="c" * 64,
            data_checksum="d" * 64,
        ),
        signature=build_signature(("C1", "C13")),
    )
    run_id = log_run(client, ensure_experiment(client, "test"), run, NOW)
    version = register_model(client, MODEL_NAME, source=f"runs:/{run_id}/model", run_id=run_id)
    return read_run(client, MODEL_NAME, version)


@pytest.fixture
def manifest(tmp_path: Path) -> Manifest:
    (tmp_path / "scored_test.parquet").write_bytes(b"rows")
    clean = build_manifest(
        repo_root=tmp_path,
        artifact=f"{MODEL_NAME} v1 model card",
        config=SETTINGS.model_dump(),
        data_paths=[tmp_path / "scored_test.parquet"],
        seeds={"model_fit": 42},
        regeneration_command="uv run python scripts/regenerate_model_card.py",
        now=NOW,
    )
    # The rendering tests are about how each provenance state is *presented*; whether
    # those states are detected correctly is `test_manifest.py`. tmp_path is neither a
    # git checkout nor next to a lockfile, so the pins are set here explicitly rather
    # than by standing up a second throwaway repository.
    return replace(clean, git_dirty=False, lock_digest="l" * 64)


def test_the_card_reports_the_runs_measurements_not_a_transcription(
    source: CardSource, manifest: Manifest
) -> None:
    card = render(source, manifest, NOW)

    assert "0.9045" in card
    assert "0.0027" in card
    assert "2.6552" in card
    assert "92427" in card
    assert "150-181 (out-of-time)" in card
    assert "isotonic" in card
    assert source.run_id in card


def test_a_param_the_run_never_logged_is_shown_as_missing(
    source: CardSource, manifest: Manifest
) -> None:
    """A blank in a model card is a finding about the training driver.

    Rendering "seed: 42" from a constant would make the card claim a pin the run does not
    actually carry, which is worse than an empty row because it cannot be discovered.
    """
    assert "seed" not in source.params
    assert NOT_RECORDED in render(source, manifest, NOW)


def test_a_dirty_manifest_makes_the_card_say_the_artifact_is_not_reproducible(
    source: CardSource, manifest: Manifest
) -> None:
    dirty = replace(manifest, git_dirty=True)

    card = render(source, dirty, NOW)

    assert "not exactly reproducible" in card
    assert "modified working tree" in card
    assert manifest.regeneration_command in card


def test_a_clean_manifest_does_not_claim_a_caveat(source: CardSource, manifest: Manifest) -> None:
    card = render(source, manifest, NOW)

    assert "not exactly reproducible" not in card
    assert "Every pin above was present" in card


def test_the_card_carries_the_limitations_at_the_findings_confidence(
    source: CardSource, manifest: Manifest
) -> None:
    """Stated no more firmly than docs/findings/verification-report.md states them.

    The LTV line is the one that matters: the naive $1,999 figure has no generating code
    and no log, so the card must call the survivorship effect "roughly 6x" rather than
    quoting a number as if it were measured.
    """
    card = render(source, manifest, NOW)

    assert "selected on the test set" in card
    assert "No confidence intervals" in card
    assert "roughly 6x" in card
    assert "$1,999" not in card


def test_both_provenance_stamps_are_shown_so_a_mismatch_is_visible(
    source: CardSource, manifest: Manifest
) -> None:
    """Training-time and card-generation-time provenance are different facts.

    Showing only one hides the case where a card is regenerated from a different checkout
    than the model was trained on — the common way a card starts lying with nobody
    editing it.
    """
    card = render(source, manifest, NOW)

    assert "a" * 40 in card
    assert manifest.config_hash in card
    assert "c" * 64 in card


def test_a_naive_generation_timestamp_is_refused(source: CardSource, manifest: Manifest) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        render(source, manifest, dt.datetime(2026, 8, 5))  # noqa: DTZ001
