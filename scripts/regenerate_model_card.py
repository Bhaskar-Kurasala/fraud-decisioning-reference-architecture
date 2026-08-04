#!/usr/bin/env python
"""Regenerate the champion's MLflow run and its model card from the persisted artifacts.

§9a: "Any published number regenerable by a single documented command." The card is a
published artifact, so it needs one, and it needs the run behind it to be regenerable
too — a card generated from a run nobody can recreate is a screenshot.

    uv run --extra tracking --extra streaming python scripts/regenerate_model_card.py

Reads `data/scored_test.parquet` (the persisted output of `research/03b_calibrate.py`),
re-derives the metric set through the *deployed* three-action policy, logs it as an
MLflow run with the provenance triple, registers it, and renders the card.

It re-scores nothing. Regenerating the score itself is `bash run_all.sh` (~15 min);
this script starts from that pipeline's output on purpose, so the card can be refreshed
after a config change without a retrain, and so the card's data checksum pins the exact
artifact the numbers came from.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from fraudlens.config import SETTINGS
from fraudlens.economics import (
    false_negative_cost,
    false_positive_cost,
    realised_cost,
    tenure_bucket,
)
from fraudlens.lineage.manifest import build_manifest
from fraudlens.lineage.model_card import read_run, render
from fraudlens.models.metrics import evaluate_scores
from fraudlens.models.provenance import collect_provenance
from fraudlens.models.registry import Stage, register_model, transition_stage
from fraudlens.models.tracking import (
    TrainingRun,
    build_signature,
    ensure_experiment,
    local_tracking_uri,
    log_training_run,
)
from fraudlens.policy import decide

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SCORED = DATA / "scored_test.parquet"
FEATURES = DATA / "feats.json"
CARD_DIR = REPO_ROOT / "docs" / "model-cards"
EXPERIMENT = "fraudlens-champion"
MODEL_NAME = "fraudlens-fraud-score"
COMMAND = "uv run --extra tracking --extra streaming python scripts/regenerate_model_card.py"

# The deployed policy, matching `serving.decisioning.INCLUDE_REVIEW`. Stated rather than
# defaulted: the four-action variant costs $583/yr more on this window, and quoting a
# cost metric without saying which policy produced it is how those two numbers get
# conflated.
INCLUDE_REVIEW = False


def load_root_config() -> ModuleType:
    """Import the repository's root `config.py` without mutating `sys.path`.

    The seeds and the temporal split live there (ADR-0001: it is the provenance copy the
    research scripts import). Reading them rather than restating them is what stops the
    card's "training window: days 0-119" from surviving a change to `TRAIN_END`.
    """
    spec = importlib.util.spec_from_file_location("fraudlens_root_config", REPO_ROOT / "config.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--now",
        default=None,
        help="ISO-8601 timestamp to stamp the run and the card with. Defaults to now.",
    )
    args = parser.parse_args()
    now = (
        dt.datetime.now(dt.timezone.utc)
        if args.now is None
        else dt.datetime.fromisoformat(args.now)
    )
    if now.tzinfo is None:
        raise SystemExit("--now must carry a timezone offset")

    if not SCORED.exists():
        # A setup state, not a failure: `data/` is git-ignored (652 MB of raw CSV).
        # Naming the regeneration command is the difference between a blocked reader and
        # a reader who runs one line.
        print(f"{SCORED} missing — regenerate with `bash run_all.sh`", file=sys.stderr)
        return 1

    cfg = load_root_config()
    scored = pd.read_parquet(SCORED)
    y = scored["isFraud"].to_numpy(dtype=np.int64)
    p = scored["p"].to_numpy(dtype=np.float64)
    amount = scored["TransactionAmt"].to_numpy(dtype=np.float64)
    tenure = tenure_bucket(scored["D1"].to_numpy(dtype=np.float64))
    fn = false_negative_cost(amount)
    fp = false_positive_cost(amount, tenure)
    actions = decide(p, fn, fp, include_review=INCLUDE_REVIEW)
    metrics = evaluate_scores(y, p, realised_cost(actions, y, fn, fp))

    provenance = collect_provenance(REPO_ROOT, SETTINGS.model_dump(), [SCORED])
    feature_names = tuple(json.loads(FEATURES.read_text()))
    run = TrainingRun(
        run_name="champion-isotonic",
        params={
            "model_type": "HistGradientBoostingClassifier",
            "train_days": f"0-{cfg.TRAIN_END - 1}",
            "calib_days": f"{cfg.TRAIN_END}-{cfg.CALIB_END - 1}",
            "test_days": f"{cfg.CALIB_END}-{int(scored['day'].max())} (out-of-time)",
            "calibration_method": "isotonic, fit on the calibration window, clipped",
            "policy": f"EV argmax, review arm {'on' if INCLUDE_REVIEW else 'off'}",
            "seed": cfg.SEED,
            **{f"hgb_{k}": v for k, v in cfg.MODEL.items()},
        },
        metrics=metrics,
        provenance=provenance,
        signature=build_signature(feature_names),
    )

    uri = local_tracking_uri(REPO_ROOT / "mlruns")
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    run_id = log_training_run(client, ensure_experiment(client, EXPERIMENT), run, now)
    version = register_model(client, MODEL_NAME, source=f"runs:/{run_id}/model", run_id=run_id)
    transition_stage(
        client,
        MODEL_NAME,
        version,
        Stage.PRODUCTION,
        justification="Champion regenerated from data/scored_test.parquet for the model card.",
        actor="scripts/regenerate_model_card.py",
        now=now,
        archive_existing=True,
    )

    manifest = build_manifest(
        repo_root=REPO_ROOT,
        artifact=f"{MODEL_NAME} v{version} model card",
        config=SETTINGS.model_dump(),
        data_paths=[SCORED],
        seeds={"model_fit": cfg.SEED, "label_sim": cfg.SEED_LABEL_SIM},
        regeneration_command=COMMAND,
        now=now,
    )

    CARD_DIR.mkdir(parents=True, exist_ok=True)
    card_path = CARD_DIR / f"{MODEL_NAME}.md"
    card_path.write_text(render(read_run(client, MODEL_NAME, version), manifest, now))
    manifest_path = CARD_DIR / f"{MODEL_NAME}.manifest.json"
    manifest_path.write_text(manifest.to_json())
    print(f"wrote {card_path} and {manifest_path} from run {run_id}")
    if not manifest.reproducible:
        for caveat in manifest.caveats:
            print(f"  caveat: {caveat}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
