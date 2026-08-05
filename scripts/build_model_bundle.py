#!/usr/bin/env python
"""Build a scoring bundle for the compose stack.

**This does not produce the champion, and the version string says so.** The champion is
`research/03_model.py` + `03b_calibrate.py`, and neither persists the fitted estimator —
they write out *scores* (`data/p_te_raw.npy`, `data/scored_test.parquet`) and throw the
model away. So there is no artifact in this repository to deploy, and the gap is real:
until a training run logs an estimator, the deployed stack can either run degraded or run
something that is not the champion. Pretending a re-fit here is the champion would be the
worse of the two, so the bundle is versioned `reference-*` and the model card is not
touched.

What it is for: the integration and degradation drills need a service that *can* score, so
that "the model went away" is an observable transition rather than the only state the stack
has ever been in.

Two deliberate narrowings from the research pipeline, both forced by the serving contract:

* **Numeric features only.** `DecideRequest.features` is `dict[str, float]`; there is no
  wire representation for `ProductCD='W'`. FEATURE_VERSION is `request-supplied-v1` — the
  caller assembles and encodes the vector — so an artifact deployed behind this API can
  only consume what the API can carry.
* **Fitted on a subsample.** This is a fixture for a drill, not a model anyone should
  decide with, and the run has to fit in a test setup rather than in fifteen minutes.

    uv run python scripts/build_model_bundle.py --out deploy/compose/bundles

Splits follow §6 exactly (train 0-119, calib 120-149) because the one property that must
hold even for a reference model is that the calibrator never sees a row the discriminator
was fitted on — an isotonic fitted on training scores is optimistic in exactly the
direction the $4.36M/yr calibration finding warns about.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

_NOT_FEATURES = frozenset({"isFraud", "day", "TransactionID", "TransactionDT"})

TRAIN_LAST_DAY = 119
CALIB_LAST_DAY = 149


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "deploy" / "compose" / "bundles")
    parser.add_argument("--sample", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    for name in ("X.parquet", "core.parquet"):
        if not (DATA / name).is_file():
            print(f"{DATA / name} missing — regenerate with `bash run_all.sh`", file=sys.stderr)
            return 2

    core = pd.read_parquet(DATA / "core.parquet", columns=["isFraud", "day"])
    features = pd.read_parquet(DATA / "X.parquet").select_dtypes(include=[np.number])
    # `X.parquet` carries the label and the split key alongside the model inputs. Selecting
    # "every numeric column" therefore trains on `isFraud`, which scores perfectly and is
    # worthless — found the first time this bundle was built, which is exactly the class of
    # mistake a numeric-dtype filter invites. `day` is excluded for the same reason in
    # miniature: it is the split key, so it encodes time-in-window and would let the model
    # learn the replay schedule.
    features = features.drop(columns=list(_NOT_FEATURES), errors="ignore")
    names = tuple(features.columns)

    train = core["day"] <= TRAIN_LAST_DAY
    calib = (core["day"] > TRAIN_LAST_DAY) & (core["day"] <= CALIB_LAST_DAY)
    rng = np.random.default_rng(args.seed)
    train_index = np.flatnonzero(train.to_numpy())
    if len(train_index) > args.sample:
        train_index = rng.choice(train_index, size=args.sample, replace=False)

    discriminator = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=0.1, max_leaf_nodes=31, random_state=args.seed
    )
    discriminator.fit(features.iloc[train_index], core["isFraud"].iloc[train_index])

    # Out-of-bounds clipping and the [1e-6, 1-1e-6] range are `research/03b_calibrate.py`'s
    # settings, kept because a calibrator that can return exactly 0 or 1 produces a decision
    # the cost model treats as certain, and nothing about fraud is certain.
    calib_raw = discriminator.predict_proba(features.loc[calib])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(
        calib_raw, core["isFraud"].loc[calib]
    )

    digest = hashlib.sha256(
        f"{args.seed}:{args.sample}:{len(names)}:{len(train_index)}".encode()
    ).hexdigest()[:8]
    bundle = {
        # `reference-`, never `champion-`. This string is written into every ledger row and
        # every response, so a decision taken by this artifact is distinguishable from one
        # taken by the real model forever after, without anyone having to remember.
        "version": f"reference-hgb-{digest}",
        "feature_names": list(names),
        "discriminator": discriminator,
        "calibrator": calibrator,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "scorer.pkl"
    path.write_bytes(pickle.dumps(bundle))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) version={bundle['version']}")
    print(f"features={len(names)} train_rows={len(train_index)} calib_rows={int(calib.sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
