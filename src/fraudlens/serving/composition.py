"""The composition root: the production wiring `create_app` has never had a caller for.

`app.create_app` takes a `ModelLoader` and a `DecisionWriter` and constructs neither.
That was deliberate — a module that opens a socket at import time cannot be unit tested —
but it left a hole: every caller of `create_app` in this repository is a test, so the
service has never been assembled the way it would run. This module is that assembly, and
it is the only place in `src/` that reads `os.environ` at request-serving time.

Three things are wired here and nowhere else.

**The model loader.** Two sources, in priority order: a bundle mounted into the image, or
the MLflow registry. The mounted path exists so the scoring container does not need
MLflow's dependency tree at runtime — pyproject splits the extras by deployment unit for
exactly that reason, and an image that carries MLflow anyway makes the split decorative.
So the `mlflow` import lives inside the function that needs it, and MLflow is used only to
*resolve and fetch* an artifact, never to run one.

**The artifact format.** A pickled bundle of (discriminator, calibrator, feature order)
rather than `mlflow.pyfunc`. Forced by the `CalibratedScorer` port: §4.3 budgets inference
(25 ms) and calibration (15 ms) as separate lines and the histogram is labelled by stage,
so a single `predict()` that does both would make the stage that decays first invisible.
The feature *order* is in the bundle because the request supplies a `dict` and a dict has
no order the model agreed to; assembling the row from the caller's iteration order would
mis-score silently, which is the most expensive failure mode available here.

**The ledger writer.** `streaming` is imported at module scope, not lazily. The scoring
container does carry SQLAlchemy and psycopg: auditability is ADR-0002's priority #2, and
an image that can serve decisions but not record them is not the production deployment
unit. It is MLflow, not the database, that the split was about.

Run it with::

    uvicorn fraudlens.serving.composition:build_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from sqlalchemy import create_engine

from fraudlens.serving.app import create_app
from fraudlens.serving.runtime import CalibratedScorer, DecisionWriter, ModelUnavailableError
from fraudlens.streaming.ledger import DecisionLedger, DecisionRecord

logger = logging.getLogger(__name__)

# Where the artifact comes from. Absolute path to a pickled bundle, or empty to fall
# through to the registry.
ENV_BUNDLE = "FRAUDLENS_MODEL_BUNDLE"
# Registry coordinates. The version is a hard pin with no "latest" fallback: "latest"
# means a replay of last Tuesday's traffic is silently decided by this Tuesday's model,
# and the resulting ledger cannot be reproduced. Unset means "whatever is in Production",
# which is resolved once at startup and then recorded per decision.
ENV_MODEL_NAME = "FRAUDLENS_MODEL_NAME"
ENV_MODEL_VERSION = "FRAUDLENS_MODEL_VERSION"
# MLflow's own variable, deliberately not renamed to a FRAUDLENS_ one: the client reads it
# directly and a second name would let the two disagree about which server we are talking to.
ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_DATABASE_URL = "FRAUDLENS_DATABASE_URL"

DEFAULT_MODEL_NAME = "fraudlens-fraud-score"

# The file inside a bundle directory. Named rather than globbed so a directory holding two
# artifacts fails loudly instead of picking one by sort order.
BUNDLE_FILENAME = "scorer.pkl"

_REQUIRED_BUNDLE_KEYS = frozenset({"version", "feature_names", "discriminator", "calibrator"})


class BundledScorer:
    """The deployed artifact: a discriminator, the isotonic fitted on top of it, and the
    feature order the discriminator was trained with.

    Satisfies `CalibratedScorer` structurally. Not a subclass of anything — the port is a
    Protocol with two real implementations (this and the test stub), which is what keeps it
    from being a speculative abstraction.
    """

    def __init__(
        self,
        *,
        version: str,
        feature_names: Sequence[str],
        discriminator: Any,
        calibrator: Any,
    ) -> None:
        self._version = version
        self._feature_names = tuple(feature_names)
        self._discriminator = discriminator
        self._calibrator = calibrator

    @property
    def version(self) -> str:
        return self._version

    def score(self, features: Mapping[str, float]) -> float:
        """Raw discriminator output for the positive class.

        Missing features raise rather than defaulting. A zero-filled column reads to a
        gradient-boosted model as a real observation, so the decision would look scored and
        be wrong; raising sends the request down the documented fail-safe ladder instead,
        where it is marked degraded and excluded from calibration analysis. Extra features
        the model was not trained on are ignored, not an error: the caller owns feature
        assembly (FEATURE_VERSION is `request-supplied-v1`) and may legitimately send more
        than this artifact consumes.
        """
        missing = [name for name in self._feature_names if name not in features]
        if missing:
            raise ValueError(f"request is missing {len(missing)} model feature(s): {missing[:5]}")
        row = [[float(features[name]) for name in self._feature_names]]
        return float(self._discriminator.predict_proba(row)[0][1])

    def calibrate(self, raw_score: float) -> float:
        """Isotonic map from the raw score to a calibrated P(fraud).

        The value of this one line is measured: an uncalibrated score routed through the
        same EV policy costs $4.36M/yr (docs/findings/fit-balanced-empirical-result.md).
        Out-of-range output is not clamped here — `decisioning` treats it as a scoring
        failure, because a calibrator that has started returning 1.2 is broken and quietly
        clipping it would produce a plausible-looking decision from a broken model.
        """
        return float(self._calibrator.predict([raw_score])[0])


def load_scorer(pin: str | None) -> CalibratedScorer:
    """The production `ModelLoader`. Raises `ModelUnavailableError`; never returns a stub.

    `ServiceState.load` catches this and the service comes up not-ready and degraded, which
    is the designed behaviour (ADR-0002 #4). Returning a placeholder scorer here instead
    would be the single most expensive line in the system: the service would look healthy
    and would be deciding on a fabricated probability.
    """
    bundle = os.environ.get(ENV_BUNDLE, "").strip()
    if bundle:
        return load_bundle(Path(bundle))
    tracking_uri = os.environ.get(ENV_TRACKING_URI, "").strip()
    if tracking_uri:
        return load_bundle(_fetch_from_registry(tracking_uri, pin))
    raise ModelUnavailableError(
        f"no artifact source configured: set {ENV_BUNDLE} or {ENV_TRACKING_URI}"
    )


def load_bundle(path: Path) -> CalibratedScorer:
    """Read a bundle from a file or from `<dir>/scorer.pkl`.

    Every failure below is converted to `ModelUnavailableError` on purpose. The caller's
    contract is "an artifact, or the documented unavailable error"; letting an
    `UnpicklingError` or a sklearn version mismatch escape as itself would take the process
    down at startup, which converts a model problem into a checkout outage.
    """
    file = path / BUNDLE_FILENAME if path.is_dir() else path
    if not file.is_file():
        raise ModelUnavailableError(f"no scoring bundle at {file}")
    try:
        # Unpickling arbitrary bytes executes arbitrary code. The bytes here come from an
        # artifact we built and mounted into our own image, so trusting them is the same
        # act as trusting the image; the control is on what may be mounted, not on the
        # deserialiser. A safetensors-style format would not help — the payload is a
        # fitted sklearn estimator, which has no non-pickle serialisation.
        with file.open("rb") as handle:
            bundle: Any = pickle.load(handle)  # noqa: S301
    except Exception as exc:
        raise ModelUnavailableError(f"scoring bundle at {file} could not be read: {exc}") from exc
    if not isinstance(bundle, dict) or not set(bundle) >= _REQUIRED_BUNDLE_KEYS:
        missing = sorted(_REQUIRED_BUNDLE_KEYS - set(bundle if isinstance(bundle, dict) else {}))
        raise ModelUnavailableError(f"scoring bundle at {file} is missing key(s): {missing}")
    if not bundle["feature_names"]:
        # An empty feature order would score every transaction identically, which reads as
        # a working model with a degenerate score distribution rather than as a broken one.
        raise ModelUnavailableError(f"scoring bundle at {file} declares no features")
    return BundledScorer(
        version=str(bundle["version"]),
        feature_names=bundle["feature_names"],
        discriminator=bundle["discriminator"],
        calibrator=bundle["calibrator"],
    )


def _fetch_from_registry(tracking_uri: str, pin: str | None) -> Path:
    """Resolve a registry version to a local artifact directory.

    Imported here, not at module scope: this is the whole reason the loader has two
    sources. An image built with only the `serving` and `streaming` extras must be able to
    import this module and serve from a mounted bundle, and a top-level `import mlflow`
    would make that an ImportError at startup.
    """
    from mlflow.artifacts import download_artifacts
    from mlflow.tracking import MlflowClient

    from fraudlens.models.registry import current_champion

    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    name = os.environ.get(ENV_MODEL_NAME, "").strip() or DEFAULT_MODEL_NAME
    try:
        version = (
            client.get_model_version(name, pin)
            if pin
            # No pin means "whoever is champion now". Resolved once, at startup, and the
            # resolved version is what every decision records — so the ledger is still
            # reproducible even though the request did not name a version.
            else current_champion(client, name)
        )
        if version is None:
            raise ModelUnavailableError(f"no version of {name!r} is in Production")
        return Path(download_artifacts(artifact_uri=version.source))
    except ModelUnavailableError:
        raise
    except Exception as exc:
        raise ModelUnavailableError(f"registry lookup for {name!r} failed: {exc}") from exc


class LedgerWriter:
    """Adapts `streaming.ledger.DecisionLedger` to the `DecisionWriter` port.

    The port's argument names match the ledger's columns one to one specifically so this
    stays a splat and never becomes a translation — a translation layer between a decision
    and its audit record is a place where the two can disagree. The arguments are written
    out rather than swallowed by `**kwargs` for that reason: with `**kwargs` this would
    type-check forever, and the day someone renames a ledger column it would start failing
    at runtime, one row at a time, in production. Spelled out, mypy fails the build.
    """

    def __init__(self, ledger: DecisionLedger) -> None:
        self._ledger = ledger

    def record_decision(
        self,
        *,
        transaction_id: int,
        transaction_at: dt.datetime,
        decided_at: dt.datetime,
        score: float | None,
        calibrated_probability: float | None,
        action: str,
        reason_codes: tuple[str, ...],
        model_version: str,
        policy_version: str,
        feature_version: str,
        config_hash: str,
        input_hash: str,
        degraded: bool,
        degraded_reason: str | None,
    ) -> None:
        # The return value (new vs duplicate) is dropped deliberately. The replay producer
        # delivers at-least-once, so a duplicate is the expected consequence of a restart
        # rather than an incident; `DecisionLedger.record` is idempotent and the count of
        # genuinely-new rows is a ledger-side question, not a serving-side one.
        self._ledger.record(
            DecisionRecord(
                transaction_id=transaction_id,
                transaction_at=transaction_at,
                decided_at=decided_at,
                score=score,
                calibrated_probability=calibrated_probability,
                action=action,
                reason_codes=reason_codes,
                model_version=model_version,
                policy_version=policy_version,
                feature_version=feature_version,
                config_hash=config_hash,
                input_hash=input_hash,
                degraded=degraded,
                degraded_reason=degraded_reason,
            )
        )


def ledger_writer(database_url: str) -> DecisionWriter:
    """A writer over `database_url`.

    `pool_pre_ping` because the database is the component this deployment is most likely to
    restart under a running service (the degradation drill does exactly that). Without it
    the pool hands out connections that died with the server and every write fails until
    the pool churns, which turns a ten-second database restart into a minutes-long hole in
    the audit trail.
    """
    return LedgerWriter(DecisionLedger(create_engine(database_url, pool_pre_ping=True)))


def build_app() -> FastAPI:
    """Assemble the service from the environment. The uvicorn entry point.

    A missing `FRAUDLENS_DATABASE_URL` yields no writer rather than an error, because that
    is a real deployment: the load test and the k8s smoke run the scoring path without a
    ledger. It is logged at warning, since it is also how you would accidentally ship a
    service that answers correctly and records nothing.
    """
    database_url = os.environ.get(ENV_DATABASE_URL, "").strip()
    if not database_url:
        logger.warning("%s unset: decisions will be served but not recorded", ENV_DATABASE_URL)
    return create_app(
        loader=load_scorer,
        model_version_pin=os.environ.get(ENV_MODEL_VERSION, "").strip() or None,
        writer=ledger_writer(database_url) if database_url else None,
    )
