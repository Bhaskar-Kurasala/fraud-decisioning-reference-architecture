"""The composition root, asserted without a container, a registry or a database server.

Two things are load-bearing here and neither is obvious.

**Feature order.** The request supplies `features` as a JSON object and a JSON object has
no order the model agreed to. If the scorer assembled its row from the caller's iteration
order the service would keep returning 200s, keep reporting itself ready, and score every
transaction against a permuted feature vector. Nothing would fail; the model would simply
be wrong, and the first evidence would be a chargeback curve three months later. So the
order comes from the bundle and there is a test that a permuted request still scores the
same.

**`ModelUnavailableError`, not a stub.** Every failure path in the loader converts to that
one error, because `ServiceState.load` catches it and brings the service up degraded. A
loader that returned a placeholder scorer on failure would be the most expensive line in
the repository: the service would look healthy while deciding on a fabricated probability.
"""

from __future__ import annotations

import datetime as dt
import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.pool import StaticPool

from fraudlens.serving.composition import (
    BUNDLE_FILENAME,
    ENV_BUNDLE,
    ENV_DATABASE_URL,
    ENV_MODEL_VERSION,
    ENV_TRACKING_URI,
    BundledScorer,
    LedgerWriter,
    build_app,
    load_bundle,
    load_scorer,
)
from fraudlens.serving.runtime import ModelUnavailableError
from fraudlens.streaming.ledger import DecisionLedger
from fraudlens.streaming.migrate import migrate
from fraudlens.streaming.schema import decision_ledger

NOW = dt.datetime(2026, 8, 5, 14, 32, 10, tzinfo=dt.timezone.utc)


class OrderSensitiveDiscriminator:
    """Returns a probability that depends on the *position* of each value.

    A discriminator whose output ignored order could not distinguish a correctly assembled
    row from a permuted one, so the order test would pass vacuously. This one weights the
    columns unequally, which is what any real tree ensemble does.
    """

    def predict_proba(self, rows: list[list[float]]) -> list[list[float]]:
        weighted = sum(value * (index + 1) for index, value in enumerate(rows[0]))
        p = weighted / 100.0
        return [[1.0 - p, p]]


class PassThroughCalibrator:
    def predict(self, raw: list[float]) -> list[float]:
        return [min(1.0, raw[0] * 0.5)]


def write_bundle(directory: Path, **overrides: Any) -> Path:
    payload: dict[str, Any] = {
        "version": "test-bundle-v1",
        "feature_names": ["C1", "C13", "V257"],
        "discriminator": OrderSensitiveDiscriminator(),
        "calibrator": PassThroughCalibrator(),
    }
    payload.update(overrides)
    path = directory / BUNDLE_FILENAME
    path.write_bytes(pickle.dumps(payload))
    return path


@pytest.fixture
def engine() -> Iterator[Engine]:
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    migrate(eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture(autouse=True)
def _no_ambient_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader reads the environment, so an ambient value would make results depend on
    the shell the suite was started from."""
    for name in (ENV_BUNDLE, ENV_TRACKING_URI, ENV_DATABASE_URL, ENV_MODEL_VERSION):
        monkeypatch.delenv(name, raising=False)


# ======================================================================================
# The artifact
# ======================================================================================


def test_the_bundle_dictates_feature_order_not_the_request(tmp_path: Path) -> None:
    """The failure this file exists to prevent: a silently permuted feature vector."""
    scorer = load_bundle(write_bundle(tmp_path))
    ordered = scorer.score({"C1": 1.0, "C13": 2.0, "V257": 3.0})
    permuted = scorer.score({"V257": 3.0, "C1": 1.0, "C13": 2.0})
    assert ordered == permuted
    # 1*1 + 2*2 + 3*3 = 14, i.e. the bundle's order, not the request's.
    assert ordered == pytest.approx(0.14)


def test_extra_features_are_ignored_and_missing_ones_raise(tmp_path: Path) -> None:
    """Asymmetric on purpose. The caller owns feature assembly (FEATURE_VERSION is
    `request-supplied-v1`) and may legitimately send more than this artifact consumes; a
    missing column has no safe default, because zero reads to a tree model as a real
    observation and would produce a scored-looking decision from an incomplete row."""
    scorer = load_bundle(write_bundle(tmp_path))
    assert scorer.score({"C1": 1.0, "C13": 2.0, "V257": 3.0, "UNUSED": 99.0}) == pytest.approx(0.14)
    with pytest.raises(ValueError, match="missing 1 model feature"):
        scorer.score({"C1": 1.0, "C13": 2.0})


def test_a_directory_and_a_file_path_both_resolve(tmp_path: Path) -> None:
    file = write_bundle(tmp_path)
    assert load_bundle(tmp_path).version == load_bundle(file).version == "test-bundle-v1"


@pytest.mark.parametrize(
    "make,expected",
    [
        (lambda d: d / "absent", "no scoring bundle"),
        (lambda d: _write_bytes(d / BUNDLE_FILENAME, b"not a pickle"), "could not be read"),
    ],
)
def test_an_unreadable_bundle_is_model_unavailable(
    tmp_path: Path, make: Any, expected: str
) -> None:
    """Not an UnpicklingError escaping to the lifespan hook. Anything that escapes takes the
    process down, and a process that will not start cannot serve fail-safe decisions."""
    with pytest.raises(ModelUnavailableError, match=expected):
        load_bundle(make(tmp_path))


def _write_bytes(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_a_bundle_missing_a_key_names_the_key(tmp_path: Path) -> None:
    path = tmp_path / BUNDLE_FILENAME
    path.write_bytes(pickle.dumps({"version": "v1", "feature_names": ["C1"]}))
    with pytest.raises(ModelUnavailableError, match="calibrator"):
        load_bundle(path)


def test_a_bundle_with_no_features_is_refused(tmp_path: Path) -> None:
    """It would load, and then score every transaction identically — which reads as a
    working model with a degenerate score distribution, not as a broken one."""
    with pytest.raises(ModelUnavailableError, match="no features"):
        load_bundle(write_bundle(tmp_path, feature_names=[]))


def test_the_calibrated_probability_is_not_clamped() -> None:
    """`decisioning` treats out-of-range output as a scoring failure and degrades. Clamping
    it here would hide a broken calibrator behind a plausible-looking decision, and
    calibration is the $4.36M/yr line."""

    class Runaway:
        def predict(self, raw: list[float]) -> list[float]:
            return [1.7]

    scorer = BundledScorer(
        version="v", feature_names=["C1"], discriminator=None, calibrator=Runaway()
    )
    assert scorer.calibrate(0.5) == 1.7


# ======================================================================================
# Source selection
# ======================================================================================


def test_no_configured_source_is_unavailable_not_an_empty_scorer() -> None:
    with pytest.raises(ModelUnavailableError, match="no artifact source configured"):
        load_scorer(None)


def test_a_mounted_bundle_wins_over_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Priority matters operationally: pinning a bundle into the image is how you roll back
    when the registry is the thing that is broken, and it only works if it takes precedence."""
    write_bundle(tmp_path)
    monkeypatch.setenv(ENV_BUNDLE, str(tmp_path))
    monkeypatch.setenv(ENV_TRACKING_URI, "http://mlflow.invalid:5000")
    assert load_scorer(None).version == "test-bundle-v1"


def test_the_registry_path_does_not_import_mlflow_until_it_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scoring image is built without the `tracking` extra. If the import were at module
    scope this module would raise ImportError at startup in that image and the service would
    not come up at all — a strictly worse outcome than coming up degraded."""
    import sys

    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.setattr("builtins.__import__", _refuse_mlflow(__import__))
    with pytest.raises(ModelUnavailableError):
        load_scorer(None)  # no source configured; must not have needed mlflow to say so


def _refuse_mlflow(real: Any) -> Any:
    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] == "mlflow":
            raise AssertionError("mlflow was imported on a path that must not need it")
        return real(name, *args, **kwargs)

    return guarded


# ======================================================================================
# The ledger adapter
# ======================================================================================


def test_the_writer_is_a_field_for_field_splat(engine: Engine) -> None:
    """Every column arrives, with the argument names the port declares. The value of this
    test is that it fails when someone renames a ledger column and updates only one side."""
    LedgerWriter(DecisionLedger(engine)).record_decision(
        transaction_id=2987000,
        transaction_at=NOW,
        decided_at=NOW,
        score=0.81,
        calibrated_probability=0.62,
        action="deny",
        reason_codes=("SCORE_ABOVE_DENY_BOUNDARY",),
        model_version="champion-v7",
        policy_version="ev-argmax-3action-v1",
        feature_version="request-supplied-v1",
        config_hash="abc",
        input_hash="def",
        degraded=False,
        degraded_reason=None,
    )
    with engine.connect() as conn:
        row = conn.execute(select(decision_ledger)).mappings().one()
    assert row["action"] == "deny"
    assert row["calibrated_probability"] == pytest.approx(0.62)
    assert row["model_version"] == "champion-v7"
    assert row["degraded"] is False


def test_a_replayed_decision_is_absorbed_rather_than_raising(engine: Engine) -> None:
    """The replay producer delivers at-least-once, so a duplicate is the expected outcome of
    a restart. If this raised, a producer restart would count as a ledger write failure and
    the auditability metric would report an incident that did not happen."""
    writer = LedgerWriter(DecisionLedger(engine))
    for _ in range(2):
        writer.record_decision(
            transaction_id=1,
            transaction_at=NOW,
            decided_at=NOW,
            score=None,
            calibrated_probability=None,
            action="challenge",
            reason_codes=("DEGRADED_MODEL_UNAVAILABLE",),
            model_version="unavailable",
            policy_version="ev-argmax-3action-v1",
            feature_version="request-supplied-v1",
            config_hash="abc",
            input_hash="def",
            degraded=True,
            degraded_reason="DEGRADED_MODEL_UNAVAILABLE",
        )
    assert DecisionLedger(engine).count() == 1


# ======================================================================================
# The assembled service
# ======================================================================================


def test_the_assembled_service_comes_up_degraded_with_nothing_configured() -> None:
    """This is a supported deployment state, not a broken one. `docker compose up` with no
    bundle and no registry lands here, and what it must do is answer, never `allow`, and
    report itself not-ready — which is exactly what the degradation drill asserts against
    the real container."""
    with TestClient(build_app()) as client:
        ready = client.get("/health/ready")
        assert ready.status_code == 503
        assert client.get("/health/live").status_code == 200

        response = client.post(
            "/v1/decide",
            json={
                "transaction_id": 2987000,
                "transaction_at": "2026-08-05T14:32:10Z",
                "amount": 249.99,
                "days_since_first_seen": 120.0,
                "features": {"C1": 1.0},
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["action"] != "allow"
    assert body["calibrated_probability"] is None
    assert body["versions"]["model_version"] == "unavailable"


def test_the_assembled_service_scores_when_a_bundle_is_mounted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_bundle(tmp_path)
    monkeypatch.setenv(ENV_BUNDLE, str(tmp_path))
    with TestClient(build_app()) as client:
        assert client.get("/health/ready").status_code == 200
        response = client.post(
            "/v1/decide",
            json={
                "transaction_id": 2987001,
                "transaction_at": "2026-08-05T14:32:10Z",
                "amount": 249.99,
                "days_since_first_seen": 120.0,
                "features": {"C1": 1.0, "C13": 2.0, "V257": 3.0},
            },
        )
    body = response.json()
    assert body["degraded"] is False
    assert body["versions"]["model_version"] == "test-bundle-v1"
    assert body["calibrated_probability"] == pytest.approx(0.07)
