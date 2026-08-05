"""§7's Integration row, end to end: stack up, replay, decisions land, labels reveal,
metrics move.

Each clause of that sentence is a separate way for the system to be broken in a way that
no unit test can see, because each one is a *seam*:

* the ledger schema is asserted on SQLite in the default suite and deployed on Postgres —
  the append-only triggers and the timestamp type are spelled differently on the two;
* the label revealer's "no label is readable before its reveal time" invariant is enforced
  by a WHERE clause on the read path, and a WHERE clause is a thing a real query planner
  gets to have opinions about;
* Prometheus scraping the api is configuration in a third file, and a scrape job that
  points at the wrong host produces an empty dashboard rather than an error.

The replay uses the real `ReplayProducer` rather than a loop of `client.post`. It is the
component that decides ordering and pacing, and its at-least-once contract is the reason
the ledger writes are idempotent — driving the API without it would test neither.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest
from sqlalchemy import Engine, create_engine, func, select

from fraudlens.streaming.labels import LabelRevealer
from fraudlens.streaming.replay import ReplayProducer
from fraudlens.streaming.schema import decision_ledger
from tests.deploy.conftest import BUNDLE, DATABASE_URL, GRAFANA, MLFLOW, PROMETHEUS, wait_until

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"

# Enough rows to move a histogram and to span more than one replay day, few enough that the
# drill runs in seconds. The claim under test is that the path works, not what it costs —
# that is the load test's job and it uses a different generator for it.
REPLAY_ROWS = 250
EPOCH = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
TEST_WINDOW_FIRST_DAY = 150


@pytest.fixture(scope="session")
def engine() -> Iterator[Engine]:
    eng = create_engine(DATABASE_URL, pool_pre_ping=True)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture(scope="session")
def replay_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A slice of the real test window, carrying the artifact's own feature names.

    `data/scored_test.parquet` holds the scored output but not the feature matrix, and the
    deployed artifact consumes 207 numeric columns. Sending the wrong columns would make
    every request take the fail-safe path — which still returns 200, still writes a ledger
    row, and would let this whole file pass while asserting nothing about scoring.
    """
    import pickle

    for name in ("X.parquet", "core.parquet"):
        if not (DATA / name).is_file():
            pytest.skip(f"{DATA / name} missing — regenerate with `bash run_all.sh`")
    with BUNDLE.open("rb") as handle:
        names = list(pickle.load(handle)["feature_names"])  # noqa: S301

    core = pd.read_parquet(DATA / "core.parquet", columns=["TransactionID", "isFraud", "day"])
    # `X.parquet` carries `isFraud` and `day` as well as the model inputs, so the two frames
    # overlap. Dropped from the feature side rather than the core side: `core` is the
    # producer's contract (`_REQUIRED_COLUMNS`) and the duplicate would silently become an
    # extra model feature named `isFraud` — a label leak that would score perfectly.
    features = pd.read_parquet(DATA / "X.parquet", columns=names).drop(
        columns=["isFraud", "day", "TransactionID"], errors="ignore"
    )
    window = core.index[core["day"] >= TEST_WINDOW_FIRST_DAY][:REPLAY_ROWS]
    frame = pd.concat([core.loc[window], features.loc[window]], axis=1)
    path = tmp_path_factory.mktemp("replay") / "window.parquet"
    frame.to_parquet(path)
    return path


def _wire(value: Any) -> float:
    """A feature value the request contract can carry.

    IEEE-CIS is mostly NaN — `V1..V339` are sparse by construction — and
    `HistGradientBoostingClassifier` consumes NaN natively as "missing", which is *better*
    than any imputation because the model learned a split for it. JSON cannot represent NaN,
    and `DecideRequest.features` is `dict[str, float]` with no null branch, so the wire
    format loses a signal the model wants.

    Imputing zero here is therefore a real distortion and is done knowingly, because the
    alternative is that this drill cannot send a real transaction at all. The gap belongs to
    the contract, not to this file: `features: dict[str, float | None]` would close it, and
    that is a change to `serving.contracts`, which this epic does not own. Recorded in
    docs/operations/running-the-stack.md.
    """
    number = float(value)
    return 0.0 if number != number else number


def _decide(client: httpx.Client, event: Any, names: list[str]) -> httpx.Response:
    amount = _wire(event.payload.get("TransactionAmt", 100.0))
    return client.post(
        "/v1/decide",
        json={
            "transaction_id": event.transaction_id,
            "transaction_at": event.transaction_at.isoformat(),
            "amount": amount if amount > 0 else 100.0,
            "days_since_first_seen": None,
            "features": {name: _wire(event.payload[name]) for name in names},
        },
    )


def test_replay_lands_scored_decisions_in_the_postgres_ledger(
    api: httpx.Client, engine: Engine, replay_source: Path
) -> None:
    """The seam: everything upstream of this is asserted on SQLite."""
    import pickle

    with BUNDLE.open("rb") as handle:
        bundle = pickle.load(handle)  # noqa: S301
    names = list(bundle["feature_names"])

    with engine.connect() as conn:
        before = conn.execute(select(func.count()).select_from(decision_ledger)).scalar_one()

    # Pace 0: the producer's timing is asserted in the streaming unit tests, and waiting out
    # even one simulated day here would buy nothing.
    producer = ReplayProducer(replay_source, epoch=EPOCH, real_seconds_per_replay_day=0.0)
    replayed = 0
    for event in producer.stream():
        response = _decide(api, event, names)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["degraded"] is False, body["degraded_reason"]
        assert body["versions"]["model_version"] == bundle["version"]
        producer.commit(event)
        replayed += 1

    assert replayed == REPLAY_ROWS
    with engine.connect() as conn:
        after = conn.execute(select(func.count()).select_from(decision_ledger)).scalar_one()
        rows = conn.execute(
            select(decision_ledger).order_by(decision_ledger.c.transaction_id).limit(5)
        ).mappings().all()  # fmt: skip
    assert after - before == replayed
    for row in rows:
        # A scored decision must carry its score. The pairing is a DB CHECK, so this is
        # asserting that the *serving* path filled both — a degraded decision writing NULLs
        # would also satisfy the constraint.
        assert row["calibrated_probability"] is not None
        assert row["score"] is not None
        assert row["degraded"] is False
        assert row["input_hash"] and row["config_hash"]


def test_the_replay_is_idempotent_across_a_restart(
    api: httpx.Client, engine: Engine, replay_source: Path
) -> None:
    """At-least-once delivery means a producer restart replays events it already committed.
    If the ledger were not idempotent the second pass would either raise or duplicate, and a
    duplicated decision is an audit trail that answers "what did you decide" with two rows."""
    import pickle

    with BUNDLE.open("rb") as handle:
        names = list(pickle.load(handle)["feature_names"])  # noqa: S301

    with engine.connect() as conn:
        before = conn.execute(select(func.count()).select_from(decision_ledger)).scalar_one()
    producer = ReplayProducer(replay_source, epoch=EPOCH, real_seconds_per_replay_day=0.0)
    for event in producer.stream():
        assert _decide(api, event, names).status_code == 200
    with engine.connect() as conn:
        after = conn.execute(select(func.count()).select_from(decision_ledger)).scalar_one()
    assert after == before


def test_labels_stay_invisible_until_their_simulated_dispute_lands(
    stack: None, engine: Engine
) -> None:
    """The property the whole flywheel rests on. A leak here inflates model performance for
    a quarter before anything contradicts it, because there is nothing to contradict it
    with — the labels are the ground truth.

    The labels are scheduled against transactions that were actually replayed, not against
    synthetic ids. That is not tidiness: `revealed_labels.transaction_id` is a foreign key
    to `decision_ledger`, so a label can only exist for a decision that was made. Discovered
    by writing this test with invented ids and watching Postgres refuse them — the SQLite
    mirror used by the default suite enforces the same constraint, but nothing in the suite
    had ever exercised it against a decision the *service* wrote.
    """
    del stack
    revealer = LabelRevealer(engine, label_source="integration_drill")
    transaction_at = dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc)
    with engine.connect() as conn:
        decided = [
            int(row)
            for row in conn.execute(
                select(decision_ledger.c.transaction_id)
                .order_by(decision_ledger.c.transaction_id)
                .limit(20)
            ).scalars()
        ]
    assert len(decided) == 20, "the replay must have landed before labels can be revealed"
    pending = [
        revealer.schedule(transaction_id, transaction_at, is_fraud=bool(index % 2))
        for index, transaction_id in enumerate(decided)
    ]

    # One day on, nothing is knowable: the shortest simulated lag is a chargeback that has
    # not been raised and a clean window that has not closed.
    still_pending = revealer.reveal_due(pending, transaction_at + dt.timedelta(days=1))
    assert len(still_pending) == len(pending)
    assert revealer.matured(transaction_at + dt.timedelta(days=1)) == []

    # 200 days on, past both the lognormal fraud tail and the 90-day clean window.
    remaining = revealer.reveal_due(pending, transaction_at + dt.timedelta(days=200))
    assert remaining == []
    matured = revealer.matured(transaction_at + dt.timedelta(days=200))
    assert {label.transaction_id for label in matured} == {p.transaction_id for p in pending}
    # Fraud disputes arrive on a lognormal around 34 days; a clean transaction is only known
    # good when the 90-day window closes. If these were equal the revealer would be a
    # constant delay wearing a distribution's name.
    fraud_lags = [label.dispute_lag_days for label in matured if label.is_fraud]
    clean_lags = {label.dispute_lag_days for label in matured if not label.is_fraud}
    assert clean_lags == {90.0}
    assert len(set(fraud_lags)) > 1


def test_prometheus_has_actually_scraped_the_api(stack: None) -> None:
    """A scrape job pointing at the wrong host produces empty panels, not an error. This is
    the only assertion in the repository that the target name in prometheus.yml matches the
    service name in the compose file."""
    del stack

    def scraped() -> bool:
        try:
            response = httpx.get(
                f"{PROMETHEUS}/api/v1/query",
                params={"query": 'up{job="fraudlens-api"}'},
                timeout=5.0,
            )
        except httpx.HTTPError:
            return False
        result = response.json()["data"]["result"]
        return bool(result) and result[0]["value"][1] == "1"

    assert wait_until(scraped, timeout=90.0), "prometheus never scraped fraudlens-api"


def test_the_decision_counter_moves_and_the_dashboards_can_read_it(stack: None) -> None:
    """§7's "metrics move on the dashboards". Queried through Prometheus rather than off
    /metrics, because the question is whether the *displayed* number exists — a counter that
    the exporter serves and the scraper never ingests is invisible to every panel."""
    del stack

    def has_decisions() -> bool:
        try:
            response = httpx.get(
                f"{PROMETHEUS}/api/v1/query",
                params={"query": "sum(fraudlens_decisions_total)"},
                timeout=5.0,
            )
        except httpx.HTTPError:
            return False
        result = response.json()["data"]["result"]
        return bool(result) and float(result[0]["value"][1]) > 0

    assert wait_until(has_decisions, timeout=90.0), "no decisions visible in prometheus"


def test_grafana_provisioned_the_four_dashboards_and_the_datasource(stack: None) -> None:
    """Provisioning fails silently: Grafana logs a warning and serves an empty folder. The
    dashboards are asserted as *files* in test_dashboards.py; this asserts they were loaded."""
    del stack
    assert wait_until(lambda: _grafana_ready(), timeout=120.0), "grafana never provisioned"
    search = httpx.get(f"{GRAFANA}/api/search", params={"type": "dash-db"}, timeout=10.0).json()
    uids = {item["uid"] for item in search}
    assert {
        "fraudlens-executive",
        "fraudlens-model-health",
        "fraudlens-serving-slo",
        "fraudlens-data-quality",
    } <= uids, sorted(uids)


def _grafana_ready() -> bool:
    try:
        response = httpx.get(f"{GRAFANA}/api/datasources/uid/fraudlens-prometheus", timeout=5.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def test_the_mlflow_backend_is_shared_rather_than_local(stack: None) -> None:
    """docs/lineage.md gap 8: the model card's run id points into a store nobody else has,
    so two people regenerating the same card get different run ids for the same numbers. A
    reachable server is what makes that pointer resolvable by someone other than its author.
    It does not close the gap on its own — the card still has to be regenerated against this
    URI — but without it the gap cannot be closed at all."""
    del stack
    assert wait_until(lambda: _mlflow_ready(), timeout=120.0), "mlflow server never answered"
    experiments = httpx.post(
        f"{MLFLOW}/api/2.0/mlflow/experiments/search", json={"max_results": 10}, timeout=10.0
    )
    assert experiments.status_code == 200, experiments.text


def _mlflow_ready() -> bool:
    try:
        return httpx.get(f"{MLFLOW}/health", timeout=5.0).status_code == 200
    except httpx.HTTPError:
        return False
