"""The running stack, for the tests that need one.

Everything here is skipped when Docker is absent. CI has no Docker and the design spec
(§1.2) records that the development machine has no Kubernetes either, so "the suite is
green" must not depend on either being installed — a test that silently requires
infrastructure is a test that gets marked xfail the first time it runs somewhere else.

The stack is brought up once per session and torn down at the end. Per-test isolation
would be cleaner and is not affordable: a cold start is ~20 s and the drills deliberately
mutate the stack (stopping Postgres, removing the model artifact), so each drill restores
what it broke and asserts that the restoration worked. That assertion is not overhead —
"does it recover" is half of what a degradation drill is for.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO / "deploy" / "compose" / "docker-compose.yml"
BUNDLE_DIR = REPO / "deploy" / "compose" / "bundles"
BUNDLE = BUNDLE_DIR / "scorer.pkl"


# Read from the same variables the compose file reads, with the same defaults. Not a
# convenience: 3000, 5000 and 5432 are the three most contended ports on a developer
# machine, and a hard-coded port turns "this laptop already runs something on 3000" into a
# drill that fails with an assertion about Grafana provisioning. The failure would be read
# as a bug in the stack rather than in the harness.
def _port(variable: str, default: int) -> int:
    return int(os.environ.get(variable, default))


API = f"http://localhost:{_port('API_PORT', 8000)}"
PROMETHEUS = f"http://localhost:{_port('PROMETHEUS_PORT', 9090)}"
GRAFANA = f"http://localhost:{_port('GRAFANA_PORT', 3000)}"
MLFLOW = f"http://localhost:{_port('MLFLOW_PORT', 5000)}"

DATABASE_URL = (
    f"postgresql+psycopg://fraudlens:fraudlens@localhost:{_port('POSTGRES_PORT', 5432)}/fraudlens"
)


def compose(
    *args: str, check: bool = True, timeout: float = 300.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["docker", "compose", "-f", str(COMPOSE_FILE), "--profile", "full", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
        cwd=COMPOSE_FILE.parent,
    )


def wait_until(predicate: object, *, timeout: float = 60.0, interval: float = 0.5) -> bool:
    """Poll until true or the deadline. Returns the outcome rather than raising.

    Polling rather than a fixed sleep: the interesting quantity in a drill is often *how
    long* recovery took, and a sleep long enough to be reliable is long enough to hide a
    regression from ten seconds to fifty.
    """
    assert callable(predicate)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _reachable(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2.0).status_code < 500
    except httpx.HTTPError:
        return False


@pytest.fixture(scope="session")
def stack() -> Iterator[None]:
    """The `full` profile, up and scraped.

    `full` rather than `core` even though the drills only need the api and the ledger:
    without Prometheus and Grafana the §7 integration row's last clause — "metrics move on
    the dashboards" — is unverifiable, and it is the clause most likely to be quietly wrong,
    because a dashboard that queries a metric nobody exports looks identical to a quiet one.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    if not BUNDLE.is_file():
        pytest.skip(f"no scoring bundle at {BUNDLE} — build one with scripts/build_model_bundle.py")
    try:
        compose("up", "-d", "--wait", timeout=600.0)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"compose stack did not come up: {exc}")
    if not wait_until(lambda: _reachable(f"{API}/health/live"), timeout=90.0):
        compose("logs", "--tail", "50", check=False)
        pytest.fail("the api container came up but never answered /health/live")
    yield
    compose("down", "-v", check=False, timeout=180.0)


@pytest.fixture
def api(stack: None) -> Iterator[httpx.Client]:
    del stack
    with httpx.Client(base_url=API, timeout=15.0) as client:
        yield client
