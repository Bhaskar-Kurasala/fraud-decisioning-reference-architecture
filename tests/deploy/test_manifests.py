"""The deployment descriptors, checked for the mistakes that do not announce themselves.

A YAML typo fails loudly at apply time and costs a minute. The failures worth a test are
the ones that apply cleanly and are wrong:

* liveness aliased to readiness — the pod gets restarted for a model outage, which does
  not produce a model and turns a degradation into a checkout outage;
* a PodDisruptionBudget whose selector matches nothing — it reports Healthy and protects
  no pod, and you find out during the node drain;
* a credential in the ConfigMap — readable by anyone with namespace read, and invisible
  because it works;
* a container with no resource limits — schedules fine, then evicts something else.

None of these produce an error message. All four are one line each to assert.

`kubeconform` and Docker are optional: this file must pass on a machine with neither,
because CI has no cluster and the design spec (§1.2) records that Kubernetes is not
installed on the development machine either.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
K8S = REPO / "deploy" / "k8s"
COMPOSE = REPO / "deploy" / "compose" / "docker-compose.yml"
DOCKERFILES = sorted((REPO / "deploy" / "docker").glob("*.Dockerfile"))


def _documents() -> list[dict[str, Any]]:
    return [
        doc
        for path in sorted(K8S.glob("*.yaml"))
        for doc in yaml.safe_load_all(path.read_text(encoding="utf-8"))
        if doc
    ]


def _of_kind(kind: str) -> dict[str, Any]:
    matches = [doc for doc in _documents() if doc["kind"] == kind]
    assert len(matches) == 1, f"expected exactly one {kind}, found {len(matches)}"
    return matches[0]


def _api_container() -> dict[str, Any]:
    return _of_kind("Deployment")["spec"]["template"]["spec"]["containers"][0]


def _instructions(path: Path) -> str:
    """A Dockerfile with the commentary stripped.

    These files carry more prose than instructions, and the prose discusses the very
    strings the assertions look for — `/health/ready` and `--all-extras` are both named in
    comments explaining why they are *not* used. Matching against the raw text would make
    the tests fail on a correct file for explaining itself.
    """
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    )


# ======================================================================================
# Probes: the distinction the whole degradation design rests on
# ======================================================================================


def test_liveness_and_readiness_ask_different_questions() -> None:
    """If these two ever point at the same endpoint, a model outage becomes a restart loop.

    The service is explicit that a not-ready instance is still answering fail-safe and must
    not be restarted, because restarting it does not produce a model. That contract lives in
    three places — `ServiceState.ready`, the Dockerfile HEALTHCHECK, and here — and this is
    the only one of the three that a reviewer cannot check by reading a docstring.
    """
    container = _api_container()
    assert container["livenessProbe"]["httpGet"]["path"] == "/health/live"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health/ready"
    assert container["startupProbe"]["httpGet"]["path"] == "/health/live"


def test_startup_has_its_own_budget_larger_than_liveness() -> None:
    """Model load happens once, in the lifespan hook, before uvicorn accepts connections. A
    load slower than the liveness threshold gets the pod killed and retried forever, and from
    outside that is indistinguishable from a crash loop."""
    container = _api_container()
    startup = container["startupProbe"]
    liveness = container["livenessProbe"]
    startup_budget = startup["periodSeconds"] * startup["failureThreshold"]
    liveness_budget = liveness["periodSeconds"] * liveness["failureThreshold"]
    assert startup_budget > liveness_budget


def test_the_container_healthcheck_is_liveness_not_readiness() -> None:
    """Compose has no readiness concept, so HEALTHCHECK is the only probe it has — and
    `restart: unless-stopped` acts on it. Pointing it at /health/ready would restart a
    correctly-degraded container in a loop."""
    api = _instructions(REPO / "deploy" / "docker" / "api.Dockerfile")
    assert "/health/live" in api
    assert "/health/ready" not in api


# ======================================================================================
# Resources, disruption, and the config/secret boundary
# ======================================================================================


def test_the_container_declares_both_requests_and_limits() -> None:
    """A request with no limit is a noisy neighbour on the checkout path; a limit with no
    request schedules onto a node that cannot honour it."""
    resources = _api_container()["resources"]
    assert set(resources["requests"]) == {"cpu", "memory"}
    assert set(resources["limits"]) == {"cpu", "memory"}


def test_the_cpu_limit_is_at_least_one_core() -> None:
    """CFS throttling quantises at 100 ms. Below a full core a single in-flight decision can
    be stalled until the next period — up to 100 ms against a 150 ms p99 budget (§4.3), from
    a limit that looks generous on an average-utilisation graph."""
    limit = _api_container()["resources"]["limits"]["cpu"]
    millicores = int(limit[:-1]) if str(limit).endswith("m") else int(float(limit) * 1000)
    assert millicores >= 1000


def test_the_disruption_budget_actually_selects_the_pods() -> None:
    """A PDB whose selector matches nothing reports Healthy and protects nothing. There is no
    warning; the evidence arrives when a node drain takes the last scoring pod."""
    pod_labels = _of_kind("Deployment")["spec"]["template"]["metadata"]["labels"]
    selector = _of_kind("PodDisruptionBudget")["spec"]["selector"]["matchLabels"]
    assert selector.items() <= pod_labels.items()
    assert _of_kind("Service")["spec"]["selector"].items() <= pod_labels.items()


def test_the_credential_is_in_the_secret_and_nowhere_else() -> None:
    """The DSN carries the password. In a ConfigMap it is readable by anyone with namespace
    read and shows up in `kubectl describe` output pasted into incident channels."""
    config = _of_kind("ConfigMap")["data"]
    secret = _of_kind("Secret")["stringData"]
    assert "FRAUDLENS_DATABASE_URL" in secret
    assert not set(config) & set(secret), "a key defined in both loses to whichever is last"
    assert not any("://" in value and "@" in value for value in config.values())


def test_the_pod_cannot_escalate_or_write_its_own_root_filesystem() -> None:
    """The service unpickles a model artifact at startup and parses attacker-influenced JSON
    on the request path. Both are places where a container escape starts with "and it was
    running as uid 0"."""
    pod = _of_kind("Deployment")["spec"]["template"]["spec"]
    container = _api_container()
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]


def test_nothing_declared_here_is_a_topology_adr_0002_declined() -> None:
    """ADR-0002 rules out multi-region, horizontal autoscaling and a service mesh by name. An
    epic that quietly delivers what an ADR declined is worse than one that skips it, and the
    cheapest way for that to happen is a manifest copied from a template."""
    kinds = {doc["kind"] for doc in _documents()}
    assert not kinds & {
        "HorizontalPodAutoscaler",
        "VerticalPodAutoscaler",
        "VirtualService",
        "DestinationRule",
        "PeerAuthentication",
        "Gateway",
    }


# ======================================================================================
# Images
# ======================================================================================


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.name)
def test_every_base_image_is_pinned_by_digest(path: Path) -> None:
    """A tag is a moving target. §9a's "regenerable by a single documented command" is false
    for an image whose base changed underneath it."""
    for line in _instructions(path).splitlines():
        if line.startswith("FROM "):
            assert "@sha256:" in line, f"{path.name}: {line}"


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: p.name)
def test_every_image_drops_to_a_non_root_user(path: Path) -> None:
    body = _instructions(path)
    assert "USER " in body
    assert body.index("USER ") < body.index("CMD "), f"{path.name}: USER must precede CMD"


def test_the_scoring_image_does_not_install_the_tracking_extra() -> None:
    """pyproject splits the extras by deployment unit because "the scoring container has no
    reason to carry MLflow's dependency tree". An image built with --all-extras makes that
    split decorative, and the composition root's lazy `import mlflow` pointless."""
    body = _instructions(REPO / "deploy" / "docker" / "api.Dockerfile")
    assert "[serving,streaming]" in body
    assert "all-extras" not in body
    assert "tracking" not in body


# ======================================================================================
# Validation by the real tools, when the real tools are present
# ======================================================================================


def test_the_manifests_validate_against_the_kubernetes_schemas() -> None:
    """Structural assertions above cannot catch a misspelled field: Kubernetes accepts
    unknown keys in many places and silently ignores them, so `readinessProbe` under the wrong
    parent is a probe that never runs.

    Skips rather than fails when kubeconform is absent. The alternative — vendoring the
    schemas — is 40 MB of JSON to check a property CI can check when it has the tool.
    """
    kubeconform = shutil.which("kubeconform")
    if kubeconform is None:
        pytest.skip("kubeconform not installed; manifests validated structurally only")
    result = subprocess.run(  # noqa: S603
        [kubeconform, "-strict", "-summary", "-output", "json", *map(str, K8S.glob("*.yaml"))],
        capture_output=True,
        text=True,
        check=False,
    )
    summary = json.loads(result.stdout)["summary"]
    assert result.returncode == 0, result.stdout
    assert summary["invalid"] == 0 and summary["errors"] == 0
    # A file that validates because every resource was skipped is not validated.
    assert summary["valid"] == len(_documents())


def test_the_compose_file_resolves() -> None:
    """`include:` and profiles are both places where a file parses as YAML and still does not
    resolve — an included path that moved, a variable with no default. Only the Compose
    binary knows.

    Not marked `integration`: it needs the Compose *binary*, not the stack, and it skips
    cleanly without one. Marking it would exclude it from the CI job whose entire purpose is
    to catch a compose file that stopped resolving."""
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    result = subprocess.run(  # noqa: S603
        ["docker", "compose", "-f", str(COMPOSE), "--profile", "full", "config"],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=COMPOSE.parent,
    )
    assert result.returncode == 0, result.stderr
    resolved = yaml.safe_load(result.stdout)
    # postgres comes from the included file; if `include:` silently failed the stack would
    # come up with no database and the api would report itself ready while recording nothing.
    assert {"postgres", "migrate", "api", "mlflow", "prometheus", "grafana"} <= set(
        resolved["services"]
    )


def test_the_core_profile_is_the_smallest_thing_that_can_record_a_decision() -> None:
    """§10: profiles exist so the stack fits on a laptop. `core` must be api + ledger and
    must not drag in Grafana; `full` must contain everything, or a service lands in no
    profile and silently never starts."""
    services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
    for name, service in services.items():
        assert service.get("profiles"), f"{name} is in no profile and will never start"
        assert "full" in service["profiles"], f"{name} is missing from the full stack"
    core = {name for name, s in services.items() if "core" in s["profiles"]}
    assert core == {"migrate", "api"}
    assert "grafana" not in core
