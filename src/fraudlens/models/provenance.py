"""The code, config and data that produced an artifact.

§9a makes this non-negotiable. A scored decision is an adverse action against an
identifiable customer and may be disputed months later; if the model artifact
cannot be traced back to the exact commit, configuration and input data that
produced it, the dispute cannot be answered and the model cannot be revalidated.
An untraceable artifact is worthless for audit regardless of how good its
metrics are.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_READ_CHUNK = 1 << 20
_UNKNOWN_SHA = "unknown"


@dataclass(frozen=True, slots=True)
class Provenance:
    """The three coordinates that identify a training run's inputs."""

    git_sha: str
    git_dirty: bool
    config_hash: str
    data_checksum: str

    def as_tags(self) -> dict[str, str]:
        """Flatten to MLflow tags.

        `git_dirty` is recorded separately rather than folded into the SHA
        because a dirty tree means the SHA does *not* identify the code that
        ran. A reviewer needs to see that distinction, not have it hidden.
        """
        return {
            "fraudlens.git_sha": self.git_sha,
            "fraudlens.git_dirty": str(self.git_dirty).lower(),
            "fraudlens.config_hash": self.config_hash,
            "fraudlens.data_checksum": self.data_checksum,
        }


def _git(repo_root: Path, *args: str) -> str | None:
    """Run a read-only git command, returning None when git cannot answer.

    Returning None rather than raising is deliberate: provenance collection must
    not be able to break a training run in an environment without git (a
    container, a notebook). The absence is then recorded explicitly as
    "unknown", which is honest and visible, rather than silently omitted.
    """
    try:
        # S603/S607: fixed argv, no shell, arguments are literals supplied by
        # this module. The only variable is the repo path, passed as `cwd`.
        completed = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_state(repo_root: Path) -> tuple[str, bool]:
    """Return (commit SHA, working-tree-is-dirty)."""
    sha = _git(repo_root, "rev-parse", "HEAD")
    if sha is None:
        return _UNKNOWN_SHA, True
    status = _git(repo_root, "status", "--porcelain")
    return sha, bool(status)


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable hash of the business + model configuration.

    Business constants (COGS, P_CHURN_ON_DECLINE, F_PASS, ...) are model inputs
    in exactly the same sense as the features are: change one and every cost in
    the ledger changes. Hashing them puts a config change on the same footing as
    a code change when explaining why two runs disagree.
    """
    canonical = json.dumps(config, sort_keys=True, default=repr, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def data_checksum(paths: Sequence[Path]) -> str:
    """Checksum of the input dataset, order-independent but name-sensitive.

    Names are folded into the digest so that swapping two files' contents cannot
    produce the same checksum — a silent train/test swap is exactly the kind of
    error this is here to catch.
    """
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(_READ_CHUNK):
                digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(
    repo_root: Path,
    config: Mapping[str, Any],
    data_paths: Sequence[Path],
) -> Provenance:
    """Assemble the full provenance stamp for a training run."""
    sha, dirty = git_state(repo_root)
    return Provenance(
        git_sha=sha,
        git_dirty=dirty,
        config_hash=config_hash(config),
        data_checksum=data_checksum(data_paths),
    )
