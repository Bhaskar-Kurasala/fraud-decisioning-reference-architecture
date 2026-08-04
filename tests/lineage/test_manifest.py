"""The manifest has to be *right about being wrong*, which is the harder half.

Recording a git SHA is trivial. Refusing to let a dirty tree pass as reproducible is the
part that has value, because that is the state an artifact is most often produced in and
the state that most looks fine at a glance.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from fraudlens.config import BusinessConstants
from fraudlens.lineage.manifest import LOCKFILE, Manifest, build_manifest

NOW = dt.datetime(2026, 8, 5, tzinfo=dt.timezone.utc)
COMMAND = "bash run_all.sh"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        # Identity is passed per-invocation rather than written to the repo config so the
        # test cannot depend on, or disturb, the developer's global git identity.
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway git repo with one committed input and a lockfile."""
    (tmp_path / "input.parquet").write_bytes(b"rows")
    (tmp_path / LOCKFILE).write_text("locked")
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "initial")
    return tmp_path


def _build(repo: Path, **overrides: Any) -> Manifest:
    kwargs: dict[str, Any] = {
        "repo_root": repo,
        "artifact": "policy ladder table",
        "config": BusinessConstants().model_dump(),
        "data_paths": [repo / "input.parquet"],
        "seeds": {"model_fit": 42},
        "regeneration_command": COMMAND,
        "now": NOW,
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


def test_a_clean_tree_with_every_pin_present_is_reproducible(repo: Path) -> None:
    manifest = _build(repo)

    assert manifest.reproducible
    assert manifest.caveats == ()
    assert manifest.git_dirty is False
    assert manifest.lock_digest is not None


def test_a_dirty_tree_is_marked_dirty_and_not_reproducible(repo: Path) -> None:
    """An uncommitted change means the SHA does not identify the code that ran.

    Recording the last commit anyway — which is what a naive `rev-parse HEAD` does — is
    worse than recording nothing: it produces a manifest that points confidently at code
    which was not executed.
    """
    (repo / "input.parquet").write_bytes(b"rows and one more")

    manifest = _build(repo)

    assert manifest.git_dirty is True
    assert not manifest.reproducible
    assert "modified working tree" in manifest.caveats[0]


def test_an_untracked_file_also_counts_as_dirty(repo: Path) -> None:
    """`--porcelain` includes untracked files, and that is the behaviour we want.

    The usual way a run becomes irreproducible is a new scratch module that was imported
    and never committed, which a tracked-changes-only check reports as clean.
    """
    (repo / "scratch.py").write_text("PATCH = True\n")
    assert _build(repo).git_dirty is True


def test_absent_input_data_is_recorded_rather_than_checksummed_as_nothing(repo: Path) -> None:
    """`data/` is git-ignored, so a fresh clone genuinely has no inputs to hash."""
    manifest = _build(repo, data_paths=[repo / "input.parquet", repo / "gone.parquet"])

    assert manifest.data_checksum is None
    assert manifest.missing_inputs == (str(repo / "gone.parquet"),)
    assert not manifest.reproducible


def test_declaring_no_inputs_is_refused(repo: Path) -> None:
    """A manifest pinning nothing would report itself reproducible, which is a lie."""
    with pytest.raises(ValueError, match="no declared inputs"):
        _build(repo, data_paths=[])


def test_a_naive_timestamp_is_refused(repo: Path) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _build(repo, now=dt.datetime(2026, 8, 5))  # noqa: DTZ001


def test_a_changed_business_constant_changes_the_config_hash(repo: Path) -> None:
    """The config is an input. Two artifacts computed under different constants must not
    claim the same provenance, and the hash is what makes that a mechanical check."""
    audited = _build(repo)
    sensitivity = _build(repo, config=BusinessConstants(f_pass=0.20).model_dump())

    assert audited.config_hash != sensitivity.config_hash


def test_the_json_form_carries_the_caveats_a_reader_needs(repo: Path) -> None:
    """The manifest travels beside the artifact as JSON; the reasons must travel with it.

    Serialising only the fields and leaving `reproducible`/`caveats` as Python-side
    properties would put the honest part behind an import.
    """
    (repo / "scratch.py").write_text("PATCH = True\n")
    payload = json.loads(_build(repo).to_json())

    assert payload["reproducible"] is False
    assert payload["caveats"]
    assert payload["regeneration_command"] == COMMAND
    assert payload["seeds"] == {"model_fit": 42}
