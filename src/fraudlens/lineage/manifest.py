"""Everything needed to regenerate one published number, recorded when it is published.

§9a lists the pins: git SHA, config hash, input data checksum, seeds, a committed
lockfile, and a single documented command that regenerates the number. This record
carries all six in one object so that "is this figure reproducible?" is a property that
can be read off an artifact rather than a claim someone makes about it.

The design choice worth stating is that this record can be *negative*. A manifest is
still written when the tree is dirty, when the input data is absent, or when the
lockfile is missing — it just says so, and `reproducible` is then False. The alternative
(refuse to build a manifest unless everything is pinned) sounds stricter and is worse:
it produces artifacts with no manifest at all, and an artifact with no provenance record
is indistinguishable from one nobody checked. An honest negative is auditable; a silence
is not.

Hashing is `models.provenance`'s, not a second scheme. ADR-0001 exists because two
definitions of the same quantity agree on the day they are written and drift silently
afterwards; two definitions of "the config hash" would break the ledger's own
`config_hash` column as a join key the first time one of them changed.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraudlens.models.provenance import config_hash, data_checksum, git_state

LOCKFILE = "uv.lock"


@dataclass(frozen=True, slots=True)
class Manifest:
    """The provenance of one artifact, including the reasons it may not be reproducible."""

    artifact: str
    generated_at: dt.datetime
    git_sha: str
    git_dirty: bool
    config_hash: str
    # None when the declared inputs were not on disk at generation time. Distinct from
    # the digest of an empty input set, which is a real hex string and would read as a
    # successful checksum of nothing.
    data_checksum: str | None
    missing_inputs: tuple[str, ...]
    seeds: Mapping[str, int]
    lock_digest: str | None
    regeneration_command: str
    # Not covered by uv.lock, and it moves numbers: the interpreter version determines
    # float repr, dict ordering guarantees and the C extension ABI the pinned wheels are
    # built against.
    python_version: str

    @property
    def caveats(self) -> tuple[str, ...]:
        """Every reason this artifact cannot be regenerated exactly, in reader's language.

        Rendered verbatim into the model card. A caveat that only exists as a boolean
        field gets summarised away by whoever writes the card; a sentence does not.
        """
        reasons: list[str] = []
        if self.git_dirty:
            reasons.append(
                f"Produced from a modified working tree. Commit {self.git_sha} does NOT "
                "identify the code that ran, so this artifact cannot be regenerated from "
                "the repository alone."
            )
        if self.data_checksum is None:
            missing = ", ".join(self.missing_inputs) or "unknown"
            reasons.append(
                f"Input data was not present when this record was written ({missing}), so "
                "no checksum pins the data this artifact was computed from."
            )
        if self.lock_digest is None:
            reasons.append(
                f"No {LOCKFILE} beside the repository root; dependency versions are not "
                "pinned by this record."
            )
        return tuple(reasons)

    @property
    def reproducible(self) -> bool:
        """True only when every pin is present. `caveats` says why when it is not."""
        return not self.caveats

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "generated_at": self.generated_at.isoformat(),
            "git_sha": self.git_sha,
            "git_dirty": self.git_dirty,
            "config_hash": self.config_hash,
            "data_checksum": self.data_checksum,
            "missing_inputs": list(self.missing_inputs),
            "seeds": dict(self.seeds),
            "lock_digest": self.lock_digest,
            "regeneration_command": self.regeneration_command,
            "python_version": self.python_version,
            "reproducible": self.reproducible,
            "caveats": list(self.caveats),
        }

    def to_json(self) -> str:
        # sort_keys so two manifests of the same artifact diff to nothing when nothing
        # changed. A manifest whose byte order moves on every write is noise in review.
        return json.dumps(self.to_dict(), sort_keys=True, indent=2) + "\n"


def build_manifest(
    *,
    repo_root: Path,
    artifact: str,
    config: Mapping[str, Any],
    data_paths: Sequence[Path],
    seeds: Mapping[str, int],
    regeneration_command: str,
    now: dt.datetime,
) -> Manifest:
    """Collect the pins for `artifact` as they stand right now.

    `now` is injected rather than read from the clock so a backfilled manifest records
    the time it represents, matching `models.tracking.log_training_run`.

    Declaring no inputs is an error, not an empty set: a manifest that pins no data
    cannot support the claim it exists to make, and would silently report itself
    reproducible.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not data_paths:
        raise ValueError("a manifest with no declared inputs cannot pin anything")

    missing = tuple(str(p) for p in data_paths if not p.exists())
    sha, dirty = git_state(repo_root)
    lock = repo_root / LOCKFILE
    return Manifest(
        artifact=artifact,
        generated_at=now,
        git_sha=sha,
        git_dirty=dirty,
        config_hash=config_hash(config),
        data_checksum=None if missing else data_checksum(tuple(data_paths)),
        missing_inputs=missing,
        seeds=dict(seeds),
        # The lockfile is checksummed rather than parsed. We need to answer "were the
        # dependencies the same", not "which versions were they" — the file itself is
        # committed, so the digest is a pointer into git history that a diff can resolve.
        lock_digest=data_checksum((lock,)) if lock.exists() else None,
        regeneration_command=regeneration_command,
        python_version=platform.python_version(),
    )
