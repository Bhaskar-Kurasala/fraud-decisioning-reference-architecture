from __future__ import annotations

from pathlib import Path

from fraudlens.models.provenance import collect_provenance, config_hash, data_checksum, git_state


def test_config_hash_is_order_independent_but_value_sensitive() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_data_checksum_binds_content_to_filename(tmp_path: Path) -> None:
    """A train/test swap must not produce the same checksum."""
    first, second = tmp_path / "train.bin", tmp_path / "test.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"beta")
    original = data_checksum([first, second])

    first.write_bytes(b"beta")
    second.write_bytes(b"alpha")
    assert data_checksum([first, second]) != original


def test_data_checksum_ignores_argument_order(tmp_path: Path) -> None:
    first, second = tmp_path / "a.bin", tmp_path / "b.bin"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    assert data_checksum([first, second]) == data_checksum([second, first])


def test_git_state_degrades_honestly_outside_a_repo(tmp_path: Path) -> None:
    """Missing provenance is reported as unknown-and-dirty, never as clean."""
    sha, dirty = git_state(tmp_path)
    if sha == "unknown":
        assert dirty is True
    else:
        # tmp_path can sit inside an enclosing repo on some machines; the
        # invariant we care about is that a SHA is either real or flagged.
        assert len(sha) == 40


def test_collect_provenance_produces_four_tags(tmp_path: Path) -> None:
    payload = tmp_path / "d.bin"
    payload.write_bytes(b"x")
    provenance = collect_provenance(tmp_path, {"SEED": 42}, [payload])
    tags = provenance.as_tags()
    assert set(tags) == {
        "fraudlens.git_sha",
        "fraudlens.git_dirty",
        "fraudlens.config_hash",
        "fraudlens.data_checksum",
    }
    assert all(isinstance(v, str) for v in tags.values())
