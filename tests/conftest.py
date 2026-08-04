"""Shared fixtures.

The golden tests run against the persisted research artifacts in `data/`, which is
git-ignored (652 MB of raw CSV). Rather than fail on a fresh clone they skip with the
command that regenerates them -- a missing artifact is a setup state, not a regression,
and conflating the two teaches people to ignore red.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"


def _load(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{path} missing — regenerate with `bash run_all.sh`")
    return pd.read_parquet(path)


@pytest.fixture(scope="session")
def scored_test() -> pd.DataFrame:
    """Model output for the 92,427 test-window transactions: p, p_raw, p_bal + raw inputs."""
    return _load("scored_test.parquet")


@pytest.fixture(scope="session")
def econ_test() -> pd.DataFrame:
    """`scored_test` plus the L, M, boundary and action columns written by 05_economics.py."""
    return _load("econ_test.parquet")
