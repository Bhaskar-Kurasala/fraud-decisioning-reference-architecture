"""The extracted config must equal the research config, constant for constant.

ADR-0001 keeps `research/` as the provenance of every published number and
extracts its logic into `fraudlens`. That only holds if the two agree on the
inputs. A drift here would not raise anything — it would silently change what
the service charges a customer while the findings continue to quote the old
figure, and the first symptom would be a P&L discrepancy nobody can source.

This is the test that makes ADR-0001 an enforced constraint rather than an
intention.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from fraudlens.config.settings import SETTINGS

# Root config.py is the research entry point, not an installed module. Load it
# by path so this test does not depend on how pytest was invoked.
_ROOT = Path(__file__).resolve().parents[2]


def _load_research_config() -> object:
    spec = importlib.util.spec_from_file_location("_research_config", _ROOT / "config.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_research_config"] = module
    spec.loader.exec_module(module)
    return module


# (research name, extracted name). Every business constant the extracted library
# redefines must appear here — adding a constant to settings.py without adding it
# to this list is the drift this test exists to catch.
PARITY = [
    ("COGS", "cogs"),
    ("MARGIN", "margin"),
    ("CB_FEE", "cb_fee"),
    ("OPS_DISPUTE", "ops_dispute"),
    ("F_PASS", "f_pass"),
    ("A_ABANDON", "a_abandon"),
    ("Q_ANALYST", "q_analyst"),
    ("C_REVIEW", "c_review"),
    ("D_DELAY", "d_delay"),
    ("CHARGEBACK_MEDIAN_DAYS", "chargeback_median_days"),
    ("CHARGEBACK_LOG_SIGMA", "chargeback_log_sigma"),
    ("SEED_LABEL_SIM", "label_sim_seed"),
]


@pytest.mark.golden
@pytest.mark.parametrize(("research_name", "extracted_name"), PARITY)
def test_extracted_constant_matches_research(research_name: str, extracted_name: str) -> None:
    research = _load_research_config()
    expected = getattr(research, research_name)
    actual = getattr(SETTINGS, extracted_name)
    assert actual == expected, (
        f"{research_name}={expected} in research/config.py but "
        f"{extracted_name}={actual} in fraudlens.config. The published findings and "
        f"the production service would disagree about the cost of a decision."
    )


@pytest.mark.golden
def test_tenure_buckets_match_research() -> None:
    """Boundaries must align, or a transaction is priced in the wrong band."""
    research = _load_research_config()
    from fraudlens.config.settings import TENURE_EDGES, TENURE_LABELS

    assert list(TENURE_EDGES) == list(research.TENURE_EDGES)
    assert list(TENURE_LABELS) == list(research.TENURE_LABELS)
