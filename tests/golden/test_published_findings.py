"""Golden values: the extracted library must reproduce the published findings exactly.

These lock `fraud-decisioning-findings.md` as verified in `docs/findings/`. A failure
here means a number a reader was given has moved, which is a correctness incident, not
a test to update. Nothing in `src/fraudlens/` may be tuned to make one pass.

Tolerances are stated per assertion. The principle: tolerance = the precision the
figure was published at, never wider. Float64 summation over 92,427 terms carries an
absolute error around 1e-6 dollars, so a $1 tolerance on an annual total is ~6 orders
of magnitude above numerical noise -- it accommodates the published rounding and
nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.config import SETTINGS
from fraudlens.economics import (
    annualise,
    false_negative_cost,
    false_positive_cost,
    realised_cost,
    tenure_bucket,
)
from fraudlens.policy import boundaries, decide, rank_for_review

pytestmark = pytest.mark.golden

# The test window is days 150-181, with at least one transaction on each of 32 days.
# Every published annual figure is a window total x 365/32 (`data/consts.json`).
TEST_WINDOW_DAYS = 32

# Published to the nearest dollar / third decimal. See module docstring.
DOLLARS = 1.0
PROBABILITY = 5e-4


@pytest.fixture(scope="module")
def costed(scored_test: pd.DataFrame) -> pd.DataFrame:
    """L and M rebuilt from raw inputs through the library, not read from the artifact."""
    d = scored_test.copy()
    d["tenure"] = tenure_bucket(d.D1.to_numpy())
    d["L"] = false_negative_cost(d.TransactionAmt.to_numpy())
    d["M"] = false_positive_cost(d.TransactionAmt.to_numpy(), d.tenure.to_numpy())
    return d


def test_cost_columns_match_the_research_artifact(
    costed: pd.DataFrame, econ_test: pd.DataFrame
) -> None:
    """Tightest assertion in the suite: the extraction is bit-equivalent to 05_economics.py.

    Tolerance 1e-9 rather than exact equality only because the tenure -> relationship
    cost lookup goes through a dict here and a CSV round-trip there.
    """
    assert costed.TransactionID.tolist() == econ_test.TransactionID.tolist()
    np.testing.assert_allclose(costed.L, econ_test.L, atol=1e-9)
    np.testing.assert_allclose(costed.M, econ_test.M, atol=1e-9)


def test_median_per_transaction_costs(costed: pd.DataFrame) -> None:
    """Findings §3: the asymmetry is smaller than fraud teams assume -- L is below M."""
    ratio = float(np.median(costed.L / costed.M))
    assert ratio == pytest.approx(0.8049, abs=5e-5)  # published to 4dp
    assert float(np.median(costed.L)) == pytest.approx(84.95, abs=5e-3)  # published to cents
    assert float(np.median(costed.M)) == pytest.approx(144.00, abs=5e-3)


def test_break_even_by_tenure(costed: pd.DataFrame) -> None:
    """Findings §6. A one-week-old account is the hardest to decline: it has the most
    residual LTV left to lose per dollar of basket."""
    b = boundaries(costed.L.to_numpy(), costed.M.to_numpy())["allow_to_deny"]
    median = pd.Series(b).groupby(costed.tenure.to_numpy()).median()
    for label, published in [
        ("new(0d)", 0.642),
        ("1-7d", 0.740),
        ("31-90d", 0.534),
        ("400d+", 0.379),
    ]:
        assert median[label] == pytest.approx(published, abs=PROBABILITY), label


def test_break_even_by_amount_band(costed: pd.DataFrame) -> None:
    """Findings §7. The boundary falls as the basket grows, which inverts the usual
    'scrutinise big-ticket orders hardest' instinct."""
    band = pd.cut(
        costed.TransactionAmt,
        [0, 25, 50, 100, 250, 500, 1e6],
        labels=["<25", "25-50", "50-100", "100-250", "250-500", "500+"],
    )
    b = boundaries(costed.L.to_numpy(), costed.M.to_numpy())["allow_to_deny"]
    median = pd.Series(b).groupby(band.to_numpy(), observed=True).median()
    for label, published in [
        ("<25", 0.731),
        ("50-100", 0.608),
        ("250-500", 0.437),
        ("500+", 0.369),
    ]:
        assert median[label] == pytest.approx(published, abs=PROBABILITY), label


def _annual_cost(costed: pd.DataFrame, score: str, *, include_review: bool) -> float:
    fn, fp = costed.L.to_numpy(), costed.M.to_numpy()
    actions = decide(costed[score].to_numpy(), fn, fp, include_review=include_review)
    total = float(realised_cost(actions, costed.isFraud.to_numpy(), fn, fp).sum())
    return annualise(total, TEST_WINDOW_DAYS)


def test_champion_three_action_annual_cost(costed: pd.DataFrame) -> None:
    """$2,799,214 — the allow/challenge/deny policy, the one that would actually ship.

    This is the figure in docs/findings/fit-balanced-empirical-result.md, where it is
    labelled only by its Allow/Challenge/Deny columns.
    """
    assert _annual_cost(costed, "p", include_review=False) == pytest.approx(2_799_214, abs=DOLLARS)


def test_champion_four_action_annual_cost(costed: pd.DataFrame) -> None:
    """$2,799,797 — findings §1 P4, the same policy with analyst review available.

    Uncapped review is worth $583/yr *less* than not having it: at $7.97 a case it
    clears its own cost on 5 of 92,427 transactions.
    """
    assert _annual_cost(costed, "p", include_review=True) == pytest.approx(2_799_797, abs=DOLLARS)


def test_uncalibrated_score_costs_more_through_the_same_policy(costed: pd.DataFrame) -> None:
    """Findings §2: $2,871,342 raw vs $2,799,797 isotonic, at essentially equal AUC.

    Same policy, same costs, same transactions — the entire $71,545 is calibration.
    """
    assert _annual_cost(costed, "p_raw", include_review=True) == pytest.approx(
        2_871_342, abs=DOLLARS
    )


def test_isotonic_score_ties_at_the_analyst_queue_cut(costed: pd.DataFrame) -> None:
    """Locks the measurement that forces the explicit tie-break in `policy.queue`.

    Verification report §3.1: 153 distinct isotonic values, and 120 transactions tied
    at the rank-1,920 score. If a future calibrator removes the ties this test fails
    and the tie-break can be reconsidered — until then it is load-bearing.
    """
    budget = SETTINGS.daily_review_slots * TEST_WINDOW_DAYS
    assert budget == 1920
    p = costed.p.to_numpy()
    assert len(np.unique(p)) == 153

    selected = rank_for_review(p, costed.TransactionID.to_numpy(), budget)
    cut_value = p[selected[-1]]
    assert cut_value == pytest.approx(0.41463, abs=5e-6)
    assert int((p == cut_value).sum()) == 120
