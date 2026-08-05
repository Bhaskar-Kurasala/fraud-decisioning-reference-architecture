"""Re-run value-of-review with the label-acquisition term included.

    uv run python scripts/analyse_review_label_value.py

Regenerates every number in `docs/findings/review-as-label-acquisition.md` from
`data/econ_test.parquet` and `data/p_te_bal_fitted.npy` (both products of `run_all.sh`
plus `FIT_BALANCED=1 python3 research/03_model.py`). Read-only: prints, writes nothing.

The §5 queue is reproduced here rather than imported from `research/06_full.py`, which is
provenance under ADR-0001 and must not be modified — with one deliberate deviation, a
deterministic tie-break on TransactionID. §5's own footnote records that `argsort` on the
153-valued isotonic score made two of its four rows irreproducible run-to-run; a number
this note asks anyone to act on has to survive a re-run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudlens.config import SETTINGS
from fraudlens.economics import (
    ALLOW,
    CHALLENGE,
    DENY,
    REVIEW,
    action_expected_values,
    annualise,
    realised_cost,
)
from fraudlens.economics.label_value import (
    break_even_events_per_year,
    days_to_detect_rate_shift,
    days_to_maturity_floor,
    maturity_fraction,
)
from fraudlens.monitoring.baseline import numeric_reference
from fraudlens.monitoring.drift import PSI_ALERT_THRESHOLD, numeric_drift

REPO_ROOT = Path(__file__).resolve().parents[1]
ECON = REPO_ROOT / "data" / "econ_test.parquet"
REBALANCED = REPO_ROOT / "data" / "p_te_bal_fitted.npy"

# The measured cost of operating a model that passed an AUC gate and should not have.
# docs/findings/fit-balanced-empirical-result.md. Used as the severity anchor throughout.
MEASURED_EXCESS_COST = 4_358_096.0

# Queue sizes to price. 60/day is the published §5 configuration (~1 analyst FTE).
SLOT_GRID = (5, 10, 20, 30, 60, 120)
# Fraud-rate drops in the reviewed cohort to size detection against. 0.13 is roughly a
# fifth of the champion queue's precision; 0.25 is the scale of a genuine break.
SHIFT_GRID = (0.05, 0.13, 0.25)


def main() -> None:
    d = pd.read_parquet(ECON)
    amount_days = int(d.day.nunique())
    fn_cost = d.L.to_numpy(dtype=np.float64)
    fp_cost = d.M.to_numpy(dtype=np.float64)
    is_fraud = d.isFraud.to_numpy(dtype=np.float64)
    champion = d.p.to_numpy(dtype=np.float64)
    txn_id = d.TransactionID.to_numpy()
    n = len(d)
    fraud_rate = float(is_fraud.mean())

    def policy_cost(actions: np.ndarray) -> float:
        window = float(realised_cost(actions, is_fraud, fn_cost, fp_cost).sum())
        return annualise(window, amount_days)

    ev = action_expected_values(champion, fn_cost, fp_cost)
    no_review = policy_cost(np.asarray(ev.argmax(axis=0)))
    auto_actions = np.array([ALLOW, CHALLENGE, DENY])[ev[[ALLOW, CHALLENGE, DENY]].argmax(axis=0)]
    value_of_review = ev[REVIEW] - ev[[ALLOW, CHALLENGE, DENY]].max(axis=0)
    # Descending VoR, ties broken by TransactionID so the queue is reproducible.
    ranked = np.lexsort((txn_id, -value_of_review))

    print("=" * 78)
    print("1. WHAT THE QUEUE COSTS, AND WHAT IT BUYS")
    print("=" * 78)
    print(f"no-review four-action argmax          ${no_review:,.0f}/yr")
    print(f"value-of-review positive on           {int((value_of_review > 0).sum())} of {n:,}")

    rows = []
    for slots in SLOT_GRID:
        budget = slots * amount_days
        selected = np.zeros(n, dtype=bool)
        selected[ranked[:budget]] = True
        actions = np.where(selected, REVIEW, auto_actions)
        cost = policy_cost(actions)
        labels_per_year = budget * 365.0 / amount_days
        rows.append(
            {
                "slots/day": slots,
                "labels/yr": labels_per_year,
                "queue cost/yr": cost - no_review,
                "$/label": (cost - no_review) / labels_per_year,
                "queue fraud rate": float(is_fraud[selected].mean()),
                "min p in queue": float(champion[selected].min()),
            }
        )
    queue = pd.DataFrame(rows).set_index("slots/day")
    print()
    print(queue.to_string(float_format=lambda v: f"{v:,.4f}"))
    print("\nThe queue is a purchase of same-day labels. Every row is a price per label.")

    print()
    print("=" * 78)
    print("2. THE DEADLINE: WHEN CHARGEBACK LABELS BECOME USABLE")
    print("=" * 78)
    for age in (7.0, 30.0, 60.0, 89.0, 90.0, 120.0):
        print(f"  maturity at t+{age:5.0f}d : {maturity_fraction(age, fraud_rate):.4f}")
    wall = days_to_maturity_floor(fraud_rate)
    print(f"\n80% maturity floor reached at t+{wall:.0f}d. Nothing label-dependent is")
    print("computable before then; monitoring.maturity refuses it.")

    print()
    print("=" * 78)
    print("3. DOES THE QUEUE ACTUALLY SEE DECAY? (measured, not assumed)")
    print("=" * 78)
    reference = numeric_reference("score", champion)
    for name, score in (
        ("champion (isotonic)", champion),
        ("raw uncalibrated", d.p_raw.to_numpy(dtype=np.float64)),
        ("rebalanced, refit", np.load(REBALANCED)),
    ):
        e2 = action_expected_values(score, fn_cost, fp_cost)
        v2 = e2[REVIEW] - e2[[ALLOW, CHALLENGE, DENY]].max(axis=0)
        sel2 = np.zeros(n, dtype=bool)
        sel2[np.lexsort((txn_id, -v2))[: SETTINGS.daily_review_slots * amount_days]] = True
        psi = numeric_drift(reference, score).psi
        print(
            f"  {name:22s} queue fraud rate {is_fraud[sel2].mean():.4f}   "
            f"PSI vs champion {psi:7.4f} "
            f"({'ALERTS' if psi > PSI_ALERT_THRESHOLD else 'silent'})"
        )
    print("\nThe queue sees the one decay event we have measured (0.58 -> 0.20). So does")
    print("PSI on the score, on day zero, for free. The label's marginal value over the")
    print("free signal is zero for this failure mode.")

    print()
    print("=" * 78)
    print("4. THE MODE THE QUEUE IS BLIND TO: DRIFT BELOW THE QUEUE'S FLOOR")
    print("=" * 78)
    p90 = float(np.quantile(champion, 0.90))
    top_queue = np.zeros(n, dtype=bool)
    top_queue[ranked[: SETTINGS.daily_review_slots * amount_days]] = True
    low_score_good = (champion < p90) & (is_fraud == 0)
    print(f"  good transactions below p90 of score : {int(low_score_good.sum()):,}")
    print(f"  of those, sampled by the 60/day queue: {int((low_score_good & top_queue).sum())}")
    for uplift in (0.005, 0.01, 0.02):
        added = int(low_score_good.sum() * uplift)
        shifted = (is_fraud.sum() + added) / n
        audit = days_to_detect_rate_shift(fraud_rate, shifted, 60.0, analyst_agreement=1.0)
        print(
            f"  +{uplift:.1%} fraud in that region (+{added:,} cases): population rate "
            f"{fraud_rate:.4f} -> {shifted:.4f}; queue sees nothing; "
            f"random audit @60/day detects in {audit:,.0f}d"
        )
    print("\nAn adversary who finds a blind spot moves nothing the queue samples, and no")
    print(f"affordable random audit beats the {wall:.0f}-day chargeback wall either.")

    print()
    print("=" * 78)
    print("5. BREAK-EVEN: EVENTS/YR NEEDED TO JUSTIFY THE QUEUE AS AN INSTRUMENT")
    print(f"   (severity fixed at the measured ${MEASURED_EXCESS_COST:,.0f}/yr)")
    print("=" * 78)
    grid = []
    for slots in SLOT_GRID:
        r0 = float(queue.loc[slots, "queue fraud rate"])
        cost = float(queue.loc[slots, "queue cost/yr"])
        row = {"slots/day": slots, "$/yr": cost}
        for shift in SHIFT_GRID:
            detect = days_to_detect_rate_shift(r0, r0 - shift, float(slots))
            saved = wall - detect
            row[f"d@-{shift:.2f}"] = detect
            row[f"BE@-{shift:.2f}"] = break_even_events_per_year(cost, saved, MEASURED_EXCESS_COST)
        grid.append(row)
    print(pd.DataFrame(grid).set_index("slots/day").to_string(float_format=lambda v: f"{v:,.3f}"))
    print("\nd@-x  = days for the reviewed cohort's fraud rate to reveal a drop of x")
    print("BE@-x = decay events per year at which the queue pays for itself. inf means")
    print("        detection is slower than the chargeback wall, so it buys nothing.")


if __name__ == "__main__":
    main()
