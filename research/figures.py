"""Regenerate the two figures embedded in the README.

Not a pipeline stage — it computes nothing new. Both panels read artifacts that
stages 3b and 5 already wrote, so this file cannot become a second source of
truth for a published number. Run it after `05_economics.py`.

    python3 research/figures.py   ->  docs/images/headline.png
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

FIG = config.ROOT / "docs" / "images" / "headline.png"
N_BINS = 20


def _reliability(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal-count bins, not equal-width: at a 3.5% base rate almost every
    transaction lands in the first equal-width bin and the curve says nothing."""
    order = np.argsort(p)
    xs, ys = [], []
    for chunk in np.array_split(order, N_BINS):
        xs.append(p[chunk].mean())
        ys.append(y[chunk].mean())
    return np.array(xs), np.array(ys)


def main() -> None:
    scored = pd.read_parquet(config.DATA / "scored_test.parquet")
    y = scored["isFraud"].to_numpy()
    policies = pd.read_csv(config.DATA / "policies.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Log axes: the decision boundaries live between 0.3 and 0.8, but 96% of the
    # mass sits below p=0.05. On linear axes both curves collapse into the corner
    # and the panel shows nothing.
    lims = (2e-4, 1.0)
    ax1.plot(lims, lims, color="0.7", lw=1, ls="--", label="perfect")
    curves = (
        ("p_bal", "class-rebalanced (AUC 0.0021 lower)", "^-"),
        ("p_raw", "raw GBDT score", "o-"),
        ("p", "isotonic — the shipped scorer", "s-"),
    )
    for col, label, style in curves:
        xs, ys = _reliability(scored[col].to_numpy(), y)
        ax1.plot(xs, ys, style, ms=4, lw=1.6, label=label)
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel="predicted P(fraud)",
        ylabel="observed fraud rate",
        title="Calibration on the out-of-time test window",
        xlim=lims,
        ylim=lims,
    )
    ax1.legend(frameon=False, fontsize=9, loc="upper left")

    annual = policies["annual"] / 1e6
    names = ["P0\napprove all", "P1\nbinary thr", "P2\ntop 1%", "P3\nper-txn EV", "P4\n4-action EV"]
    colors = ["0.75"] * (len(annual) - 1) + ["#1f77b4"]
    bars = ax2.bar(names, annual, color=colors)
    for bar, v in zip(bars, annual, strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"${v:.2f}M", ha="center", fontsize=9)
    ax2.set(
        ylabel="annual fraud cost ($M)",
        title="Cost of the action set (test window, annualised)",
        ylim=(0, annual.max() * 1.18),
    )
    ax2.tick_params(axis="x", labelsize=8)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
        ax2.spines[side].set_visible(False)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG, dpi=144)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
