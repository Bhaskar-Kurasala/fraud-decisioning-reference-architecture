"""Label maturity: the gate that stands in front of every Tier 2 number.

An AUC computed over a window that is 30% matured is not an AUC. It is a statement about
which disputes happened to land first, and it is biased in a knowable direction — two
directions, in fact, depending on which of the two available mistakes you make:

- **Evaluate on the revealed rows only.** Fraud discloses at a 34-day median; a clean
  transaction is only confirmed clean when the 90-day dispute window closes silently. So
  early in maturation the revealed set is *fraud-enriched*. Observed fraud rate reads
  high, the model looks like it systematically under-predicts, and the calibration
  intercept is biased upward. A challenger that predicts higher probabilities looks
  better calibrated than it is.
- **Treat unrevealed rows as non-fraud.** The opposite error, and the one findings §6
  measured: the observed fraud rate collapses to **6.9% of the true rate** in the final
  20 days. Everything reads as over-prediction, and the model that allows more traffic
  looks cheapest.

Both are wrong, in opposite directions, by an amount that depends on how far through
maturation the window is. There is no correction to apply because the bias depends on the
unobserved labels. So this module refuses rather than warns. A warning next to a number
gets read as a number; §9a's rule is that errors which lose money fail loudly, and a
promotion decided on a 30%-matured window is exactly that.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

# t+7/30/60/90 per spec §5 Tier 2. 7 catches the fast-dispute head, 90 is the network
# dispute-window close, and 60 is where the gate's 0.80 floor is typically reached — the
# curve is only interesting if it brackets the threshold somebody has to defend.
MATURITY_HORIZONS: tuple[int, ...] = (7, 30, 60, 90)

# Same floor as `models.checks.GateThresholds.min_label_maturity`. Duplicated as a
# default rather than imported: the gate's floor is a promotion decision and this one is
# a reporting decision, and they are free to diverge — a dashboard may reasonably show a
# number at 0.70 that a promotion may not be decided on. Anything that must agree with
# the gate should pass the gate's value in explicitly.
DEFAULT_MATURITY_FLOOR = 0.80


class ImmatureWindowError(RuntimeError):
    """Raised when a label-dependent metric is requested on an unusable window."""


@dataclass(frozen=True, slots=True)
class MaturityPoint:
    """Fraction of labels that had landed by t+`horizon_days`.

    `n_eligible` is the transactions old enough at `as_of` for the horizon to have
    elapsed. Rows too recent to have had their t+90 are excluded rather than counted as
    unlabelled, which would otherwise drag every long-horizon point toward zero purely
    because the window is recent.
    """

    horizon_days: int
    n_eligible: int
    n_revealed: int

    @property
    def ratio(self) -> float:
        return self.n_revealed / self.n_eligible if self.n_eligible else float("nan")


@dataclass(frozen=True, slots=True)
class LabelMaturity:
    """How much of one window is knowable, as of one instant."""

    window_start: dt.datetime
    window_end: dt.datetime
    as_of: dt.datetime
    n_decisions: int
    n_revealed: int
    curve: tuple[MaturityPoint, ...]

    @property
    def ratio(self) -> float:
        return self.n_revealed / self.n_decisions if self.n_decisions else 0.0

    def stamp(self) -> str:
        """The maturity stamp §5 requires on every Tier 2 panel.

        Every label-dependent number this package emits carries one. A Tier 2 figure
        without it is not interpretable, and an uninterpretable number on an operator's
        screen is worse than a blank panel because it will be acted on.
        """
        return (
            f"{self.ratio:.1%} of {self.n_decisions} labels matured as of {self.as_of.isoformat()}"
        )


def label_maturity(
    transaction_times: Sequence[dt.datetime],
    revealed_times: Sequence[dt.datetime | None],
    *,
    as_of: dt.datetime,
    window_start: dt.datetime,
    window_end: dt.datetime,
    horizons: Sequence[int] = MATURITY_HORIZONS,
) -> LabelMaturity:
    """Measure maturity of a decision window.

    `revealed_times[i]` is when transaction `i`'s label landed, or None if it has not.
    Reveal times after `as_of` are treated as not-yet-revealed rather than trusted, so a
    caller that queried the label table without the `revealed_at <= as_of` filter cannot
    leak a future label into a maturity figure — the one number whose whole job is to
    say how much of the future we cannot see yet.
    """
    if len(transaction_times) != len(revealed_times):
        raise ValueError("transaction_times and revealed_times must be the same length")
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")

    # A reveal time after `as_of` becomes None rather than dropping the row. Dropping it
    # would shrink the denominator instead of the numerator, so a window where nothing has
    # matured would report 100% matured and certify itself — the exact leak the gate
    # exists to prevent, inverted.
    known = [
        (txn_at, revealed_at if revealed_at is not None and revealed_at <= as_of else None)
        for txn_at, revealed_at in zip(transaction_times, revealed_times, strict=True)
    ]
    n_revealed = sum(1 for _, revealed_at in known if revealed_at is not None)

    curve: list[MaturityPoint] = []
    for horizon in horizons:
        delta = dt.timedelta(days=horizon)
        eligible = [(t, r) for t, r in known if t + delta <= as_of]
        curve.append(
            MaturityPoint(
                horizon_days=horizon,
                n_eligible=len(eligible),
                n_revealed=sum(1 for t, r in eligible if r is not None and r <= t + delta),
            )
        )
    return LabelMaturity(
        window_start=window_start,
        window_end=window_end,
        as_of=as_of,
        n_decisions=len(transaction_times),
        n_revealed=n_revealed,
        curve=tuple(curve),
    )


def require_mature(maturity: LabelMaturity, floor: float = DEFAULT_MATURITY_FLOOR) -> None:
    """Refuse the window if it is not matured enough to compute a Tier 2 metric on.

    Refuse, not warn. The alternative — emit the number with a caveat — has been tried
    everywhere in this industry and the caveat is what gets dropped when the figure is
    pasted into a deck.
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"maturity floor must be in (0, 1]; got {floor}")
    if maturity.ratio >= floor:
        return
    raise ImmatureWindowError(
        f"refusing to compute a label-dependent metric: {maturity.stamp()}, "
        f"below the {floor:.0%} floor. On the revealed rows alone the window is "
        f"fraud-enriched (34-day dispute median against a 90-day clean-window close), "
        f"so observed fraud rate reads high and the model reads as under-predicting; "
        f"counting the unrevealed rows as clean inverts that (findings §6: observed "
        f"rate falls to 6.9% of true). Neither is correctable from what is observed."
    )
