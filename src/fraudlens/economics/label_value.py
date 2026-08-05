"""What an early label is worth, when the alternative is waiting for a chargeback.

Findings §5 prices analyst review as a decision: does a human profitably change *this*
decision? At this merchant, almost never — value-of-review is positive on 5 of 92,427
transactions. That accounting is right and complete *for the decision*, and it omits the
other thing an analyst produces. An analyst adjudication is a label **today**. A
chargeback is a label after a 34-day median dispute, and a good transaction is only
confirmed good when the 90-day window closes silently. §6 measured what that latency
costs; `monitoring.maturity` refuses every label-dependent metric until a window is 80%
matured, which under the dispute-lag model happens on day 90 and not one day earlier.

So a review queue is also an instrument, and these functions price it as one: how many
days of degraded operation it removes, and what that is worth against the measured
$4,358,096/yr cost of running a miscalibrated model.

They do **not** price labels as training data. The queue samples the top ~2% by
value-of-review — a censored, non-random sample sitting entirely above the decision
boundary — and `docs/findings/review-as-label-acquisition.md` shows that mechanism is
worth nothing. Only the monitoring mechanism survives, because a detector needs its
sampling rule to be *stationary*, not unbiased.

Pure functions over scalars. No I/O, no state.
"""

from __future__ import annotations

import math
from statistics import NormalDist

from fraudlens.config import SETTINGS, BusinessConstants
from fraudlens.economics.expected_value import DAYS_PER_YEAR

_NORMAL = NormalDist()

# The maturity floor `monitoring.maturity` refuses below. Duplicated as a default rather
# than imported: economics sits below monitoring in the layered contract, and inverting
# that to share one constant would let a reporting decision reach into the cost model.
DEFAULT_MATURITY_FLOOR = 0.80


def maturity_fraction(age_days: float, fraud_rate: float, c: BusinessConstants = SETTINGS) -> float:
    """Share of a transaction cohort whose true label is knowable `age_days` later.

    Two arrival processes, and the small one is the fast one. Fraud discloses lognormally
    (median 34d); a good transaction is never confirmed by a positive event, so it lands
    as a step at the 90-day dispute-window close. With a 3.5% fraud rate that step is 96%
    of the mass, which is why maturity is a cliff rather than a curve.
    """
    if age_days <= 0:
        return 0.0
    z = (math.log(age_days) - math.log(c.chargeback_median_days)) / c.chargeback_log_sigma
    disclosed_fraud = fraud_rate * _NORMAL.cdf(z)
    clean = (1.0 - fraud_rate) if age_days >= c.clean_window_days else 0.0
    return disclosed_fraud + clean


def days_to_maturity_floor(
    fraud_rate: float,
    floor: float = DEFAULT_MATURITY_FLOOR,
    c: BusinessConstants = SETTINGS,
) -> float:
    """How long a window must age before a label-dependent metric may be computed on it.

    This is the deadline the whole analysis turns on. Solved in closed form rather than by
    search because the function is a step: before the clean-window close the most that can
    have matured is `fraud_rate` (~3.5%), so any floor above that is met on the close day
    and not before. No floor an operator would set is below 3.5%, so in practice this
    always returns `clean_window_days` — and it is written this way so that the day it
    does not, the reason is visible.
    """
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"maturity floor must be in (0, 1]; got {floor}")
    if floor > fraud_rate:
        return c.clean_window_days
    z = _NORMAL.inv_cdf(floor / fraud_rate)
    return float(math.exp(math.log(c.chargeback_median_days) + c.chargeback_log_sigma * z))


def days_to_detect_rate_shift(
    baseline_rate: float,
    shifted_rate: float,
    cases_per_day: float,
    *,
    analyst_agreement: float = SETTINGS.q_analyst,
    sigmas: float = 3.0,
) -> float:
    """Days of analyst labels needed to see a shift in the reviewed cohort's fraud rate.

    A fixed-sample `sigmas`-deviation test on a Bernoulli proportion, deliberately crude.
    A CUSUM or a sequential test would fire sooner on the same data, so this **overstates**
    detection time and therefore understates the value of the labels — the direction to
    err in when the conclusion is "the labels are worth buying".

    Analyst error enters as symmetric misclassification: an observed rate of
    `(1-q) + (2q-1)r`. The shift the chart sees is attenuated by `2q-1` (0.82 at q=0.91),
    which is the whole cost of imperfect analysts here. It slows detection; it does not
    bias the *direction*, which is why a noisy label still works as a detector.
    """
    if not 0.5 < analyst_agreement <= 1.0:
        # At q <= 0.5 the label carries no information about the direction of the shift.
        raise ValueError(f"analyst_agreement must be in (0.5, 1]; got {analyst_agreement}")
    if cases_per_day <= 0:
        raise ValueError(f"cases_per_day must be positive; got {cases_per_day}")
    attenuation = 2.0 * analyst_agreement - 1.0
    observed_baseline = (1.0 - analyst_agreement) + attenuation * baseline_rate
    delta = attenuation * (shifted_rate - baseline_rate)
    if delta == 0.0:
        return math.inf
    cases = sigmas**2 * observed_baseline * (1.0 - observed_baseline) / delta**2
    return cases / cases_per_day


def label_acquisition_value(
    days_saved: float,
    excess_cost_per_year: float,
    events_per_year: float,
) -> float:
    """Annual value of detecting decay `days_saved` sooner.

    The whole causal chain, stated as arithmetic so it can be argued with: earlier labels
    remove `days_saved` from every episode of operating a degraded model, each episode
    costs `excess_cost_per_year` while it lasts, and episodes arrive `events_per_year`
    times. It assumes detection is immediately actionable — if rollback takes a fortnight,
    subtract a fortnight from `days_saved`.
    """
    return max(days_saved, 0.0) / DAYS_PER_YEAR * excess_cost_per_year * events_per_year


def break_even_events_per_year(
    programme_cost_per_year: float,
    days_saved: float,
    excess_cost_per_year: float,
) -> float:
    """Decay-event frequency at which the queue pays for itself purely as an instrument.

    Returned rather than a yes/no because the hazard rate is the one input this dataset
    cannot supply — one 32-day window, zero observed decay events — so it is the operator's
    number, and the useful output is the threshold they have to beat.
    """
    per_event = label_acquisition_value(days_saved, excess_cost_per_year, 1.0)
    return math.inf if per_event <= 0.0 else programme_cost_per_year / per_event
