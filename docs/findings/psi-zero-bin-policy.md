# PSI is undefined when a bin empties, and the fix you pick moves the alert

**Date:** 2026-08-05
**Epic:** E7
**Status:** Decided

## The problem

Spec §5.2 pages on `PSI > 0.25`. PSI is

```
PSI = Σ (a_i − r_i) · ln(a_i / r_i)
```

which is `−inf`, `+inf` or `nan` the moment any bin's actual or reference mass is zero.
That is not an edge case in this system. The calibrated score is spiky near zero, deciles
are narrow at the bottom of the distribution, and an upstream feature going constant
empties several bins at once. So the alert threshold is only as meaningful as the choice
made about empty bins — and that choice is never printed on the dashboard next to the
number it produced.

## The decision

**Floor both proportion vectors at ε = 1e-4 before the sum.**

Computed consequence, at ten bins: one wholly emptied decile contributes

```
(1e-4 − 0.1) · ln(1e-4 / 0.1) = 0.690085
```

on its own — **2.8× the 0.25 alert threshold**. A single vanished decile therefore pages
by itself, before any other bin is considered. That is the intended behaviour: a tenth of
traffic disappearing from a value range is a breakage, not drift.

## Why an arbitrary constant is nonetheless defensible

The magnitude of ε is arbitrary; its effect on the *decision* is not. The empty-bin term
scales as `ln(1/ε)`:

| ε | contribution of one emptied decile | fires at 0.25? |
|---|---|---|
| 1e-3 | 0.456 | yes |
| 1e-4 | **0.690** | yes |
| 1e-6 | 1.151 | yes |

Three orders of magnitude move the reported severity by 2.5× and change the alert
outcome not at all. The number is a signal, not an estimate, and it should be read that
way: above 0.25 means act, and the specific value above 0.25 is not a measurement of
anything.

Secondary effect, stated so nobody rediscovers it: flooring breaks the sum-to-one of both
vectors by at most `n_bins · ε = 1e-3`, which is far below the resolution of any decision
taken on this number.

## Rejected: merge empty bins into their neighbours

Numerically the more principled fix, and operationally backwards. The worse the shift,
the more bins merge; the more bins merge, the fewer terms the sum has; the fewer terms,
the smaller the reported PSI. It monotonically suppresses precisely the event the alert
exists for. A total distribution collapse would merge to one bin and report PSI ≈ 0.

## Rejected: refuse to report when a bin empties

Correct in a research notebook, indefensible on call. It turns the most severe drift into
a missing datapoint, and a gap on a Grafana panel reads as "the exporter is down", not
"act now". It also silently disables the alert: Prometheus cannot fire `> 0.25` on a
series that has no samples.

## Related choices that also move the number

- **Ten bins.** The 0.25 convention was established on deciles. More bins raises PSI on
  the same data. Fixed rather than configurable, and the bin edges travel *inside* the
  baseline artifact so a baseline captured with different bins cannot be compared to one
  captured with these.
- **Nulls are excluded from the bins, not folded into the lowest one.** A missing value
  is not a small value. Folding them in makes a data-pipeline outage indistinguishable
  from a population shift — two incidents with entirely different runbooks. The null rate
  is measured and alerted separately (§5.2, "Data breakage").
- **Unseen categorical levels are pooled into a bucket rather than dropped.** Dropping is
  the common implementation and it makes the surviving distribution look stable at exactly
  the moment the population changed underneath it. The bucket's reference mass is zero by
  construction, so it is floored, and entity churn alerts.

## Friction found while building on the existing code

`models.metrics.expected_calibration_error` is reused rather than reimplemented, per the
correction recorded in `fit-balanced-empirical-result.md`. It exposes only the scalar,
not the bin assignment that produced it, so `monitoring.calibration.reliability_curve`
has to re-derive the same quantile edges to plot the curve — which is the exact
re-derivation that produced the 0.0035-vs-0.0027 error in the first place.

Extending `models/` is not in E7's scope, so the duplication is converted into a checked
invariant instead: `tests/monitoring/test_calibration.py` asserts that the curve's
mass-weighted gaps sum to `expected_calibration_error` to machine precision. If the two
binnings ever diverge the test fails, rather than a dashboard reporting a calibration
figure the promotion gate disagrees with.

**Owed to `models/metrics.py`:** export the bin assignment (edges + index) alongside the
scalar, so there is one binning rather than one binning and a pinned copy of it.
