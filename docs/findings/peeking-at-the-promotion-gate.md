# What continuous peeking costs, measured

**Date:** 2026-08-05
**Epic:** E8
**Code:** `src/fraudlens/models/sequential.py`, `tests/flywheel/test_confidence_sequence.py`

---

## The problem this note records

The promotion gate compares champion and challenger on realised cost, which needs matured
labels. Labels do not arrive in a batch — chargebacks trickle in over 30–90 days. So the
comparison is not something you run once at a pre-registered sample size. It is something
that gets re-run every time more of the window matures, and that gets *acted on* the first
time it looks conclusive.

That pattern breaks a fixed-horizon interval. §4.2 asserts this and specifies a sequential
test; this note is the measurement behind the assertion, because "sequential testing
because peeking" is the kind of claim that gets repeated without anyone checking the
magnitude.

## Measured

400 simulated null paths (champion and challenger genuinely equal-cost), 60 looks per path
log-spaced from n=30 to n=20,000, α = 0.05. Recorded: the fraction of paths where the
interval excluded zero at *any* look — i.e. where the gate would have promoted or blocked
on a difference that does not exist.

| Interval | Null | False conclusion at ≥1 of 60 looks |
|---|---|---|
| Robbins normal-mixture confidence sequence | Gaussian | **0.25%** |
| Robbins normal-mixture confidence sequence | Heavy-tailed (differenced Γ(0.4, 60)) | **0.25%** |
| Fixed-horizon 95% interval | Gaussian | **49.25%** |

A fixed-horizon interval, read the way this gate necessarily reads it, is wrong about
half the time under the null. Not "slightly inflated" — a coin flip. If the gate used one,
roughly every second challenger that is truly no better than the champion would at some
point during maturation look certifiably better, and the gate would promote it.

## The guarantee, stated precisely

For the boundary in `sequential._mixture_half_width` with mixture parameter ρ² = 1/h:

> P( ∃ n ≥ 2 : the true mean paired cost difference lies outside CI_n ) ≤ α

One α for the entire, unbounded sequence of looks — not one α per look. Consequences that
matter operationally: no alpha-spending schedule, no pre-registered horizon, and the gate
may stop and promote at the first look that clears zero without any correction.

**Where it is weaker than the citation.** Howard et al. (2021) §3 state this for a known or
sub-Gaussian-bounded scale. `paired_cost_delta` plugs in the sample standard deviation, so
coverage is asymptotic rather than exact, and finite-n coverage depends on the tail of the
cost distribution. That distribution is genuinely heavy here — one high-amount chargeback
dominates a window — which is why the heavy-tailed row above is in the table and not
omitted. An exactly-valid nonparametric version needs an empirical-Bernstein or
betting-style boundary. This is not one, and calling it one would be the kind of overclaim
this note exists to avoid.

## What it costs

0.25% against a 5% budget means the boundary is conservative by roughly 20x in error rate,
which shows up as width: `half_width · √n / s` is 3.0–8.2 depending on n, against 1.96 for
the one-shot interval. The gate therefore needs a **1.5–4x larger effect** than a
fixed-horizon test would, at the same n.

That is not free, and it is worth being explicit about which effects it excludes:

| Effect (findings) | Per-txn | Certifiable on one 92,427-row window? |
|---|---|---|
| Class-rebalanced regression, §2 | ~$4.14 | **Yes** |
| Raw-vs-isotonic penalty, §2 | ~$0.07 | **No** |

The $0.07 case is instructive. On the seed used in the test, the *sample mean* of a true
$0.07/txn saving comes out at +$0.12 — the wrong sign — inside $60 of per-transaction
dispersion. Any gate deciding on the point estimate would flip on the seed. So the
sequence declining to certify it is correct behaviour, and the honest reading is that
improvements of that size are not shippable on one window's evidence at all. That agrees
with the blueprint's §4.8 power calculation from the other direction: a 1% relative
improvement needs ~112M per arm, which is 140 days, which is longer than the retrain
cadence it would be justifying.

## What this does not settle

The conservatism is a real loss and it is not obviously the right trade. A tighter
anytime-valid boundary (empirical-Bernstein, or a betting martingale) would recover much
of the 20x, and would also remove the plug-in-variance caveat. Not done here; the current
boundary is defensible and the alternative is a research task, not an implementation one.
The reason it is worth revisiting is that the gap between $4.14 and $0.07 is where most
real challengers live.
