# ADR-0003: Absence must not render as a value

**Status:** Accepted
**Date:** 2026-08-05
**Supersedes:** nothing. Generalises four fixes made independently across E4–E7.

---

## Context

This decision was not designed up front. It was extracted after the same defect
was found and fixed four times, by four different pieces of work, in four
modules that share no code:

| Where | The placeholder | What it did |
|---|---|---|
| `streaming.ledger` (948022b) | degraded decision wrote `calibrated_probability = 0.0` | biased the score distribution toward zero in proportion to outage length |
| `monitoring.maturity` (66abb01) | rows with no label yet were dropped from the denominator | a 0%-matured window reported 100% matured and certified itself |
| `monitoring.report` (4cb19f5) | refused Tier 2 gauges left unset | Prometheus holds the last value forever, so yesterday's reassuring ECE stayed on the panel |
| `serving.metrics` (1c22b4b) | degraded path would have observed probability `0.0` | same distribution bias, one layer up |

Each was found in isolation and each looked like a local slip. Together they are
one thing, and it is not a coding mistake — it is a modelling mistake. In every
case the code had a state meaning *"we did not observe this"* and stored it in a
representation that can only express *"we observed this value"*.

What makes it worth an ADR rather than four commit messages is the direction of
the resulting error. It is never neutral:

> In every instance, the placeholder made the system look **healthier** than it
> was, and it did so **in proportion to how broken the system was**.

Longer outage → more fabricated zeros → calmer drift chart. Less mature labels →
fewer eligible rows → higher reported maturity. A monitoring job that stops
computing → a gauge frozen at its last good value. The dashboard is quietest
exactly when someone should be looking at it. That is worse than no dashboard,
because a blank panel prompts a question and a green one ends it.

This matters more here than it would in most systems. Chargebacks mature over
30–90 days, so there is no ground truth arriving to contradict a flattering
number. A conventional system self-corrects when reality shows up. This one has
a quarter in which nothing can.

## Decision

**A value that was not observed is represented as absent, at every layer it
crosses, and absence is made visible rather than filled in.**

Concretely, and enforced by test rather than by review:

1. **Storage** — nullable column plus a `CHECK` constraint pairing the null with
   the flag that explains it. `ck_decision_score_presence` makes
   "`calibrated_probability IS NULL`" a *guarantee* that no model scored the row,
   not a convention a later query has to remember.
2. **Computation** — a metric that cannot be honestly computed **refuses**; it
   does not warn and return a number. `require_mature` raises. A caveat beside a
   number is not read as part of the number, and it is the first thing dropped
   when the figure is pasted into a deck.
3. **Emission** — a gauge that could not be computed is set to `NaN`, never left
   unset. Prometheus has no concept of "stale"; an untouched gauge is
   indistinguishable from a healthy one. `NaN` renders as a gap.
4. **Dashboards** — a panel with no data source yet is labelled with the epic
   that owes it. An empty panel is ambiguous between "no traffic", "exporter
   down" and "not built", and on a dashboard ambiguity reads as calm.
5. **Denominators** — a row that has not been observed yet is counted as
   unobserved, never excluded. Excluding it moves the denominator and inverts the
   statistic, which is how a maturity gate certifies an immature window.

The general form: **when in doubt, fail toward the state that prompts a
question.** This is the observability analogue of ADR-0002's fail-safe rule —
there, the safe state is not `allow`; here, the safe state is not `0.0`.

## Consequences

**Accepted costs.**

- Nullable columns push handling onto every consumer. That is the intent: the
  compiler and the type checker now ask the question the placeholder used to
  answer silently. `float | None` is a prompt.
- `NaN` in Prometheus renders as a gap, and a gap looks like an outage of the
  *exporter*. That confusion is real and is the better of the two: it sends
  someone to look, which is the point.
- Refusing to compute a metric means a dashboard has holes during immature
  windows. Those holes are the honest shape of what is knowable at that moment.

**Rejected alternatives.**

- *Sentinel values (`-1`, `-999`).* Same defect wearing a disguise. It survives
  a `mean()`, an `avg_over_time()` and a histogram bucket without complaint, and
  it is worse than `0.0` because it is not even in the valid range — so the first
  aggregate that touches it is wrong by an unbounded amount instead of a bounded
  one.
- *Filtering degraded rows in every downstream query.* This was the original
  workaround for the ledger case, as a comment telling future queries to filter
  `degraded = false`. It is correctness by documentation: it holds until the
  first analyst who writes their own query, which is the population the audit
  trail exists to serve.
- *Imputing at read time.* Moves the fabrication somewhere less visible and gives
  two consumers licence to impute differently.

**What this does not cover.** Absence of a *row* is a different problem — a
decision that was made but never reached the ledger is invisible to this rule,
which is why `fraudlens_ledger_write_failures_total` exists (1184708) and why
gap 5 in `docs/lineage.md` (the ledger is append-only by trigger, not sealed)
stays open.

## Notes

That this pattern was found four times independently, by separate work, is the
strongest evidence for it. Nobody was looking for it. It kept being the answer.
The corollary is uncomfortable and worth writing down: there are almost certainly
more instances in this codebase that have not been noticed yet, and the way to
find them is to grep for the places where a default value stands in for a
measurement — not to trust that this ADR finished the job.
