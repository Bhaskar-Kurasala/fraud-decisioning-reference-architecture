# ADR-0001: Preserve the research scripts, extract their logic

**Status:** Accepted
**Date:** 2026-08-05
**Epic:** E0

## Context

The research pipeline (`01..06_*.py`) produced `fraud-decisioning-findings.md`,
which makes specific, falsifiable claims: a policy worth $2.4M/yr, a $6.6M/yr
miscalibration penalty, break-even thresholds that invert with basket size.

Those claims are the reason the project continues. Anything that weakens our
ability to say *"this number came from this code, run on this data"* destroys
the asset.

Meanwhile the production system needs the same logic — the same feature
definitions, the same cost functions, the same policy — available as a tested,
importable library that can serve a request in under 150 ms.

## Decision

Preserve the research scripts under `research/`, unmodified, and extract their
logic into `src/fraudlens/`. Then refactor the research scripts to import from
the extracted package.

The first commit of this repository contains the scripts exactly as they ran.

## Alternatives considered

**Sidecar — leave the research alone, build production beside it.**
Rejected. This produces two definitions of every feature and every cost
constant. They agree on the day they are written and drift silently
thereafter. The failure is discovered when the production system declines
customers using economics that no longer match the analysis justifying the
policy — and there is no test that can catch it, because the two definitions
are never compared.

**Full rewrite — replace the scripts with the package.**
Rejected. It severs the link between the findings and the code that produced
them. Six months from now, "why is the break-even 0.642 for new customers?" has
no answer that can be checked.

## Consequences

Positive:

- Drift becomes a test failure rather than an incident. The golden-value suite
  asserts the extracted library reproduces the published numbers; if a
  refactor changes a cost function, a test naming the affected finding fails.
- The research scripts stay executable, so any published number can be
  regenerated from source.
- One definition of the economics, used by both the analysis and the money.

Negative:

- Extraction is not free, and it is the phase most likely to introduce a subtle
  numerical change. Mitigated by doing verification (E1) *first*, so the
  extraction has a locked reference to be checked against.
- `research/` will accumulate as unmaintained code. Accepted deliberately: it is
  a record, and it is linted for correctness but exempt from style enforcement
  and coverage requirements.

## Notes

The ordering matters and is not incidental. Verification precedes extraction
because extraction without a verified reference is unfalsifiable — if a number
changes, we could not tell whether the extraction broke it or the original was
wrong.
