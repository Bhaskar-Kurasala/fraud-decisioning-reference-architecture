# Architecture Decision Records

Four decisions, each one that a reviewer has asked about or that changed the
shape of the code. Ordered by the problem they solve, not the day they were
written — ADR-0003 generalises four independent fixes, so its date is the day
the pattern was named, not the day it was first encountered.

| | Title | Status | The decision in one sentence |
|---|---|---|---|
| [0001](0001-preserve-research-scripts-as-provenance.md) | Preserve research scripts as provenance | Accepted | The research scripts are frozen as they ran; reformatting them would mean the committed code is no longer the code that produced the published numbers. |
| [0002](0002-quality-attribute-priorities.md) | Quality attribute priorities and explicit trade-offs | Accepted | Reliability > Auditability > Latency > Consistency > Observability — ranked for *this* system, with the trade-offs named. |
| [0003](0003-absence-must-not-render-as-a-value.md) | Absence must not render as a value | Accepted | A value that was not observed is represented as absent at every layer, and absence is made visible rather than filled in. Found four times independently before it was named. |
| [0004](0004-no-analyst-review-queue.md) | No analyst review queue, and what the case API is for instead | Accepted | Review loses money at this merchant ($14,783/yr vs no queue); the four humans in the system are served by an investigation API, not a work queue. |

---

## When to write a new ADR

When a decision is hard to reverse, has been questioned, or has a rejected
alternative that someone will reasonably propose again. Not for implementation
choices that a commit message covers. The four above each cost something to get
right, and each has a section explaining what was rejected and why — that
section is the part that saves the next person from re-litigating.
