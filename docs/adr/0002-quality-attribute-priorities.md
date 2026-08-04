# ADR-0002: Quality attribute priorities and explicit trade-offs

**Status:** Accepted
**Date:** 2026-08-05
**Epic:** E0

## Context

Production systems are evaluated along a long list of quality attributes —
reliability, availability, latency, scalability, security, auditability,
portability, and a dozen more. They cannot all be maximised simultaneously;
every architectural decision trades one against another.

The architect's job is not to pursue all of them. It is to rank them for *this*
system, make the trade-offs explicit, and say which attributes are deliberately
not being optimised — so that a later reader can tell a considered omission from
an oversight.

## Decision

### The ranking for this system

A fraud decisioning service sits synchronously in the checkout path and makes
per-customer adverse decisions that are subject to dispute. That combination
sets the order:

**1. Correctness of the decision economics.**
The system's entire value is that it declines the right transactions. A
correctness bug in the cost functions is not a degraded experience; it is
direct, silent, uncapped financial loss. E1 measured the scale: a model
0.0021 AUC below champion — indistinguishable to any ranking-based gate —
costs $4.37M/yr because its calibration is 40× worse. This ranks above
availability, which is unusual and deliberate.

**2. Auditability.**
Every decline is an adverse action against an identifiable customer. It must be
reconstructable after the fact: which model version, which feature values, which
policy version, which threshold, what the score was. This is a regulatory
posture (SR 11-7, EU AI Act) but also an operational one — disputes are answered
from the audit log, not from memory.

**3. Latency.**
The decision blocks a checkout. Budget: p99 ≤ 150 ms end to end. Tail latency is
the SLO, not the mean — the mean is served fine while the p99 abandons carts.

**4. Reliability and graceful degradation.**
Fail-safe means *fail to the safe state*, and for fraud the safe state is not
"approve". On model unavailability the service degrades to a documented
rule-based fallback and marks the decision as degraded in the ledger, so the
downstream analysis can exclude it rather than silently absorb it.

**5. Observability.**
Distinguished from the above because the system's failure mode is *silence*:
labels arrive 30–90 days late, so a broken model looks exactly like a working
one for a quarter. Tier 0 leading indicators exist for this reason.

### What we are deliberately not optimising, and why

Stating these prevents a reader from mistaking absence for oversight:

| Attribute | Decision | Rationale |
|---|---|---|
| **Multi-region / HA topology** | Not built | Single-node Compose. The interesting failure modes here are model-level (drift, calibration decay, label latency), not infrastructural. Multi-AZ would add operational surface without exercising any decision this system exists to make. Health and readiness probes *are* implemented, since they are cheap and shape the deployment contract. |
| **Horizontal autoscaling / elasticity** | Not built | The service is designed stateless so it *can* scale horizontally, and that property is asserted by test. Actually running an autoscaler against a replayed stream would demonstrate nothing. |
| **Accessibility (WCAG)** | Not applicable | No user interface. Grafana is operator tooling and is not a delivered product surface. |
| **i18n / l10n** | Not applicable | No user-facing strings. Reason codes are stable machine identifiers, deliberately not display text — localisation belongs to whatever renders them. |
| **PCI-DSS scope** | Avoided by design | The system never touches a PAN. IEEE-CIS ships pre-tokenised card identifiers (`card1`–`card6`); the production analogue is to receive tokens and never handle raw card data. The cheapest compliance posture is staying out of scope. |
| **GDPR erasure workflow** | Documented, not implemented | The lineage design records where customer-linked data lives so the deletion path is *identifiable*. Building the workflow against a static public dataset with no real subjects would be theatre. |
| **Zero-trust service mesh** | Not built | One service. mTLS between two containers demonstrates configuration, not architecture. |

### The trade-offs actually being made

| Choice | Buys | Costs |
|---|---|---|
| Synchronous scoring in the checkout path | Decision before money moves | Latency becomes a hard constraint; no expensive features |
| Append-only decision ledger | Auditability, replay, flywheel training data | Write amplification; storage grows unbounded without retention policy |
| Isotonic calibration on a held-out slice | $4.37M/yr (E1, measured) | A third data split, so less training data |
| Fail-safe to rules, not to approve | No fraud window during an outage | False declines during degradation; must be measured, not assumed |
| Static replay instead of synthetic traffic | Real drift, real label latency, honest measurement | Cannot exceed the dataset's 182 days or its adversary regime |
| One model, not a portfolio | Fits the dataset and the evidence | Cannot demonstrate per-fraud-class economics (FN:FP spans 1:8 to 355:1 per the blueprint) |

### Where the framework's defaults do not fit ML

Two standard practices need adjustment, and the adjustment is the point:

**Error budgets** assume failure is observable. Here the dominant failure —
model decay — is invisible for 30–90 days. A conventional error budget burns on
5xx responses while the model quietly costs millions. The budget is therefore
defined on *Tier 0 leading indicators* (drift, calibration on matured labels,
training-serving skew), not solely on request success.

**"Every service has a golden-signals dashboard"** (latency, traffic, errors,
saturation) is necessary and insufficient. All four can be green while the
system is at its most expensive. The default landing dashboard is business P&L,
with golden signals one layer down — an inversion of the usual order, justified
by the measurement in E1.

## Consequences

- Reliability work targets decision correctness and fail-safe behaviour rather
  than topology.
- The audit trail is a first-class component with its own tests, not a logging
  side effect.
- Load testing validates the p99 budget rather than peak throughput.
- Reviewers can distinguish "not built, and here is why" from "not considered".
