# Runbook: FraudLensDataBreakage

**Alert:** `fraudlens:unknown_tenure_rate:10m` above 2× its trailing 6h rate and ≥ 2% of
traffic, for 15m · **Tier 0** · dashboard `03 Data quality`

## Symptom

`days_since_first_seen` is arriving null on a materially increased share of requests, and
those customers are being priced at the **median relationship cost** instead of their own.

This matters more than a generic null-rate alert because that field is not decoration: it
selects the relationship-cost term in `M = Amount × 0.30 + relationship_cost(tenure)`,
which is the false-positive arm, which sets the break-even probability. Break-even runs
**0.642 for a brand-new account and 0.379 at 400d+**. Pricing an unknown customer at the
median silently moves the boundary for every one of them.

Nothing crashes. Every request returns 200 with a valid decision and a
`UNKNOWN_TENURE_PRICED_AS_MEDIAN` reason code. That is exactly why it needs an alert.

## Scope note

§5.2 specifies "null rate breach vs training baseline" across features. That comparison
needs the training baseline and belongs to E7's monitoring workers. This alert covers the
one missing input the *serving process* can see unaided — and it is the one that costs
money. When E7 lands, the same alert gains the per-feature baseline comparison.

## Likely cause

1. **The caller's tenure/account lookup is failing.** The service has no feature store;
   the caller assembles the vector (`FEATURE_VERSION = request-supplied-v1`), so a null
   here is *their* dependency, not ours. Almost always the answer.
2. **A new caller or a new integration path** that never populated the field. Check
   whether the increase is total or concentrated.
3. **A genuine surge in first-time customers** — a campaign, a new market. Real, and the
   correct response is different: nothing is broken, but the median pricing is now being
   applied to a large cohort it was not sized for.

## First diagnostic query

```promql
# Is it the tenure field alone, or is everything degrading?
fraudlens:unknown_tenure_rate:10m

# Did the decision mix move with it? This is the money question — if the deny/challenge
# share moved, customers are being treated differently right now.
fraudlens:decisions:rate5m / scalar(sum(fraudlens:decisions:rate5m))

# Volume context: a rate spike on collapsed traffic is a different incident.
sum(rate(fraudlens_requests_total{route="/v1/decide"}[5m]))
```

## Remediation

- Page the caller / feature-assembly owner. **We cannot fix this from the scoring
  service**, and we should not try: a default value invented here would be exactly the
  fabricated input the fail-safe ladder refuses to feed the cost model.
- Confirm the blast radius by joining the ledger on `UNKNOWN_TENURE_PRICED_AS_MEDIAN`.
  Those decisions are auditable and, if the pricing turns out to have been wrong for
  them, individually reversible — that is what the reason codes are for.
- Do **not** disable the reason code or the alert to quiet the pager. The code is on the
  customer's decision record; suppressing it removes the evidence, not the problem.

## Who decides

On-call engineer owns diagnosis and the page to the upstream owner. If the cause is a
genuine cohort shift rather than a break, the fraud risk owner decides whether the median
relationship cost is still the right price for that cohort — that is a repricing
decision, not an operational one.
