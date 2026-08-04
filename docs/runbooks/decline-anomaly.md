# Runbook: FraudLensDeclineAnomaly

**Alert:** `fraudlens:decline_rate:10m` more than 3σ from its trailing 6h mean (floor
0.5pp), for 15m · **Tier 1** · dashboard `00 Executive`

## Symptom

The share of transactions being hard-declined has left its own recent distribution. On
the published policy the deny arm is **0.35% of traffic** and declines **11 good
customers** on the test window, so this band is narrow and a real move is obvious.

**The alert is bidirectional and that is deliberate.** A decline rate that collapses is as
alarming as one that spikes — it means the boundary or the score moved and we are now
approving transactions we would have declined yesterday. The loss from that lands 30–90
days later, when nothing can be done about it. A spike hurts customers today; a collapse
hurts the P&L next quarter.

## Likely cause

1. **A model was promoted or reloaded.** Check first, always. The rebalanced model in
   findings §2 challenged 52.2% of traffic instead of 6.8% — a change this alert would
   have caught within 15 minutes of deploy, a quarter before a single chargeback settled.
2. **The service is degraded.** The fail-safe ladder declines high-amount new accounts and
   challenges everything else, so an outage moves both the deny and challenge shares.
   `fraudlens:degraded_rate:5m` settles this in one query.
3. **An input broke.** See the data-breakage runbook — mispriced customers get
   misclassified customers.
4. **Traffic mix changed.** A promotion attracting new accounts genuinely raises the
   decline rate, because new accounts genuinely need a higher bar (break-even 0.642 new
   vs 0.379 at 400d+). Nothing is wrong; the volume mix moved.
5. **Real attack.** A decline spike that is *correct*. Do not suppress it.

Note the ±3σ band has a 0.5pp floor. In a quiet window σ collapses toward zero and a bare
3σ test fires on rounding; 0.5pp is below anything a human would call a policy shift, so
the floor costs no sensitivity that matters.

## First diagnostic query

```promql
fraudlens:decline_rate:10m
avg_over_time(fraudlens:decline_rate:10m[6h])
stddev_over_time(fraudlens:decline_rate:10m[6h])

# Cause 2, in one line.
fraudlens:degraded_rate:5m

# Which explanation the service is giving. A shift between SCORE_ABOVE_* codes is the
# boundary moving; a surge in FALLBACK_RULE_* is an outage wearing a policy costume.
sum by (code) (rate(fraudlens_reason_codes_total[15m]))

# Dollars, not counts. Ten declines on $5 baskets and ten on $5,000 baskets are not the
# same incident.
fraudlens:gmv_usd:rate1h
```

## Remediation

- **Degraded?** Fix the model availability; the decline rate is a symptom. See the
  latency and score-drift runbooks for the underlying causes.
- **Promotion?** Roll back to the previous artifact. It is reversible and the version pin
  in the ledger makes the affected decisions enumerable afterwards.
- **Input break?** Data-breakage runbook. Do not touch the policy.
- **Mix change or real attack?** Do nothing. The policy is behaving as designed. Record
  the finding so the next person to see this band does not re-investigate it.

Under no circumstance widen the ±3σ band to stop the page. If the band is wrong, the
trailing window is wrong, and that is a change to `recording.yml` with a reason in the
commit body.

## Who decides

On-call engineer owns artifact rollback and the degradation fix. **Changing the decision
boundary is the fraud risk owner's call** — the boundary is derived from the cost model,
and moving it by hand replaces a measured $2.4M/yr policy with an opinion. If the decline
rate is correct but commercially unacceptable, that is a conversation about the cost
constants (and about `P(churn|declined)`), not about a threshold.
