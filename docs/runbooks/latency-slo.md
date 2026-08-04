# Runbook: FraudLensLatencySLOBreach

**Alert:** `fraudlens:decide_latency_seconds:p99_5m > 0.150` for 10m · **Tier 3** ·
dashboard `02 Serving SLO`

## Symptom

The p99 of `POST /v1/decide` has been over the 150 ms budget for ten minutes. The
decision blocks a checkout (ADR-0002 priority #3), so this is cart abandonment in
progress, not a graph looking wrong. The mean is almost certainly fine — that is why the
SLO is written on the tail.

## Likely cause

Go straight to the **per-stage p99 panel**. The §4.3 allocation is features 60 ms,
inference 25 ms, calibration + policy 15 ms, with 50 ms of overhead inside the 150 ms
total, and whichever stage crossed its own line names the cause:

| Stage over budget | Cause |
|---|---|
| `features` | Economic assembly — tenure bucketing and the two cost arrays. Pure arithmetic on a handful of values, so a breach here is a CPU-starvation or GC signal, not a slow query. There is no feature store in this deployment. |
| `inference` | The model. A larger artifact after a promotion, thread contention, or a host without the CPU the artifact expects. |
| `calibration_policy` | Isotonic lookup plus the EV argmax. Should be microseconds; a breach means something is wrong with the artifact itself. |
| **No stage over budget, total over** | The 50 ms of overhead: request deserialisation, strict validation, response serialisation, or the ASGI/event-loop layer. Look at payload size — `features` is caller-supplied and an unusually wide vector costs validation time. |

## First diagnostic query

```promql
# Which stage. This is the whole diagnosis.
histogram_quantile(0.99, sum by (le, stage) (rate(fraudlens_decision_stage_seconds_bucket{stage!="total"}[5m])))

# Breach counter by stage — catches a stage eating another's headroom while the total is
# still inside 150 ms.
sum by (stage) (rate(fraudlens_latency_budget_breaches_total[5m])) * 60

# Is this load, or is it per-request cost?
sum(rate(fraudlens_requests_total{route="/v1/decide"}[5m]))

# Restarts and memory — a p99 that is really a cold start after a crash loop.
changes(process_start_time_seconds[1h])
process_resident_memory_bytes
```

**Trap:** check `fraudlens:degraded_rate:5m` before concluding anything is fine. A
degraded decision skips inference and calibration entirely, so a model outage *improves*
these percentiles. Latency going green while the degraded rate climbs is a worse
situation, not a resolved one.

If a trace backend is wired, the spans carry `fraudlens.model_version` — the question
"are the slow requests all on one artifact" is one query away and the histogram cannot
answer it.

## Remediation

- Rolling restart if memory or restart count implicates the process, but check that it is
  not a crash loop first: restarting a pod that is failing to load a model produces
  another pod that fails to load a model.
- Roll back the model artifact if the breach starts at a promotion boundary. Latency is a
  legitimate rollback reason on its own; it does not need a cost argument.
- Scale out. The service is stateless by design (ADR-0002) so this is safe, and it is the
  right move when throughput and latency rose together.
- If the breach is in the overhead band and payloads have grown, that is a contract
  conversation with the caller, not a tuning exercise here.

## Who decides

On-call engineer owns restarts, scaling and artifact rollback — all reversible, none
change decision economics. Relaxing the 150 ms budget is **not** an on-call decision: the
number is an SLO locked by a test (`test_budget_matches_the_spec_allocation`) precisely so
it cannot be edited under pressure. That is a product and fraud-risk decision.
