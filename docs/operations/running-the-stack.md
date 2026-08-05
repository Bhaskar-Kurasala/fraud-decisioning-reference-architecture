# Running the stack

What it is, how to start it, what the health endpoints mean, and what to do when it will
not come up. Everything below was executed on the machine described in design spec §1.2
(10 cores, 11 GB, Docker 29.5.2, Compose v5.1.4, no Kubernetes) on 2026-08-05; numbers that
were measured say so, and numbers that were not are absent rather than estimated.

---

## 1. What runs

| Service | Profile | Port | Why it exists |
|---|---|---|---|
| `postgres` | core | 5432 | The decision ledger. ADR-0002 ranks auditability second of five, so it is in the smallest runnable stack, not in the optional tier. |
| `migrate` | core | — | One-shot. Applies the numbered migrations and exits. |
| `api` | core | 8000 | The scoring service. |
| `mlflow` | full | 5000 | Shared tracking + registry backend (closes half of `docs/lineage.md` gap 8). |
| `prometheus` | full | 9090 | Scrapes `api:8000/metrics`, evaluates the seven §5.2 alert rules. |
| `grafana` | full | 3000 | The four provisioned dashboards, landing on business P&L. |

`core` is the smallest thing that can take a decision **and record it**. `full` adds
everything needed to watch it and nothing that changes what it decides.

There is no second replica, no autoscaler and no service mesh, because ADR-0002 rules all
three out by name. The Kubernetes manifests in `deploy/k8s/` are authored and validated
statically; **they have never been applied**, because no cluster is available here.

---

## 2. Starting it

```bash
# smallest useful stack
docker compose -f deploy/compose/docker-compose.yml --profile core up -d --wait

# everything, including the dashboards
docker compose -f deploy/compose/docker-compose.yml --profile full up -d --wait
```

Every published port is overridable, because 3000, 5000 and 5432 are the three most
contended ports on a developer machine:

```bash
GRAFANA_PORT=3001 docker compose -f deploy/compose/docker-compose.yml --profile full up -d
```

The test harness reads the same variables with the same defaults, so a machine with a
conflict runs the drills unchanged.

---

## 3. Getting a model into it

**There is no champion artifact in this repository.** `research/03_model.py` and
`03b_calibrate.py` write out *scores* (`data/p_te_raw.npy`, `data/scored_test.parquet`) and
discard the fitted estimator, so there is nothing to deploy. The stack therefore starts
**not ready and serving fail-safe decisions**, which is a correct and supported state — see
§4 — and not a misconfiguration to be hidden behind a default artifact.

To give it something to score with:

```bash
uv run python scripts/build_model_bundle.py     # writes deploy/compose/bundles/scorer.pkl
docker compose -f deploy/compose/docker-compose.yml --profile core restart api
```

That bundle is versioned `reference-hgb-*`, never `champion-*`. It is a fixture for the
drills — a 120k-row subsample, numeric features only — and the version string travels into
every ledger row and every API response so a decision taken by it is distinguishable from
one taken by the real model forever after.

The alternative source is the registry: set `MLFLOW_TRACKING_URI` and
`FRAUDLENS_MODEL_VERSION` and leave `FRAUDLENS_MODEL_BUNDLE` empty. A mounted bundle wins
over the registry when both are set, which is how you roll back when the registry is the
thing that is broken.

---

## 4. What the health endpoints mean

This is the most important section on this page, because getting it wrong converts a model
outage into a checkout outage.

| Endpoint | Question | Green when | Consequence of red |
|---|---|---|---|
| `GET /health/live` | Can this process answer at all? | Always, once started. Never consults the model. | The container is restarted. |
| `GET /health/ready` | Can this instance produce a *scored* decision? | The artifact loaded. | Traffic is routed to a peer that can. **The instance keeps answering.** |

The rule: **a not-ready instance must not be restarted.** Restarting it does not produce a
model; it produces the same instance, a few seconds later, with the same missing artifact —
and meanwhile the decisions it was correctly serving fail-safe are 502s instead.

That is why the Dockerfile `HEALTHCHECK` is wired to `/health/live` and why the Kubernetes
`livenessProbe` is too, with `/health/ready` on the readiness probe alone.

While readiness is red, `POST /v1/decide` still returns **200** with:

```json
{"action": "challenge", "degraded": true,
 "degraded_reason": "DEGRADED_MODEL_UNAVAILABLE", "calibrated_probability": null}
```

It never returns `allow` on that path. `action` is `deny` only for a large basket on an
account we have never seen — the one case where an undetected fraud is most expensive and
we have the least reason to believe the account.

**The known tension.** In Kubernetes, if *every* pod is not-ready the Service has no
endpoints and callers get a connection failure — which converts "degraded but answering"
into "unreachable" at exactly the moment the fail-safe ladder was built for. A model
artifact is a shared dependency, so fleet-wide is the *likely* shape of that failure.
`publishNotReadyAddresses: true` would fix it and is deliberately not set, because it would
also disable endpoint draining on every rollout. The resolution taken is a contract this
repository can state and cannot enforce: **the caller must treat a scoring timeout as a
decline-or-challenge, never as an approve.** Open item.

---

## 5. When it will not come up

| Symptom | Cause | What to do |
|---|---|---|
| `Bind for 0.0.0.0:3000 failed: port is already allocated` | Something else owns the port. | `GRAFANA_PORT=3001 docker compose ... up -d`. Same for `API_PORT`, `POSTGRES_PORT`, `MLFLOW_PORT`, `PROMETHEUS_PORT`. |
| `api` never starts; `migrate` exited non-zero | The migrations did not apply, so `api` is still waiting on `service_completed_successfully`. | `docker compose logs migrate`. A healthy database with no tables is exactly the state in which the api would come up, report itself ready, and fail every ledger write — which is why the dependency is on the migration completing, not on Postgres being healthy. |
| `/health/ready` → 503, `"no scoring bundle at /models/scorer.pkl"` | No artifact. | Expected on a fresh clone. §3. |
| `/health/ready` → 503, `ModelUnavailableError: ... could not be read` | The bundle was pickled by a different scikit-learn than the image carries (pinned at 1.7.2). | Rebuild the bundle with the same environment, or rebuild the image. |
| Every decision is `degraded` with `DEGRADED_SCORING_ERROR` | The request is not carrying the artifact's feature names. | The names live in the bundle; `scripts/load_test.py --bundle <dir>` reads them for exactly this reason. |
| Grafana panels all say "No data" | Prometheus is not scraping, or the datasource uid drifted. | `curl localhost:9090/api/v1/query?query=up{job="fraudlens-api"}`. `tests/deploy/test_stack.py` asserts this end to end. |
| `fraudlens_ledger_write_failures_total` climbing | Postgres is unreachable. Decisions are still correct and still being served. | This is an **auditability** incident, not a request-path one. Restore Postgres; writes resume without restarting the api (`pool_pre_ping`), which the drill asserts. |

---

## 6. The drills

```bash
uv run pytest tests/deploy -q                       # unit + integration + degradation
uv run pytest tests/deploy -m "not integration" -q  # what CI runs; no Docker needed
```

The integration and degradation tests bring the `full` profile up themselves and tear it
down — including volumes — at the end of the session. Running them destroys whatever was in
your local ledger.

The degradation drill removes the model artifact, restarts the container, and asserts:
decisions still served, never `allow`, marked degraded in both the response and the ledger,
readiness red, liveness green, and Docker's own `RestartCount` unchanged. Then it stops
Postgres and asserts that decisions are still served, still **not** degraded (the model is
fine; only the record-keeping broke), readiness still green, and the write failures counted.

---

## 7. Measured numbers

**Latency.** `scripts/load_test.py`, open loop (arrivals on a fixed cadence, each request
timed from its *scheduled* arrival so queueing cannot hide as a reduced rate). 2026-08-05,
10 cores / 11 GB, `api` container capped at 1.0 CPU and 768 MB, real `reference-hgb`
artifact, 205 features per request, ledger writes to Postgres, client and server on the same
host over loopback. 30 s per point, warmed first.

| Offered RPS | Achieved | p50 | p95 | p99 | max | Budget breaches (server-reported) |
|---|---|---|---|---|---|---|
| 10 | 10.0 | 21.1 ms | 26.2 ms | **32.2 ms** | 124.2 ms | 0 / 300 |
| 20 | 20.0 | 17.5 ms | 22.3 ms | **23.6 ms** | 26.5 ms | 0 / 600 |
| 30 | 30.0 | 16.7 ms | 20.8 ms | **24.1 ms** | 30.4 ms | 0 / 900 |
| 40 | 33.1 | 2166.5 ms | 6309.5 ms | **6623.4 ms** | 6729.8 ms | 838 / 1200 |
| 50 | 45.4 | 16.9 ms | 2311.7 ms | **3026.3 ms** | 3172.5 ms | 694 / 1500 |

**The §4.3 claim of p99 ≤ 150 ms holds through 30 RPS and fails hard by 40 RPS on one CPU.**
At the 10 RPS target derived from the blueprint's volume assumption (800k auths/arm/day) the
tail has ~4.7x margin.

Two things in that table are worth more than the headline:

- **There is no warning shoulder.** Between 30 and 40 RPS the p99 moves by a factor of 275
  while throughput barely changes. That is the normal shape of a saturated single-threaded
  server, and it is the argument for the SLO alert being on p99 rather than on error rate or
  utilisation — nothing else in the system moves first.
- **The saturated rows are not reproducible run to run.** 40 RPS measured worse than 50 RPS
  here, and an earlier sweep put the collapse between 30 and 50 with different magnitudes.
  Past saturation the number is a property of scheduling noise, not of the service. Only the
  *location* of the knee is a finding; the magnitudes past it are not.

Caveats that apply to every row, stated because a load test that reports a flattering number
under conditions it does not state is worse than none:

- Loopback, not a network. No TLS, no proxy, no cross-AZ hop.
- Client and server share the host. The container is capped at 1 of 10 CPUs so the client is
  not starving it, but this is not a clean single-tenant measurement.
- The artifact is `reference-hgb`, **not the champion**: a 120k-row, 205-feature
  HistGradientBoosting. Same class of model and same inference shape, different tree count.
- Feature values are random, so the score distribution is not the real one. Latency is
  insensitive to that; the action mix in these runs is not meaningful.
- In-process, with a stub scorer and no socket (`--in-process`), the same generator reports
  p99 7-11 ms from 10 to 200 RPS and saturates at ~220 RPS. That number excludes networking,
  the ledger write and inference, and is reported here only to locate where the time is *not*
  going.

**Where the time actually goes.** Service-side stage histogram, counters reset immediately
before a clean 20 RPS / 40 s run (800 scored decisions, zero breaches):

| Stage | §4.3 budget | Measured mean | Budget breaches |
|---|---|---|---|
| Feature assembly | 60 ms | 0.44 ms | 0 |
| Inference | 25 ms | 2.43 ms | 0 |
| Calibration + policy | 15 ms | 0.54 ms | 0 |
| **Total (in handler)** | 150 ms | **3.66 ms** | 0 |
| Client-observed p99 | | 23.6 ms | |

**The §4.3 split is mis-proportioned, and that is this epic's most useful measurement.**
The service spends 3.66 ms inside the handler and the customer waits ~20 ms; the difference
is deserialisation, strict validation of a 205-key object, uvicorn and loopback — the line
§4.3 calls "overhead 50 ms", which turns out to be the dominant term rather than the
rounding error the allocation implies. Meanwhile feature assembly is allocated 60 ms and
uses 0.44 ms, because there is no feature store in this deployment: "assembly" is two
cost-model calls on a vector the caller already built.

The total budget is not at risk and nothing here is a regression, so the allocation is not
changed by this epic. It is recorded because the per-stage breach counters are wired to
those allocations, and a budget whose lines do not match where the time goes produces alerts
that point at the wrong stage.

**Resources.** 135 MB resident with numpy / pandas / scikit-learn / SQLAlchemy imported and
the app constructed (measured). Hence `requests.memory: 256Mi` (room for the artifact) and
`limits.memory: 768Mi` — tight enough that a leak surfaces in days, loose enough that a
champion pickle does not OOMKill the checkout path. The CPU *limit* is one full core and the
reason is the SLO rather than throughput: CFS throttling quantises at 100 ms, so a limit
below a core can stall a single in-flight decision until the next period — up to 100 ms
against a 150 ms budget, from a limit that looks generous on a utilisation graph.

**Image size.** 984 MB for `[serving,streaming]`; 1.27 GB with `tracking` added. The 286 MB
difference is MLflow's server-side tree, which is why the composition root imports MLflow
lazily and the scoring image does not install it.

---

## 8. Open items

Each of these is a thing this epic found and did not fix, recorded here rather than left for
the next person to rediscover.

1. **No champion artifact exists.** Until a training run persists an estimator, the deployed
   stack runs either degraded or `reference-*`. `scripts/build_model_bundle.py` is a fixture,
   not a fix.
2. **The request contract cannot express a missing feature.** `DecideRequest.features` is
   `dict[str, float]` and JSON has no NaN. IEEE-CIS is mostly NaN and
   `HistGradientBoostingClassifier` consumes NaN natively as "missing" — better than any
   imputation, because the model learned a split for it. The wire format loses that signal
   and forces the caller to impute. `dict[str, float | None]` would close it; that is a
   change to `serving.contracts`, which this epic does not own.
3. **No OpenTelemetry collector in the stack.** The tracing instrumentation exists and is
   inert without an injected provider, and wiring an OTLP exporter needs
   `opentelemetry-exporter-otlp`, which is not a declared dependency. §5's "end-to-end trace
   spans" is therefore instrumented but not collected.
4. **Fleet-wide not-ready black-holes the Kubernetes Service.** §4.
5. **`pandas` and `pyarrow` are base dependencies**, so the scoring container carries them
   although the request path never reads a parquet file. Moving them into an extra is a
   pyproject change with consequences for the research and streaming units.
6. **Gap 8 is half closed.** The MLflow server exists and is reachable; the model card still
   has to be regenerated against it before its run pointer resolves for anyone but its
   author.
