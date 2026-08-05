#!/usr/bin/env python
"""Open-loop load generator for `POST /v1/decide`, to test the §4.3 claim of p99 <= 150 ms.

**Open loop on purpose.** The obvious implementation — N workers each sending the next
request as soon as the last one returns — is closed loop, and closed loop cannot measure a
latency SLO. When the service slows down a closed-loop client sends fewer requests, so the
queue never builds and the measured p99 is the p99 of a system that was never actually
offered the load. That is coordinated omission, and it is why load tests routinely report
a passing tail on a service that browns out in production. Here arrivals are scheduled
against a fixed wall-clock cadence and a request that starts late is *timed from when it
should have started*, so queueing shows up in the number instead of hiding in the rate.

**What "target RPS" is.** The spec does not state one, and the dataset cannot supply it:
92,427 transactions over a 32-day test window is 0.03 RPS, which would make any latency
claim meaningless. The blueprint's power calculation (quoted in `models.sequential`)
assumes 800k auths/arm/day, so 10 RPS is the defensible steady-state figure and is the
default here. The interesting question is not whether 10 RPS passes but where it stops
passing, which is what `--rps` sweeping is for.

    uv run python scripts/load_test.py --url http://localhost:8000 --rps 10 --duration 30
    uv run python scripts/load_test.py --in-process --rps 200 --duration 20

`--in-process` drives the ASGI app directly with a stub scorer: no sockets, no model. It
measures the decision path and nothing else, and its numbers must be labelled as such —
they exclude kernel networking, TLS, the ledger write and real inference, all of which are
in the end-to-end budget the SLO is written against.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import pickle
import random
import statistics
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

# A fixed body shape, with amount and tenure varied. Not a single constant request: the
# break-even boundary moves with amount and tenure, so a constant body would exercise one
# branch of the policy and one cost arm, and the p99 of a single branch is not the p99 of
# the service.
#
# The default 50 names are a placeholder for `--in-process`, where the stub scorer ignores
# them. Against a real service they must be the artifact's own feature names, or every
# request takes the fail-safe path — which is fast for the wrong reason and would report a
# flattering p99 for a service that scored nothing. `--bundle` exists to make that mistake
# hard to make by accident.
_FEATURE_NAMES = tuple(f"V{i}" for i in range(1, 51))


def _feature_names(bundle: str | None) -> tuple[str, ...]:
    if bundle is None:
        return _FEATURE_NAMES
    path = Path(bundle)
    path = path / "scorer.pkl" if path.is_dir() else path
    with path.open("rb") as handle:
        return tuple(pickle.load(handle)["feature_names"])  # noqa: S301


def _body(rng: random.Random, sequence: int, names: tuple[str, ...]) -> dict[str, Any]:
    return {
        "transaction_id": 900_000_000 + sequence,
        "transaction_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "amount": round(rng.lognormvariate(4.0, 1.0), 2),
        "days_since_first_seen": rng.choice([None, 0.0, 3.0, 45.0, 500.0]),
        "features": {name: rng.random() for name in names},
    }


class Percentiles:
    """p50/p95/p99 by nearest-rank on the sorted sample.

    Nearest-rank rather than an interpolating estimator: at a few thousand samples the
    difference is under a millisecond, and a percentile that is a real observation can be
    pointed at in a trace. Interpolation invents a latency nobody experienced.
    """

    def __init__(self, samples_ms: list[float]) -> None:
        self._sorted = sorted(samples_ms)

    def at(self, quantile: float) -> float:
        if not self._sorted:
            return float("nan")
        index = min(len(self._sorted) - 1, int(quantile * len(self._sorted)))
        return self._sorted[index]

    def summary(self) -> dict[str, float]:
        return {
            "count": float(len(self._sorted)),
            "p50_ms": self.at(0.50),
            "p95_ms": self.at(0.95),
            "p99_ms": self.at(0.99),
            "max_ms": self._sorted[-1] if self._sorted else float("nan"),
            "mean_ms": statistics.fmean(self._sorted) if self._sorted else float("nan"),
        }


async def _drive(
    client: httpx.AsyncClient,
    url: str,
    rps: float,
    duration: float,
    seed: int,
    names: tuple[str, ...],
) -> dict[str, Any]:
    rng = random.Random(seed)  # noqa: S311 — traffic shape, not a security decision
    interval = 1.0 / rps
    latencies: list[float] = []
    statuses: dict[int, int] = {}
    degraded = 0
    breached = 0
    timeouts = 0
    started = time.perf_counter()
    tasks: list[asyncio.Task[None]] = []

    async def one(scheduled_at: float, sequence: int) -> None:
        nonlocal degraded, breached, timeouts
        payload = _body(rng, sequence, names)
        try:
            response = await client.post(f"{url}/v1/decide", json=payload)
        except httpx.HTTPError:
            # A timed-out request is the most important sample in a saturation run and the
            # easiest one to lose: letting the exception escape would abort the whole run and
            # report nothing, and discarding it would compute a percentile over only the
            # requests that succeeded — which is exactly the flattering number this generator
            # exists to avoid. It is recorded at its full observed wait.
            timeouts += 1
            latencies.append((time.perf_counter() - scheduled_at) * 1000.0)
            return
        # Measured from the scheduled arrival, not from the send. This is the whole
        # correction for coordinated omission: if the client itself was late because the
        # event loop was saturated, the customer waited for that too.
        latencies.append((time.perf_counter() - scheduled_at) * 1000.0)
        statuses[response.status_code] = statuses.get(response.status_code, 0) + 1
        if response.status_code == 200:
            body = response.json()
            degraded += int(body["degraded"])
            breached += int(body["latency"]["budget_breached"])

    sequence = 0
    while True:
        scheduled_at = started + sequence * interval
        if scheduled_at - started >= duration:
            break
        delay = scheduled_at - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(scheduled_at, sequence)))
        sequence += 1
    await asyncio.gather(*tasks)

    wall = time.perf_counter() - started
    return {
        "offered_rps": rps,
        "achieved_rps": len(latencies) / wall if wall else 0.0,
        "duration_s": wall,
        "statuses": statuses,
        "degraded_responses": degraded,
        "client_timeouts": timeouts,
        "self_reported_budget_breaches": breached,
        **Percentiles(latencies).summary(),
    }


def _in_process_client() -> tuple[httpx.AsyncClient, Any]:
    """The app with a stub scorer, driven over ASGI with no socket.

    The stub returns a constant probability in constant time, so what this measures is
    validation, the cost model, the policy argmax, the metrics and the response — every
    part of the request path except inference. Inference is the 25 ms line of the §4.3
    budget and is therefore the largest thing these numbers omit.
    """
    from fraudlens.serving.app import create_app
    from fraudlens.serving.runtime import CalibratedScorer

    class _Stub:
        version = "load-test-stub"

        def score(self, features: Mapping[str, float]) -> float:
            return 0.31

        def calibrate(self, raw_score: float) -> float:
            return 0.31

    def loader(pin: str | None) -> CalibratedScorer:
        return _Stub()

    app = create_app(loader=loader)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://in-process"
    )
    return client, app


async def _main(args: argparse.Namespace) -> int:
    names = _feature_names(args.bundle)
    if args.in_process:
        client, app = _in_process_client()
        # The ASGI transport does not run the lifespan, so `ServiceState.load` never fires
        # and the app would serve degraded. Run it explicitly rather than accepting a
        # measurement of the fail-safe path labelled as a measurement of the scored one.
        async with client, app.router.lifespan_context(app):
            result = await _drive(client, "", args.rps, args.duration, args.seed, names)
    else:
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await _drive(
                client, args.url.rstrip("/"), args.rps, args.duration, args.seed, names
            )

    result["mode"] = "in-process ASGI, stub scorer" if args.in_process else f"HTTP {args.url}"
    result["features_per_request"] = len(names)
    print(json.dumps(result, indent=2, sort_keys=True))
    # Exit non-zero on a breach so this is usable as a gate, not only as a report.
    return 0 if result["p99_ms"] <= args.budget_ms else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument("--rps", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--budget-ms", type=float, default=150.0)
    parser.add_argument(
        "--bundle",
        default=None,
        help="path to the deployed scorer bundle; requests are shaped from its feature names",
    )
    return asyncio.run(_main(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
