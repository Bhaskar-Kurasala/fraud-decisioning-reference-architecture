"""Drift, calibration and label-maturity monitoring — the answer to "is it still working?".

The layer exists because of one constraint: chargebacks mature over 30-90 days, so the
question this package is named after is unanswerable in real time. Everything here is
therefore split by *how long you have to wait to know it*, not by subject:

- `drift` and `baseline` need no labels at all and answer today (Tier 0).
- `calibration` needs labels and answers in a quarter (Tier 2).
- `maturity` is the gate between the two, and it refuses rather than warns.

Nothing on the request path may import this package (see the import-linter contract in
`pyproject.toml`): a PSI computation reads a whole window and would eat the 150 ms
decision budget whole. The contract makes that a build failure rather than an incident.

Deliberately empty of re-exports, matching `fraudlens.models`: a forwarding `__init__`
hides which module a symbol lives in, which is the first thing a reader needs.
"""
