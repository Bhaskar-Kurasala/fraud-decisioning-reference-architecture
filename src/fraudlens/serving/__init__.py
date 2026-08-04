"""The scoring API: the only component of this system on the checkout path.

Responsibility (design spec §4.1): economic features -> model score -> calibration ->
policy -> action + reason codes, synchronously, within the p99 150 ms budget of §4.3.

It computes no economics and defines no policy. `fraudlens.economics` prices the
outcomes and `fraudlens.policy` chooses the action; this layer assembles their inputs,
times each stage, and renders the result as an HTTP contract. Two corrections have
already been needed on this project because a caller re-derived the policy from its
description instead of importing it, so the rule is explicit: nothing here recomputes a
number that a lower layer already computes.

Deliberately empty of re-exports (§9a) -- a package `__init__` that mirrors every public
name is a forwarding layer with no behaviour, and it hides where a symbol actually lives.
"""
