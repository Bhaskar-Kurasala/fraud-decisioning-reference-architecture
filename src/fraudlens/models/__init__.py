"""Model evaluation, experiment tracking, and the promotion gate.

This layer sits below `economics` and `policy` in the dependency order, so it
cannot import them. That constraint is load-bearing rather than incidental: the
gate consumes *per-transaction realised cost arrays* supplied by the caller,
which is what makes it testable without a policy simulator and reusable for any
cost definition the business later adopts.

Deliberately empty of re-exports. A package `__init__` that mirrors every public
name is a forwarding layer with no behaviour of its own (§9a), and it hides which
module a symbol actually lives in — which is the first thing a reader needs.
"""
