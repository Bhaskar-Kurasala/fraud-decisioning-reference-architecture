"""Lineage: what produced a number, what produced a decision, and where that chain breaks.

Three questions, three modules. `manifest` answers "what would I need to regenerate
this artifact"; `model_card` answers "what is this model, measured how, and what is it
known to get wrong"; `replay` answers "given the ledger row, does the recorded action
still fall out of the recorded versions".

This layer reads the ledger and the tracking store; nothing on the request path may read
it. The import contract places it below `serving` for that reason — a replay or a card
render cannot end up inside the 150 ms budget by accident, because the import would fail
the layer check. The cost of that placement is stated rather than hidden: the serving
fail-safe ladder lives above this layer, so `replay` cannot reconstruct a degraded
decision. See docs/lineage.md, gap 2.

Deliberately empty of re-exports, matching `fraudlens.models`: a package `__init__` that
mirrors every public name is a forwarding layer with no behaviour of its own (§9a).
"""
