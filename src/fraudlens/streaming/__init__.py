"""Streaming layer: replay the snapshot, record every decision, delay the truth.

Three pieces, one purpose — turn assumptions about label latency and model
decay into measurements. The ledger accepts decisions rather than computing
them, so this package does not import the model or policy layers.
"""

from fraudlens.streaming.labels import (
    DisputeLagModel,
    EffectiveLabel,
    HumanAdjudication,
    LabelOrigin,
    LabelRevealer,
    PendingLabel,
    RevealedLabel,
    effective_labels,
    reconcile,
    record_adjudication,
)
from fraudlens.streaming.ledger import DecisionLedger, DecisionRecord
from fraudlens.streaming.migrate import migrate
from fraudlens.streaming.replay import ReplayCheckpoint, ReplayEvent, ReplayProducer

__all__ = [
    "DecisionLedger",
    "DecisionRecord",
    "DisputeLagModel",
    "EffectiveLabel",
    "HumanAdjudication",
    "LabelOrigin",
    "LabelRevealer",
    "PendingLabel",
    "ReplayCheckpoint",
    "ReplayEvent",
    "ReplayProducer",
    "RevealedLabel",
    "effective_labels",
    "migrate",
    "reconcile",
    "record_adjudication",
]
