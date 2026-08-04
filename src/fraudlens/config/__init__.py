"""Typed business configuration. Bottom of the dependency stack: imports nothing else."""

from fraudlens.config.settings import (
    SETTINGS,
    TENURE_EDGES,
    TENURE_LABELS,
    UNKNOWN_TENURE,
    BusinessConstants,
)

__all__ = [
    "SETTINGS",
    "TENURE_EDGES",
    "TENURE_LABELS",
    "UNKNOWN_TENURE",
    "BusinessConstants",
]
