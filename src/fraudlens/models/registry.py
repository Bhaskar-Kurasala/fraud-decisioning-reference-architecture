"""Model registry: versions, stages, and the record of why each stage changed.

The registry's job in an audited system is not storage — it is answering "which
model decided this transaction, and who decided that model should be deciding".
Every stage transition therefore carries an actor, a timestamp and a
justification, and *rejections are written too*. A registry that only records
promotions cannot show that a bad model was considered and blocked, which is the
evidence a model-risk review actually asks for.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from fraudlens.models.gate import GateDecision

# Model-version tag keys. Namespaced so they cannot collide with MLflow's own.
_TAG_DECISION = "fraudlens.promotion.decision"
_TAG_JUSTIFICATION = "fraudlens.promotion.justification"
_TAG_ACTOR = "fraudlens.promotion.actor"
_TAG_AT = "fraudlens.promotion.decided_at"
_TAG_CHECKS = "fraudlens.promotion.checks"


class Stage(str, Enum):
    """MLflow's stage vocabulary, restated so callers cannot pass a typo.

    MLflow has deprecated stages in favour of free-form aliases and emits a
    FutureWarning on every transition. Stages are kept anyway: the design spec
    specifies this exact four-state lifecycle, and an ordered, closed vocabulary
    is what makes "which model was in Production on 3 June" answerable. If MLflow
    removes them, the migration is to write the same four values as aliases —
    contained entirely within this module, which is why the enum exists rather
    than raw strings at the call sites.
    """

    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    """What was written to the registry, returned for logging by the caller."""

    model_name: str
    version: int
    promoted: bool
    to_stage: Stage
    actor: str
    decided_at: datetime
    justification: str


def register_model(
    client: Any,
    name: str,
    source: str,
    run_id: str,
    tags: dict[str, str] | None = None,
) -> int:
    """Create the registered model if needed and add a version. Returns version.

    Existence is probed by search rather than by catching a not-found exception,
    because the exception type differs per backing store and a broad `except`
    here would swallow a genuine connectivity failure as "does not exist" — and
    then create a second, empty registered model beside the real one.
    """
    if not client.search_registered_models(f"name='{name}'"):
        client.create_registered_model(name)
    version = client.create_model_version(name=name, source=source, run_id=run_id, tags=tags)
    return int(version.version)


def current_champion(client: Any, name: str) -> Any | None:
    """The version currently in Production, or None.

    Returns None rather than raising when nothing is in Production: the first
    promotion into an empty registry is a normal state, not an error, and the
    caller has to handle it either way.
    """
    versions = client.search_model_versions(f"name='{name}'")
    production = [v for v in versions if v.current_stage == Stage.PRODUCTION.value]
    if not production:
        return None
    # Highest version wins if a manual transition ever leaves two in Production;
    # surfacing the newest is the safer default than picking arbitrarily.
    return max(production, key=lambda v: int(v.version))


def transition_stage(
    client: Any,
    name: str,
    version: int,
    stage: Stage,
    justification: str,
    actor: str,
    now: datetime,
    archive_existing: bool = False,
) -> None:
    """Move a version to a stage, recording who decided and why."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    if not justification.strip():
        raise ValueError("a stage transition without a justification is not auditable")
    client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage=stage.value,
        archive_existing_versions=archive_existing,
    )
    client.set_model_version_tag(name, str(version), _TAG_JUSTIFICATION, justification)
    client.set_model_version_tag(name, str(version), _TAG_ACTOR, actor)
    client.set_model_version_tag(name, str(version), _TAG_AT, now.isoformat())


def apply_gate_decision(
    client: Any,
    name: str,
    version: int,
    decision: GateDecision,
    actor: str,
    now: datetime,
) -> PromotionRecord:
    """Write the gate's verdict to the registry and act on it.

    A blocked challenger is tagged and left where it is, not deleted and not
    silently skipped. The failed checks stay attached to the version so the next
    person to propose the same model sees why it was refused.
    """
    checks = [
        {"name": c.name, "status": c.status.value, "reason": c.reason} for c in decision.checks
    ]
    client.set_model_version_tag(
        name, str(version), _TAG_DECISION, "promote" if decision.promote else "blocked"
    )
    client.set_model_version_tag(name, str(version), _TAG_CHECKS, json.dumps(checks))

    justification = decision.summary()
    target = Stage.PRODUCTION if decision.promote else Stage.STAGING
    transition_stage(
        client,
        name,
        version,
        target,
        justification=justification,
        actor=actor,
        now=now,
        # Only a promotion archives the incumbent. A blocked challenger must not
        # disturb the model currently taking decisions.
        archive_existing=decision.promote,
    )
    return PromotionRecord(
        model_name=name,
        version=version,
        promoted=decision.promote,
        to_stage=target,
        actor=actor,
        decided_at=now,
        justification=justification,
    )
