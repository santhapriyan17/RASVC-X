"""Decision contract: final pipeline action and corrective routing target.

Owned exclusively by decision/engine.py. This module only defines the
contract; it contains no decision logic itself (no thresholds, no business
rules) -- see architecture rule: orchestrator/schemas contain no business
decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rasvcx.schemas.common import CandidateId


class DecisionAction(str, Enum):
    """Final action selected by the decision engine for this request."""

    ANSWER = "answer"
    WARNING = "warning"
    REPAIR = "repair"
    REGENERATE = "regenerate"
    ABSTAIN = "abstain"


class CorrectiveTarget(str, Enum):
    """Where a REPAIR or REGENERATE action re-enters the pipeline.

    REPAIR always targets GENERATION (generation-level correction).
    REGENERATE targets VERIFIED_CONTEXT or RETRIEVAL depending on the
    computed root cause; the decision engine computes this, the orchestrator
    only executes it.
    """

    GENERATION = "generation"
    VERIFIED_CONTEXT = "verified_context"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True, slots=True)
class Decision:
    """Immutable output of the decision engine for one pipeline pass.

    corrective_target is required when action is REPAIR or REGENERATE and
    must be None otherwise -- enforced here so the orchestrator can execute
    decision.corrective_target without re-deriving or validating the
    action/target relationship itself (see orchestrator rule: composition
    and sequencing only, no business decisions).
    """

    action: DecisionAction
    confidence: float
    corrective_target: CorrectiveTarget | None = None
    contributing_candidate_ids: frozenset[CandidateId] = frozenset()
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Decision.confidence must be in [0, 1], got {self.confidence}")

        requires_target = self.action in (DecisionAction.REPAIR, DecisionAction.REGENERATE)
        if requires_target and self.corrective_target is None:
            raise ValueError(
                f"Decision.action == {self.action.value!r} requires a corrective_target"
            )
        if not requires_target and self.corrective_target is not None:
            raise ValueError(
                f"Decision.action == {self.action.value!r} must not set corrective_target"
            )
        if self.action == DecisionAction.REPAIR and self.corrective_target != CorrectiveTarget.GENERATION:
            raise ValueError("REPAIR must target CorrectiveTarget.GENERATION")
        if self.action == DecisionAction.REGENERATE and self.corrective_target not in (
            CorrectiveTarget.VERIFIED_CONTEXT,
            CorrectiveTarget.RETRIEVAL,
        ):
            raise ValueError(
                "REGENERATE must target CorrectiveTarget.VERIFIED_CONTEXT or CorrectiveTarget.RETRIEVAL"
            )