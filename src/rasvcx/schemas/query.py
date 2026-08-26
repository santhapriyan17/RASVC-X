"""Query request contract and RiskProfile.

RiskProfile is created once by routing/risk_router.py and is read-only
thereafter. It allocates validation effort (budgets/eligibility); it does
not itself decide ANSWER/WARNING/REPAIR/REGENERATE/ABSTAIN -- that belongs
to decision/engine.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from rasvcx.schemas.common import QueryId


class ValidationDepth(str, Enum):
    """Advisory validation-effort tier computed by the risk router.

    Advisory only: it optimizes computational cost. It can never cause a
    query to bypass the safety floor (deterministic validation always runs;
    safety-critical patterns always force at least contextual validation
    and NLI-eligibility, regardless of this tier).
    """

    SHALLOW = "shallow"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass(frozen=True, slots=True)
class RiskFeatureScores:
    """Individual feature scores that fed into the overall risk score.

    Kept as a small, explicit structure (not an untyped dict) so that the
    contributing signals are inspectable and testable in isolation.
    """

    numeric_content_score: float = 0.0
    unit_sensitive_score: float = 0.0
    safety_critical_structure_score: float = 0.0
    query_complexity_score: float = 0.0
    ambiguity_score: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("numeric_content_score", self.numeric_content_score),
            ("unit_sensitive_score", self.unit_sensitive_score),
            ("safety_critical_structure_score", self.safety_critical_structure_score),
            ("query_complexity_score", self.query_complexity_score),
            ("ambiguity_score", self.ambiguity_score),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"RiskFeatureScores.{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class RiskProfile:
    """Read-only, request-scoped output of the risk/complexity router.

    Immutable after construction (frozen dataclass): once the router
    produces a RiskProfile, no downstream stage may mutate it. Downstream
    stages that need a different allocation must not construct a new
    RiskProfile competing with this one for the same request -- there is
    exactly one RiskProfile per EvidenceBundle.
    """

    overall_risk_score: float
    feature_scores: RiskFeatureScores
    validation_depth: ValidationDepth

    # Advisory allocation -- consumed by downstream bounded-loop logic, never
    # authoritative over the global corrective attempt budget itself.
    retrieval_retry_budget: int
    nli_call_allowance: int

    # Forced by the safety floor (see validation/selective_router.py) when
    # safety-critical deterministic patterns are detected in the query or
    # its candidate evidence, independent of validation_depth.
    safety_floor_forced: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.overall_risk_score <= 1.0:
            raise ValueError(
                f"overall_risk_score must be in [0, 1], got {self.overall_risk_score}"
            )
        if self.retrieval_retry_budget < 0:
            raise ValueError("retrieval_retry_budget must be non-negative")
        if self.nli_call_allowance < 0:
            raise ValueError("nli_call_allowance must be non-negative")


@dataclass(frozen=True, slots=True)
class QueryRequest:
    """Normalized, validated incoming query.

    Constructed after input validation and query normalization; this is the
    contract those two stages produce and everything downstream consumes.
    """

    query_id: QueryId
    raw_text: str
    normalized_text: str
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw_text.strip():
            raise ValueError("QueryRequest.raw_text must be non-empty")
        if not self.normalized_text.strip():
            raise ValueError("QueryRequest.normalized_text must be non-empty")