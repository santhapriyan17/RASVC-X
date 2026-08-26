"""Validation result and evidence relationship contracts.

Validation must preserve uncertainty: a ValidationResult carries an explicit
label plus a confidence score, and NLI failure/timeout must map to
UNCERTAIN -- never to SUPPORTED. Label and confidence are never collapsed
into a boolean.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rasvcx.schemas.common import CandidateId, ClaimId, EvidenceRelationship, NLILabel


class ValidationLabel(str, Enum):
    """Outcome of validating a claim against evidence (or evidence pair)."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PARTIAL = "partial"
    CONTRADICTION = "contradiction"
    UNCERTAIN = "uncertain"


class ValidationStage(str, Enum):
    """Which validation stage produced this result."""

    DETERMINISTIC = "deterministic"
    CONTEXTUAL = "contextual"
    SELECTIVE_NLI = "selective_nli"


@dataclass(frozen=True, slots=True)
class NLISignal:
    """Raw NLI model output for one candidate, kept distinct from the final
    ValidationLabel. Optional on a ValidationResult: not every result comes
    from an NLI call (deterministic/contextual stages may produce results
    without invoking NLI at all).
    """

    label: NLILabel
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"NLISignal.confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of validating one CandidatePair (or claim) at one stage.

    confidence is always present and is never discarded or reduced to a
    boolean, even when label is UNCERTAIN. On NLI failure/timeout, callers
    must construct this with label=ValidationLabel.UNCERTAIN and a
    confidence reflecting the failure (typically 0.0) -- constructing
    SUPPORTED on failure is a contract violation left to caller discipline
    and covered by tests, not enforceable purely at the type level.
    """

    candidate_id: CandidateId
    stage: ValidationStage
    label: ValidationLabel
    confidence: float
    nli_signal: NLISignal | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"ValidationResult.confidence must be in [0, 1], got {self.confidence}")
        if self.stage is ValidationStage.SELECTIVE_NLI and self.nli_signal is None:
            raise ValueError(
                "ValidationResult.stage == SELECTIVE_NLI requires an nli_signal"
            )


@dataclass(frozen=True, slots=True)
class EvidenceRelationshipResult:
    """Final resolved relationship for a candidate pair.

    relationship uses the fixed taxonomy (EvidenceRelationship), not raw
    NLI labels. contributing_claim_ids records which claims informed the
    resolution, by ID only.
    """

    candidate_id: CandidateId
    relationship: EvidenceRelationship
    confidence: float
    contributing_claim_ids: frozenset[ClaimId]
    rationale: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"EvidenceRelationshipResult.confidence must be in [0, 1], got {self.confidence}"
            )