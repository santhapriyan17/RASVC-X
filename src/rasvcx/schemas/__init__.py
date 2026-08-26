"""RASVC-X core schemas: canonical typed data contracts.

Re-exports the public surface of each schema module so downstream packages
can do e.g. ``from rasvcx.schemas import EvidenceBundle, Claim, Decision``
instead of reaching into individual files.
"""

from __future__ import annotations

from rasvcx.schemas.claims import Claim, ClaimRegistry, ClaimSource, ClaimType
from rasvcx.schemas.common import (
    UNKNOWN,
    CandidateId,
    ChunkId,
    ClaimId,
    EvidenceItemId,
    EvidenceRelationship,
    NLILabel,
    PipelineStage,
    QueryId,
    SourceType,
    Unknown,
)
from rasvcx.schemas.confidence import ConfidenceFeatures, ConfidenceScore
from rasvcx.schemas.decision import CorrectiveTarget, Decision, DecisionAction
from rasvcx.schemas.evidence import (
    BundleMetadata,
    CandidatePair,
    ConflictLabel,
    EvidenceBundle,
    EvidenceItem,
    Provenance,
)
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.schemas.validation import (
    EvidenceRelationshipResult,
    NLISignal,
    ValidationLabel,
    ValidationResult,
    ValidationStage,
)

__all__ = [
    "UNKNOWN",
    "Unknown",
    "QueryId",
    "ChunkId",
    "ClaimId",
    "CandidateId",
    "EvidenceItemId",
    "SourceType",
    "NLILabel",
    "EvidenceRelationship",
    "PipelineStage",
    "Claim",
    "ClaimRegistry",
    "ClaimSource",
    "ClaimType",
    "Provenance",
    "EvidenceItem",
    "ConflictLabel",
    "CandidatePair",
    "BundleMetadata",
    "EvidenceBundle",
    "ValidationDepth",
    "RiskFeatureScores",
    "RiskProfile",
    "QueryRequest",
    "ValidationLabel",
    "ValidationStage",
    "NLISignal",
    "ValidationResult",
    "EvidenceRelationshipResult",
    "CorrectiveTarget",
    "DecisionAction",
    "Decision",
    "ConfidenceFeatures",
    "ConfidenceScore",
]