"""Shared primitives for RASVC-X schema contracts.

This module has no intra-package dependencies. Every other module under
``rasvcx.schemas`` may depend on it; it must never import from them.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Literal, NewType

# ---------------------------------------------------------------------------
# ID type aliases
#
# These are structural aliases (plain ``str`` at runtime) used so that
# ID-keyed lookups (e.g. ``Dict[ClaimId, Claim]``) are self-documenting and
# statically distinguishable from arbitrary strings, without introducing a
# wrapper class or any runtime overhead.
# ---------------------------------------------------------------------------

QueryId = NewType("QueryId", str)
ChunkId = NewType("ChunkId", str)
ClaimId = NewType("ClaimId", str)
CandidateId = NewType("CandidateId", str)
EvidenceItemId = NewType("EvidenceItemId", str)


# ---------------------------------------------------------------------------
# UNKNOWN semantics (see architecture spec, section 10)
#
# Missing provenance/context fields (date, jurisdiction, population,
# dosage context) must never be fabricated. They must be represented as
# UNKNOWN, and UNKNOWN must never be treated as "compatible" -- it means
# "insufficient information to rule out conflict."
#
# A dedicated sentinel class (rather than ``None`` or a bare string) is used
# so that:
#   1. ``UNKNOWN`` cannot be confused with "not yet set" (``None`` reserved
#      for that, if ever needed) or with a legitimate empty string.
#   2. Equality/identity checks are explicit and type-checked.
#   3. The conservative meaning is documented at the type level, not by
#      convention.
# ---------------------------------------------------------------------------


class _UnknownType:
    """Sentinel type for conservatively-missing provenance/context data.

    There is exactly one instance of this class: :data:`UNKNOWN`. It is not
    falsy-by-convention (unlike ``None`` or ``""``) so that accidental
    truthiness checks do not silently treat missing context as absent
    information versus present-but-empty information.
    """

    __slots__ = ()
    _instance: "_UnknownType | None" = None

    def __new__(cls) -> "_UnknownType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        # Explicit: UNKNOWN is not "falsy absence", it is a distinct state.
        # Callers must check identity/equality against UNKNOWN rather than
        # relying on truthiness.
        return True

    def __reduce__(self) -> "tuple[type[_UnknownType], tuple[()]]":
        # Ensures pickling/copying returns the singleton rather than a
        # second instance.
        return (_UnknownType, ())


UNKNOWN: Final[_UnknownType] = _UnknownType()
"""Singleton sentinel meaning: insufficient information to rule out conflict."""

Unknown = _UnknownType
"""Type alias for use in type hints, e.g. ``date: str | Unknown``."""


# ---------------------------------------------------------------------------
# Finite-state enums
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    """Provenance source category for a retrieved evidence chunk."""

    CLINICAL_GUIDELINE = "clinical_guideline"
    REGULATORY_DOCUMENT = "regulatory_document"
    PEER_REVIEWED_LITERATURE = "peer_reviewed_literature"
    DRUG_LABEL = "drug_label"
    INSTITUTIONAL_POLICY = "institutional_policy"
    OTHER = "other"


class NLILabel(str, Enum):
    """Raw natural language inference output label.

    These are signals consumed during evidence resolution; they are
    deliberately NOT the final evidence relationship taxonomy (see
    :class:`EvidenceRelationship`). An entailment/contradiction label alone
    does not indicate *why* two evidence items agree or disagree (e.g.
    differing population vs. genuine conflict).
    """

    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    NEUTRAL = "neutral"


class EvidenceRelationship(str, Enum):
    """Final, resolved relationship between two pieces of evidence.

    Produced by evidence resolution after combining deterministic checks,
    contextual validation, and (where eligible) selective NLI signals.
    """

    COMPATIBLE = "compatible"
    POPULATION_DIFF = "population-diff"
    TEMPORAL_DIFF = "temporal-diff"
    JURISDICTION_DIFF = "jurisdiction-diff"
    DOSAGE_DIFF = "dosage-diff"
    GENUINE_CONFLICT = "genuine-conflict"
    UNRESOLVED = "unresolved"


# ---------------------------------------------------------------------------
# Misc shared literals
# ---------------------------------------------------------------------------

PipelineStage = Literal[
    "input_validation",
    "query_normalization",
    "risk_routing",
    "hybrid_retrieval",
    "reranking",
    "sufficiency_gate",
    "targeted_retrieval",
    "provenance_context",
    "claim_representation",
    "deterministic_validation",
    "candidate_generation",
    "contextual_validation",
    "selective_nli",
    "evidence_resolution",
    "verified_context",
    "generation",
    "atomic_claim_extraction",
    "post_generation_verification",
    "confidence_estimation",
    "calibration",
    "decision",
]
"""Named pipeline stages, used as keys for per-stage elapsed-time metadata."""