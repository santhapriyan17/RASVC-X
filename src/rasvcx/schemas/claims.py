"""Claim schema: single source of truth for extracted/atomic claims.

Claim objects are stored once (keyed by ClaimId) inside EvidenceBundle.claims
and referenced everywhere else by ID only. This module must not import from
rasvcx.schemas.evidence to avoid a circular dependency (evidence references
claims by ID, not by object).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet

from rasvcx.schemas.common import ChunkId, ClaimId, Unknown


class ClaimSource(str, Enum):
    """Which pipeline stage produced this claim."""

    EVIDENCE_EXTRACTION = "evidence_extraction"
    POST_GENERATION = "post_generation"


class ClaimType(str, Enum):
    """Coarse structural category of a claim, used for routing/validation."""

    FACTUAL = "factual"
    NUMERIC = "numeric"
    DOSAGE = "dosage"
    TEMPORAL = "temporal"
    RECOMMENDATION = "recommendation"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Claim:
    """An atomic, independently-verifiable statement.

    Immutable once constructed. Claims are never mutated in place; if a
    claim needs to change, a new Claim with a new ClaimId is created and
    the old one remains for audit purposes.
    """

    claim_id: ClaimId
    text: str
    source: ClaimSource
    claim_type: ClaimType

    # Chunk this claim was extracted from, when source is EVIDENCE_EXTRACTION.
    # UNKNOWN when source is POST_GENERATION (claim came from the generated
    # answer, not a single retrieval chunk).
    origin_chunk_id: ChunkId | Unknown

    # Normalized/canonicalized text used for downstream matching and NLI
    # premise/hypothesis construction. Populated by claims/normalizer.py.
    normalized_text: str | None = None

    # Numeric/unit-sensitive content requires it to be forced into the
    # safety floor path (see routing/risk safety floor rule). This flag is
    # set by claims/extractor.py using deterministic pattern rules, never by
    # an LLM judgment call.
    is_safety_critical: bool = False

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("Claim.claim_id must be non-empty")
        if not self.text:
            raise ValueError("Claim.text must be non-empty")


@dataclass(frozen=True, slots=True)
class ClaimRegistry:
    """Request-scoped, append-only mapping of ClaimId -> Claim.

    This is the enforcement point for "claims are stored once": registering
    a claim_id that already exists with different content is a hard error,
    preventing accidental duplication of claim objects across pipeline
    stages. Registering the identical Claim twice (idempotent re-registration)
    is allowed and is a no-op.
    """

    _claims: dict[ClaimId, Claim] = field(default_factory=dict)

    def register(self, claim: Claim) -> None:
        existing = self._claims.get(claim.claim_id)
        if existing is not None and existing != claim:
            raise ValueError(
                f"Claim {claim.claim_id!r} already registered with different "
                f"content; claims must be stored once and referenced by ID"
            )
        self._claims[claim.claim_id] = claim

    def get(self, claim_id: ClaimId) -> Claim:
        try:
            return self._claims[claim_id]
        except KeyError as exc:
            raise KeyError(f"Unknown claim_id: {claim_id!r}") from exc

    def __contains__(self, claim_id: ClaimId) -> bool:
        return claim_id in self._claims

    def __len__(self) -> int:
        return len(self._claims)

    def ids(self) -> FrozenSet[ClaimId]:
        return frozenset(self._claims.keys())

    def as_dict(self) -> dict[ClaimId, Claim]:
        """Read-only snapshot for serialization; not for mutation."""
        return dict(self._claims)