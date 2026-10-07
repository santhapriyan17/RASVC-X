"""Internal (non-schema) support types for Module 8 -- Validation.

These are configuration/DI/intermediate types used *inside* the validation
package. They are distinct from the frozen public contracts in
``rasvcx.schemas.validation`` (``ValidationResult``, ``NLISignal``,
``EvidenceRelationshipResult``), which this module never redefines or
shadows.

Design constraints:
  - Pure data holders / configuration -- no I/O, no ML.
  - Immutable (frozen dataclasses) where the value is fixed at construction.
  - Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rasvcx.schemas.claims import ClaimType
from rasvcx.schemas.common import ClaimId, EvidenceItemId


@dataclass(frozen=True, slots=True)
class DeterministicConfig:
    """Tunable thresholds for the deterministic validator.

    Kept separate from ``RiskRouterWeights``/``RiskRouterThresholds`` (M2
    routing) because these are purely about numeric/text comparison
    tolerance, not risk allocation.
    """

    # Relative tolerance for treating two same-unit numeric values as
    # "approximately equal" rather than requiring bit-exact match. Applied
    # as |a - b| <= relative_tolerance * max(|a|, |b|, 1e-9).
    numeric_relative_tolerance: float = 0.01

    # Minimum count of significant (non-stopword) shared normalized tokens
    # required before two claims are considered "about the same subject"
    # and therefore eligible for numeric/temporal comparison at all.
    min_shared_tokens_for_comparison: int = 2

    # Cap on the number of claim pairs compared per CandidatePair, so a
    # pathological evidence item with hundreds of claims cannot blow up
    # per-candidate latency.
    max_claim_pairs_per_candidate: int = 25

    # A numeric/temporal DIFFERENCE between two claims is only a
    # contradiction when both claims state the same proposition.  Sharing a
    # couple of words is not enough: two sentences of one drug label both
    # mention "dose" and "daily" yet give different numbers for different
    # indications.  This is the minimum overlap coefficient
    # |A & B| / min(|A|, |B|) of the claims' significant non-numeric tokens
    # required before a difference is reported as CONTRADICTION.  Pairs
    # below it yield no deterministic verdict and are left to contextual
    # validation and selective NLI.
    min_proposition_overlap_for_contradiction: float = 0.5


@dataclass(frozen=True, slots=True)
class CandidateGenerationConfig:
    """Tunable thresholds for candidate generation (Section 9)."""

    max_candidates: int = 500

    # Minimum token length considered "significant" for the inverted index
    # (short tokens like "is"/"of" are excluded as noise / stopwords).
    min_token_length: int = 4

    # Posting lists longer than this are truncated to this many items
    # (deterministically, by sorted EvidenceItemId) before pairing, so a
    # token shared by very many evidence items still contributes bounded,
    # reproducible candidate coverage rather than being dropped entirely.
    max_posting_list_size: int = 40

    # Only claim types below matter for candidate generation; FACTUAL /
    # RECOMMENDATION claims rarely admit deterministic conflict detection
    # and are excluded to keep the index small and focused.
    eligible_claim_types: frozenset[ClaimType] = field(
        default_factory=lambda: frozenset(
            {ClaimType.NUMERIC, ClaimType.DOSAGE, ClaimType.TEMPORAL}
        )
    )


@dataclass(frozen=True, slots=True)
class SelectiveRoutingConfig:
    """Tunable thresholds for selective NLI routing (Section 12)."""

    # Confidence below which a deterministic/contextual result is treated
    # as "inconclusive" and therefore a candidate for NLI escalation.
    inconclusive_confidence_threshold: float = 0.6


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Top-level configuration bundle threaded through the M8 pipeline."""

    deterministic: DeterministicConfig = field(default_factory=DeterministicConfig)
    candidate_generation: CandidateGenerationConfig = field(
        default_factory=CandidateGenerationConfig
    )
    selective_routing: SelectiveRoutingConfig = field(
        default_factory=SelectiveRoutingConfig
    )
    conflict_detection_max_candidates: int = 500


@dataclass(frozen=True, slots=True)
class NumericExtraction:
    """One numeric token pulled out of claim text, with its surrounding unit.

    ``unit`` is the raw lowercase unit string as found in the text (e.g.
    "mg", "%", "iu"); ``None`` when no unit token immediately follows the
    number. ``span`` records the character offsets in the *source* string
    the number was extracted from, for audit/debug purposes only.
    """

    value: float
    unit: str | None
    span: tuple[int, int]


@dataclass(frozen=True, slots=True)
class CandidateClaimPair:
    """One pair of claims (one from each evidence item) considered together
    while deterministically/contextually validating a CandidatePair.

    Not a schema type: purely an internal working structure used to keep
    claim -> evidence -> decision traceability inside the validators
    without re-deriving it downstream.
    """

    claim_id_a: ClaimId
    claim_id_b: ClaimId
    item_id_a: EvidenceItemId
    item_id_b: EvidenceItemId
    shared_token_count: int