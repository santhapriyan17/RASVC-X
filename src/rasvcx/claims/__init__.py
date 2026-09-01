"""Module 7 — Atomic Claims + Claim/Evidence Linkage.

Public surface for ``src/rasvcx/claims``.

Downstream modules (M8+) should import from this package rather than
reaching into individual submodules directly, so M7's internal file
layout is free to evolve without breaking consumers.

Public API:
    ClaimExtractor, ExtractionConfig, DEFAULT_EXTRACTION_CONFIG
        -- evidence text → Claim objects (sentence segmentation,
           classification, safety-critical detection, ID generation).
    split_sentences, classify_claim_type, is_safety_critical,
    generate_claim_id
        -- individual extraction primitives for direct use/testing.
    normalize_claim_text, apply_normalization
        -- deterministic, idempotent claim text normalization.
    ClaimLinker
        -- O(n+m) chunk_id-indexed claim → evidence linking.
    AtomicClaimPipeline, AtomicClaimExtractionResult
        -- full extraction pipeline orchestration.
"""

from __future__ import annotations

from rasvcx.claims.extractor import (
    ClaimExtractor,
    DEFAULT_EXTRACTION_CONFIG,
    ExtractionConfig,
    classify_claim_type,
    generate_claim_id,
    is_safety_critical,
    split_sentences,
)
from rasvcx.claims.claim_normalizer import (
    apply_normalization,
    normalize_claim_text,
)
from rasvcx.claims.claim_linker import ClaimLinker
from rasvcx.claims.atomic_claims import (
    AtomicClaimExtractionResult,
    AtomicClaimPipeline,
)

__all__ = [
    # Extractor
    "ClaimExtractor",
    "ExtractionConfig",
    "DEFAULT_EXTRACTION_CONFIG",
    "split_sentences",
    "classify_claim_type",
    "is_safety_critical",
    "generate_claim_id",
    # Normalizer
    "normalize_claim_text",
    "apply_normalization",
    # Linker
    "ClaimLinker",
    # Pipeline
    "AtomicClaimPipeline",
    "AtomicClaimExtractionResult",
]