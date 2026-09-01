"""Atomic claim extraction pipeline (Module 7).

Orchestrates the full deterministic claim extraction workflow:

    EvidenceBundle
        → sentence segmentation
        → claim classification
        → safety-critical detection
        → deterministic claim ID generation
        → text normalization
        → ClaimRegistry registration
        → claim → evidence linkage
        → EvidenceItem.extracted_claim_ids update
        → AtomicClaimExtractionResult
        → timing recorded on EvidenceBundle

Architectural boundaries:
  - M7 does NOT modify M1 schemas.
  - M7 does NOT perform validation, NLI, conflict detection, or generation.
  - M7 does NOT call LLMs, networks, or databases.
  - M7 produces structured claim representations for downstream safety layers.
  - EvidenceItem updates use ``dataclasses.replace()`` (M4 established pattern).
  - Claims are registered before evidence items are updated (M1 contract).
  - Timing uses ``bundle.record_stage_elapsed("atomic_claim_extraction", ...)``.

Idempotency:
  Running the pipeline twice on the same bundle produces the same claims
  (same IDs, same text, same types, same flags).  ClaimRegistry's idempotent
  re-registration ensures no duplicate logical claims accumulate.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import Mapping

from rasvcx.claims.claim_linker import ClaimLinker
from rasvcx.claims.extractor import (
    ClaimExtractor,
)
from rasvcx.schemas.claims import Claim, ClaimType
from rasvcx.schemas.common import ClaimId, EvidenceItemId
from rasvcx.schemas.evidence import EvidenceBundle


# ═══════════════════════════════════════════════════════════════════════════
# RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class AtomicClaimExtractionResult:
    """Immutable, auditable result of the atomic claim extraction pipeline.

    Provides deterministic statistics about the extraction run.  All counts
    are internally consistent (safety_critical_claim_count <= total_claims,
    type counts sum to total_claims, etc.).

    Attributes:
        claim_ids_by_item:              Mapping from EvidenceItemId to the
                                         frozenset of ClaimIds extracted from
                                         and linked to that item.
        total_claims:                   Total number of unique claims extracted.
        safety_critical_claim_count:    Count of claims flagged as safety-critical.
        factual_claim_count:            Count of FACTUAL claims.
        numeric_claim_count:            Count of NUMERIC claims.
        dosage_claim_count:             Count of DOSAGE claims.
        temporal_claim_count:           Count of TEMPORAL claims.
        recommendation_claim_count:     Count of RECOMMENDATION claims.
        other_claim_count:              Count of OTHER claims.
        unlinked_claim_count:           Claims registered but not linked to any
                                         evidence item (e.g. orphan chunks).
    """

    claim_ids_by_item: Mapping[EvidenceItemId, frozenset[ClaimId]]
    total_claims: int
    safety_critical_claim_count: int
    factual_claim_count: int = 0
    numeric_claim_count: int = 0
    dosage_claim_count: int = 0
    temporal_claim_count: int = 0
    recommendation_claim_count: int = 0
    other_claim_count: int = 0
    unlinked_claim_count: int = 0

    def __post_init__(self) -> None:
        if self.total_claims < 0:
            raise ValueError(
                f"total_claims must be non-negative, got {self.total_claims}"
            )
        if self.safety_critical_claim_count < 0:
            raise ValueError(
                f"safety_critical_claim_count must be non-negative, "
                f"got {self.safety_critical_claim_count}"
            )
        if self.safety_critical_claim_count > self.total_claims:
            raise ValueError(
                f"safety_critical_claim_count ({self.safety_critical_claim_count}) "
                f"cannot exceed total_claims ({self.total_claims})"
            )
        # Validate type counts sum
        type_sum = (
            self.factual_claim_count
            + self.numeric_claim_count
            + self.dosage_claim_count
            + self.temporal_claim_count
            + self.recommendation_claim_count
            + self.other_claim_count
        )
        if type_sum != self.total_claims:
            raise ValueError(
                f"Type counts must sum to total_claims ({self.total_claims}), "
                f"got {type_sum}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


class AtomicClaimPipeline:
    """Orchestrator for deterministic atomic claim extraction.

    Composes ClaimExtractor and ClaimLinker into a single pipeline that:
      1. Extracts claims from each evidence item's text.
      2. Registers all claims in the bundle's ClaimRegistry.
      3. Links claims back to evidence items via chunk_id.
      4. Updates EvidenceItems with their extracted_claim_ids.
      5. Records elapsed time on the bundle.

    Dependency-injectable: extractor and linker can be replaced for testing.

    Thread-safety: this class holds no mutable state.  Multiple concurrent
    calls to ``run()`` on *different* bundles are safe.  Concurrent calls
    on the *same* bundle are NOT safe (EvidenceBundle is not thread-safe).
    """

    def __init__(
        self,
        extractor: ClaimExtractor | None = None,
        linker: ClaimLinker | None = None,
    ) -> None:
        self._extractor = extractor or ClaimExtractor()
        self._linker = linker or ClaimLinker()

    def run(self, bundle: EvidenceBundle) -> AtomicClaimExtractionResult:
        """Run the atomic claim extraction pipeline on a bundle.

        Mutates the bundle:
          - Registers extracted claims in ``bundle.claims``.
          - Replaces evidence items in ``bundle.evidence_items`` with new
            frozen instances carrying populated ``extracted_claim_ids``.
          - Records elapsed time under ``"atomic_claim_extraction"``.

        Args:
            bundle: The EvidenceBundle to process.  Must have evidence_items
                already populated (post-retrieval, post-reranking).

        Returns:
            AtomicClaimExtractionResult with extraction statistics.
        """
        start = time.perf_counter()
        try:
            return self._run_inner(bundle)
        finally:
            elapsed = time.perf_counter() - start
            bundle.record_stage_elapsed("atomic_claim_extraction", elapsed)

    def _run_inner(
        self, bundle: EvidenceBundle
    ) -> AtomicClaimExtractionResult:
        """Core extraction logic, separated from timing."""
        items = bundle.evidence_items

        # Empty bundle → empty result
        if not items:
            return AtomicClaimExtractionResult(
                claim_ids_by_item={},
                total_claims=0,
                safety_critical_claim_count=0,
                factual_claim_count=0,
                numeric_claim_count=0,
                dosage_claim_count=0,
                temporal_claim_count=0,
                recommendation_claim_count=0,
                other_claim_count=0,
                unlinked_claim_count=0,
            )

        # ── Phase 1: Extract + register claims ────────────────────────
        all_claims: list[Claim] = []

        # Process items in deterministic order (sorted by item_id)
        sorted_item_ids = sorted(items.keys())

        for item_id in sorted_item_ids:
            item = items[item_id]
            extracted = self._extractor.extract(
                item_id=item.item_id,
                chunk_id=item.chunk_id,
                text=item.text,
            )
            for claim in extracted:
                # Register each claim.  ClaimRegistry handles idempotent
                # re-registration for repeated pipeline runs.
                bundle.register_claim(claim)

            all_claims.extend(extracted)

        # ── Phase 2: Link claims → evidence items ─────────────────────
        linkage = self._linker.link(all_claims, items)

        # ── Phase 3: Update evidence items with claim IDs ─────────────
        # Use dataclasses.replace() to create new frozen EvidenceItem
        # instances.  This is the established M4 pattern.
        for item_id in sorted_item_ids:
            linked_claim_ids = linkage.get(item_id, frozenset())
            current_item = bundle.evidence_items[item_id]

            # Merge with any previously-extracted claim IDs (idempotency)
            merged_ids = current_item.extracted_claim_ids | linked_claim_ids

            if merged_ids != current_item.extracted_claim_ids:
                new_item = dataclasses.replace(
                    current_item, extracted_claim_ids=merged_ids
                )
                bundle.evidence_items[item_id] = new_item

        # ── Phase 4: Compute statistics ───────────────────────────────
        # Deduplicate claims by claim_id (in case the same claim was
        # extracted multiple times in an idempotent re-run)
        unique_claims: dict[str, Claim] = {}
        for claim in all_claims:
            unique_claims[claim.claim_id] = claim

        total_claims = len(unique_claims)
        safety_count = sum(
            1 for c in unique_claims.values() if c.is_safety_critical
        )

        type_counts: dict[ClaimType, int] = {ct: 0 for ct in ClaimType}
        for claim in unique_claims.values():
            type_counts[claim.claim_type] += 1

        # Count unlinked claims: claims not assigned to any evidence item
        all_linked_claim_ids: set[str] = set()
        for claim_ids in linkage.values():
            all_linked_claim_ids.update(claim_ids)

        unlinked_count = sum(
            1 for cid in unique_claims if cid not in all_linked_claim_ids
        )

        return AtomicClaimExtractionResult(
            claim_ids_by_item=linkage,
            total_claims=total_claims,
            safety_critical_claim_count=safety_count,
            factual_claim_count=type_counts.get(ClaimType.FACTUAL, 0),
            numeric_claim_count=type_counts.get(ClaimType.NUMERIC, 0),
            dosage_claim_count=type_counts.get(ClaimType.DOSAGE, 0),
            temporal_claim_count=type_counts.get(ClaimType.TEMPORAL, 0),
            recommendation_claim_count=type_counts.get(
                ClaimType.RECOMMENDATION, 0
            ),
            other_claim_count=type_counts.get(ClaimType.OTHER, 0),
            unlinked_claim_count=unlinked_count,
        )