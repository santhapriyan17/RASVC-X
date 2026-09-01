"""Claim-to-evidence linking (Module 7).

Builds a deterministic index mapping extracted claims back to the evidence
items they were extracted from, using ``Claim.origin_chunk_id`` →
``EvidenceItem.chunk_id`` as the primary relationship key.

Design constraints:
  - O(n + m) complexity where n = number of evidence items, m = number of
    claims.  Two linear passes: one to build the chunk_id → item_ids index,
    one to resolve each claim.
  - No O(n × m) nested iteration.
  - UNKNOWN origin_chunk_id produces no link (not an error).
  - Missing chunk_id (claim's chunk not in the bundle) produces no link.
  - Multiple evidence items sharing the same chunk_id are all linked.
  - Deterministic output ordering (sorted by EvidenceItemId).
  - Pure computation: no I/O, no ML, no mutation of inputs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from rasvcx.schemas.claims import Claim
from rasvcx.schemas.common import ChunkId, ClaimId, EvidenceItemId, _UnknownType
from rasvcx.schemas.evidence import EvidenceItem


class ClaimLinker:
    """Links claims to evidence items via chunk_id index.

    Usage::

        linker = ClaimLinker()
        mapping = linker.link(claims, evidence_items)
        # mapping: {EvidenceItemId: frozenset[ClaimId]}

    The returned mapping contains only items that have at least one linked
    claim.  Items with no claims are absent from the mapping (not present
    with an empty frozenset).
    """

    def link(
        self,
        claims: list[Claim],
        evidence_items: Mapping[EvidenceItemId, EvidenceItem],
    ) -> dict[EvidenceItemId, frozenset[ClaimId]]:
        """Link claims to evidence items using chunk_id.

        Pass 1 (O(n)): Build ``chunk_id → list[EvidenceItemId]`` index.
        Pass 2 (O(m)): Resolve each claim's ``origin_chunk_id`` against
        the index.

        Args:
            claims: Extracted claims (may include claims with UNKNOWN
                origin_chunk_id; these are silently skipped).
            evidence_items: The current evidence items mapping from the
                bundle (not mutated).

        Returns:
            Mapping from EvidenceItemId to the frozenset of ClaimIds
            linked to that item.  Only items with at least one linked
            claim are present.
        """
        if not claims or not evidence_items:
            return {}

        # --- Pass 1: Build chunk_id → item_ids index ---
        chunk_to_items: dict[ChunkId, list[EvidenceItemId]] = defaultdict(list)
        for item_id, item in evidence_items.items():
            chunk_to_items[item.chunk_id].append(item_id)

        # --- Pass 2: Resolve each claim's origin_chunk_id ---
        item_to_claims: dict[EvidenceItemId, set[ClaimId]] = defaultdict(set)

        for claim in claims:
            # UNKNOWN origin → no link (not an error)
            if isinstance(claim.origin_chunk_id, _UnknownType):
                continue

            chunk_id = claim.origin_chunk_id
            linked_item_ids = chunk_to_items.get(chunk_id)

            if linked_item_ids is None:
                # Claim's chunk is not in the bundle — orphan claim.
                # This is not an error; the claim remains registered but
                # unlinked.
                continue

            for item_id in linked_item_ids:
                item_to_claims[item_id].add(claim.claim_id)

        # Convert to frozensets for immutability
        return {
            item_id: frozenset(sorted(claim_ids))
            for item_id, claim_ids in item_to_claims.items()
        }