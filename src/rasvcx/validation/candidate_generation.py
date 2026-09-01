"""Candidate generation (Section 9).

Identifies pairs of evidence items worth validating against each other,
using an inverted index over significant claim tokens rather than brute-
force O(items^2) comparison. Preserves full claim -> evidence ->
CandidatePair traceability: every generated CandidatePair records exactly
which two EvidenceItemIds it concerns, and the caller can always recover
the contributing claims via EvidenceItem.extracted_claim_ids.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass

from rasvcx.schemas.common import CandidateId, EvidenceItemId
from rasvcx.schemas.evidence import CandidatePair, ConflictLabel, EvidenceBundle, EvidenceItem
from rasvcx.validation.deterministic import extract_numbers, significant_tokens
from rasvcx.validation.validation_types import CandidateGenerationConfig


@dataclass(frozen=True, slots=True)
class _PairKey:
    item_id_a: EvidenceItemId
    item_id_b: EvidenceItemId


def _ordered_pair(a: EvidenceItemId, b: EvidenceItemId) -> _PairKey:
    # Canonical ordering so (a, b) and (b, a) collapse to one key.
    return _PairKey(a, b) if a <= b else _PairKey(b, a)


def _stable_candidate_id(item_a: EvidenceItemId, item_b: EvidenceItemId) -> CandidateId:
    """Deterministic candidate id derived from the (ordered) item pair.

    Deterministic generation is required for reproducibility across
    repeated pipeline runs on the same bundle (Section 21: "repeated
    pipeline execution" / "deterministic repeated execution"). Uses
    uuid5 over a fixed namespace rather than a counter or uuid4, so the
    same item pair always yields the same CandidateId.
    """
    namespace = uuid.UUID("6f1f7f2e-2f2b-4c2e-9d3e-2b7d2b9a5e11")
    name = f"{item_a}::{item_b}"
    return CandidateId(str(uuid.uuid5(namespace, name)))


def _label_for(item_a: EvidenceItem, item_b: EvidenceItem, shared_tokens: frozenset[str]) -> ConflictLabel:
    """Pick the ConflictLabel best explaining *why* the pair was flagged.

    This is a preliminary label (see ConflictLabel docstring in
    schemas/evidence.py) -- it only records the generation-time rationale,
    not the resolved EvidenceRelationship computed later.
    """
    text_a = item_a.text
    text_b = item_b.text
    if extract_numbers(text_a) and extract_numbers(text_b):
        return ConflictLabel.POTENTIAL_NUMERIC_CONFLICT

    prov_a, prov_b = item_a.provenance, item_b.provenance
    if (
        prov_a.population != prov_b.population
        or prov_a.jurisdiction != prov_b.jurisdiction
        or prov_a.dosage_context != prov_b.dosage_context
    ):
        return ConflictLabel.POTENTIAL_CONTEXTUAL_CONFLICT

    return ConflictLabel.POTENTIAL_SEMANTIC_CONFLICT


class CandidateGenerator:
    """Generates bounded CandidatePairs from an EvidenceBundle's claims.

    Complexity: O(total_claim_tokens) to build the inverted index, plus
    O(sum of posting_list_size^2) to expand pairs, with each posting list
    capped at ``config.max_posting_list_size`` -- avoids full O(n^2) over
    all evidence items when the bundle is large.
    """

    def __init__(self, config: CandidateGenerationConfig | None = None) -> None:
        self._config = config or CandidateGenerationConfig()

    def generate(self, bundle: EvidenceBundle) -> list[CandidatePair]:
        """Populate ``bundle.conflict_candidates`` and return the list added.

        Idempotent with respect to already-present candidates: if a
        candidate with the same (deterministic) id already exists in the
        bundle it is skipped rather than raising, so repeated invocation
        on the same bundle is safe.
        """
        token_index = self._build_token_index(bundle)

        seen_pairs: set[_PairKey] = set()
        generated: list[CandidatePair] = []
        existing_ids = {c.candidate_id for c in bundle.conflict_candidates}

        for token, item_ids in token_index.items():
            ordered = sorted(item_ids)
            if len(ordered) > self._config.max_posting_list_size:
                # A token this common is not discriminative on its own, but
                # dropping it outright would silently zero out candidate
                # coverage for an entire topic once evidence volume grows
                # (a real, measured failure mode: with heavily duplicated
                # claim vocabulary, *every* token can exceed the cap
                # simultaneously, yielding zero candidates for the whole
                # bundle). Bound the list deterministically instead of
                # discarding it, so large topics still get partial,
                # reproducible coverage rather than none.
                ordered = ordered[: self._config.max_posting_list_size]

            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    item_id_a, item_id_b = ordered[i], ordered[j]
                    pair_key = _ordered_pair(item_id_a, item_id_b)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    if len(generated) + len(existing_ids) >= self._config.max_candidates:
                        continue

                    candidate_id = _stable_candidate_id(pair_key.item_id_a, pair_key.item_id_b)
                    if candidate_id in existing_ids:
                        continue

                    item_a = bundle.evidence_items[pair_key.item_id_a]
                    item_b = bundle.evidence_items[pair_key.item_id_b]
                    tokens_a = significant_tokens(item_a.text)
                    tokens_b = significant_tokens(item_b.text)
                    shared = tokens_a & tokens_b
                    label = _label_for(item_a, item_b, shared)
                    priority = min(1.0, len(shared) / 10.0)

                    candidate = CandidatePair(
                        candidate_id=candidate_id,
                        item_id_a=pair_key.item_id_a,
                        item_id_b=pair_key.item_id_b,
                        label=label,
                        priority=priority,
                    )
                    bundle.add_conflict_candidate(candidate, max_candidates=self._config.max_candidates)
                    generated.append(candidate)

        # Highest-priority candidates first: downstream selective routing
        # (Section 12) consumes this ordering to spend expensive validation
        # budget on the most-worth-checking pairs first.
        generated.sort(key=lambda c: c.priority, reverse=True)
        return generated

    def _build_token_index(
        self, bundle: EvidenceBundle
    ) -> dict[str, set[EvidenceItemId]]:
        """Map significant claim token -> set of EvidenceItemIds whose
        linked claims mention it, restricted to eligible claim types.
        """
        index: dict[str, set[EvidenceItemId]] = defaultdict(set)
        eligible_types = self._config.eligible_claim_types

        for item_id, item in bundle.evidence_items.items():
            if not item.extracted_claim_ids:
                continue
            for claim_id in item.extracted_claim_ids:
                claim = bundle.claims.get(claim_id)
                if claim.claim_type not in eligible_types:
                    continue
                text = claim.normalized_text or claim.text
                for token in significant_tokens(text):
                    if len(token) < self._config.min_token_length:
                        continue
                    index[token].add(item_id)

        return index