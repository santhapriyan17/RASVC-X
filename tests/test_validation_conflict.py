"""Tests for rasvcx.validation.candidate_generation.CandidateGenerator.

Covers: candidates generated for related claims; unrelated claims never
paired; deterministic candidate ids across repeated invocation; idempotent
re-generation on the same bundle; bounded output respecting max_candidates;
and coverage under heavy token duplication (the regression this file
guards against: previously, a token whose posting list exceeded
max_posting_list_size was dropped entirely, and once *every* token in a
bundle exceeded that cap -- realistic once evidence volume grows -- this
silently produced zero candidates for the whole bundle).
"""

from __future__ import annotations

from rasvcx.schemas.common import ChunkId, ClaimId, EvidenceItemId, QueryId, SourceType
from rasvcx.schemas.claims import Claim, ClaimSource, ClaimType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.validation.candidate_generation import CandidateGenerator
from rasvcx.validation.validation_types import CandidateGenerationConfig


def _risk_profile() -> RiskProfile:
    return RiskProfile(
        overall_risk_score=0.3, feature_scores=RiskFeatureScores(),
        validation_depth=ValidationDepth.STANDARD, retrieval_retry_budget=1, nli_call_allowance=5,
    )


def _provenance() -> Provenance:
    return Provenance(source_type=SourceType.DRUG_LABEL, date="2023-01-01", jurisdiction="US",
                       population="adults", dosage_context="general")


def _add_dosage_item(bundle: EvidenceBundle, idx: int, drug: str, dose: int) -> None:
    claim_id = ClaimId(f"c{idx}")
    chunk_id = ChunkId(f"ch{idx}")
    text = f"The recommended dose of {drug} is {dose} mg twice daily for adults."
    claim = Claim(claim_id=claim_id, text=text, source=ClaimSource.EVIDENCE_EXTRACTION,
                  claim_type=ClaimType.DOSAGE, origin_chunk_id=chunk_id, normalized_text=text.lower())
    bundle.register_claim(claim)
    item = EvidenceItem(item_id=EvidenceItemId(f"e{idx}"), chunk_id=chunk_id, text=text,
                         retrieval_score=0.5, provenance=_provenance(),
                         extracted_claim_ids=frozenset({claim_id}))
    bundle.add_evidence_item(item)


def _empty_bundle() -> EvidenceBundle:
    return EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())


def test_empty_bundle_yields_no_candidates():
    bundle = _empty_bundle()
    generated = CandidateGenerator().generate(bundle)
    assert generated == []
    assert bundle.conflict_candidates == []


def test_single_item_yields_no_candidates():
    bundle = _empty_bundle()
    _add_dosage_item(bundle, 0, "ibuprofen", 200)
    generated = CandidateGenerator().generate(bundle)
    assert generated == []


def test_related_items_are_paired():
    bundle = _empty_bundle()
    _add_dosage_item(bundle, 0, "ibuprofen", 200)
    _add_dosage_item(bundle, 1, "ibuprofen", 400)
    generated = CandidateGenerator().generate(bundle)
    assert len(generated) == 1
    pair_ids = {generated[0].item_id_a, generated[0].item_id_b}
    assert pair_ids == {EvidenceItemId("e0"), EvidenceItemId("e1")}


def test_unrelated_topics_are_not_paired_when_no_shared_token():
    bundle = _empty_bundle()
    claim_a = Claim(claim_id=ClaimId("ca"), text="Dose is 200 mg.", source=ClaimSource.EVIDENCE_EXTRACTION,
                     claim_type=ClaimType.DOSAGE, origin_chunk_id=ChunkId("cha"), normalized_text="alpha")
    claim_b = Claim(claim_id=ClaimId("cb"), text="Dose is 200 mg.", source=ClaimSource.EVIDENCE_EXTRACTION,
                     claim_type=ClaimType.DOSAGE, origin_chunk_id=ChunkId("chb"), normalized_text="zulu")
    bundle.register_claim(claim_a)
    bundle.register_claim(claim_b)
    bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("e0"), chunk_id=ChunkId("cha"),
                                           text="alpha quixotic narwhal", retrieval_score=0.5,
                                           provenance=_provenance(), extracted_claim_ids=frozenset({ClaimId("ca")})))
    bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("e1"), chunk_id=ChunkId("chb"),
                                           text="zulu whimsical penguin", retrieval_score=0.5,
                                           provenance=_provenance(), extracted_claim_ids=frozenset({ClaimId("cb")})))
    generated = CandidateGenerator().generate(bundle)
    assert generated == []


def test_candidate_ids_are_deterministic_across_separate_generators():
    bundle1 = _empty_bundle()
    _add_dosage_item(bundle1, 0, "ibuprofen", 200)
    _add_dosage_item(bundle1, 1, "ibuprofen", 400)
    ids1 = {c.candidate_id for c in CandidateGenerator().generate(bundle1)}

    bundle2 = _empty_bundle()
    _add_dosage_item(bundle2, 0, "ibuprofen", 200)
    _add_dosage_item(bundle2, 1, "ibuprofen", 400)
    ids2 = {c.candidate_id for c in CandidateGenerator().generate(bundle2)}

    assert ids1 == ids2


def test_regenerating_on_the_same_bundle_is_idempotent():
    bundle = _empty_bundle()
    _add_dosage_item(bundle, 0, "ibuprofen", 200)
    _add_dosage_item(bundle, 1, "ibuprofen", 400)
    generator = CandidateGenerator()
    first = generator.generate(bundle)
    second = generator.generate(bundle)
    assert len(first) == 1
    assert second == []  # already present; not re-added
    assert len(bundle.conflict_candidates) == 1


def test_max_candidates_is_respected():
    bundle = _empty_bundle()
    for i in range(12):
        _add_dosage_item(bundle, i, "ibuprofen", 100 + i)
    generator = CandidateGenerator(CandidateGenerationConfig(max_candidates=5))
    generated = generator.generate(bundle)
    assert len(generated) <= 5
    assert len(bundle.conflict_candidates) <= 5


def test_heavy_token_duplication_still_yields_bounded_nonzero_coverage():
    # Regression test for the posting-list coverage collapse: with every
    # item sharing the same handful of high-frequency tokens (as happens
    # once evidence volume grows), candidate generation must still return
    # *some* bounded, deterministic coverage rather than silently zero.
    bundle = _empty_bundle()
    drugs = ["ibuprofen", "acetaminophen", "amoxicillin", "metformin", "lisinopril"]
    for i in range(150):
        _add_dosage_item(bundle, i, drugs[i % len(drugs)], 100 + (i % 5) * 50)
    generator = CandidateGenerator(CandidateGenerationConfig(max_posting_list_size=10, max_candidates=50))
    generated = generator.generate(bundle)
    assert len(generated) > 0
    assert len(generated) <= 50


def test_priority_ordering_is_highest_first():
    bundle = _empty_bundle()
    _add_dosage_item(bundle, 0, "ibuprofen", 200)
    _add_dosage_item(bundle, 1, "ibuprofen", 400)
    _add_dosage_item(bundle, 2, "acetaminophen", 500)
    generated = CandidateGenerator().generate(bundle)
    priorities = [c.priority for c in generated]
    assert priorities == sorted(priorities, reverse=True)