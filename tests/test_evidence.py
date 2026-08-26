"""Tests for rasvcx.schemas.evidence -- the canonical EvidenceBundle contract."""

import pytest

from rasvcx.schemas.claims import Claim, ClaimSource, ClaimType
from rasvcx.schemas.common import (
    UNKNOWN,
    CandidateId,
    ChunkId,
    ClaimId,
    EvidenceItemId,
    QueryId,
    SourceType,
)
from rasvcx.schemas.evidence import (
    CandidatePair,
    ConflictLabel,
    EvidenceBundle,
    EvidenceItem,
    Provenance,
)


def make_provenance(**overrides):
    defaults = dict(
        source_type=SourceType.DRUG_LABEL,
        date=UNKNOWN,
        jurisdiction="US",
        population=UNKNOWN,
        dosage_context="adult",
    )
    defaults.update(overrides)
    return Provenance(**defaults)


def make_bundle():
    return EvidenceBundle(query_id=QueryId("q1"), risk_profile=None)


def make_claim(claim_id="c1"):
    return Claim(
        claim_id=ClaimId(claim_id),
        text="10mg dose",
        source=ClaimSource.EVIDENCE_EXTRACTION,
        claim_type=ClaimType.DOSAGE,
        origin_chunk_id=ChunkId("chunk1"),
    )


def make_item(item_id="e1", claim_ids=frozenset()):
    return EvidenceItem(
        item_id=EvidenceItemId(item_id),
        chunk_id=ChunkId("chunk1"),
        text="some evidence text",
        retrieval_score=0.9,
        provenance=make_provenance(),
        extracted_claim_ids=claim_ids,
    )


def test_bundle_construction():
    bundle = make_bundle()
    assert bundle.query_id == "q1"
    assert bundle.risk_profile is None
    assert len(bundle.evidence_items) == 0
    assert len(bundle.claims) == 0


def test_unknown_provenance_fields():
    prov = make_provenance()
    assert prov.date is UNKNOWN
    assert prov.population is UNKNOWN
    assert prov.jurisdiction == "US"


def test_evidence_item_requires_nonempty_text():
    with pytest.raises(ValueError):
        EvidenceItem(
            item_id=EvidenceItemId("e1"),
            chunk_id=ChunkId("chunk1"),
            text="",
            retrieval_score=0.5,
            provenance=make_provenance(),
        )


def test_claim_registration_before_evidence_item_reference():
    bundle = make_bundle()
    claim = make_claim()
    bundle.register_claim(claim)
    item = make_item(claim_ids=frozenset({ClaimId("c1")}))
    bundle.add_evidence_item(item)
    assert bundle.evidence_items[EvidenceItemId("e1")] is item


def test_evidence_item_rejects_unregistered_claim_reference():
    bundle = make_bundle()
    item = make_item(claim_ids=frozenset({ClaimId("missing")}))
    with pytest.raises(ValueError):
        bundle.add_evidence_item(item)


def test_duplicate_evidence_item_rejected():
    bundle = make_bundle()
    item = make_item()
    bundle.add_evidence_item(item)
    with pytest.raises(ValueError):
        bundle.add_evidence_item(item)


def test_no_duplicate_claim_object_storage():
    bundle = make_bundle()
    claim = make_claim()
    bundle.register_claim(claim)
    assert bundle.claims.get(ClaimId("c1")) is claim

    different_content = Claim(
        claim_id=ClaimId("c1"),
        text="a completely different claim",
        source=ClaimSource.EVIDENCE_EXTRACTION,
        claim_type=ClaimType.DOSAGE,
        origin_chunk_id=ChunkId("chunk1"),
    )
    with pytest.raises(ValueError):
        bundle.register_claim(different_content)

    dup_same_content = Claim(
        claim_id=ClaimId("c1"),
        text="10mg dose",
        source=ClaimSource.EVIDENCE_EXTRACTION,
        claim_type=ClaimType.DOSAGE,
        origin_chunk_id=ChunkId("chunk1"),
    )
    bundle.register_claim(dup_same_content)  # identical content: no-op, allowed
    assert len(bundle.claims) == 1


def test_conflict_candidate_storage_and_bound():
    bundle = make_bundle()
    cand = CandidatePair(
        candidate_id=CandidateId("cand1"),
        item_id_a=EvidenceItemId("e1"),
        item_id_b=EvidenceItemId("e2"),
        label=ConflictLabel.POTENTIAL_NUMERIC_CONFLICT,
    )
    bundle.add_conflict_candidate(cand, max_candidates=30)
    assert len(bundle.conflict_candidates) == 1

    cand2 = CandidatePair(
        candidate_id=CandidateId("cand2"),
        item_id_a=EvidenceItemId("e1"),
        item_id_b=EvidenceItemId("e3"),
        label=ConflictLabel.POTENTIAL_SEMANTIC_CONFLICT,
    )
    with pytest.raises(ValueError):
        bundle.add_conflict_candidate(cand2, max_candidates=1)


def test_conflict_candidate_duplicate_id_rejected():
    bundle = make_bundle()
    cand = CandidatePair(
        candidate_id=CandidateId("cand1"),
        item_id_a=EvidenceItemId("e1"),
        item_id_b=EvidenceItemId("e2"),
        label=ConflictLabel.POTENTIAL_NUMERIC_CONFLICT,
    )
    bundle.add_conflict_candidate(cand, max_candidates=30)
    with pytest.raises(ValueError):
        bundle.add_conflict_candidate(cand, max_candidates=30)


def test_validation_result_accumulates_not_overwrites():
    bundle = make_bundle()
    bundle.add_validation_result(CandidateId("cand1"), "result-a")
    assert bundle.validation_results[CandidateId("cand1")] == "result-a"
    with pytest.raises(ValueError):
        bundle.add_validation_result(CandidateId("cand1"), "result-b")


def test_resolution_accumulates():
    bundle = make_bundle()
    bundle.add_resolution("relationship-1")
    bundle.add_resolution("relationship-2")
    assert bundle.resolution == ["relationship-1", "relationship-2"]


def test_metadata_counters():
    bundle = make_bundle()
    bundle.record_retrieval_call()
    bundle.record_retrieval_call()
    bundle.record_nli_calls(5)
    bundle.record_targeted_retrieval_used()
    bundle.record_corrective_attempt()
    bundle.record_stage_elapsed("hybrid_retrieval", 0.42)

    assert bundle.metadata.retrieval_calls == 2
    assert bundle.metadata.nli_calls == 5
    assert bundle.metadata.targeted_retrieval_used is True
    assert bundle.metadata.corrective_attempts_used == 1
    assert bundle.metadata.elapsed_per_stage["hybrid_retrieval"] == 0.42


def test_metadata_rejects_negative_nli_calls():
    bundle = make_bundle()
    with pytest.raises(ValueError):
        bundle.record_nli_calls(-1)


def test_targeted_retrieval_flag_defaults_false():
    bundle = make_bundle()
    assert bundle.metadata.targeted_retrieval_used is False


def test_no_global_state_leakage_between_bundles():
    bundle_a = make_bundle()
    bundle_a.register_claim(make_claim())
    bundle_a.record_retrieval_call()

    bundle_b = EvidenceBundle(query_id=QueryId("q2"), risk_profile=None)
    assert len(bundle_b.claims) == 0
    assert bundle_b.metadata.retrieval_calls == 0