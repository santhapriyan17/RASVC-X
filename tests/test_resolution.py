"""Tests for rasvcx.validation.resolution.EvidenceResolver.

Covers: full EvidenceRelationship taxonomy mapping (COMPATIBLE,
POPULATION_DIFF, JURISDICTION_DIFF, DOSAGE_DIFF, TEMPORAL_DIFF,
GENUINE_CONFLICT, UNRESOLVED); the regression this file specifically
guards against (a deterministic CONTRADICTION being silently overridden by
an unrelated contextual SUPPORTED result); UNKNOWN context never being
promoted to a confident GENUINE_CONFLICT; confidence bounds; and
determinism.
"""

from __future__ import annotations

from rasvcx.schemas.common import (
    UNKNOWN, ChunkId, ClaimId, EvidenceItemId, EvidenceRelationship, NLILabel, QueryId, SourceType,
)
from rasvcx.schemas.claims import Claim, ClaimSource, ClaimType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.schemas.validation import NLISignal, ValidationLabel, ValidationResult, ValidationStage
from rasvcx.validation.resolution import EvidenceResolver


def _risk_profile() -> RiskProfile:
    return RiskProfile(
        overall_risk_score=0.3, feature_scores=RiskFeatureScores(),
        validation_depth=ValidationDepth.STANDARD, retrieval_retry_budget=1, nli_call_allowance=5,
    )


def _bundle(prov_a: Provenance, prov_b: Provenance) -> EvidenceBundle:
    bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
    claim_a = Claim(claim_id=ClaimId("ca"), text="Dose is 500 mg.", source=ClaimSource.EVIDENCE_EXTRACTION,
                     claim_type=ClaimType.DOSAGE, origin_chunk_id=ChunkId("ch1"))
    claim_b = Claim(claim_id=ClaimId("cb"), text="Dose is 250 mg.", source=ClaimSource.EVIDENCE_EXTRACTION,
                     claim_type=ClaimType.DOSAGE, origin_chunk_id=ChunkId("ch2"))
    bundle.register_claim(claim_a)
    bundle.register_claim(claim_b)
    bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("e1"), chunk_id=ChunkId("ch1"),
                                           text="Dose is 500 mg.", retrieval_score=0.9, provenance=prov_a,
                                           extracted_claim_ids=frozenset({ClaimId("ca")})))
    bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("e2"), chunk_id=ChunkId("ch2"),
                                           text="Dose is 250 mg.", retrieval_score=0.8, provenance=prov_b,
                                           extracted_claim_ids=frozenset({ClaimId("cb")})))
    return bundle


def _matching_provenance(**overrides) -> tuple[Provenance, Provenance]:
    base = dict(source_type=SourceType.DRUG_LABEL, date="2023-01-01", jurisdiction="US",
                population="adults", dosage_context="general")
    base.update(overrides)
    return Provenance(**base), Provenance(**base)


def _result(stage: ValidationStage, label: ValidationLabel, confidence: float) -> ValidationResult:
    return ValidationResult(candidate_id="cand1", stage=stage, label=label, confidence=confidence)


def test_all_stages_supported_and_known_context_is_compatible():
    prov_a, prov_b = _matching_provenance()
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.SUPPORTED, 0.9)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.SUPPORTED, 0.8)
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, ctx, None)
    assert result.relationship is EvidenceRelationship.COMPATIBLE


def test_deterministic_contradiction_is_not_overridden_by_benign_contextual_result():
    # Regression test: a prior version of this resolver keyed only off the
    # "deepest stage that ran" and let a benign contextual SUPPORTED
    # result silently erase a genuine deterministic numeric contradiction.
    prov_a, prov_b = _matching_provenance()  # fully known, fully matching context
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.CONTRADICTION, 0.9)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.SUPPORTED, 0.8)
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, ctx, None)
    assert result.relationship is EvidenceRelationship.GENUINE_CONFLICT
    assert result.relationship is not EvidenceRelationship.COMPATIBLE


def test_contradiction_explained_by_known_population_mismatch_is_population_diff():
    prov_a, prov_b = _matching_provenance(population="adults")
    prov_b = Provenance(source_type=prov_b.source_type, date=prov_b.date, jurisdiction=prov_b.jurisdiction,
                         population="pediatric", dosage_context=prov_b.dosage_context)
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.CONTRADICTION, 0.9)
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, None, None)
    assert result.relationship is EvidenceRelationship.POPULATION_DIFF


def test_contradiction_explained_by_known_jurisdiction_mismatch_is_jurisdiction_diff():
    prov_a, prov_b = _matching_provenance(jurisdiction="US")
    prov_b = Provenance(source_type=prov_b.source_type, date=prov_b.date, jurisdiction="EU",
                         population=prov_b.population, dosage_context=prov_b.dosage_context)
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.CONTRADICTION, 0.9)
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, None, None)
    assert result.relationship is EvidenceRelationship.JURISDICTION_DIFF


def test_contradiction_explained_by_known_dosage_context_mismatch_is_dosage_diff():
    prov_a, prov_b = _matching_provenance(dosage_context="general")
    prov_b = Provenance(source_type=prov_b.source_type, date=prov_b.date, jurisdiction=prov_b.jurisdiction,
                         population=prov_b.population, dosage_context="renal_impairment")
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.CONTRADICTION, 0.9)
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, None, None)
    assert result.relationship is EvidenceRelationship.DOSAGE_DIFF


def test_contradiction_with_unknown_context_is_unresolved_not_genuine_conflict():
    # Absence of contextual information must never be promoted into a
    # confident GENUINE_CONFLICT verdict.
    prov_a, prov_b = _matching_provenance(dosage_context=UNKNOWN)
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.CONTRADICTION, 0.9)
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, None, None)
    assert result.relationship is EvidenceRelationship.UNRESOLVED
    assert result.relationship is not EvidenceRelationship.GENUINE_CONFLICT
    assert result.confidence <= 0.5


def _dated_pair(lc_a=None, lc_b=None):
    import dataclasses
    from rasvcx.schemas.evidence import SourceLifecycle, SourceRef

    prov_a = Provenance(source_type=SourceType.DRUG_LABEL, date="2015-01-01", jurisdiction="US",
                         population="adults", dosage_context="general")
    prov_b = Provenance(source_type=SourceType.DRUG_LABEL, date="2023-01-01", jurisdiction="US",
                         population="adults", dosage_context="general")
    bundle = _bundle(prov_a, prov_b)
    for iid, lc in (("e1", lc_a), ("e2", lc_b)):
        if lc is not None:
            it = bundle.evidence_items[EvidenceItemId(iid)]
            bundle.evidence_items[EvidenceItemId(iid)] = dataclasses.replace(
                it, source=SourceRef(doc_id=f"doc_{iid}", lifecycle=SourceLifecycle(**lc)))
    return bundle


def _resolve(bundle, label=ValidationLabel.CONTRADICTION):
    det = _result(ValidationStage.DETERMINISTIC, label, 0.9)
    return EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, None, None)


def test_temporal_divergence_alone_does_not_resolve_a_contradiction():
    # Recency is not evidence of supersession: an 8-year gap with no declared
    # lifecycle is an UNRESOLVED contradiction (formerly TEMPORAL_DIFF).
    result = _resolve(_dated_pair())
    assert result.relationship is EvidenceRelationship.UNRESOLVED
    assert "no source declares supersession" in result.rationale


def test_declared_superseded_source_resolves_as_temporal_diff():
    result = _resolve(_dated_pair(lc_a={"status": "superseded", "superseded_by": "doc_e2"},
                                  lc_b={"status": "current"}))
    assert result.relationship is EvidenceRelationship.TEMPORAL_DIFF
    assert "resolved by declared lifecycle: e1 is superseded" in result.rationale


def test_withdrawn_source_resolves_as_temporal_diff():
    result = _resolve(_dated_pair(lc_a={"status": "withdrawn"}))
    assert result.relationship is EvidenceRelationship.TEMPORAL_DIFF


def test_two_current_sources_that_disagree_are_a_genuine_conflict():
    result = _resolve(_dated_pair(lc_a={"status": "current"}, lc_b={"status": "current"}))
    assert result.relationship is EvidenceRelationship.GENUINE_CONFLICT


def test_two_stale_sources_are_not_resolved_by_lifecycle():
    result = _resolve(_dated_pair(lc_a={"status": "withdrawn"}, lc_b={"status": "historical"}))
    assert result.relationship is EvidenceRelationship.UNRESOLVED


def test_temporal_divergence_without_disagreement_is_benign_temporal_diff():
    result = _resolve(_dated_pair(), label=ValidationLabel.SUPPORTED)
    assert result.relationship is EvidenceRelationship.TEMPORAL_DIFF


def test_uncertain_everything_is_unresolved_never_fabricated_compatible():
    prov_a, prov_b = _matching_provenance(jurisdiction=UNKNOWN)
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.UNCERTAIN, 0.0)
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, None, None)
    assert result.relationship is EvidenceRelationship.UNRESOLVED


def test_nli_result_contradiction_participates_in_conflict_detection():
    prov_a, prov_b = _matching_provenance()
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.UNCERTAIN, 0.0)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.UNCERTAIN, 0.3)
    nli = ValidationResult(candidate_id="cand1", stage=ValidationStage.SELECTIVE_NLI,
                            label=ValidationLabel.CONTRADICTION, confidence=0.85,
                            nli_signal=NLISignal(label=NLILabel.CONTRADICTION, confidence=0.85))
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, ctx, nli)
    assert result.relationship is EvidenceRelationship.GENUINE_CONFLICT


def test_confidence_is_always_in_valid_range():
    prov_a, prov_b = _matching_provenance()
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.SUPPORTED, 0.98)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.SUPPORTED, 0.8)
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, ctx, None)
    assert 0.0 <= result.confidence <= 1.0


def test_contributing_claim_ids_trace_back_to_both_items():
    prov_a, prov_b = _matching_provenance()
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.SUPPORTED, 0.9)
    result = EvidenceResolver().resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, None, None)
    assert result.contributing_claim_ids == frozenset({ClaimId("ca"), ClaimId("cb")})


def test_resolution_is_deterministic_across_repeated_calls():
    prov_a, prov_b = _matching_provenance(population=UNKNOWN)
    bundle = _bundle(prov_a, prov_b)
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.CONTRADICTION, 0.9)
    resolver = EvidenceResolver()
    first = resolver.resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, None, None)
    second = resolver.resolve(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"), det, None, None)
    assert first.relationship == second.relationship
    assert first.confidence == second.confidence
    assert first.rationale == second.rationale