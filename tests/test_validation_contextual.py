"""Tests for rasvcx.validation.contextual_validator.ContextualValidator.

Covers: matching context -> SUPPORTED, known mismatch -> PARTIAL, UNKNOWN
fields -> UNCERTAIN (never silently treated as a match), temporal
divergence, and repeated-invocation determinism.
"""

from __future__ import annotations

import pytest

from rasvcx.schemas.common import UNKNOWN, ChunkId, EvidenceItemId, QueryId, SourceType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.schemas.validation import ValidationLabel, ValidationStage
from rasvcx.validation.contextual_validator import ContextualValidator


def _risk_profile() -> RiskProfile:
    return RiskProfile(
        overall_risk_score=0.2,
        feature_scores=RiskFeatureScores(),
        validation_depth=ValidationDepth.STANDARD,
        retrieval_retry_budget=1,
        nli_call_allowance=5,
    )


def _bundle_with_items(prov_a: Provenance, prov_b: Provenance) -> EvidenceBundle:
    bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
    item_a = EvidenceItem(
        item_id=EvidenceItemId("e1"), chunk_id=ChunkId("c1"), text="Dose is 500 mg.",
        retrieval_score=0.9, provenance=prov_a,
    )
    item_b = EvidenceItem(
        item_id=EvidenceItemId("e2"), chunk_id=ChunkId("c2"), text="Dose is 250 mg.",
        retrieval_score=0.8, provenance=prov_b,
    )
    bundle.add_evidence_item(item_a)
    bundle.add_evidence_item(item_b)
    return bundle


def test_matching_known_context_is_supported():
    prov_a = Provenance(source_type=SourceType.DRUG_LABEL, date="2023-01-01",
                         jurisdiction="US", population="adults", dosage_context="general")
    prov_b = Provenance(source_type=SourceType.DRUG_LABEL, date="2023-02-01",
                         jurisdiction="US", population="adults", dosage_context="general")
    bundle = _bundle_with_items(prov_a, prov_b)
    result = ContextualValidator().validate(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"))
    assert result.stage is ValidationStage.CONTEXTUAL
    assert result.label is ValidationLabel.SUPPORTED
    assert 0.0 <= result.confidence <= 1.0


def test_known_population_mismatch_is_partial_not_contradiction():
    prov_a = Provenance(source_type=SourceType.DRUG_LABEL, date="2023-01-01",
                         jurisdiction="US", population="adults", dosage_context="general")
    prov_b = Provenance(source_type=SourceType.DRUG_LABEL, date="2023-01-01",
                         jurisdiction="US", population="pediatric", dosage_context="general")
    bundle = _bundle_with_items(prov_a, prov_b)
    result = ContextualValidator().validate(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"))
    assert result.label is ValidationLabel.PARTIAL
    assert result.label is not ValidationLabel.CONTRADICTION


@pytest.mark.parametrize("unknown_field", ["jurisdiction", "population", "dosage_context"])
def test_unknown_provenance_field_never_becomes_supported(unknown_field: str):
    base = dict(source_type=SourceType.DRUG_LABEL, date="2023-01-01",
                jurisdiction="US", population="adults", dosage_context="general")
    overridden = dict(base)
    overridden[unknown_field] = UNKNOWN
    prov_a = Provenance(**base)
    prov_b = Provenance(**overridden)
    bundle = _bundle_with_items(prov_a, prov_b)
    result = ContextualValidator().validate(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"))
    # UNKNOWN must never be silently treated as a compatible match.
    assert result.label is ValidationLabel.UNCERTAIN
    assert result.label is not ValidationLabel.SUPPORTED


def test_temporal_divergence_beyond_threshold_is_partial():
    prov_a = Provenance(source_type=SourceType.DRUG_LABEL, date="2015-01-01",
                         jurisdiction="US", population="adults", dosage_context="general")
    prov_b = Provenance(source_type=SourceType.DRUG_LABEL, date="2023-01-01",
                         jurisdiction="US", population="adults", dosage_context="general")
    bundle = _bundle_with_items(prov_a, prov_b)
    result = ContextualValidator().validate(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"))
    assert result.label is ValidationLabel.PARTIAL


def test_repeated_invocation_is_deterministic():
    prov_a = Provenance(source_type=SourceType.DRUG_LABEL, date="2023-01-01",
                         jurisdiction="US", population="adults", dosage_context=UNKNOWN)
    prov_b = Provenance(source_type=SourceType.DRUG_LABEL, date="2023-02-01",
                         jurisdiction="US", population="adults", dosage_context=UNKNOWN)
    bundle = _bundle_with_items(prov_a, prov_b)
    validator = ContextualValidator()
    first = validator.validate(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"))
    second = validator.validate(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"))
    assert first.label == second.label
    assert first.confidence == second.confidence
    assert first.rationale == second.rationale