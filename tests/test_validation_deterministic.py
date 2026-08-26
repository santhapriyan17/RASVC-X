"""Tests for rasvcx.schemas.validation -- ValidationResult, NLISignal contracts.

Named test_validation_deterministic.py to match the pre-existing scaffolded
test file list; covers the schema layer (not the deterministic validator
implementation itself, which belongs to a later module).
"""

import pytest

from rasvcx.schemas.common import CandidateId, ClaimId, EvidenceRelationship, NLILabel
from rasvcx.schemas.validation import (
    EvidenceRelationshipResult,
    NLISignal,
    ValidationLabel,
    ValidationResult,
    ValidationStage,
)


def test_validation_result_construction():
    result = ValidationResult(
        candidate_id=CandidateId("c1"),
        stage=ValidationStage.DETERMINISTIC,
        label=ValidationLabel.SUPPORTED,
        confidence=0.95,
    )
    assert result.label is ValidationLabel.SUPPORTED
    assert result.confidence == 0.95


def test_nli_failure_maps_to_uncertain_never_supported():
    failure_result = ValidationResult(
        candidate_id=CandidateId("c1"),
        stage=ValidationStage.DETERMINISTIC,
        label=ValidationLabel.UNCERTAIN,
        confidence=0.0,
    )
    assert failure_result.label is ValidationLabel.UNCERTAIN
    assert failure_result.label is not ValidationLabel.SUPPORTED


def test_selective_nli_stage_requires_nli_signal():
    with pytest.raises(ValueError):
        ValidationResult(
            candidate_id=CandidateId("c1"),
            stage=ValidationStage.SELECTIVE_NLI,
            label=ValidationLabel.UNCERTAIN,
            confidence=0.0,
        )


def test_selective_nli_stage_with_signal_succeeds():
    sig = NLISignal(label=NLILabel.CONTRADICTION, confidence=0.87)
    result = ValidationResult(
        candidate_id=CandidateId("c1"),
        stage=ValidationStage.SELECTIVE_NLI,
        label=ValidationLabel.CONTRADICTION,
        confidence=0.87,
        nli_signal=sig,
    )
    assert result.nli_signal.label is NLILabel.CONTRADICTION


def test_confidence_never_discarded_even_when_uncertain():
    result = ValidationResult(
        candidate_id=CandidateId("c1"),
        stage=ValidationStage.CONTEXTUAL,
        label=ValidationLabel.UNCERTAIN,
        confidence=0.42,
    )
    assert result.confidence == 0.42


def test_validation_result_confidence_range_validation():
    with pytest.raises(ValueError):
        ValidationResult(
            candidate_id=CandidateId("c1"),
            stage=ValidationStage.DETERMINISTIC,
            label=ValidationLabel.SUPPORTED,
            confidence=1.5,
        )


def test_nli_signal_confidence_range_validation():
    with pytest.raises(ValueError):
        NLISignal(label=NLILabel.NEUTRAL, confidence=-0.1)


def test_evidence_relationship_result_uses_fixed_taxonomy():
    result = EvidenceRelationshipResult(
        candidate_id=CandidateId("c1"),
        relationship=EvidenceRelationship.POPULATION_DIFF,
        confidence=0.6,
        contributing_claim_ids=frozenset({ClaimId("claim1")}),
    )
    assert result.relationship is EvidenceRelationship.POPULATION_DIFF


def test_evidence_relationship_result_is_immutable():
    result = EvidenceRelationshipResult(
        candidate_id=CandidateId("c1"),
        relationship=EvidenceRelationship.COMPATIBLE,
        confidence=0.9,
        contributing_claim_ids=frozenset(),
    )
    with pytest.raises(Exception):
        result.confidence = 0.1


def test_evidence_relationship_result_confidence_range_validation():
    with pytest.raises(ValueError):
        EvidenceRelationshipResult(
            candidate_id=CandidateId("c1"),
            relationship=EvidenceRelationship.GENUINE_CONFLICT,
            confidence=2.0,
            contributing_claim_ids=frozenset(),
        )