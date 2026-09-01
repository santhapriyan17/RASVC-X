"""Tests for rasvcx.validation.selective_router.

Covers: accepting conclusive deterministic/contextual results without
escalation; escalating inconclusive, safety-critical pairs; the safety
floor being forced by EITHER claim-level Claim.is_safety_critical OR
RiskProfile.safety_floor_forced; low-risk non-safety-critical pairs never
escalating; and NLI-budget exhaustion never being silently upgraded to a
positive result.
"""

from __future__ import annotations

from rasvcx.schemas.common import ChunkId, ClaimId, EvidenceItemId, QueryId
from rasvcx.schemas.claims import Claim, ClaimSource, ClaimType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.schemas.validation import ValidationLabel, ValidationResult, ValidationStage
from rasvcx.validation.selective_router import (
    RoutingAction,
    SelectiveValidationRouter,
    candidate_is_safety_critical,
)


def _provenance() -> Provenance:
    return Provenance(source_type="drug_label", date="2023-01-01", jurisdiction="US",
                       population="adults", dosage_context="general")


def _bundle(is_safety_critical_a: bool, is_safety_critical_b: bool = False) -> EvidenceBundle:
    risk_profile = RiskProfile(
        overall_risk_score=0.3, feature_scores=RiskFeatureScores(),
        validation_depth=ValidationDepth.SHALLOW, retrieval_retry_budget=1, nli_call_allowance=2,
    )
    bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=risk_profile)
    claim_a = Claim(claim_id=ClaimId("ca"), text="Dose is 500 mg.", source=ClaimSource.EVIDENCE_EXTRACTION,
                     claim_type=ClaimType.DOSAGE, origin_chunk_id=ChunkId("ch1"),
                     is_safety_critical=is_safety_critical_a)
    claim_b = Claim(claim_id=ClaimId("cb"), text="Dose is 250 mg.", source=ClaimSource.EVIDENCE_EXTRACTION,
                     claim_type=ClaimType.DOSAGE, origin_chunk_id=ChunkId("ch2"),
                     is_safety_critical=is_safety_critical_b)
    bundle.register_claim(claim_a)
    bundle.register_claim(claim_b)
    item_a = EvidenceItem(item_id=EvidenceItemId("e1"), chunk_id=ChunkId("ch1"), text="Dose is 500 mg.",
                           retrieval_score=0.9, provenance=_provenance(),
                           extracted_claim_ids=frozenset({ClaimId("ca")}))
    item_b = EvidenceItem(item_id=EvidenceItemId("e2"), chunk_id=ChunkId("ch2"), text="Dose is 250 mg.",
                           retrieval_score=0.8, provenance=_provenance(),
                           extracted_claim_ids=frozenset({ClaimId("cb")}))
    bundle.add_evidence_item(item_a)
    bundle.add_evidence_item(item_b)
    return bundle


def _result(stage: ValidationStage, label: ValidationLabel, confidence: float) -> ValidationResult:
    return ValidationResult(candidate_id="cand1", stage=stage, label=label, confidence=confidence)


def _risk_profile(depth: ValidationDepth, allowance: int, safety_floor_forced: bool = False) -> RiskProfile:
    return RiskProfile(
        overall_risk_score=0.3, feature_scores=RiskFeatureScores(), validation_depth=depth,
        retrieval_retry_budget=1, nli_call_allowance=allowance, safety_floor_forced=safety_floor_forced,
    )


def test_candidate_is_safety_critical_detects_either_side():
    bundle = _bundle(is_safety_critical_a=True, is_safety_critical_b=False)
    assert candidate_is_safety_critical(bundle, EvidenceItemId("e1"), EvidenceItemId("e2")) is True
    bundle2 = _bundle(is_safety_critical_a=False, is_safety_critical_b=False)
    assert candidate_is_safety_critical(bundle2, EvidenceItemId("e1"), EvidenceItemId("e2")) is False


def test_conclusive_deterministic_result_accepted_without_escalation():
    bundle = _bundle(False, False)
    router = SelectiveValidationRouter()
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.CONTRADICTION, 0.9)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.SUPPORTED, 0.8)
    decision = router.route(
        bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"),
        det, ctx, _risk_profile(ValidationDepth.DEEP, 5), nli_available=True, nli_calls_used=0,
    )
    assert decision.action is RoutingAction.ACCEPT_DETERMINISTIC


def test_low_risk_non_safety_critical_inconclusive_pair_does_not_escalate():
    bundle = _bundle(False, False)
    router = SelectiveValidationRouter()
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.UNCERTAIN, 0.0)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.UNCERTAIN, 0.3)
    decision = router.route(
        bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"),
        det, ctx, _risk_profile(ValidationDepth.SHALLOW, 5), nli_available=True, nli_calls_used=0,
    )
    assert decision.action is not RoutingAction.ESCALATE_TO_NLI
    assert decision.is_safety_critical is False


def test_safety_critical_claim_escalates_even_under_shallow_depth():
    bundle = _bundle(True, False)
    router = SelectiveValidationRouter()
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.UNCERTAIN, 0.0)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.UNCERTAIN, 0.3)
    decision = router.route(
        bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"),
        det, ctx, _risk_profile(ValidationDepth.SHALLOW, 5), nli_available=True, nli_calls_used=0,
    )
    assert decision.action is RoutingAction.ESCALATE_TO_NLI
    assert decision.is_safety_critical is True


def test_risk_profile_safety_floor_forced_escalates_even_without_safety_critical_claims():
    # No claim is individually flagged is_safety_critical, but the
    # query-time risk router forced the safety floor -- this must still
    # force NLI eligibility (RiskProfile.safety_floor_forced is an
    # independent, OR'd signal, not overridden by claim-level state).
    bundle = _bundle(False, False)
    router = SelectiveValidationRouter()
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.UNCERTAIN, 0.0)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.UNCERTAIN, 0.3)
    decision = router.route(
        bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"),
        det, ctx, _risk_profile(ValidationDepth.SHALLOW, 5, safety_floor_forced=True),
        nli_available=True, nli_calls_used=0,
    )
    assert decision.action is RoutingAction.ESCALATE_TO_NLI


def test_nli_budget_exhausted_never_fabricates_a_positive_result():
    bundle = _bundle(True, False)
    router = SelectiveValidationRouter()
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.UNCERTAIN, 0.0)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.UNCERTAIN, 0.3)
    decision = router.route(
        bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"),
        det, ctx, _risk_profile(ValidationDepth.DEEP, 1), nli_available=True, nli_calls_used=1,
    )
    assert decision.action is RoutingAction.SKIP_NLI_BUDGET_EXHAUSTED


def test_nli_unavailable_never_fabricates_escalation():
    bundle = _bundle(True, False)
    router = SelectiveValidationRouter()
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.UNCERTAIN, 0.0)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.UNCERTAIN, 0.3)
    decision = router.route(
        bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"),
        det, ctx, _risk_profile(ValidationDepth.DEEP, 5), nli_available=False, nli_calls_used=0,
    )
    assert decision.action is RoutingAction.SKIP_NLI_BUDGET_EXHAUSTED


def test_routing_is_deterministic_across_repeated_calls():
    bundle = _bundle(True, False)
    router = SelectiveValidationRouter()
    det = _result(ValidationStage.DETERMINISTIC, ValidationLabel.PARTIAL, 0.4)
    ctx = _result(ValidationStage.CONTEXTUAL, ValidationLabel.UNCERTAIN, 0.3)
    rp = _risk_profile(ValidationDepth.STANDARD, 5)
    first = router.route(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"),
                          det, ctx, rp, nli_available=True, nli_calls_used=0)
    second = router.route(bundle, "cand1", EvidenceItemId("e1"), EvidenceItemId("e2"),
                           det, ctx, rp, nli_available=True, nli_calls_used=0)
    assert first.action == second.action
    assert first.reason == second.reason