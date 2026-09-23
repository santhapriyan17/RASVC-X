from __future__ import annotations

import pytest

from rasvcx.confidence.calibration import CalibrationOutcome
from rasvcx.decision import DecisionAction, DecisionEngine, DecisionThresholds, SafetyGate
from rasvcx.schemas.common import ChunkId, EvidenceItemId, QueryId, SourceType
from rasvcx.schemas.confidence import ConfidenceFeatures, ConfidenceScore
from rasvcx.schemas.decision import CorrectiveTarget
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.validation import NLIService, NullNLIBackend, ValidationPipeline
from rasvcx.verification import VerificationPipeline


def _risk_profile(**overrides) -> RiskProfile:
    base = dict(
        overall_risk_score=0.3, feature_scores=RiskFeatureScores(),
        validation_depth=ValidationDepth.STANDARD, retrieval_retry_budget=1, nli_call_allowance=5,
    )
    base.update(overrides)
    return RiskProfile(**base)


def _provenance(**overrides) -> Provenance:
    base = dict(source_type=SourceType.PEER_REVIEWED_LITERATURE, date="2022-01-01",
                jurisdiction="US", population="adults", dosage_context="general")
    base.update(overrides)
    return Provenance(**base)


def _bundle_with_item(item_id: str = "E1", text: str = "x", **prov_overrides) -> EvidenceBundle:
    bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
    bundle.add_evidence_item(
        EvidenceItem(item_id=EvidenceItemId(item_id), chunk_id=ChunkId(f"ch-{item_id}"), text=text,
                     retrieval_score=0.9, provenance=_provenance(**prov_overrides))
    )
    return bundle


def _empty_bundle() -> EvidenceBundle:
    return EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())


def _features(**overrides) -> ConfidenceFeatures:
    base = dict(evidence_agreement=0.8, claim_verification_support=0.8, contradiction_penalty=0.0,
                provenance_quality=0.8, source_diversity=0.5, retrieval_quality=0.8, rerank_quality=0.8,
                resolution_uncertainty_penalty=0.0)
    base.update(overrides)
    return ConfidenceFeatures(**base)


def _outcome(value: float, status: str = "uncalibrated", **feature_overrides) -> CalibrationOutcome:
    features = _features(**feature_overrides)
    score = ConfidenceScore(value=value, features=features, is_calibrated=(status == "calibrated"),
                             calibration_method="platt" if status == "calibrated" else None)
    return CalibrationOutcome(score=score, status=status)


class TestSafetyGate:
    def test_no_evidence_at_high_risk_blocks(self):
        bundle = _empty_bundle()
        rp = _risk_profile(overall_risk_score=0.9)
        result = SafetyGate().evaluate(bundle, rp, None, None)
        assert result.blocked is True

    def test_no_evidence_at_low_risk_does_not_block_at_gate_level(self):
        bundle = _empty_bundle()
        rp = _risk_profile(overall_risk_score=0.1)
        result = SafetyGate().evaluate(bundle, rp, None, None)
        assert result.blocked is False

    def test_safety_critical_verification_failure_blocks_regardless_of_risk(self):
        bundle = _bundle_with_item(text="The recommended dose is 500 mg twice daily.")
        rp = _risk_profile(overall_risk_score=0.1)
        vp = ValidationPipeline(nli_service=NLIService(NullNLIBackend()))
        query = QueryRequest(query_id=QueryId("q1"), raw_text="x", normalized_text="x")
        vs = vp.run(bundle, query, rp)
        verify = VerificationPipeline().verify(
            "The recommended dose is 250 mg twice daily. [E1]", bundle, query=query, validation_summary=vs
        )
        result = SafetyGate().evaluate(bundle, rp, vs, verify)
        assert result.blocked is True


class TestDecisionEngineThresholds:
    def test_high_score_low_risk_is_answer(self):
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.1)
        engine = DecisionEngine()
        decision = engine.decide(_outcome(0.9), bundle, rp, None, None)
        assert decision.action is DecisionAction.ANSWER

    def test_moderate_score_is_warning(self):
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.1)
        engine = DecisionEngine()
        decision = engine.decide(_outcome(0.6), bundle, rp, None, None)
        assert decision.action is DecisionAction.WARNING

    def test_low_score_is_regenerate_or_repair(self):
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.1)
        engine = DecisionEngine()
        decision = engine.decide(_outcome(0.4), bundle, rp, None, None)
        assert decision.action in (DecisionAction.REGENERATE, DecisionAction.REPAIR)
        assert decision.corrective_target is not None

    def test_very_low_score_is_abstain(self):
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.1)
        engine = DecisionEngine()
        decision = engine.decide(_outcome(0.1), bundle, rp, None, None)
        assert decision.action is DecisionAction.ABSTAIN

    def test_high_risk_requires_stricter_threshold_for_answer(self):
        bundle = _bundle_with_item()
        rp_low = _risk_profile(overall_risk_score=0.1)
        rp_high = _risk_profile(overall_risk_score=0.9)
        engine = DecisionEngine()
        low_risk_decision = engine.decide(_outcome(0.78), bundle, rp_low, None, None)
        high_risk_decision = engine.decide(_outcome(0.78), bundle, rp_high, None, None)
        assert low_risk_decision.action is DecisionAction.ANSWER
        assert high_risk_decision.action is not DecisionAction.ANSWER

    def test_threshold_boundary_is_inclusive(self):
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.1)
        thresholds = DecisionThresholds()
        engine = DecisionEngine(thresholds=thresholds)
        decision = engine.decide(_outcome(thresholds.answer_min), bundle, rp, None, None)
        assert decision.action is DecisionAction.ANSWER

    def test_no_evidence_yields_abstain_even_at_perfect_score(self):
        bundle = _empty_bundle()
        rp = _risk_profile(overall_risk_score=0.1)
        engine = DecisionEngine()
        decision = engine.decide(_outcome(1.0), bundle, rp, None, None)
        assert decision.action is DecisionAction.ABSTAIN

    def test_repair_targets_generation_when_evidence_is_fine(self):
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.1)
        engine = DecisionEngine()
        decision = engine.decide(
            _outcome(0.4, retrieval_quality=0.9, evidence_agreement=0.9, resolution_uncertainty_penalty=0.0),
            bundle, rp, None, None,
        )
        assert decision.corrective_target is CorrectiveTarget.GENERATION

    def test_regenerate_targets_retrieval_when_retrieval_quality_is_poor(self):
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.1)
        engine = DecisionEngine()
        decision = engine.decide(
            _outcome(0.4, retrieval_quality=0.1, evidence_agreement=0.1), bundle, rp, None, None
        )
        assert decision.corrective_target is CorrectiveTarget.RETRIEVAL

    def test_rationale_reports_calibration_status(self):
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.1)
        engine = DecisionEngine()
        decision = engine.decide(_outcome(0.9, status="calibrated"), bundle, rp, None, None)
        assert "calibration=calibrated" in decision.rationale

    def test_deterministic_across_repeated_calls(self):
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.3)
        engine = DecisionEngine()
        d1 = engine.decide(_outcome(0.5), bundle, rp, None, None)
        d2 = engine.decide(_outcome(0.5), bundle, rp, None, None)
        assert d1.action == d2.action
        assert d1.confidence == d2.confidence
        assert d1.rationale == d2.rationale

    def test_confidence_slightly_out_of_range_rejected_by_schema(self):
        with pytest.raises(ValueError):
            ConfidenceScore(value=1.0001, features=_features(), is_calibrated=False)


class TestDecisionThresholdsValidation:
    def test_inverted_thresholds_rejected(self):
        with pytest.raises(ValueError):
            DecisionThresholds(answer_min=0.5, warning_min=0.7)


class TestM10_1HardeningRegressions:
    """Regression tests for bugs found and fixed in the M10.1 forensic pass."""

    def test_safety_critical_failure_reason_code_is_contradicted_not_unsupported(self):
        # Prior bug: safety_critical_failure_count (M9) counts CONTRADICTED
        # safety-critical claims, but the gate labelled it
        # SAFETY_CRITICAL_UNSUPPORTED -- a real semantic mislabel.
        from rasvcx.decision import DecisionReasonCode
        bundle = _bundle_with_item(text="The recommended dose is 500 mg twice daily.")
        rp = _risk_profile(overall_risk_score=0.1)
        vp = ValidationPipeline(nli_service=NLIService(NullNLIBackend()))
        query = QueryRequest(query_id=QueryId("q1"), raw_text="x", normalized_text="x")
        vs = vp.run(bundle, query, rp)
        verify = VerificationPipeline().verify(
            "The recommended dose is 250 mg twice daily. [E1]", bundle, query=query, validation_summary=vs
        )
        result = SafetyGate().evaluate(bundle, rp, vs, verify)
        assert result.reason_code is DecisionReasonCode.SAFETY_CRITICAL_CONTRADICTED
        assert result.reason_code is not DecisionReasonCode.SAFETY_CRITICAL_UNSUPPORTED

    def test_citation_blocker_requires_same_claim_association(self):
        # Prior bug: ANY invalid citation anywhere + ANY contradicted claim
        # anywhere triggered the blocker, even if they were unrelated claims.
        from rasvcx.schemas.verification import (
            CitationResult, CitationStatus, ClaimVerificationResult, SupportLabel,
            VerificationReasonCode, VerificationStage,
        )

        class _FakeVS:
            def __init__(self, claim_results, citation_results):
                self.claim_results = claim_results
                self.citation_results = citation_results
                self.safety_critical_failure_count = 0

        claim_a = ClaimVerificationResult(
            claim_id="A", label=SupportLabel.CONTRADICTED, confidence=0.9,
            stage=VerificationStage.DETERMINISTIC,
            reason_code=VerificationReasonCode.NUMERIC_MISMATCH, rationale="x",
        )
        claim_b = ClaimVerificationResult(
            claim_id="B", label=SupportLabel.SUPPORTED, confidence=0.9,
            stage=VerificationStage.DETERMINISTIC,
            reason_code=VerificationReasonCode.DIRECT_EVIDENCE_SUPPORT, rationale="x",
        )
        cite_a_ok = CitationResult(claim_id="A", status=CitationStatus.CORRECT,
                                    cited_item_ids=frozenset(), rationale="x")
        cite_b_bad = CitationResult(claim_id="B", status=CitationStatus.INCORRECT,
                                     cited_item_ids=frozenset(), rationale="x")

        rp = _risk_profile(overall_risk_score=0.1)
        bundle = _bundle_with_item()
        unrelated = _FakeVS([claim_a, claim_b], [cite_a_ok, cite_b_bad])
        result = SafetyGate().evaluate(bundle, rp, None, unrelated)
        assert result.blocked is False

        claim_c = ClaimVerificationResult(
            claim_id="C", label=SupportLabel.CONTRADICTED, confidence=0.9,
            stage=VerificationStage.DETERMINISTIC,
            reason_code=VerificationReasonCode.NUMERIC_MISMATCH, rationale="x",
        )
        cite_c_bad = CitationResult(claim_id="C", status=CitationStatus.INCORRECT,
                                     cited_item_ids=frozenset(), rationale="x")
        same_claim = _FakeVS([claim_c], [cite_c_bad])
        result2 = SafetyGate().evaluate(bundle, rp, None, same_claim)
        assert result2.blocked is True

    def test_decision_engine_and_safety_gate_share_one_high_risk_threshold(self):
        # Prior bug: abstention_policy.py hardcoded 0.6 independently of
        # DecisionThresholds.high_risk_threshold, so changing the
        # configured threshold silently left the safety gate stale.
        custom = DecisionThresholds(high_risk_threshold=0.2)
        engine = DecisionEngine(thresholds=custom)
        bundle = _empty_bundle()
        # 0.25 is above the custom 0.2 threshold but below the stale
        # hardcoded 0.6 that used to live in the safety gate -- this only
        # passes if the gate actually uses the configured value.
        rp = _risk_profile(overall_risk_score=0.25)
        decision = engine.decide(_outcome(1.0), bundle, rp, None, None)
        assert decision.action is DecisionAction.ABSTAIN


class TestM10_2FeatureCompletenessGate:
    """Regression tests: FeatureCompleteness must reach the safety gate.

    Prior gap: DecisionEngine.decide() had no way to know whether M8/M9
    actually ran versus simply being absent, so a sufficiently high raw
    score (achievable under a differently-weighted, still-valid
    EstimatorWeights configuration) could reach WARNING/ANSWER at high
    risk despite a required upstream stage never having executed.
    """

    def test_missing_validation_at_high_risk_hard_blocks_regardless_of_score(self):
        from rasvcx.confidence import FeatureCompleteness

        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.9)
        engine = DecisionEngine()
        completeness = FeatureCompleteness(
            had_evidence=True, had_validation=False, had_verification=True,
            had_rerank_scores=True, had_provenance=True,
        )
        # A near-perfect score would normally be ANSWER; completeness must
        # override it.
        decision = engine.decide(_outcome(0.95), bundle, rp, None, None, completeness)
        assert decision.action is DecisionAction.ABSTAIN
        assert "missing_required_validation" in decision.rationale

    def test_missing_verification_at_high_risk_hard_blocks(self):
        from rasvcx.confidence import FeatureCompleteness

        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.9)
        engine = DecisionEngine()
        completeness = FeatureCompleteness(
            had_evidence=True, had_validation=True, had_verification=False,
            had_rerank_scores=True, had_provenance=True,
        )
        decision = engine.decide(_outcome(0.95), bundle, rp, None, None, completeness)
        assert decision.action is DecisionAction.ABSTAIN
        assert "failed_required_verification" in decision.rationale

    def test_missing_stages_do_not_block_at_low_risk(self):
        # The hard block is specifically a high-risk conservatism
        # requirement (Section: "HIGH-RISK POLICY"), not a blanket rule
        # that a missing stage always aborts -- low-risk queries may
        # legitimately proceed on raw-score policy alone.
        from rasvcx.confidence import FeatureCompleteness

        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.1)
        engine = DecisionEngine()
        completeness = FeatureCompleteness(
            had_evidence=True, had_validation=False, had_verification=False,
            had_rerank_scores=True, had_provenance=True,
        )
        decision = engine.decide(_outcome(0.9), bundle, rp, None, None, completeness)
        assert decision.action is DecisionAction.ANSWER

    def test_complete_pipeline_at_high_risk_is_not_blocked_by_completeness(self):
        from rasvcx.confidence import FeatureCompleteness

        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.9)
        engine = DecisionEngine()
        completeness = FeatureCompleteness(
            had_evidence=True, had_validation=True, had_verification=True,
            had_rerank_scores=True, had_provenance=True,
        )
        decision = engine.decide(_outcome(0.95), bundle, rp, None, None, completeness)
        assert decision.action is DecisionAction.ANSWER

    def test_decide_without_completeness_argument_still_works(self):
        # Backward compatibility: completeness is optional, existing
        # call sites that don't pass it must not break.
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.9)
        engine = DecisionEngine()
        decision = engine.decide(_outcome(0.95), bundle, rp, None, None)
        assert decision.action is DecisionAction.ANSWER

    def test_real_world_escape_case_is_closed(self):
        # This reproduces the exact scenario found during forensic
        # testing: a legitimately different (still validly-summing,
        # still-bounded) EstimatorWeights configuration that upweights
        # verification/provenance/retrieval over evidence_agreement can
        # push a high-risk, M8-absent case to a raw score of 0.70 --
        # enough for WARNING under the default thresholds. The
        # completeness gate must still block it.
        from rasvcx.confidence import ConfidencePipeline, EstimatorWeights, RawReliabilityEstimator
        from rasvcx.schemas.common import ChunkId, EvidenceItemId, QueryId, SourceType
        from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
        from rasvcx.schemas.query import QueryRequest

        prov = Provenance(source_type=SourceType.PEER_REVIEWED_LITERATURE, date="2022-01-01",
                           jurisdiction="US", population="adults", dosage_context="general")
        rp = _risk_profile(overall_risk_score=0.9)
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=rp)
        bundle.add_evidence_item(
            EvidenceItem(item_id=EvidenceItemId("E1"), chunk_id=ChunkId("c1"),
                         text="RASVC-X achieved 98.7% accuracy.", retrieval_score=1.0,
                         provenance=prov, rerank_score=1.0)
        )
        query = QueryRequest(query_id=QueryId("q1"), raw_text="x", normalized_text="x")
        verify = VerificationPipeline().verify(
            "RASVC-X achieved 98.7% accuracy. [E1]", bundle, query=query, validation_summary=None
        )

        custom_weights = EstimatorWeights(
            evidence_agreement=0.05, claim_verification_support=0.40, contradiction_penalty=0.10,
            provenance_quality=0.20, source_diversity=0.0, retrieval_quality=0.10,
            rerank_quality=0.10, resolution_uncertainty_penalty=0.05,
        )
        cp = ConfidencePipeline(estimator=RawReliabilityEstimator(custom_weights))
        outcome, completeness = cp.compute(bundle, None, verify)
        assert outcome.score.value >= 0.55  # would clear WARNING without the gate

        engine = DecisionEngine()
        decision = engine.decide(outcome, bundle, rp, None, verify, completeness)
        assert decision.action is DecisionAction.ABSTAIN