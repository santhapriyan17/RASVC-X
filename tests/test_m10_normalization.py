"""M10.1 regression tests — positive-ceiling normalization fix.

These tests verify the mathematical correction in estimator.py that
normalizes the raw score by W_P (sum of positive weights) so that
perfect positive evidence with zero penalties produces raw = 1.0.

Test groups:
    A. Perfect features → raw = 1.0 → ANSWER through the full decision chain
    B. Strong-but-imperfect features → ANSWER reachable
    C. Penalty monotonicity
    D. Boundary clamping at 0 and 1
    E. Custom weights: W_P computed correctly for non-default configs
"""

from __future__ import annotations

import pytest

from rasvcx.confidence import (
    Calibrator,
    CalibrationOutcome,
    ConfidencePipeline,
    EstimatorWeights,
    FeatureCompleteness,
    FeatureExtractor,
    RawReliabilityEstimator,
)
from rasvcx.confidence.calibration import CalibrationMethod, CalibrationArtifact, FEATURE_SCHEMA_VERSION
from rasvcx.decision import DecisionAction, DecisionEngine, DecisionThresholds
from rasvcx.schemas.common import ChunkId, EvidenceItemId, QueryId, SourceType
from rasvcx.schemas.confidence import ConfidenceFeatures, ConfidenceScore
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth


# ── Helpers ───────────────────────────────────────────────────────────


def _features(**overrides) -> ConfidenceFeatures:
    base = dict(
        evidence_agreement=0.8,
        claim_verification_support=0.8,
        contradiction_penalty=0.0,
        provenance_quality=0.8,
        source_diversity=0.5,
        retrieval_quality=0.8,
        rerank_quality=0.8,
        resolution_uncertainty_penalty=0.0,
    )
    base.update(overrides)
    return ConfidenceFeatures(**base)


def _perfect_features() -> ConfidenceFeatures:
    return ConfidenceFeatures(
        evidence_agreement=1.0,
        claim_verification_support=1.0,
        contradiction_penalty=0.0,
        provenance_quality=1.0,
        source_diversity=1.0,
        retrieval_quality=1.0,
        rerank_quality=1.0,
        resolution_uncertainty_penalty=0.0,
    )


def _risk_profile(**overrides) -> RiskProfile:
    base = dict(
        overall_risk_score=0.3,
        feature_scores=RiskFeatureScores(),
        validation_depth=ValidationDepth.STANDARD,
        retrieval_retry_budget=1,
        nli_call_allowance=5,
    )
    base.update(overrides)
    return RiskProfile(**base)


def _outcome(value: float, features: ConfidenceFeatures | None = None,
             status: str = "uncalibrated") -> CalibrationOutcome:
    f = features or _features()
    score = ConfidenceScore(
        value=value, features=f, is_calibrated=(status == "calibrated"),
        calibration_method="platt" if status == "calibrated" else None,
    )
    return CalibrationOutcome(score=score, status=status)


def _completeness_all_present() -> FeatureCompleteness:
    return FeatureCompleteness(
        had_evidence=True,
        had_validation=True,
        had_verification=True,
        had_rerank_scores=True,
        had_provenance=True,
    )


def _bundle_with_item() -> EvidenceBundle:
    prov = Provenance(
        source_type=SourceType.PEER_REVIEWED_LITERATURE,
        date="2024-01-01", jurisdiction="US",
        population="adults", dosage_context="general",
    )
    bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
    bundle.add_evidence_item(
        EvidenceItem(
            item_id=EvidenceItemId("E1"), chunk_id=ChunkId("c1"),
            text="Evidence text.", retrieval_score=0.95, provenance=prov,
        )
    )
    return bundle


# ── A. Perfect features → ANSWER ──────────────────────────────────────


class TestPerfectFeaturesProduceAnswer:
    """With every positive feature at 1.0 and every penalty at 0.0 the
    normalized raw score must be exactly 1.0, and the decision engine
    must select ANSWER under the standard-risk configuration.
    """

    def test_perfect_raw_score_is_one(self):
        score = RawReliabilityEstimator().estimate(_perfect_features())
        assert score.value == 1.0

    def test_perfect_score_is_not_calibrated(self):
        score = RawReliabilityEstimator().estimate(_perfect_features())
        assert score.is_calibrated is False
        assert score.calibration_method is None

    def test_perfect_through_calibration_remains_valid(self):
        score = RawReliabilityEstimator().estimate(_perfect_features())
        outcome = Calibrator(None).calibrate(score)
        assert outcome.status == "uncalibrated"
        assert outcome.score.value == 1.0

    def test_perfect_score_exceeds_answer_threshold(self):
        t = DecisionThresholds()
        score = RawReliabilityEstimator().estimate(_perfect_features())
        assert score.value >= t.answer_min

    def test_perfect_score_exceeds_high_risk_answer_threshold(self):
        t = DecisionThresholds()
        score = RawReliabilityEstimator().estimate(_perfect_features())
        assert score.value >= t.high_risk_answer_min

    def test_decision_engine_produces_answer_standard_risk(self):
        features = _perfect_features()
        score = RawReliabilityEstimator().estimate(features)
        outcome = Calibrator(None).calibrate(score)
        engine = DecisionEngine()
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.3)
        completeness = _completeness_all_present()
        decision = engine.decide(outcome, bundle, rp, None, None, completeness)
        assert decision.action is DecisionAction.ANSWER

    def test_decision_engine_produces_answer_high_risk(self):
        features = _perfect_features()
        score = RawReliabilityEstimator().estimate(features)
        outcome = Calibrator(None).calibrate(score)
        engine = DecisionEngine()
        bundle = _bundle_with_item()
        rp = _risk_profile(overall_risk_score=0.9)
        completeness = _completeness_all_present()
        decision = engine.decide(outcome, bundle, rp, None, None, completeness)
        assert decision.action is DecisionAction.ANSWER


# ── B. Strong-but-imperfect → ANSWER reachable ────────────────────────


class TestStrongEvidenceCanReachAnswer:
    """Verify that realistic strong evidence now reaches ANSWER under
    standard-risk thresholds, which was impossible before the fix.
    """

    def test_strong_features_exceed_answer_min(self):
        features = ConfidenceFeatures(
            evidence_agreement=0.9,
            claim_verification_support=0.85,
            contradiction_penalty=0.0,
            provenance_quality=0.8,
            source_diversity=0.4,
            retrieval_quality=0.9,
            rerank_quality=0.88,
            resolution_uncertainty_penalty=0.0,
        )
        score = RawReliabilityEstimator().estimate(features)
        assert score.value >= DecisionThresholds().answer_min

    def test_answer_tolerates_small_penalty(self):
        """A small contradiction penalty should not prevent ANSWER when
        all positive features are near-perfect."""
        features = ConfidenceFeatures(
            evidence_agreement=1.0,
            claim_verification_support=1.0,
            contradiction_penalty=0.05,
            provenance_quality=1.0,
            source_diversity=1.0,
            retrieval_quality=1.0,
            rerank_quality=1.0,
            resolution_uncertainty_penalty=0.05,
        )
        score = RawReliabilityEstimator().estimate(features)
        assert score.value >= DecisionThresholds().answer_min

    def test_moderate_penalty_pushes_below_answer(self):
        """Heavy penalties should push the score below ANSWER threshold.
        With normalization, pre_norm = 0.70 - (0.20*0.8 + 0.10*0.6) = 0.47,
        normalized = 0.47/0.70 ≈ 0.671, which is below answer_min=0.75."""
        features = ConfidenceFeatures(
            evidence_agreement=1.0,
            claim_verification_support=1.0,
            contradiction_penalty=0.8,
            provenance_quality=1.0,
            source_diversity=1.0,
            retrieval_quality=1.0,
            rerank_quality=1.0,
            resolution_uncertainty_penalty=0.6,
        )
        score = RawReliabilityEstimator().estimate(features)
        assert score.value < DecisionThresholds().answer_min


# ── C. Penalty monotonicity ───────────────────────────────────────────


class TestPenaltyMonotonicity:
    """Increasing a penalty feature must strictly decrease confidence.
    Decreasing it must strictly increase confidence. This must hold
    after normalization.
    """

    def test_contradiction_penalty_is_monotonically_decreasing(self):
        estimator = RawReliabilityEstimator()
        prev_score = None
        for cp in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            score = estimator.estimate(_features(contradiction_penalty=cp))
            if prev_score is not None:
                assert score.value < prev_score, (
                    f"Contradiction penalty {cp}: score {score.value} must be "
                    f"strictly less than previous {prev_score}"
                )
            prev_score = score.value

    def test_resolution_uncertainty_is_monotonically_decreasing(self):
        estimator = RawReliabilityEstimator()
        prev_score = None
        for rup in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            score = estimator.estimate(_features(resolution_uncertainty_penalty=rup))
            if prev_score is not None:
                assert score.value < prev_score, (
                    f"Resolution uncertainty {rup}: score {score.value} must be "
                    f"strictly less than previous {prev_score}"
                )
            prev_score = score.value

    def test_positive_feature_is_monotonically_increasing(self):
        estimator = RawReliabilityEstimator()
        prev_score = None
        for ea in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            score = estimator.estimate(_features(evidence_agreement=ea))
            if prev_score is not None:
                assert score.value > prev_score, (
                    f"Evidence agreement {ea}: score {score.value} must be "
                    f"strictly greater than previous {prev_score}"
                )
            prev_score = score.value


# ── D. Boundary clamping ──────────────────────────────────────────────


class TestBoundaryClamping:
    """Score must always be in [0, 1] after normalization."""

    def test_perfect_features_clamp_at_one(self):
        score = RawReliabilityEstimator().estimate(_perfect_features())
        assert score.value == 1.0

    def test_worst_features_clamp_at_zero(self):
        features = ConfidenceFeatures(
            evidence_agreement=0.0,
            claim_verification_support=0.0,
            contradiction_penalty=1.0,
            provenance_quality=0.0,
            source_diversity=0.0,
            retrieval_quality=0.0,
            rerank_quality=0.0,
            resolution_uncertainty_penalty=1.0,
        )
        score = RawReliabilityEstimator().estimate(features)
        assert score.value == 0.0

    def test_all_zero_features_produce_zero(self):
        features = ConfidenceFeatures(
            evidence_agreement=0.0,
            claim_verification_support=0.0,
            contradiction_penalty=0.0,
            provenance_quality=0.0,
            source_diversity=0.0,
            retrieval_quality=0.0,
            rerank_quality=0.0,
            resolution_uncertainty_penalty=0.0,
        )
        score = RawReliabilityEstimator().estimate(features)
        assert score.value == 0.0

    def test_score_is_always_in_unit_interval(self):
        """Sweep a range of feature combinations and verify [0, 1]."""
        estimator = RawReliabilityEstimator()
        for ea in [0.0, 0.5, 1.0]:
            for cp in [0.0, 0.5, 1.0]:
                for rup in [0.0, 0.5, 1.0]:
                    features = _features(
                        evidence_agreement=ea,
                        contradiction_penalty=cp,
                        resolution_uncertainty_penalty=rup,
                    )
                    score = estimator.estimate(features)
                    assert 0.0 <= score.value <= 1.0, (
                        f"ea={ea}, cp={cp}, rup={rup} → {score.value}"
                    )


# ── E. Custom weights: W_P computed correctly ─────────────────────────


class TestCustomWeightsNormalization:
    """When non-default weights are used, W_P must be recomputed from
    the new positive weights, and the normalization must still produce
    1.0 for perfect positive features with zero penalties.
    """

    def test_custom_weights_perfect_features_yield_one(self):
        custom = EstimatorWeights(
            evidence_agreement=0.05,
            claim_verification_support=0.40,
            contradiction_penalty=0.10,
            provenance_quality=0.20,
            source_diversity=0.0,
            retrieval_quality=0.10,
            rerank_quality=0.10,
            resolution_uncertainty_penalty=0.05,
        )
        estimator = RawReliabilityEstimator(custom)
        score = estimator.estimate(_perfect_features())
        assert abs(score.value - 1.0) < 1e-9

    def test_custom_weights_w_positive_is_correct(self):
        custom = EstimatorWeights(
            evidence_agreement=0.05,
            claim_verification_support=0.40,
            contradiction_penalty=0.10,
            provenance_quality=0.20,
            source_diversity=0.0,
            retrieval_quality=0.10,
            rerank_quality=0.10,
            resolution_uncertainty_penalty=0.05,
        )
        estimator = RawReliabilityEstimator(custom)
        expected_wp = 0.05 + 0.40 + 0.20 + 0.0 + 0.10 + 0.10
        assert abs(estimator._w_positive - expected_wp) < 1e-9

    def test_default_weights_w_positive_is_0_70(self):
        estimator = RawReliabilityEstimator()
        assert abs(estimator._w_positive - 0.70) < 1e-9

    def test_custom_weights_penalty_still_reduces_score(self):
        custom = EstimatorWeights(
            evidence_agreement=0.10,
            claim_verification_support=0.30,
            contradiction_penalty=0.25,
            provenance_quality=0.15,
            source_diversity=0.05,
            retrieval_quality=0.05,
            rerank_quality=0.05,
            resolution_uncertainty_penalty=0.05,
        )
        estimator = RawReliabilityEstimator(custom)
        score_no_penalty = estimator.estimate(_features(contradiction_penalty=0.0))
        score_with_penalty = estimator.estimate(_features(contradiction_penalty=0.5))
        assert score_with_penalty.value < score_no_penalty.value