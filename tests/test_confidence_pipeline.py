from __future__ import annotations

import pytest

from rasvcx.confidence import (
    Calibrator,
    CalibrationArtifact,
    CalibrationMethod,
    ConfidencePipeline,
    EstimatorWeights,
    FeatureExtractor,
    RawReliabilityEstimator,
    fit_isotonic,
    fit_platt,
)
from rasvcx.confidence.calibration import FEATURE_SCHEMA_VERSION
from rasvcx.schemas.common import ChunkId, EvidenceItemId, QueryId, SourceType
from rasvcx.schemas.confidence import ConfidenceFeatures, ConfidenceScore
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.validation import NLIService, NullNLIBackend, ValidationPipeline


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


def _bundle_with_items(n: int, **prov_overrides) -> EvidenceBundle:
    bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
    for i in range(n):
        bundle.add_evidence_item(
            EvidenceItem(
                item_id=EvidenceItemId(f"E{i}"), chunk_id=ChunkId(f"c{i}"), text=f"fact {i}",
                retrieval_score=0.8, provenance=_provenance(**prov_overrides),
            )
        )
    return bundle


class TestFeatureExtractor:
    def test_empty_bundle_yields_all_zero_features(self):
        bundle = _bundle_with_items(0)
        result = FeatureExtractor().extract(bundle, None, None)
        assert result.features.evidence_agreement == 0.0
        assert result.features.retrieval_quality == 0.0
        assert result.completeness.had_evidence is False

    def test_single_item_no_validation_run_is_not_penalized_as_uncertain(self):
        bundle = _bundle_with_items(1)
        result = FeatureExtractor().extract(bundle, None, None)
        assert result.completeness.had_validation is False

    def test_missing_validation_never_becomes_positive_agreement(self):
        bundle = _bundle_with_items(2)
        result = FeatureExtractor().extract(bundle, None, None)
        assert result.features.evidence_agreement == 0.0

    def test_missing_verification_never_becomes_positive_support(self):
        bundle = _bundle_with_items(1)
        result = FeatureExtractor().extract(bundle, None, None)
        assert result.features.claim_verification_support == 0.0

    def test_all_features_bounded_in_unit_interval(self):
        bundle = _bundle_with_items(5)
        result = FeatureExtractor().extract(bundle, None, None)
        for field in (
            "evidence_agreement", "claim_verification_support", "contradiction_penalty",
            "provenance_quality", "source_diversity", "retrieval_quality", "rerank_quality",
            "resolution_uncertainty_penalty",
        ):
            value = getattr(result.features, field)
            assert 0.0 <= value <= 1.0

    def test_out_of_range_retrieval_score_is_clamped_not_propagated(self):
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
        bundle.add_evidence_item(
            EvidenceItem(item_id=EvidenceItemId("E0"), chunk_id=ChunkId("c0"), text="x",
                         retrieval_score=5.0, provenance=_provenance())
        )
        result = FeatureExtractor().extract(bundle, None, None)
        assert result.features.retrieval_quality == 1.0


class TestRawReliabilityEstimator:
    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            EstimatorWeights(evidence_agreement=0.5, claim_verification_support=0.5,
                              contradiction_penalty=0.5, provenance_quality=0.0,
                              source_diversity=0.0, retrieval_quality=0.0, rerank_quality=0.0,
                              resolution_uncertainty_penalty=0.0)

    def test_estimate_never_calibrated(self):
        features = ConfidenceFeatures(evidence_agreement=0.9, claim_verification_support=0.9,
                                       contradiction_penalty=0.0, provenance_quality=0.9,
                                       source_diversity=0.5, retrieval_quality=0.9, rerank_quality=0.9,
                                       resolution_uncertainty_penalty=0.0)
        score = RawReliabilityEstimator().estimate(features)
        assert score.is_calibrated is False
        assert score.calibration_method is None

    def test_full_contradiction_and_uncertainty_floors_at_zero(self):
        features = ConfidenceFeatures(evidence_agreement=0.0, claim_verification_support=0.0,
                                       contradiction_penalty=1.0, provenance_quality=0.0,
                                       source_diversity=0.0, retrieval_quality=0.0, rerank_quality=0.0,
                                       resolution_uncertainty_penalty=1.0)
        score = RawReliabilityEstimator().estimate(features)
        assert score.value == 0.0

    def test_score_always_in_unit_interval(self):
        features = ConfidenceFeatures(evidence_agreement=1.0, claim_verification_support=1.0,
                                       contradiction_penalty=0.0, provenance_quality=1.0,
                                       source_diversity=1.0, retrieval_quality=1.0, rerank_quality=1.0,
                                       resolution_uncertainty_penalty=0.0)
        score = RawReliabilityEstimator().estimate(features)
        assert 0.0 <= score.value <= 1.0


class TestCalibration:
    @staticmethod
    def _features() -> ConfidenceFeatures:
        return ConfidenceFeatures(evidence_agreement=0.5, claim_verification_support=0.5,
                                   contradiction_penalty=0.0, provenance_quality=0.5,
                                   source_diversity=0.5, retrieval_quality=0.5, rerank_quality=0.5,
                                   resolution_uncertainty_penalty=0.0)

    def test_no_artifact_is_uncalibrated_and_preserves_raw_value(self):
        raw = ConfidenceScore(value=0.6, features=self._features(), is_calibrated=False)
        outcome = Calibrator(None).calibrate(raw)
        assert outcome.status == "uncalibrated"
        assert outcome.score.value == 0.6
        assert outcome.score.is_calibrated is False

    def test_feature_schema_version_mismatch_is_unavailable_not_a_crash(self):
        raw = ConfidenceScore(value=0.6, features=self._features(), is_calibrated=False)
        bad = CalibrationArtifact(method=CalibrationMethod.PLATT, version="v1",
                                   feature_schema_version="wrong", params=(1.0, 0.0), fitted_at="t")
        outcome = Calibrator(bad).calibrate(raw)
        assert outcome.status == "unavailable"
        assert outcome.score.value == 0.6

    def test_malformed_platt_params_rejected(self):
        raw = ConfidenceScore(value=0.6, features=self._features(), is_calibrated=False)
        bad = CalibrationArtifact(method=CalibrationMethod.PLATT, version="v1",
                                   feature_schema_version=FEATURE_SCHEMA_VERSION,
                                   params=(1.0, 2.0, 3.0), fitted_at="t")
        outcome = Calibrator(bad).calibrate(raw)
        assert outcome.status == "unavailable"

    def test_non_monotonic_isotonic_table_rejected(self):
        raw = ConfidenceScore(value=0.6, features=self._features(), is_calibrated=False)
        bad = CalibrationArtifact(method=CalibrationMethod.ISOTONIC, version="v1",
                                   feature_schema_version=FEATURE_SCHEMA_VERSION,
                                   params=(0.0, 0.5, 0.5, 0.1), fitted_at="t")
        outcome = Calibrator(bad).calibrate(raw)
        assert outcome.status == "unavailable"

    def test_isotonic_params_with_nan_rejected(self):
        raw = ConfidenceScore(value=0.6, features=self._features(), is_calibrated=False)
        bad = CalibrationArtifact(method=CalibrationMethod.ISOTONIC, version="v1",
                                   feature_schema_version=FEATURE_SCHEMA_VERSION,
                                   params=(0.0, 0.0, 0.5, float("nan")), fitted_at="t")
        outcome = Calibrator(bad).calibrate(raw)
        assert outcome.status == "unavailable"

    def test_nan_raw_score_is_rejected(self):
        with pytest.raises(ValueError):
            ConfidenceScore(value=float("nan"), features=self._features(), is_calibrated=False)

    def test_platt_output_bounded_in_unit_interval(self):
        raw_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        artifact = fit_platt(raw_scores, labels)
        for x in (0.0, 0.5, 1.0):
            raw = ConfidenceScore(value=x, features=self._features(), is_calibrated=False)
            outcome = Calibrator(artifact).calibrate(raw)
            assert 0.0 <= outcome.score.value <= 1.0

    def test_isotonic_fit_is_monotone_over_full_range(self):
        raw_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        labels = [0, 1, 0, 1, 1]
        artifact = fit_isotonic(raw_scores, labels)
        values = []
        for i in range(0, 101):
            x = i / 100
            raw = ConfidenceScore(value=x, features=self._features(), is_calibrated=False)
            values.append(Calibrator(artifact).calibrate(raw).score.value)
        assert all(values[i] <= values[i + 1] + 1e-9 for i in range(len(values) - 1))

    def test_fit_platt_requires_both_classes(self):
        with pytest.raises(ValueError):
            fit_platt([0.1, 0.2, 0.3], [1, 1, 1])

    def test_fit_platt_deterministic(self):
        raw_scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        labels = [0, 0, 1, 1, 1]
        a1 = fit_platt(raw_scores, labels)
        a2 = fit_platt(raw_scores, labels)
        assert a1.params == a2.params


class TestConfidencePipeline:
    def test_import_performs_no_model_loading(self):
        import importlib
        import rasvcx.confidence as pkg
        importlib.reload(pkg)
        assert hasattr(pkg, "ConfidencePipeline")

    def test_default_pipeline_is_always_uncalibrated(self):
        bundle = _bundle_with_items(1)
        outcome, _ = ConfidencePipeline().compute(bundle, None, None)
        assert outcome.status == "uncalibrated"
        assert outcome.score.is_calibrated is False

    def test_deterministic_across_repeated_calls(self):
        bundle = _bundle_with_items(3)
        pipeline = ConfidencePipeline()
        o1, _ = pipeline.compute(bundle, None, None)
        o2, _ = pipeline.compute(bundle, None, None)
        assert o1.score.value == o2.score.value

    def test_end_to_end_with_real_m8_validation_summary(self):
        bundle = _bundle_with_items(2)
        rp = bundle.risk_profile
        vp = ValidationPipeline(nli_service=NLIService(NullNLIBackend()))
        query = QueryRequest(query_id=QueryId("q1"), raw_text="x", normalized_text="x")
        vs = vp.run(bundle, query, rp)
        outcome, completeness = ConfidencePipeline().compute(bundle, vs, None)
        assert completeness.had_validation is True
        assert 0.0 <= outcome.score.value <= 1.0


class TestM10_1HardeningRegressions:
    """Regression tests for bugs found and fixed in the M10.1 forensic pass."""

    def test_nan_never_becomes_maximum_confidence(self):
        # Prior bug: max(0.0, min(1.0, nan)) == 1.0 in raw Python, so a
        # single malformed retrieval/rerank score could silently become
        # the highest possible confidence value.
        from rasvcx.confidence.features import FeatureExtractor
        assert FeatureExtractor._mean_clamped([float("nan")]) == 0.0
        assert FeatureExtractor._mean_clamped([float("inf")]) == 0.0
        assert FeatureExtractor._mean_clamped([float("-inf")]) == 0.0

    def test_nan_dropped_not_averaged_in(self):
        from rasvcx.confidence.features import FeatureExtractor
        # A NaN alongside valid values must not corrupt the mean of the
        # valid ones (dropped, not treated as 0 or propagated as NaN).
        assert FeatureExtractor._mean_clamped([float("nan"), 0.8]) == 0.8

    def test_estimator_weights_reject_negative_value(self):
        with pytest.raises(ValueError):
            EstimatorWeights(evidence_agreement=-0.1, claim_verification_support=0.35,
                              contradiction_penalty=0.2, provenance_quality=0.1,
                              source_diversity=0.05, retrieval_quality=0.15, rerank_quality=0.15,
                              resolution_uncertainty_penalty=0.1)

    def test_estimator_weights_reject_greater_than_one(self):
        with pytest.raises(ValueError):
            EstimatorWeights(evidence_agreement=1.5, claim_verification_support=-0.5,
                              contradiction_penalty=0.0, provenance_quality=0.0,
                              source_diversity=0.0, retrieval_quality=0.0, rerank_quality=0.0,
                              resolution_uncertainty_penalty=0.0)

    def test_estimator_weights_reject_nan(self):
        with pytest.raises(ValueError):
            EstimatorWeights(evidence_agreement=float("nan"), claim_verification_support=0.25,
                              contradiction_penalty=0.2, provenance_quality=0.1,
                              source_diversity=0.05, retrieval_quality=0.15, rerank_quality=0.15,
                              resolution_uncertainty_penalty=0.1)

    def test_calibration_artifact_rejects_nan_platt_params(self):
        bad = CalibrationArtifact(method=CalibrationMethod.PLATT, version="v1",
                                   feature_schema_version=FEATURE_SCHEMA_VERSION,
                                   params=(float("nan"), 0.0), fitted_at="t")
        outcome = Calibrator(bad).calibrate(
            ConfidenceScore(value=0.5, features=TestCalibration._features(), is_calibrated=False)
        )
        assert outcome.status == "unavailable"

    def test_calibration_artifact_rejects_inf_platt_params(self):
        bad = CalibrationArtifact(method=CalibrationMethod.PLATT, version="v1",
                                   feature_schema_version=FEATURE_SCHEMA_VERSION,
                                   params=(float("inf"), 0.0), fitted_at="t")
        outcome = Calibrator(bad).calibrate(
            ConfidenceScore(value=0.5, features=TestCalibration._features(), is_calibrated=False)
        )
        assert outcome.status == "unavailable"

    def test_fit_platt_rejects_non_binary_labels(self):
        with pytest.raises(ValueError):
            fit_platt([0.1, 0.2, 0.3], [-1, 0, 1])
        with pytest.raises(ValueError):
            fit_platt([0.1, 0.2, 0.3], [0, 2, 1])

    def test_fit_isotonic_rejects_non_binary_labels(self):
        with pytest.raises(ValueError):
            fit_isotonic([0.1, 0.2, 0.3], [0, 2, 1])

    def test_fit_platt_rejects_non_finite_raw_scores(self):
        with pytest.raises(ValueError):
            fit_platt([0.1, float("nan"), 0.9], [0, 0, 1])

    def test_single_source_evidence_is_neutral_not_high_agreement(self):
        # Prior bug: a single-source, no-M8-conflict-check-possible case
        # defaulted to evidence_agreement=0.9 -- indistinguishable from
        # genuine multi-source corroboration.
        bundle = _bundle_with_items(1)
        FeatureExtractor().extract(bundle, None, None)
        # No validation summary at all in this call -> agreement is 0.0
        # (see test_missing_validation_never_becomes_positive_agreement);
        # this test targets the internal helper directly for the
        # single-item-with-a-validation-summary-but-no-candidates case.
        from rasvcx.validation.verified_context import ValidationSummary
        fake_summary = ValidationSummary(
            candidates_generated=0, validation_results={}, resolutions=[], nli_calls_used=0
        )
        agreement, _, _, _ = FeatureExtractor._validation_signals(fake_summary, had_evidence=True)
        assert agreement == 0.5