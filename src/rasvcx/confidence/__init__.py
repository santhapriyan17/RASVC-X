from __future__ import annotations

from rasvcx.confidence.calibration import (
    Calibrator,
    CalibrationArtifact,
    CalibrationArtifactError,
    CalibrationMethod,
    CalibrationOutcome,
    fit_isotonic,
    fit_platt,
)
from rasvcx.confidence.estimator import EstimatorWeights, RawReliabilityEstimator
from rasvcx.confidence.features import FeatureCompleteness, FeatureExtractionResult, FeatureExtractor
from rasvcx.provenance.source_quality import SourceQualityScorer
from rasvcx.schemas.confidence import ConfidenceFeatures, ConfidenceScore
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.validation.verified_context import ValidationSummary
from rasvcx.verification import VerificationSummary

__all__ = [
    "Calibrator",
    "CalibrationArtifact",
    "CalibrationArtifactError",
    "CalibrationMethod",
    "CalibrationOutcome",
    "fit_isotonic",
    "fit_platt",
    "EstimatorWeights",
    "RawReliabilityEstimator",
    "FeatureCompleteness",
    "FeatureExtractionResult",
    "FeatureExtractor",
    "ConfidenceFeatures",
    "ConfidenceScore",
    "ConfidencePipeline",
]


class ConfidencePipeline:
    """Feature extraction -> raw estimation -> calibration, in one call.
    Never fits a calibrator; pass a fitted CalibrationArtifact, or leave
    calibrator unset for an always-uncalibrated raw score.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor | None = None,
        estimator: RawReliabilityEstimator | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self._feature_extractor = feature_extractor or FeatureExtractor(SourceQualityScorer())
        self._estimator = estimator or RawReliabilityEstimator()
        self._calibrator = calibrator or Calibrator(None)

    def compute(
        self,
        bundle: EvidenceBundle,
        validation_summary: ValidationSummary | None,
        verification_summary: VerificationSummary | None,
    ) -> tuple[CalibrationOutcome, FeatureCompleteness]:
        extraction = self._feature_extractor.extract(bundle, validation_summary, verification_summary)
        raw_score = self._estimator.estimate(extraction.features)
        outcome = self._calibrator.calibrate(raw_score)
        return outcome, extraction.completeness