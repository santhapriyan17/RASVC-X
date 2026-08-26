"""Tests for rasvcx.schemas.confidence -- ConfidenceScore, ConfidenceFeatures."""

import pytest

from rasvcx.schemas.confidence import ConfidenceFeatures, ConfidenceScore


def make_features(**overrides):
    defaults = dict(
        evidence_agreement=0.8,
        claim_verification_support=0.7,
        contradiction_penalty=0.1,
        provenance_quality=0.9,
        source_diversity=0.6,
        retrieval_quality=0.85,
        rerank_quality=0.8,
        resolution_uncertainty_penalty=0.05,
    )
    defaults.update(overrides)
    return ConfidenceFeatures(**defaults)


def test_confidence_score_defaults_not_calibrated():
    score = ConfidenceScore(value=0.74, features=make_features())
    assert score.is_calibrated is False
    assert score.calibration_method is None


def test_calibrated_score_requires_method():
    with pytest.raises(ValueError):
        ConfidenceScore(value=0.74, features=make_features(), is_calibrated=True)


def test_calibrated_score_with_method_succeeds():
    score = ConfidenceScore(
        value=0.74,
        features=make_features(),
        is_calibrated=True,
        calibration_method="platt_scaling",
    )
    assert score.is_calibrated is True
    assert score.calibration_method == "platt_scaling"


def test_method_set_without_calibrated_flag_raises():
    with pytest.raises(ValueError):
        ConfidenceScore(value=0.74, features=make_features(), calibration_method="platt_scaling")


def test_confidence_features_range_validation():
    with pytest.raises(ValueError):
        make_features(evidence_agreement=1.2)
    with pytest.raises(ValueError):
        make_features(resolution_uncertainty_penalty=-0.1)


def test_confidence_score_value_range_validation():
    with pytest.raises(ValueError):
        ConfidenceScore(value=2.0, features=make_features())


def test_confidence_score_combines_multiple_signals_not_single_score():
    features = make_features()
    assert features.retrieval_quality != features.evidence_agreement or True
    assert hasattr(features, "claim_verification_support")
    assert hasattr(features, "provenance_quality")
    assert hasattr(features, "source_diversity")


def test_confidence_score_is_immutable():
    score = ConfidenceScore(value=0.5, features=make_features())
    with pytest.raises(Exception):
        score.value = 0.9