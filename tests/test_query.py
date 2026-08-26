"""Tests for rasvcx.schemas.query -- QueryRequest and RiskProfile."""

import pytest

from rasvcx.schemas.common import QueryId
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth


def make_scores(**overrides):
    defaults = dict(
        numeric_content_score=0.5,
        unit_sensitive_score=0.5,
        safety_critical_structure_score=0.5,
        query_complexity_score=0.5,
        ambiguity_score=0.5,
    )
    defaults.update(overrides)
    return RiskFeatureScores(**defaults)


def test_query_request_construction():
    qr = QueryRequest(query_id=QueryId("q1"), raw_text=" hello ", normalized_text="hello")
    assert qr.raw_text == " hello "
    assert qr.normalized_text == "hello"


def test_query_request_rejects_empty_raw_text():
    with pytest.raises(ValueError):
        QueryRequest(query_id=QueryId("q1"), raw_text="   ", normalized_text="x")


def test_query_request_rejects_empty_normalized_text():
    with pytest.raises(ValueError):
        QueryRequest(query_id=QueryId("q1"), raw_text="hello", normalized_text="  ")


def test_risk_feature_scores_range_validation():
    make_scores()  # should not raise
    with pytest.raises(ValueError):
        make_scores(numeric_content_score=1.5)
    with pytest.raises(ValueError):
        make_scores(ambiguity_score=-0.1)


def test_risk_profile_construction():
    scores = make_scores(safety_critical_structure_score=1.0)
    profile = RiskProfile(
        overall_risk_score=0.75,
        feature_scores=scores,
        validation_depth=ValidationDepth.DEEP,
        retrieval_retry_budget=2,
        nli_call_allowance=10,
        safety_floor_forced=True,
    )
    assert profile.validation_depth is ValidationDepth.DEEP
    assert profile.safety_floor_forced is True


def test_risk_profile_is_immutable():
    profile = RiskProfile(
        overall_risk_score=0.5,
        feature_scores=make_scores(),
        validation_depth=ValidationDepth.STANDARD,
        retrieval_retry_budget=1,
        nli_call_allowance=5,
    )
    with pytest.raises(Exception):
        profile.overall_risk_score = 0.9


def test_risk_profile_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        RiskProfile(
            overall_risk_score=1.2,
            feature_scores=make_scores(),
            validation_depth=ValidationDepth.SHALLOW,
            retrieval_retry_budget=0,
            nli_call_allowance=0,
        )


def test_risk_profile_rejects_negative_budgets():
    with pytest.raises(ValueError):
        RiskProfile(
            overall_risk_score=0.5,
            feature_scores=make_scores(),
            validation_depth=ValidationDepth.SHALLOW,
            retrieval_retry_budget=-1,
            nli_call_allowance=0,
        )
    with pytest.raises(ValueError):
        RiskProfile(
            overall_risk_score=0.5,
            feature_scores=make_scores(),
            validation_depth=ValidationDepth.SHALLOW,
            retrieval_retry_budget=0,
            nli_call_allowance=-1,
        )


def test_risk_profile_safety_floor_defaults_false():
    profile = RiskProfile(
        overall_risk_score=0.1,
        feature_scores=make_scores(),
        validation_depth=ValidationDepth.SHALLOW,
        retrieval_retry_budget=0,
        nli_call_allowance=0,
    )
    assert profile.safety_floor_forced is False