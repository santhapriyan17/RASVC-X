"""Tests for rasvcx.schemas.decision -- Decision, DecisionAction, CorrectiveTarget."""

import pytest

from rasvcx.schemas.common import CandidateId
from rasvcx.schemas.decision import CorrectiveTarget, Decision, DecisionAction


def test_answer_decision_has_no_target():
    decision = Decision(action=DecisionAction.ANSWER, confidence=0.92)
    assert decision.corrective_target is None


def test_warning_decision_has_no_target():
    decision = Decision(action=DecisionAction.WARNING, confidence=0.6)
    assert decision.corrective_target is None


def test_abstain_decision_has_no_target():
    decision = Decision(action=DecisionAction.ABSTAIN, confidence=0.1)
    assert decision.corrective_target is None


def test_repair_must_target_generation():
    decision = Decision(
        action=DecisionAction.REPAIR,
        confidence=0.5,
        corrective_target=CorrectiveTarget.GENERATION,
    )
    assert decision.corrective_target is CorrectiveTarget.GENERATION


def test_repair_rejects_non_generation_target():
    with pytest.raises(ValueError):
        Decision(
            action=DecisionAction.REPAIR,
            confidence=0.5,
            corrective_target=CorrectiveTarget.RETRIEVAL,
        )


def test_regenerate_must_target_verified_context_or_retrieval():
    for target in (CorrectiveTarget.VERIFIED_CONTEXT, CorrectiveTarget.RETRIEVAL):
        decision = Decision(action=DecisionAction.REGENERATE, confidence=0.3, corrective_target=target)
        assert decision.corrective_target is target


def test_regenerate_rejects_generation_target():
    with pytest.raises(ValueError):
        Decision(
            action=DecisionAction.REGENERATE,
            confidence=0.3,
            corrective_target=CorrectiveTarget.GENERATION,
        )


def test_repair_missing_target_raises():
    with pytest.raises(ValueError):
        Decision(action=DecisionAction.REPAIR, confidence=0.5)


def test_regenerate_missing_target_raises():
    with pytest.raises(ValueError):
        Decision(action=DecisionAction.REGENERATE, confidence=0.5)


def test_non_corrective_action_with_target_raises():
    with pytest.raises(ValueError):
        Decision(
            action=DecisionAction.ANSWER,
            confidence=0.9,
            corrective_target=CorrectiveTarget.GENERATION,
        )


def test_decision_confidence_range_validation():
    with pytest.raises(ValueError):
        Decision(action=DecisionAction.ANSWER, confidence=1.4)


def test_decision_is_immutable():
    decision = Decision(action=DecisionAction.ANSWER, confidence=0.8)
    with pytest.raises(Exception):
        decision.confidence = 0.1


def test_decision_carries_contributing_candidate_ids():
    decision = Decision(
        action=DecisionAction.REGENERATE,
        confidence=0.3,
        corrective_target=CorrectiveTarget.RETRIEVAL,
        contributing_candidate_ids=frozenset({CandidateId("cand1")}),
    )
    assert decision.contributing_candidate_ids == frozenset({CandidateId("cand1")})