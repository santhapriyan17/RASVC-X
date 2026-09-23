"""Converts a ConfidenceScore + upstream signals into the schemas/decision.py
Decision contract. Thresholds are applied to the CALIBRATED score when
calibration succeeded; otherwise the raw score is used and the rationale
says so explicitly (an uncalibrated score is not silently treated as
equally trustworthy).

Boundary convention: all thresholds are inclusive lower bounds (>=), so a
score exactly equal to a threshold takes the higher action. This is fixed
and deterministic -- no floating-point tie-breaking randomness.
"""

from __future__ import annotations

from rasvcx.confidence.calibration import CalibrationOutcome
from rasvcx.confidence.features import FeatureCompleteness
from rasvcx.decision.abstention_policy import SafetyGate
from rasvcx.decision.decision_types import DecisionReasonCode, DecisionThresholds
from rasvcx.schemas.confidence import ConfidenceFeatures
from rasvcx.schemas.decision import CorrectiveTarget, Decision, DecisionAction
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.query import RiskProfile
from rasvcx.validation.verified_context import ValidationSummary
from rasvcx.verification import SupportLabel, VerificationSummary


class DecisionEngine:
    def __init__(
        self,
        thresholds: DecisionThresholds | None = None,
        safety_gate: SafetyGate | None = None,
    ) -> None:
        self._thresholds = thresholds or DecisionThresholds()
        self._safety_gate = safety_gate or SafetyGate(self._thresholds)

    def decide(
        self,
        calibration_outcome: CalibrationOutcome,
        bundle: EvidenceBundle,
        risk_profile: RiskProfile,
        validation_summary: ValidationSummary | None,
        verification_summary: VerificationSummary | None,
        completeness: FeatureCompleteness | None = None,
    ) -> Decision:
        score = calibration_outcome.score

        gate = self._safety_gate.evaluate(
            bundle, risk_profile, validation_summary, verification_summary, completeness
        )
        if gate.blocked:
            return Decision(
                action=DecisionAction.ABSTAIN,
                confidence=score.value,
                rationale=self._rationale(gate.reason_code, gate.detail, calibration_outcome),
            )

        if not bundle.evidence_items:
            return Decision(
                action=DecisionAction.ABSTAIN,
                confidence=score.value,
                rationale=self._rationale(DecisionReasonCode.NO_EVIDENCE, None, calibration_outcome),
            )

        is_high_risk = risk_profile.overall_risk_score >= self._thresholds.high_risk_threshold
        answer_min, warning_min, regenerate_min = self._active_thresholds(is_high_risk)

        value = score.value
        if value >= answer_min:
            return Decision(
                action=DecisionAction.ANSWER,
                confidence=value,
                rationale=self._rationale(
                    DecisionReasonCode.MEETS_ANSWER_THRESHOLD, None, calibration_outcome
                ),
            )

        if value >= warning_min:
            reason = self._weak_support_reason(verification_summary)
            return Decision(
                action=DecisionAction.WARNING,
                confidence=value,
                rationale=self._rationale(reason, None, calibration_outcome),
            )

        if value >= regenerate_min:
            action, target = self._select_corrective(score.features)
            return Decision(
                action=action,
                confidence=value,
                corrective_target=target,
                rationale=self._rationale(
                    DecisionReasonCode.LOW_CONFIDENCE, None, calibration_outcome
                ),
            )

        return Decision(
            action=DecisionAction.ABSTAIN,
            confidence=value,
            rationale=self._rationale(DecisionReasonCode.LOW_CONFIDENCE, None, calibration_outcome),
        )

    def _active_thresholds(self, is_high_risk: bool) -> tuple[float, float, float]:
        t = self._thresholds
        if is_high_risk:
            return t.high_risk_answer_min, t.high_risk_warning_min, t.high_risk_regenerate_min
        return t.answer_min, t.warning_min, t.regenerate_min

    def _weak_support_reason(
        self, verification_summary: VerificationSummary | None
    ) -> DecisionReasonCode:
        if verification_summary is not None and any(
            r.label is SupportLabel.UNSUPPORTED for r in verification_summary.claim_results
        ):
            return DecisionReasonCode.UNSUPPORTED_GENERATED_CLAIM
        return DecisionReasonCode.PARTIAL_CLAIM_COVERAGE

    def _select_corrective(
        self, features: ConfidenceFeatures
    ) -> tuple[DecisionAction, CorrectiveTarget]:
        if features.retrieval_quality < 0.4 or features.evidence_agreement < 0.2:
            return DecisionAction.REGENERATE, CorrectiveTarget.RETRIEVAL
        if features.resolution_uncertainty_penalty > 0.5:
            return DecisionAction.REGENERATE, CorrectiveTarget.VERIFIED_CONTEXT
        return DecisionAction.REPAIR, CorrectiveTarget.GENERATION

    @staticmethod
    def _rationale(
        reason: DecisionReasonCode | None, detail: str | None, calibration: CalibrationOutcome
    ) -> str:
        parts = [reason.value if reason is not None else "unspecified"]
        if detail:
            parts.append(detail)
        parts.append(f"calibration={calibration.status}")
        return "; ".join(parts)