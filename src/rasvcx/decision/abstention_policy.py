"""Hard safety blockers (Section 8 of the M10 prompt). Evaluated before
any confidence threshold: a blocker here overrides ANSWER/WARNING/REPAIR
regardless of how high the calibrated score is (Invariant: many supported
claims must never wash out one severe contradiction).

Consumes M8 ValidationSummary and M9 VerificationSummary directly; never
re-derives claim/evidence/conflict logic that already belongs to them.
"""

from __future__ import annotations

from rasvcx.confidence.features import FeatureCompleteness
from rasvcx.decision.decision_types import DecisionReasonCode, DecisionThresholds, SafetyGateResult
from rasvcx.schemas.common import EvidenceRelationship
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.query import RiskProfile
from rasvcx.validation.verified_context import ValidationSummary
from rasvcx.verification import CitationStatus, SupportLabel, VerificationSummary

_CRITICAL_RELATIONSHIPS = frozenset({
    EvidenceRelationship.GENUINE_CONFLICT,
    EvidenceRelationship.UNRESOLVED,
})


class SafetyGate:
    def __init__(self, thresholds: DecisionThresholds | None = None) -> None:
        # Same threshold source as DecisionEngine (decision_types.py) --
        # never a second, independently-hardcoded risk cutoff.
        self._thresholds = thresholds or DecisionThresholds()

    def evaluate(
        self,
        bundle: EvidenceBundle,
        risk_profile: RiskProfile,
        validation_summary: ValidationSummary | None,
        verification_summary: VerificationSummary | None,
        completeness: FeatureCompleteness | None = None,
    ) -> SafetyGateResult:
        is_high_risk = (
            risk_profile.overall_risk_score >= self._thresholds.high_risk_threshold
            or risk_profile.safety_floor_forced
        )

        # A high, even calibrated, score computed WITHOUT a required
        # pipeline stage having run at all must never look equivalent to
        # one where that stage genuinely ran and found nothing wrong.
        # completeness distinguishes "ran, found no issue" (safe defaults
        # elsewhere in this class already handle that) from "never ran"
        # (which no downstream score can compensate for at high risk).
        if is_high_risk and completeness is not None:
            if not completeness.had_validation:
                return SafetyGateResult(
                    blocked=True,
                    reason_code=DecisionReasonCode.MISSING_REQUIRED_VALIDATION,
                    detail="M8 evidence validation did not run for a high-risk query",
                )
            if not completeness.had_verification:
                return SafetyGateResult(
                    blocked=True,
                    reason_code=DecisionReasonCode.FAILED_REQUIRED_VERIFICATION,
                    detail="M9 post-generation verification did not run for a high-risk query",
                )

        if verification_summary is not None and verification_summary.safety_critical_failure_count > 0:
            # safety_critical_failure_count (M9) counts CONTRADICTED
            # safety-critical claims specifically, not merely unsupported
            # ones -- the reason code must say what actually happened.
            return SafetyGateResult(
                blocked=True,
                reason_code=DecisionReasonCode.SAFETY_CRITICAL_CONTRADICTED,
                detail=(
                    f"{verification_summary.safety_critical_failure_count} safety-critical "
                    f"generated claim(s) contradicted"
                ),
            )

        if verification_summary is not None:
            citation_by_claim = {c.claim_id: c for c in verification_summary.citation_results}
            for result in verification_summary.claim_results:
                if result.label is not SupportLabel.CONTRADICTED:
                    continue
                citation = citation_by_claim.get(result.claim_id)
                if citation is not None and citation.status in (
                    CitationStatus.INCORRECT,
                    CitationStatus.CONTRADICTORY,
                ):
                    return SafetyGateResult(
                        blocked=True,
                        reason_code=DecisionReasonCode.INVALID_CRITICAL_CITATION,
                        detail=(
                            f"Claim {result.claim_id!r} is contradicted and its own citation "
                            f"is {citation.status.value}"
                        ),
                    )

        if validation_summary is not None:
            critical = [
                r for r in validation_summary.resolutions if r.relationship in _CRITICAL_RELATIONSHIPS
            ]
            if critical and is_high_risk:
                genuine = any(
                    r.relationship is EvidenceRelationship.GENUINE_CONFLICT for r in critical
                )
                return SafetyGateResult(
                    blocked=True,
                    reason_code=(
                        DecisionReasonCode.HIGH_SEVERITY_CONTRADICTION
                        if genuine
                        else DecisionReasonCode.UNRESOLVED_CRITICAL_CONFLICT
                    ),
                    detail=f"{len(critical)} unresolved/genuine-conflict evidence relationship(s) at high risk",
                )

        if is_high_risk and not bundle.evidence_items:
            return SafetyGateResult(
                blocked=True,
                reason_code=DecisionReasonCode.INSUFFICIENT_EVIDENCE_HIGH_RISK,
                detail="No evidence available for a high-risk query",
            )

        return SafetyGateResult(blocked=False)