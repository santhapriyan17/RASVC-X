"""Internal decision-layer types. schemas/decision.py (Decision,
DecisionAction, CorrectiveTarget) is the frozen public contract this
module produces output for -- it is never redefined here.

DecisionReasonCode is package-internal: schemas/decision.py's Decision has
only a free-text `rationale: str | None` field (no structured reason-code
field exists in the frozen contract). Rather than add a field to a frozen
M1 schema without a proven integration break, this enum is embedded as a
stable, machine-parseable prefix in the rationale string
(engine.py's _rationale()).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DecisionReasonCode(str, Enum):
    HIGH_SEVERITY_CONTRADICTION = "high_severity_contradiction"
    SAFETY_CRITICAL_CONTRADICTED = "safety_critical_contradicted"
    SAFETY_CRITICAL_UNSUPPORTED = "safety_critical_unsupported"
    UNRESOLVED_CRITICAL_CONFLICT = "unresolved_critical_conflict"
    INSUFFICIENT_EVIDENCE_HIGH_RISK = "insufficient_evidence_high_risk"
    INVALID_CRITICAL_CITATION = "invalid_critical_citation"
    SEVERE_PROVENANCE_UNCERTAINTY = "severe_provenance_uncertainty"
    FAILED_REQUIRED_VERIFICATION = "failed_required_verification"
    MISSING_REQUIRED_VALIDATION = "missing_required_validation"
    LOW_CONFIDENCE = "low_confidence"
    PARTIAL_CLAIM_COVERAGE = "partial_claim_coverage"
    UNSUPPORTED_GENERATED_CLAIM = "unsupported_generated_claim"
    MEETS_ANSWER_THRESHOLD = "meets_answer_threshold"
    NO_EVIDENCE = "no_evidence"


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    """Confidence-value cut points, applied to the CALIBRATED score when
    available, else the raw score (never silently treated as equivalent --
    see engine.py). Risk-tier-specific overrides let a high-risk query
    require a strictly higher bar without duplicating the policy.
    """

    answer_min: float = 0.75
    warning_min: float = 0.55
    regenerate_min: float = 0.35
    # Below regenerate_min -> ABSTAIN.

    high_risk_answer_min: float = 0.85
    high_risk_warning_min: float = 0.65
    high_risk_regenerate_min: float = 0.45
    high_risk_threshold: float = 0.6  # RiskProfile.overall_risk_score >= this counts as high-risk

    def __post_init__(self) -> None:
        pairs = (
            (self.regenerate_min, self.warning_min),
            (self.warning_min, self.answer_min),
            (self.high_risk_regenerate_min, self.high_risk_warning_min),
            (self.high_risk_warning_min, self.high_risk_answer_min),
        )
        for lower, upper in pairs:
            if not 0.0 <= lower <= upper <= 1.0:
                raise ValueError(
                    f"DecisionThresholds must satisfy 0 <= lower <= upper <= 1, got {lower}, {upper}"
                )
        if not 0.0 <= self.high_risk_threshold <= 1.0:
            raise ValueError("high_risk_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SafetyGateResult:
    blocked: bool
    reason_code: DecisionReasonCode | None = None
    detail: str | None = None