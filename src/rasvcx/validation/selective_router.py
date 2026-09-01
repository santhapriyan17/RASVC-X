"""Selective validation routing (Section 12).

Decides whether a candidate pair's inconclusive deterministic/contextual
result should be escalated to NLI, and enforces the safety floor: queries
with safety-critical claims must remain *eligible* for the strongest
available validation regardless of the (advisory-only) RiskProfile
produced by routing/risk_router.py.

Per routing/risk_router.py's own docstring, this module must not rely
solely on ``RiskProfile.safety_floor_forced`` -- it re-derives
safety-critical eligibility directly from the claims linked to the
candidate's evidence items (``Claim.is_safety_critical``, set
deterministically by claims/extractor.py).

The routing policy itself is deterministic and explainable: given the same
inputs it always makes the same decision, and ``RoutingDecision.reason``
records why.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rasvcx.schemas.claims import Claim
from rasvcx.schemas.common import CandidateId, EvidenceItemId
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.query import RiskProfile, ValidationDepth
from rasvcx.schemas.validation import ValidationLabel, ValidationResult
from rasvcx.validation.validation_types import SelectiveRoutingConfig


class RoutingAction(str, Enum):
    """What the selective router decided to do with a candidate pair."""

    ACCEPT_DETERMINISTIC = "accept_deterministic"
    ACCEPT_CONTEXTUAL = "accept_contextual"
    ESCALATE_TO_NLI = "escalate_to_nli"
    SKIP_NLI_BUDGET_EXHAUSTED = "skip_nli_budget_exhausted"


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    action: RoutingAction
    is_safety_critical: bool
    reason: str


def candidate_is_safety_critical(
    bundle: EvidenceBundle, item_id_a: EvidenceItemId, item_id_b: EvidenceItemId
) -> bool:
    """True if any claim linked to either evidence item is safety-critical.

    Re-derives eligibility from claims/evidence directly (per the module
    docstring above) rather than trusting ``RiskProfile.safety_floor_forced``
    alone.
    """
    for item_id in (item_id_a, item_id_b):
        item = bundle.evidence_items.get(item_id)
        if item is None:
            continue
        for claim_id in item.extracted_claim_ids:
            claim: Claim = bundle.claims.get(claim_id)
            if claim.is_safety_critical:
                return True
    return False


def _is_inconclusive(result: ValidationResult, threshold: float) -> bool:
    if result.label in (ValidationLabel.UNCERTAIN, ValidationLabel.PARTIAL):
        return True
    return result.confidence < threshold


class SelectiveValidationRouter:
    """Routes candidate pairs to the appropriate validation depth.

    Policy (evaluated in order):
      1. If the deterministic result is already conclusive (SUPPORTED or
         CONTRADICTION at/above the confidence threshold), accept it --
         no need to spend contextual/NLI budget.
      2. Otherwise, if the contextual result is conclusive, accept it.
      3. Otherwise the pair is inconclusive. Escalate to NLI only if:
           a. an NLI backend is available, and
           b. the remaining NLI budget (RiskProfile.nli_call_allowance
              minus calls already made) is > 0, and
           c. the pair is safety-critical (by claim-level re-derivation
              OR risk_profile.safety_floor_forced) OR the RiskProfile's
              advisory validation_depth is STANDARD/DEEP (SHALLOW +
              non-safety-critical claims do not warrant the expensive
              path).
      4. If escalation is warranted but the NLI budget is exhausted, the
         result stays UNCERTAIN (never fabricated as SUPPORTED) -- this is
         a safe, explicit degradation, not a silent skip.

    Safety-critical pairs are never silently downgraded: they are always
    *eligible* for step 3 even under a SHALLOW risk tier, subject only to
    NLI availability and budget (never to a fabricated relaxation of the
    inconclusiveness bar).
    """

    def __init__(self, config: SelectiveRoutingConfig | None = None) -> None:
        self._config = config or SelectiveRoutingConfig()

    def route(
        self,
        bundle: EvidenceBundle,
        candidate_id: CandidateId,
        item_id_a: EvidenceItemId,
        item_id_b: EvidenceItemId,
        deterministic_result: ValidationResult,
        contextual_result: ValidationResult | None,
        risk_profile: RiskProfile,
        nli_available: bool,
        nli_calls_used: int,
    ) -> RoutingDecision:
        threshold = self._config.inconclusive_confidence_threshold
        is_safety_critical = candidate_is_safety_critical(bundle, item_id_a, item_id_b)

        if not _is_inconclusive(deterministic_result, threshold):
            return RoutingDecision(
                action=RoutingAction.ACCEPT_DETERMINISTIC,
                is_safety_critical=is_safety_critical,
                reason=(
                    f"Deterministic result conclusive: label={deterministic_result.label.value} "
                    f"confidence={deterministic_result.confidence:.2f}"
                ),
            )

        if contextual_result is not None and not _is_inconclusive(contextual_result, threshold):
            return RoutingDecision(
                action=RoutingAction.ACCEPT_CONTEXTUAL,
                is_safety_critical=is_safety_critical,
                reason=(
                    f"Contextual result conclusive: label={contextual_result.label.value} "
                    f"confidence={contextual_result.confidence:.2f}"
                ),
            )

        depth_warrants_nli = risk_profile.validation_depth in (
            ValidationDepth.STANDARD,
            ValidationDepth.DEEP,
        )
        # Two independent safety-floor signals are ORed, never ANDed:
        # risk_profile.safety_floor_forced is the query-time advisory signal
        # (routing/risk_router.py, computed before any evidence/claims
        # exist); is_safety_critical is the post-extraction, claim-level
        # re-derivation this module is required to make on its own (see
        # module docstring). Either one is sufficient to force eligibility;
        # neither is trusted to the exclusion of the other.
        eligible_for_nli = (
            is_safety_critical or risk_profile.safety_floor_forced or depth_warrants_nli
        )

        if not eligible_for_nli:
            return RoutingDecision(
                action=RoutingAction.ACCEPT_CONTEXTUAL
                if contextual_result is not None
                else RoutingAction.ACCEPT_DETERMINISTIC,
                is_safety_critical=is_safety_critical,
                reason=(
                    "Inconclusive but low-risk and not safety-critical; "
                    "avoiding unnecessary NLI cost"
                ),
            )

        remaining_budget = risk_profile.nli_call_allowance - nli_calls_used
        if not nli_available or remaining_budget <= 0:
            return RoutingDecision(
                action=RoutingAction.SKIP_NLI_BUDGET_EXHAUSTED,
                is_safety_critical=is_safety_critical,
                reason=(
                    "Escalation warranted but "
                    + ("no NLI backend available" if not nli_available else "NLI budget exhausted")
                ),
            )

        return RoutingDecision(
            action=RoutingAction.ESCALATE_TO_NLI,
            is_safety_critical=is_safety_critical,
            reason="Inconclusive deterministic/contextual result eligible for NLI escalation",
        )