"""Selective semantic verification (Section 23 of master prompt; NLI
guidance from the continuation prompt).

Reuses M8's ``NLIService``/``NLIBackend`` abstraction directly rather than
duplicating it -- this module never defines its own NLI backend, model
loading, or failure-handling logic; ``NLIService.predict_safe`` already
provides the "never raises, never fabricates a signal on failure"
contract this module depends on.

Escalation is selective, matching M8's ``selective_router.py`` philosophy:
NLI is not invoked for every claim, only when deterministic verification
was inconclusive (returned ``None``), and is skipped entirely once the
per-answer NLI budget is exhausted.

Hard invariants enforced here:
  - NLI unavailable/failed => UNCERTAIN, never SUPPORTED (Invariant 2).
  - Budget exhausted => NOT_VERIFIABLE with an explicit reason code, never
    a fabricated verdict (Section 38).
  - High NLI confidence never overrides an already-conclusive
    deterministic signal -- this module is only ever consulted when the
    deterministic stage returned None (Section 26/65).
"""

from __future__ import annotations

from rasvcx.schemas.common import NLILabel
from rasvcx.schemas.verification import (
    ClaimVerificationResult,
    GeneratedClaim,
    SupportLabel,
    VerificationReasonCode,
    VerificationStage,
)
from rasvcx.validation.deterministic import significant_tokens
from rasvcx.validation.nli_interface import NLIService

_NLI_LABEL_TO_SUPPORT_LABEL: dict[NLILabel, SupportLabel] = {
    NLILabel.ENTAILMENT: SupportLabel.SUPPORTED,
    NLILabel.CONTRADICTION: SupportLabel.CONTRADICTED,
    # Neutral confirms neither support nor contradiction -- conservative
    # mapping to UNCERTAIN, matching M8's identical policy for the same
    # ambiguous NLI outcome (verified_context.py).
    NLILabel.NEUTRAL: SupportLabel.UNCERTAIN,
}


class SelectiveSemanticVerifier:
    """Escalates a deterministically-inconclusive claim to NLI, subject to
    a per-answer call budget.

    Complexity: O(1) NLI calls per claim it is actually invoked for (never
    invoked when the deterministic stage was already conclusive).
    """

    def __init__(self, nli_service: NLIService) -> None:
        self._nli_service = nli_service

    def verify(
        self,
        claim: GeneratedClaim,
        candidate_evidence_texts: dict[str, str],
        calls_used: int,
        call_budget: int,
    ) -> tuple[ClaimVerificationResult, int]:
        """Attempt semantic verification of *claim* against the best
        available evidence text.

        Returns ``(result, calls_made)`` -- ``calls_made`` is 0 or 1 and is
        the caller's responsibility to add to its running budget count
        (this method does not mutate any shared state itself).
        """
        if calls_used >= call_budget:
            return (
                self._budget_exhausted_result(claim),
                0,
            )

        premise, item_id = self._select_best_evidence_text(claim, candidate_evidence_texts)
        if premise is None:
            return (
                ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    label=SupportLabel.UNSUPPORTED,
                    confidence=0.85,
                    stage=VerificationStage.SELECTIVE_NLI,
                    reason_code=VerificationReasonCode.NO_EVIDENCE,
                    rationale="No candidate evidence text available for semantic comparison",
                ),
                0,
            )

        signal = self._nli_service.predict_safe(premise=premise, hypothesis=claim.text)
        if signal is None:
            # Graceful degradation (Invariant 2): NLI unavailable/failed
            # never becomes SUPPORTED, and is not silently retried.
            return (
                ClaimVerificationResult(
                    claim_id=claim.claim_id,
                    label=SupportLabel.UNCERTAIN,
                    confidence=0.0,
                    stage=VerificationStage.SELECTIVE_NLI,
                    reason_code=VerificationReasonCode.NLI_UNAVAILABLE,
                    rationale="NLI backend unavailable or failed; no semantic signal obtained",
                ),
                0,
            )

        label = _NLI_LABEL_TO_SUPPORT_LABEL[signal.label]
        if label is SupportLabel.CONTRADICTED:
            reason = VerificationReasonCode.EVIDENCE_CONTRADICTION
        elif label is SupportLabel.UNCERTAIN:
            reason = VerificationReasonCode.NLI_UNCERTAIN
        else:
            reason = VerificationReasonCode.DIRECT_EVIDENCE_SUPPORT

        item_ids = frozenset({item_id}) if item_id is not None else frozenset()
        return (
            ClaimVerificationResult(
                claim_id=claim.claim_id,
                label=label,
                confidence=signal.confidence,
                stage=VerificationStage.SELECTIVE_NLI,
                reason_code=reason,
                rationale=f"NLI backend returned {signal.label.value} (confidence={signal.confidence:.2f})",
                supporting_item_ids=item_ids if label is SupportLabel.SUPPORTED else frozenset(),
                contradicting_item_ids=item_ids if label is SupportLabel.CONTRADICTED else frozenset(),
            ),
            1,
        )

    # -- internal helpers -----------------------------------------------

    def _select_best_evidence_text(
        self, claim: GeneratedClaim, candidate_evidence_texts: dict[str, str]
    ) -> tuple[str | None, str | None]:
        """Rank among *given* candidates by lexical overlap; does not
        itself decide relevance -- the caller (verdict_aggregator) is
        responsible for scoping ``candidate_evidence_texts`` to items
        already worth comparing (via citation or deterministic_check's
        lexical fallback). Rejecting a candidate here purely on low
        overlap would defeat the purpose of semantic verification, which
        exists precisely for cases where wording differs substantially
        (Section 23: "wording differs substantially" is itself a listed
        escalation trigger, not a disqualifying signal).
        """
        if not candidate_evidence_texts:
            return None, None

        claim_tokens = significant_tokens(claim.normalized_text)
        best_item_id: str | None = None
        best_text: str | None = None
        best_overlap = -1
        for item_id, text in sorted(candidate_evidence_texts.items()):
            overlap = len(claim_tokens & significant_tokens(text))
            if overlap > best_overlap:
                best_overlap = overlap
                best_item_id = item_id
                best_text = text

        return best_text, best_item_id

    def _budget_exhausted_result(self, claim: GeneratedClaim) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            label=SupportLabel.NOT_VERIFIABLE,
            confidence=0.0,
            stage=VerificationStage.SELECTIVE_NLI,
            reason_code=VerificationReasonCode.BUDGET_EXHAUSTED,
            rationale="Per-answer semantic verification budget exhausted before this claim",
        )