"""Module 9 -- Post-Generation Verification.

Verifies a *generated answer* against the pipeline's evidence bundle
(optionally informed by Module 8's ``ValidationSummary``). This is a
distinct question from Module 8's: M8 asks "how do retrieved evidence
items relate to each other?"; M9 asks "does the generated answer
accurately represent the validated evidence?" (see module docstrings in
verification/deterministic_check.py and schemas/verification.py for the
full architectural rationale).

Pipeline (deterministic-first, matching M8's established philosophy):

    generated_answer
        -> GeneratedClaimExtractor            (claim + citation extraction)
        -> check_citation                      (citation presence/validity)
        -> DeterministicClaimVerifier           (numeric/negation/qualifier/
                                                  comparative/temporal/scope)
        -> SelectiveSemanticVerifier            (NLI, only when deterministic
                                                  was inconclusive, budgeted)
        -> apply_m8_conflict_context             (surface M8 evidence conflicts)
        -> VerdictAggregator                     (answer-level verdict)
        -> VerificationSummary

Importing this package performs no model loading, no network access, and
no expensive initialization -- ``SelectiveSemanticVerifier`` only calls
into its injected ``NLIService``, which itself never loads a model except
lazily inside a concrete backend's first ``predict`` call (see
validation/nli_model.py's lazy-loading contract, reused unchanged here).
"""

from __future__ import annotations

import time

from rasvcx.schemas.common import EvidenceItemId
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.query import QueryRequest
from rasvcx.schemas.verification import (
    AnswerVerdict,
    CitationResult,
    CitationStatus,
    ClaimVerificationResult,
    GeneratedClaim,
    GeneratedClaimId,
    SupportLabel,
    VerificationReasonCode,
    VerificationStage,
    VerificationSummary,
)
from rasvcx.validation.nli_interface import NLIService
from rasvcx.validation.nli_model import NullNLIBackend
from rasvcx.validation.verified_context import ValidationSummary
from rasvcx.verification.claim_extraction_post import GeneratedClaimExtractor, orphan_citation_item_ids
from rasvcx.verification.deterministic_check import (
    DeterministicClaimVerifier,
    check_citation,
    find_lexically_relevant_items,
)
from rasvcx.verification.semantic_check import SelectiveSemanticVerifier
from rasvcx.verification.verdict_aggregator import (
    VerdictAggregator,
    VerificationConfig,
    apply_m8_conflict_context,
)

__all__ = [
    "AnswerVerdict",
    "CitationResult",
    "CitationStatus",
    "ClaimVerificationResult",
    "GeneratedClaim",
    "GeneratedClaimId",
    "SupportLabel",
    "VerificationReasonCode",
    "VerificationStage",
    "VerificationSummary",
    "GeneratedClaimExtractor",
    "orphan_citation_item_ids",
    "DeterministicClaimVerifier",
    "check_citation",
    "find_lexically_relevant_items",
    "SelectiveSemanticVerifier",
    "VerdictAggregator",
    "VerificationConfig",
    "apply_m8_conflict_context",
    "VerificationPipeline",
]


class VerificationPipeline:
    """Top-level Module 9 orchestrator.

    All collaborators are injected (dependency injection, matching M8's
    ``ValidationPipeline`` convention): no global mutable state, no
    singleton model objects. Constructing a ``VerificationPipeline`` never
    loads an NLI model -- pass an ``NLIService`` wrapping a real backend
    only if semantic escalation should actually be capable of running;
    the default (``NullNLIBackend``) makes the deterministic-only path the
    safe out-of-the-box behavior.
    """

    def __init__(
        self,
        nli_service: NLIService | None = None,
        config: VerificationConfig | None = None,
        claim_extractor: GeneratedClaimExtractor | None = None,
        deterministic_verifier: DeterministicClaimVerifier | None = None,
        semantic_verifier: SelectiveSemanticVerifier | None = None,
        aggregator: VerdictAggregator | None = None,
    ) -> None:
        self._config = config or VerificationConfig()
        self._claim_extractor = claim_extractor or GeneratedClaimExtractor()
        self._deterministic_verifier = deterministic_verifier or DeterministicClaimVerifier()
        self._nli_service = nli_service or NLIService(NullNLIBackend())
        self._semantic_verifier = semantic_verifier or SelectiveSemanticVerifier(self._nli_service)
        self._aggregator = aggregator or VerdictAggregator(self._config)

    def verify(
        self,
        generated_answer: str,
        evidence_bundle: EvidenceBundle,
        query: QueryRequest | None = None,
        validation_summary: ValidationSummary | None = None,
    ) -> VerificationSummary:
        """Verify *generated_answer* against *evidence_bundle*.

        Empty or whitespace-only answers (Section 58) return a valid,
        non-crashing ``VerificationSummary`` with no claims -- never a
        verified/supported result.
        """
        start = time.perf_counter()
        try:
            return self._verify_inner(generated_answer, evidence_bundle, validation_summary)
        finally:
            evidence_bundle.record_stage_elapsed(
                "post_generation_verification", time.perf_counter() - start
            )

    # -- internal --------------------------------------------------------

    def _verify_inner(
        self,
        generated_answer: str,
        evidence_bundle: EvidenceBundle,
        validation_summary: ValidationSummary | None,
    ) -> VerificationSummary:
        all_claims = self._claim_extractor.extract(generated_answer, evidence_bundle)
        claims = all_claims[: self._config.max_claims]
        budget_exhausted_by_claim_cap = len(claims) < len(all_claims)

        citation_results: list[CitationResult] = []
        claim_results: list[ClaimVerificationResult] = []
        claim_is_safety_critical: dict[GeneratedClaimId, bool] = {}
        nli_calls_used = 0

        for claim in claims:
            claim_is_safety_critical[claim.claim_id] = claim.is_safety_critical
            citation = check_citation(claim, evidence_bundle)
            citation_results.append(citation)

            result = self._deterministic_verifier.verify(claim, evidence_bundle, citation)
            if result is None:
                candidate_texts = self._candidate_texts(claim, evidence_bundle, citation)
                result, calls_made = self._semantic_verifier.verify(
                    claim,
                    candidate_texts,
                    calls_used=nli_calls_used,
                    call_budget=self._config.max_nli_calls_per_answer,
                )
                nli_calls_used += calls_made

            claim_results.append(result)

        claim_results = apply_m8_conflict_context(claim_results, validation_summary, evidence_bundle)
        orphan_ids = orphan_citation_item_ids(claims, evidence_bundle)

        return self._aggregator.aggregate(
            claim_results=claim_results,
            citation_results=citation_results,
            orphan_citation_ids=orphan_ids,
            semantic_verification_calls=nli_calls_used,
            budget_exhausted=budget_exhausted_by_claim_cap
            or nli_calls_used >= self._config.max_nli_calls_per_answer,
            claim_is_safety_critical=claim_is_safety_critical,
        )

    def _candidate_texts(
        self, claim: GeneratedClaim, bundle: EvidenceBundle, citation: CitationResult
    ) -> dict[EvidenceItemId, str]:
        if citation.status is CitationStatus.CORRECT:
            return {iid: bundle.evidence_items[iid].text for iid in citation.cited_item_ids}
        items = find_lexically_relevant_items(claim, bundle)[
            : self._config.max_fallback_evidence_candidates
        ]
        return {item.item_id: item.text for item in items}