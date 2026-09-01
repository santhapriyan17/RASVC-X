"""Module 8 orchestration: produces the pipeline's "verified context".

Wires together candidate generation, deterministic validation, contextual
validation, selective NLI routing, and evidence resolution into a single
``ValidationPipeline.run(...)`` call that operates on an EvidenceBundle in
place and returns a summary. This is the only place in M8 that mutates the
bundle (via its documented incremental-mutation methods) -- individual
validators/resolvers are pure with respect to the bundle.

M8 never generates a final answer; ``ValidationSummary`` describes the
state of claim/evidence validation only, for consumption by the future
Generation/Verification/Confidence/Decision modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rasvcx.provenance.source_quality import SourceQualityScorer
from rasvcx.schemas.common import CandidateId, NLILabel
from rasvcx.schemas.evidence import CandidatePair, EvidenceBundle
from rasvcx.schemas.query import QueryRequest, RiskProfile
from rasvcx.schemas.validation import (
    EvidenceRelationshipResult,
    ValidationLabel,
    ValidationResult,
    ValidationStage,
)
from rasvcx.validation.candidate_generation import CandidateGenerator
from rasvcx.validation.contextual_validator import ContextualValidator
from rasvcx.validation.deterministic import DeterministicValidator
from rasvcx.validation.nli_interface import NLIService
from rasvcx.validation.resolution import EvidenceResolver
from rasvcx.validation.selective_router import RoutingAction, SelectiveValidationRouter
from rasvcx.validation.validation_types import ValidationConfig

_NLI_LABEL_TO_VALIDATION_LABEL: dict[NLILabel, ValidationLabel] = {
    NLILabel.ENTAILMENT: ValidationLabel.SUPPORTED,
    NLILabel.CONTRADICTION: ValidationLabel.CONTRADICTION,
    # Neutral does not confirm either agreement or disagreement; treated
    # conservatively as uncertain rather than as support.
    NLILabel.NEUTRAL: ValidationLabel.UNCERTAIN,
}


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Result of running the M8 pipeline once over an EvidenceBundle."""

    candidates_generated: int
    validation_results: dict[CandidateId, ValidationResult]
    resolutions: list[EvidenceRelationshipResult]
    nli_calls_used: int

    @property
    def unresolved_count(self) -> int:
        from rasvcx.schemas.common import EvidenceRelationship

        return sum(1 for r in self.resolutions if r.relationship is EvidenceRelationship.UNRESOLVED)

    @property
    def genuine_conflict_count(self) -> int:
        from rasvcx.schemas.common import EvidenceRelationship

        return sum(
            1 for r in self.resolutions if r.relationship is EvidenceRelationship.GENUINE_CONFLICT
        )


class ValidationPipeline:
    """Top-level Module 8 orchestrator.

    All collaborators are injected (dependency injection, Section 23): no
    global mutable state, no singleton model objects. Constructing a
    ValidationPipeline never loads an NLI model -- ``NLIService`` only
    calls into its backend lazily, on first ``predict_safe`` invocation
    during ``run``.
    """

    def __init__(
        self,
        nli_service: NLIService,
        config: ValidationConfig | None = None,
        candidate_generator: CandidateGenerator | None = None,
        deterministic_validator: DeterministicValidator | None = None,
        contextual_validator: ContextualValidator | None = None,
        selective_router: SelectiveValidationRouter | None = None,
        resolver: EvidenceResolver | None = None,
    ) -> None:
        self._config = config or ValidationConfig()
        self._nli_service = nli_service
        self._candidate_generator = candidate_generator or CandidateGenerator(
            self._config.candidate_generation
        )
        self._deterministic_validator = deterministic_validator or DeterministicValidator(
            self._config.deterministic
        )
        self._contextual_validator = contextual_validator or ContextualValidator()
        self._selective_router = selective_router or SelectiveValidationRouter(
            self._config.selective_routing
        )
        self._resolver = resolver or EvidenceResolver(SourceQualityScorer())

    def run(
        self,
        bundle: EvidenceBundle,
        query: QueryRequest,
        risk_profile: RiskProfile,
    ) -> ValidationSummary:
        overall_start = time.perf_counter()

        candidates = self._generate_candidates(bundle)

        validation_results: dict[CandidateId, ValidationResult] = {}
        resolutions: list[EvidenceRelationshipResult] = []
        nli_calls_used = 0

        deterministic_elapsed = 0.0
        contextual_elapsed = 0.0
        selective_nli_elapsed = 0.0
        resolution_elapsed = 0.0

        nli_available = self._nli_service.is_available()

        # Highest-priority candidates first, so a bounded NLI budget is
        # spent on the pairs most worth checking (Section 9/12).
        for candidate in sorted(candidates, key=lambda c: c.priority, reverse=True):
            det_start = time.perf_counter()
            deterministic_result = self._deterministic_validator.validate(
                bundle, candidate.candidate_id, candidate.item_id_a, candidate.item_id_b
            )
            deterministic_elapsed += time.perf_counter() - det_start

            ctx_start = time.perf_counter()
            contextual_result = self._contextual_validator.validate(
                bundle, candidate.candidate_id, candidate.item_id_a, candidate.item_id_b
            )
            contextual_elapsed += time.perf_counter() - ctx_start

            decision = self._selective_router.route(
                bundle=bundle,
                candidate_id=candidate.candidate_id,
                item_id_a=candidate.item_id_a,
                item_id_b=candidate.item_id_b,
                deterministic_result=deterministic_result,
                contextual_result=contextual_result,
                risk_profile=risk_profile,
                nli_available=nli_available,
                nli_calls_used=nli_calls_used,
            )

            nli_result: ValidationResult | None = None
            if decision.action is RoutingAction.ESCALATE_TO_NLI:
                nli_start = time.perf_counter()
                nli_result = self._run_nli(bundle, candidate)
                selective_nli_elapsed += time.perf_counter() - nli_start
                if nli_result is not None:
                    nli_calls_used += 1
                    bundle.record_nli_calls(1)

            final_result = self._select_final_result(
                decision.action, deterministic_result, contextual_result, nli_result
            )
            bundle.add_validation_result(candidate.candidate_id, final_result)
            validation_results[candidate.candidate_id] = final_result

            res_start = time.perf_counter()
            resolution_result = self._resolver.resolve(
                bundle,
                candidate.candidate_id,
                candidate.item_id_a,
                candidate.item_id_b,
                deterministic_result,
                contextual_result,
                nli_result,
            )
            resolution_elapsed += time.perf_counter() - res_start
            bundle.add_resolution(resolution_result)
            resolutions.append(resolution_result)

        bundle.record_stage_elapsed("deterministic_validation", deterministic_elapsed)
        bundle.record_stage_elapsed("contextual_validation", contextual_elapsed)
        bundle.record_stage_elapsed("selective_nli", selective_nli_elapsed)
        bundle.record_stage_elapsed("evidence_resolution", resolution_elapsed)
        bundle.record_stage_elapsed("verified_context", time.perf_counter() - overall_start)

        return ValidationSummary(
            candidates_generated=len(candidates),
            validation_results=validation_results,
            resolutions=resolutions,
            nli_calls_used=nli_calls_used,
        )

    # -- internal helpers -----------------------------------------------

    def _generate_candidates(self, bundle: EvidenceBundle) -> list[CandidatePair]:
        start = time.perf_counter()
        candidates = self._candidate_generator.generate(bundle)
        bundle.record_stage_elapsed("candidate_generation", time.perf_counter() - start)
        return candidates

    def _run_nli(self, bundle: EvidenceBundle, candidate: CandidatePair) -> ValidationResult | None:
        item_a = bundle.evidence_items[candidate.item_id_a]
        item_b = bundle.evidence_items[candidate.item_id_b]

        signal = self._nli_service.predict_safe(premise=item_a.text, hypothesis=item_b.text)
        if signal is None:
            # Graceful degradation: NLI failure/timeout/unavailability
            # never produces a stage=SELECTIVE_NLI ValidationResult (the
            # schema requires an nli_signal for that stage) and never maps
            # to SUPPORTED. The caller falls back to the contextual result.
            return None

        label = _NLI_LABEL_TO_VALIDATION_LABEL[signal.label]
        return ValidationResult(
            candidate_id=candidate.candidate_id,
            stage=ValidationStage.SELECTIVE_NLI,
            label=label,
            confidence=signal.confidence,
            nli_signal=signal,
            rationale=f"NLI backend returned {signal.label.value} (confidence={signal.confidence:.2f})",
        )

    def _select_final_result(
        self,
        action: RoutingAction,
        deterministic_result: ValidationResult,
        contextual_result: ValidationResult,
        nli_result: ValidationResult | None,
    ) -> ValidationResult:
        if action is RoutingAction.ESCALATE_TO_NLI and nli_result is not None:
            return nli_result
        if action is RoutingAction.ACCEPT_CONTEXTUAL:
            return contextual_result
        if action is RoutingAction.ACCEPT_DETERMINISTIC:
            return deterministic_result
        # SKIP_NLI_BUDGET_EXHAUSTED, or ESCALATE_TO_NLI that failed to
        # produce a signal: fall back to the best already-computed result
        # rather than fabricating certainty. Contextual is preferred over
        # deterministic here because it incorporates provenance signal the
        # deterministic stage does not see.
        return contextual_result