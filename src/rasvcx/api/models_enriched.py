"""src/rasvcx/api/models_enriched.py

Enriched Pydantic v2 response models for the RASVC-X M16 API layer.

Extends the thin M14 models (models.py) with full claim verification,
conflict resolution, and validation detail — used by routes_eval.py
and the enriched variant of /query.

All field names are verified against the M1-M12 source types this session:
  VerificationSummary: answer_verdict, overall_confidence, claim_results,
                       citation_results, orphan_citation_ids,
                       semantic_verification_calls,
                       safety_critical_failure_count, budget_exhausted
  ClaimVerificationResult: claim_id, label, confidence, stage, reason_code,
                           rationale, supporting_item_ids,
                           contradicting_item_ids
  ValidationSummary: candidates_generated, validation_results, resolutions,
                     nli_calls_used
                     (genuine_conflict_count, unresolved_count are @property)
  EvidenceRelationshipResult: candidate_id, relationship, confidence,
                              contributing_claim_ids, rationale
  ValidationResult: candidate_id, stage, label, confidence, nli_signal,
                    rationale
  Decision: action, confidence, corrective_target, contributing_candidate_ids,
            rationale
  RiskProfile: overall_risk_score, feature_scores, validation_depth,
               retrieval_retry_budget, nli_call_allowance, safety_floor_forced
  RiskFeatureScores: numeric_content_score, unit_sensitive_score,
                     safety_critical_structure_score, query_complexity_score,
                     ambiguity_score

LIMITATIONS:
  - answer_verdict calibration is an operational proxy, not a measure of
    clinical correctness.
  - No field in this module constitutes medical advice or clinical guidance.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Claim verification models
# ---------------------------------------------------------------------------


class ClaimVerificationModel(BaseModel):
    """Serialised form of one ClaimVerificationResult.

    Verified fields from rasvcx.schemas.verification.ClaimVerificationResult.
    """

    claim_id: str
    text: str = ""
    """The generated sentence this result refers to."""
    cited_evidence_ids: list[str] = Field(default_factory=list)
    citation_status: str | None = None
    """CitationStatus value for this claim, or None if not checked."""
    is_safety_critical: bool = False
    label: str
    """SupportLabel value: supported | partially_supported | contradicted |
    unsupported | uncertain | not_verifiable"""
    confidence: float = Field(ge=0.0, le=1.0)
    stage: str
    """VerificationStage value string."""
    reason_code: str | None = None
    """VerificationReasonCode value string, or None."""
    rationale: str | None = None
    supporting_item_ids: list[str] = Field(default_factory=list)
    contradicting_item_ids: list[str] = Field(default_factory=list)


class VerificationSummaryModel(BaseModel):
    """Serialised form of VerificationSummary.

    Verified fields from rasvcx.schemas.verification.VerificationSummary.
    """

    answer_verdict: str
    """AnswerVerdict value: verified | partially_verified | unverified |
    unsafe | insufficient_evidence"""
    overall_confidence: float = Field(ge=0.0, le=1.0)
    claim_results: list[ClaimVerificationModel] = Field(default_factory=list)
    semantic_verification_calls: int = Field(default=0, ge=0)
    safety_critical_failure_count: int = Field(default=0, ge=0)
    budget_exhausted: bool = False


# ---------------------------------------------------------------------------
# Conflict / validation models
# ---------------------------------------------------------------------------


class ValidationResultModel(BaseModel):
    """Serialised form of one ValidationResult.

    Verified fields from rasvcx.validation.verified_context.ValidationResult.
    """

    candidate_id: str
    stage: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    nli_signal: str | None = None
    rationale: str | None = None


class ConflictResolutionModel(BaseModel):
    """Serialised form of one EvidenceRelationshipResult.

    Verified fields from
    rasvcx.validation.verified_context.EvidenceRelationshipResult.
    """

    candidate_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    """The two evidence ids this resolution compares."""
    validation_stage: str | None = None
    """Which validator produced the final label for this pair:
    deterministic | contextual | selective_nli."""
    validation_label: str | None = None
    relationship: str
    """EvidenceRelationship value: compatible | population-diff |
    temporal-diff | jurisdiction-diff | dosage-diff |
    genuine-conflict | unresolved"""
    confidence: float = Field(ge=0.0, le=1.0)
    contributing_claim_ids: list[str] = Field(default_factory=list)
    rationale: str | None = None


class ValidationSummaryModel(BaseModel):
    """Serialised form of ValidationSummary.

    genuine_conflict_count and unresolved_count are @property on the
    source type; they are computed and stored here as plain ints.
    Verified fields from
    rasvcx.validation.verified_context.ValidationSummary.
    """

    candidates_generated: int = Field(ge=0)
    nli_calls_used: int = Field(ge=0)
    nli_failures: int = Field(default=0, ge=0)
    genuine_conflict_count: int = Field(default=0, ge=0)
    unresolved_count: int = Field(default=0, ge=0)
    resolutions: list[ConflictResolutionModel] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk profile model (enriched)
# ---------------------------------------------------------------------------


class RiskFeatureScoresModel(BaseModel):
    """Serialised form of RiskFeatureScores.

    Verified fields from rasvcx.schemas.query.RiskFeatureScores.
    """

    numeric_content_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unit_sensitive_score: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_critical_structure_score: float = Field(default=0.0, ge=0.0, le=1.0)
    query_complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ambiguity_score: float = Field(default=0.0, ge=0.0, le=1.0)


class RiskProfileEnrichedModel(BaseModel):
    """Full serialised form of RiskProfile including feature scores.

    Verified fields from rasvcx.schemas.query.RiskProfile.
    """

    overall_risk_score: float = Field(ge=0.0, le=1.0)
    validation_depth: str
    """ValidationDepth value: shallow | standard | deep"""
    retrieval_retry_budget: int = Field(ge=0)
    nli_call_allowance: int = Field(ge=0)
    safety_floor_forced: bool = False
    feature_scores: RiskFeatureScoresModel = Field(
        default_factory=RiskFeatureScoresModel
    )


# ---------------------------------------------------------------------------
# Evidence / provenance / trace models
# ---------------------------------------------------------------------------


class ProvenanceModel(BaseModel):
    """Contextual metadata of one evidence chunk. null == UNKNOWN."""

    source_type: str
    date: str | None = None
    jurisdiction: str | None = None
    population: str | None = None
    dosage_context: str | None = None


class ProvenanceAnalysisModel(BaseModel):
    """M6 analysis of one evidence item against the query context.

    Verdicts: known_match | known_mismatch | unknown.
    """

    source_known: bool
    quality_score: float = Field(ge=0.0, le=1.0)
    temporal: str
    jurisdiction: str
    population: str
    dosage_context: str


class EvidenceModel(BaseModel):
    """One evidence item the decision was based on."""

    evidence_id: str
    """Citation label used in the answer text, e.g. "E1"."""
    chunk_id: str
    doc_id: str | None = None
    title: str | None = None
    source_url: str | None = None
    filename: str | None = None
    section: str | None = None
    page: int | None = None
    text: str
    retrieval_score: float
    rerank_score: float | None = None
    cited: bool = False
    provenance: ProvenanceModel
    analysis: ProvenanceAnalysisModel | None = None
    role: str = "RETRIEVED"
    """RETRIEVED | RELEVANT | SUPPORTING | CONTRADICTORY | IRRELEVANT | SUPERSEDED"""
    role_reason: str | None = None
    temporal_status: str = "UNKNOWN"
    """CURRENT | SUPERSEDED | HISTORICAL | WITHDRAWN | UNKNOWN (declared lifecycle only)"""
    temporal_reason: str | None = None
    supports_answer: bool = False
    contradicts_answer: bool = False
    lifecycle: dict[str, Any] | None = None
    authority_tier: str | None = None


class ProvenanceSummaryModel(BaseModel):
    """Bundle-level M6 result."""

    unique_source_count: int = Field(ge=0)
    unknown_provenance_ratio: float = Field(ge=0.0, le=1.0)
    mismatch_present: bool
    strict_context_required: bool


class StageTraceModel(BaseModel):
    """One stage the pipeline executed (or explicitly skipped)."""

    stage: str
    status: str
    """ok | failed | skipped"""
    elapsed_ms: float = Field(ge=0.0)
    detail: str | None = None
    attempt: int = Field(default=0, ge=0)


class RuntimeModeModel(BaseModel):
    """Which mode and components produced this response."""

    execution_mode: str
    offline: bool
    """True only in offline_test: the response came from test doubles."""
    llm_provider: str
    llm_model: str | None = None
    mock_llm: bool
    retrieval_mode: str
    reranker: bool
    nli: bool


#: Stage names of the execution trace that represent each pipeline module.
MODULE_STAGES: dict[str, tuple[str, ...]] = {
    "risk_routing": ("risk_routing",),
    "bm25": ("bm25_retrieval",),
    "qdrant": ("dense_retrieval",),
    "rrf": ("rrf_fusion",),
    "reranker": ("reranking",),
    "evidence_sufficiency": ("sufficiency_gate",),
    "provenance": ("provenance_context",),
    "claim_extraction": ("atomic_claim_extraction",),
    "deterministic_validation": ("deterministic_validation",),
    "contextual_validation": ("contextual_validation",),
    # NLI runs in two places: M8 evidence-pair validation (selective_nli)
    # and M9 generated-claim verification (semantic_verification).
    "nli": ("selective_nli", "semantic_verification"),
    "conflict_resolution": ("evidence_resolution",),
    "generation": ("generation",),
    "post_generation_verification": ("post_generation_verification",),
    "confidence": ("confidence_estimation",),
    "calibration": ("calibration",),
    "decision_engine": ("decision",),
}


# ---------------------------------------------------------------------------
# Enriched query response
# ---------------------------------------------------------------------------


class EnrichedQueryResponse(BaseModel):
    """Full response for POST /query with enriched=true.

    This is the contract the chat UI consumes.  Every field is derived from
    the PipelineResult of the request; nothing is filled in from
    configuration to make a stage look as if it ran.
    """

    query_id: str
    request_id: str | None = None
    success: bool
    """False when a pipeline stage failed (see pipeline_error)."""

    decision: str
    """Canonical decision: ANSWER | ANSWER_WITH_WARNING | REPAIR |
    REGENERATE | ABSTAIN."""
    action: str
    """Internal DecisionAction value (answer | warning | ...); kept for
    backward compatibility -- clients should read `decision`."""
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None

    answer: str = ""
    """The answer text.  Empty unless decision is ANSWER or
    ANSWER_WITH_WARNING: text the pipeline declined to stand behind is
    never returned."""
    generated_text: str = ""
    """Same as `answer` (backward-compatible name)."""
    has_answer: bool = False

    evidence: list[EvidenceModel] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    """Evidence ids cited by the answer, in evidence order."""
    retrieved_chunk_ids: list[str] = Field(default_factory=list)

    verification: VerificationSummaryModel | None = None
    validation: ValidationSummaryModel | None = None
    provenance: ProvenanceSummaryModel | None = None
    risk_profile: RiskProfileEnrichedModel | None = None

    retrieval: dict[str, Any] = Field(default_factory=dict)
    """What retrieval executed: mode, bm25_hits, dense_hits, rrf_fused, ..."""
    kb_version_id: str | None = None

    trace: list[StageTraceModel] = Field(default_factory=list)
    modules: dict[str, str] = Field(default_factory=dict)
    """module -> executed | failed | skipped | not_reached, derived from
    the trace of this request."""
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float = Field(default=0.0, ge=0.0)

    nli_calls: int = Field(default=0, ge=0)
    calibration_status: str | None = None
    """uncalibrated | calibrated | invalidated | unavailable.  Only
    "calibrated" confidence is an estimated probability of correctness."""
    calibration_version: str | None = None
    calibration_dataset_hash: str | None = None
    corrective_attempts: int = Field(default=0, ge=0)

    warnings: list[str] = Field(default_factory=list)
    degraded: bool = False
    """True when a stage failed or a component produced no signal."""
    pipeline_error: dict[str, Any] | None = None

    mode: RuntimeModeModel | None = None
    mock_llm: bool = Field(
        default=True,
        description=(
            "True when MockLLMClient was used. "
            "Results must not be used to claim real model accuracy."
        ),
    )
    limitations: str = ""
    cached: bool = False
    """True when served from the answer cache (same question, same KB version)."""
    request_class: str | None = None
    """COLD_UNCACHED | WARM_UNCACHED | CACHE_HIT | PROVIDER_ERROR | SYSTEM_ERROR"""
    provider: dict[str, Any] = Field(default_factory=dict)
    """LLM calls, HTTP attempts, retries, retry wait, provider seconds, quota."""
    kb: dict[str, Any] | None = None
    run_identity: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Eval run summary models (used by routes_eval.py)
# ---------------------------------------------------------------------------


class EvalRunSummaryModel(BaseModel):
    """Summary of one completed evaluation run for GET /eval/runs/{run_id}."""

    run_id: str
    baseline_id: str
    dataset_id: str
    dataset_version: str
    split: str
    corpus_fingerprint: str
    corpus_condition: str
    execution_mode: str
    mock_llm: bool
    total_cases: int = Field(ge=0)
    offered: int = Field(ge=0)
    accepted: int = Field(ge=0)
    completed: int = Field(ge=0)
    error: int = Field(ge=0)
    skipped: int = Field(ge=0)
    rejected_overload: int = Field(ge=0)
    cancelled_deadline: int = Field(ge=0)
    not_offered_deadline: int = Field(ge=0)
    baseline_init_seconds: float = Field(ge=0.0)
    total_run_wall_seconds: float = Field(ge=0.0)
    start_utc: str = ""
    end_utc: str = ""
    results_jsonl_path: str = ""
    integrity_error: str | None = None
    limitations: str = Field(
        default=(
            "MockLLMClient results are not evidence of real model accuracy. "
            "Calibration metrics are an operational proxy only. "
            "No result constitutes medical advice."
        ),
    )


class EvalRunListModel(BaseModel):
    """Response for GET /eval/runs — list of run summaries."""

    runs: list[EvalRunSummaryModel] = Field(default_factory=list)
    total: int = Field(ge=0)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


_MOCK_LIMITATIONS = (
    "OFFLINE TEST MODE: this response was produced by MockLLMClient and "
    "other test doubles. It is not evidence of real model accuracy. "
    "No field constitutes medical advice."
)
_REAL_LIMITATIONS = (
    "Research prototype. Answers are generated from retrieved documents and "
    "automatically validated; validation can be wrong. Confidence is "
    "uncalibrated unless calibration_status is 'calibrated'. "
    "No field constitutes medical advice."
)


def limitations_notice(mock_llm: bool) -> str:
    return _MOCK_LIMITATIONS if mock_llm else _REAL_LIMITATIONS


def _unknown_to_none(value: Any) -> str | None:
    from rasvcx.schemas.common import UNKNOWN

    return None if value is UNKNOWN or value is None else str(value)


def module_states(trace: Any) -> dict[str, str]:
    """Per-module execution state derived ONLY from the request's trace."""
    by_stage: dict[str, list[str]] = {}
    for entry in trace:
        by_stage.setdefault(entry.stage, []).append(entry.status)
    states: dict[str, str] = {}
    for module, stages in MODULE_STAGES.items():
        statuses = [st for stage in stages for st in by_stage.get(stage, [])]
        if not statuses:
            states[module] = "not_reached"
        elif "failed" in statuses:
            states[module] = "failed"
        elif "ok" in statuses:
            states[module] = "executed"
        else:
            states[module] = "skipped"
    return states


def stage_latencies_ms(trace: Any) -> dict[str, float]:
    """Total measured milliseconds per stage (summed over corrective passes)."""
    totals: dict[str, float] = {}
    corrective = 0.0
    for entry in trace:
        if entry.status == "skipped":
            continue
        totals[entry.stage] = totals.get(entry.stage, 0.0) + entry.elapsed_ms
        # Stages re-run by REPAIR / REGENERATE passes (attempt > 0).  Nested
        # entries (nli_inference, provider_backoff) are already inside
        # their enclosing stage and are not added twice.
        if entry.attempt > 0 and entry.stage not in _NESTED_STAGES:
            corrective += entry.elapsed_ms
    if corrective:
        totals["corrective_passes"] = corrective
    return {stage: round(ms, 3) for stage, ms in totals.items()}


#: Trace entries measured INSIDE another stage (reported for attribution).
_NESTED_STAGES = frozenset({
    "nli_inference", "provider_backoff", "bm25_retrieval", "dense_retrieval", "rrf_fusion",
    "candidate_generation", "deterministic_validation", "contextual_validation",
    "selective_nli", "evidence_resolution", "semantic_verification",
})


def pipeline_result_to_enriched(
    result: Any,
    request_id: str | None = None,
    mock_llm: bool = True,
    mode: RuntimeModeModel | None = None,
) -> EnrichedQueryResponse:
    """Convert a PipelineResult to an EnrichedQueryResponse.

    All imports are local to avoid circular dependencies at module load.
    """
    from rasvcx.schemas.decision import DecisionAction

    cited_ids = {str(i) for i in result.cited_item_ids}

    # Verification summary
    verification: VerificationSummaryModel | None = None
    if result.verification_summary is not None:
        vs = result.verification_summary
        claims_by_id = {str(c.claim_id): c for c in getattr(vs, "claims", ())}
        citation_by_id = {str(c.claim_id): c for c in vs.citation_results}
        claim_models = []
        for cr in vs.claim_results:
            claim = claims_by_id.get(str(cr.claim_id))
            citation = citation_by_id.get(str(cr.claim_id))
            claim_models.append(
                ClaimVerificationModel(
                    claim_id=str(cr.claim_id),
                    text=claim.text if claim is not None else "",
                    cited_evidence_ids=sorted(
                        str(i) for i in (claim.cited_item_ids if claim is not None else ())
                    ),
                    citation_status=citation.status.value if citation is not None else None,
                    is_safety_critical=bool(claim.is_safety_critical) if claim else False,
                    label=cr.label.value,
                    confidence=cr.confidence,
                    stage=cr.stage.value,
                    reason_code=cr.reason_code.value if cr.reason_code else None,
                    rationale=cr.rationale,
                    supporting_item_ids=sorted(str(i) for i in cr.supporting_item_ids),
                    contradicting_item_ids=sorted(
                        str(i) for i in cr.contradicting_item_ids
                    ),
                )
            )
        verification = VerificationSummaryModel(
            answer_verdict=vs.answer_verdict.value,
            overall_confidence=vs.overall_confidence,
            claim_results=claim_models,
            semantic_verification_calls=vs.semantic_verification_calls,
            safety_critical_failure_count=vs.safety_critical_failure_count,
            budget_exhausted=vs.budget_exhausted,
        )

    # Validation summary
    validation: ValidationSummaryModel | None = None
    if result.validation_summary is not None:
        vsum = result.validation_summary
        pairs = getattr(result, "candidate_pairs", {}) or {}
        resolution_models = []
        for r in vsum.resolutions:
            final = vsum.validation_results.get(r.candidate_id)
            resolution_models.append(
                ConflictResolutionModel(
                    candidate_id=str(r.candidate_id),
                    evidence_ids=list(pairs.get(str(r.candidate_id), ())),
                    validation_stage=final.stage.value if final is not None else None,
                    validation_label=final.label.value if final is not None else None,
                    relationship=r.relationship.value,
                    confidence=r.confidence,
                    contributing_claim_ids=[
                        str(c) for c in r.contributing_claim_ids
                    ],
                    rationale=r.rationale,
                )
            )
        validation = ValidationSummaryModel(
            candidates_generated=vsum.candidates_generated,
            nli_calls_used=vsum.nli_calls_used,
            nli_failures=getattr(vsum, "nli_failures", 0),
            genuine_conflict_count=vsum.genuine_conflict_count,
            unresolved_count=vsum.unresolved_count,
            resolutions=resolution_models,
        )

    # Risk profile
    risk_profile: RiskProfileEnrichedModel | None = None
    if result.risk_profile is not None:
        rp = result.risk_profile
        fs = rp.feature_scores
        risk_profile = RiskProfileEnrichedModel(
            overall_risk_score=rp.overall_risk_score,
            validation_depth=rp.validation_depth.value,
            retrieval_retry_budget=rp.retrieval_retry_budget,
            nli_call_allowance=rp.nli_call_allowance,
            safety_floor_forced=rp.safety_floor_forced,
            feature_scores=RiskFeatureScoresModel(
                numeric_content_score=fs.numeric_content_score,
                unit_sensitive_score=fs.unit_sensitive_score,
                safety_critical_structure_score=(
                    fs.safety_critical_structure_score
                ),
                query_complexity_score=fs.query_complexity_score,
                ambiguity_score=fs.ambiguity_score,
            ),
        )

    # Provenance (M6)
    prov_result = getattr(result, "provenance", None)
    per_item = getattr(prov_result, "per_item", {}) or {}
    provenance: ProvenanceSummaryModel | None = None
    if prov_result is not None:
        provenance = ProvenanceSummaryModel(
            unique_source_count=prov_result.unique_source_count,
            unknown_provenance_ratio=prov_result.unknown_provenance_ratio,
            mismatch_present=prov_result.mismatch_present,
            strict_context_required=prov_result.strict_context_required,
        )

    # Evidence
    from rasvcx.provenance.source_quality import SourceQualityScorer

    assessments = getattr(result, "evidence_assessments", {}) or {}
    tier_scorer = SourceQualityScorer()
    evidence: list[EvidenceModel] = []
    for item in getattr(result, "evidence", ()):
        prov = item.provenance
        src = item.source
        analysis = per_item.get(item.item_id)
        assessed = assessments.get(str(item.item_id))
        tier = getattr(getattr(getattr(tier_scorer.score(item), "signal", None), "tier", None), "value", None)
        evidence.append(
            EvidenceModel(
                role=assessed.role.value if assessed else "RETRIEVED",
                role_reason=assessed.reason if assessed else None,
                temporal_status=assessed.temporal_status.value if assessed else "UNKNOWN",
                temporal_reason=assessed.temporal_reason if assessed else None,
                supports_answer=bool(assessed and assessed.supports_answer),
                contradicts_answer=bool(assessed and assessed.contradicts_answer),
                lifecycle=src.lifecycle.to_record() if src and src.lifecycle else None,
                authority_tier=str(tier) if tier is not None else None,
                evidence_id=str(item.item_id),
                chunk_id=str(item.chunk_id),
                doc_id=src.doc_id if src else None,
                title=src.title if src else None,
                source_url=src.source_url if src else None,
                filename=src.filename if src else None,
                section=src.heading if src else None,
                page=src.page if src else None,
                text=item.text,
                retrieval_score=item.retrieval_score,
                rerank_score=item.rerank_score,
                cited=str(item.item_id) in cited_ids,
                provenance=ProvenanceModel(
                    source_type=prov.source_type.value,
                    date=_unknown_to_none(prov.date),
                    jurisdiction=_unknown_to_none(prov.jurisdiction),
                    population=_unknown_to_none(prov.population),
                    dosage_context=_unknown_to_none(prov.dosage_context),
                ),
                analysis=(
                    ProvenanceAnalysisModel(
                        source_known=analysis.is_known,
                        quality_score=analysis.quality_score,
                        temporal=analysis.temporal.value,
                        jurisdiction=analysis.jurisdiction.value,
                        population=analysis.population.value,
                        dosage_context=analysis.dosage_context.value,
                    )
                    if analysis is not None
                    else None
                ),
            )
        )

    # Pipeline error
    pipeline_error_dict: dict[str, Any] | None = None
    if result.pipeline_error is not None:
        pipeline_error_dict = {
            "stage": result.pipeline_error.stage,
            "message": result.pipeline_error.message,
            "is_retryable": result.pipeline_error.is_retryable,
        }

    has_answer = result.decision.action in (
        DecisionAction.ANSWER, DecisionAction.WARNING
    )
    answer = (result.generated_text or "") if has_answer else ""
    trace = list(getattr(result, "trace", ()))
    warnings = list(getattr(result, "warnings", ()))
    # Degraded == a component failed or produced no signal.  A safety
    # abstention by a stage that ran correctly (e.g. the sufficiency gate
    # finding the evidence insufficient) is NOT degradation.
    degraded = (
        result.generation_error is not None
        or any(t.status == "failed" for t in trace)
        or (validation is not None and validation.nli_failures > 0)
        or any(
            c.reason_code == "nli_unavailable"
            for c in (verification.claim_results if verification else [])
        )
    )

    return EnrichedQueryResponse(
        query_id=str(result.query_id),
        request_id=request_id,
        success=result.success,
        decision=result.decision.action.canonical,
        action=result.decision.action.value,
        confidence=result.decision.confidence,
        rationale=result.decision.rationale,
        answer=answer,
        generated_text=answer,
        has_answer=has_answer,
        evidence=evidence,
        citations=[e.evidence_id for e in evidence if e.cited],
        retrieved_chunk_ids=[e.chunk_id for e in evidence],
        verification=verification,
        validation=validation,
        provenance=provenance,
        risk_profile=risk_profile,
        retrieval={
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in (getattr(result, "retrieval", {}) or {}).items()
        },
        kb_version_id=getattr(result, "kb_version_id", None),
        trace=[
            StageTraceModel(
                stage=t.stage, status=t.status, elapsed_ms=round(t.elapsed_ms, 3),
                detail=t.detail, attempt=t.attempt,
            )
            for t in trace
        ],
        modules=module_states(trace),
        stage_latencies_ms=stage_latencies_ms(trace),
        total_latency_ms=round(getattr(result, "total_seconds", 0.0) * 1000.0, 3),
        nli_calls=getattr(result, "nli_calls", 0)
        + (verification.semantic_verification_calls if verification else 0),
        provider=dict(getattr(result, "provider_stats", {}) or {}),
        calibration_status=getattr(result, "calibration_status", None),
        calibration_version=getattr(result, "calibration_version", None),
        calibration_dataset_hash=getattr(result, "calibration_dataset_hash", None),
        corrective_attempts=result.corrective_attempts,
        warnings=warnings,
        degraded=degraded,
        pipeline_error=pipeline_error_dict,
        mode=mode,
        mock_llm=mock_llm,
        limitations=limitations_notice(mock_llm),
    )


def run_record_to_summary(record: Any) -> EvalRunSummaryModel:
    """Convert a RunRecord to an EvalRunSummaryModel."""
    return EvalRunSummaryModel(
        run_id=record.run_id,
        baseline_id=record.baseline_id,
        dataset_id=record.dataset_id,
        dataset_version=record.dataset_version,
        split=(
            record.split.value
            if hasattr(record.split, "value")
            else str(record.split)
        ),
        corpus_fingerprint=record.corpus_fingerprint,
        corpus_condition=(
            record.corpus_condition.value
            if hasattr(record.corpus_condition, "value")
            else str(record.corpus_condition)
        ),
        execution_mode=record.execution_mode,
        mock_llm=record.mock_llm,
        total_cases=record.total_cases,
        offered=record.offered,
        accepted=record.accepted,
        completed=record.completed,
        error=record.error,
        skipped=record.skipped,
        rejected_overload=record.rejected_overload,
        cancelled_deadline=record.cancelled_deadline,
        not_offered_deadline=record.not_offered_deadline,
        baseline_init_seconds=record.baseline_init_seconds,
        total_run_wall_seconds=record.total_run_wall_seconds,
        start_utc=record.start_utc,
        end_utc=record.end_utc,
        results_jsonl_path=record.results_jsonl_path,
        integrity_error=getattr(record, "integrity_error", None),
    )


__all__ = [
    "ClaimVerificationModel",
    "VerificationSummaryModel",
    "ValidationResultModel",
    "ConflictResolutionModel",
    "ValidationSummaryModel",
    "RiskFeatureScoresModel",
    "RiskProfileEnrichedModel",
    "EnrichedQueryResponse",
    "EvidenceModel",
    "ProvenanceModel",
    "ProvenanceAnalysisModel",
    "ProvenanceSummaryModel",
    "StageTraceModel",
    "RuntimeModeModel",
    "MODULE_STAGES",
    "module_states",
    "stage_latencies_ms",
    "limitations_notice",
    "EvalRunSummaryModel",
    "EvalRunListModel",
    "pipeline_result_to_enriched",
    "run_record_to_summary",
]