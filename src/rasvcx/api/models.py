"""Pydantic v2 request and response models for the RASVC-X HTTP API (Module 14).

These models form the public API contract. They are deliberately thin:
they translate between the HTTP layer and the existing M1-M12 schema
types. No pipeline logic lives here.

Verified field names from source:
  PipelineResult: query_id, decision, generated_text, pipeline_error,
                  corrective_attempts, success (property)
  Decision:       action (DecisionAction), confidence, rationale
  DecisionAction: answer, warning, abstain, repair, regenerate
  PipelineError:  stage, message, is_retryable
  RiskProfile:    overall_risk_score, validation_depth
  ValidationDepth: shallow, standard, deep

Component readiness states (three-level):
  configured  -- present in settings, not yet verified reachable
  loaded      -- in-process resource ready (index in memory, model loaded)
  reachable   -- external service probed successfully

Security:
  No secrets (api_key, auth_token) appear in any response model.
  Generated text is included in QueryResponse but never logged by the
  API layer (enforced in routes_query.py).

M16 additions:
  QueryRequest.enriched  -- opt-in flag for EnrichedQueryResponse
  QueryResponse.mock_llm -- flags MockLLMClient results
  QueryResponse.limitations -- mandatory limitations notice
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

QUERY_CONTEXT_KEYS = frozenset(
    {"population", "jurisdiction", "dosage_context", "time_sensitivity"}
)


class QueryRequest(BaseModel):
    """POST /query request body."""

    query: str = Field(
        ...,
        min_length=1,
        description="The natural-language query text.",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional caller-supplied request identifier for tracing.",
    )
    enriched: bool = Field(
        default=False,
        description=(
            "When True, return an EnrichedQueryResponse with full claim "
            "verification, conflict resolution, and validation detail. "
            "When False (default), return the thin QueryResponse."
        ),
    )

    bypass_cache: bool = Field(
        default=False,
        description=(
            "Run the pipeline even if an identical answer is cached (the "
            "result is still stored). Benchmarks use this to measure "
            "inference rather than cache lookups."
        ),
    )

    context: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional clinical context of the question, used by provenance "
            "analysis to check that evidence applies. Allowed keys: "
            "population, jurisdiction, dosage_context, time_sensitivity."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_not_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be whitespace-only")
        return v

    @field_validator("context")
    @classmethod
    def context_keys_allowed(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is None:
            return None
        unknown = set(v) - QUERY_CONTEXT_KEYS
        if unknown:
            raise ValueError(
                f"unknown context key(s) {sorted(unknown)}; "
                f"allowed: {sorted(QUERY_CONTEXT_KEYS)}"
            )
        cleaned = {k: val.strip() for k, val in v.items() if val and val.strip()}
        for key, val in cleaned.items():
            if len(val) > 128:
                raise ValueError(f"context[{key!r}] exceeds 128 characters")
        return cleaned or None


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class DecisionActionEnum(str, Enum):
    """Maps DecisionAction enum values to the API surface."""
    ANSWER = "answer"
    WARNING = "warning"
    ABSTAIN = "abstain"
    REPAIR = "repair"
    REGENERATE = "regenerate"


class DecisionModel(BaseModel):
    """Serialised form of M10 Decision.

    `decision` is the canonical external value (ANSWER |
    ANSWER_WITH_WARNING | REPAIR | REGENERATE | ABSTAIN).  `action` is the
    internal DecisionAction value it maps from.
    """
    action: DecisionActionEnum
    decision: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str | None = None


class PipelineErrorModel(BaseModel):
    """Serialised form of PipelineError."""
    stage: str
    message: str
    is_retryable: bool


class RiskProfileModel(BaseModel):
    """Serialised form of RiskProfile -- scores only, no internal budgets."""
    overall_risk_score: float = Field(ge=0.0, le=1.0)
    validation_depth: str  # ValidationDepth.value: shallow | standard | deep


# ---------------------------------------------------------------------------
# Query response
# ---------------------------------------------------------------------------


class QueryResponse(BaseModel):
    """POST /query response body."""
    query_id: str
    success: bool
    decision: DecisionModel
    generated_text: str = ""
    risk_profile: RiskProfileModel | None = None
    pipeline_error: PipelineErrorModel | None = None
    corrective_attempts: int = Field(default=0, ge=0)
    request_id: str | None = None
    has_answer: bool = Field(
        default=False,
        description="True when action is answer or warning (text is present).",
    )
    mock_llm: bool = Field(
        default=True,
        description=(
            "True when MockLLMClient was used. "
            "Results must not be used to claim real model accuracy."
        ),
    )
    limitations: str = Field(
        default=(
            "MockLLMClient results are not evidence of real model accuracy. "
            "No field constitutes medical advice."
        ),
        description="Mandatory limitations notice.",
    )
    execution_mode: str | None = None
    offline: bool | None = Field(
        default=None,
        description="True only in offline_test: produced by test doubles.",
    )
    kb_version_id: str | None = None
    total_latency_ms: float | None = None
    warnings: list[str] = Field(default_factory=list)
    cached: bool = False
    calibration_status: str | None = None
    calibration_version: str | None = None
    calibration_dataset_hash: str | None = None
    request_class: str | None = None
    """COLD_UNCACHED | WARM_UNCACHED | CACHE_HIT | PROVIDER_ERROR | SYSTEM_ERROR"""
    kb: dict[str, Any] | None = None
    """Identity of the knowledge base that answered (version, source, hashes)."""
    run_identity: dict[str, Any] | None = None
    """Config hash, prompt version and model versions that produced it."""


#: request_class values (see routes_query.classify_request).
REQUEST_CLASSES = (
    "COLD_UNCACHED", "WARM_UNCACHED", "CACHE_HIT", "PROVIDER_ERROR", "SYSTEM_ERROR",
)


# ---------------------------------------------------------------------------
# Health and readiness
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """GET /health response -- always 200 if the process is alive."""
    status: str = "ok"
    version: str = "0.1.0"


class ComponentState(str, Enum):
    """Three-level readiness state for a pipeline component."""
    CONFIGURED = "configured"    # in settings, not yet verified
    LOADED = "loaded"            # in-process resource ready
    REACHABLE = "reachable"      # external service probed OK
    UNAVAILABLE = "unavailable"  # probe failed or deps missing
    DISABLED = "disabled"        # not part of the configured mode
    STUB = "stub"                # offline_test double, not a real component


class ComponentStatus(BaseModel):
    """Readiness status for one pipeline component."""
    name: str
    state: ComponentState
    detail: str | None = None


class ReadinessResponse(BaseModel):
    """GET /ready response -- 200 when all required components are ready."""
    ready: bool
    execution_mode: str
    offline: bool = False
    llm_provider: str = ""
    kb_version_id: str | None = None
    kb_source: str | None = None
    """published_kb | seed_fallback | smoke_test"""
    kb_corpus_hash: str | None = None
    kb_doc_count: int | None = None
    kb_chunk_count: int | None = None
    calibration: dict[str, Any] | None = None
    """status (uncalibrated | calibrated | invalidated), version, dataset hash."""
    kb_warnings: list[str] = Field(default_factory=list)
    """Knowledge-base configuration states the operator must see (seed
    fallback, stray pointer files)."""
    components: list[ComponentStatus] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------


class AdminConfigResponse(BaseModel):
    """GET /admin/config response -- sanitised settings snapshot.

    Secrets (api_key, auth_token) are never included.
    """
    execution_mode: str
    retrieval_mode: str
    reranker_enabled: bool
    nli_enabled: bool
    llm_provider: str
    llm_model_name: str
    max_corrective_attempts: int
    offline: bool = False
    qdrant_mode: str | None = None
    decision_thresholds: dict[str, float] = Field(default_factory=dict)
    effective_routing: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Generic error response body."""
    error: str
    detail: str | None = None
    request_id: str | None = None


# ---------------------------------------------------------------------------
# Conversion helper
# ---------------------------------------------------------------------------


def pipeline_result_to_response(
    result: Any,
    request_id: str | None = None,
    mock_llm: bool = True,
    execution_mode: str | None = None,
) -> QueryResponse:
    """Convert a M12 PipelineResult to a QueryResponse.

    Imports are local to avoid circular dependencies at module load time.
    """
    from rasvcx.api.models_enriched import limitations_notice
    from rasvcx.schemas.decision import DecisionAction

    decision = DecisionModel(
        action=DecisionActionEnum(result.decision.action.value),
        decision=result.decision.action.canonical,
        confidence=result.decision.confidence,
        rationale=result.decision.rationale,
    )

    pipeline_error = None
    if result.pipeline_error is not None:
        pipeline_error = PipelineErrorModel(
            stage=result.pipeline_error.stage,
            message=result.pipeline_error.message,
            is_retryable=result.pipeline_error.is_retryable,
        )

    risk_profile = None
    if hasattr(result, "risk_profile") and result.risk_profile is not None:
        risk_profile = RiskProfileModel(
            overall_risk_score=result.risk_profile.overall_risk_score,
            validation_depth=result.risk_profile.validation_depth.value,
        )

    has_answer = result.decision.action in (
        DecisionAction.ANSWER, DecisionAction.WARNING
    )

    return QueryResponse(
        query_id=str(result.query_id),
        success=result.success,
        decision=decision,
        # Text the pipeline declined to stand behind is never returned.
        generated_text=(result.generated_text or "") if has_answer else "",
        risk_profile=risk_profile,
        pipeline_error=pipeline_error,
        corrective_attempts=result.corrective_attempts,
        request_id=request_id,
        has_answer=has_answer,
        mock_llm=mock_llm,
        limitations=limitations_notice(mock_llm),
        execution_mode=execution_mode,
        offline=(execution_mode == "offline_test") if execution_mode else None,
        kb_version_id=getattr(result, "kb_version_id", None),
        total_latency_ms=round(getattr(result, "total_seconds", 0.0) * 1000.0, 3),
        calibration_status=getattr(result, "calibration_status", None),
        calibration_version=getattr(result, "calibration_version", None),
        calibration_dataset_hash=getattr(result, "calibration_dataset_hash", None),
        warnings=list(getattr(result, "warnings", ())),
    )


__all__ = [
    "QUERY_CONTEXT_KEYS",
    "QueryRequest",
    "QueryResponse",
    "DecisionModel",
    "DecisionActionEnum",
    "PipelineErrorModel",
    "RiskProfileModel",
    "HealthResponse",
    "ComponentState",
    "ComponentStatus",
    "ReadinessResponse",
    "AdminConfigResponse",
    "ErrorResponse",
    "pipeline_result_to_response",
]