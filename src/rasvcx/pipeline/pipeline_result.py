"""Pipeline-level result and error contracts (Module 12).

PipelineResult is the single output of PipelineOrchestrator.run().
It carries the complete outcome of one pipeline pass — success,
failure, or abstention — without duplicating any M1–M11 contract.
Every field either references an existing frozen contract from M1–M11
or is a minimal scalar that M12 itself owns.

PipelineError represents a stage failure that prevented the pipeline
from reaching the decision stage.  It is NOT a replacement for
GenerationError (M11) or DecisionAction.ABSTAIN (M10) — those are
stage-level outcomes that flow normally.  PipelineError captures
structural/operational failures: retrieval backend down, M8 raised
an unexpected exception, etc.

Stage identification uses plain ``str`` values drawn from the
canonical ``PipelineStage`` Literal defined in ``schemas.common``.
No duplicate enum is introduced.

No existing M1–M10 contract is modified, duplicated, or replaced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rasvcx.generation.generation_types import GenerationError
from rasvcx.schemas.common import QueryId
from rasvcx.schemas.decision import Decision, DecisionAction
from rasvcx.schemas.evidence import EvidenceItem
from rasvcx.schemas.query import RiskProfile
from rasvcx.validation.verified_context import ValidationSummary
from rasvcx.verification import VerificationSummary


# ---------------------------------------------------------------------------
# Pipeline error
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineError:
    """Structural/operational failure that halted the pipeline.

    Distinguished from:
      - GenerationError (M11): expected generation failure, flows normally
      - DecisionAction.ABSTAIN (M10): a valid decision outcome, not an error
      - VerificationSummary (M9): verification completes even on bad answers

    PipelineError means the pipeline COULD NOT reach a decision at all.

    ``stage`` is a plain ``str`` matching one of the canonical
    ``PipelineStage`` literal values defined in ``schemas.common``
    (e.g. ``"risk_routing"``, ``"generation"``).  No duplicate stage
    enum is introduced — the canonical Literal is the single taxonomy.
    """

    stage: str
    message: str
    is_retryable: bool

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("PipelineError.stage must be non-empty")
        if not self.message:
            raise ValueError("PipelineError.message must be non-empty")


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StageTrace:
    """One executed (or explicitly skipped) pipeline stage.

    The trace is a record of what actually ran for this request, in order.
    A stage appears here only because the orchestrator invoked it (or
    decided not to and says why) -- it is never derived from configuration.

    status:  "ok" | "failed" | "skipped"
    attempt: 0 for the first pass; n for the n-th corrective pass.
    """

    stage: str
    status: str
    elapsed_ms: float
    detail: str | None = None
    attempt: int = 0


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Immutable output of one pipeline pass.

    On a successful run (pipeline_error is None):
      - decision is the M10 Decision (ANSWER/WARNING/REPAIR/REGENERATE/ABSTAIN)
      - generated_text carries the answer (empty on generation failure
        or abstention)
      - risk_profile, validation_summary, verification_summary carry
        stage outputs for observability and research auditability

    On a pipeline-level failure (pipeline_error is not None):
      - decision is a synthetic ABSTAIN with confidence 0.0
      - the error describes which stage failed and why

    corrective_attempts mirrors the final authoritative count
    maintained by the existing EvidenceBundle contract through its
    ``record_corrective_attempt()`` mechanism.  The orchestrator
    reads the bundle's authoritative count and copies it here so
    that downstream consumers can inspect it on the frozen result
    without requiring access to the mutable bundle.
    """

    query_id: QueryId
    decision: Decision
    generated_text: str = ""
    generation_error: GenerationError | None = None
    verification_summary: VerificationSummary | None = None
    validation_summary: ValidationSummary | None = None
    risk_profile: RiskProfile | None = None
    corrective_attempts: int = 0
    pipeline_error: PipelineError | None = None
    # -- observability / auditability (all optional, default-empty) --------
    evidence: tuple[EvidenceItem, ...] = ()
    """Evidence items the decision was based on, in final (reranked) order."""
    provenance: object | None = None
    """provenance.ProvenanceAnalysisResult, when the stage ran."""
    cited_item_ids: frozenset[str] = frozenset()
    candidate_pairs: dict[str, tuple[str, str]] = field(default_factory=dict)
    """candidate_id -> (evidence id a, evidence id b) for every pair M8
    validated, so a resolution can be shown against the items it compares."""
    trace: tuple[StageTrace, ...] = ()
    retrieval: dict[str, object] = field(default_factory=dict)
    """What retrieval executed: per-retriever hit counts, KB version."""
    kb_version_id: str | None = None
    total_seconds: float = 0.0
    nli_calls: int = 0
    calibration_status: str | None = None
    model_id: str | None = None
    warnings: tuple[str, ...] = ()
    evidence_assessments: dict[str, object] = field(default_factory=dict)
    """item_id -> provenance.evidence_roles.EvidenceAssessment (role +
    temporal state) for every evidence item."""
    calibration_version: str | None = None
    calibration_dataset_hash: str | None = None
    provider_stats: dict[str, object] = field(default_factory=dict)
    """llm_calls, http_attempts, retries, retry_wait_seconds,
    provider_seconds, failed_calls, statuses, quota (see orchestrator)."""

    def __post_init__(self) -> None:
        if self.corrective_attempts < 0:
            raise ValueError(
                f"corrective_attempts must be >= 0, got {self.corrective_attempts}"
            )

    @property
    def success(self) -> bool:
        """True when the pipeline reached a decision without structural failure."""
        return self.pipeline_error is None

    @property
    def has_answer(self) -> bool:
        """True when the decision is ANSWER or WARNING (an answer is returned)."""
        return self.decision.action in (DecisionAction.ANSWER, DecisionAction.WARNING)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_error_result(
    query_id: QueryId,
    error: PipelineError,
    risk_profile: RiskProfile | None = None,
    validation_summary: ValidationSummary | None = None,
    corrective_attempts: int = 0,
    **observability: object,
) -> PipelineResult:
    """Construct a PipelineResult for a pipeline-level failure.

    Creates a synthetic ABSTAIN decision with confidence 0.0 and the
    error's stage + message as the rationale.  This is a convenience
    factory, not a new contract — it returns a standard PipelineResult.
    """
    return PipelineResult(
        query_id=query_id,
        decision=Decision(
            action=DecisionAction.ABSTAIN,
            confidence=0.0,
            rationale=f"pipeline_error at {error.stage}: {error.message}",
        ),
        risk_profile=risk_profile,
        validation_summary=validation_summary,
        corrective_attempts=corrective_attempts,
        pipeline_error=error,
        **observability,  # type: ignore[arg-type]
    )


__all__ = [
    "PipelineError",
    "PipelineResult",
    "StageTrace",
    "make_error_result",
]