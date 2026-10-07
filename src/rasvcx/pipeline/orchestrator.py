"""Pipeline orchestrator (Module 12).

Connects M2-M11 into a deterministic pipeline:

    QueryRequest
      -> M2  risk routing
      -> retrieval (injected callable: BM25 + dense + RRF)
      -> M4  reranking
      -> M5  sufficiency gate (with targeted-retrieval loop)
      -> M6  provenance / context analysis
      -> M7  atomic claim extraction
      -> M8  evidence validation
      -> M11 generation (with repair feedback on REPAIR loops)
      -> M9  post-generation verification
      -> M10 confidence + decision
      -> PipelineResult

M12 is a thin orchestrator.  All intelligence lives inside M2-M11.

Design decisions:
  - Retrieval is injected as a callable bound to ONE knowledge-base
    snapshot.  run() accepts the callables of the snapshot the request has
    leased; every retrieval in the request (initial, targeted, REGENERATE
    re-entry) uses that same snapshot.
  - Targeted retrieval is a separate optional callable; when None,
    targeted retrieval recommendations are treated as unactionable
    (safe-fail, not silent continue).
  - Corrective loops (REPAIR/REGENERATE) are bounded by
    max_corrective_attempts (default 2).
  - REPAIR derives feedback from M9's actual VerificationSummary
    and passes it through Generator.generate(verification_feedback=...).
  - REGENERATE->RETRIEVAL re-enters the full safety path on a fresh
    bundle (M7/M8 state is recomputed, not carried over stale).
  - CONSERVATIVE sufficiency (high-risk + ambiguous evidence) attempts
    targeted retrieval if recommended and available, then re-checks;
    if still not SUFFICIENT, the pipeline safe-fails.
  - The authoritative corrective-attempt counter lives on the
    EvidenceBundle (record_corrective_attempt()).  Retrieval calls are
    counted by the retrieval callables themselves, once per call.
  - All per-request state lives in a _Run object local to run().  The
    orchestrator instance holds only immutable collaborators, so one
    instance is safe to call from many worker threads at once.
  - Every stage is timed and recorded in the result's execution trace.

No existing M1-M11 contract is modified.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rasvcx.claims import AtomicClaimPipeline
from rasvcx.confidence import ConfidencePipeline
from rasvcx.decision import DecisionEngine
from rasvcx.generation import Generator
from rasvcx.pipeline.pipeline_result import (
    PipelineError,
    PipelineResult,
    StageTrace,
    make_error_result,
)
from rasvcx.provenance import ProvenanceAnalyzer
from rasvcx.provenance.evidence_roles import (
    NOT_CURRENT,
    EvidenceAssessment,
    TemporalStatus,
    assess_evidence,
)
from rasvcx.reranking import RerankingService
from rasvcx.routing import route_query
from rasvcx.schemas.common import Unknown
from rasvcx.security.prompt_injection_guard import detect_instruction_override
from rasvcx.schemas.decision import CorrectiveTarget, Decision, DecisionAction
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.query import QueryRequest, RiskProfile
from rasvcx.schemas.verification import (
    CitationStatus,
    SupportLabel,
    VerificationReasonCode,
    VerificationSummary,
)
from rasvcx.sufficiency import SufficiencyGateConfig, SufficiencyVerdict, evaluate_sufficiency
from rasvcx.validation import ValidationPipeline, ValidationSummary
from rasvcx.validation.nli_interface import NLIService
from rasvcx.verification import VerificationPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callable type aliases for injected retrieval
# ---------------------------------------------------------------------------

RetrievalFn = Callable[[QueryRequest, RiskProfile, EvidenceBundle], None]
"""Primary retrieval callable: populates bundle.evidence_items in place.
Raises on failure (M12 catches operational errors)."""

TargetedRetrievalFn = Callable[[QueryRequest, RiskProfile, EvidenceBundle], None]
"""Targeted retrieval callable: adds supplementary evidence items to the
bundle.  Called when M5 recommends targeted retrieval and this callable
is provided.  Raises on failure (M12 catches operational errors)."""


# ---------------------------------------------------------------------------
# Repair feedback derivation (from actual M9 contracts)
# ---------------------------------------------------------------------------


def _build_repair_feedback(vs: VerificationSummary) -> str:
    """Derive a repair instruction string from M9's verification findings.

    Uses the exact ClaimVerificationResult.label, .rationale, .claim_id
    and CitationResult.status, .rationale fields defined in
    schemas/verification.py.  Does not invent any fields.

    Returns an empty string when there are no actionable issues (the
    caller treats empty feedback the same as None).
    """
    issues: list[str] = []
    for r in vs.claim_results:
        if r.label in (SupportLabel.UNSUPPORTED, SupportLabel.CONTRADICTED):
            issues.append(
                f"- Claim {r.claim_id} [{r.label.value}]: {r.rationale}"
            )
    for c in vs.citation_results:
        if c.status in (
            CitationStatus.MISSING,
            CitationStatus.INCORRECT,
            CitationStatus.CONTRADICTORY,
        ):
            issues.append(
                f"- Citation for claim {c.claim_id} [{c.status.value}]: {c.rationale}"
            )
    return "\n".join(issues)


_CURRENT_QUESTION_RE = __import__("re").compile(
    r"\b(?:current(?:ly)?|latest|now|today|at\s+present|up[- ]to[- ]date|most\s+recent)\b",
    __import__("re").IGNORECASE,
)


def _asks_for_current(query: QueryRequest) -> bool:
    """The question explicitly asks about the present state."""
    sensitivity = str((getattr(query, "metadata", None) or {}).get("time_sensitivity", "")).lower()
    return sensitivity in ("current", "high") or bool(_CURRENT_QUESTION_RE.search(query.raw_text))


def _record_nli(run: "_Run", where: str) -> None:
    """Record NLI model inference time accumulated on this thread as its
    own trace entry (it is nested inside the named enclosing stage)."""
    seconds, calls = NLIService.take_thread_timing()
    if calls:
        run.record("nli_inference", "ok", seconds, f"{where}; calls={calls}")


def _context_mismatched_support(run: "_Run", assessed: dict) -> list[str]:
    """Supporting items whose population / jurisdiction / dosage context is
    a KNOWN mismatch with the context the question stated (M6 analysis).
    The temporal check is excluded: it reflects document age, which the
    lifecycle rules handle."""
    per_item = getattr(run.provenance, "per_item", {}) or {}
    out = []
    for iid, a in per_item.items():
        if not assessed.get(str(iid)) or not assessed[str(iid)].supports_answer:
            continue
        if any(getattr(getattr(a, f, None), "value", None) == "known_mismatch"
               for f in ("population", "jurisdiction", "dosage_context")):
            out.append(str(iid))
    return sorted(out)


# ---------------------------------------------------------------------------
# Per-request state
# ---------------------------------------------------------------------------

# Sub-stages that M8 times internally and records on the bundle; surfaced
# in the trace after validation so each one is individually visible.
_VALIDATION_SUBSTAGES: tuple[str, ...] = (
    "candidate_generation",
    "deterministic_validation",
    "contextual_validation",
    "selective_nli",
    "evidence_resolution",
)


class _Run:
    """Mutable state of ONE pipeline request.  Never shared between requests."""

    def __init__(
        self,
        query: QueryRequest,
        retrieval_fn: RetrievalFn,
        targeted_retrieval_fn: TargetedRetrievalFn | None,
        kb_version_id: str | None,
    ) -> None:
        self.query = query
        self.retrieval_fn = retrieval_fn
        self.targeted_retrieval_fn = targeted_retrieval_fn
        self.kb_version_id = kb_version_id
        self.started = time.perf_counter()
        self.trace: list[StageTrace] = []
        self.warnings: list[str] = []
        self.attempt = 0
        self.risk_profile: RiskProfile | None = None
        self.bundle: EvidenceBundle | None = None
        self.provenance: object | None = None
        self.validation_summary: ValidationSummary | None = None
        self.calibration_status: str | None = None
        self.model_id: str | None = None
        self.cited_item_ids: frozenset[str] = frozenset()
        # True when the sufficiency gate let a high-risk query proceed on a
        # single authoritative source type (diversity waiver).
        self.single_source_waiver = False
        self.verification_summary: VerificationSummary | None = None
        # Provider accounting for this request: every generate() call is
        # one LLM call; each may make several HTTP attempts (retries).
        self.provider = {"llm_calls": 0, "http_attempts": 0, "retries": 0,
                         "retry_wait_seconds": 0.0, "provider_seconds": 0.0,
                         "failed_calls": 0, "statuses": [], "quota": None}
        self.min_relevance_score = 0.0
        self.calibration_version: str | None = None
        self.calibration_dataset_hash: str | None = None

    def assessments(self) -> dict[str, EvidenceAssessment]:
        bundle = self.bundle
        if bundle is None:
            return {}
        return assess_evidence(
            bundle.evidence_items.values(), self.verification_summary,
            self.validation_summary,
            {str(c.candidate_id): (str(c.item_id_a), str(c.item_id_b))
             for c in bundle.conflict_candidates},
            self.min_relevance_score,
        )

    def _evidence_warnings(self, assessed: dict[str, EvidenceAssessment]) -> None:
        """Provenance warnings about the evidence the ANSWER relies on.

        A retrieved passage that does not support the answer is not the
        answer's evidence: its age, population or jurisdiction must not
        produce a warning about the answer.
        """
        supporting = {iid for iid, a in assessed.items() if a.supports_answer}
        if not supporting:
            return
        prov = self.provenance
        per_item = getattr(prov, "per_item", {}) or {}
        for field_name, message in (
            ("temporal", "supporting evidence may be outdated (publication date is old for this question)"),
            ("population", "supporting evidence is for a different population than the question states"),
            ("jurisdiction", "supporting evidence is from a different jurisdiction than the question states"),
            ("dosage_context", "supporting evidence is for a different dosage context than the question states"),
        ):
            ids = sorted(
                str(iid) for iid, a in per_item.items()
                if str(iid) in supporting and getattr(a, field_name).value == "known_mismatch"
            )
            if ids:
                self.warn(f"{message}: {', '.join(ids)}")
        stale = sorted(
            iid for iid in supporting if assessed[iid].temporal_status not in (
                TemporalStatus.CURRENT, TemporalStatus.UNKNOWN, TemporalStatus.UNDATED)
        )
        for iid in stale:
            self.warn(f"supporting evidence {iid}: {assessed[iid].temporal_reason}")
        items = [self.bundle.evidence_items[i] for i in self.bundle.evidence_items if str(i) in supporting]  # type: ignore[union-attr]
        incomplete = sum(
            1 for it in items
            if any(
                isinstance(v, Unknown)
                for v in (
                    it.provenance.date, it.provenance.jurisdiction,
                    it.provenance.population, it.provenance.dosage_context,
                )
            )
        )
        if items and incomplete / len(items) > 0.5:
            self.warn(
                f"{incomplete} of {len(items)} supporting evidence items have incomplete "
                "provenance (date, jurisdiction, population or dosage context unknown)"
            )

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, str | None]]:
        """Time one stage and append it to the trace.

        The yielded dict lets the stage attach a short detail string.
        A stage that raises is recorded as failed and the exception
        propagates to the caller's handler.
        """
        info: dict[str, str | None] = {"detail": None}
        t0 = time.perf_counter()
        try:
            yield info
        except Exception as exc:
            self.record(name, "failed", time.perf_counter() - t0, f"{type(exc).__name__}: {exc}")
            raise
        self.record(name, "ok", time.perf_counter() - t0, info["detail"])

    def record(self, name: str, status: str, seconds: float, detail: str | None = None) -> None:
        self.trace.append(
            StageTrace(
                stage=name, status=status, elapsed_ms=seconds * 1000.0,
                detail=detail, attempt=self.attempt,
            )
        )

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def observability(self) -> dict[str, object]:
        """Fields every PipelineResult of this run carries, success or not."""
        bundle = self.bundle
        assessed = self.assessments()
        self._evidence_warnings(assessed)
        return {
            "provider_stats": dict(self.provider),
            "evidence_assessments": assessed,
            "calibration_version": self.calibration_version,
            "calibration_dataset_hash": self.calibration_dataset_hash,
            "evidence": tuple(bundle.evidence_items.values()) if bundle else (),
            "provenance": self.provenance,
            "cited_item_ids": self.cited_item_ids,
            "candidate_pairs": {
                str(c.candidate_id): (str(c.item_id_a), str(c.item_id_b))
                for c in (bundle.conflict_candidates if bundle else ())
            },
            "trace": tuple(self.trace),
            "retrieval": dict(bundle.retrieval_trace) if bundle else {},
            "kb_version_id": self.kb_version_id,
            "total_seconds": time.perf_counter() - self.started,
            "nli_calls": bundle.metadata.nli_calls if bundle else 0,
            "calibration_status": self.calibration_status,
            "model_id": self.model_id,
            "warnings": tuple(self.warnings),
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Module 12: end-to-end pipeline orchestrator.

    All collaborators are injected.  Constructing a PipelineOrchestrator
    performs no model loading, no network access, no retrieval.

    max_corrective_attempts bounds the total number of REPAIR/REGENERATE
    loops across the entire pipeline run.  Default 2.  Set to 0 to
    disable corrective loops entirely.  The authoritative counter is
    maintained on the EvidenceBundle via record_corrective_attempt().

    sufficiency_config: optional SufficiencyGateConfig forwarded to
    evaluate_sufficiency() at every call site.  When None (the default),
    evaluate_sufficiency() uses its own SufficiencyGateConfig() default
    (min_scored_items=1).  The factory injects a config built from
    Settings so offline_test mode can set min_scored_items=0.

    provenance_analyzer: M6 analyzer run after the sufficiency gate.  When
    None the stage is recorded in the trace as skipped -- it is never
    reported as executed.

    retrieval_fn / targeted_retrieval_fn given here are the defaults; run()
    accepts per-request overrides bound to a leased KB snapshot.
    """

    def __init__(
        self,
        retrieval_fn: RetrievalFn,
        reranking_service: RerankingService,
        claim_pipeline: AtomicClaimPipeline,
        validation_pipeline: ValidationPipeline,
        generator: Generator,
        verification_pipeline: VerificationPipeline,
        confidence_pipeline: ConfidencePipeline,
        decision_engine: DecisionEngine,
        max_corrective_attempts: int = 2,
        targeted_retrieval_fn: TargetedRetrievalFn | None = None,
        sufficiency_config: SufficiencyGateConfig | None = None,
        provenance_analyzer: ProvenanceAnalyzer | None = None,
    ) -> None:
        if max_corrective_attempts < 0:
            raise ValueError(
                f"max_corrective_attempts must be >= 0, got {max_corrective_attempts}"
            )
        self._retrieval_fn = retrieval_fn
        self._targeted_retrieval_fn = targeted_retrieval_fn
        self._reranking = reranking_service
        self._claims = claim_pipeline
        self._validation = validation_pipeline
        self._generator = generator
        self._verification = verification_pipeline
        self._confidence = confidence_pipeline
        self._decision = decision_engine
        self._max_corrective = max_corrective_attempts
        self._sufficiency_config = sufficiency_config
        self._provenance = provenance_analyzer

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        query: QueryRequest,
        *,
        retrieval_fn: RetrievalFn | None = None,
        targeted_retrieval_fn: TargetedRetrievalFn | None = None,
        kb_version_id: str | None = None,
    ) -> PipelineResult:
        """Execute the full RASVC-X pipeline for one query.

        retrieval_fn / targeted_retrieval_fn: the callables of the KB
        snapshot this request has leased.  When retrieval_fn is given,
        targeted_retrieval_fn is taken from the same call (None disables
        targeted retrieval) -- the two are never mixed across snapshots.
        When omitted, the orchestrator's construction-time defaults are used.

        Returns a PipelineResult on every path.  Never raises for
        expected operational failures.
        """
        if retrieval_fn is None:
            retrieval_fn = self._retrieval_fn
            targeted_retrieval_fn = self._targeted_retrieval_fn
        run = _Run(query, retrieval_fn, targeted_retrieval_fn, kb_version_id)
        NLIService.take_thread_timing()  # discard anything left on this worker thread
        run.min_relevance_score = float(
            getattr(getattr(self._reranking, "config", None), "min_relevance_score", None) or 0.0
        )
        qid = query.query_id

        # Stage 0: Input validation -- a "question" that is an attempt to
        # redirect the model is rejected before anything is retrieved.
        t_guard = time.perf_counter()
        rule = detect_instruction_override(query.raw_text)
        if rule is not None:
            run.record("input_validation", "ok", time.perf_counter() - t_guard, f"rejected: {rule}")
            run.warn("the request was rejected as a prompt-injection attempt, not a medical question")
            return make_error_result(
                qid,
                PipelineError(
                    stage="input_validation",
                    message=f"query rejected by prompt-injection guard ({rule})",
                    is_retryable=False,
                ),
                **run.observability(),
            )
        run.record("input_validation", "ok", time.perf_counter() - t_guard, "accepted")

        # Stage 1: Risk routing
        try:
            with run.stage("risk_routing") as info:
                risk_profile = route_query(query.normalized_text)
                info["detail"] = (
                    f"risk={risk_profile.overall_risk_score:.2f} "
                    f"depth={risk_profile.validation_depth.value} "
                    f"nli_allowance={risk_profile.nli_call_allowance}"
                )
        except Exception as exc:
            return make_error_result(
                qid,
                PipelineError(
                    stage="risk_routing",
                    message=str(exc) or "risk routing failed",
                    is_retryable=False,
                ),
                **run.observability(),
            )
        run.risk_profile = risk_profile

        # Evidence acquisition -> validation
        run.bundle = EvidenceBundle(query_id=qid, risk_profile=risk_profile)
        failure = self._acquire_and_validate(run)
        if failure is not None:
            return failure

        # Generation -> verification -> confidence -> decision
        return self._generate_verify_decide_loop(run)

    # ------------------------------------------------------------------
    # Evidence acquisition: retrieval -> reranking -> sufficiency ->
    #                       provenance -> claims -> validation
    # ------------------------------------------------------------------

    def _fail(self, run: _Run, error: PipelineError) -> PipelineResult:
        assert run.bundle is not None
        return make_error_result(
            run.query.query_id, error,
            risk_profile=run.risk_profile,
            validation_summary=run.validation_summary,
            corrective_attempts=run.bundle.metadata.corrective_attempts_used,
            **run.observability(),
        )

    def _acquire_and_validate(self, run: _Run) -> PipelineResult | None:
        """Run retrieval through validation on run.bundle.

        Returns a PipelineResult on failure (safe ABSTAIN), or None on
        success (proceed to generation) with run.validation_summary set.
        """
        run.validation_summary = None
        run.verification_summary = None  # item ids of a previous bundle mean nothing here
        run.provenance = None
        bundle = run.bundle
        assert bundle is not None

        err = self._run_retrieval(run)
        if err is not None:
            return self._fail(run, err)

        if not bundle.evidence_items:
            return self._fail(
                run,
                PipelineError(
                    stage="hybrid_retrieval",
                    message="retrieval returned no evidence items",
                    is_retryable=False,
                ),
            )

        for stage_runner in (
            self._run_reranking,
            self._run_sufficiency_gate,
            self._run_provenance,
            self._run_claims,
            self._run_validation,
        ):
            err = stage_runner(run)
            if err is not None:
                return self._fail(run, err)

        return None  # success -- proceed to generation

    # ------------------------------------------------------------------
    # Individual stage runners
    # ------------------------------------------------------------------

    def _run_retrieval(self, run: _Run) -> PipelineError | None:
        bundle = run.bundle
        assert bundle is not None and run.risk_profile is not None
        try:
            with run.stage("hybrid_retrieval") as info:
                # The retrieval callable records its own call on the bundle.
                run.retrieval_fn(run.query, run.risk_profile, bundle)
                info["detail"] = f"evidence_items={len(bundle.evidence_items)}"
        except Exception as exc:
            logger.exception("Retrieval failed for query %s", run.query.query_id)
            self._record_retrieval_substages(run)
            return PipelineError(
                stage="hybrid_retrieval",
                message=str(exc) or "retrieval failed",
                is_retryable=True,
            )
        self._record_retrieval_substages(run)
        return None

    @staticmethod
    def _record_retrieval_substages(run: _Run) -> None:
        """Surface what the retrieval bridge reports it executed.

        Entries are created only from the bridge's own record of the calls
        it made, so 'dense_retrieval' appears only if Qdrant was queried.
        """
        assert run.bundle is not None
        rt = run.bundle.retrieval_trace
        if "bm25_seconds" in rt:
            run.record(
                "bm25_retrieval", "ok", float(rt["bm25_seconds"]),  # type: ignore[arg-type]
                f"hits={rt.get('bm25_hits')}",
            )
        if "dense_error" in rt:
            run.record("dense_retrieval", "failed", 0.0, str(rt["dense_error"]))
        elif "dense_seconds" in rt:
            run.record(
                "dense_retrieval", "ok", float(rt["dense_seconds"]),  # type: ignore[arg-type]
                f"hits={rt.get('dense_hits')} collection={rt.get('dense_collection')}",
            )
        if "rrf_seconds" in rt:
            run.record(
                "rrf_fusion", "ok", float(rt["rrf_seconds"]),  # type: ignore[arg-type]
                f"fused={rt.get('rrf_fused')} from_both={rt.get('rrf_from_both')}",
            )

    def _run_reranking(self, run: _Run) -> PipelineError | None:
        bundle = run.bundle
        assert bundle is not None
        if getattr(self._reranking, "enabled", True) is False:
            # offline_test: no cross-encoder exists. The no-op service keeps
            # the bundle's stage bookkeeping; the trace says what happened.
            self._reranking.rerank_bundle(bundle, run.query.normalized_text)
            run.record("reranking", "skipped", 0.0, "reranker disabled (offline_test)")
            return None
        try:
            with run.stage("reranking") as info:
                self._reranking.rerank_bundle(bundle, run.query.normalized_text)
                scored = sum(
                    1 for it in bundle.evidence_items.values() if it.rerank_score is not None
                )
                info["detail"] = f"scored={scored}/{len(bundle.evidence_items)}"
                rerank_error = bundle.retrieval_trace.get("rerank_error")
                if rerank_error:
                    # The service absorbed a model failure; surface it as a
                    # failed stage instead of continuing on unscored evidence.
                    raise RuntimeError(f"cross-encoder failed: {rerank_error}")
            return None
        except Exception as exc:
            logger.exception("Reranking failed for query %s", run.query.query_id)
            return PipelineError(
                stage="reranking",
                message=str(exc) or "reranking failed",
                is_retryable=True,
            )

    def _evaluate_sufficiency(self, run: _Run):  # -> SufficiencyResult
        assert run.bundle is not None and run.risk_profile is not None
        with run.stage("sufficiency_gate") as info:
            sufficiency = evaluate_sufficiency(
                run.bundle, run.risk_profile, config=self._sufficiency_config
            )
            info["detail"] = sufficiency.verdict.value + "".join(
                f"; {note}" for note in sufficiency.notes
            )
        # The latest evaluation decides: a re-entry that finds corroborating
        # sources clears the flag.
        run.single_source_waiver = sufficiency.diversity_waived
        return sufficiency

    def _run_sufficiency_gate(self, run: _Run) -> PipelineError | None:
        """Evaluate sufficiency, attempt targeted retrieval if recommended
        and available, then re-evaluate.

        M5 semantics (from source):
          SUFFICIENT   -> proceed
          INSUFFICIENT -> targeted retrieval recommended
          CONSERVATIVE -> high-risk + ambiguous evidence; targeted retrieval
                         recommended if not already used

        Both INSUFFICIENT and CONSERVATIVE set recommend_targeted_retrieval
        when targeted retrieval has not been used.  After targeted retrieval,
        sufficiency is re-evaluated; if still not SUFFICIENT, pipeline
        safe-fails.
        """
        bundle = run.bundle
        assert bundle is not None and run.risk_profile is not None
        try:
            sufficiency = self._evaluate_sufficiency(run)
        except Exception as exc:
            logger.exception("Sufficiency evaluation failed for query %s", run.query.query_id)
            return PipelineError(
                stage="sufficiency_gate",
                message=str(exc) or "sufficiency evaluation failed",
                is_retryable=False,
            )

        if sufficiency.verdict is SufficiencyVerdict.SUFFICIENT:
            return None

        # INSUFFICIENT or CONSERVATIVE: attempt targeted retrieval if
        # recommended and a targeted retrieval callable was provided.
        if sufficiency.recommend_targeted_retrieval and run.targeted_retrieval_fn is not None:
            try:
                with run.stage("targeted_retrieval") as info:
                    before = len(bundle.evidence_items)
                    run.targeted_retrieval_fn(run.query, run.risk_profile, bundle)
                    # Idempotent flag: M5 must not recommend a second pass.
                    bundle.record_targeted_retrieval_used()
                    info["detail"] = f"added={len(bundle.evidence_items) - before}"
            except Exception as exc:
                logger.warning(
                    "Targeted retrieval failed for query %s: %s", run.query.query_id, exc,
                )
                run.warn(f"targeted retrieval failed: {type(exc).__name__}")
                # Targeted retrieval failure is not fatal -- fall through to
                # the re-evaluation which will safe-fail on the original evidence.
            if bundle.retrieval_trace.get("targeted_dense_error"):
                run.warn(
                    "targeted retrieval ran BM25-only: dense retrieval failed "
                    f"({bundle.retrieval_trace['targeted_dense_error']})"
                )

            # Re-rank after targeted retrieval added new items
            rerank_err = self._run_reranking(run)
            if rerank_err is not None:
                return rerank_err

            # Re-evaluate sufficiency
            try:
                sufficiency = self._evaluate_sufficiency(run)
            except Exception as exc:
                logger.exception(
                    "Sufficiency re-evaluation failed for query %s", run.query.query_id
                )
                return PipelineError(
                    stage="sufficiency_gate",
                    message=str(exc) or "sufficiency re-evaluation failed",
                    is_retryable=False,
                )

            if sufficiency.verdict is SufficiencyVerdict.SUFFICIENT:
                return None

        # Still not sufficient after targeted retrieval (or targeted
        # retrieval was not available / not recommended).
        return PipelineError(
            stage="sufficiency_gate",
            message=(
                f"evidence {sufficiency.verdict.value}: "
                f"{'; '.join(sufficiency.reasons)}"
            ),
            is_retryable=False,
        )

    def _run_provenance(self, run: _Run) -> PipelineError | None:
        bundle = run.bundle
        assert bundle is not None and run.risk_profile is not None
        if self._provenance is None:
            run.record("provenance_context", "skipped", 0.0, "no provenance analyzer configured")
            return None
        try:
            with run.stage("provenance_context") as info:
                result = self._provenance.analyze(bundle, run.query, run.risk_profile)
                info["detail"] = (
                    f"sources={result.unique_source_count} "
                    f"unknown_ratio={result.unknown_provenance_ratio:.2f} "
                    f"mismatch={result.mismatch_present}"
                )
        except Exception as exc:
            logger.exception("Provenance analysis failed for query %s", run.query.query_id)
            return PipelineError(
                stage="provenance_context",
                message=str(exc) or "provenance analysis failed",
                is_retryable=False,
            )
        run.provenance = result
        # Mismatch / incomplete-provenance warnings are emitted when the
        # result is built, and only for evidence the answer relies on
        # (_Run._evidence_warnings): a retrieved passage that does not
        # support the answer must not produce a warning about it.
        return None

    def _run_claims(self, run: _Run) -> PipelineError | None:
        bundle = run.bundle
        assert bundle is not None
        try:
            with run.stage("atomic_claim_extraction") as info:
                self._claims.run(bundle)
                info["detail"] = f"claims={len(bundle.claims)}"
            return None
        except Exception as exc:
            logger.exception("Claim extraction failed")
            return PipelineError(
                stage="atomic_claim_extraction",
                message=str(exc) or "claim extraction failed",
                is_retryable=False,
            )

    def _run_validation(self, run: _Run) -> PipelineError | None:
        bundle = run.bundle
        assert bundle is not None and run.risk_profile is not None
        try:
            with run.stage("verified_context") as info:
                summary = self._validation.run(bundle, run.query, run.risk_profile)
                info["detail"] = (
                    f"candidates={summary.candidates_generated} "
                    f"nli_calls={summary.nli_calls_used} "
                    f"genuine_conflicts={summary.genuine_conflict_count} "
                    f"unresolved={summary.unresolved_count}"
                )
        except Exception as exc:
            logger.exception("Validation failed for query %s", run.query.query_id)
            return PipelineError(
                stage="verified_context",
                message=str(exc) or "validation failed",
                is_retryable=False,
            )
        run.validation_summary = summary
        _record_nli(run, "m8 evidence-pair NLI (inside verified_context)")
        if summary.nli_failures:
            run.warn(
                f"NLI produced no signal for {summary.nli_failures} escalated "
                "evidence pair(s); those pairs were resolved without semantic "
                "validation"
            )

        # M8 runs these inside one call and times each on the bundle.
        elapsed = bundle.metadata.elapsed_per_stage
        for sub in _VALIDATION_SUBSTAGES:
            if sub not in elapsed:
                continue
            detail = None
            if sub == "selective_nli":
                detail = (
                    f"nli_calls={summary.nli_calls_used}"
                    if summary.nli_calls_used
                    else "policy selected no pair for NLI"
                )
            elif sub == "candidate_generation":
                detail = f"candidates={summary.candidates_generated}"
            elif sub == "evidence_resolution":
                detail = f"resolutions={len(summary.resolutions)}"
            # selective_nli with zero calls did not run an NLI model: the
            # router decided no pair needed it.  Report that as skipped.
            status = "skipped" if sub == "selective_nli" and not summary.nli_calls_used else "ok"
            run.record(sub, status, elapsed[sub], detail)  # type: ignore[index]
        return None

    # ------------------------------------------------------------------
    # Generation -> verification -> confidence -> decision loop
    # ------------------------------------------------------------------

    def _result(
        self,
        run: _Run,
        decision: Decision,
        generated_text: str = "",
        verification_summary: VerificationSummary | None = None,
        generation_error: object | None = None,
    ) -> PipelineResult:
        assert run.bundle is not None
        return PipelineResult(
            query_id=run.query.query_id,
            decision=decision,
            generated_text=generated_text,
            generation_error=generation_error,  # type: ignore[arg-type]
            verification_summary=verification_summary,
            validation_summary=run.validation_summary,
            risk_profile=run.risk_profile,
            corrective_attempts=run.bundle.metadata.corrective_attempts_used,
            **run.observability(),  # type: ignore[arg-type]
        )

    def _generate_verify_decide_loop(self, run: _Run) -> PipelineResult:
        query = run.query
        qid = query.query_id
        risk_profile = run.risk_profile
        assert risk_profile is not None
        verification_feedback: str | None = None

        while True:
            bundle = run.bundle
            validation_summary = run.validation_summary
            assert bundle is not None
            attempts_used = bundle.metadata.corrective_attempts_used
            run.attempt = attempts_used

            # Generation
            t_gen = time.perf_counter()
            try:
                gen_result = self._generator.generate(
                    query, bundle, validation_summary,
                    verification_feedback=verification_feedback,
                )
            except Exception as exc:
                run.record(
                    "generation", "failed", time.perf_counter() - t_gen,
                    f"{type(exc).__name__}: {exc}",
                )
                return self._fail(
                    run,
                    PipelineError(
                        stage="generation",
                        message=str(exc) or "generation raised unexpected exception",
                        is_retryable=True,
                    ),
                )
            pstats = self._generator.last_provider_stats() if hasattr(
                self._generator, "last_provider_stats") else None
            if pstats:
                p = run.provider
                p["llm_calls"] += 1
                p["http_attempts"] += pstats.get("attempts", 0)
                p["retries"] += max(0, pstats.get("attempts", 0) - 1)
                p["retry_wait_seconds"] += pstats.get("retry_wait_seconds", 0.0)
                p["provider_seconds"] += pstats.get("elapsed_seconds", 0.0)
                p["failed_calls"] += 1 if pstats.get("failed") else 0
                p["statuses"].extend(pstats.get("statuses", []))
                p["quota"] = pstats.get("quota") or p["quota"]
                if pstats.get("retry_wait_seconds"):
                    # Backoff is part of the generation stage's time; it is
                    # also recorded on its own so latency reports can
                    # separate provider waiting from model time.
                    run.record("provider_backoff", "ok", pstats["retry_wait_seconds"],
                               f"retries={pstats.get('attempts', 1) - 1}")
            if gen_result.success:
                run.record(
                    "generation", "ok", time.perf_counter() - t_gen,
                    f"model={gen_result.model_id} "
                    f"chars={len(gen_result.generated_text)} "
                    f"cited={len(gen_result.cited_item_ids)}",
                )
            else:
                # The provider was called (or pre-flight refused) and no
                # answer exists: this is a failed stage, not an executed one.
                run.record(
                    "generation", "failed", time.perf_counter() - t_gen,
                    f"{gen_result.error.code.value}: {gen_result.error.message}",
                )

            # Generation-level failure (M11 returned a structured error)
            if not gen_result.success:
                run.warn(
                    f"generation failed ({gen_result.error.code.value}): "
                    "no answer was produced"
                )
                return self._result(
                    run,
                    Decision(
                        action=DecisionAction.ABSTAIN,
                        confidence=0.0,
                        rationale=(
                            f"generation_error: {gen_result.error.code.value}: "
                            f"{gen_result.error.message}"
                        ),
                    ),
                    generation_error=gen_result.error,
                )
            run.model_id = gen_result.model_id
            run.cited_item_ids = frozenset(str(i) for i in gen_result.cited_item_ids)

            # Verification
            try:
                with run.stage("post_generation_verification") as info:
                    verification_summary = self._verification.verify(
                        gen_result.generated_text,
                        bundle,
                        query=query,
                        validation_summary=validation_summary,
                    )
                    info["detail"] = (
                        f"verdict={verification_summary.answer_verdict.value} "
                        f"claims={len(verification_summary.claim_results)} "
                        f"supported={verification_summary.supported_count} "
                        f"unsupported={verification_summary.unsupported_count} "
                        f"contradicted={verification_summary.contradicted_count} "
                        f"nli_calls={verification_summary.semantic_verification_calls}"
                    )
            except Exception as exc:
                return self._fail(
                    run,
                    PipelineError(
                        stage="post_generation_verification",
                        message=str(exc) or "verification failed",
                        is_retryable=False,
                    ),
                )

            run.verification_summary = verification_summary
            _record_nli(run, "m9 claim NLI (inside post_generation_verification)")

            # M9 escalates individual claims to the NLI model; make that
            # visible as its own entry (the time is part of the stage above).
            m9_calls = verification_summary.semantic_verification_calls
            run.record(
                "semantic_verification",
                "ok" if m9_calls else "skipped",
                0.0,
                f"nli_calls={m9_calls}" if m9_calls
                else "no generated claim required NLI",
            )
            nli_unavailable = sum(
                1 for r in verification_summary.claim_results
                if r.reason_code is VerificationReasonCode.NLI_UNAVAILABLE
            )
            if nli_unavailable:
                run.warn(
                    f"semantic verification was unavailable for {nli_unavailable} "
                    "generated claim(s)"
                )

            # Confidence (+ calibration, applied inside the confidence pipeline)
            try:
                with run.stage("confidence_estimation") as info:
                    calibration_outcome, completeness = self._confidence.compute(
                        bundle, validation_summary, verification_summary,
                        kb_version_id=run.kb_version_id,
                    )
                    info["detail"] = f"score={calibration_outcome.score.value:.3f}"
            except Exception as exc:
                return self._fail(
                    run,
                    PipelineError(
                        stage="confidence_estimation",
                        message=str(exc) or "confidence computation failed",
                        is_retryable=False,
                    ),
                )
            run.calibration_status = calibration_outcome.status
            run.calibration_version = calibration_outcome.calibration_version
            run.calibration_dataset_hash = calibration_outcome.calibration_dataset_hash
            if calibration_outcome.status == "invalidated":
                run.warn(f"confidence calibration not applied: {calibration_outcome.reason}")
            run.record(
                "calibration", "ok", 0.0,
                f"status={calibration_outcome.status}"
                + (f" reason={calibration_outcome.reason}" if calibration_outcome.reason else ""),
            )

            # Decision
            try:
                with run.stage("decision") as info:
                    decision = self._decision.decide(
                        calibration_outcome,
                        bundle,
                        risk_profile,
                        validation_summary,
                        verification_summary,
                        completeness,
                    )
                    info["detail"] = (
                        f"{decision.action.canonical} confidence={decision.confidence:.3f}"
                    )
            except Exception as exc:
                return self._fail(
                    run,
                    PipelineError(
                        stage="decision",
                        message=str(exc) or "decision engine failed",
                        is_retryable=False,
                    ),
                )

            # An answer must not rest on evidence its own knowledge base
            # declares withdrawn or superseded.  First try a repair that
            # names the stale items; when the budget is spent, a high-risk
            # answer is withheld and a low-risk one carries a warning.
            # Declared-historical support caps the answer at a warning.
            if decision.action in (DecisionAction.ANSWER, DecisionAction.WARNING):
                assessed = run.assessments()
                stale = sorted(
                    iid for iid, a in assessed.items()
                    if a.supports_answer and a.temporal_status in NOT_CURRENT
                )
                historical = sorted(
                    iid for iid, a in assessed.items()
                    if a.supports_answer and a.temporal_status is TemporalStatus.HISTORICAL
                )
                if stale and attempts_used < self._max_corrective:
                    bundle.record_corrective_attempt()
                    verification_feedback = "\n".join(
                        f"- Evidence {iid} is {assessed[iid].temporal_status.value.lower()} "
                        f"({assessed[iid].temporal_reason}); do not base the answer on it."
                        for iid in stale
                    )
                    run.record("decision", "ok", 0.0, f"repair: answer relied on non-current evidence {stale}")
                    continue
                if stale:
                    thresholds = getattr(self._decision, "_thresholds", None)
                    high_risk = risk_profile.safety_floor_forced or (
                        thresholds is not None
                        and risk_profile.overall_risk_score >= thresholds.high_risk_threshold
                    )
                    decision = Decision(
                        action=DecisionAction.ABSTAIN if high_risk else DecisionAction.WARNING,
                        confidence=decision.confidence,
                        rationale=(
                            f"answer relies on withdrawn/superseded evidence {', '.join(stale)}"
                            + ("; withheld at high risk" if high_risk else "")
                        ),
                    )
                elif decision.action is DecisionAction.ANSWER and (
                    mismatched := _context_mismatched_support(run, assessed)
                ):
                    # The question stated a context (population, jurisdiction,
                    # dosage) and the answer relies on evidence for another
                    # one: answerable, never unqualified.
                    decision = Decision(
                        action=DecisionAction.WARNING, confidence=decision.confidence,
                        corrective_target=decision.corrective_target,
                        contributing_candidate_ids=decision.contributing_candidate_ids,
                        rationale=f"{decision.rationale}; capped: supporting evidence "
                                  f"{', '.join(mismatched)} does not match the question's stated context",
                    )
                elif historical and decision.action is DecisionAction.ANSWER:
                    decision = Decision(
                        action=DecisionAction.WARNING, confidence=decision.confidence,
                        corrective_target=decision.corrective_target,
                        contributing_candidate_ids=decision.contributing_candidate_ids,
                        rationale=f"{decision.rationale}; capped: supporting evidence "
                                  f"{', '.join(historical)} is declared historical",
                    )

            # A question about the CURRENT state ("current dose", "latest
            # guidance", or context time_sensitivity=current/high) answered
            # without any supporting source DECLARED current is answerable
            # but temporally unqualified: never an unqualified ANSWER.
            if decision.action is DecisionAction.ANSWER and _asks_for_current(run.query):
                assessed = run.assessments()
                if not any(a.supports_answer and a.temporal_status is TemporalStatus.CURRENT
                           for a in assessed.values()):
                    decision = Decision(
                        action=DecisionAction.WARNING, confidence=decision.confidence,
                        corrective_target=decision.corrective_target,
                        contributing_candidate_ids=decision.contributing_candidate_ids,
                        rationale=f"{decision.rationale}; capped: the question asks for current "
                                  "information and no supporting source is declared current",
                    )
                    run.warn("the question asks for current information, but no supporting "
                             "source is declared current (lifecycle not recorded)")

            # An answer that rests on a single, uncorroborated source type
            # for a high-risk question is answerable but never unqualified:
            # the strongest outcome it can receive is ANSWER_WITH_WARNING.
            if decision.action is DecisionAction.ANSWER and run.single_source_waiver:
                decision = Decision(
                    action=DecisionAction.WARNING,
                    confidence=decision.confidence,
                    corrective_target=decision.corrective_target,
                    contributing_candidate_ids=decision.contributing_candidate_ids,
                    rationale=(
                        f"{decision.rationale}; capped at ANSWER_WITH_WARNING: high-risk "
                        "answer supported by a single source type (not corroborated)"
                    ),
                )
                run.warn(
                    "this high-risk answer is supported by a single source type and "
                    "is not corroborated by an independent source"
                )

            # Terminal decisions
            if decision.action in (
                DecisionAction.ANSWER,
                DecisionAction.WARNING,
                DecisionAction.ABSTAIN,
            ):
                return self._result(
                    run, decision,
                    generated_text=gen_result.generated_text,
                    verification_summary=verification_summary,
                )

            # Corrective budget check
            if attempts_used >= self._max_corrective:
                logger.warning(
                    "Corrective budget exhausted (%d/%d) for query %s; "
                    "returning ABSTAIN.",
                    attempts_used, self._max_corrective, qid,
                )
                return self._result(
                    run,
                    Decision(
                        action=DecisionAction.ABSTAIN,
                        confidence=decision.confidence,
                        rationale=(
                            f"corrective budget exhausted after {attempts_used} "
                            f"attempts; last action was {decision.action.value} "
                            f"targeting {decision.corrective_target.value}"
                        ),
                    ),
                    generated_text=gen_result.generated_text,
                    verification_summary=verification_summary,
                )

            # Record corrective attempt on the authoritative counter.
            bundle.record_corrective_attempt()

            # REPAIR -> re-generate with verification feedback
            if decision.action is DecisionAction.REPAIR:
                verification_feedback = _build_repair_feedback(verification_summary)
                logger.info(
                    "REPAIR (attempt %d) for query %s: re-generating with "
                    "verification feedback.",
                    bundle.metadata.corrective_attempts_used, qid,
                )
                continue

            # REGENERATE -> VERIFIED_CONTEXT: re-generate
            if (
                decision.action is DecisionAction.REGENERATE
                and decision.corrective_target is CorrectiveTarget.VERIFIED_CONTEXT
            ):
                verification_feedback = None
                logger.info(
                    "REGENERATE->VERIFIED_CONTEXT (attempt %d) for query %s.",
                    bundle.metadata.corrective_attempts_used, qid,
                )
                continue

            # REGENERATE -> RETRIEVAL: full re-entry on the SAME KB snapshot
            if (
                decision.action is DecisionAction.REGENERATE
                and decision.corrective_target is CorrectiveTarget.RETRIEVAL
            ):
                logger.info(
                    "REGENERATE->RETRIEVAL (attempt %d) for query %s: "
                    "re-entering full safety path.",
                    bundle.metadata.corrective_attempts_used, qid,
                )
                run.attempt = bundle.metadata.corrective_attempts_used
                run.bundle = EvidenceBundle(
                    query_id=qid,
                    risk_profile=risk_profile,
                    metadata=bundle.metadata,
                )
                failure = self._acquire_and_validate(run)
                if failure is not None:
                    return failure
                verification_feedback = None
                continue


__all__ = [
    "PipelineOrchestrator",
    "RetrievalFn",
    "TargetedRetrievalFn",
]
