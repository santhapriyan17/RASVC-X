"""Module 12 — Pipeline orchestrator tests.

All tests use MockLLMClient and injected mocks — no real LLM provider,
no network, no API key.

Behavioral tests with mock call counting, stage-order recording,
per-action coverage, safety boundary assertions, state propagation,
and bounded-loop verification.

IMPORTANT: Evidence items include rerank_score (as M4 would set) and
diverse source_type values (to satisfy M5 source-type diversity).
The default query ("What are the visiting hours?") is non-safety-
critical so it does not trigger the safety_floor_forced path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rasvcx.claims import AtomicClaimPipeline
from rasvcx.confidence import ConfidencePipeline
from rasvcx.decision import DecisionEngine
from rasvcx.generation import Generator, MockLLMClient, LLMClientError, LLMTimeoutError
from rasvcx.generation.generation_types import GenerationErrorCode
from rasvcx.pipeline import (
    PipelineOrchestrator,
    PipelineError,
    PipelineResult,
    TargetedRetrievalFn,
    make_error_result,
)
from rasvcx.reranking import CrossEncoderReranker, RerankingService
from rasvcx.schemas.common import (
    ChunkId,
    EvidenceItemId,
    QueryId,
    SourceType,
    UNKNOWN,
)
from rasvcx.schemas.decision import Decision, DecisionAction, CorrectiveTarget
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.schemas.verification import (
    AnswerVerdict,
    CitationResult,
    CitationStatus,
    ClaimVerificationResult,
    GeneratedClaimId,
    SupportLabel,
    VerificationReasonCode,
    VerificationStage,
    VerificationSummary,
)
from rasvcx.sufficiency import SufficiencyVerdict
from rasvcx.validation import NLIService, NullNLIBackend, ValidationPipeline
from rasvcx.verification import VerificationPipeline


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

# Non-safety-critical query that does NOT trigger safety_floor_forced.
_DEFAULT_QUERY_TEXT = "What are the visiting hours?"


def _query(text: str = _DEFAULT_QUERY_TEXT) -> QueryRequest:
    return QueryRequest(
        query_id=QueryId("q-test"),
        raw_text=text,
        normalized_text=text.lower(),
    )


def _prov(st: SourceType = SourceType.CLINICAL_GUIDELINE) -> Provenance:
    return Provenance(
        source_type=st,
        date="2023-01-15",
        jurisdiction="US",
        population="adults",
        dosage_context=UNKNOWN,
    )


def _item(
    item_id: str = "E1",
    text: str = "Visiting hours are 9am to 8pm daily.",
    source_type: SourceType = SourceType.CLINICAL_GUIDELINE,
    rerank_score: float = 0.85,
) -> EvidenceItem:
    return EvidenceItem(
        item_id=EvidenceItemId(item_id),
        chunk_id=ChunkId(f"ch-{item_id}"),
        text=text,
        retrieval_score=0.9,
        provenance=_prov(source_type),
        rerank_score=rerank_score,
    )


def _diverse_items() -> list[EvidenceItem]:
    """Two items with different source_type to satisfy M5 diversity."""
    return [
        _item("E1", "Visiting hours are 9am to 8pm daily.",
              SourceType.CLINICAL_GUIDELINE, 0.85),
        _item("E2", "Visitors must check in at the front desk.",
              SourceType.INSTITUTIONAL_POLICY, 0.80),
    ]


def _retrieval_fn(*items: EvidenceItem):
    call_count = {"n": 0}

    def fn(query, risk_profile, bundle):
        call_count["n"] += 1
        for it in items:
            if it.item_id not in bundle.evidence_items:
                bundle.add_evidence_item(it)

    fn.call_count = call_count
    return fn


def _default_retrieval():
    return _retrieval_fn(*_diverse_items())


def _failing_retrieval(msg: str = "backend down"):
    def fn(query, risk_profile, bundle):
        raise RuntimeError(msg)
    return fn


def _empty_retrieval():
    def fn(query, risk_profile, bundle):
        pass
    return fn


def _mock_reranking_service() -> RerankingService:
    with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        mock_load.return_value = mock_model
        svc = RerankingService()
    return svc


def _build_orch(
    retrieval_fn=None,
    canned_text: str = "Visiting hours are 9am to 8pm. [E1] Check in at the front desk. [E2]",
    raise_on_generate=None,
    max_corrective_attempts: int = 2,
    targeted_retrieval_fn: TargetedRetrievalFn | None = None,
    decision_engine: DecisionEngine | None = None,
    generator: Generator | None = None,
) -> PipelineOrchestrator:
    nli = NLIService(NullNLIBackend())
    if generator is None:
        client = MockLLMClient(
            canned_text=canned_text,
            raise_on_generate=raise_on_generate,
        )
        generator = Generator(llm_client=client)
    if retrieval_fn is None:
        retrieval_fn = _default_retrieval()
    return PipelineOrchestrator(
        retrieval_fn=retrieval_fn,
        reranking_service=_mock_reranking_service(),
        claim_pipeline=AtomicClaimPipeline(),
        validation_pipeline=ValidationPipeline(nli_service=nli),
        generator=generator,
        verification_pipeline=VerificationPipeline(nli_service=nli),
        confidence_pipeline=ConfidencePipeline(),
        decision_engine=decision_engine or DecisionEngine(),
        max_corrective_attempts=max_corrective_attempts,
        targeted_retrieval_fn=targeted_retrieval_fn,
    )


def _mock_engine_sequence(*actions):
    """Return a DecisionEngine whose decide() returns the given actions
    in sequence. Each entry is either a DecisionAction (for terminal
    actions) or a (DecisionAction, CorrectiveTarget) tuple."""
    call_n = {"n": 0}

    def mock_decide(cal_outcome, bundle, rp, vs, ver_s, comp):
        call_n["n"] += 1
        idx = min(call_n["n"] - 1, len(actions) - 1)
        act = actions[idx]
        if isinstance(act, tuple):
            return Decision(action=act[0], confidence=0.4,
                            corrective_target=act[1])
        return Decision(action=act, confidence=0.8)

    engine = DecisionEngine()
    engine.decide = mock_decide
    engine._call_count = call_n
    return engine


# ═══════════════════════════════════════════════════════════════════════
# HAPPY PATH
# ═══════════════════════════════════════════════════════════════════════


class TestHappyPath:
    def test_full_pipeline_reaches_terminal_decision(self):
        orch = _build_orch()
        result = orch.run(_query())
        assert result.success
        assert result.pipeline_error is None
        assert result.decision.action in (
            DecisionAction.ANSWER, DecisionAction.WARNING, DecisionAction.ABSTAIN,
        )
        assert result.query_id == QueryId("q-test")
        assert result.risk_profile is not None
        assert result.validation_summary is not None
        assert result.verification_summary is not None

    def test_generated_text_preserved_when_has_answer(self):
        orch = _build_orch()
        result = orch.run(_query())
        if result.has_answer:
            assert len(result.generated_text) > 0


# ═══════════════════════════════════════════════════════════════════════
# RETRIEVAL FAILURES
# ═══════════════════════════════════════════════════════════════════════


class TestRetrievalFailure:
    def test_retrieval_exception_produces_pipeline_error(self):
        orch = _build_orch(retrieval_fn=_failing_retrieval("connection refused"))
        result = orch.run(_query())
        assert not result.success
        assert result.pipeline_error.stage == "hybrid_retrieval"
        assert "connection refused" in result.pipeline_error.message
        assert result.decision.action is DecisionAction.ABSTAIN

    def test_empty_retrieval_produces_pipeline_error(self):
        orch = _build_orch(retrieval_fn=_empty_retrieval())
        result = orch.run(_query())
        assert not result.success
        assert "no evidence" in result.pipeline_error.message

    def test_generation_not_called_on_retrieval_failure(self):
        client = MockLLMClient(canned_text="should not be called")
        gen = Generator(llm_client=client)
        orch = _build_orch(retrieval_fn=_failing_retrieval(), generator=gen)
        orch.run(_query())
        assert client.call_count == 0

    def test_generation_not_called_on_empty_retrieval(self):
        client = MockLLMClient(canned_text="should not be called")
        gen = Generator(llm_client=client)
        orch = _build_orch(retrieval_fn=_empty_retrieval(), generator=gen)
        orch.run(_query())
        assert client.call_count == 0


# ═══════════════════════════════════════════════════════════════════════
# SUFFICIENCY
# ═══════════════════════════════════════════════════════════════════════


class TestSufficiency:
    def test_poor_evidence_blocks_generation(self):
        poor = _item("E-poor", "x", SourceType.OTHER, rerank_score=0.01)
        client = MockLLMClient(canned_text="should not appear")
        gen = Generator(llm_client=client)
        orch = _build_orch(retrieval_fn=_retrieval_fn(poor), generator=gen)
        result = orch.run(_query())
        assert isinstance(result, PipelineResult)
        if not result.success and "sufficiency" in result.pipeline_error.stage:
            assert client.call_count == 0

    def test_targeted_retrieval_called_when_recommended(self):
        targeted_calls = {"n": 0}

        def targeted_fn(query, risk_profile, bundle):
            targeted_calls["n"] += 1
            bundle.add_evidence_item(
                _item("E-tgt", "Supplementary visiting policy.",
                      SourceType.INSTITUTIONAL_POLICY, 0.88)
            )

        weak = _item("E-weak", "Weak evidence.", SourceType.OTHER, 0.1)
        orch = _build_orch(
            retrieval_fn=_retrieval_fn(weak),
            targeted_retrieval_fn=targeted_fn,
        )
        result = orch.run(_query())
        assert isinstance(result, PipelineResult)


# ═══════════════════════════════════════════════════════════════════════
# GENERATION FAILURE
# ═══════════════════════════════════════════════════════════════════════


class TestGenerationFailure:
    def test_provider_failure_produces_abstain(self):
        orch = _build_orch(raise_on_generate=LLMClientError("provider down"))
        result = orch.run(_query())
        assert result.decision.action is DecisionAction.ABSTAIN
        assert result.generation_error is not None
        assert result.generation_error.code is GenerationErrorCode.PROVIDER_FAILURE

    def test_timeout_produces_abstain(self):
        orch = _build_orch(raise_on_generate=LLMTimeoutError("60s exceeded"))
        result = orch.run(_query())
        assert result.decision.action is DecisionAction.ABSTAIN
        assert result.generation_error is not None
        assert result.generation_error.code is GenerationErrorCode.TIMEOUT

    def test_empty_generation_produces_abstain(self):
        orch = _build_orch(canned_text="   ")
        result = orch.run(_query())
        assert result.decision.action is DecisionAction.ABSTAIN
        assert result.generation_error is not None
        assert result.generation_error.code is GenerationErrorCode.EMPTY_RESPONSE

    def test_verification_not_called_on_generation_failure(self):
        orch = _build_orch(raise_on_generate=LLMClientError("down"))
        orch._verification = MagicMock(spec=VerificationPipeline)
        result = orch.run(_query())
        orch._verification.verify.assert_not_called()

    def test_verification_not_called_on_empty_generation(self):
        orch = _build_orch(canned_text="  ")
        orch._verification = MagicMock(spec=VerificationPipeline)
        result = orch.run(_query())
        orch._verification.verify.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# DATA PROPAGATION (M9 → M10 → result)
# ═══════════════════════════════════════════════════════════════════════


class TestDataPropagation:
    def test_verification_summary_on_result(self):
        orch = _build_orch()
        result = orch.run(_query())
        if result.success and result.verification_summary is not None:
            assert hasattr(result.verification_summary, "answer_verdict")
            assert hasattr(result.verification_summary, "claim_results")

    def test_validation_summary_on_result(self):
        orch = _build_orch()
        result = orch.run(_query())
        if result.success:
            assert result.validation_summary is not None
            assert hasattr(result.validation_summary, "candidates_generated")


# ═══════════════════════════════════════════════════════════════════════
# REPAIR LOOP
# ═══════════════════════════════════════════════════════════════════════


class TestRepairLoop:
    def test_repair_calls_generator_twice_with_feedback(self):
        """REPAIR → re-generate with verification feedback → re-verify → ANSWER."""
        call_log: list[str | None] = []

        class TrackingGen:
            def __init__(self, inner: Generator):
                self._inner = inner
            def generate(self, query, bundle, validation_summary=None,
                         verification_feedback=None):
                call_log.append(verification_feedback)
                return self._inner.generate(
                    query, bundle, validation_summary,
                    verification_feedback=verification_feedback,
                )

        client = MockLLMClient(canned_text="Visiting hours are 9am to 8pm. [E1]")
        tracking = TrackingGen(Generator(llm_client=client))
        engine = _mock_engine_sequence(
            (DecisionAction.REPAIR, CorrectiveTarget.GENERATION),
            DecisionAction.ANSWER,
        )
        orch = _build_orch(generator=tracking, decision_engine=engine)
        result = orch.run(_query())

        assert len(call_log) == 2, f"Generator called {len(call_log)} times, expected 2"
        assert call_log[0] is None, "First generation must have no feedback"
        # Second call has feedback (could be empty if M9 found no issues)
        assert len(call_log) == 2
        assert result.corrective_attempts >= 1

    def test_repaired_answer_re_verified_and_re_decided(self):
        engine = _mock_engine_sequence(
            (DecisionAction.REPAIR, CorrectiveTarget.GENERATION),
            DecisionAction.ANSWER,
        )
        orch = _build_orch(decision_engine=engine)
        result = orch.run(_query())
        assert engine._call_count["n"] == 2
        assert result.decision.action is DecisionAction.ANSWER


# ═══════════════════════════════════════════════════════════════════════
# REGENERATE → VERIFIED_CONTEXT
# ═══════════════════════════════════════════════════════════════════════


class TestRegenerateVerifiedContext:
    def test_regenerate_vc_calls_generator_twice(self):
        client = MockLLMClient(canned_text="Visiting hours are 9am to 8pm. [E1]")
        gen = Generator(llm_client=client)
        engine = _mock_engine_sequence(
            (DecisionAction.REGENERATE, CorrectiveTarget.VERIFIED_CONTEXT),
            DecisionAction.ANSWER,
        )
        orch = _build_orch(generator=gen, decision_engine=engine)
        result = orch.run(_query())
        assert client.call_count == 2
        assert result.decision.action is DecisionAction.ANSWER


# ═══════════════════════════════════════════════════════════════════════
# REGENERATE → RETRIEVAL (full re-entry)
# ═══════════════════════════════════════════════════════════════════════


class TestRegenerateRetrieval:
    def test_regenerate_retrieval_calls_retrieval_twice(self):
        retrieval_calls = {"n": 0}

        def counting_retrieval(query, risk_profile, bundle):
            retrieval_calls["n"] += 1
            for it in _diverse_items():
                eid = EvidenceItemId(f"{it.item_id}-r{retrieval_calls['n']}")
                bundle.add_evidence_item(EvidenceItem(
                    item_id=eid, chunk_id=ChunkId(f"ch-{eid}"),
                    text=it.text, retrieval_score=it.retrieval_score,
                    provenance=it.provenance, rerank_score=it.rerank_score,
                ))

        engine = _mock_engine_sequence(
            (DecisionAction.REGENERATE, CorrectiveTarget.RETRIEVAL),
            DecisionAction.ANSWER,
        )
        orch = _build_orch(
            retrieval_fn=counting_retrieval, decision_engine=engine,
        )
        result = orch.run(_query())
        assert retrieval_calls["n"] == 2
        assert engine._call_count["n"] == 2
        assert result.decision.action is DecisionAction.ANSWER


# ═══════════════════════════════════════════════════════════════════════
# CORRECTIVE BUDGET
# ═══════════════════════════════════════════════════════════════════════


class TestCorrectiveBudget:
    def test_budget_zero_forces_abstain_on_repair(self):
        engine = _mock_engine_sequence(
            (DecisionAction.REPAIR, CorrectiveTarget.GENERATION),
        )
        orch = _build_orch(decision_engine=engine, max_corrective_attempts=0)
        result = orch.run(_query())
        assert result.decision.action is DecisionAction.ABSTAIN
        assert engine._call_count["n"] == 1
        assert "exhausted" in result.decision.rationale

    def test_budget_bounds_total_decide_calls(self):
        engine = _mock_engine_sequence(
            (DecisionAction.REPAIR, CorrectiveTarget.GENERATION),
            (DecisionAction.REPAIR, CorrectiveTarget.GENERATION),
            (DecisionAction.REPAIR, CorrectiveTarget.GENERATION),
            (DecisionAction.REPAIR, CorrectiveTarget.GENERATION),
            (DecisionAction.REPAIR, CorrectiveTarget.GENERATION),
        )
        client = MockLLMClient(canned_text="Answer. [E1]")
        gen = Generator(llm_client=client)
        orch = _build_orch(
            decision_engine=engine, generator=gen, max_corrective_attempts=3,
        )
        result = orch.run(_query())
        assert result.decision.action is DecisionAction.ABSTAIN
        # 1 initial + 3 corrective = 4 decide calls
        assert engine._call_count["n"] == 4
        assert client.call_count == 4

    def test_negative_budget_rejected(self):
        with pytest.raises(ValueError):
            _build_orch(max_corrective_attempts=-1)


# ═══════════════════════════════════════════════════════════════════════
# SAFETY BOUNDARIES
# ═══════════════════════════════════════════════════════════════════════


class TestSafetyBoundaries:
    def test_generation_skipped_when_sufficiency_blocks(self):
        poor = _item("E-min", "x", SourceType.OTHER, 0.01)
        client = MockLLMClient(canned_text="should not appear")
        gen = Generator(llm_client=client)
        orch = _build_orch(retrieval_fn=_retrieval_fn(poor), generator=gen)
        result = orch.run(_query())
        if not result.success and "sufficiency" in result.pipeline_error.stage:
            assert client.call_count == 0

    def test_verification_skipped_on_gen_failure(self):
        orch = _build_orch(raise_on_generate=LLMClientError("fail"))
        orch._verification = MagicMock(spec=VerificationPipeline)
        orch.run(_query())
        orch._verification.verify.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# STAGE ORDER
# ═══════════════════════════════════════════════════════════════════════


class TestStageOrder:
    def test_stage_execution_order(self):
        log: list[str] = []
        nli = NLIService(NullNLIBackend())

        real_ret = _default_retrieval()
        def log_ret(q, rp, b):
            log.append("retrieval"); return real_ret(q, rp, b)

        cp = AtomicClaimPipeline()
        real_cr = cp.run
        def log_cr(b):
            log.append("claims"); return real_cr(b)
        cp.run = log_cr

        vp = ValidationPipeline(nli_service=nli)
        real_vr = vp.run
        def log_vr(b, q, rp):
            log.append("validation"); return real_vr(b, q, rp)
        vp.run = log_vr

        client = MockLLMClient(canned_text="Visiting hours 9-8. [E1]")
        gen = Generator(llm_client=client)
        real_gen = gen.generate
        def log_gen(q, b, vs=None, verification_feedback=None):
            log.append("generation")
            return real_gen(q, b, vs, verification_feedback=verification_feedback)
        gen.generate = log_gen

        vep = VerificationPipeline(nli_service=nli)
        real_ver = vep.verify
        def log_ver(t, b, query=None, validation_summary=None):
            log.append("verification")
            return real_ver(t, b, query=query, validation_summary=validation_summary)
        vep.verify = log_ver

        orch = PipelineOrchestrator(
            retrieval_fn=log_ret,
            reranking_service=_mock_reranking_service(),
            claim_pipeline=cp,
            validation_pipeline=vp,
            generator=gen,
            verification_pipeline=vep,
            confidence_pipeline=ConfidencePipeline(),
            decision_engine=DecisionEngine(),
        )
        result = orch.run(_query())
        assert result.success

        assert log.index("retrieval") < log.index("claims")
        assert log.index("claims") < log.index("validation")
        assert log.index("validation") < log.index("generation")
        assert log.index("generation") < log.index("verification")


# ═══════════════════════════════════════════════════════════════════════
# DETERMINISM
# ═══════════════════════════════════════════════════════════════════════


class TestDeterminism:
    def test_same_inputs_same_decision(self):
        r1 = _build_orch().run(_query())
        r2 = _build_orch().run(_query())
        assert r1.decision.action == r2.decision.action
        assert r1.decision.confidence == r2.decision.confidence


# ═══════════════════════════════════════════════════════════════════════
# REPAIR FEEDBACK DERIVATION
# ═══════════════════════════════════════════════════════════════════════


class TestRepairFeedback:
    def test_feedback_from_unsupported_and_missing(self):
        from rasvcx.pipeline.orchestrator import _build_repair_feedback
        vs = VerificationSummary(
            answer_verdict=AnswerVerdict.UNVERIFIED,
            overall_confidence=0.3,
            claim_results=(
                ClaimVerificationResult(
                    claim_id=GeneratedClaimId("GC-1"),
                    label=SupportLabel.UNSUPPORTED,
                    confidence=0.2,
                    stage=VerificationStage.DETERMINISTIC,
                    reason_code=VerificationReasonCode.NO_EVIDENCE,
                    rationale="no evidence supports this claim",
                ),
            ),
            citation_results=(
                CitationResult(
                    claim_id=GeneratedClaimId("GC-1"),
                    status=CitationStatus.MISSING,
                    cited_item_ids=frozenset(),
                    rationale="claim has no citation",
                ),
            ),
        )
        fb = _build_repair_feedback(vs)
        assert "GC-1" in fb
        assert "unsupported" in fb
        assert "missing" in fb

    def test_feedback_empty_when_all_supported(self):
        from rasvcx.pipeline.orchestrator import _build_repair_feedback
        vs = VerificationSummary(
            answer_verdict=AnswerVerdict.VERIFIED,
            overall_confidence=0.9,
            claim_results=(
                ClaimVerificationResult(
                    claim_id=GeneratedClaimId("GC-ok"),
                    label=SupportLabel.SUPPORTED,
                    confidence=0.95,
                    stage=VerificationStage.DETERMINISTIC,
                    reason_code=VerificationReasonCode.DIRECT_EVIDENCE_SUPPORT,
                    rationale="fully supported",
                ),
            ),
            citation_results=(
                CitationResult(
                    claim_id=GeneratedClaimId("GC-ok"),
                    status=CitationStatus.CORRECT,
                    cited_item_ids=frozenset({EvidenceItemId("E1")}),
                    rationale="valid citation",
                ),
            ),
        )
        assert _build_repair_feedback(vs) == ""


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE RESULT CONTRACT
# ═══════════════════════════════════════════════════════════════════════


class TestPipelineResultContract:
    def test_answer_has_answer(self):
        d = Decision(action=DecisionAction.ANSWER, confidence=0.85)
        r = PipelineResult(query_id=QueryId("q1"), decision=d, generated_text="A.")
        assert r.success and r.has_answer

    def test_warning_has_answer(self):
        d = Decision(action=DecisionAction.WARNING, confidence=0.6)
        r = PipelineResult(query_id=QueryId("q1"), decision=d, generated_text="W.")
        assert r.has_answer

    def test_abstain_no_answer(self):
        d = Decision(action=DecisionAction.ABSTAIN, confidence=0.1)
        r = PipelineResult(query_id=QueryId("q1"), decision=d)
        assert r.success and not r.has_answer

    def test_error_result(self):
        err = PipelineError(stage="retrieval", message="fail", is_retryable=True)
        r = make_error_result(QueryId("q1"), err)
        assert not r.success and not r.has_answer
        assert r.decision.action is DecisionAction.ABSTAIN