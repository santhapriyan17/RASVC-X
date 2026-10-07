"""Tests for pipeline factory and M12 sufficiency_config injection (Module 13).

Covers:
  - build_pipeline() constructs a working PipelineOrchestrator in offline_test mode.
  - DisabledRerankingService is used when reranker.enabled=False.
  - SufficiencyGateConfig is forwarded from Settings to the orchestrator.
  - M12 backward-compatibility: default sufficiency_config=None preserves
    existing behavior (scored_item_count=0 -> INSUFFICIENT with default config).
  - Injected SufficiencyGateConfig(min_scored_items=0) lets the pipeline
    pass sufficiency with all rerank_score=None.
  - get_effective_routing_info() returns accurate M2 defaults.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rasvcx.claims import AtomicClaimPipeline
from rasvcx.confidence import ConfidencePipeline
from rasvcx.config.settings import (
    ConfigurationError,
    ExecutionMode,
    LLMSettings,
    NLISettings,
    RerankerSettings,
    RetrievalSettings,
    Settings,
)
from rasvcx.decision import DecisionEngine
from rasvcx.generation import Generator
from rasvcx.generation.llm_client import MockLLMClient
from rasvcx.pipeline.factory import build_pipeline, get_effective_routing_info
from rasvcx.pipeline.orchestrator import PipelineOrchestrator
from rasvcx.retrieval.bridge import DisabledRerankingService
from rasvcx.reranking import RerankingService
from rasvcx.reranking.cross_encoder import CrossEncoderReranker
from rasvcx.schemas.common import ChunkId, EvidenceItemId, QueryId, SourceType, UNKNOWN
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.sufficiency.gate import SufficiencyGateConfig
from rasvcx.validation import NLIService, NullNLIBackend, ValidationPipeline
from rasvcx.verification import VerificationPipeline


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CANNED = "Visiting hours are 9am to 8pm. [E1] Check in at the front desk. [E2]"


def _query(text: str = "what are the visiting hours") -> QueryRequest:
    return QueryRequest(
        query_id=QueryId("q-test"),
        raw_text=text,
        normalized_text=text,
    )


def _provenance() -> Provenance:
    return Provenance(
        source_type=SourceType.INSTITUTIONAL_POLICY,
        date=UNKNOWN,
        jurisdiction=UNKNOWN,
        population=UNKNOWN,
        dosage_context=UNKNOWN,
    )


def _item(item_id: str, text: str = "Visiting hours are 9am to 8pm.") -> EvidenceItem:
    return EvidenceItem(
        item_id=EvidenceItemId(item_id),
        chunk_id=ChunkId(item_id),
        text=text,
        retrieval_score=0.8,
        provenance=_provenance(),
        rerank_score=None,
    )


def _rp() -> RiskProfile:
    return RiskProfile(
        overall_risk_score=0.1,
        feature_scores=RiskFeatureScores(),
        validation_depth=ValidationDepth.SHALLOW,
        retrieval_retry_budget=0,
        nli_call_allowance=0,
    )


def _retrieval_fn_with_none_scores(
    query: QueryRequest,
    risk_profile: RiskProfile,
    bundle: EvidenceBundle,
) -> None:
    """Adds three items with rerank_score=None. Simulates disabled reranker."""
    bundle.add_evidence_item(_item("E1", "Visiting hours are 9am to 8pm."))
    bundle.add_evidence_item(_item("E2", "Check in at the front desk."))
    bundle.add_evidence_item(_item("E3", "Maximum two visitors per patient."))
    bundle.record_retrieval_call()


def _mock_reranking_service() -> RerankingService:
    """Real RerankingService with mocked cross-encoder.
    Items receive rerank_score=0.5 so scored_item_count > 0."""
    with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5]
        mock_load.return_value = mock_model
        svc = RerankingService()
    return svc


def _noop_reranking_service() -> DisabledRerankingService:
    """DisabledRerankingService: all rerank_score remain None."""
    return DisabledRerankingService()


def _build_orch(
    retrieval_fn=None,
    reranking_service=None,
    sufficiency_config: SufficiencyGateConfig | None = None,
    canned_text: str = _CANNED,
    max_corrective_attempts: int = 2,
) -> PipelineOrchestrator:
    nli = NLIService(NullNLIBackend())
    client = MockLLMClient(canned_text=canned_text)
    generator = Generator(llm_client=client)
    if retrieval_fn is None:
        retrieval_fn = _retrieval_fn_with_none_scores
    if reranking_service is None:
        reranking_service = _mock_reranking_service()
    return PipelineOrchestrator(
        retrieval_fn=retrieval_fn,
        reranking_service=reranking_service,
        claim_pipeline=AtomicClaimPipeline(),
        validation_pipeline=ValidationPipeline(nli_service=nli),
        generator=generator,
        verification_pipeline=VerificationPipeline(nli_service=nli),
        confidence_pipeline=ConfidencePipeline(),
        decision_engine=DecisionEngine(),
        max_corrective_attempts=max_corrective_attempts,
        sufficiency_config=sufficiency_config,
    )


# ---------------------------------------------------------------------------
# M12 sufficiency_config injection — backward-compatibility
# ---------------------------------------------------------------------------


class TestSufficiencyConfigInjection:
    def test_default_config_none_gates_on_zero_scored_items(self):
        """Without sufficiency_config, SufficiencyGateConfig() default applies:
        min_scored_items=1. With all rerank_score=None, scored_item_count=0 < 1
        -> INSUFFICIENT. Proves backward-compatibility of the M12 change."""
        orch = _build_orch(
            reranking_service=_noop_reranking_service(),
            sufficiency_config=None,
        )
        result = orch.run(_query())
        assert result.pipeline_error is not None, (
            "Expected sufficiency failure with default config and rerank_score=None"
        )
        assert result.pipeline_error.stage == "sufficiency_gate"
        assert "scored_item_count=0" in result.pipeline_error.message

    def test_injected_zero_min_scored_items_passes_sufficiency(self):
        """With SufficiencyGateConfig(min_scored_items=0), a bundle where all
        items have rerank_score=None must pass sufficiency and reach a
        downstream stage."""
        orch = _build_orch(
            reranking_service=_noop_reranking_service(),
            sufficiency_config=SufficiencyGateConfig(min_scored_items=0),
        )
        result = orch.run(_query())
        if result.pipeline_error is not None:
            assert result.pipeline_error.stage != "sufficiency_gate", (
                f"Pipeline still failing at sufficiency: {result.pipeline_error.message}"
            )

    def test_injected_config_with_real_reranking_passes(self):
        """With a real (mocked) reranker, scored_item_count > 0. Both default
        and injected configs must pass sufficiency."""
        for cfg in [None, SufficiencyGateConfig(min_scored_items=1)]:
            orch = _build_orch(
                reranking_service=_mock_reranking_service(),
                sufficiency_config=cfg,
            )
            result = orch.run(_query())
            if result.pipeline_error:
                assert result.pipeline_error.stage != "sufficiency_gate", (
                    f"Unexpected sufficiency failure with cfg={cfg}: "
                    f"{result.pipeline_error.message}"
                )

    def test_sufficiency_config_stored_on_orchestrator(self):
        cfg = SufficiencyGateConfig(min_scored_items=0)
        orch = _build_orch(sufficiency_config=cfg)
        assert orch._sufficiency_config is cfg

    def test_no_sufficiency_config_stored_as_none(self):
        orch = _build_orch(sufficiency_config=None)
        assert orch._sufficiency_config is None


# ---------------------------------------------------------------------------
# build_pipeline — offline_test mode
# ---------------------------------------------------------------------------


class TestOfflineTestMode:
    def test_builds_without_error(self):
        pipeline = build_pipeline(Settings())
        assert isinstance(pipeline, PipelineOrchestrator)

    def test_uses_disabled_reranker(self):
        pipeline = build_pipeline(Settings())
        assert isinstance(pipeline._reranking, DisabledRerankingService)

    def test_sufficiency_config_has_zero_min_scored_items(self):
        """Factory must inject SufficiencyGateConfig(min_scored_items=0)
        in offline_test mode (Settings default, reranker disabled)."""
        pipeline = build_pipeline(Settings())
        assert pipeline._sufficiency_config is not None
        assert pipeline._sufficiency_config.min_scored_items == 0

    def test_run_does_not_fail_at_sufficiency_gate(self):
        """Full pipeline run in offline_test mode must not error at
        sufficiency_gate -- min_scored_items=0 must be honoured end-to-end."""
        pipeline = build_pipeline(Settings())
        result = pipeline.run(_query())
        if result.pipeline_error:
            assert result.pipeline_error.stage != "sufficiency_gate", (
                f"Sufficiency gate blocked offline_test run: "
                f"{result.pipeline_error.message}"
            )

    def test_pipeline_result_has_query_id(self):
        pipeline = build_pipeline(Settings())
        result = pipeline.run(_query())
        assert result.query_id == QueryId("q-test")

    def test_no_external_dns_during_build(self):
        """build_pipeline() in offline_test mode must not resolve any
        external hostname."""
        import socket
        original = socket.getaddrinfo
        external: list[str] = []

        def spy(host, *args, **kwargs):
            if host not in ("localhost", "127.0.0.1", "::1", None):
                external.append(str(host))
            return original(host, *args, **kwargs)

        with patch("socket.getaddrinfo", side_effect=spy):
            build_pipeline(Settings())

        assert not external, (
            f"build_pipeline made unexpected external DNS lookups: {external}"
        )


# ---------------------------------------------------------------------------
# build_pipeline — mode enforcement
# ---------------------------------------------------------------------------


class TestModeEnforcement:
    def test_hybrid_without_dense_deps_raises(self):
        """research_hybrid without required deps must raise — never succeed silently.
        FileNotFoundError is also acceptable when corpus artifacts are missing."""
        s = Settings(
            execution_mode=ExecutionMode.RESEARCH_HYBRID,
            retrieval=RetrievalSettings(mode="hybrid"),
            reranker=RerankerSettings(enabled=True),
            nli=NLISettings(enabled=True),
            llm=LLMSettings(provider="gemini", _api_key="key"),
        )
        with pytest.raises((ConfigurationError, ImportError, FileNotFoundError)):
            build_pipeline(s)

    def test_default_settings_are_offline_test(self):
        s = Settings()
        assert s.execution_mode is ExecutionMode.OFFLINE_TEST
        assert not s.reranker.enabled
        assert not s.nli.enabled
        assert s.sufficiency.min_scored_items == 0


# ---------------------------------------------------------------------------
# get_effective_routing_info — research transparency
# ---------------------------------------------------------------------------


class TestEffectiveRoutingInfo:
    def test_returns_required_keys(self):
        info = get_effective_routing_info()
        assert "weights" in info
        assert "thresholds" in info
        assert "note" in info

    def test_weights_are_floats(self):
        for k, v in get_effective_routing_info()["weights"].items():
            assert isinstance(v, float), f"{k} must be float, got {type(v)}"

    def test_thresholds_are_numeric(self):
        for k, v in get_effective_routing_info()["thresholds"].items():
            assert isinstance(v, (int, float)), f"{k} must be numeric"

    def test_note_documents_deferred_status(self):
        note = get_effective_routing_info()["note"].lower()
        assert "not yet configurable" in note or "deferred" in note

    def test_is_deterministic(self):
        assert get_effective_routing_info() == get_effective_routing_info()