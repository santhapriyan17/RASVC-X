"""Tests for Module 13 configuration (settings + loader)."""
from __future__ import annotations
import os
import tempfile
import pytest
from rasvcx.config.settings import (
    Settings, ExecutionMode, ConfigurationError, offline_test_settings,
    RerankerSettings, NLISettings, LLMSettings, SufficiencySettings,
    RetrievalSettings, DecisionSettings, ConfidenceSettings,
    ChunkingSettings, APISettings,
)
from rasvcx.config.loader import load_settings


class TestDefaultSettings:
    def test_default_is_offline_test(self):
        s = Settings()
        assert s.execution_mode is ExecutionMode.OFFLINE_TEST

    def test_default_does_not_raise(self):
        Settings()

    def test_offline_test_settings_alias(self):
        assert offline_test_settings() == Settings()

    def test_default_reranker_disabled(self):
        assert not Settings().reranker.enabled

    def test_default_nli_disabled(self):
        assert not Settings().nli.enabled

    def test_default_llm_stub(self):
        assert Settings().llm.provider == "stub"

    def test_default_min_scored_items_zero(self):
        assert Settings().sufficiency.min_scored_items == 0

    def test_default_retrieval_bm25_only(self):
        assert Settings().retrieval.mode == "bm25_only"


class TestModeCoercion:
    def test_string_coercion(self):
        s = Settings(execution_mode="offline_test")
        assert s.execution_mode is ExecutionMode.OFFLINE_TEST

    def test_invalid_mode_raises(self):
        with pytest.raises(ConfigurationError, match="execution_mode"):
            Settings(execution_mode="invalid_mode")


class TestModeConsistency:
    def test_offline_requires_bm25_only(self):
        with pytest.raises(ConfigurationError, match="bm25_only"):
            Settings(
                execution_mode=ExecutionMode.OFFLINE_TEST,
                retrieval=RetrievalSettings(mode="hybrid"),
            )

    def test_offline_rejects_enabled_reranker(self):
        with pytest.raises(ConfigurationError, match="reranker"):
            Settings(
                execution_mode=ExecutionMode.OFFLINE_TEST,
                reranker=RerankerSettings(enabled=True),
            )

    def test_offline_rejects_enabled_nli(self):
        with pytest.raises(ConfigurationError, match="nli"):
            Settings(
                execution_mode=ExecutionMode.OFFLINE_TEST,
                nli=NLISettings(enabled=True),
            )

    def test_offline_rejects_real_llm(self):
        with pytest.raises(ConfigurationError, match="llm"):
            Settings(
                execution_mode=ExecutionMode.OFFLINE_TEST,
                llm=LLMSettings(provider="gemini", _api_key="key"),
            )

    def test_research_bm25_requires_reranker(self):
        with pytest.raises(ConfigurationError, match="reranker"):
            Settings(
                execution_mode=ExecutionMode.RESEARCH_BM25,
                reranker=RerankerSettings(enabled=False),
                nli=NLISettings(enabled=True),
                llm=LLMSettings(provider="gemini", _api_key="key"),
            )

    def test_research_bm25_requires_nli(self):
        with pytest.raises(ConfigurationError, match="nli"):
            Settings(
                execution_mode=ExecutionMode.RESEARCH_BM25,
                reranker=RerankerSettings(enabled=True),
                nli=NLISettings(enabled=False),
                llm=LLMSettings(provider="gemini", _api_key="key"),
            )

    def test_research_bm25_rejects_stub_llm(self):
        with pytest.raises(ConfigurationError, match="LLM provider"):
            Settings(
                execution_mode=ExecutionMode.RESEARCH_BM25,
                reranker=RerankerSettings(enabled=True),
                nli=NLISettings(enabled=True),
                llm=LLMSettings(provider="stub"),
            )

    def test_research_hybrid_requires_hybrid_retrieval(self):
        with pytest.raises(ConfigurationError, match="hybrid"):
            Settings(
                execution_mode=ExecutionMode.RESEARCH_HYBRID,
                retrieval=RetrievalSettings(mode="bm25_only"),
                reranker=RerankerSettings(enabled=True),
                nli=NLISettings(enabled=True),
                llm=LLMSettings(provider="gemini", _api_key="key"),
            )


class TestFieldValidation:
    def test_retrieval_bad_mode(self):
        with pytest.raises(ConfigurationError):
            RetrievalSettings(mode="bad")

    def test_retrieval_bad_top_k(self):
        with pytest.raises(ConfigurationError):
            RetrievalSettings(bm25_top_k=0)

    def test_retrieval_bad_port(self):
        with pytest.raises(ConfigurationError):
            RetrievalSettings(qdrant_port=99999)

    def test_decision_ordering(self):
        with pytest.raises(ConfigurationError):
            DecisionSettings(answer_min=0.3, warning_min=0.6)

    def test_confidence_weight_out_of_range(self):
        with pytest.raises(ConfigurationError):
            ConfidenceSettings(evidence_agreement=1.5)

    def test_chunking_overlap_ge_max_tokens(self):
        with pytest.raises(ConfigurationError):
            ChunkingSettings(max_tokens=100, overlap=100)

    def test_chunking_bad_strategy(self):
        with pytest.raises(ConfigurationError):
            ChunkingSettings(strategy="bad")

    def test_api_bad_port(self):
        with pytest.raises(ConfigurationError):
            APISettings(port=0)

    def test_api_require_auth_without_token(self):
        with pytest.raises(ConfigurationError):
            APISettings(require_auth=True)

    def test_llm_gemini_without_key(self):
        with pytest.raises(ConfigurationError):
            LLMSettings(provider="gemini")

    def test_llm_bad_temperature(self):
        with pytest.raises(ConfigurationError):
            LLMSettings(temperature=3.0)

    def test_sufficiency_bad_ratio(self):
        with pytest.raises(ConfigurationError):
            SufficiencySettings(max_unknown_provenance_ratio_high_risk=1.5)


class TestSecretHandling:
    def test_api_key_not_in_repr(self):
        s = LLMSettings(provider="gemini", _api_key="supersecret")
        assert "supersecret" not in repr(s)

    def test_auth_token_not_in_repr(self):
        s = APISettings(_auth_token="mytoken")
        assert "mytoken" not in repr(s)

    def test_api_key_accessible_via_property(self):
        s = LLMSettings(provider="gemini", _api_key="mykey")
        assert s.api_key == "mykey"


class TestLoader:
    def test_default_is_research_hybrid_not_offline(self, monkeypatch):
        # No config, no mode: the application default is research_hybrid.
        from rasvcx.config.loader import DEFAULT_CONFIG_PATH, resolve_config_path
        assert resolve_config_path() == DEFAULT_CONFIG_PATH
        monkeypatch.setenv("RASVCX_LLM_API_KEY", "test-key-not-real")
        s = load_settings()
        assert s.execution_mode is ExecutionMode.RESEARCH_HYBRID
        assert s.llm.provider == "gemini"
        assert s.retrieval.mode == "hybrid"
        assert s.reranker.enabled and s.nli.enabled

    def test_default_without_api_key_fails_loudly(self):
        # research_hybrid needs a real LLM: there is no fallback to the stub.
        with pytest.raises(ConfigurationError, match="api_key"):
            load_settings()

    def test_offline_requires_explicit_request(self, monkeypatch):
        monkeypatch.setenv("RASVCX_EXECUTION_MODE", "offline_test")
        s = load_settings()
        assert s.execution_mode is ExecutionMode.OFFLINE_TEST
        assert s.llm.provider == "stub"
        assert s.is_offline is True

    def test_yaml_without_mode_is_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("")
            p = f.name
        try:
            with pytest.raises(ConfigurationError, match="execution_mode"):
                load_settings(p)
        finally:
            os.unlink(p)

    def test_env_overrides_deployment_values(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RASVCX_LLM_API_KEY", "test-key-not-real")
        monkeypatch.setenv("RASVCX_QDRANT_MODE", "memory")
        monkeypatch.setenv("RASVCX_LLM_MODEL", "some-model")
        monkeypatch.setenv("RASVCX_CORPUS_DIR", str(tmp_path / "kb"))
        s = load_settings()
        assert s.retrieval.qdrant_mode == "memory"
        assert s.llm.model_name == "some-model"
        assert s.ingestion.corpus_versions_dir == str(tmp_path / "kb")
        assert s.ingestion.active_version_path == str(tmp_path / "kb" / "active_version.json")

    def test_bad_qdrant_mode_rejected(self, monkeypatch):
        monkeypatch.setenv("RASVCX_LLM_API_KEY", "test-key-not-real")
        monkeypatch.setenv("RASVCX_QDRANT_MODE", "cloud")
        with pytest.raises(ConfigurationError, match="qdrant_mode"):
            load_settings()

    def test_base_yaml_loads(self):
        from pathlib import Path
        base = Path("config/base.yaml")
        if base.exists():
            s = load_settings(base)
            assert s.execution_mode is ExecutionMode.OFFLINE_TEST

    def test_unknown_top_level_key_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("bad_key: 123\n")
            p = f.name
        try:
            with pytest.raises(ConfigurationError, match="bad_key"):
                load_settings(p)
        finally:
            os.unlink(p)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_settings("/nonexistent/path.yaml")

    def test_env_mode_override(self, monkeypatch):
        monkeypatch.setenv("RASVCX_EXECUTION_MODE", "offline_test")
        s = load_settings()
        assert s.execution_mode is ExecutionMode.OFFLINE_TEST

    def test_env_bad_mode_raises(self, monkeypatch):
        monkeypatch.setenv("RASVCX_EXECUTION_MODE", "bad_mode")
        with pytest.raises(ConfigurationError):
            load_settings()

    def test_api_key_not_from_yaml(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("execution_mode: offline_test\nllm:\n  api_key: secret\n")
            p = f.name
        try:
            with pytest.raises(ConfigurationError, match="api_key"):
                load_settings(p)
        finally:
            os.unlink(p)