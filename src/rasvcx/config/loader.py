"""Configuration loader for RASVC-X (Module 13).

M17 patch: added 'ingestion' to _TOP_LEVEL_KEYS and _build_ingestion().
All existing behaviour is preserved exactly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from rasvcx.config.settings import (
    APISettings,
    ChunkingSettings,
    ConfidenceSettings,
    ConfigurationError,
    CorpusSettings,
    DecisionSettings,
    ExecutionMode,
    IngestionSettings,
    LLMSettings,
    NLISettings,
    PipelineSettings,
    RerankerSettings,
    RetrievalSettings,
    Settings,
    SufficiencySettings,
    ValidationSettings,
    VerificationSettings,
)

_CONFIG_SCHEMA_VERSION = 1

# The profile a normally-started application uses.  offline_test is never
# selected implicitly: it must be requested with RASVCX_EXECUTION_MODE or
# an explicit config file.
DEFAULT_CONFIG_PATH = "config/research_hybrid.yaml"

_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "execution_mode",
    "corpus", "chunking", "retrieval", "reranker",
    "sufficiency", "nli", "validation", "verification",
    "confidence", "decision", "llm", "pipeline", "api",
    "ingestion",                                          # M17 addition
})


def _parse_mode(raw: str) -> ExecutionMode:
    try:
        return ExecutionMode(raw)
    except ValueError:
        valid = [m.value for m in ExecutionMode]
        raise ConfigurationError(
            f"execution_mode must be one of {valid}, got {raw!r}"
        )


def _check_unknown_keys(data: dict, allowed: frozenset, context: str) -> None:
    unknown = set(data.keys()) - allowed
    if unknown:
        raise ConfigurationError(
            f"Unknown configuration key(s) under '{context}': {sorted(unknown)}."
            "  Check for typos."
        )


def _section(data: dict, key: str) -> dict:
    val = data.get(key, {})
    if not isinstance(val, dict):
        raise ConfigurationError(
            f"Configuration section '{key}' must be a mapping, "
            f"got {type(val).__name__!r}"
        )
    return val


def _build_corpus(r: dict) -> CorpusSettings:
    a = frozenset({"store_path", "bm25_path", "manifest_path", "smoke_corpus_path"})
    _check_unknown_keys(r, a, "corpus")
    return CorpusSettings(**{k: v for k, v in r.items() if k in a})


def _build_chunking(r: dict) -> ChunkingSettings:
    a = frozenset({"strategy", "max_tokens", "overlap"})
    _check_unknown_keys(r, a, "chunking")
    return ChunkingSettings(**{k: v for k, v in r.items() if k in a})


def _build_retrieval(r: dict) -> RetrievalSettings:
    a = frozenset({
        "mode", "bm25_top_k", "dense_top_k", "rrf_k", "rrf_top_k",
        "dense_model_name", "qdrant_host", "qdrant_port", "qdrant_collection",
        "targeted_bm25_top_k", "targeted_dense_top_k", "targeted_rrf_k",
        "qdrant_mode", "qdrant_path",
    })
    _check_unknown_keys(r, a, "retrieval")
    return RetrievalSettings(**{k: v for k, v in r.items() if k in a})


def _build_reranker(r: dict) -> RerankerSettings:
    a = frozenset({
        "enabled", "model_name", "max_candidates", "batch_size", "device",
        "min_relevance_score",
    })
    _check_unknown_keys(r, a, "reranker")
    return RerankerSettings(**{k: v for k, v in r.items() if k in a})


def _build_sufficiency(r: dict) -> SufficiencySettings:
    a = frozenset({
        "min_evidence_items", "min_scored_items", "min_top_rerank_score",
        "min_rerank_score_margin", "max_unknown_provenance_ratio_high_risk",
        "min_source_diversity_high_risk", "high_risk_score_threshold",
        "margin_check_below_top_score",
        "diversity_waiver_source_types", "diversity_waiver_min_top_score",
    })
    _check_unknown_keys(r, a, "sufficiency")
    kw: dict[str, Any] = {k: v for k, v in r.items() if k in a}
    if "diversity_waiver_source_types" in kw:
        raw_types = kw["diversity_waiver_source_types"]
        if not isinstance(raw_types, (list, tuple)):
            raise ConfigurationError(
                "sufficiency.diversity_waiver_source_types must be a list of strings"
            )
        kw["diversity_waiver_source_types"] = tuple(str(t) for t in raw_types)
    return SufficiencySettings(**kw)


def _build_nli(r: dict) -> NLISettings:
    a = frozenset({"enabled", "model_name", "device"})
    _check_unknown_keys(r, a, "nli")
    return NLISettings(**{k: v for k, v in r.items() if k in a})


def _build_validation(r: dict) -> ValidationSettings:
    a = frozenset({
        "numeric_relative_tolerance", "min_shared_tokens_for_comparison",
        "max_claim_pairs_per_candidate", "max_candidates",
        "inconclusive_confidence_threshold",
    })
    _check_unknown_keys(r, a, "validation")
    return ValidationSettings(**{k: v for k, v in r.items() if k in a})


def _build_verification(r: dict) -> VerificationSettings:
    a = frozenset({
        "max_claims", "max_nli_calls_per_answer",
        "max_fallback_evidence_candidates",
        "unsupported_fraction_for_partial",
        "unsupported_fraction_for_unverified",
    })
    _check_unknown_keys(r, a, "verification")
    return VerificationSettings(**{k: v for k, v in r.items() if k in a})


def _build_confidence(r: dict) -> ConfidenceSettings:
    a = frozenset({
        "evidence_agreement", "claim_verification_support",
        "contradiction_penalty", "provenance_quality", "source_diversity",
        "retrieval_quality", "rerank_quality", "resolution_uncertainty_penalty",
        "calibration_artifact_path",
    })
    _check_unknown_keys(r, a, "confidence")
    return ConfidenceSettings(**{k: v for k, v in r.items() if k in a})


def _build_decision(r: dict) -> DecisionSettings:
    a = frozenset({
        "answer_min", "warning_min", "regenerate_min",
        "high_risk_answer_min", "high_risk_warning_min",
        "high_risk_regenerate_min", "high_risk_threshold",
    })
    _check_unknown_keys(r, a, "decision")
    return DecisionSettings(**{k: v for k, v in r.items() if k in a})


def _build_llm(r: dict, api_key: str) -> LLMSettings:
    if "api_key" in r:
        raise ConfigurationError(
            "llm.api_key must not be placed in the config file. "
            "Set it via the RASVCX_LLM_API_KEY environment variable."
        )
    a = frozenset({
        "provider", "model_name", "temperature", "max_tokens",
        "timeout_seconds", "stub_response", "max_retries",
    })
    _check_unknown_keys(r, a, "llm")
    kw = {k: v for k, v in r.items() if k in a}
    kw["_api_key"] = api_key
    return LLMSettings(**kw)


def _build_pipeline(r: dict) -> PipelineSettings:
    a = frozenset({"max_corrective_attempts", "max_context_chars"})
    _check_unknown_keys(r, a, "pipeline")
    return PipelineSettings(**{k: v for k, v in r.items() if k in a})


def _build_api(r: dict, auth_token: str) -> APISettings:
    if "auth_token" in r or "admin_token" in r:
        raise ConfigurationError(
            "api.auth_token / api.admin_token must not be placed in the config file. "
            "Set them via RASVCX_AUTH_TOKEN / RASVCX_ADMIN_TOKEN."
        )
    a = frozenset({
        "host", "port", "max_workers", "max_queue_depth",
        "request_timeout_seconds", "max_request_bytes",
        "max_query_chars", "require_auth", "cache_size", "cache_ttl_seconds",
        "rate_limit_per_minute",
    })
    _check_unknown_keys(r, a, "api")
    kw = {k: v for k, v in r.items() if k in a}
    kw["_auth_token"] = auth_token
    kw["_admin_token"] = os.environ.get("RASVCX_ADMIN_TOKEN", "")
    return APISettings(**kw)


def _build_ingestion(r: dict) -> IngestionSettings:
    """Parse the ingestion: YAML section into IngestionSettings.

    allowed_url_schemes arrives from YAML as a list — converted to tuple.
    db_path and temp_dir can be overridden by env vars after YAML is
    parsed; that override happens in load_settings(), not here.
    """
    a = frozenset({
        "max_upload_bytes", "max_upload_bytes_docx", "max_upload_bytes_pptx",
        "max_pdf_pages", "max_csv_rows", "max_xlsx_rows",
        "parser_timeout_seconds", "db_path", "max_concurrent_jobs", "temp_dir",
        "corpus_versions_dir", "active_version_path", "gc_delay_seconds",
        "max_source_response_bytes", "source_connect_timeout_seconds",
        "source_read_timeout_seconds", "allowed_url_schemes",
        "scheduled_sync_interval_seconds", "max_redirects", "keep_versions",
        "require_published_kb",
    })
    _check_unknown_keys(r, a, "ingestion")
    kw: dict[str, Any] = {k: v for k, v in r.items() if k in a}
    if "allowed_url_schemes" in kw:
        raw_schemes = kw["allowed_url_schemes"]
        if not isinstance(raw_schemes, (list, tuple)):
            raise ConfigurationError(
                "ingestion.allowed_url_schemes must be a list of strings"
            )
        kw["allowed_url_schemes"] = tuple(str(s) for s in raw_schemes)
    return IngestionSettings(**kw)


def resolve_config_path(config_path: str | None = None) -> str | None:
    """Decide which config file a process should load.

    Precedence:
      1. an explicit `config_path` argument
      2. RASVCX_CONFIG
      3. RASVCX_EXECUTION_MODE=offline_test  -> None (built-in offline defaults)
      4. DEFAULT_CONFIG_PATH (research_hybrid)

    Offline mode is therefore always an explicit choice; an application
    started with no configuration runs research_hybrid or fails loudly.
    """
    if config_path:
        return config_path
    env_path = os.environ.get("RASVCX_CONFIG", "").strip()
    if env_path:
        return env_path
    if os.environ.get("RASVCX_EXECUTION_MODE", "").strip() == ExecutionMode.OFFLINE_TEST.value:
        return None
    return DEFAULT_CONFIG_PATH


def _env_overrides() -> dict[str, dict[str, Any]]:
    """Environment overrides applied on top of the YAML sections.

    Only deployment-specific values (where services live, which model id)
    are overridable; safety thresholds are config-file only.
    """
    def _get(name: str) -> str:
        return os.environ.get(name, "").strip()

    retrieval: dict[str, Any] = {}
    if _get("RASVCX_QDRANT_MODE"):
        retrieval["qdrant_mode"] = _get("RASVCX_QDRANT_MODE")
    if _get("RASVCX_QDRANT_PATH"):
        retrieval["qdrant_path"] = _get("RASVCX_QDRANT_PATH")
    if _get("RASVCX_QDRANT_HOST"):
        retrieval["qdrant_host"] = _get("RASVCX_QDRANT_HOST")
    if _get("RASVCX_QDRANT_PORT"):
        try:
            retrieval["qdrant_port"] = int(_get("RASVCX_QDRANT_PORT"))
        except ValueError as exc:
            raise ConfigurationError("RASVCX_QDRANT_PORT must be an integer") from exc

    llm: dict[str, Any] = {}
    if _get("RASVCX_LLM_MODEL"):
        llm["model_name"] = _get("RASVCX_LLM_MODEL")

    ingestion: dict[str, Any] = {}
    if _get("RASVCX_INGEST_DB_PATH"):
        ingestion["db_path"] = _get("RASVCX_INGEST_DB_PATH")
    if _get("RASVCX_INGEST_TEMP_DIR"):
        ingestion["temp_dir"] = _get("RASVCX_INGEST_TEMP_DIR")
    if _get("RASVCX_CORPUS_DIR"):
        ingestion["corpus_versions_dir"] = _get("RASVCX_CORPUS_DIR")
    if _get("RASVCX_ACTIVE_VERSION_PATH"):
        ingestion["active_version_path"] = _get("RASVCX_ACTIVE_VERSION_PATH")
    elif _get("RASVCX_CORPUS_DIR"):
        ingestion["active_version_path"] = str(
            Path(_get("RASVCX_CORPUS_DIR")) / "active_version.json"
        )
    if _get("RASVCX_REQUIRE_PUBLISHED_KB"):
        ingestion["require_published_kb"] = _get("RASVCX_REQUIRE_PUBLISHED_KB").lower() in ("1", "true", "yes")
    corpus: dict[str, Any] = {}
    if _get("RASVCX_SEED_DIR"):
        seed = Path(_get("RASVCX_SEED_DIR"))
        corpus = {
            "store_path": str(seed / "corpus_store.json"),
            "bm25_path": str(seed / "bm25_index.pkl"),
            "manifest_path": str(seed / "manifest.json"),
        }
    confidence: dict[str, Any] = {}
    if _get("RASVCX_CALIBRATION_ARTIFACT"):
        confidence["calibration_artifact_path"] = _get("RASVCX_CALIBRATION_ARTIFACT")
    return {"retrieval": retrieval, "llm": llm, "ingestion": ingestion, "corpus": corpus,
            "confidence": confidence}


def load_settings(config_path: str | None = None) -> Settings:
    """Load and validate application settings.

    Args:
        config_path: Path to a YAML config file.  None resolves through
            resolve_config_path(): RASVCX_CONFIG, else the built-in
            offline defaults when RASVCX_EXECUTION_MODE=offline_test,
            else the research_hybrid profile.

    Environment variables applied AFTER YAML:
        RASVCX_EXECUTION_MODE       overrides execution_mode
        RASVCX_LLM_API_KEY          injected as llm._api_key (never from YAML)
        RASVCX_AUTH_TOKEN           injected as api._auth_token (never from YAML)
        RASVCX_LLM_MODEL            overrides llm.model_name
        RASVCX_QDRANT_MODE/_PATH/_HOST/_PORT   override retrieval.qdrant_*
        RASVCX_SEED_DIR             seed index directory (corpus_store.json,
                                    bm25_index.pkl, manifest.json)
        RASVCX_CORPUS_DIR           overrides ingestion.corpus_versions_dir
        RASVCX_ACTIVE_VERSION_PATH  overrides ingestion.active_version_path
        RASVCX_INGEST_DB_PATH       overrides ingestion.db_path
        RASVCX_INGEST_TEMP_DIR      overrides ingestion.temp_dir

    Raises:
        ConfigurationError, FileNotFoundError
    """
    api_key = os.environ.get("RASVCX_LLM_API_KEY", "")
    auth_token = os.environ.get("RASVCX_AUTH_TOKEN", "")
    mode_override = os.environ.get("RASVCX_EXECUTION_MODE", "").strip()
    overrides = _env_overrides()

    config_path = resolve_config_path(config_path)

    if config_path is None:
        # Explicit offline_test with no config file: built-in defaults.
        # Model/Qdrant overrides are meaningless for the offline doubles.
        return Settings(
            execution_mode=_parse_mode(mode_override),
            corpus=CorpusSettings(**overrides["corpus"]),
            llm=LLMSettings(_api_key=api_key),
            api=APISettings(_auth_token=auth_token,
                            _admin_token=os.environ.get("RASVCX_ADMIN_TOKEN", "")),
            ingestion=IngestionSettings(**overrides["ingestion"]),
        )

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"Failed to parse YAML config {path}: {exc}"
        ) from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Configuration file {path} must be a YAML mapping at the top level"
        )

    _check_unknown_keys(raw, _TOP_LEVEL_KEYS, "<root>")

    schema_ver = raw.get("schema_version")
    if schema_ver is not None and int(schema_ver) != _CONFIG_SCHEMA_VERSION:
        raise ConfigurationError(
            f"Config schema_version {schema_ver} is not supported "
            f"(expected {_CONFIG_SCHEMA_VERSION})"
        )

    raw_mode = mode_override or raw.get("execution_mode")
    if not raw_mode:
        # A config file that names no mode must not quietly become
        # offline_test (test doubles) -- the mode is always stated.
        raise ConfigurationError(
            f"{path} does not set execution_mode. Set it to one of "
            f"{[m.value for m in ExecutionMode]} (or RASVCX_EXECUTION_MODE)."
        )
    mode = _parse_mode(raw_mode)

    def _merged(key: str) -> dict:
        section = dict(_section(raw, key))
        section.update(overrides.get(key, {}))
        return section

    try:
        return Settings(
            execution_mode=mode,
            corpus=_build_corpus(_merged("corpus")),
            chunking=_build_chunking(_section(raw, "chunking")),
            retrieval=_build_retrieval(_merged("retrieval")),
            reranker=_build_reranker(_section(raw, "reranker")),
            sufficiency=_build_sufficiency(_section(raw, "sufficiency")),
            nli=_build_nli(_section(raw, "nli")),
            validation=_build_validation(_section(raw, "validation")),
            verification=_build_verification(_section(raw, "verification")),
            confidence=_build_confidence(_merged("confidence")),
            decision=_build_decision(_section(raw, "decision")),
            llm=_build_llm(_merged("llm"), api_key),
            pipeline=_build_pipeline(_section(raw, "pipeline")),
            api=_build_api(_section(raw, "api"), auth_token),
            ingestion=_build_ingestion(_merged("ingestion")),
        )
    except TypeError as exc:
        raise ConfigurationError(
            f"Invalid configuration value in {path}: {exc}"
        ) from exc


__all__ = ["DEFAULT_CONFIG_PATH", "load_settings", "resolve_config_path"]
