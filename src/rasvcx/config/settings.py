"""Typed, validated application settings for RASVC-X (Module 13).

All tuneable parameters across M1-M12 are centralised here.  No module
reads environment variables or config files directly; every component
receives a typed value via build_pipeline() or the API lifecycle.

Design constraints:
- Pure dataclasses; no Pydantic dependency in this file.
- ConfigurationError is raised eagerly on invalid combinations so
  misconfiguration surfaces at startup, not at request time.
- Secrets (API keys, auth tokens) are never logged, never returned by
  API admin endpoints, and do not appear in repr().

Execution modes
---------------
offline_test       BM25-only retrieval, reranker disabled, NullNLI,
                   stub LLM.  No model downloads or external services.
                   min_scored_items is 0 (no cross-encoder scores exist).

research_bm25      BM25-only retrieval, real cross-encoder reranker,
                   real NLI, real LLM provider.

research_hybrid    BM25 + Dense/Qdrant retrieval, real cross-encoder,
                   real NLI, real LLM provider.

Settings() with no arguments produces a fully valid offline_test
configuration that passes its own mode-consistency validation.

M17 addition: IngestionSettings
--------------------------------
Controls the document ingestion pipeline (upload limits, job concurrency,
parser timeouts, corpus versioning, external source sync).
Ingestion routes are available in all execution modes; the ingestion
pipeline does not require the LLM provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


class ConfigurationError(Exception):
    """Raised when settings are internally inconsistent or a dependency
    required by the configured execution mode is unavailable."""


class ExecutionMode(str, Enum):
    OFFLINE_TEST = "offline_test"
    RESEARCH_BM25 = "research_bm25"
    RESEARCH_HYBRID = "research_hybrid"


@dataclass(frozen=True)
class CorpusSettings:
    store_path: str = "corpus/corpus_store.json"
    bm25_path: str = "corpus/bm25_index.pkl"
    manifest_path: str = "corpus/manifest.json"
    smoke_corpus_path: str = "corpus/smoke_corpus.json"


@dataclass(frozen=True)
class RetrievalSettings:
    mode: str = "bm25_only"
    bm25_top_k: int = 20
    dense_top_k: int = 20
    rrf_k: int = 60
    rrf_top_k: int = 10
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    # Prefix of the per-version Qdrant collections: each published KB
    # version owns the collection "<qdrant_collection>_<version_id>".
    qdrant_collection: str = "rasvcx_chunks"
    targeted_bm25_top_k: int = 10
    targeted_dense_top_k: int = 10
    targeted_rrf_k: int = 60
    # server   -> Qdrant service at qdrant_host:qdrant_port
    # embedded -> qdrant-client local mode persisted under qdrant_path
    # memory   -> qdrant-client local mode, in-process, rebuilt at startup
    qdrant_mode: str = "server"
    qdrant_path: str = "data/qdrant"

    def __post_init__(self) -> None:
        if self.mode not in ("bm25_only", "hybrid"):
            raise ConfigurationError(
                f"retrieval.mode must be 'bm25_only' or 'hybrid', got {self.mode!r}"
            )
        if self.qdrant_mode not in ("server", "embedded", "memory"):
            raise ConfigurationError(
                "retrieval.qdrant_mode must be 'server', 'embedded' or 'memory', "
                f"got {self.qdrant_mode!r}"
            )
        if not self.qdrant_collection:
            raise ConfigurationError("retrieval.qdrant_collection must be non-empty")
        for name, val in [
            ("bm25_top_k", self.bm25_top_k),
            ("dense_top_k", self.dense_top_k),
            ("rrf_top_k", self.rrf_top_k),
            ("targeted_bm25_top_k", self.targeted_bm25_top_k),
            ("targeted_dense_top_k", self.targeted_dense_top_k),
            ("rrf_k", self.rrf_k),
            ("targeted_rrf_k", self.targeted_rrf_k),
        ]:
            if val < 1:
                raise ConfigurationError(f"retrieval.{name} must be >= 1, got {val}")
        if not (1 <= self.qdrant_port <= 65535):
            raise ConfigurationError(
                f"retrieval.qdrant_port must be in [1, 65535], got {self.qdrant_port}"
            )


@dataclass(frozen=True)
class RerankerSettings:
    """M4 cross-encoder reranking.

    CrossEncoderConfig fields verified from reranking/cross_encoder.py:
    model_name, max_candidates, batch_size, device.
    max_length does NOT exist in CrossEncoderConfig.
    """
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_candidates: int = 30
    batch_size: int = 32
    device: str | None = None
    # Drop evidence the cross-encoder scores below this (see
    # RerankingServiceConfig.min_relevance_score). None = keep everything.
    min_relevance_score: float | None = None

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ConfigurationError(
                f"reranker.max_candidates must be >= 1, got {self.max_candidates}"
            )
        if self.batch_size < 1:
            raise ConfigurationError(
                f"reranker.batch_size must be >= 1, got {self.batch_size}"
            )


@dataclass(frozen=True)
class SufficiencySettings:
    """M5 SufficiencyGateConfig.

    Default min_scored_items=0 matches offline_test mode where reranker
    is disabled and all items have rerank_score=None.  Research modes
    should set min_scored_items=1 to restore the standard gate behaviour.
    """
    min_evidence_items: int = 2
    min_scored_items: int = 0
    min_top_rerank_score: float = 0.0
    min_rerank_score_margin: float = 0.05
    max_unknown_provenance_ratio_high_risk: float = 0.5
    min_source_diversity_high_risk: int = 2
    high_risk_score_threshold: float = 0.7
    # See SufficiencyGateConfig.margin_check_below_top_score. None = always.
    margin_check_below_top_score: float | None = None
    # See SufficiencyGateConfig.diversity_waiver_*. Empty/None = diversity
    # is always enforced for high-risk queries.
    diversity_waiver_source_types: tuple[str, ...] = ()
    diversity_waiver_min_top_score: float | None = None

    def __post_init__(self) -> None:
        from rasvcx.schemas.common import SourceType

        valid = {s.value for s in SourceType}
        unknown = set(self.diversity_waiver_source_types) - valid
        if unknown:
            raise ConfigurationError(
                f"sufficiency.diversity_waiver_source_types: unknown {sorted(unknown)}; "
                f"valid: {sorted(valid)}"
            )
        if bool(self.diversity_waiver_source_types) != (self.diversity_waiver_min_top_score is not None):
            raise ConfigurationError(
                "sufficiency.diversity_waiver_source_types and "
                "diversity_waiver_min_top_score must be set together"
            )
        if self.min_evidence_items < 0:
            raise ConfigurationError(
                "sufficiency.min_evidence_items must be >= 0"
            )
        if self.min_scored_items < 0:
            raise ConfigurationError(
                "sufficiency.min_scored_items must be >= 0"
            )
        if self.min_rerank_score_margin < 0.0:
            raise ConfigurationError(
                "sufficiency.min_rerank_score_margin must be >= 0"
            )
        if not (0.0 <= self.max_unknown_provenance_ratio_high_risk <= 1.0):
            raise ConfigurationError(
                "sufficiency.max_unknown_provenance_ratio_high_risk must be in [0, 1]"
            )
        if self.min_source_diversity_high_risk < 0:
            raise ConfigurationError(
                "sufficiency.min_source_diversity_high_risk must be >= 0"
            )
        if not (0.0 <= self.high_risk_score_threshold <= 1.0):
            raise ConfigurationError(
                "sufficiency.high_risk_score_threshold must be in [0, 1]"
            )


@dataclass(frozen=True)
class NLISettings:
    """M8 NLI backend.

    When enabled=False, NullNLIBackend is injected.  NLI-eligible pairs
    receive SKIP_NLI_BUDGET_EXHAUSTED with reason 'no NLI backend
    available' per selective_router.py:173.
    """
    enabled: bool = False
    model_name: str = "microsoft/deberta-large-mnli"
    device: str = "cpu"


@dataclass(frozen=True)
class ValidationSettings:
    """M8 ValidationConfig sub-fields.

    Verified from validation/validation_types.py.
    """
    numeric_relative_tolerance: float = 0.01
    min_shared_tokens_for_comparison: int = 2
    max_claim_pairs_per_candidate: int = 25
    max_candidates: int = 500
    inconclusive_confidence_threshold: float = 0.6

    def __post_init__(self) -> None:
        if self.numeric_relative_tolerance < 0.0:
            raise ConfigurationError(
                "validation.numeric_relative_tolerance must be >= 0"
            )
        if self.min_shared_tokens_for_comparison < 0:
            raise ConfigurationError(
                "validation.min_shared_tokens_for_comparison must be >= 0"
            )
        if self.max_claim_pairs_per_candidate < 1:
            raise ConfigurationError(
                "validation.max_claim_pairs_per_candidate must be >= 1"
            )
        if self.max_candidates < 1:
            raise ConfigurationError("validation.max_candidates must be >= 1")
        if not (0.0 <= self.inconclusive_confidence_threshold <= 1.0):
            raise ConfigurationError(
                "validation.inconclusive_confidence_threshold must be in [0, 1]"
            )


@dataclass(frozen=True)
class VerificationSettings:
    """M9 VerificationConfig fields.

    Verified from verification/verdict_aggregator.py:31.
    """
    max_claims: int = 200
    max_nli_calls_per_answer: int = 20
    max_fallback_evidence_candidates: int = 5
    unsupported_fraction_for_partial: float = 0.2
    unsupported_fraction_for_unverified: float = 0.5

    def __post_init__(self) -> None:
        if self.max_claims < 1:
            raise ConfigurationError("verification.max_claims must be >= 1")
        if self.max_nli_calls_per_answer < 0:
            raise ConfigurationError(
                "verification.max_nli_calls_per_answer must be >= 0"
            )
        if not (0.0 <= self.unsupported_fraction_for_partial <= 1.0):
            raise ConfigurationError(
                "verification.unsupported_fraction_for_partial must be in [0, 1]"
            )
        if not (0.0 <= self.unsupported_fraction_for_unverified <= 1.0):
            raise ConfigurationError(
                "verification.unsupported_fraction_for_unverified must be in [0, 1]"
            )
        if self.unsupported_fraction_for_partial > self.unsupported_fraction_for_unverified:
            raise ConfigurationError(
                "verification.unsupported_fraction_for_partial must be <= "
                "unsupported_fraction_for_unverified"
            )


@dataclass(frozen=True)
class ConfidenceSettings:
    """M10 EstimatorWeights.

    Range-checked here; sum-to-1.0 validated downstream by
    EstimatorWeights.__post_init__ (confidence/estimator.py:50-69).
    """
    evidence_agreement: float = 0.20
    claim_verification_support: float = 0.25
    contradiction_penalty: float = 0.20
    provenance_quality: float = 0.10
    source_diversity: float = 0.05
    retrieval_quality: float = 0.05
    rerank_quality: float = 0.05
    resolution_uncertainty_penalty: float = 0.10
    # Fitted calibration artifact (scripts/calibrate.py fit).  None = no
    # calibration: confidence is the raw score, labelled "uncalibrated".
    # Excluded from config_hash (the artifact records the hash of the
    # configuration that produced the scores it was fitted on).
    calibration_artifact_path: str | None = None

    def __post_init__(self) -> None:
        for name, val in [
            ("evidence_agreement", self.evidence_agreement),
            ("claim_verification_support", self.claim_verification_support),
            ("contradiction_penalty", self.contradiction_penalty),
            ("provenance_quality", self.provenance_quality),
            ("source_diversity", self.source_diversity),
            ("retrieval_quality", self.retrieval_quality),
            ("rerank_quality", self.rerank_quality),
            ("resolution_uncertainty_penalty", self.resolution_uncertainty_penalty),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ConfigurationError(
                    f"confidence.{name} must be in [0, 1], got {val}"
                )


@dataclass(frozen=True)
class DecisionSettings:
    """M10 DecisionThresholds fields.

    Ordering: 0 <= regenerate_min <= warning_min <= answer_min <= 1
    and equivalently for high-risk variants.
    Verified from decision/decision_types.py:37.
    """
    answer_min: float = 0.75
    warning_min: float = 0.55
    regenerate_min: float = 0.35
    high_risk_answer_min: float = 0.85
    high_risk_warning_min: float = 0.65
    high_risk_regenerate_min: float = 0.45
    high_risk_threshold: float = 0.6

    def __post_init__(self) -> None:
        if not (0.0 <= self.regenerate_min <= self.warning_min <= self.answer_min <= 1.0):
            raise ConfigurationError(
                "decision thresholds must satisfy "
                "0 <= regenerate_min <= warning_min <= answer_min <= 1"
            )
        if not (
            0.0
            <= self.high_risk_regenerate_min
            <= self.high_risk_warning_min
            <= self.high_risk_answer_min
            <= 1.0
        ):
            raise ConfigurationError(
                "decision high-risk thresholds must satisfy "
                "0 <= high_risk_regenerate_min <= high_risk_warning_min "
                "<= high_risk_answer_min <= 1"
            )
        if not (0.0 <= self.high_risk_threshold <= 1.0):
            raise ConfigurationError(
                "decision.high_risk_threshold must be in [0, 1]"
            )


@dataclass(frozen=True)
class LLMSettings:
    """M11 provider configuration.

    _api_key is never logged, never returned by admin endpoint,
    and does not appear in repr().
    """
    provider: str = "stub"
    model_name: str = "gemini-3.5-flash-lite"
    temperature: float = 0.0
    # Includes the model's thinking tokens, which count against the limit.
    max_tokens: int = 4096
    timeout_seconds: float = 60.0
    # Retries of transient provider overload only (HTTP 429/500/503).
    max_retries: int = 2
    stub_response: str = (
        "Based on the retrieved evidence, visiting hours are 9am to 8pm. [E1]"
    )
    _api_key: str = field(default="", repr=False)

    @property
    def api_key(self) -> str:
        return self._api_key

    def __post_init__(self) -> None:
        if self.provider not in ("stub", "gemini"):
            raise ConfigurationError(
                f"llm.provider must be 'stub' or 'gemini', got {self.provider!r}"
            )
        if self.provider == "gemini" and not self._api_key:
            raise ConfigurationError(
                "llm.api_key must be set when llm.provider is 'gemini'. "
                "Set via RASVCX_LLM_API_KEY environment variable."
            )
        if not (0.0 <= self.temperature <= 2.0):
            raise ConfigurationError(
                f"llm.temperature must be in [0, 2], got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ConfigurationError(
                f"llm.max_tokens must be >= 1, got {self.max_tokens}"
            )
        if self.timeout_seconds <= 0:
            raise ConfigurationError("llm.timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise ConfigurationError("llm.max_retries must be >= 0")


@dataclass(frozen=True)
class PipelineSettings:
    """M12 PipelineOrchestrator settings."""
    max_corrective_attempts: int = 2
    max_context_chars: int = 100_000

    def __post_init__(self) -> None:
        if self.max_corrective_attempts < 0:
            raise ConfigurationError(
                "pipeline.max_corrective_attempts must be >= 0"
            )
        if self.max_context_chars < 1:
            raise ConfigurationError("pipeline.max_context_chars must be >= 1")


@dataclass(frozen=True)
class ChunkingSettings:
    """Chunking parameters used during corpus ingestion.

    Must match the corpus manifest fingerprint.  ChunkStrategy values
    verified from retrieval/chunking.py:31-33.
    """
    strategy: str = "fixed"
    max_tokens: int = 256
    overlap: int = 32

    def __post_init__(self) -> None:
        if self.strategy not in ("fixed", "sentence_window"):
            raise ConfigurationError(
                f"chunking.strategy must be 'fixed' or 'sentence_window', "
                f"got {self.strategy!r}"
            )
        if self.max_tokens < 1:
            raise ConfigurationError(
                f"chunking.max_tokens must be >= 1, got {self.max_tokens}"
            )
        if self.overlap < 0:
            raise ConfigurationError(
                f"chunking.overlap must be >= 0, got {self.overlap}"
            )
        if self.overlap >= self.max_tokens:
            raise ConfigurationError(
                f"chunking.overlap ({self.overlap}) must be < "
                f"chunking.max_tokens ({self.max_tokens})"
            )


@dataclass(frozen=True)
class APISettings:
    """HTTP API layer settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_workers: int = 4
    max_queue_depth: int = 8
    request_timeout_seconds: float = 120.0
    max_request_bytes: int = 65_536
    max_query_chars: int = 2_048
    require_auth: bool = False
    # Answer cache: identical question + context against the same
    # knowledge-base version returns the stored response. 0 disables.
    cache_size: int = 0
    cache_ttl_seconds: float = 3600.0
    # Per-principal request budget for /query, /feedback and ingestion
    # writes (token bucket, in-process).  0 disables.  A principal is the
    # bearer token (hashed) when auth is on, else the client address.
    rate_limit_per_minute: int = 0
    _auth_token: str = field(default="", repr=False)
    # Optional separate credential for ingestion and /admin (writes to the
    # knowledge base).  Empty = the auth token is also the admin token.
    _admin_token: str = field(default="", repr=False)

    @property
    def auth_token(self) -> str:
        return self._auth_token

    @property
    def admin_token(self) -> str:
        return self._admin_token or self._auth_token

    def __post_init__(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ConfigurationError(
                f"api.port must be in [1, 65535], got {self.port}"
            )
        if self.max_workers < 1:
            raise ConfigurationError(
                f"api.max_workers must be >= 1, got {self.max_workers}"
            )
        if self.max_queue_depth < 0:
            raise ConfigurationError(
                f"api.max_queue_depth must be >= 0, got {self.max_queue_depth}"
            )
        if self.request_timeout_seconds <= 0:
            raise ConfigurationError("api.request_timeout_seconds must be > 0")
        if self.max_request_bytes < 1:
            raise ConfigurationError(
                f"api.max_request_bytes must be >= 1, got {self.max_request_bytes}"
            )
        if self.max_query_chars < 1:
            raise ConfigurationError(
                f"api.max_query_chars must be >= 1, got {self.max_query_chars}"
            )
        if self.rate_limit_per_minute < 0:
            raise ConfigurationError("api.rate_limit_per_minute must be >= 0")
        if self.require_auth and not self._auth_token:
            raise ConfigurationError(
                "api.auth_token must be set when api.require_auth is True. "
                "Set via RASVCX_AUTH_TOKEN environment variable."
            )


# ---------------------------------------------------------------------------
# M17 — IngestionSettings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestionSettings:
    """M17 document ingestion pipeline configuration.

    Upload size limits are enforced before parsing.  Per-format limits
    reflect verified parser memory constraints:
      - PDF/TXT/HTML/CSV/XLSX: streaming or lazy — up to 1 GB
      - DOCX: full ZIP load — limited to 100 MB
      - PPTX: full ZIP load — limited to 50 MB

    Parser isolation:
      Each document is parsed in an isolated subprocess via
      multiprocessing.Process.  The subprocess is killed after
      parser_timeout_seconds if it does not complete.  This is the only
      cross-platform mechanism that guarantees termination (signal.SIGALRM
      is not available on Windows; threading.Timer cannot kill blocked I/O).

    Job store:
      SQLite with WAL mode.  Single Uvicorn worker only
      (--workers 1).  Multi-process Uvicorn is not supported.

    Versioned publication:
      Each completed ingestion creates a versioned snapshot under
      corpus_versions_dir/<version_id>/.  active_version_path is
      written atomically (os.replace) as the commit pointer.

    External source sync:
      scheduled_sync_interval_seconds controls how often the background
      asyncio task checks for due source synchronizations.
      Set to 0 to disable scheduled sync (manual refresh only).
    """

    # --------------- Upload limits ---------------
    max_upload_bytes: int = 1_073_741_824           # 1 GB
    max_upload_bytes_docx: int = 104_857_600        # 100 MB
    max_upload_bytes_pptx: int = 52_428_800         # 50 MB
    max_pdf_pages: int = 10_000
    max_csv_rows: int = 10_000_000
    max_xlsx_rows: int = 1_000_000

    # --------------- Parser isolation ---------------
    parser_timeout_seconds: float = 300.0

    # --------------- Job store ---------------
    db_path: str = "corpus/ingestion_jobs.db"
    max_concurrent_jobs: int = 2

    # --------------- Temp storage ---------------
    temp_dir: str = ""

    # --------------- Versioned corpus publication ---------------
    corpus_versions_dir: str = "corpus"
    active_version_path: str = "corpus/active_version.json"
    # Retention: a superseded version is deleted only when no request holds
    # a lease on it, it has been inactive for gc_delay_seconds, and it is
    # not among the keep_versions most recent superseded versions.
    gc_delay_seconds: float = 300.0
    keep_versions: int = 2
    # When True, startup FAILS instead of serving the seed corpus if no
    # published version exists at active_version_path (production setting).
    require_published_kb: bool = False

    # --------------- External source sync ---------------
    max_source_response_bytes: int = 104_857_600    # 100 MB
    source_connect_timeout_seconds: float = 10.0
    source_read_timeout_seconds: float = 60.0
    allowed_url_schemes: tuple[str, ...] = ("https",)
    scheduled_sync_interval_seconds: float = 3600.0
    max_redirects: int = 3

    def __post_init__(self) -> None:
        if self.max_upload_bytes < 1:
            raise ConfigurationError("ingestion.max_upload_bytes must be >= 1")
        if self.max_upload_bytes_docx < 1:
            raise ConfigurationError("ingestion.max_upload_bytes_docx must be >= 1")
        if self.max_upload_bytes_pptx < 1:
            raise ConfigurationError("ingestion.max_upload_bytes_pptx must be >= 1")
        if self.max_pdf_pages < 1:
            raise ConfigurationError("ingestion.max_pdf_pages must be >= 1")
        if self.parser_timeout_seconds <= 0:
            raise ConfigurationError("ingestion.parser_timeout_seconds must be > 0")
        if self.max_concurrent_jobs < 1:
            raise ConfigurationError("ingestion.max_concurrent_jobs must be >= 1")
        if self.gc_delay_seconds < 0:
            raise ConfigurationError("ingestion.gc_delay_seconds must be >= 0")
        if self.keep_versions < 0:
            raise ConfigurationError("ingestion.keep_versions must be >= 0")
        if self.max_source_response_bytes < 1:
            raise ConfigurationError("ingestion.max_source_response_bytes must be >= 1")
        if self.source_connect_timeout_seconds <= 0:
            raise ConfigurationError(
                "ingestion.source_connect_timeout_seconds must be > 0"
            )
        if self.source_read_timeout_seconds <= 0:
            raise ConfigurationError(
                "ingestion.source_read_timeout_seconds must be > 0"
            )
        if not self.allowed_url_schemes:
            raise ConfigurationError("ingestion.allowed_url_schemes must not be empty")
        if self.max_redirects < 0:
            raise ConfigurationError("ingestion.max_redirects must be >= 0")
        if self.scheduled_sync_interval_seconds < 0:
            raise ConfigurationError(
                "ingestion.scheduled_sync_interval_seconds must be >= 0"
            )


# ---------------------------------------------------------------------------
# Root Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    """Root configuration object.

    Constructed by load_settings() and passed to build_pipeline().
    Never mutated after construction.

    Settings() with no arguments is a valid offline_test configuration.
    """
    execution_mode: ExecutionMode = ExecutionMode.OFFLINE_TEST
    corpus: CorpusSettings = field(default_factory=CorpusSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    reranker: RerankerSettings = field(default_factory=RerankerSettings)
    sufficiency: SufficiencySettings = field(default_factory=SufficiencySettings)
    nli: NLISettings = field(default_factory=NLISettings)
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    verification: VerificationSettings = field(default_factory=VerificationSettings)
    confidence: ConfidenceSettings = field(default_factory=ConfidenceSettings)
    decision: DecisionSettings = field(default_factory=DecisionSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    api: APISettings = field(default_factory=APISettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)

    def __post_init__(self) -> None:
        if not isinstance(self.execution_mode, ExecutionMode):
            try:
                object.__setattr__(
                    self, "execution_mode", ExecutionMode(self.execution_mode)
                )
            except ValueError:
                valid = [m.value for m in ExecutionMode]
                raise ConfigurationError(
                    f"execution_mode must be one of {valid}, "
                    f"got {self.execution_mode!r}"
                )
        self._validate_mode_consistency()

    def _validate_mode_consistency(self) -> None:
        mode = self.execution_mode

        if mode is ExecutionMode.OFFLINE_TEST:
            if self.retrieval.mode != "bm25_only":
                raise ConfigurationError(
                    "offline_test mode requires retrieval.mode='bm25_only'"
                )
            if self.reranker.enabled:
                raise ConfigurationError(
                    "offline_test mode requires reranker.enabled=False"
                )
            if self.nli.enabled:
                raise ConfigurationError(
                    "offline_test mode requires nli.enabled=False"
                )
            if self.llm.provider != "stub":
                raise ConfigurationError(
                    "offline_test mode requires llm.provider='stub'"
                )

        if mode in (ExecutionMode.RESEARCH_BM25, ExecutionMode.RESEARCH_HYBRID):
            if not self.reranker.enabled:
                raise ConfigurationError(
                    f"{mode.value} mode requires reranker.enabled=True"
                )
            if not self.nli.enabled:
                raise ConfigurationError(
                    f"{mode.value} mode requires nli.enabled=True"
                )
            if self.llm.provider == "stub":
                raise ConfigurationError(
                    f"{mode.value} mode requires a real LLM provider (not 'stub')"
                )

        if mode is ExecutionMode.RESEARCH_HYBRID:
            if self.retrieval.mode != "hybrid":
                raise ConfigurationError(
                    "research_hybrid mode requires retrieval.mode='hybrid'"
                )


    @property
    def is_offline(self) -> bool:
        """True only for the explicit offline_test mode (test doubles in use)."""
        return self.execution_mode is ExecutionMode.OFFLINE_TEST


def offline_test_settings() -> Settings:
    """Return a valid offline_test Settings object. Identical to Settings()."""
    return Settings()


__all__ = [
    "ConfigurationError",
    "ExecutionMode",
    "Settings",
    "CorpusSettings",
    "ChunkingSettings",
    "RetrievalSettings",
    "RerankerSettings",
    "SufficiencySettings",
    "NLISettings",
    "ValidationSettings",
    "VerificationSettings",
    "ConfidenceSettings",
    "DecisionSettings",
    "LLMSettings",
    "PipelineSettings",
    "APISettings",
    "IngestionSettings",
    "offline_test_settings",
]