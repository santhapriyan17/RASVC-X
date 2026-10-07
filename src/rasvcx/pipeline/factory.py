"""Pipeline factory for RASVC-X (Module 13).

build_runtime(settings) assembles everything one process needs to serve
queries and returns it as a PipelineRuntime:

  orchestrator      PipelineOrchestrator with every M2-M11 stage wired
  snapshot_loader   builds KBSnapshot objects (store + BM25 + Qdrant) for
                    this configuration; used at startup and by publication
  dense_backend     shared encoder + Qdrant client (hybrid mode), else None
  initial_snapshot  the knowledge-base version to serve first

build_pipeline(settings) is the orchestrator-only convenience wrapper.

Which knowledge base is served at startup
-----------------------------------------
  1. If ingestion.active_version_path exists, that published version is
     loaded.  A pointer that cannot be loaded is a startup error -- the
     process never falls back to some other corpus behind the operator's
     back.
  2. Otherwise the seed corpus (corpus.store_path / corpus.bm25_path, built
     by scripts/build_index.py) is served as the bootstrap version.
  3. offline_test only: if the seed index does not exist either, the smoke
     corpus is indexed in memory.

In hybrid mode the version's Qdrant collection must hold exactly one point
per chunk.  A missing collection is rebuilt from the corpus store (it is a
derived index; this is what makes qdrant_mode=memory usable) and the
rebuild is logged.  A collection with the wrong point count is an error.

No silent substitution
----------------------
Research modes require a real LLM provider, a real reranker and a real NLI
model (enforced by Settings).  Every one of them is constructed -- and the
models loaded -- here, at startup, so a missing dependency, an unreachable
Qdrant or an unloadable model stops the process instead of surfacing later
as a quietly degraded answer.  No billable provider call is made.

M2 routing (route_query) is called per-request inside
PipelineOrchestrator.run() and is not injected by this factory.

Raises:
  ConfigurationError: missing dependency, unreachable service, bad KB.
  FileNotFoundError:  seed corpus / smoke corpus not found.
  StaleIndexError:    seed manifest does not match the seed store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasvcx.claims import AtomicClaimPipeline
from rasvcx.confidence import ConfidencePipeline
from rasvcx.confidence.estimator import EstimatorWeights, RawReliabilityEstimator
from rasvcx.config.settings import (
    ConfigurationError,
    ExecutionMode,
    Settings,
)
from rasvcx.decision import DecisionEngine
from rasvcx.decision.decision_types import DecisionThresholds
from rasvcx.generation import Generator
from rasvcx.generation.generation_types import LLMConfig
from rasvcx.generation.llm_client import MockLLMClient
from rasvcx.pipeline.orchestrator import PipelineOrchestrator
from rasvcx.provenance import ContextExtractor, ProvenanceAnalyzer, SourceQualityScorer
from rasvcx.reranking import RerankingService, RerankingServiceConfig
from rasvcx.reranking.cross_encoder import CrossEncoderConfig
from rasvcx.retrieval.bm25 import BM25Index
from rasvcx.retrieval.bridge import DisabledRerankingService
from rasvcx.retrieval.chunking import ChunkConfig, ChunkStrategy
from rasvcx.retrieval.corpus import (
    CorpusManifest,
    CorpusStore,
    StaleIndexError,
    build_corpus_store,
    load_corpus_json,
)
from rasvcx.retrieval.knowledge_base import (
    KB_SOURCE_PUBLISHED,
    KB_SOURCE_SEED,
    KB_SOURCE_SMOKE,
    KBIntegrityError,
    KBSnapshot,
    SnapshotLoader,
    file_sha256,
)
from rasvcx.routing.risk_router import RiskRouterThresholds, RiskRouterWeights
from rasvcx.sufficiency.gate import SufficiencyGateConfig
from rasvcx.validation import (
    NLIService,
    NullNLIBackend,
    TransformersNLIBackend,
    ValidationPipeline,
)
from rasvcx.validation.validation_types import (
    CandidateGenerationConfig,
    DeterministicConfig,
    SelectiveRoutingConfig,
    ValidationConfig,
)
from rasvcx.verification import VerificationPipeline
from rasvcx.verification.verdict_aggregator import VerificationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineRuntime:
    """Everything one process needs to serve queries. Immutable.

    The orchestrator and the models it holds are built once and shared by
    all requests.  Knowledge-base versions are NOT part of the orchestrator:
    each request leases a KBSnapshot and passes that snapshot's retrieval
    callables to orchestrator.run().
    """

    settings: Settings
    orchestrator: PipelineOrchestrator
    snapshot_loader: SnapshotLoader
    dense_backend: Any | None
    initial_snapshot: KBSnapshot
    nli_service: NLIService
    reranking_service: Any
    llm_client: Any

    def identity(self) -> dict[str, Any]:
        """What produced an answer, apart from the KB version: the
        configuration, the prompt template, every model in the path, and
        the calibration applied to its confidence."""
        ident = runtime_identity(self.settings)
        cal = self.calibration
        ident["calibration"] = {
            "status": cal["status"], "calibration_version": cal.get("calibration_version"),
            "calibration_dataset_hash": cal.get("calibration_dataset_hash"),
            "reason": cal.get("reason"),
        }
        return ident

    calibration: dict[str, Any] = None  # type: ignore[assignment]  # set by build_runtime


def runtime_identity(s: Settings) -> dict[str, Any]:
    from rasvcx.generation.prompt_builder import PROMPT_VERSION

    return {
        "execution_mode": s.execution_mode.value,
        "config_hash": config_hash(s),
        "prompt_version": PROMPT_VERSION,
        "model_versions": {
            "llm": None if s.llm.provider == "stub" else f"{s.llm.provider}:{s.llm.model_name}",
            "dense": s.retrieval.dense_model_name if s.retrieval.mode == "hybrid" else None,
            "reranker": s.reranker.model_name if s.reranker.enabled else None,
            "nli": s.nli.model_name if s.nli.enabled else None,
        },
    }


#: Settings fields that describe WHERE an artifact lives, not how the
#: pipeline behaves; excluded so pointing at a calibration artifact does
#: not change the identity that artifact was fitted under.
_HASH_EXCLUDED_FIELDS = frozenset({"calibration_artifact_path"})


def config_hash(settings: Settings) -> str:
    """SHA-256 (16 hex) of every non-secret setting.

    Private fields (the API key, the auth token) are excluded, so the hash
    can be published with results without leaking anything.
    """
    import dataclasses
    import hashlib
    import json

    def _clean(value: Any) -> Any:
        if dataclasses.is_dataclass(value):
            return {
                f.name: _clean(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if not f.name.startswith("_") and f.name not in _HASH_EXCLUDED_FIELDS
            }
        if isinstance(value, (list, tuple)):
            return [_clean(v) for v in value]
        if hasattr(value, "value") and not isinstance(value, (int, float, str)):
            return value.value
        return value

    blob = json.dumps(_clean(settings), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_pipeline(settings: Settings) -> PipelineOrchestrator:
    """Construct and return a fully-wired PipelineOrchestrator.

    Equivalent to build_runtime(settings).orchestrator.  The orchestrator's
    default retrieval callables are bound to the initial KB snapshot.
    """
    return build_runtime(settings).orchestrator


def build_runtime(settings: Settings) -> PipelineRuntime:
    """Construct the full query runtime and validate it at startup.

    Raises:
        ConfigurationError, FileNotFoundError, StaleIndexError, ImportError
    """
    mode = settings.execution_mode
    logger.info("Building pipeline (mode=%s)", mode.value)

    # 1. Retrieval backends + initial knowledge-base snapshot
    dense_backend = _build_dense_backend(settings)
    try:
        snapshot_loader = SnapshotLoader(settings.retrieval, dense_backend)
        snapshot = _resolve_initial_snapshot(settings, snapshot_loader)
    except KBIntegrityError as exc:
        raise ConfigurationError(f"Knowledge base cannot be served: {exc}") from exc

    # 2. NLI backend (shared by M8 validation and M9 verification)
    nli_service = _build_nli_service(settings)

    # 3. M6 provenance
    provenance_analyzer = ProvenanceAnalyzer(SourceQualityScorer(), ContextExtractor())

    # 4. M7 claims
    claim_pipeline = AtomicClaimPipeline()

    # 5. M8 validation
    validation_pipeline = _build_validation_pipeline(settings, nli_service)

    # 6. M9 verification
    verification_pipeline = _build_verification_pipeline(settings, nli_service)

    # 7. M10 confidence + decision
    calibrator, calibration_info = _build_calibrator(settings)
    confidence_pipeline = _build_confidence_pipeline(settings, calibrator)
    decision_engine = _build_decision_engine(settings)

    # 8. M4 reranker
    reranking_service = _build_reranking_service(settings)

    # 9. M11 LLM + Generator
    generator, llm_client = _build_generator(settings)

    # 10. M12 orchestrator -- inject SufficiencyGateConfig from settings
    sufficiency_cfg = SufficiencyGateConfig(
        min_scored_items=settings.sufficiency.min_scored_items,
        min_evidence_items=settings.sufficiency.min_evidence_items,
        min_top_rerank_score=settings.sufficiency.min_top_rerank_score,
        min_rerank_score_margin=settings.sufficiency.min_rerank_score_margin,
        max_unknown_provenance_ratio_high_risk=settings.sufficiency.max_unknown_provenance_ratio_high_risk,
        min_source_diversity_high_risk=settings.sufficiency.min_source_diversity_high_risk,
        high_risk_score_threshold=settings.sufficiency.high_risk_score_threshold,
        margin_check_below_top_score=settings.sufficiency.margin_check_below_top_score,
        diversity_waiver_source_types=settings.sufficiency.diversity_waiver_source_types,
        diversity_waiver_min_top_score=settings.sufficiency.diversity_waiver_min_top_score,
    )

    orchestrator = PipelineOrchestrator(
        retrieval_fn=snapshot.retrieval_fn,
        reranking_service=reranking_service,
        claim_pipeline=claim_pipeline,
        validation_pipeline=validation_pipeline,
        generator=generator,
        verification_pipeline=verification_pipeline,
        confidence_pipeline=confidence_pipeline,
        decision_engine=decision_engine,
        max_corrective_attempts=settings.pipeline.max_corrective_attempts,
        targeted_retrieval_fn=snapshot.targeted_retrieval_fn,
        sufficiency_config=sufficiency_cfg,
        provenance_analyzer=provenance_analyzer,
    )

    logger.info(
        "Pipeline built successfully (mode=%s kb_version=%s chunks=%d qdrant=%s)",
        mode.value, snapshot.version_id, snapshot.chunk_count, snapshot.qdrant_collection,
    )
    return PipelineRuntime(
        settings=settings,
        orchestrator=orchestrator,
        snapshot_loader=snapshot_loader,
        dense_backend=dense_backend,
        initial_snapshot=snapshot,
        nli_service=nli_service,
        reranking_service=reranking_service,
        llm_client=llm_client,
        calibration=calibration_info,
    )


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------


def _build_dense_backend(settings: Settings) -> Any | None:
    """Encoder + Qdrant client for hybrid mode; None for bm25_only."""
    if settings.retrieval.mode != "hybrid":
        return None

    s = settings.retrieval
    logger.info(
        "Hybrid mode -- dense backend: qdrant_mode=%s host=%s port=%d path=%s model=%s",
        s.qdrant_mode, s.qdrant_host, s.qdrant_port, s.qdrant_path, s.dense_model_name,
    )
    try:
        from rasvcx.retrieval.dense import DenseBackend, _QDRANT_AVAILABLE, _ST_AVAILABLE
    except ImportError as exc:  # pragma: no cover - module itself always imports
        raise ConfigurationError(str(exc)) from exc
    if not _QDRANT_AVAILABLE or not _ST_AVAILABLE:
        raise ConfigurationError(
            "research_hybrid mode requires qdrant-client and sentence-transformers "
            "(pip install -e \".[research]\")"
        )
    if s.qdrant_mode == "embedded":
        Path(s.qdrant_path).mkdir(parents=True, exist_ok=True)
    try:
        backend = DenseBackend(
            s.dense_model_name,
            mode=s.qdrant_mode,
            host=s.qdrant_host,
            port=s.qdrant_port,
            path=s.qdrant_path,
        )
        backend.list_collections()  # connectivity probe
    except Exception as exc:
        target = (
            f"{s.qdrant_host}:{s.qdrant_port}" if s.qdrant_mode == "server" else s.qdrant_mode
        )
        raise ConfigurationError(
            f"research_hybrid requires Qdrant, but it is not usable ({target}): {exc}"
        ) from exc
    return backend


def _resolve_initial_snapshot(settings: Settings, loader: SnapshotLoader) -> KBSnapshot:
    """Pick and load the knowledge-base version to serve at startup."""
    from rasvcx.ingestion.publisher import read_active_version

    pointer_path = settings.ingestion.active_version_path
    if Path(pointer_path).exists():
        active = read_active_version(pointer_path)
        if not active or not active.get("version_id"):
            raise KBIntegrityError(
                f"active version pointer {pointer_path} exists but is unreadable"
            )
        store_path = Path(str(active.get("store_path") or ""))
        bm25_path = Path(str(active.get("bm25_path") or ""))
        if not store_path.is_file() or not bm25_path.is_file():
            raise KBIntegrityError(
                f"active version {active['version_id']} (from {pointer_path}) "
                f"references missing artifacts: {store_path}, {bm25_path}"
            )
        store = CorpusStore.load(store_path)
        bm25 = BM25Index.load(bm25_path)
        logger.info(
            "Serving published KB version %s (%d chunks) from %s",
            active["version_id"], len(store), pointer_path,
        )
        return _snapshot_with_dense(
            loader, str(active["version_id"]), store, bm25,
            qdrant_collection=active.get("qdrant_collection"),
            published_at=active.get("published_at"),
            kb_source=KB_SOURCE_PUBLISHED,
            index_hash=file_sha256(bm25_path),
        )

    store, bm25, kb_source, index_hash = _load_seed_corpus(settings)
    version_id = seed_version_id(store)
    # WARNING, not INFO: an operator who expected a published version (for
    # example a mistyped RASVCX_ACTIVE_VERSION_PATH) must see that the seed
    # is being served.  Every response also carries kb_source.
    logger.warning(
        "No published KB version at %s -- serving %s corpus as %s (%d chunks)",
        pointer_path, kb_source, version_id, len(store),
    )
    return _snapshot_with_dense(
        loader, version_id, store, bm25, kb_source=kb_source, index_hash=index_hash,
    )


def seed_version_id(store: CorpusStore) -> str:
    """Version id of a corpus, derived from its content (same scheme as
    ingestion builds, so re-ingesting nothing reproduces the same id)."""
    from rasvcx.ingestion.corpus_builder import _fingerprint

    return f"v_{_fingerprint(store.to_records())[:12]}"


def _snapshot_with_dense(
    loader: SnapshotLoader,
    version_id: str,
    store: CorpusStore,
    bm25: BM25Index,
    qdrant_collection: str | None = None,
    published_at: str | None = None,
    kb_source: str = KB_SOURCE_PUBLISHED,
    index_hash: str | None = None,
) -> KBSnapshot:
    """Build the snapshot, (re)building a MISSING Qdrant collection first."""
    origin = "persisted"
    if loader.hybrid:
        backend = loader.dense_backend
        qdrant_collection = qdrant_collection or loader.collection_name(version_id)
        if backend.collection_count(qdrant_collection) is None:
            logger.warning(
                "Qdrant collection %s for KB version %s does not exist -- "
                "building it from the corpus store (%d chunks)",
                qdrant_collection, version_id, len(store),
            )
            backend.build_collection(qdrant_collection, store.to_documents_list())
            origin = "rebuilt_from_store"
    return loader.build(
        version_id=version_id,
        corpus_store=store,
        bm25_index=bm25,
        qdrant_collection=qdrant_collection,
        published_at=published_at,
        kb_source=kb_source,
        index_hash=index_hash,
        dense_index_origin=origin,
    )


def _load_seed_corpus(
    settings: Settings,
) -> tuple[CorpusStore, BM25Index, str, str | None]:
    """Load the seed CorpusStore and BM25Index, validating the manifest.

    Returns (store, bm25, kb_source, index_hash).

    In offline_test mode, the smoke corpus is used if the seed index files
    do not exist (so tests can run without running build_index).  In
    research modes the seed index must exist.
    """
    mode = settings.execution_mode
    corpus_path = Path(settings.corpus.store_path)
    bm25_path = Path(settings.corpus.bm25_path)
    manifest_path = Path(settings.corpus.manifest_path)

    if mode is ExecutionMode.OFFLINE_TEST:
        if not corpus_path.exists() or not bm25_path.exists():
            logger.info(
                "Index not found at %s; using smoke corpus for offline_test",
                corpus_path,
            )
            smoke_path = Path(settings.corpus.smoke_corpus_path)
            if not smoke_path.exists():
                raise FileNotFoundError(
                    f"Smoke corpus not found: {smoke_path}. "
                    f"Run: python scripts/build_index.py "
                    f"--corpus {smoke_path} --store {corpus_path} "
                    f"--bm25 {bm25_path} --manifest {manifest_path}"
                )
            docs = load_corpus_json(smoke_path)
            chunk_cfg = ChunkConfig(
                strategy=ChunkStrategy(settings.chunking.strategy),
                max_tokens=settings.chunking.max_tokens,
                overlap=settings.chunking.overlap,
            )
            store, pairs = build_corpus_store(docs, chunk_cfg)
            bm25 = BM25Index.build(pairs)
            logger.info(
                "Offline_test: built in-memory corpus (%d chunks) from %s",
                len(store), smoke_path,
            )
            return store, bm25, KB_SOURCE_SMOKE, None

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"CorpusStore not found: {corpus_path}. "
            f"Run scripts/build_index.py to build the index."
        )

    store = CorpusStore.load(corpus_path)
    bm25 = BM25Index.load(bm25_path)

    if manifest_path.exists():
        manifest = CorpusManifest.load(manifest_path)
        if manifest.chunk_count != len(store):
            raise StaleIndexError(
                f"CorpusStore has {len(store)} chunks but manifest records "
                f"{manifest.chunk_count}.  Rebuild with scripts/build_index.py"
            )
        logger.info(
            "Corpus validated: %d chunks, fingerprint %s...",
            len(store), manifest.corpus_fingerprint[:16],
        )
    else:
        logger.warning(
            "No corpus manifest found at %s; skipping fingerprint check",
            manifest_path,
        )

    return store, bm25, KB_SOURCE_SEED, file_sha256(bm25_path)


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------


def _build_nli_service(settings: Settings) -> NLIService:
    if not settings.nli.enabled:
        logger.info("NLI disabled -- using NullNLIBackend")
        return NLIService(NullNLIBackend())

    logger.info(
        "NLI enabled -- loading TransformersNLIBackend (model=%s device=%s)",
        settings.nli.model_name, settings.nli.device,
    )
    backend = TransformersNLIBackend(
        model_name=settings.nli.model_name,
        device=settings.nli.device,
    )
    # Load now, not on the first request: an NLI model that cannot be
    # loaded must stop startup rather than turn every escalation into a
    # silent "no signal".
    try:
        backend.load()
    except Exception as exc:
        raise ConfigurationError(
            f"nli.enabled=True but the NLI model could not be loaded: {exc}"
        ) from exc
    return NLIService(backend)


def _build_validation_pipeline(
    settings: Settings, nli_service: NLIService,
) -> ValidationPipeline:
    s = settings.validation
    config = ValidationConfig(
        deterministic=DeterministicConfig(
            numeric_relative_tolerance=s.numeric_relative_tolerance,
            min_shared_tokens_for_comparison=s.min_shared_tokens_for_comparison,
            max_claim_pairs_per_candidate=s.max_claim_pairs_per_candidate,
        ),
        candidate_generation=CandidateGenerationConfig(
            max_candidates=s.max_candidates,
        ),
        selective_routing=SelectiveRoutingConfig(
            inconclusive_confidence_threshold=s.inconclusive_confidence_threshold,
        ),
    )
    return ValidationPipeline(nli_service=nli_service, config=config)


def _build_verification_pipeline(
    settings: Settings, nli_service: NLIService,
) -> VerificationPipeline:
    s = settings.verification
    config = VerificationConfig(
        max_claims=s.max_claims,
        max_nli_calls_per_answer=s.max_nli_calls_per_answer,
        max_fallback_evidence_candidates=s.max_fallback_evidence_candidates,
        unsupported_fraction_for_partial=s.unsupported_fraction_for_partial,
        unsupported_fraction_for_unverified=s.unsupported_fraction_for_unverified,
    )
    return VerificationPipeline(nli_service=nli_service, config=config)


def _build_calibrator(settings: Settings) -> tuple[Any, dict[str, Any]]:
    """Calibrator for this process + its status for /ready and responses.

    No artifact configured  -> uncalibrated (raw score, labelled raw).
    Artifact unreadable     -> startup error (a configured calibration that
                               cannot be loaded must not silently vanish).
    Artifact for another config / prompt / model -> invalidated: the raw
                               score is used and every response says why.
    """
    from rasvcx.confidence import Calibrator
    from rasvcx.confidence.calibration_artifact import load_artifact, runtime_mismatch

    path = settings.confidence.calibration_artifact_path
    if not path:
        return Calibrator(None), {"status": "uncalibrated", "reason": "no calibration artifact configured"}
    try:
        bound = load_artifact(path)
    except Exception as exc:
        raise ConfigurationError(f"calibration artifact {path} cannot be loaded: {exc}") from exc
    meta = {
        "calibration_version": bound.calibration_version,
        "calibration_dataset_hash": bound.calibration_dataset_hash,
        "kb_version_id": bound.kb_version_id,
    }
    mismatch = runtime_mismatch(bound, runtime_identity(settings))
    if mismatch:
        logger.warning("calibration artifact %s INVALIDATED: %s", path, mismatch)
        return Calibrator(None, invalid_reason=mismatch, **{k: meta[k] for k in (
            "calibration_version", "calibration_dataset_hash")}), {
            "status": "invalidated", "reason": mismatch, **meta}
    logger.info("calibration %s loaded (fitted on KB %s)", bound.calibration_version, bound.kb_version_id)
    return Calibrator(
        bound.artifact, kb_version_id=bound.kb_version_id,
        calibration_version=bound.calibration_version,
        calibration_dataset_hash=bound.calibration_dataset_hash,
    ), {"status": "calibrated", **meta}


def _build_confidence_pipeline(settings: Settings, calibrator: Any = None) -> ConfidencePipeline:
    s = settings.confidence
    weights = EstimatorWeights(
        evidence_agreement=s.evidence_agreement,
        claim_verification_support=s.claim_verification_support,
        contradiction_penalty=s.contradiction_penalty,
        provenance_quality=s.provenance_quality,
        source_diversity=s.source_diversity,
        retrieval_quality=s.retrieval_quality,
        rerank_quality=s.rerank_quality,
        resolution_uncertainty_penalty=s.resolution_uncertainty_penalty,
    )
    return ConfidencePipeline(estimator=RawReliabilityEstimator(weights=weights), calibrator=calibrator)


def _build_decision_engine(settings: Settings) -> DecisionEngine:
    s = settings.decision
    thresholds = DecisionThresholds(
        answer_min=s.answer_min,
        warning_min=s.warning_min,
        regenerate_min=s.regenerate_min,
        high_risk_answer_min=s.high_risk_answer_min,
        high_risk_warning_min=s.high_risk_warning_min,
        high_risk_regenerate_min=s.high_risk_regenerate_min,
        high_risk_threshold=s.high_risk_threshold,
    )
    return DecisionEngine(thresholds=thresholds)


def _build_reranking_service(
    settings: Settings,
) -> RerankingService | DisabledRerankingService:
    if not settings.reranker.enabled:
        logger.info("Reranker disabled -- using DisabledRerankingService")
        return DisabledRerankingService()

    s = settings.reranker
    logger.info(
        "Reranker enabled -- loading CrossEncoderReranker (model=%s)", s.model_name
    )
    config = RerankingServiceConfig(
        cross_encoder=CrossEncoderConfig(
            model_name=s.model_name,
            max_candidates=s.max_candidates,
            batch_size=s.batch_size,
            device=s.device,
        ),
        pre_sort_by_retrieval=True,
        min_relevance_score=s.min_relevance_score,
    )
    try:
        return RerankingService(config)
    except Exception as exc:
        raise ConfigurationError(
            f"reranker.enabled=True but the cross-encoder could not be loaded: {exc}"
        ) from exc


def _build_generator(settings: Settings) -> tuple[Generator, Any]:
    s = settings.llm
    llm_config = LLMConfig(
        model_id=s.model_name,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        timeout_seconds=s.timeout_seconds,
    )

    if s.provider == "stub":
        # Reachable only in offline_test: Settings rejects provider='stub'
        # in every research mode, so no online path can end up here.
        if settings.execution_mode is not ExecutionMode.OFFLINE_TEST:
            raise ConfigurationError(
                f"{settings.execution_mode.value} must not use the stub LLM"
            )
        logger.info("LLM provider: stub (offline_test)")
        client: Any = MockLLMClient(canned_text=s.stub_response)
    elif s.provider == "gemini":
        logger.info("LLM provider: gemini (model=%s)", s.model_name)
        try:
            from rasvcx.generation.gemini_client import GeminiLLMClient
            client = GeminiLLMClient(
                api_key=s.api_key,
                model_name=s.model_name,
                timeout_seconds=s.timeout_seconds,
                max_retries=s.max_retries,
            )
        except ImportError as exc:
            raise ConfigurationError(
                "gemini provider requires the google-genai SDK "
                "(pip install -e \".[gemini]\"): " + str(exc)
            ) from exc
    else:
        raise ConfigurationError(
            f"Unknown LLM provider: {s.provider!r}"
        )

    from rasvcx.generation.prompt_builder import PromptBuilder
    generator = Generator(
        llm_client=client,
        prompt_builder=PromptBuilder(max_context_chars=settings.pipeline.max_context_chars),
        llm_config=llm_config,
    )
    return generator, client


# ---------------------------------------------------------------------------
# Research transparency: effective routing defaults
# ---------------------------------------------------------------------------


def get_effective_routing_info() -> dict[str, object]:
    """Return the effective M2 routing defaults for research transparency.

    M2 routing parameters (RiskRouterWeights, RiskRouterThresholds) are
    not exposed via factory Settings (deferred for ablation study work).
    This function documents the in-code defaults so the admin endpoint
    can report them accurately.
    """
    w = RiskRouterWeights()
    t = RiskRouterThresholds()
    return {
        "note": (
            "M2 routing uses default RiskRouterWeights and RiskRouterThresholds. "
            "These are not yet configurable via settings (deferred)."
        ),
        "weights": {
            "numeric_content_weight": w.numeric_content_weight,
            "unit_sensitive_weight": w.unit_sensitive_weight,
            "safety_critical_structure_weight": w.safety_critical_structure_weight,
            "query_complexity_weight": w.query_complexity_weight,
            "ambiguity_weight": w.ambiguity_weight,
        },
        "thresholds": {
            "standard_threshold": t.standard_threshold,
            "deep_threshold": t.deep_threshold,
            "shallow_retrieval_retry_budget": t.shallow_retrieval_retry_budget,
            "standard_retrieval_retry_budget": t.standard_retrieval_retry_budget,
            "deep_retrieval_retry_budget": t.deep_retrieval_retry_budget,
            "shallow_nli_call_allowance": t.shallow_nli_call_allowance,
            "standard_nli_call_allowance": t.standard_nli_call_allowance,
            "deep_nli_call_allowance": t.deep_nli_call_allowance,
        },
    }


__all__ = [
    "PipelineRuntime",
    "build_pipeline",
    "config_hash",
    "build_runtime",
    "get_effective_routing_info",
    "seed_version_id",
]
