"""Production retrieval bridge for RASVC-X (Module 13).

Wires the existing M3 retrieval components into the M12 RetrievalFn and
TargetedRetrievalFn callables, and provides DisabledRerankingService for
offline_test mode.

  make_retrieval_fn()          Returns a RetrievalFn that populates an
                               EvidenceBundle from BM25 + dense + RRF.
  make_targeted_retrieval_fn() Returns a TargetedRetrievalFn that calls
                               run_targeted_retrieval() and adds only new
                               chunks to the bundle.
  DisabledRerankingService     Replaces RerankingService when the reranker
                               is disabled (offline_test only).

Bridge call sequence (RetrievalFn):
  1. bm25_index.query(normalized_text, top_k)     -> list[BM25Result]
  2. dense_retriever.query(normalized_text, top_k) -> list[DenseResult]
     (when a dense retriever is configured)
  3. reciprocal_rank_fusion(bm25, dense, top_k=rrf_top_k) -> list[FusedResult]
  4. For each FusedResult, in fused rank order:
       text, provenance = corpus_store.lookup(chunk_id)
       source           = corpus_store.lookup_source(chunk_id)
       item_id          = "E<n>"   (n = 1-based rank within the bundle)
       bundle.add_evidence_item(EvidenceItem(...))

No silent degradation:
  When a dense retriever is configured (hybrid mode) and it fails, the
  failure propagates as RetrievalBackendError.  The orchestrator turns it
  into an explicit pipeline error (ABSTAIN) -- a hybrid request is never
  silently answered from BM25 alone.

Evidence ids:
  item_id is the short citation label "E1", "E2", ... that the generation
  prompt asks the model to cite and that the M9 citation checker resolves.
  chunk_id keeps the knowledge-base identity.  Chunk ids are not usable as
  citation labels: ingested ids contain "::" and can exceed the citation
  token grammar.

Observability:
  bundle.retrieval_trace records what actually executed (per-retriever hit
  counts and timings, fusion output size).  It is the evidence that BM25,
  Qdrant and RRF each ran for this request.

Score semantics:
  retrieval_score = FusedResult.rrf_score  (not a probability)
  rerank_score    = None                   (set later by M4 RerankingService)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from rasvcx.retrieval.bm25 import BM25Index
from rasvcx.retrieval.corpus import CorpusStore
from rasvcx.retrieval.fusion import reciprocal_rank_fusion
from rasvcx.retrieval.targeted_retrieval import (
    TargetedRetrievalConfig,
    run_targeted_retrieval,
)
from rasvcx.schemas.common import ChunkId, EvidenceItemId
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem
from rasvcx.schemas.query import QueryRequest, RiskProfile

logger = logging.getLogger(__name__)

# Type alias matching M12 pipeline/orchestrator.py
RetrievalFn = Callable[[QueryRequest, RiskProfile, EvidenceBundle], None]
TargetedRetrievalFn = Callable[[QueryRequest, RiskProfile, EvidenceBundle], None]


class RetrievalBackendError(RuntimeError):
    """A configured retrieval backend failed during a request."""


def _next_item_id(bundle: EvidenceBundle) -> EvidenceItemId:
    n = len(bundle.evidence_items) + 1
    while EvidenceItemId(f"E{n}") in bundle.evidence_items:
        n += 1
    return EvidenceItemId(f"E{n}")


def _add_chunk(
    bundle: EvidenceBundle,
    corpus_store: CorpusStore,
    chunk_id: ChunkId,
    rrf_score: float,
) -> bool:
    entry = corpus_store.lookup(chunk_id)
    if entry is None:
        logger.warning("chunk_id %r not found in CorpusStore -- skipping", chunk_id)
        return False
    text, provenance = entry
    bundle.add_evidence_item(
        EvidenceItem(
            item_id=_next_item_id(bundle),
            chunk_id=chunk_id,
            text=text,
            retrieval_score=rrf_score,
            provenance=provenance,
            rerank_score=None,  # set later by M4 RerankingService
            source=corpus_store.lookup_source(chunk_id),
        )
    )
    return True


# ---------------------------------------------------------------------------
# Primary retrieval bridge
# ---------------------------------------------------------------------------


def make_retrieval_fn(
    bm25_index: BM25Index,
    dense_retriever: object | None,
    corpus_store: CorpusStore,
    bm25_top_k: int = 20,
    dense_top_k: int = 20,
    rrf_top_k: int = 10,
    rrf_k: int = 60,
    kb_version_id: str | None = None,
) -> RetrievalFn:
    """Return a RetrievalFn that populates an EvidenceBundle.

    Args:
        bm25_index:      Loaded BM25Index (required).
        dense_retriever: Loaded DenseRetriever, or None for BM25-only.
        corpus_store:    Loaded CorpusStore (ChunkId -> text + Provenance).
        bm25_top_k:      BM25 candidates to retrieve.
        dense_top_k:     Dense candidates to retrieve (ignored if no retriever).
        rrf_top_k:       Maximum evidence items after RRF.
        rrf_k:           RRF smoothing constant.
        kb_version_id:   Knowledge-base version all three artifacts belong to
                         (recorded in the retrieval trace).

    Returns:
        A callable matching the M12 RetrievalFn signature.
    """

    def retrieval_fn(
        query: QueryRequest,
        risk_profile: RiskProfile,
        bundle: EvidenceBundle,
    ) -> None:
        query_text = query.normalized_text
        trace: dict[str, object] = {
            "mode": "hybrid" if dense_retriever is not None else "bm25_only",
            "kb_version_id": kb_version_id,
        }

        # Step 1: BM25 retrieval
        t0 = time.perf_counter()
        bm25_results = bm25_index.query(query_text, top_k=bm25_top_k)
        trace["bm25_hits"] = len(bm25_results)
        trace["bm25_seconds"] = time.perf_counter() - t0

        # Step 2: Dense retrieval (hybrid mode). Failure is NOT absorbed.
        dense_results = []
        if dense_retriever is not None:
            t0 = time.perf_counter()
            try:
                dense_results = dense_retriever.query(query_text, top_k=dense_top_k)
            except Exception as exc:
                trace["dense_error"] = type(exc).__name__
                bundle.retrieval_trace = trace
                bundle.record_retrieval_call()
                raise RetrievalBackendError(
                    f"dense retrieval failed ({type(exc).__name__}): {exc}"
                ) from exc
            trace["dense_hits"] = len(dense_results)
            trace["dense_seconds"] = time.perf_counter() - t0
            trace["dense_collection"] = getattr(dense_retriever, "collection_name", None)

        # Step 3: RRF fusion
        t0 = time.perf_counter()
        fused = (
            reciprocal_rank_fusion(
                bm25_results=bm25_results,
                dense_results=dense_results,
                top_k=rrf_top_k,
                rrf_k=rrf_k,
            )
            if (bm25_results or dense_results)
            else []
        )
        trace["rrf_fused"] = len(fused)
        trace["rrf_seconds"] = time.perf_counter() - t0
        trace["rrf_from_both"] = sum(
            1 for fr in fused if fr.bm25_score is not None and fr.dense_score is not None
        )

        # Step 4: Resolve chunk_id -> EvidenceItem via CorpusStore
        added = sum(
            1 for fr in fused if _add_chunk(bundle, corpus_store, fr.chunk_id, fr.rrf_score)
        )
        trace["evidence_items"] = added
        bundle.retrieval_trace = trace
        # The retrieval callable owns the authoritative call counter.
        bundle.record_retrieval_call()

        logger.debug(
            "Retrieval complete: bm25=%d dense=%d fused=%d added=%d (query_id=%s)",
            len(bm25_results), len(dense_results), len(fused), added, query.query_id,
        )

    return retrieval_fn


# ---------------------------------------------------------------------------
# Targeted retrieval bridge
# ---------------------------------------------------------------------------


def make_targeted_retrieval_fn(
    bm25_index: BM25Index,
    dense_retriever: object | None,
    corpus_store: CorpusStore,
    targeted_bm25_top_k: int = 10,
    targeted_dense_top_k: int = 10,
    targeted_rrf_k: int = 60,
) -> TargetedRetrievalFn:
    """Return a TargetedRetrievalFn that adds supplementary evidence items.

    Called by M12 when M5 recommends targeted retrieval (INSUFFICIENT or
    CONSERVATIVE verdict with recommend_targeted_retrieval=True).

    Only chunks whose chunk_id appears in
    TargetedRetrievalResult.new_chunk_ids are added to the bundle.
    Items already present from initial retrieval are skipped.
    """

    targeted_config = TargetedRetrievalConfig(
        bm25_top_k=targeted_bm25_top_k,
        dense_top_k=targeted_dense_top_k,
        rrf_k=targeted_rrf_k,
        augment_query=True,
    )

    def targeted_retrieval_fn(
        query: QueryRequest,
        risk_profile: RiskProfile,
        bundle: EvidenceBundle,
    ) -> None:
        from rasvcx.retrieval.fusion import FusedResult as FR
        existing: list[FR] = [
            FR(
                chunk_id=item.chunk_id,
                rrf_score=item.retrieval_score,
                bm25_score=None,
                dense_score=None,
            )
            for item in bundle.evidence_items.values()
        ]

        t0 = time.perf_counter()
        result = run_targeted_retrieval(
            query_text=query.normalized_text,
            existing_results=existing,
            bm25_index=bm25_index,
            dense_retriever=dense_retriever,  # type: ignore[arg-type]
            config=targeted_config,
        )

        present = {item.chunk_id for item in bundle.evidence_items.values()}
        scores = {fr.chunk_id: fr.rrf_score for fr in result.merged_results}
        added = 0
        for chunk_id in result.new_chunk_ids:
            if chunk_id in present:
                continue
            if _add_chunk(bundle, corpus_store, chunk_id, scores.get(chunk_id, 0.0)):
                added += 1

        trace = dict(bundle.retrieval_trace)
        trace["targeted_added"] = added
        trace["targeted_bm25_hits"] = result.targeted_bm25_count
        trace["targeted_dense_hits"] = result.targeted_dense_count
        trace["targeted_seconds"] = time.perf_counter() - t0
        if result.dense_error:
            trace["targeted_dense_error"] = result.dense_error
        bundle.retrieval_trace = trace
        bundle.record_retrieval_call()
        bundle.record_targeted_retrieval_used()

        logger.info(
            "Targeted retrieval: %d new items added (query_id=%s, bm25=%d dense=%d)",
            added, query.query_id,
            result.targeted_bm25_count, result.targeted_dense_count,
        )

    return targeted_retrieval_fn


# ---------------------------------------------------------------------------
# DisabledRerankingService -- offline_test mode
# ---------------------------------------------------------------------------


class DisabledRerankingService:
    """Reranking service stub for offline_test mode.

    Leaves all EvidenceItem.rerank_score values as None.
    Records elapsed=0.0 under the 'reranking' pipeline stage so that
    bundle.metadata.elapsed_per_stage is consistent with real-mode output.

    This is NOT a pass-through that sets rerank_score=retrieval_score.
    The M5 SufficiencyGateConfig must have min_scored_items=0 when this
    service is used (configured by factory.py in offline_test mode).
    """

    #: Read by the orchestrator so the execution trace reports reranking as
    #: skipped rather than executed.
    enabled = False

    def rerank_bundle(self, bundle: EvidenceBundle, query: str) -> None:
        """No-op: records stage timing only. rerank_score remains None."""
        bundle.record_stage_elapsed("reranking", 0.0)


__all__ = [
    "RetrievalBackendError",
    "RetrievalFn",
    "TargetedRetrievalFn",
    "make_retrieval_fn",
    "make_targeted_retrieval_fn",
    "DisabledRerankingService",
]