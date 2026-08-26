"""Targeted retrieval for RASVC-X — conditional second-pass retrieval.

Invoked ONLY when the evidence sufficiency gate fails after initial hybrid
retrieval.

BOUNDED LOOP INVARIANT:
    The orchestrator owns the global corrective attempt budget and must not
    call this function once the budget is exhausted.  This module executes
    retrieval mechanics only; it does not own or decrement the budget.

GLOBAL CORRECTIVE ATTEMPT BUDGET:
    Targeted retrieval, REPAIR, and REGENERATE share one request-scoped
    budget (default 3).  This module consumes the retrieval slot.

Design constraints:
- No business logic: retrieval mechanics only.
- Returns merged FusedResult list; reranking is the caller's responsibility.
- Query augmentation is keyword extraction — not a generative step.
- If both BM25 and dense return empty, original results returned unchanged.

CORRECTION (continuation-session review finding #5):
Merging previously used `reciprocal_rank_fusion()` (a 2-list BM25+dense
function) by relabeling each existing FusedResult's `rrf_score` as if it
were a fresh `BM25Result.score`, then re-running fusion against the new
targeted BM25/dense lists. This was semantically wrong in two ways:
  1. It mislabeled prior RRF scores as BM25 scores, corrupting the
     `bm25_score` field of the returned FusedResult for any chunk that
     originally came from dense-only retrieval (or from both).
  2. It discarded each existing chunk's original bm25_score/dense_score
     provenance and its true rank position within *its own* originating
     list(s), collapsing three logically distinct ranked lists (existing
     fused results, new BM25 hits, new dense hits) into two.

The fix performs a genuine three-list rank fusion locally: existing
results retain their own established rank (by their prior rrf_score,
re-sorted defensively), and each of the two new lists (targeted BM25,
targeted dense) contributes its own RRF term. Original bm25_score /
dense_score provenance is preserved and only overwritten by a newer,
more specific score when the same chunk_id reappears in a fresh
targeted-retrieval list. `retrieval/fusion.py`'s two-list function and
formula are unchanged; this module does not modify or reinterpret them,
it simply does not misuse them for a three-list scenario.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from rasvcx.retrieval.bm25 import BM25Index, BM25Result
from rasvcx.retrieval.dense import DenseRetriever, DenseResult, QdrantUnavailableError
from rasvcx.retrieval.fusion import FusedResult
from rasvcx.schemas.common import ChunkId

logger = logging.getLogger(__name__)

_AUGMENT_TERM_RE = re.compile(
    r"\b(?:\d+(?:\.\d+)?(?:\s*(?:mg|mcg|g|kg|ml|l|percent|%))?|[A-Z][a-z]{2,})\b"
)
_MAX_AUGMENT_TERMS = 5


@dataclass(frozen=True, slots=True)
class TargetedRetrievalConfig:
    """Configuration for a targeted retrieval pass.

    Attributes:
        bm25_top_k:    BM25 results to retrieve in targeted pass.
        dense_top_k:   Dense results to retrieve in targeted pass.
        rrf_k:         RRF smoothing constant used for the merge.
        augment_query: Augment query with extracted keywords if True.
    """

    bm25_top_k: int = 10
    dense_top_k: int = 10
    rrf_k: int = 60
    augment_query: bool = True

    def __post_init__(self) -> None:
        if self.bm25_top_k < 1:
            raise ValueError(f"bm25_top_k must be >= 1, got {self.bm25_top_k}")
        if self.dense_top_k < 1:
            raise ValueError(f"dense_top_k must be >= 1, got {self.dense_top_k}")
        if self.rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {self.rrf_k}")


@dataclass(frozen=True, slots=True)
class TargetedRetrievalResult:
    """Output of a targeted retrieval pass.

    Attributes:
        merged_results:       Fused results (existing + targeted) by RRF score.
        augmented_query:      Query string used for the targeted pass.
        new_chunk_ids:        Chunk IDs new to this pass (not in existing).
        targeted_bm25_count:  BM25 result count from targeted pass.
        targeted_dense_count: Dense result count from targeted pass.
    """

    merged_results: list[FusedResult]
    augmented_query: str
    new_chunk_ids: frozenset[ChunkId]
    targeted_bm25_count: int
    targeted_dense_count: int


def _augment_query(query_text: str) -> str:
    """Extract salient terms and append to query for targeted retrieval."""
    terms = _AUGMENT_TERM_RE.findall(query_text)
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        t_stripped = t.strip()
        if t_stripped and t_stripped not in seen:
            seen.add(t_stripped)
            unique.append(t_stripped)
        if len(unique) >= _MAX_AUGMENT_TERMS:
            break
    if not unique:
        return query_text
    logger.debug("Query augmented with %d terms: %s", len(unique), unique)
    return f"{query_text} {' '.join(unique)}"


def _merge_three_ranked_lists(
    existing_results: list[FusedResult],
    targeted_bm25: list[BM25Result],
    targeted_dense: list[DenseResult],
    *,
    top_k: int,
    rrf_k: int,
) -> list[FusedResult]:
    """Fuse three ranked sources into one list using the standard RRF formula.

    Each source contributes independently, by its own rank position within
    itself: RRF(d) = sum_source [ 1 / (rrf_k + rank_source(d)) ]. This
    generalizes retrieval/fusion.py's two-list formula to three lists
    without reinterpreting any list's score as belonging to a different
    source (the bug being fixed here).

    `existing_results` is re-sorted defensively by its own rrf_score
    before rank positions are assigned, since a caller-provided ordering
    is not otherwise guaranteed at this call site.
    """
    rrf_scores: dict[ChunkId, float] = {}
    bm25_score_map: dict[ChunkId, float] = {}
    dense_score_map: dict[ChunkId, float] = {}

    ordered_existing = sorted(existing_results, key=lambda r: r.rrf_score, reverse=True)
    for rank_0, r in enumerate(ordered_existing):
        rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + rank_0 + 1)
        if r.bm25_score is not None:
            bm25_score_map[r.chunk_id] = r.bm25_score
        if r.dense_score is not None:
            dense_score_map[r.chunk_id] = r.dense_score

    for rank_0, r in enumerate(targeted_bm25):
        rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + rank_0 + 1)
        bm25_score_map[r.chunk_id] = r.score  # freshest BM25 score wins

    for rank_0, r in enumerate(targeted_dense):
        rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + rank_0 + 1)
        dense_score_map[r.chunk_id] = r.score  # freshest dense score wins

    fused = [
        FusedResult(
            chunk_id=cid,
            rrf_score=score,
            bm25_score=bm25_score_map.get(cid),
            dense_score=dense_score_map.get(cid),
        )
        for cid, score in rrf_scores.items()
    ]
    fused.sort(key=lambda r: r.rrf_score, reverse=True)
    return fused[: min(top_k, len(fused))]


def run_targeted_retrieval(
    query_text: str,
    existing_results: list[FusedResult],
    bm25_index: BM25Index,
    dense_retriever: DenseRetriever | None,
    config: TargetedRetrievalConfig,
) -> TargetedRetrievalResult:
    """Execute a targeted retrieval pass and merge with existing results.

    Args:
        query_text:       Normalized query string.
        existing_results: FusedResult list from initial hybrid retrieval.
        bm25_index:       Initialised BM25Index.
        dense_retriever:  Initialised DenseRetriever, or None for BM25-only.
        config:           Targeted retrieval configuration.

    Returns:
        TargetedRetrievalResult with merged results and metadata.
    """
    augmented = _augment_query(query_text) if config.augment_query else query_text
    if augmented != query_text:
        logger.info("Targeted retrieval: using augmented query")

    targeted_bm25: list[BM25Result] = bm25_index.query(augmented, top_k=config.bm25_top_k)
    logger.debug("Targeted BM25: %d results", len(targeted_bm25))

    targeted_dense: list[DenseResult] = []
    if dense_retriever is not None:
        try:
            targeted_dense = dense_retriever.query(augmented, top_k=config.dense_top_k)
            logger.debug("Targeted dense: %d results", len(targeted_dense))
        except QdrantUnavailableError as exc:
            logger.warning(
                "Targeted dense retrieval failed (Qdrant unavailable): %s — "
                "continuing with BM25-only targeted results", exc,
            )

    if not targeted_bm25 and not targeted_dense:
        logger.warning("Targeted retrieval produced no results; returning existing unchanged")
        return TargetedRetrievalResult(
            merged_results=existing_results,
            augmented_query=augmented,
            new_chunk_ids=frozenset(),
            targeted_bm25_count=0,
            targeted_dense_count=0,
        )

    merged = _merge_three_ranked_lists(
        existing_results=existing_results,
        targeted_bm25=targeted_bm25,
        targeted_dense=targeted_dense,
        top_k=max(
            len(existing_results) + config.bm25_top_k + config.dense_top_k,
            config.bm25_top_k,
        ),
        rrf_k=config.rrf_k,
    )

    prior_ids: frozenset[ChunkId] = frozenset(r.chunk_id for r in existing_results)
    targeted_ids: frozenset[ChunkId] = (
        frozenset(r.chunk_id for r in targeted_bm25)
        | frozenset(r.chunk_id for r in targeted_dense)
    )
    new_ids = targeted_ids - prior_ids

    logger.info(
        "Targeted retrieval complete: existing=%d targeted_bm25=%d "
        "targeted_dense=%d new=%d merged=%d",
        len(existing_results), len(targeted_bm25),
        len(targeted_dense), len(new_ids), len(merged),
    )

    return TargetedRetrievalResult(
        merged_results=merged,
        augmented_query=augmented,
        new_chunk_ids=new_ids,
        targeted_bm25_count=len(targeted_bm25),
        targeted_dense_count=len(targeted_dense),
    )