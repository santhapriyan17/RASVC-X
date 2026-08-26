"""Reciprocal Rank Fusion (RRF) for RASVC-X.

Merges two ranked result lists (BM25 + dense) into a single unified ranking
using the standard RRF formula:

    RRF(d) = sum_r [ 1 / (k + rank_r(d)) ]

where k is a smoothing constant (default 60, per the original paper) and
rank_r(d) is the 1-based rank of document d in ranked list r.

Design constraints:
- Pure function: no I/O, no side effects, no external dependencies.
- O(N) fusion after O(N log N) sorting of inputs; N = total unique chunks.
- Scores are RRF scores only — not probabilities.
- Documents absent from a list receive no score from it.
- The caller controls top_k; fusion returns at most top_k results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rasvcx.retrieval.bm25 import BM25Result
from rasvcx.retrieval.dense import DenseResult
from rasvcx.schemas.common import ChunkId

logger = logging.getLogger(__name__)

_DEFAULT_RRF_K: int = 60


@dataclass(frozen=True, slots=True)
class FusedResult:
    """A single result after RRF fusion.

    Attributes:
        chunk_id:    ID of the chunk.
        rrf_score:   Aggregate RRF score. Higher is better. Not a probability.
        bm25_score:  Original BM25 score if present, else None.
        dense_score: Original dense similarity score if present, else None.
    """

    chunk_id: ChunkId
    rrf_score: float
    bm25_score: float | None
    dense_score: float | None


def reciprocal_rank_fusion(
    bm25_results: list[BM25Result],
    dense_results: list[DenseResult],
    *,
    top_k: int,
    rrf_k: int = _DEFAULT_RRF_K,
) -> list[FusedResult]:
    """Fuse BM25 and dense ranked lists using Reciprocal Rank Fusion.

    Args:
        bm25_results:  BM25 results, sorted descending by score.
        dense_results: Dense results, sorted descending by score.
        top_k:         Maximum number of fused results to return.
        rrf_k:         RRF smoothing constant. Must be >= 1. Default 60.

    Returns:
        List of FusedResult sorted by descending rrf_score, length <= top_k.
        Empty list if both inputs are empty.

    Raises:
        ValueError: If top_k < 1 or rrf_k < 1.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if rrf_k < 1:
        raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")

    if not bm25_results and not dense_results:
        logger.debug("RRF: both input lists empty; returning empty fusion result")
        return []

    rrf_scores: dict[ChunkId, float] = {}
    bm25_score_map: dict[ChunkId, float] = {}
    dense_score_map: dict[ChunkId, float] = {}

    for rank_0, result in enumerate(bm25_results):
        cid = result.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank_0 + 1)
        bm25_score_map[cid] = result.score

    for rank_0, result in enumerate(dense_results):
        cid = result.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank_0 + 1)
        dense_score_map[cid] = result.score

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

    result_count = min(top_k, len(fused))
    logger.debug(
        "RRF fusion: bm25=%d dense=%d unique=%d returning=%d (k=%d)",
        len(bm25_results), len(dense_results), len(fused), result_count, rrf_k,
    )
    return fused[:result_count]