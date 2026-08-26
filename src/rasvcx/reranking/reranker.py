# src/rasvcx/reranking/reranker.py
"""RerankingService: bundle-level orchestration around CrossEncoderReranker.

Responsibilities (see docs/ARCHITECTURE.md Module 4):
  1. Pull EvidenceItems out of an EvidenceBundle.
  2. Optionally pre-sort by retrieval_score so the highest-retrieval-score
     candidates reach the (bounded) cross-encoder first.
  3. Enforce the candidate cap (delegated to CrossEncoderReranker).
  4. Attach rerank scores via immutable replacement -- EvidenceItem is
     frozen, so scored items are new objects, never in-place mutation.
  5. Leave items beyond the cap in the bundle with rerank_score=None.
  6. Reorder bundle.evidence_items: scored items first (descending
     rerank_score), then unscored items, preserving their relative order.
  7. Degrade gracefully on model failure: no scores are attached, but the
     request does not crash and elapsed time is still recorded.

No business/domain decision logic lives here -- this module only scores,
orders, and enriches evidence.
"""

from __future__ import annotations

import dataclasses
import logging
import time

from rasvcx.reranking.cross_encoder import CrossEncoderConfig, CrossEncoderReranker
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class RerankingServiceConfig:
    """Configuration for the bundle-level reranking service.

    Attributes:
        cross_encoder:         Configuration forwarded to CrossEncoderReranker.
        pre_sort_by_retrieval: If True, candidates are sorted descending by
                                retrieval_score before the candidate cap is
                                applied, so the most-relevant-by-retrieval
                                items are the ones sent to the (expensive)
                                cross-encoder. If False, bundle iteration
                                order is used as-is.
    """

    cross_encoder: CrossEncoderConfig = dataclasses.field(
        default_factory=CrossEncoderConfig
    )
    pre_sort_by_retrieval: bool = True


class RerankingService:
    """Reranks the evidence items in an EvidenceBundle, in place.

    The underlying cross-encoder model is loaded once at construction time
    (via CrossEncoderReranker.__init__) and reused across calls to
    ``rerank_bundle``.
    """

    def __init__(self, config: RerankingServiceConfig | None = None) -> None:
        self._config = config or RerankingServiceConfig()
        self._reranker = CrossEncoderReranker(self._config.cross_encoder)

    @property
    def config(self) -> RerankingServiceConfig:
        return self._config

    def rerank_bundle(self, bundle: EvidenceBundle, query: str) -> None:
        """Score and reorder ``bundle.evidence_items`` for ``query``.

        Mutates the bundle's ``evidence_items`` mapping in place, replacing
        entries with new (frozen) EvidenceItem objects carrying an attached
        ``rerank_score``. Items beyond the configured candidate cap, or all
        items if scoring fails or the query is empty, are left with
        ``rerank_score=None``.

        Always records elapsed time under the ``"reranking"`` pipeline
        stage, including on empty-bundle, empty-query, and model-failure
        paths.
        """
        start = time.perf_counter()
        try:
            self._rerank_bundle_inner(bundle, query)
        finally:
            elapsed = time.perf_counter() - start
            bundle.record_stage_elapsed("reranking", elapsed)

    def _rerank_bundle_inner(self, bundle: EvidenceBundle, query: str) -> None:
        items = list(bundle.evidence_items.values())
        if not items:
            return

        if not query or not query.strip():
            logger.warning(
                "rerank_bundle() called with empty/whitespace query; "
                "leaving all rerank_score fields unset."
            )
            return

        if self._config.pre_sort_by_retrieval:
            items = sorted(items, key=lambda it: it.retrieval_score, reverse=True)

        candidates = [(it.chunk_id, it.text) for it in items]

        try:
            results = self._reranker.rerank(query, candidates)
        except Exception:  # noqa: BLE001 -- graceful degradation, never crash
            logger.exception(
                "Cross-encoder reranking failed; leaving rerank_score unset "
                "for all %d evidence item(s).",
                len(items),
            )
            return

        score_by_chunk_id = {r.chunk_id: r.rerank_score for r in results}

        scored_items: list[EvidenceItem] = []
        unscored_items: list[EvidenceItem] = []
        for item in items:
            score = score_by_chunk_id.get(item.chunk_id)
            if score is None:
                unscored_items.append(item)
                continue
            scored_items.append(dataclasses.replace(item, rerank_score=score))

        scored_items.sort(key=lambda it: (-it.rerank_score, it.chunk_id))  # type: ignore[operator]

        ordered = scored_items + unscored_items
        bundle.evidence_items = {it.item_id: it for it in ordered}