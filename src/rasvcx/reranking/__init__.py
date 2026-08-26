"""Module 4 -- Reranking.

Public API:
    CrossEncoderConfig, CrossEncoderReranker, RerankResult
        -- low-level, pure (query, passage) scoring.
    RerankingServiceConfig, RerankingService
        -- bundle-level orchestration (candidate cap, pre-sort, immutable
           score attachment, graceful degradation).
"""

from __future__ import annotations

from rasvcx.reranking.cross_encoder import (
    CrossEncoderConfig,
    CrossEncoderReranker,
    RerankResult,
)
from rasvcx.reranking.reranker import RerankingService, RerankingServiceConfig

__all__ = [
    "CrossEncoderConfig",
    "CrossEncoderReranker",
    "RerankResult",
    "RerankingService",
    "RerankingServiceConfig",
]