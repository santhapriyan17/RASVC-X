"""RASVC-X retrieval package."""

from rasvcx.retrieval.bm25 import BM25Index, BM25Result
from rasvcx.retrieval.chunking import Chunk, ChunkConfig, ChunkStrategy, chunk_document
from rasvcx.retrieval.dense import DenseResult, DenseRetriever, QdrantUnavailableError
from rasvcx.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from rasvcx.retrieval.metadata_filter import (
    MetadataFilterCriteria, ProvenanceSnapshot, filter_by_metadata,
)
from rasvcx.retrieval.qdrant_store import ChunkPayload, QdrantStore
from rasvcx.retrieval.targeted_retrieval import (
    TargetedRetrievalConfig, TargetedRetrievalResult, run_targeted_retrieval,
)

__all__ = [
    "BM25Index", "BM25Result",
    "Chunk", "ChunkConfig", "ChunkStrategy", "chunk_document",
    "DenseResult", "DenseRetriever", "QdrantUnavailableError",
    "FusedResult", "reciprocal_rank_fusion",
    "MetadataFilterCriteria", "ProvenanceSnapshot", "filter_by_metadata",
    "ChunkPayload", "QdrantStore",
    "TargetedRetrievalConfig", "TargetedRetrievalResult", "run_targeted_retrieval",
]