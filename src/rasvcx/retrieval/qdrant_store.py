"""Qdrant collection lifecycle and payload management for RASVC-X.

This module owns all direct Qdrant client interactions that are NOT part of
query-time dense retrieval (which lives in retrieval/dense.py).

Responsibilities:
  - Collection creation / deletion / existence checks.
  - Chunk payload storage and lookup by chunk_id.
  - Point deletion.
  - Health / connectivity check.

Design constraints:
- No embedding logic here; that belongs in DenseRetriever.
- All public methods raise QdrantUnavailableError on connection failure.
- No global mutable state.
- Payload schema: every point must carry {"chunk_id": str} at minimum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rasvcx.retrieval.dense import QdrantUnavailableError
from rasvcx.schemas.common import ChunkId

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointIdsList,
        VectorParams,
    )
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

_CHUNK_ID_KEY = "chunk_id"


@dataclass(frozen=True, slots=True)
class ChunkPayload:
    """Stored payload for a single Qdrant point."""

    chunk_id: ChunkId
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_qdrant_payload(self) -> dict[str, Any]:
        return {_CHUNK_ID_KEY: self.chunk_id, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_qdrant_payload(cls, payload: dict[str, Any]) -> "ChunkPayload":
        if _CHUNK_ID_KEY not in payload:
            raise ValueError(f"Qdrant payload missing required key '{_CHUNK_ID_KEY}': {payload}")
        return cls(
            chunk_id=ChunkId(payload[_CHUNK_ID_KEY]),
            text=payload.get("text", ""),
            metadata=payload.get("metadata", {}),
        )


class QdrantStore:
    """Qdrant collection lifecycle and payload management."""

    def __init__(
        self,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        vector_size: int,
        *,
        prefer_grpc: bool = False,
        grpc_port: int = 6334,
    ) -> None:
        if not _QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for QdrantStore.")
        if vector_size < 1:
            raise ValueError(f"vector_size must be >= 1, got {vector_size}")
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._client = QdrantClient(
            host=qdrant_host, port=qdrant_port,
            prefer_grpc=prefer_grpc, grpc_port=grpc_port,
        )
        logger.info("QdrantStore initialised: host=%s port=%d collection=%s",
                    qdrant_host, qdrant_port, collection_name)

    @classmethod
    def build(
        cls,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        vector_size: int,
        *,
        prefer_grpc: bool = False,
        grpc_port: int = 6334,
    ) -> "QdrantStore":
        instance = cls(
            qdrant_host=qdrant_host, qdrant_port=qdrant_port,
            collection_name=collection_name, vector_size=vector_size,
            prefer_grpc=prefer_grpc, grpc_port=grpc_port,
        )
        instance.health_check()
        return instance

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            raise QdrantUnavailableError(f"Qdrant health check failed: {exc}") from exc

    def collection_exists(self) -> bool:
        try:
            names = [c.name for c in self._client.get_collections().collections]
            return self._collection_name in names
        except Exception as exc:
            raise QdrantUnavailableError(f"Failed to list Qdrant collections: {exc}") from exc

    def ensure_collection(self) -> None:
        try:
            if not self.collection_exists():
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection '%s' (dim=%d, cosine)",
                            self._collection_name, self._vector_size)
            else:
                logger.debug("Qdrant collection '%s' already exists; skipping creation",
                             self._collection_name)
        except QdrantUnavailableError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Failed to ensure collection '{self._collection_name}': {exc}") from exc

    def delete_collection(self) -> None:
        try:
            if self.collection_exists():
                self._client.delete_collection(self._collection_name)
                logger.info("Deleted Qdrant collection '%s'", self._collection_name)
        except QdrantUnavailableError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Failed to delete collection '{self._collection_name}': {exc}") from exc

    def count(self) -> int:
        try:
            if not self.collection_exists():
                return 0
            return self._client.count(collection_name=self._collection_name).count
        except QdrantUnavailableError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Failed to count points in '{self._collection_name}': {exc}") from exc

    def get_payload(self, chunk_id: ChunkId) -> ChunkPayload | None:
        try:
            results, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(must=[FieldCondition(key=_CHUNK_ID_KEY, match=MatchValue(value=chunk_id))]),
                limit=1, with_payload=True, with_vectors=False,
            )
        except Exception as exc:
            raise QdrantUnavailableError(f"Qdrant scroll failed for chunk_id '{chunk_id}': {exc}") from exc
        if not results:
            return None
        try:
            return ChunkPayload.from_qdrant_payload(results[0].payload or {})
        except ValueError as exc:
            logger.warning("Malformed payload for chunk_id '%s': %s", chunk_id, exc)
            return None

    def get_payloads(self, chunk_ids: list[ChunkId]) -> dict[ChunkId, ChunkPayload]:
        if not chunk_ids:
            return {}
        unique_ids = list(dict.fromkeys(chunk_ids))
        try:
            hits, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(should=[
                    FieldCondition(key=_CHUNK_ID_KEY, match=MatchValue(value=cid))
                    for cid in unique_ids
                ]),
                limit=len(unique_ids), with_payload=True, with_vectors=False,
            )
        except Exception as exc:
            raise QdrantUnavailableError(f"Qdrant batch scroll failed: {exc}") from exc
        result: dict[ChunkId, ChunkPayload] = {}
        for hit in hits:
            try:
                cp = ChunkPayload.from_qdrant_payload(hit.payload or {})
                result[cp.chunk_id] = cp
            except ValueError as exc:
                logger.warning("Malformed payload in batch scroll: %s", exc)
        return result

    def delete_points_by_chunk_ids(self, chunk_ids: list[ChunkId]) -> int:
        if not chunk_ids:
            return 0
        unique_ids = list(dict.fromkeys(chunk_ids))
        try:
            hits, _ = self._client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(should=[
                    FieldCondition(key=_CHUNK_ID_KEY, match=MatchValue(value=cid))
                    for cid in unique_ids
                ]),
                limit=len(unique_ids), with_payload=False, with_vectors=False,
            )
        except Exception as exc:
            raise QdrantUnavailableError(f"Qdrant scroll (pre-delete) failed: {exc}") from exc
        if not hits:
            return 0
        point_ids = [hit.id for hit in hits]
        try:
            self._client.delete(
                collection_name=self._collection_name,
                points_selector=PointIdsList(points=point_ids),
            )
        except Exception as exc:
            raise QdrantUnavailableError(f"Qdrant delete failed: {exc}") from exc
        logger.info("Deleted %d points from '%s'", len(point_ids), self._collection_name)
        return len(point_ids)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def vector_size(self) -> int:
        return self._vector_size