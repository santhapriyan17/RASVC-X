"""Dense vector retrieval for RASVC-X.

Encodes queries with a sentence-transformer model and retrieves nearest
neighbours from a Qdrant collection.  The encoder is initialised once and
reused across all requests -- never loaded per-request.

Design constraints:
- Encoder initialisation is explicit (call DenseRetriever.build()).
- No global mutable state; each DenseRetriever is a self-contained object.
- Qdrant client is created once and reused.
- Scores returned are Qdrant cosine similarity scores in [-1, 1]; they are
  NOT probabilities.  Do not interpret as confidence values.
- Qdrant unavailability raises QdrantUnavailableError, which the pipeline
  must handle (fall back to BM25-only where appropriate).

CORRECTION (continuation-session review finding #4):
Qdrant point IDs are now derived from a deterministic digest of the
chunk_id rather than Python's built-in hash(). Python's hash() for str
objects is randomized per-process (PYTHONHASHSEED) unless explicitly
disabled, so two different processes (e.g. an ingestion job and the API
server, or two API worker processes) could previously compute two
different point IDs for the same chunk_id, silently duplicating or
orphaning vectors in the collection across restarts/processes. The
replacement uses SHA-256 over the UTF-8 encoded chunk_id, truncated to
63 bits so it remains a valid Qdrant unsigned point ID, and is stable
across processes, restarts, and machines.

If src/rasvcx/utils/hashing.py is implemented later as the project's
shared hashing utility, this local helper should be replaced by a call
into it rather than duplicated -- it is kept local here only because
utils/hashing.py is still an empty placeholder in the current
repository state.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

from rasvcx.schemas.common import ChunkId

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


class QdrantUnavailableError(RuntimeError):
    """Raised when Qdrant cannot be reached or returns an unexpected error."""


class EncoderNotInitializedError(RuntimeError):
    """Raised when query() is called before the encoder is ready."""


def deterministic_point_id(chunk_id: str) -> int:
    """Derive a stable, process-independent Qdrant point ID from a chunk_id.

    Uses SHA-256 rather than Python's built-in hash() because str hashing
    is randomized per-process by default (PYTHONHASHSEED), which would
    make point IDs non-reproducible across ingestion runs, API worker
    processes, or restarts. The digest is truncated to 63 bits so the
    result always fits Qdrant's unsigned-integer point ID requirement.

    This function is pure and deterministic: the same chunk_id always
    produces the same point ID, regardless of process, machine, or time.
    """
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    # Take the first 8 bytes as a big-endian unsigned integer, then mask
    # to 63 bits (not 64) to stay safely within Qdrant's accepted range
    # and avoid any sign-bit ambiguity across client/server integer types.
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value & ((1 << 63) - 1)


@dataclass(frozen=True, slots=True)
class DenseResult:
    chunk_id: ChunkId
    score: float


class DenseRetriever:
    def __init__(
        self,
        model_name: str,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        *,
        prefer_grpc: bool = False,
        grpc_port: int = 6334,
    ) -> None:
        if not _ST_AVAILABLE:
            raise ImportError("sentence-transformers is required for DenseRetriever.")
        if not _QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for DenseRetriever.")

        self._model_name = model_name
        self._collection_name = collection_name

        logger.info("Loading sentence-transformer model: %s", model_name)
        self._encoder = SentenceTransformer(model_name)
        self._vector_size: int = self._encoder.get_sentence_embedding_dimension()
        logger.info("Encoder ready: model=%s vector_size=%d", model_name, self._vector_size)

        logger.info("Connecting to Qdrant: host=%s port=%d grpc=%s", qdrant_host, qdrant_port, prefer_grpc)
        self._client = QdrantClient(
            host=qdrant_host,
            port=qdrant_port,
            prefer_grpc=prefer_grpc,
            grpc_port=grpc_port,
        )

    @classmethod
    def build(
        cls,
        model_name: str,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        *,
        prefer_grpc: bool = False,
        grpc_port: int = 6334,
    ) -> "DenseRetriever":
        instance = cls(
            model_name=model_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name,
            prefer_grpc=prefer_grpc,
            grpc_port=grpc_port,
        )
        instance._verify_collection()
        return instance

    def _verify_collection(self) -> None:
        try:
            collections = [c.name for c in self._client.get_collections().collections]
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Cannot reach Qdrant at collection '{self._collection_name}': {exc}"
            ) from exc
        if self._collection_name not in collections:
            logger.warning("Qdrant collection '%s' does not exist yet.", self._collection_name)

    def ensure_collection(self) -> None:
        try:
            existing = [c.name for c in self._client.get_collections().collections]
            if self._collection_name not in existing:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection '%s' (dim=%d, cosine)", self._collection_name, self._vector_size)
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Failed to ensure Qdrant collection '{self._collection_name}': {exc}"
            ) from exc

    def upsert(self, documents: list[tuple[ChunkId, str]], batch_size: int = 64) -> None:
        if not documents:
            raise ValueError("upsert() requires at least one document")
        total = len(documents)
        upserted = 0
        for batch_start in range(0, total, batch_size):
            batch = documents[batch_start: batch_start + batch_size]
            chunk_ids = [cid for cid, _ in batch]
            texts = [text for _, text in batch]
            vectors = self._encoder.encode(
                texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True
            ).tolist()
            points = [
                PointStruct(
                    id=deterministic_point_id(str(cid)),
                    vector=vec,
                    payload={"chunk_id": cid},
                )
                for cid, vec in zip(chunk_ids, vectors)
            ]
            try:
                self._client.upsert(collection_name=self._collection_name, points=points)
            except Exception as exc:
                raise QdrantUnavailableError(f"Qdrant upsert failed (batch at {batch_start}): {exc}") from exc
            upserted += len(batch)
            logger.debug("Upserted %d / %d documents", upserted, total)
        logger.info("Dense upsert complete: %d documents", total)

    def query(self, query_text: str, top_k: int) -> list[DenseResult]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not query_text.strip():
            logger.debug("Dense query is empty; returning empty results")
            return []
        vector: list[float] = self._encoder.encode(
            query_text, show_progress_bar=False, normalize_embeddings=True
        ).tolist()
        try:
            hits: list[ScoredPoint] = self._client.search(
                collection_name=self._collection_name,
                query_vector=vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Qdrant search failed for collection '{self._collection_name}': {exc}"
            ) from exc
        results: list[DenseResult] = []
        for hit in hits:
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id is None:
                logger.warning("Qdrant hit missing 'chunk_id' in payload (id=%s); skipping", hit.id)
                continue
            results.append(DenseResult(chunk_id=ChunkId(chunk_id), score=hit.score))
        return results

    @property
    def vector_size(self) -> int:
        return self._vector_size

    @property
    def collection_name(self) -> str:
        return self._collection_name