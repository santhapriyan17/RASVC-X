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
- Qdrant unavailability raises QdrantUnavailableError.  The retrieval
  bridge propagates it: a hybrid request never silently degrades to
  BM25-only.

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
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from rasvcx.schemas.common import ChunkId

logger = logging.getLogger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
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


QDRANT_MODES = ("server", "embedded", "memory")


class DenseBackend:
    """One sentence-transformer encoder + one Qdrant client, shared by every
    knowledge-base version.

    Each published KB version owns its own Qdrant collection; this object
    builds those collections and hands out DenseRetriever views bound to
    one collection.  The encoder is loaded exactly once per process.

    Qdrant modes (all use the same qdrant-client API):
      server    Qdrant service at host:port.
      embedded  qdrant-client local mode persisted under `path` (single
                process only).
      memory    qdrant-client local mode, in-process and non-persistent:
                collections must be rebuilt after every restart.

    Local modes are not thread-safe, so every client call is serialised
    with a lock; the lock is uncontended overhead in server mode.
    """

    def __init__(
        self,
        model_name: str,
        *,
        mode: str = "server",
        host: str = "localhost",
        port: int = 6333,
        path: str | None = None,
        encoder: Any | None = None,
        client: Any | None = None,
    ) -> None:
        if mode not in QDRANT_MODES:
            raise ValueError(f"qdrant mode must be one of {QDRANT_MODES}, got {mode!r}")
        if encoder is None and not _ST_AVAILABLE:
            raise ImportError("sentence-transformers is required for dense retrieval.")
        if client is None and not _QDRANT_AVAILABLE:
            raise ImportError("qdrant-client is required for dense retrieval.")
        if mode == "embedded" and not path and client is None:
            raise ValueError("qdrant mode 'embedded' requires a storage path")

        self._model_name = model_name
        self._mode = mode
        self._lock = threading.RLock()

        if encoder is None:
            logger.info("Loading sentence-transformer model: %s", model_name)
            encoder = SentenceTransformer(model_name)
        self._encoder = encoder
        self._vector_size = int(_embedding_dimension(encoder))
        logger.info("Encoder ready: model=%s vector_size=%d", model_name, self._vector_size)

        if client is None:
            if mode == "server":
                logger.info("Connecting to Qdrant server: host=%s port=%d", host, port)
                client = QdrantClient(host=host, port=port, timeout=10)
            elif mode == "embedded":
                logger.info("Opening embedded Qdrant at %s", path)
                client = QdrantClient(path=path)
            else:
                logger.info("Using in-memory Qdrant (non-persistent)")
                client = QdrantClient(":memory:")
        self._client = client

    # -- introspection -------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def vector_size(self) -> int:
        return self._vector_size

    def list_collections(self) -> list[str]:
        """Collection names. Raises QdrantUnavailableError if unreachable."""
        try:
            with self._lock:
                return [c.name for c in self._client.get_collections().collections]
        except Exception as exc:
            raise QdrantUnavailableError(f"Cannot reach Qdrant ({self._mode}): {exc}") from exc

    def collection_count(self, collection_name: str) -> int | None:
        """Exact point count, or None if the collection does not exist."""
        if collection_name not in self.list_collections():
            return None
        try:
            with self._lock:
                return int(self._client.count(collection_name=collection_name, exact=True).count)
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Qdrant count failed for collection '{collection_name}': {exc}"
            ) from exc

    # -- embedding -----------------------------------------------------------

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> list[list[float]]:
        with self._lock:
            return self._encoder.encode(
                list(texts), batch_size=batch_size,
                show_progress_bar=False, normalize_embeddings=True,
            ).tolist()

    # -- collection lifecycle ------------------------------------------------

    def build_collection(
        self,
        collection_name: str,
        documents: Sequence[tuple[ChunkId, str]],
        batch_size: int = 64,
    ) -> int:
        """(Re)create `collection_name` holding exactly `documents`.

        Verifies the resulting point count equals len(documents) and raises
        QdrantUnavailableError otherwise, so a partially-built dense index
        can never be published.  Returns the point count.
        """
        if not documents:
            raise ValueError("build_collection() requires at least one document")
        try:
            with self._lock:
                if collection_name in [c.name for c in self._client.get_collections().collections]:
                    self._client.delete_collection(collection_name)
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
                )
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Failed to create Qdrant collection '{collection_name}': {exc}"
            ) from exc

        total = len(documents)
        for batch_start in range(0, total, batch_size):
            batch = documents[batch_start: batch_start + batch_size]
            vectors = self.embed([text for _, text in batch], batch_size=batch_size)
            points = [
                PointStruct(
                    id=deterministic_point_id(str(cid)),
                    vector=vec,
                    payload={"chunk_id": cid},
                )
                for (cid, _), vec in zip(batch, vectors)
            ]
            try:
                with self._lock:
                    self._client.upsert(collection_name=collection_name, points=points)
            except Exception as exc:
                raise QdrantUnavailableError(
                    f"Qdrant upsert failed (batch at {batch_start}): {exc}"
                ) from exc

        count = self.collection_count(collection_name)
        if count != total:
            raise QdrantUnavailableError(
                f"Qdrant collection '{collection_name}' holds {count} points, "
                f"expected {total}"
            )
        logger.info("Dense collection built: %s (%d points)", collection_name, total)
        return total

    def delete_collection(self, collection_name: str) -> None:
        try:
            with self._lock:
                self._client.delete_collection(collection_name)
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Failed to delete Qdrant collection '{collection_name}': {exc}"
            ) from exc

    def retriever(self, collection_name: str) -> "DenseRetriever":
        """A DenseRetriever view bound to one collection (no model load)."""
        return DenseRetriever(backend=self, collection_name=collection_name)

    # -- search --------------------------------------------------------------

    def search(self, collection_name: str, query_text: str, top_k: int) -> list["DenseResult"]:
        vector = self.embed([query_text])[0]
        try:
            with self._lock:
                response = self._client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    limit=top_k,
                    with_payload=True,
                )
        except Exception as exc:
            raise QdrantUnavailableError(
                f"Qdrant search failed for collection '{collection_name}': {exc}"
            ) from exc
        results: list[DenseResult] = []
        for hit in response.points:
            payload = hit.payload or {}
            chunk_id = payload.get("chunk_id")
            if chunk_id is None:
                logger.warning("Qdrant hit missing 'chunk_id' in payload (id=%s); skipping", hit.id)
                continue
            results.append(DenseResult(chunk_id=ChunkId(chunk_id), score=float(hit.score)))
        return results


def _embedding_dimension(encoder: Any) -> int:
    getter = getattr(encoder, "get_embedding_dimension", None) or getattr(
        encoder, "get_sentence_embedding_dimension"
    )
    return getter()


class DenseRetriever:
    """Dense retrieval over ONE Qdrant collection (one KB version).

    A thin view over a shared DenseBackend: constructing one never loads a
    model or opens a connection.
    """

    def __init__(self, backend: DenseBackend, collection_name: str) -> None:
        if not collection_name:
            raise ValueError("DenseRetriever requires a collection name")
        self._backend = backend
        self._collection_name = collection_name

    @classmethod
    def build(
        cls,
        model_name: str,
        qdrant_host: str,
        qdrant_port: int,
        collection_name: str,
        *,
        mode: str = "server",
        path: str | None = None,
    ) -> "DenseRetriever":
        """Create a backend and a retriever for `collection_name`, verifying
        that Qdrant is reachable (raises QdrantUnavailableError otherwise)."""
        backend = DenseBackend(
            model_name, mode=mode, host=qdrant_host, port=qdrant_port, path=path
        )
        backend.list_collections()
        return cls(backend=backend, collection_name=collection_name)

    def ensure_collection(self) -> None:
        if self._collection_name not in self._backend.list_collections():
            raise QdrantUnavailableError(
                f"Qdrant collection '{self._collection_name}' does not exist"
            )

    def upsert(self, documents: list[tuple[ChunkId, str]], batch_size: int = 64) -> None:
        """(Re)build this retriever's collection from `documents`."""
        if not documents:
            raise ValueError("upsert() requires at least one document")
        self._backend.build_collection(self._collection_name, documents, batch_size=batch_size)

    def query(self, query_text: str, top_k: int) -> list[DenseResult]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not query_text.strip():
            logger.debug("Dense query is empty; returning empty results")
            return []
        return self._backend.search(self._collection_name, query_text, top_k)

    def count(self) -> int | None:
        return self._backend.collection_count(self._collection_name)

    @property
    def backend(self) -> DenseBackend:
        return self._backend

    @property
    def vector_size(self) -> int:
        return self._backend.vector_size

    @property
    def collection_name(self) -> str:
        return self._collection_name
