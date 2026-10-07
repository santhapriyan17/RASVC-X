"""Immutable knowledge-base snapshots for RASVC-X.

A KBSnapshot is one published knowledge-base version as the query plane
sees it: the CorpusStore, the BM25 index and (in hybrid mode) the Qdrant
collection that were all built from the SAME set of chunks, plus the
retrieval callables bound to exactly those three artifacts.

This is the object that connects the knowledge plane to the query plane:

  ingestion publishes  ->  SnapshotLoader.load(active_version)  ->  KBSnapshot
  /query               ->  lease a KBSnapshot  ->  retrieve from it  ->  release

A request holds one snapshot for its whole lifetime, so it can never mix
BM25 from one version with Qdrant from another, and a publish that lands
mid-request does not change what that request retrieves from.

Integrity is checked at load time, before a snapshot can be served:
  - the BM25 index and the CorpusStore must cover the same chunk ids;
  - in hybrid mode the Qdrant collection must exist and hold exactly one
    point per chunk.
A snapshot that fails either check raises KBIntegrityError and is never
published.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rasvcx.retrieval.bm25 import BM25Index
from rasvcx.retrieval.bridge import (
    RetrievalFn,
    TargetedRetrievalFn,
    make_retrieval_fn,
    make_targeted_retrieval_fn,
)
from rasvcx.retrieval.corpus import CorpusStore

logger = logging.getLogger(__name__)


class KBIntegrityError(Exception):
    """The artifacts of a knowledge-base version are missing or inconsistent."""


def collection_name_for(prefix: str, version_id: str) -> str:
    """Qdrant collection name owned by one knowledge-base version."""
    return f"{prefix}_{version_id}"


#: Where the served knowledge base came from.
KB_SOURCE_PUBLISHED = "published_kb"   # a version published by ingestion (active pointer)
KB_SOURCE_SEED = "seed_fallback"       # no active pointer: the seed index was served
KB_SOURCE_SMOKE = "smoke_test"         # offline_test only: smoke corpus indexed in memory


def corpus_content_hash(store: CorpusStore) -> str:
    """SHA-256 over the (chunk_id, text) content of a store.

    Same scheme as ingestion version ids (v_<hash[:12]>), so a version id
    and its corpus hash always agree.
    """
    from rasvcx.ingestion.corpus_builder import _fingerprint

    return _fingerprint(store.to_records())


def file_sha256(path: str | Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class KBSnapshot:
    """One immutable, internally consistent knowledge-base version."""

    version_id: str
    corpus_store: CorpusStore
    bm25_index: BM25Index
    dense_retriever: Any | None
    qdrant_collection: str | None
    published_at: str | None
    chunk_count: int
    doc_count: int
    retrieval_fn: RetrievalFn
    targeted_retrieval_fn: TargetedRetrievalFn
    kb_source: str = KB_SOURCE_PUBLISHED
    corpus_hash: str | None = None
    index_hash: str | None = None
    """SHA-256 of the BM25 index file; None when the index was built in memory."""
    identity_scheme: str | None = None
    """kb-identity-v2, or kb-identity-v1-text-only for a legacy version id."""
    dense_index_origin: str | None = None
    """persisted (the collection existed and was opened) | rebuilt_from_store
    (it was missing, e.g. qdrant_mode=memory, and was re-embedded at load) |
    None (BM25-only)."""

    def describe(self) -> dict[str, Any]:
        """Serialisable summary (no index contents)."""
        return {
            "version_id": self.version_id,
            "kb_source": self.kb_source,
            "publication_status": "ACTIVE" if self.kb_source == KB_SOURCE_PUBLISHED else "NOT_PUBLISHED",
            "dense_index_origin": self.dense_index_origin,
            "corpus_hash": self.corpus_hash,
            "index_hash": self.index_hash,
            "identity_scheme": self.identity_scheme,
            "chunk_count": self.chunk_count,
            "doc_count": self.doc_count,
            "bm25_size": self.bm25_index.size,
            "qdrant_collection": self.qdrant_collection,
            "retrieval_mode": "hybrid" if self.dense_retriever is not None else "bm25_only",
            "published_at": self.published_at,
        }


class SnapshotLoader:
    """Builds KBSnapshot objects for one runtime configuration.

    `retrieval` is the frozen RetrievalSettings of the running process.
    `dense_backend` is the shared DenseBackend in hybrid mode, else None.
    """

    def __init__(self, retrieval: Any, dense_backend: Any | None = None) -> None:
        self._retrieval = retrieval
        self._dense_backend = dense_backend
        if retrieval.mode == "hybrid" and dense_backend is None:
            raise KBIntegrityError(
                "retrieval.mode='hybrid' requires a dense backend; refusing to "
                "build BM25-only snapshots for a hybrid configuration"
            )

    @property
    def hybrid(self) -> bool:
        return self._retrieval.mode == "hybrid"

    @property
    def dense_backend(self) -> Any | None:
        return self._dense_backend

    def collection_name(self, version_id: str) -> str:
        return collection_name_for(self._retrieval.qdrant_collection, version_id)

    def load(self, active_version: Mapping[str, Any]) -> KBSnapshot:
        """Load the version described by an active_version record from disk."""
        version_id = active_version.get("version_id")
        if not version_id:
            raise KBIntegrityError("active version record has no version_id")
        store_path = Path(str(active_version.get("store_path") or ""))
        bm25_path = Path(str(active_version.get("bm25_path") or ""))
        for label, path in (("store", store_path), ("bm25", bm25_path)):
            if not path.is_file():
                raise KBIntegrityError(
                    f"KB version {version_id}: {label} artifact not found at {path}"
                )
        try:
            store = CorpusStore.load(store_path)
            bm25 = BM25Index.load(bm25_path)
        except Exception as exc:
            raise KBIntegrityError(
                f"KB version {version_id}: artifacts unreadable: {exc}"
            ) from exc
        return self.build(
            version_id=str(version_id),
            corpus_store=store,
            bm25_index=bm25,
            qdrant_collection=active_version.get("qdrant_collection"),
            published_at=active_version.get("published_at"),
            kb_source=KB_SOURCE_PUBLISHED,
            index_hash=file_sha256(bm25_path),
        )

    def build(
        self,
        version_id: str,
        corpus_store: CorpusStore,
        bm25_index: BM25Index,
        qdrant_collection: str | None = None,
        published_at: str | None = None,
        kb_source: str = KB_SOURCE_PUBLISHED,
        index_hash: str | None = None,
        dense_index_origin: str | None = "persisted",
    ) -> KBSnapshot:
        """Assemble and integrity-check a snapshot from loaded artifacts.

        The corpus hash is recomputed from the loaded chunks, and a version
        id of the content-derived form (v_<12 hex>) must match it: a
        version label that does not describe the chunks being served is an
        integrity failure, not a cosmetic one.
        """
        if len(corpus_store) == 0:
            raise KBIntegrityError(f"KB version {version_id}: corpus store is empty")
        from rasvcx.ingestion.corpus_builder import (
            IDENTITY_SCHEME,
            LEGACY_IDENTITY_SCHEME,
            _fingerprint_v1,
        )

        corpus_hash = corpus_content_hash(corpus_store)
        identity_scheme = IDENTITY_SCHEME
        if (
            len(version_id) == 14 and version_id.startswith("v_")
            and version_id[2:] != corpus_hash[:12]
        ):
            # A version published before identity v2 carries a text-only id.
            # It is still served, but labelled legacy: its id does not cover
            # metadata.  Anything else is an integrity failure.
            if version_id[2:] == _fingerprint_v1(corpus_store.to_records())[:12]:
                identity_scheme = LEGACY_IDENTITY_SCHEME
            else:
                raise KBIntegrityError(
                    f"KB version {version_id}: the loaded chunks hash to "
                    f"v_{corpus_hash[:12]} -- artifacts do not belong to this version"
                )
        store_ids = set(corpus_store.chunk_ids())
        bm25_ids = set(bm25_index.chunk_ids())
        if store_ids != bm25_ids:
            raise KBIntegrityError(
                f"KB version {version_id}: BM25 index and corpus store disagree "
                f"(store={len(store_ids)} bm25={len(bm25_ids)} "
                f"only_store={len(store_ids - bm25_ids)} only_bm25={len(bm25_ids - store_ids)})"
            )

        dense_retriever = None
        if self.hybrid:
            if not qdrant_collection:
                raise KBIntegrityError(
                    f"KB version {version_id} has no Qdrant collection but "
                    f"retrieval.mode='hybrid'"
                )
            count = self._dense_backend.collection_count(qdrant_collection)
            if count is None:
                raise KBIntegrityError(
                    f"KB version {version_id}: Qdrant collection "
                    f"{qdrant_collection!r} does not exist"
                )
            if count != len(corpus_store):
                raise KBIntegrityError(
                    f"KB version {version_id}: Qdrant collection {qdrant_collection!r} "
                    f"holds {count} points but the corpus store has {len(corpus_store)} chunks"
                )
            dense_retriever = self._dense_backend.retriever(qdrant_collection)
        else:
            qdrant_collection = None

        r = self._retrieval
        return KBSnapshot(
            version_id=version_id,
            corpus_store=corpus_store,
            bm25_index=bm25_index,
            dense_retriever=dense_retriever,
            qdrant_collection=qdrant_collection,
            published_at=published_at,
            chunk_count=len(corpus_store),
            doc_count=len(corpus_store.doc_ids()),
            retrieval_fn=make_retrieval_fn(
                bm25_index=bm25_index,
                dense_retriever=dense_retriever,
                corpus_store=corpus_store,
                bm25_top_k=r.bm25_top_k,
                dense_top_k=r.dense_top_k,
                rrf_top_k=r.rrf_top_k,
                rrf_k=r.rrf_k,
                kb_version_id=version_id,
            ),
            targeted_retrieval_fn=make_targeted_retrieval_fn(
                bm25_index=bm25_index,
                dense_retriever=dense_retriever,
                corpus_store=corpus_store,
                targeted_bm25_top_k=r.targeted_bm25_top_k,
                targeted_dense_top_k=r.targeted_dense_top_k,
                targeted_rrf_k=r.targeted_rrf_k,
            ),
            kb_source=kb_source,
            corpus_hash=corpus_hash,
            index_hash=index_hash,
            identity_scheme=identity_scheme,
            dense_index_origin=dense_index_origin if dense_retriever is not None else None,
        )


__all__ = [
    "KB_SOURCE_PUBLISHED",
    "KB_SOURCE_SEED",
    "KB_SOURCE_SMOKE",
    "KBIntegrityError",
    "KBSnapshot",
    "SnapshotLoader",
    "collection_name_for",
    "corpus_content_hash",
    "file_sha256",
]
