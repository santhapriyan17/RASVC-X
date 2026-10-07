"""Corpus document schema and lookup store for RASVC-X (Module 13).

This module owns:
  - CorpusDocument: typed schema for a source document with full provenance.
  - CorpusStore:    in-memory lookup from ChunkId → (text, Provenance).
                    Serialised to/from JSON; the source of truth for
                    provenance at query time (Qdrant stores chunk_id only).
  - CorpusManifest: fingerprint + artifact paths written by build_index.py
                    and validated at application startup.
  - StaleIndexError: raised when the on-disk index does not match the
                    corpus fingerprint.

Design constraints:
  - No BM25Index, DenseRetriever, or pipeline code here.
  - UNKNOWN provenance represented as None in JSON; converted to the
    UNKNOWN singleton when building Provenance objects.
  - Synthetic documents MUST set is_synthetic=True.
  - No real patient data permitted.
  - JSON serialisation uses sort_keys=True for reproducible fingerprints.

Fingerprint composition (SHA-256):
  corpus content: sorted JSON of each document dict
  chunking config: strategy + max_tokens + overlap
  schema version: _CORPUS_SCHEMA_VERSION constant
  The fingerprint covers the logical corpus, not the BM25 pickle bytes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rasvcx.retrieval.chunking import Chunk, ChunkConfig, ChunkStrategy, chunk_document
from rasvcx.schemas.common import ChunkId, SourceType, UNKNOWN
from rasvcx.schemas.evidence import Provenance, SourceLifecycle, SourceRef

_CORPUS_SCHEMA_VERSION: int = 1
_INDEX_FORMAT_VERSION: str = "bm25-v1"


# ---------------------------------------------------------------------------
# Corpus document schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusDocument:
    """A single source document with provenance metadata.

    All provenance fields use None to represent genuinely unknown values.
    None is serialised to JSON as null and converted to the UNKNOWN
    sentinel when constructing Provenance objects for the pipeline.

    is_synthetic MUST be True for any document not sourced from a real
    published healthcare document.  Never set is_synthetic=False for
    invented content.
    """

    doc_id: str
    title: str
    text: str
    source_type: str                    # SourceType.value string
    date: str | None = None             # publication date; None = UNKNOWN
    jurisdiction: str | None = None     # e.g. "US", "EU"; None = UNKNOWN
    population: str | None = None       # e.g. "adults"; None = UNKNOWN
    dosage_context: str | None = None   # e.g. "oral"; None = UNKNOWN
    source_url: str | None = None       # attribution URL; None if unavailable
    is_synthetic: bool = True           # MUST be True for synthetic content
    lifecycle: dict[str, Any] | None = None  # SourceLifecycle record; None = not declared

    def __post_init__(self) -> None:
        if not self.doc_id:
            raise ValueError("CorpusDocument.doc_id must be non-empty")
        if not self.title:
            raise ValueError("CorpusDocument.title must be non-empty")
        if not self.text:
            raise ValueError("CorpusDocument.text must be non-empty")
        try:
            SourceType(self.source_type)
        except ValueError:
            valid = [st.value for st in SourceType]
            raise ValueError(
                f"CorpusDocument.source_type must be one of {valid}, "
                f"got {self.source_type!r}"
            )

    def to_provenance(self) -> Provenance:
        """Construct an M1 Provenance object from this document's metadata.

        None fields are converted to the UNKNOWN sentinel as required by
        the existing Provenance contract (schemas/evidence.py:30-43).
        """
        def _or_unknown(v: str | None) -> str | type(UNKNOWN):
            return UNKNOWN if v is None else v

        return Provenance(
            source_type=SourceType(self.source_type),
            date=_or_unknown(self.date),
            jurisdiction=_or_unknown(self.jurisdiction),
            population=_or_unknown(self.population),
            dosage_context=_or_unknown(self.dosage_context),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "source_type": self.source_type,
            "date": self.date,
            "jurisdiction": self.jurisdiction,
            "population": self.population,
            "dosage_context": self.dosage_context,
            "source_url": self.source_url,
            "is_synthetic": self.is_synthetic,
            **({"lifecycle": self.lifecycle} if self.lifecycle else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorpusDocument":
        return cls(
            doc_id=data["doc_id"],
            title=data["title"],
            text=data["text"],
            source_type=data["source_type"],
            date=data.get("date"),
            jurisdiction=data.get("jurisdiction"),
            population=data.get("population"),
            dosage_context=data.get("dosage_context"),
            source_url=data.get("source_url"),
            is_synthetic=data.get("is_synthetic", True),
            lifecycle=data.get("lifecycle") or None,
        )


_MODELLED_RECORD_KEYS = frozenset({
    "chunk_id", "text", "provenance", "doc_id", "title",
    "source_url", "filename", "heading", "page", "lifecycle",
})


def doc_id_from_chunk_id(chunk_id: str) -> str:
    """Recover the document id from a chunk id.

    Handles both id schemes in use: "{doc_id}__chunk_{index:04d}"
    (retrieval/chunking.py) and "{doc_id}::{section}::{page}::{index}"
    (ingestion/corpus_builder.py).  Falls back to the chunk id itself.
    """
    if "__chunk_" in chunk_id:
        return chunk_id.rsplit("__chunk_", 1)[0]
    if "::" in chunk_id:
        return chunk_id.split("::", 1)[0]
    return chunk_id


def metadata_signature(provenance: Any, lifecycle: Any, title: Any) -> str:
    """Canonical JSON of a document's indexed metadata (None == "")."""

    def _c(v: Any) -> Any:
        if v in ("", None):
            return None
        if isinstance(v, dict):
            return {k: _c(x) for k, x in sorted(v.items())}
        if isinstance(v, (list, tuple)):
            return [_c(x) for x in v] or None
        return v

    return json.dumps(
        {"provenance": _c(provenance), "lifecycle": _c(lifecycle), "title": _c(title)},
        sort_keys=True, ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Corpus store — ChunkId → (text, Provenance)
# ---------------------------------------------------------------------------


class CorpusStore:
    """In-memory lookup from ChunkId to chunk text, Provenance and SourceRef.

    This is the authoritative source for provenance and attribution at
    query time.  Qdrant stores only the chunk_id in its payload; text,
    provenance and source attribution come from here.

    One on-disk format serves both the offline index builder
    (scripts/build_index.py) and the ingestion knowledge plane
    (ingestion/corpus_builder.py): a JSON list of chunk records

        {"chunk_id", "text", "provenance": {...}, "doc_id", "title",
         "source_url", "filename", "heading", "page", ...}

    Only chunk_id and text are required.  A record without a provenance
    block gets source_type OTHER and UNKNOWN context fields -- missing
    context is never fabricated.  Keys the store does not model (job_id,
    content_hash, ingestion_timestamp, ...) are preserved verbatim so a
    load/merge/save cycle is lossless.
    """

    def __init__(self) -> None:
        self._store: dict[ChunkId, tuple[str, Provenance]] = {}
        self._sources: dict[ChunkId, SourceRef] = {}
        self._extras: dict[ChunkId, dict[str, Any]] = {}
        self._content_hashes: set[str] = set()
        # doc_id -> doc_id of a document that declares it `supersedes` it.
        # Derived at load; applied by lookup_source(), never written back.
        self._superseded_by: dict[str, str] = {}

    def add(
        self,
        chunk_id: ChunkId,
        text: str,
        provenance: Provenance,
        source: SourceRef | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        """Register a chunk. Raises ValueError on duplicate chunk_id."""
        if chunk_id in self._store:
            raise ValueError(
                f"CorpusStore: duplicate chunk_id {chunk_id!r}"
            )
        if not text:
            raise ValueError(f"CorpusStore: text must be non-empty for {chunk_id!r}")
        self._store[chunk_id] = (text, provenance)
        self._sources[chunk_id] = source or SourceRef(doc_id=doc_id_from_chunk_id(chunk_id))
        lifecycle = self._sources[chunk_id].lifecycle
        if lifecycle is not None:
            for old in lifecycle.supersedes:
                if old != self._sources[chunk_id].doc_id:
                    self._superseded_by.setdefault(old, self._sources[chunk_id].doc_id)
        if extras:
            self._extras[chunk_id] = dict(extras)
            content_hash = extras.get("content_hash")
            if isinstance(content_hash, str) and content_hash:
                self._content_hashes.add(content_hash)

    def lookup(self, chunk_id: ChunkId) -> tuple[str, Provenance] | None:
        """Return (text, Provenance) or None if chunk_id is not present."""
        return self._store.get(chunk_id)

    def lookup_source(self, chunk_id: ChunkId) -> SourceRef | None:
        """Return the SourceRef for chunk_id, or None if not present.

        Supersession is applied KB-wide: if another document in this store
        declares it supersedes this one, the returned lifecycle carries
        superseded_by even when this document's own metadata does not.
        """
        src = self._sources.get(chunk_id)
        if src is None:
            return None
        newer = self._superseded_by.get(src.doc_id)
        if newer is None or (src.lifecycle is not None and src.lifecycle.superseded_by):
            return src
        import dataclasses

        base = src.lifecycle or SourceLifecycle()
        return dataclasses.replace(src, lifecycle=dataclasses.replace(base, superseded_by=newer))

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._store

    def chunk_ids(self) -> list[ChunkId]:
        return list(self._store.keys())

    def doc_ids(self) -> set[str]:
        return {src.doc_id for src in self._sources.values()}

    def has_content_hash(self, content_hash: str) -> bool:
        """True if a document with this SHA-256 was ingested into this store."""
        return content_hash in self._content_hashes

    def metadata_signatures(self, content_hash: str) -> set[str]:
        """Metadata signatures of the indexed documents with this SHA-256.

        Lets ingestion tell "same file, same metadata" (a duplicate) from
        "same file, new metadata" (an update that must be published)."""
        if content_hash not in self._content_hashes:
            return set()
        sigs: set[str] = set()
        for rec in self.to_records():
            if rec.get("content_hash") == content_hash:
                sigs.add(metadata_signature(rec.get("provenance"), rec.get("lifecycle"), rec.get("title")))
        return sigs

    def to_documents_list(self) -> list[tuple[ChunkId, str]]:
        """Return (chunk_id, text) pairs for BM25Index.build()."""
        return [(cid, text) for cid, (text, _) in self._store.items()]

    # -- Serialisation -------------------------------------------------------

    def to_records(self) -> list[dict[str, Any]]:
        """Chunk records in the on-disk format. UNKNOWN is stored as null."""

        def _prov_to_dict(p: Provenance) -> dict[str, Any]:
            def _v(v: Any) -> str | None:
                return None if v is UNKNOWN else str(v)
            return {
                "source_type": p.source_type.value,
                "date": _v(p.date),
                "jurisdiction": _v(p.jurisdiction),
                "population": _v(p.population),
                "dosage_context": _v(p.dosage_context),
            }

        records: list[dict[str, Any]] = []
        for cid, (text, prov) in self._store.items():
            src = self._sources[cid]
            rec: dict[str, Any] = dict(self._extras.get(cid, {}))
            rec.update({
                "chunk_id": cid,
                "text": text,
                "provenance": _prov_to_dict(prov),
                "doc_id": src.doc_id,
                "title": src.title,
                "source_url": src.source_url,
                "filename": src.filename,
                "heading": src.heading,
                "page": src.page,
            })
            if src.lifecycle is not None:
                rec["lifecycle"] = src.lifecycle.to_record()
            records.append(rec)
        return records

    def save(self, path: Path) -> None:
        """Serialise to JSON. UNKNOWN is stored as null."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_records(), fh, sort_keys=True, ensure_ascii=False, indent=2)

    @classmethod
    def from_records(cls, records: list[dict[str, Any]]) -> "CorpusStore":
        """Build a store from chunk records. null fields -> UNKNOWN sentinel."""
        if not isinstance(records, list):
            raise ValueError("CorpusStore records must be a JSON list")

        def _or_unknown(v: Any) -> Any:
            return UNKNOWN if v is None or v == "" else v

        def _or_none(v: Any) -> Any:
            return None if v is None or v == "" else v

        store = cls()
        for rec in records:
            cid = ChunkId(rec["chunk_id"])
            pd = rec.get("provenance") or {}
            try:
                source_type = SourceType(pd.get("source_type") or SourceType.OTHER.value)
            except ValueError:
                source_type = SourceType.OTHER
            prov = Provenance(
                source_type=source_type,
                date=_or_unknown(pd.get("date")),
                jurisdiction=_or_unknown(pd.get("jurisdiction")),
                population=_or_unknown(pd.get("population")),
                dosage_context=_or_unknown(pd.get("dosage_context")),
            )
            page = rec.get("page")
            source = SourceRef(
                doc_id=str(rec.get("doc_id") or doc_id_from_chunk_id(cid)),
                title=_or_none(rec.get("title")),
                source_url=_or_none(rec.get("source_url")),
                filename=_or_none(rec.get("filename")),
                heading=_or_none(rec.get("heading")),
                page=page if isinstance(page, int) and not isinstance(page, bool) else None,
                lifecycle=SourceLifecycle.from_record(rec.get("lifecycle")),
            )
            extras = {k: v for k, v in rec.items() if k not in _MODELLED_RECORD_KEYS}
            store.add(cid, rec["text"], prov, source=source, extras=extras)
        return store

    @classmethod
    def load(cls, path: Path) -> "CorpusStore":
        """Deserialise from JSON. null fields -> UNKNOWN sentinel."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CorpusStore file not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)
        return cls.from_records(records)


# ---------------------------------------------------------------------------
# Corpus manifest — fingerprint + artifact paths
# ---------------------------------------------------------------------------


class StaleIndexError(Exception):
    """Raised when the on-disk index fingerprint does not match the current
    corpus content or chunking configuration."""


@dataclass
class CorpusManifest:
    """Records fingerprint and artifact paths produced by build_index.py.

    corpus_fingerprint: SHA-256 covering document content, chunking config,
                        and schema version.  Used to detect stale indexes.
    chunk_count:        Total chunks across all documents.
    doc_count:          Total source documents.
    store_path:         Path to CorpusStore JSON.
    bm25_path:          Path to BM25Index pickle.
    qdrant_collection:  Qdrant collection name (None if not built).
    index_format_version: _INDEX_FORMAT_VERSION constant.
    """

    corpus_fingerprint: str
    chunk_count: int
    doc_count: int
    store_path: str
    bm25_path: str
    qdrant_collection: str | None
    index_format_version: str = _INDEX_FORMAT_VERSION
    schema_version: int = _CORPUS_SCHEMA_VERSION

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, sort_keys=True, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CorpusManifest":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus manifest not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(**data)

    def validate_against(self, fingerprint: str) -> None:
        """Raise StaleIndexError if the stored fingerprint differs."""
        if self.corpus_fingerprint != fingerprint:
            raise StaleIndexError(
                f"Corpus index is stale: stored fingerprint "
                f"{self.corpus_fingerprint!r} != current {fingerprint!r}. "
                f"Rebuild with: python scripts/build_index.py"
            )


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------


def compute_corpus_fingerprint(
    documents: list[CorpusDocument],
    chunk_config: ChunkConfig,
) -> str:
    """Compute a deterministic SHA-256 fingerprint of the corpus.

    Covers: sorted document content, chunking parameters, schema version.
    Does NOT cover the BM25 pickle bytes (those are derived artifacts).
    """
    parts: list[str] = []

    for doc in sorted(documents, key=lambda d: d.doc_id):
        parts.append(json.dumps(doc.to_dict(), sort_keys=True, ensure_ascii=False))

    parts.append(json.dumps({
        "strategy": chunk_config.strategy.value,
        "max_tokens": chunk_config.max_tokens,
        "overlap": chunk_config.overlap,
    }, sort_keys=True))

    parts.append(json.dumps({"schema_version": _CORPUS_SCHEMA_VERSION}))

    combined = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


# ---------------------------------------------------------------------------
# Corpus builder — documents → CorpusStore + chunk list
# ---------------------------------------------------------------------------


def build_corpus_store(
    documents: list[CorpusDocument],
    chunk_config: ChunkConfig,
) -> tuple[CorpusStore, list[tuple[ChunkId, str]]]:
    """Chunk all documents, populate a CorpusStore, and return the
    (chunk_id, text) list ready for BM25Index.build()."""
    if not documents:
        raise ValueError("build_corpus_store requires at least one document")

    seen_doc_ids: set[str] = set()
    store = CorpusStore()
    chunk_pairs: list[tuple[ChunkId, str]] = []

    for doc in documents:
        if doc.doc_id in seen_doc_ids:
            raise ValueError(
                f"build_corpus_store: duplicate doc_id {doc.doc_id!r}"
            )
        seen_doc_ids.add(doc.doc_id)

        provenance = doc.to_provenance()
        source = SourceRef(
            doc_id=doc.doc_id, title=doc.title, source_url=doc.source_url,
            lifecycle=SourceLifecycle.from_record(doc.lifecycle),
        )
        chunks: list[Chunk] = chunk_document(doc.doc_id, doc.text, chunk_config)

        for chunk in chunks:
            store.add(
                chunk.chunk_id, chunk.text, provenance, source=source,
                extras={"is_synthetic": doc.is_synthetic},
            )
            chunk_pairs.append((chunk.chunk_id, chunk.text))

    return store, chunk_pairs


# ---------------------------------------------------------------------------
# Smoke corpus loader
# ---------------------------------------------------------------------------


def load_corpus_json(path: Path) -> list[CorpusDocument]:
    """Load a list of CorpusDocument objects from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus JSON not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    if not isinstance(records, list):
        raise ValueError(f"Corpus JSON at {path} must contain a list")
    return [CorpusDocument.from_dict(r) for r in records]


__all__ = [
    "CorpusDocument",
    "CorpusStore",
    "CorpusManifest",
    "StaleIndexError",
    "compute_corpus_fingerprint",
    "build_corpus_store",
    "load_corpus_json",
    "doc_id_from_chunk_id",
]