# src/rasvcx/ingestion/corpus_builder.py
# ---------------------------------------------------------------------------
# Builds a new versioned knowledge-base snapshot from ingested documents.
#
# The artifacts written here are the SAME formats the query plane loads
# (retrieval/corpus.py CorpusStore records, retrieval/bm25.py BM25Index
# pickle, one Qdrant collection per version), so a published version is
# directly retrievable -- there is no second, ingestion-only index format.
# ---------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_PROVENANCE_FIELDS = ("source_type", "date", "jurisdiction", "population", "dosage_context")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CorpusBuildError(Exception):
    """Raised when corpus build fails unrecoverably."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorpusBuildResult:
    version_id: str
    store_path: str
    bm25_path: str
    manifest_path: str
    qdrant_collection: str | None
    doc_count: int
    chunk_count: int
    published_at: str  # ISO-8601
    unchanged: bool = False  # True when the build reproduced the active version

    def to_active_version(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "store_path": self.store_path,
            "bm25_path": self.bm25_path,
            "manifest_path": self.manifest_path,
            "qdrant_collection": self.qdrant_collection,
            "published_at": self.published_at,
            "doc_count": self.doc_count,
            "chunk_count": self.chunk_count,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Canonical KB identity rule (v2).  A chunk's identity is every field the
#: query plane reads from the store -- retrieval (text), provenance and
#: validation (provenance, lifecycle) and attribution shown to users
#: (doc_id, title, source_url, filename, heading, page).  Ingestion
#: bookkeeping (job_id, ingestion_timestamp, content_hash, chunk_index,
#: chunker_version, section, ...) is excluded: it is volatile and is never
#: read at query time.  Empty strings and None are the same value.
IDENTITY_FIELDS = (
    "chunk_id", "text", "provenance", "lifecycle",
    "doc_id", "title", "source_url", "filename", "heading", "page",
)
IDENTITY_SCHEME = "kb-identity-v2"
LEGACY_IDENTITY_SCHEME = "kb-identity-v1-text-only"


def _canonical(value: Any) -> Any:
    if value in ("", None):
        return None
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value] or None
    return value


def _fingerprint(chunks: list[dict[str, Any]]) -> str:
    """KB identity v2: SHA-256 over every identity field of every chunk.

    A metadata-only change (provenance, lifecycle, attribution) changes the
    hash -- and therefore the version id -- because it changes what
    retrieval, validation and the UI see.  Volatile bookkeeping does not.
    """
    h = hashlib.sha256()
    rows = sorted(
        json.dumps(
            {f: _canonical(c.get(f)) for f in IDENTITY_FIELDS},
            sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        )
        for c in chunks
    )
    for row in rows:
        h.update(hashlib.sha256(row.encode("utf-8")).digest())
    return h.hexdigest()


def _fingerprint_v1(chunks: list[dict[str, Any]]) -> str:
    """Legacy identity (v1): (chunk_id, text) only.  Kept so versions
    published before v2 still load and are reported as legacy."""
    h = hashlib.sha256()
    for cid, text in sorted((c["chunk_id"], c["text"]) for c in chunks):
        h.update(cid.encode("utf-8"))
        h.update(b"\x00")
        h.update(hashlib.sha256(text.encode("utf-8")).digest())
    return h.hexdigest()


def _iso_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def normalise_provenance(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Provenance block for ingested chunks.

    Only values the uploader actually supplied are recorded.  Anything
    missing stays null (UNKNOWN at query time) -- never guessed from the
    document text.  An unrecognised source_type is rejected.
    """
    from rasvcx.schemas.common import SourceType

    metadata = metadata or {}
    raw_type = str(metadata.get("source_type") or SourceType.OTHER.value).strip().lower()
    try:
        source_type = SourceType(raw_type).value
    except ValueError as exc:
        valid = ", ".join(st.value for st in SourceType)
        raise CorpusBuildError(
            f"Unknown source_type {raw_type!r}. Valid values: {valid}"
        ) from exc
    prov: dict[str, Any] = {"source_type": source_type}
    for key in _PROVENANCE_FIELDS[1:]:
        value = metadata.get(key)
        value = str(value).strip() if value is not None else ""
        prov[key] = value or None
    return prov


def normalise_lifecycle(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Lifecycle block for ingested chunks, or None when nothing is declared.

    status must be one of current | superseded | historical | withdrawn;
    supersedes may be a comma-separated string or a list of doc_ids.
    """
    from rasvcx.schemas.evidence import LIFECYCLE_STATUSES, SourceLifecycle

    metadata = metadata or {}
    status = str(metadata.get("status") or "").strip().lower() or None
    if status is not None and status not in LIFECYCLE_STATUSES:
        raise CorpusBuildError(
            f"Unknown lifecycle status {status!r}. Valid values: {', '.join(LIFECYCLE_STATUSES)}"
        )
    raw = metadata.get("supersedes") or ()
    if isinstance(raw, str):
        raw = raw.split(",")
    lc = SourceLifecycle.from_record({
        "status": status,
        "effective_date": metadata.get("effective_date"),
        "superseded_by": metadata.get("superseded_by"),
        "supersedes": [s for s in raw if str(s).strip()],
        "version": metadata.get("version"),
    })
    return lc.to_record() if lc is not None else None


def document_metadata_signature(metadata: Mapping[str, Any] | None, filename: str | None) -> str:
    """The metadata a document would be indexed with (see
    CorpusStore.metadata_signatures).  Same file + same signature = duplicate."""
    from rasvcx.retrieval.corpus import metadata_signature

    title = str((metadata or {}).get("title") or "").strip() or (
        Path(filename).stem if filename else ""
    )
    return metadata_signature(normalise_provenance(metadata), normalise_lifecycle(metadata), title)


# ---------------------------------------------------------------------------
# Core: parse_result_to_corpus_document
# ---------------------------------------------------------------------------

def parse_result_to_corpus_document(
    parse_result: Any,        # parsers.ParseResult
    job_id: str,
    doc_id: str,
    source_url: str | None,
    filename: str | None,
    content_hash: str,
    chunker_version: str = "v1",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    doc_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a ParseResult into a list of chunk records ready for indexing.

    chunk_id is globally unique per document:  {doc_id}::{section_ordinal}::{page}::{chunk_index}
    This guarantees that distinct sections never collide during dedup, while
    re-ingesting the same document reproduces identical ids (idempotent).

    Each record is a CorpusStore chunk record (retrieval/corpus.py):
      chunk_id, text, provenance{...}, doc_id, title, heading, page,
      source_url, filename
    plus ingestion bookkeeping preserved verbatim by the store:
      job_id, section, content_hash, chunk_index, chunker_version,
      ingestion_timestamp
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    step = max(chunk_size - max(chunk_overlap, 0), 1)

    provenance = normalise_provenance(doc_metadata)
    lifecycle = normalise_lifecycle(doc_metadata)
    title = str((doc_metadata or {}).get("title") or "").strip() or (
        Path(filename).stem if filename else doc_id
    )

    chunks: list[dict[str, Any]] = []
    ts = _iso_now()

    for sec_ord, section in enumerate(parse_result.sections):
        text = section.text.strip()
        if not text:
            continue

        words = text.split()
        if not words:
            continue

        heading = getattr(section, "heading", None)
        page = getattr(section, "page", None)
        is_table = getattr(section, "is_table", False)
        section_kind = "table" if is_table else "text"

        start = 0
        idx = 0
        while start < len(words):
            window = words[start : start + chunk_size]
            chunk_text = " ".join(window)
            chunk_id = f"{doc_id}::{sec_ord}::{page or 0}::{idx}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "title": title,
                    "job_id": job_id,
                    "section": section_kind,
                    "heading": heading or "",
                    "page": page,
                    "text": chunk_text,
                    "provenance": dict(provenance),
                    "source_url": source_url or "",
                    "filename": filename or "",
                    "content_hash": content_hash,
                    "chunk_index": idx,
                    "chunker_version": chunker_version,
                    "ingestion_timestamp": ts,
                    **({"lifecycle": dict(lifecycle)} if lifecycle else {}),
                }
            )
            idx += 1
            if start + chunk_size >= len(words):
                break
            start += step

    return chunks


# ---------------------------------------------------------------------------
# Core: build_corpus_version
# ---------------------------------------------------------------------------

def build_corpus_version(
    new_chunks: list[dict[str, Any]],
    corpus_dir: str,
    active_version_path: str,
    dense_backend: Any | None = None,
    collection_prefix: str = "rasvcx_chunks",
    base_snapshot: Any | None = None,
) -> CorpusBuildResult:
    """
    Build a new knowledge-base version:
      1. Take the current corpus as the merge base: `base_snapshot` (the
         KBSnapshot the process is serving) when given, otherwise the
         version referenced by active_version_path.
      2. Merge new_chunks (deduplicate by chunk_id; new wins).
      3. Write <corpus_dir>/v_<id>/store.json + bm25.pkl + manifest.json.
      4. Build the version's Qdrant collection when dense_backend is given
         (hybrid mode).  dense_backend=None builds a BM25-only version.
      5. Verify integrity by reloading every artifact.
      6. Clean up on failure.

    Nothing here touches the active version pointer: publication is a
    separate, atomic step (publisher.publish_corpus_version).

    If the merge reproduces the currently active version exactly, the
    active artifacts are left untouched and the result has unchanged=True.

    Returns CorpusBuildResult. Raises CorpusBuildError on any failure.
    """
    from rasvcx.retrieval.bm25 import BM25Index
    from rasvcx.retrieval.corpus import CorpusStore
    from rasvcx.retrieval.knowledge_base import collection_name_for

    corpus_dir_path = Path(corpus_dir)

    # -- Load existing corpus ------------------------------------------------
    # A missing pointer means "no corpus yet".  An unreadable one is an
    # error: starting fresh would silently drop the whole knowledge base.
    active: dict[str, Any] = {}
    existing_records: list[dict[str, Any]] = []
    if Path(active_version_path).exists():
        try:
            with open(active_version_path, encoding="utf-8") as f:
                active = json.load(f)
        except Exception as exc:
            raise CorpusBuildError(
                f"Cannot read the active version pointer {active_version_path}: {exc}"
            ) from exc

    if base_snapshot is not None:
        existing_records = base_snapshot.corpus_store.to_records()
        base_version_id = base_snapshot.version_id
        logger.info(
            "Merging onto served KB version %s (%d chunks)",
            base_version_id, len(existing_records),
        )
    else:
        base_version_id = active.get("version_id")
        store_path = active.get("store_path", "")
        if store_path:
            try:
                existing_records = CorpusStore.load(Path(store_path)).to_records()
            except Exception as exc:
                raise CorpusBuildError(
                    f"Cannot load the active corpus at {store_path}: {exc}"
                ) from exc
            logger.info(
                "Loaded %d existing chunks from %s", len(existing_records), store_path
            )

    # -- Merge (additive, deduplicate by chunk_id) ---------------------------
    merged_map: dict[str, dict[str, Any]] = {c["chunk_id"]: c for c in existing_records}
    for c in new_chunks:
        merged_map[c["chunk_id"]] = c  # new version wins

    if not merged_map:
        raise CorpusBuildError("No chunks to index after merge — corpus would be empty.")

    try:
        # Round-trip through CorpusStore so every record is normalised to
        # the one on-disk schema regardless of where it came from.
        store = CorpusStore.from_records(list(merged_map.values()))
    except Exception as exc:
        raise CorpusBuildError(f"Invalid chunk record: {exc}") from exc
    merged = store.to_records()

    # -- Version ID ----------------------------------------------------------
    fp = _fingerprint(merged)
    version_id = f"v_{fp[:12]}"
    version_dir = corpus_dir_path / version_id

    store_path_str    = str(version_dir / "store.json")
    bm25_path_str     = str(version_dir / "bm25.pkl")
    manifest_path_str = str(version_dir / "manifest.json")
    doc_count = len(store.doc_ids())
    qdrant_collection = (
        collection_name_for(collection_prefix, version_id) if dense_backend is not None else None
    )

    # -- Idempotent rebuild of the active version ----------------------------
    if base_version_id == version_id:
        logger.info("Build reproduced active version %s — nothing to publish", version_id)
        same_pointer = active.get("version_id") == version_id
        return CorpusBuildResult(
            version_id=version_id,
            store_path=str(active.get("store_path", "")) if same_pointer else "",
            bm25_path=str(active.get("bm25_path", "")) if same_pointer else "",
            manifest_path=str(active.get("manifest_path", "")) if same_pointer else "",
            qdrant_collection=(
                getattr(base_snapshot, "qdrant_collection", None)
                if base_snapshot is not None else active.get("qdrant_collection")
            ),
            doc_count=doc_count,
            chunk_count=len(merged),
            published_at=str(active.get("published_at") or _iso_now()),
            unchanged=True,
        )

    version_dir.mkdir(parents=True, exist_ok=True)

    try:
        # -- Write store.json ------------------------------------------------
        with open(store_path_str, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False)
        logger.info("Wrote store.json: %d chunks -> %s", len(merged), store_path_str)

        # -- Build and write BM25 index --------------------------------------
        bm25 = BM25Index.build(store.to_documents_list())
        bm25.save(Path(bm25_path_str), extra={"chunks": merged, "version_id": version_id})
        logger.info("Wrote bm25.pkl -> %s", bm25_path_str)

        # -- Qdrant dense indexing (hybrid mode) -----------------------------
        if dense_backend is not None:
            dense_backend.build_collection(qdrant_collection, store.to_documents_list())
            logger.info("Qdrant collection created: %s", qdrant_collection)

        # -- Write manifest --------------------------------------------------
        published_at = _iso_now()
        manifest = {
            "version_id": version_id,
            "published_at": published_at,
            "doc_count": doc_count,
            "chunk_count": len(merged),
            "store_path": store_path_str,
            "bm25_path": bm25_path_str,
            "qdrant_collection": qdrant_collection,
            "fingerprint": fp,
        }
        with open(manifest_path_str, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # -- Integrity check: reload every artifact --------------------------
        reloaded_store = CorpusStore.load(Path(store_path_str))
        reloaded_bm25 = BM25Index.load(Path(bm25_path_str))
        if len(reloaded_store) != len(merged) or reloaded_bm25.size != len(merged):
            raise CorpusBuildError(
                f"Reload count mismatch: expected {len(merged)}, "
                f"store={len(reloaded_store)} bm25={reloaded_bm25.size}"
            )
        if set(reloaded_bm25.chunk_ids()) != set(reloaded_store.chunk_ids()):
            raise CorpusBuildError("BM25 index and store cover different chunk ids")
        if dense_backend is not None:
            dense_count = dense_backend.collection_count(qdrant_collection)
            if dense_count != len(merged):
                raise CorpusBuildError(
                    f"Qdrant collection {qdrant_collection} holds {dense_count} "
                    f"points, expected {len(merged)}"
                )
        logger.info("Integrity check passed for version %s", version_id)

        return CorpusBuildResult(
            version_id=version_id,
            store_path=store_path_str,
            bm25_path=bm25_path_str,
            manifest_path=manifest_path_str,
            qdrant_collection=qdrant_collection,
            doc_count=doc_count,
            chunk_count=len(merged),
            published_at=published_at,
        )

    except Exception as exc:
        logger.exception("corpus_builder: build failed for version %s — cleaning up", version_id)
        shutil.rmtree(version_dir, ignore_errors=True)
        if dense_backend is not None:
            try:
                dense_backend.delete_collection(qdrant_collection)
            except Exception:  # noqa: BLE001 - best-effort cleanup of a failed build
                logger.warning("could not remove partial Qdrant collection %s", qdrant_collection)
        if isinstance(exc, CorpusBuildError):
            raise
        raise CorpusBuildError(f"Build failed: {exc}") from exc
