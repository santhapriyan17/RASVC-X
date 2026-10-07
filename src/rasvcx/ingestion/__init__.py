"""RASVC-X M17 — Document Ingestion & Source Synchronisation package.

Public API re-exported for convenience. Callers may import directly from
submodules or from this package:

    from rasvcx.ingestion import JobStore, run_ingestion_pipeline

Submodule responsibilities:
    models.py        — Pydantic schemas: IngestJob, IngestSource, ActiveVersion
    job_store.py     — SQLite-backed job and source state machine
    security.py      — File type validation and SSRF guard
    parsers.py       — Bounded-memory document parsers (subprocess-isolated)
    corpus_builder.py — Extract→chunk→BM25+Qdrant versioned artifact builder
    publisher.py     — Atomic corpus publication and pipeline hot-swap
    source_sync.py   — External URL fetch with ETag/hash change detection
    scheduler.py     — asyncio background scheduler + shared pipeline worker
"""

from rasvcx.ingestion.corpus_builder import (
    CorpusBuildError, CorpusBuildResult,
    build_corpus_version, parse_result_to_corpus_document,
)
from rasvcx.ingestion.job_store import JobStore
from rasvcx.ingestion.models import (
    ActiveVersion, CorpusVersionInfo, IngestJob,
    IngestJobCreateResponse, IngestJobListResponse, IngestSource,
    IngestSourceCreateResponse, IngestSourceListResponse,
    IngestStatusResponse, JobStage, JobStatus, SourceType, SyncState,
)
from rasvcx.ingestion.parsers import (
    ExtractedSection, ParseResult, run_parser_subprocess,
)
from rasvcx.ingestion.publisher import (
    KBLease, KBVersionLeaseTracker, gc_old_versions, get_current_corpus_version,
    get_or_create_lease_tracker, get_or_create_pub_lock,
    publish_corpus_version, read_active_version,
    reconcile_publishing_jobs_on_startup,
)
from rasvcx.ingestion.scheduler import run_ingestion_pipeline, scheduler_loop
from rasvcx.ingestion.security import (
    SUPPORTED_FORMATS, FileValidationResult, SecurityError,
    UrlValidationResult, validate_file, validate_url,
)
from rasvcx.ingestion.source_sync import (
    SyncError, SyncFetchResult, build_ingest_job_for_source,
    compute_next_sync_at, sync_source,
)

__all__ = [
    "IngestJob", "IngestSource", "ActiveVersion", "JobStage", "JobStatus",
    "SourceType", "SyncState", "CorpusVersionInfo", "IngestJobListResponse",
    "IngestSourceListResponse", "IngestJobCreateResponse",
    "IngestSourceCreateResponse", "IngestStatusResponse",
    "JobStore",
    "SecurityError", "FileValidationResult", "UrlValidationResult",
    "SUPPORTED_FORMATS", "validate_file", "validate_url",
    "ParseResult", "ExtractedSection", "run_parser_subprocess",
    "CorpusBuildError", "CorpusBuildResult",
    "parse_result_to_corpus_document", "build_corpus_version",
    "KBLease", "KBVersionLeaseTracker", "get_or_create_pub_lock",
    "get_or_create_lease_tracker", "get_current_corpus_version",
    "read_active_version",
    "reconcile_publishing_jobs_on_startup", "publish_corpus_version",
    "gc_old_versions",
    "SyncError", "SyncFetchResult", "sync_source",
    "compute_next_sync_at", "build_ingest_job_for_source",
    "run_ingestion_pipeline", "scheduler_loop",
]