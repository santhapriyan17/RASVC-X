"""Pydantic models for RASVC-X M17 ingestion pipeline.

Covers:
  - JobStage   — ordered stages of a single ingestion job
  - JobStatus  — terminal and non-terminal job states
  - IngestJob  — full job record (stored in SQLite, returned by API)
  - IngestSource — registered external URL source
  - SourceSyncState — sync health of a registered source
  - ActiveVersion — the published corpus version pointer

Design rules:
  - All models use Pydantic v2 (model_config, model_validator).
  - Timestamps are UTC ISO-8601 strings (str, not datetime) so they
    serialise cleanly to/from SQLite TEXT columns without tz gymnastics.
  - job_id and source_id are UUID4 strings generated at creation time.
  - No field stores secrets, API keys, or PII.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class JobStage(str, Enum):
    """Ordered stages of the ingestion state machine.

    Transition path:
      PENDING -> VALIDATING -> EXTRACTING -> CHUNKING
              -> INDEXING -> PUBLISHING -> COMPLETED
      Any stage -> FAILED
      Any stage before PUBLISHING -> CANCELLED
    """
    PENDING = "pending"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobStatus(str, Enum):
    """Coarse status used for quick filtering."""
    RUNNING = "running"      # any of PENDING..PUBLISHING
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SourceType(str, Enum):
    """How the source was registered."""
    FILE_UPLOAD = "file_upload"
    EXTERNAL_URL = "external_url"


class SyncState(str, Enum):
    """Health state of a registered external source."""
    PENDING = "pending"       # never synced
    SYNCING = "syncing"       # sync job running now
    OK = "ok"                 # last sync succeeded
    UNCHANGED = "unchanged"   # last sync found no new content (ETag match)
    FAILED = "failed"         # last sync failed
    SUSPENDED = "suspended"   # too many consecutive failures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    """Return a new UUID4 string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# IngestJob — one document or URL ingestion task
# ---------------------------------------------------------------------------

class IngestJob(BaseModel):
    """Complete record for one ingestion job.

    Created when a file is uploaded or a URL is submitted.
    Updated in place by the background pipeline worker.
    Stored as a row in the SQLite jobs table.
    """

    # Identity
    job_id: str = Field(default_factory=_new_id, description="UUID4 job identifier")
    source_id: str | None = Field(
        default=None,
        description="Source ID if this job was triggered by a source sync",
    )

    # What is being ingested
    source_type: SourceType
    filename: str | None = Field(
        default=None,
        description="Original filename for file uploads; None for URL jobs",
    )
    url: str | None = Field(
        default=None,
        description="URL for external source jobs; None for file uploads",
    )
    file_size_bytes: int | None = Field(
        default=None,
        description="Upload file size in bytes; None if not yet known",
    )
    content_hash: str | None = Field(
        default=None,
        description="SHA-256 hex digest of raw file content; set after upload lands on disk",
    )
    mime_type: str | None = Field(
        default=None,
        description="MIME type detected by filetype library; set after validation",
    )

    # State machine
    stage: JobStage = Field(default=JobStage.PENDING)
    status: JobStatus = Field(default=JobStatus.RUNNING)

    # Progress within EXTRACTING stage
    pages_total: int | None = Field(
        default=None,
        description="Total pages/records in document; set after first parse pass",
    )
    pages_processed: int = Field(
        default=0,
        description="Pages/records processed so far during extraction",
    )
    chunks_produced: int = Field(
        default=0,
        description="Chunks produced during chunking stage",
    )

    # Failure details
    error_stage: str | None = Field(
        default=None,
        description="Stage name at which the job failed",
    )
    error_message: str | None = Field(
        default=None,
        description="Human-readable failure reason; never contains secrets",
    )

    # Published corpus version (set when PUBLISHING completes)
    corpus_version_id: str | None = Field(
        default=None,
        description="Version fingerprint of the corpus snapshot published by this job",
    )

    # Timestamps (UTC ISO-8601 strings)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    completed_at: str | None = Field(default=None)

    model_config = {"frozen": False}  # mutable — updated in place by worker

    @model_validator(mode="after")
    def _check_source_fields(self) -> "IngestJob":
        if self.source_type == SourceType.FILE_UPLOAD and self.url is not None:
            raise ValueError("FILE_UPLOAD jobs must not have a url field")
        if self.source_type == SourceType.EXTERNAL_URL and self.url is None:
            raise ValueError("EXTERNAL_URL jobs must have a url field")
        return self

    def mark_stage(self, stage: JobStage) -> None:
        """Advance to the given stage and refresh updated_at."""
        self.stage = stage
        self.updated_at = _utc_now()
        if stage in (JobStage.COMPLETED,):
            self.status = JobStatus.COMPLETED
            if self.completed_at is None:
                self.completed_at = self.updated_at
        elif stage == JobStage.FAILED:
            self.status = JobStatus.FAILED
            if self.completed_at is None:
                self.completed_at = self.updated_at
        elif stage == JobStage.CANCELLED:
            self.status = JobStatus.CANCELLED
            if self.completed_at is None:
                self.completed_at = self.updated_at
        else:
            self.status = JobStatus.RUNNING

    def mark_failed(self, stage: JobStage, message: str) -> None:
        """Record failure at the given stage with a message."""
        self.error_stage = stage.value
        self.error_message = message[:2000]
        self.mark_stage(JobStage.FAILED)

    def to_row(self) -> dict[str, Any]:
        """Serialise to a flat dict suitable for SQLite insertion."""
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "IngestJob":
        """Deserialise from a SQLite row dict."""
        return cls.model_validate(row)


# ---------------------------------------------------------------------------
# IngestSource — registered external URL for scheduled sync
# ---------------------------------------------------------------------------

class IngestSource(BaseModel):
    """A registered external URL that RASVC-X syncs on a schedule.

    Created by POST /ingest/sources.
    Updated after each sync attempt.
    Stored as a row in the SQLite sources table.
    """

    source_id: str = Field(default_factory=_new_id)
    url: str = Field(description="HTTPS URL of the source document or feed")
    display_name: str = Field(
        default="",
        description="Optional human-readable label for the source",
    )

    # Sync scheduling
    sync_interval_seconds: float = Field(
        default=3600.0,
        description="How often to poll this source (seconds); 0 = manual only",
    )
    next_sync_at: str | None = Field(
        default=None,
        description="UTC ISO-8601 time of next scheduled sync; None = not scheduled",
    )

    # Sync state
    sync_state: SyncState = Field(default=SyncState.PENDING)
    consecutive_failures: int = Field(default=0)
    last_sync_at: str | None = Field(default=None)
    last_sync_job_id: str | None = Field(default=None)
    last_error: str | None = Field(default=None)

    # Change detection
    etag: str | None = Field(
        default=None,
        description="Last ETag received from server; used for conditional GET",
    )
    last_modified: str | None = Field(
        default=None,
        description="Last Last-Modified header received from server",
    )
    content_hash: str | None = Field(
        default=None,
        description="SHA-256 of last fetched response body for change detection",
    )

    # Published corpus relationship
    corpus_version_id: str | None = Field(
        default=None,
        description="Version ID of the last successfully published corpus from this source",
    )

    # Document count from last successful sync
    document_count: int = Field(default=0)

    # Timestamps
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)

    model_config = {"frozen": False}

    def mark_sync_started(self, job_id: str) -> None:
        self.sync_state = SyncState.SYNCING
        self.last_sync_job_id = job_id
        self.updated_at = _utc_now()

    def mark_sync_success(
        self,
        *,
        job_id: str,
        content_hash: str | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
        unchanged: bool = False,
        corpus_version_id: str | None = None,
    ) -> None:
        self.sync_state = SyncState.UNCHANGED if unchanged else SyncState.OK
        self.last_sync_job_id = job_id
        self.last_sync_at = _utc_now()
        self.consecutive_failures = 0
        self.last_error = None
        if content_hash is not None:
            self.content_hash = content_hash
        if etag is not None:
            self.etag = etag
        if last_modified is not None:
            self.last_modified = last_modified
        if corpus_version_id is not None:
            self.corpus_version_id = corpus_version_id
        self.updated_at = _utc_now()

    def mark_sync_failed(self, message: str, suspend_after: int = 5) -> None:
        self.consecutive_failures += 1
        self.last_error = message[:2000]
        self.last_sync_at = _utc_now()
        if self.consecutive_failures >= suspend_after:
            self.sync_state = SyncState.SUSPENDED
        else:
            self.sync_state = SyncState.FAILED
        self.updated_at = _utc_now()

    def to_row(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "IngestSource":
        return cls.model_validate(row)


# ---------------------------------------------------------------------------
# ActiveVersion — the committed corpus version pointer
# ---------------------------------------------------------------------------

class ActiveVersion(BaseModel):
    """Contents of corpus/active_version.json.

    Written atomically by the publisher (os.replace) as the commit step.
    Read at startup and after each successful ingestion publication.
    """

    version_id: str = Field(
        description="Fingerprint-based version identifier, e.g. 'v_abc123def456'"
    )
    store_path: str = Field(
        description="Path to CorpusStore JSON for this version"
    )
    bm25_path: str = Field(
        description="Path to BM25 pickle for this version"
    )
    qdrant_collection: str = Field(
        description="Qdrant collection name for this version"
    )
    chunk_count: int = Field(description="Total chunks in this version")
    document_count: int = Field(description="Total documents in this version")
    ingestion_job_id: str | None = Field(
        default=None,
        description="Job ID that produced this version; None for hand-built corpora",
    )
    published_at: str = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# API response wrappers
# ---------------------------------------------------------------------------

class IngestJobListResponse(BaseModel):
    """Response for GET /ingest/jobs."""
    jobs: list[IngestJob]
    total: int


class IngestSourceListResponse(BaseModel):
    """Response for GET /ingest/sources."""
    sources: list[IngestSource]
    total: int


class IngestJobCreateResponse(BaseModel):
    """Response for POST /ingest/upload and POST /ingest/url."""
    job_id: str
    status: JobStatus
    stage: JobStage
    message: str


class IngestSourceCreateResponse(BaseModel):
    """Response for POST /ingest/sources."""
    source_id: str
    url: str
    message: str


class CorpusVersionInfo(BaseModel):
    """Lightweight corpus version summary for GET /ingest/status."""
    version_id: str | None
    chunk_count: int
    document_count: int
    published_at: str | None
    qdrant_collection: str | None


class IngestStatusResponse(BaseModel):
    """Response for GET /ingest/status."""
    active_version: CorpusVersionInfo
    running_jobs: int
    pending_jobs: int
    total_jobs: int
    total_sources: int


__all__ = [
    "JobStage",
    "JobStatus",
    "SourceType",
    "SyncState",
    "IngestJob",
    "IngestSource",
    "ActiveVersion",
    "IngestJobListResponse",
    "IngestSourceListResponse",
    "IngestJobCreateResponse",
    "IngestSourceCreateResponse",
    "CorpusVersionInfo",
    "IngestStatusResponse",
]