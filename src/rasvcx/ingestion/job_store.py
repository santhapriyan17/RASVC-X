"""SQLite-backed job and source store for RASVC-X M17 ingestion.

Flat-keyword façade API used by routes_ingest, scheduler, publisher and tests.
Schema is created in __init__ (no separate initialise() call required).
Stage/status are free-form strings:
  stage:  queued -> validating -> parsing -> chunking -> indexing
          -> building_corpus -> publishing -> completed | failed | cancelled
  status: queued (in-flight) -> completed | failed | cancelled
Thread-safe: one write lock; WAL allows concurrent reads. Single worker only.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

TERMINAL = {"completed", "failed", "cancelled"}

_JOB_COLUMNS = (
    "job_id", "source_id", "source_type", "filename", "url",
    "content_hash", "stage", "status", "error_message",
    "corpus_version_id", "created_at", "updated_at", "completed_at",
)

_SOURCE_COLUMNS = (
    "source_id", "url", "display_name",
    "sync_interval_seconds", "next_sync_at",
    "sync_state", "consecutive_failures",
    "last_sync_at", "last_sync_job_id", "last_error",
    "etag", "last_modified", "content_hash",
    "corpus_version_id", "document_count",
    "created_at", "updated_at",
)

_CREATE_JOBS = f"""
CREATE TABLE IF NOT EXISTS jobs (
    {', '.join(f'{c} TEXT' for c in _JOB_COLUMNS)},
    PRIMARY KEY (job_id)
)
"""

_CREATE_SOURCES = f"""
CREATE TABLE IF NOT EXISTS sources (
    {', '.join(f'{c} TEXT' for c in _SOURCE_COLUMNS)},
    PRIMARY KEY (source_id)
)
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def _job_view(row: sqlite3.Row) -> SimpleNamespace:
    d = dict(row)
    d["source_url"] = d.get("url")
    return SimpleNamespace(**d)


def _source_view(row: sqlite3.Row) -> SimpleNamespace:
    d = dict(row)
    for int_field in ("consecutive_failures", "document_count"):
        if d.get(int_field) is not None:
            try:
                d[int_field] = int(d[int_field])
            except (ValueError, TypeError):
                pass
    if d.get("sync_interval_seconds") is not None:
        try:
            d["sync_interval_seconds"] = float(d["sync_interval_seconds"])
        except (ValueError, TypeError):
            pass
    d["source_url"] = d.get("url")
    d["is_active"] = str(d.get("sync_state") or "").lower() not in ("suspended",)
    return SimpleNamespace(**d)


class JobStore:
    """Thread-safe SQLite store for ingestion jobs and sources (flat-kwarg API)."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._write_lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._write_lock:
            con = _connect(self._db_path)
            try:
                con.execute(_CREATE_JOBS)
                con.execute(_CREATE_SOURCES)
                con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status)")
                con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_stage ON jobs (stage)")
                con.execute("CREATE INDEX IF NOT EXISTS idx_sources_next ON sources (next_sync_at)")
                con.commit()
                logger.info("JobStore schema ready: %s", self._db_path)
            finally:
                con.close()

    def initialise(self) -> None:
        self._ensure_schema()

    # ----------------------------------------------------------------- jobs

    def create_job(
        self,
        job_id: str,
        source_id: str | None = None,
        source_type: str = "file_upload",
        filename: str | None = None,
        source_url: str | None = None,
        content_hash: str | None = None,
        stage: str = "queued",
        status: str = "queued",
    ) -> SimpleNamespace:
        now = _now()
        row = {
            "job_id": job_id,
            "source_id": source_id,
            "source_type": source_type,
            "filename": filename,
            "url": source_url,
            "content_hash": content_hash,
            "stage": stage,
            "status": status,
            "error_message": None,
            "corpus_version_id": None,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
        }
        cols = ", ".join(_JOB_COLUMNS)
        ph = ", ".join(f":{c}" for c in _JOB_COLUMNS)
        with self._write_lock:
            con = _connect(self._db_path)
            try:
                con.execute(f"INSERT INTO jobs ({cols}) VALUES ({ph})", row)
                con.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Job {job_id} already exists") from exc
            finally:
                con.close()
        return self.get_job(job_id)

    def _update_job(self, job_id: str, fields: dict[str, Any]) -> bool:
        fields = dict(fields)
        fields["updated_at"] = _now()
        sets = ", ".join(f"{k}=:{k}" for k in fields)
        params = dict(fields)
        params["job_id"] = job_id
        with self._write_lock:
            con = _connect(self._db_path)
            try:
                cur = con.execute(
                    f"UPDATE jobs SET {sets} WHERE job_id=:job_id", params
                )
                con.commit()
                return cur.rowcount > 0
            finally:
                con.close()

    def update_job_stage(self, job_id: str, stage: str) -> bool:
        fields: dict[str, Any] = {"stage": stage}
        if stage in TERMINAL:
            fields["status"] = stage
            fields["completed_at"] = _now()
        return self._update_job(job_id, fields)

    def update_job_status(
        self, job_id: str, status: str, error_message: str | None = None
    ) -> bool:
        fields: dict[str, Any] = {"status": status}
        if error_message is not None:
            fields["error_message"] = error_message[:2000]
        if status in TERMINAL:
            fields["stage"] = status
            fields["completed_at"] = _now()
        return self._update_job(job_id, fields)

    def update_job_progress(self, job_id: str, **_kwargs: Any) -> None:
        """Progress hook (chunks_produced etc.); non-fatal, columns not persisted."""
        return None

    def set_corpus_version(self, job_id: str, version_id: str) -> bool:
        return self._update_job(job_id, {"corpus_version_id": version_id})

    def get_job(self, job_id: str) -> SimpleNamespace | None:
        con = _connect(self._db_path)
        try:
            row = con.execute(
                "SELECT * FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            return _job_view(row) if row else None
        finally:
            con.close()

    def list_jobs(
        self,
        status_filter: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SimpleNamespace]:
        con = _connect(self._db_path)
        try:
            if status_filter:
                rows = con.execute(
                    "SELECT * FROM jobs WHERE status=? OR stage=? "
                    "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (status_filter, status_filter, limit, offset),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [_job_view(r) for r in rows]
        finally:
            con.close()

    def count_jobs_by_status(self) -> dict[str, int]:
        con = _connect(self._db_path)
        try:
            rows = con.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}
        finally:
            con.close()

    def cancel_job(self, job_id: str) -> bool:
        now = _now()
        with self._write_lock:
            con = _connect(self._db_path)
            try:
                cur = con.execute(
                    "UPDATE jobs SET stage='cancelled', status='cancelled', "
                    "updated_at=?, completed_at=? "
                    "WHERE job_id=? AND stage NOT IN "
                    "('publishing','completed','failed','cancelled')",
                    (now, now, job_id),
                )
                con.commit()
                return cur.rowcount > 0
            finally:
                con.close()

    def is_duplicate_content(self, content_hash: str) -> bool:
        con = _connect(self._db_path)
        try:
            row = con.execute(
                "SELECT 1 FROM jobs WHERE content_hash=? AND status='completed' LIMIT 1",
                (content_hash,),
            ).fetchone()
            return row is not None
        finally:
            con.close()

    def recover_interrupted_jobs(self) -> int:
        now = _now()
        with self._write_lock:
            con = _connect(self._db_path)
            try:
                cur = con.execute(
                    "UPDATE jobs SET status='failed', stage='failed', "
                    "error_message='Interrupted by process restart', "
                    "updated_at=?, completed_at=? "
                    "WHERE status NOT IN ('completed','failed','cancelled') "
                    "AND stage != 'publishing'",
                    (now, now),
                )
                con.commit()
                return cur.rowcount
            finally:
                con.close()

    # -------------------------------------------------------------- sources

    def create_source(
        self,
        source_id: str,
        display_name: str = "",
        source_url: str = "",
        sync_interval_seconds: float = 3600.0,
        sync_state: str = "idle",
    ) -> SimpleNamespace:
        now = _now()
        row = {c: None for c in _SOURCE_COLUMNS}
        row.update(
            {
                "source_id": source_id,
                "url": source_url,
                "display_name": display_name,
                "sync_interval_seconds": sync_interval_seconds,
                "sync_state": sync_state,
                "consecutive_failures": 0,
                "document_count": 0,
                "created_at": now,
                "updated_at": now,
            }
        )
        cols = ", ".join(_SOURCE_COLUMNS)
        ph = ", ".join(f":{c}" for c in _SOURCE_COLUMNS)
        with self._write_lock:
            con = _connect(self._db_path)
            try:
                con.execute(f"INSERT INTO sources ({cols}) VALUES ({ph})", row)
                con.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Source {source_id} already exists") from exc
            finally:
                con.close()
        return self.get_source(source_id)

    def update_source_fields(self, source_id: str, fields: dict[str, Any]) -> bool:
        fields = dict(fields)
        fields["updated_at"] = _now()
        sets = ", ".join(f"{k}=:{k}" for k in fields)
        params = dict(fields)
        params["source_id"] = source_id
        with self._write_lock:
            con = _connect(self._db_path)
            try:
                cur = con.execute(
                    f"UPDATE sources SET {sets} WHERE source_id=:source_id", params
                )
                con.commit()
                return cur.rowcount > 0
            finally:
                con.close()

    def mark_source_sync_started(self, source_id: str) -> bool:
        return self.update_source_fields(source_id, {"sync_state": "syncing"})

    def mark_source_sync_success(
        self,
        source_id: str,
        etag: str | None = None,
        last_modified: str | None = None,
        content_hash: str | None = None,
    ) -> bool:
        fields: dict[str, Any] = {
            "sync_state": "idle",
            "last_sync_at": _now(),
            "consecutive_failures": 0,
            "last_error": None,
        }
        if etag is not None:
            fields["etag"] = etag
        if last_modified is not None:
            fields["last_modified"] = last_modified
        if content_hash is not None:
            fields["content_hash"] = content_hash
        return self.update_source_fields(source_id, fields)

    def mark_source_sync_failed(self, source_id: str, error: str) -> bool:
        src = self.get_source(source_id)
        fails = (getattr(src, "consecutive_failures", 0) or 0) + 1 if src else 1
        state = "suspended" if fails >= 5 else "error"
        return self.update_source_fields(
            source_id,
            {
                "sync_state": state,
                "consecutive_failures": fails,
                "last_error": error[:2000],
                "last_sync_at": _now(),
            },
        )

    def get_source(self, source_id: str) -> SimpleNamespace | None:
        con = _connect(self._db_path)
        try:
            row = con.execute(
                "SELECT * FROM sources WHERE source_id=?", (source_id,)
            ).fetchone()
            return _source_view(row) if row else None
        finally:
            con.close()

    def list_sources(self, limit: int = 100, offset: int = 0) -> list[SimpleNamespace]:
        con = _connect(self._db_path)
        try:
            rows = con.execute(
                "SELECT * FROM sources ORDER BY created_at ASC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [_source_view(r) for r in rows]
        finally:
            con.close()

    def get_due_sources(self, now: str | None = None) -> list[SimpleNamespace]:
        now = now or _now()
        con = _connect(self._db_path)
        try:
            rows = con.execute(
                "SELECT * FROM sources "
                "WHERE next_sync_at IS NOT NULL AND next_sync_at <= ? "
                "AND sync_state NOT IN ('suspended','syncing') "
                "ORDER BY next_sync_at ASC",
                (now,),
            ).fetchall()
            return [_source_view(r) for r in rows]
        finally:
            con.close()

    def source_url_exists(self, url: str) -> bool:
        con = _connect(self._db_path)
        try:
            row = con.execute(
                "SELECT 1 FROM sources WHERE url=? LIMIT 1", (url,)
            ).fetchone()
            return row is not None
        finally:
            con.close()

    def delete_source(self, source_id: str) -> bool:
        with self._write_lock:
            con = _connect(self._db_path)
            try:
                cur = con.execute(
                    "DELETE FROM sources WHERE source_id=?", (source_id,)
                )
                con.commit()
                return cur.rowcount > 0
            finally:
                con.close()


__all__ = ["JobStore", "IngestJobStore"]

# Callers and tests use IngestJobStore
IngestJobStore = JobStore