# src/rasvcx/ingestion/publisher.py
# ---------------------------------------------------------------------------
# Atomic knowledge-base publication:
#   - KBVersionLeaseTracker: explicit request leases + the in-memory
#     registry of loaded KB snapshots (the lifecycle source of truth)
#   - publish_corpus_version: load+verify snapshot -> atomic pointer swap
#     -> in-memory activation
#   - gc_old_versions: delete old version dirs + Qdrant collections when the
#     lease tracker and the retention policy both allow it
# ---------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Version lease tracker
# ---------------------------------------------------------------------------

@dataclass
class _VersionRef:
    version_id: str
    active_request_count: int = 0
    is_inactive: bool = False        # True once a newer version is published
    snapshot: Any | None = None      # retrieval.knowledge_base.KBSnapshot
    published_at: str | None = None
    last_active_at: str | None = None
    inactive_since: float | None = None   # time.monotonic() when superseded


@dataclass(frozen=True)
class KBLease:
    """A request's pin on one knowledge-base version.

    Obtained from KBVersionLeaseTracker.lease(); must be released exactly
    once (release() is idempotent).  While the lease is held the version's
    artifacts cannot be garbage-collected.
    """

    version_id: str
    snapshot: Any | None
    _tracker: "KBVersionLeaseTracker"
    _state: dict[str, bool]

    def release(self) -> None:
        if self._state.get("released"):
            return
        self._state["released"] = True
        self._tracker.release(self.version_id)

    def __enter__(self) -> "KBLease":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class KBVersionLeaseTracker:
    """
    Thread-safe owner of knowledge-base version lifecycle.

    Explicit leases, not Python reference counting, decide when a version
    may be deleted:

        lease = tracker.lease()          # request start: pin current version
        try:
            ... query using lease.snapshot ...
        finally:
            lease.release()              # request end

    Per version the tracker records: version_id, active_request_count,
    published_at, last_active_at and status (active | inactive).

    GC is allowed only when the version is
      - inactive (a newer version is published)
      - AND active_request_count == 0
      - AND the retention policy is satisfied (see gc_old_versions).

    Activation (register) and leasing (acquire/lease) take the same lock,
    so a request always pins exactly one fully-registered version.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._versions: dict[str, _VersionRef] = {}
        self._current: str | None = None

    def register(
        self,
        version_id: str,
        snapshot: Any | None = None,
        published_at: str | None = None,
    ) -> None:
        """Make version_id the current version; mark the old current inactive."""
        with self._lock:
            if self._current and self._current != version_id:
                old = self._versions.get(self._current)
                if old:
                    old.is_inactive = True
                    old.inactive_since = time.monotonic()
            ref = self._versions.get(version_id)
            if ref is None:
                ref = _VersionRef(version_id=version_id)
                self._versions[version_id] = ref
            ref.is_inactive = False
            ref.inactive_since = None
            if snapshot is not None:
                ref.snapshot = snapshot
            if published_at is not None:
                ref.published_at = published_at
            self._current = version_id
            logger.debug("LeaseTracker: registered version %s", version_id)

    def register_inactive(
        self, version_id: str, published_at: str | None = None,
        inactive_for_seconds: float = 0.0,
    ) -> None:
        """Record an on-disk version that is not being served (GC candidate).

        inactive_for_seconds: how long it has already been inactive (e.g.
        the age of its directory), so the retention delay counts real time
        rather than restarting at every process start."""
        with self._lock:
            if version_id == self._current or version_id in self._versions:
                return
            self._versions[version_id] = _VersionRef(
                version_id=version_id, is_inactive=True, published_at=published_at,
                inactive_since=time.monotonic() - max(0.0, inactive_for_seconds),
            )

    def acquire(self) -> str | None:
        """Pin the current version for a request. Returns version_id or None."""
        with self._lock:
            ref = self._pin_current_locked()
            return ref.version_id if ref else None

    def lease(self) -> KBLease | None:
        """Pin the current version and return a KBLease (None if no version)."""
        with self._lock:
            ref = self._pin_current_locked()
            if ref is None:
                return None
            return KBLease(
                version_id=ref.version_id, snapshot=ref.snapshot,
                _tracker=self, _state={},
            )

    def _pin_current_locked(self) -> _VersionRef | None:
        if self._current is None:
            return None
        ref = self._versions.get(self._current)
        if ref is None:
            return None
        ref.active_request_count += 1
        ref.last_active_at = _iso_now()
        return ref

    def release(self, version_id: str) -> None:
        """Release a previously acquired version pin."""
        with self._lock:
            ref = self._versions.get(version_id)
            if ref and ref.active_request_count > 0:
                ref.active_request_count -= 1
                ref.last_active_at = _iso_now()

    def is_gc_safe(self, version_id: str) -> bool:
        """True if the version can be safely deleted."""
        with self._lock:
            ref = self._versions.get(version_id)
            if ref is None:
                return True
            return ref.is_inactive and ref.active_request_count == 0

    def mark_inactive(self, version_id: str) -> None:
        with self._lock:
            ref = self._versions.get(version_id)
            if ref and not ref.is_inactive:
                ref.is_inactive = True
                ref.inactive_since = time.monotonic()

    def forget(self, version_id: str) -> bool:
        """Drop a version after its artifacts were deleted. Refuses (returns
        False) unless the version is inactive with no active requests."""
        with self._lock:
            ref = self._versions.get(version_id)
            if ref is None:
                return True
            if version_id == self._current or not ref.is_inactive or ref.active_request_count:
                return False
            del self._versions[version_id]
            return True

    def inactive_seconds(self, version_id: str) -> float | None:
        with self._lock:
            ref = self._versions.get(version_id)
            if ref is None or ref.inactive_since is None:
                return None
            return time.monotonic() - ref.inactive_since

    def snapshot_of(self, version_id: str) -> Any | None:
        with self._lock:
            ref = self._versions.get(version_id)
            return ref.snapshot if ref else None

    def current_snapshot(self) -> Any | None:
        """The current version's snapshot WITHOUT taking a lease (status only)."""
        with self._lock:
            ref = self._versions.get(self._current) if self._current else None
            return ref.snapshot if ref else None

    def list_versions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "version_id": v.version_id,
                    "active_requests": v.active_request_count,
                    "active_request_count": v.active_request_count,
                    "is_inactive": v.is_inactive,
                    "status": "inactive" if v.is_inactive else "active",
                    "published_at": v.published_at,
                    "last_active_at": v.last_active_at,
                    "loaded": v.snapshot is not None,
                }
                for v in self._versions.values()
            ]

    def current_version(self) -> str | None:
        with self._lock:
            return self._current


# ---------------------------------------------------------------------------
# Global publication lock (one publish at a time)
# ---------------------------------------------------------------------------

_pub_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# App-state wiring helpers
# ---------------------------------------------------------------------------

def get_or_create_pub_lock(app_state: Any) -> asyncio.Lock:
    """Return the publication lock stored on app.state, creating it if absent."""
    lock = getattr(app_state, "pub_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        try:
            app_state.pub_lock = lock
        except Exception:
            return _pub_lock
    return lock


def get_or_create_lease_tracker(app_state: Any) -> KBVersionLeaseTracker:
    """Return the KBVersionLeaseTracker on app.state, creating it if absent."""
    tracker = getattr(app_state, "rasvcx_lease_tracker", None)
    if not isinstance(tracker, KBVersionLeaseTracker):
        tracker = KBVersionLeaseTracker()
        try:
            app_state.rasvcx_lease_tracker = tracker
        except Exception:
            pass
    return tracker


def get_current_corpus_version(active_version_path: str) -> str | None:
    """Return the version_id from active_version.json, or None if absent."""
    av = read_active_version(active_version_path)
    if av is None:
        return None
    return av.get("version_id")


def _runtime_of(app_state: Any) -> Any | None:
    """The PipelineRuntime attached at startup, or None.

    Type-checked on purpose: a missing or foreign attribute must never be
    mistaken for a runtime (test doubles return arbitrary objects for any
    attribute name).
    """
    from rasvcx.pipeline.factory import PipelineRuntime

    runtime = getattr(app_state, "rasvcx_runtime", None)
    return runtime if isinstance(runtime, PipelineRuntime) else None


# ---------------------------------------------------------------------------
# active_version.json helpers
# ---------------------------------------------------------------------------

def read_active_version(active_version_path: str) -> dict[str, Any] | None:
    """Read and return the active_version.json, or None if not present."""
    p = Path(active_version_path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read active_version.json: %s", exc)
        return None


def _write_active_version_atomic(
    active_version_path: str,
    data: dict[str, Any],
) -> None:
    """
    Atomically replace active_version.json using a sibling temp file + os.replace.
    Guarantees that readers never see a half-written file.
    """
    p = Path(active_version_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    dir_ = str(p.parent)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, str(p))
        logger.debug("Wrote active_version.json -> %s", active_version_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Startup reconciliation
# ---------------------------------------------------------------------------

def reconcile_publishing_jobs_on_startup(
    job_store: Any,
    active_version_path: str,
) -> None:
    """
    On startup: any job stuck in PUBLISHING was interrupted mid-publish.
    Mark them FAILED so they can be retried.

    Note: uses job_store.update_job_status(job_id, new_status, error_message).
    """
    try:
        stuck = job_store.list_jobs(status_filter="publishing")
        for job in stuck:
            logger.warning(
                "reconcile: job %s stuck in PUBLISHING -- marking FAILED",
                job.job_id,
            )
            job_store.update_job_status(
                job.job_id,
                "failed",
                error_message="Interrupted during publication -- retry required.",
            )
    except Exception as exc:
        logger.error("reconcile_publishing_jobs_on_startup failed: %s", exc)


# ---------------------------------------------------------------------------
# Main: publish_corpus_version
# ---------------------------------------------------------------------------

async def publish_corpus_version(
    build_result: Any,           # CorpusBuildResult
    active_version_path: str,
    app_state: Any,              # FastAPI app.state
    settings: Any,               # rasvcx Settings
    lease_tracker: KBVersionLeaseTracker,
    job_store: Any,
    job_id: str,
    executor: Any,               # loop.run_in_executor executor
) -> None:
    """
    Atomically publish a new knowledge-base version:
      1. Acquire the publication lock (one publish at a time).
      2. Load the version's artifacts into a KBSnapshot and verify integrity
         (BM25 == store == Qdrant).  Any failure aborts here: nothing has
         changed yet and the previous version keeps serving.
      3. Replace active_version.json atomically (durable commit point).
      4. Activate the snapshot in the lease tracker.  New requests lease the
         new version; in-flight requests finish on the version they leased.
      5. Mark the job COMPLETED and run retention GC.

    Steps 3 and 4 cannot diverge: step 4 is an in-memory assignment that
    cannot fail, and a crash between them leaves the pointer on the new
    version, which is exactly what the next startup loads.

    When the process has no query runtime attached (ingestion used as a
    library), step 2 is skipped and the version is registered without a
    snapshot.

    Raises on any failure; caller must mark job FAILED.
    """
    async with _pub_lock:
        version_id = build_result.version_id
        logger.info("publish_corpus_version: publishing version %s", version_id)

        if getattr(build_result, "unchanged", False):
            # The build reproduced the version already being served.
            job_store.set_corpus_version(job_id, version_id)
            job_store.update_job_status(job_id, "completed")
            logger.info("publish_corpus_version: %s already active -- no-op", version_id)
            return

        av_data = build_result.to_active_version()

        snapshot = None
        runtime = _runtime_of(app_state)
        if runtime is not None:
            loop = asyncio.get_running_loop()
            snapshot = await loop.run_in_executor(
                executor, runtime.snapshot_loader.load, av_data
            )

        _write_active_version_atomic(active_version_path, av_data)
        lease_tracker.register(
            version_id, snapshot=snapshot, published_at=build_result.published_at
        )

        job_store.set_corpus_version(job_id, version_id)
        job_store.update_job_status(job_id, "completed")
        logger.info(
            "publish_corpus_version: COMPLETED -- version=%s chunks=%d",
            version_id, build_result.chunk_count,
        )

        ingestion = getattr(settings, "ingestion", None)
        keep = getattr(ingestion, "keep_versions", 2)
        delay = getattr(ingestion, "gc_delay_seconds", 0.0)
        try:
            gc_old_versions(
                lease_tracker,
                corpus_dir=str(Path(build_result.store_path).parent.parent),
                dense_backend=runtime.dense_backend if runtime is not None else None,
                keep_versions=keep if isinstance(keep, int) else 2,
                min_inactive_seconds=float(delay) if isinstance(delay, (int, float)) else 0.0,
            )
        except Exception:  # noqa: BLE001 - GC must never fail a completed publish
            logger.warning("post-publish GC failed", exc_info=True)


# ---------------------------------------------------------------------------
# GC old versions
# ---------------------------------------------------------------------------

def gc_old_versions(
    lease_tracker: KBVersionLeaseTracker,
    corpus_dir: str,
    dense_backend: Any | None = None,
    keep_versions: int = 2,
    min_inactive_seconds: float = 0.0,
    dry_run: bool = False,
) -> list[str]:
    """
    Delete old version directories and their Qdrant collections.

    dry_run=True applies exactly the same eligibility rules and returns the
    versions that WOULD be deleted, deleting nothing.

    A version is deleted only when ALL hold:
      - it is inactive (superseded) and is not the current version
      - active_request_count == 0  (no request holds a lease on it)
      - it has been inactive for at least `min_inactive_seconds`
      - it is not among the `keep_versions` most recent inactive versions

    Only directories named <corpus_dir>/<version_id> are ever removed, so a
    bootstrap version whose artifacts live elsewhere is never touched.

    Returns the version ids that were deleted.
    """
    current = lease_tracker.current_version()
    corpus_path = Path(corpus_dir)

    def _mtime(vid: str) -> float:
        d = corpus_path / vid
        return d.stat().st_mtime if d.exists() else 0.0

    inactive = sorted(
        (v["version_id"] for v in lease_tracker.list_versions()
         if v["is_inactive"] and v["version_id"] != current),
        key=_mtime, reverse=True,
    )
    deleted: list[str] = []
    for vid in inactive[max(keep_versions, 0):]:
        if not lease_tracker.is_gc_safe(vid):
            continue  # a request still holds a lease
        idle = lease_tracker.inactive_seconds(vid)
        if idle is not None and idle < min_inactive_seconds:
            continue
        if dry_run:
            deleted.append(vid)
            continue
        snapshot = lease_tracker.snapshot_of(vid)
        if not lease_tracker.forget(vid):
            continue  # leased between the check and now

        version_dir = corpus_path / vid
        collection = getattr(snapshot, "qdrant_collection", None)
        if collection is None:
            manifest = read_active_version(str(version_dir / "manifest.json")) or {}
            collection = manifest.get("qdrant_collection")
        if version_dir.is_dir() and version_dir.parent == corpus_path:
            try:
                shutil.rmtree(version_dir)
                logger.info("GC: deleted corpus dir %s", version_dir)
            except OSError as exc:
                logger.warning("GC: failed to delete %s: %s", version_dir, exc)

        if dense_backend is not None and collection:
            try:
                dense_backend.delete_collection(collection)
                logger.info("GC: deleted Qdrant collection %s", collection)
            except Exception as exc:  # noqa: BLE001
                logger.warning("GC: Qdrant delete %s failed: %s", collection, exc)
        deleted.append(vid)
    return deleted
