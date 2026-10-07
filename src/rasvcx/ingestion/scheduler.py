# src/rasvcx/ingestion/scheduler.py
# ---------------------------------------------------------------------------
# Shared ingestion worker + scheduler loop.
#
# run_ingestion_pipeline() — full end-to-end pipeline for one job:
#   validate → parse → chunk → index (build_corpus_version) → publish → COMPLETED
#
# scheduler_loop() — background asyncio task:
#   polls job_store for due sources, fires fetch + pipeline jobs
# ---------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# One worker thread for CPU-bound work (BM25 build, pickle, parse subprocess)
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rasvcx-ingest")

# Build + publish is a read-modify-write of the knowledge base (merge onto
# the active version, then repoint to the result).  It must be serialised
# across jobs, otherwise two concurrent jobs both merge onto the same base
# and the second publication silently drops the first job's documents.
_build_publish_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_doc_id(content_hash: str, filename: str) -> str:
    """Stable doc_id: sha256(hash + filename)[:16]."""
    raw = f"{content_hash}::{filename}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _file_hash(path: str, buf: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core: run_ingestion_pipeline
# ---------------------------------------------------------------------------

async def run_ingestion_pipeline(
    job_id: str,
    temp_path: str,
    filename: str,
    content_hash: str,
    source_url: str | None,
    source_id: str | None,
    job_store: Any,
    active_version_path: str,
    corpus_dir: str,
    lease_tracker: Any,
    app_state: Any,
    settings: Any,
    doc_metadata: dict[str, Any] | None = None,
) -> None:
    """
    Full ingestion pipeline for a single job. Runs in the background.
    Updates job stage at every step. Cleans up temp_path in finally.

    When the process serves queries in hybrid mode (a PipelineRuntime with
    a dense backend is attached to app_state), the new version's Qdrant
    collection is built alongside its BM25 index and both are published
    together.  doc_metadata carries uploader-supplied provenance
    (source_type, date, jurisdiction, population, dosage_context, title).

    Stages: VALIDATING → PARSING → CHUNKING → BUILDING_CORPUS → PUBLISHING → COMPLETED
    On any error: marks job FAILED with error message.
    """
    from .models import JobStage
    from .security import validate_file, SecurityError
    from .parsers import run_parser_subprocess
    from .corpus_builder import parse_result_to_corpus_document, build_corpus_version
    from .publisher import _runtime_of, publish_corpus_version

    loop = asyncio.get_running_loop()
    runtime = _runtime_of(app_state)
    dense_backend = runtime.dense_backend if runtime is not None else None
    collection_prefix = (
        runtime.settings.retrieval.qdrant_collection if runtime is not None else "rasvcx_chunks"
    )
    ingestion_cfg = runtime.settings.ingestion if runtime is not None else None
    chunking_cfg = runtime.settings.chunking if runtime is not None else None

    def _cancelled() -> bool:
        try:
            job = job_store.get_job(job_id)
        except Exception:  # noqa: BLE001 - a store hiccup must not cancel a job
            return False
        return job is not None and getattr(job, "status", None) == "cancelled"

    def _update_stage(stage: str) -> None:
        try:
            if _cancelled():
                return
            job_store.update_job_stage(job_id, stage)
        except Exception as exc:
            logger.warning("update_job_stage(%s, %s) failed: %s", job_id, stage, exc)

    def _fail(msg: str) -> None:
        try:
            job_store.update_job_status(job_id, "failed", error_message=msg)
        except Exception as exc:
            logger.warning("update_job_status FAILED for %s: %s", job_id, exc)

    try:
        # ── VALIDATING ──────────────────────────────────────────────────────
        _update_stage("validating")
        logger.info("pipeline[%s]: VALIDATING %s", job_id, filename)

        try:
            await loop.run_in_executor(
                _EXECUTOR,
                validate_file,
                temp_path,
                filename,
            )
        except SecurityError as exc:
            from rasvcx.security.events import security_event
            security_event("upload_rejected", job_id=job_id, reason=type(exc).__name__)
            _fail(f"Validation failed: {exc}")
            return
        except Exception as exc:
            _fail(f"Validation error: {exc}")
            return

        # ── PARSING ─────────────────────────────────────────────────────────
        _update_stage("parsing")
        logger.info("pipeline[%s]: PARSING", job_id)

        parse_result = await loop.run_in_executor(
            _EXECUTOR,
            run_parser_subprocess,
            temp_path,
            filename,
            ingestion_cfg,
        )
        if _cancelled():
            logger.info("pipeline[%s]: cancelled after parsing", job_id)
            return

        if parse_result.error:
            _fail(f"Parse error: {parse_result.error}")
            return

        quality = parse_result.quality
        if quality not in ("ok", "partial"):
            _fail(f"Extraction quality too low: {quality}")
            return

        if not parse_result.sections:
            _fail("Parser returned no sections — document may be empty or unsupported.")
            return

        # ── CHUNKING ─────────────────────────────────────────────────────────
        _update_stage("chunking")
        logger.info("pipeline[%s]: CHUNKING (%d sections)", job_id, len(parse_result.sections))

        doc_id = _make_doc_id(content_hash, filename)

        chunks = await loop.run_in_executor(
            _EXECUTOR,
            _do_chunking,
            parse_result,
            job_id,
            doc_id,
            source_url,
            filename,
            content_hash,
            doc_metadata,
            chunking_cfg,
        )

        if not chunks:
            _fail("Chunking produced zero chunks.")
            return

        logger.info("pipeline[%s]: %d chunks produced", job_id, len(chunks))

        # Update job with chunk count
        try:
            job_store.update_job_progress(job_id, chunks_produced=len(chunks))
        except Exception:
            pass  # progress update is non-fatal

        async with _build_publish_lock:
            # ── BUILDING_CORPUS ───────────────────────────────────────────
            _update_stage("building_corpus")
            logger.info("pipeline[%s]: BUILDING_CORPUS", job_id)

            # Merge onto the version this process is actually serving.
            base_snapshot = (
                lease_tracker.current_snapshot() if lease_tracker is not None else None
            )
            try:
                build_result = await loop.run_in_executor(
                    _EXECUTOR,
                    build_corpus_version,
                    chunks,
                    corpus_dir,
                    active_version_path,
                    dense_backend,
                    collection_prefix,
                    base_snapshot,
                )
            except Exception as exc:
                _fail(f"Corpus build failed: {exc}")
                return
            if _cancelled():
                # Built but never published: the version dir is unreferenced
                # and is removed here rather than left for GC to discover.
                logger.info("pipeline[%s]: cancelled before publication", job_id)
                if not build_result.unchanged:
                    import shutil
                    shutil.rmtree(Path(build_result.store_path).parent, ignore_errors=True)
                    if dense_backend is not None and build_result.qdrant_collection:
                        try:
                            dense_backend.delete_collection(build_result.qdrant_collection)
                        except Exception:  # noqa: BLE001
                            logger.warning("could not remove unpublished Qdrant collection")
                return

            # ── PUBLISHING ────────────────────────────────────────────────
            _update_stage("publishing")
            logger.info(
                "pipeline[%s]: PUBLISHING version %s (%d chunks)",
                job_id, build_result.version_id, build_result.chunk_count,
            )

            try:
                await publish_corpus_version(
                    build_result=build_result,
                    active_version_path=active_version_path,
                    app_state=app_state,
                    settings=settings,
                    lease_tracker=lease_tracker,
                    job_store=job_store,
                    job_id=job_id,
                    executor=_EXECUTOR,
                )
            except Exception as exc:
                _fail(f"Publication failed: {exc}")
                # Nothing references a version whose publication failed
                # before the pointer swap; remove it instead of leaving an
                # orphan directory / Qdrant collection.  Never touch a
                # version the pointer already names.
                from .publisher import read_active_version
                pointed = (read_active_version(active_version_path) or {}).get("version_id")
                if not build_result.unchanged and pointed != build_result.version_id:
                    import shutil
                    shutil.rmtree(Path(build_result.store_path).parent, ignore_errors=True)
                    if dense_backend is not None and build_result.qdrant_collection:
                        try:
                            dense_backend.delete_collection(build_result.qdrant_collection)
                        except Exception:  # noqa: BLE001
                            logger.warning("could not remove unpublished Qdrant collection")
                return

        # publish_corpus_version marks the job COMPLETED internally
        logger.info(
            "pipeline[%s]: COMPLETED — version=%s",
            job_id, build_result.version_id,
        )

    except Exception as exc:
        logger.exception("pipeline[%s]: unexpected error: %s", job_id, exc)
        _fail(f"Unexpected pipeline error: {exc}")

    finally:
        # Always clean up the temp file
        if temp_path and Path(temp_path).exists():
            try:
                os.unlink(temp_path)
                logger.debug("pipeline[%s]: cleaned up temp file %s", job_id, temp_path)
            except OSError as exc:
                logger.warning("pipeline[%s]: failed to delete temp %s: %s", job_id, temp_path, exc)


# ---------------------------------------------------------------------------
# Chunking helper (runs in executor)
# ---------------------------------------------------------------------------

def _do_chunking(
    parse_result: Any,
    job_id: str,
    doc_id: str,
    source_url: str | None,
    filename: str,
    content_hash: str,
    doc_metadata: dict[str, Any] | None = None,
    chunking_cfg: Any | None = None,
) -> list[dict]:
    from .corpus_builder import parse_result_to_corpus_document
    kwargs: dict[str, Any] = {}
    if chunking_cfg is not None:
        kwargs = {
            "chunk_size": chunking_cfg.max_tokens,
            "chunk_overlap": chunking_cfg.overlap,
        }
    return parse_result_to_corpus_document(
        parse_result=parse_result,
        job_id=job_id,
        doc_id=doc_id,
        source_url=source_url,
        filename=filename,
        content_hash=content_hash,
        doc_metadata=doc_metadata,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Source sync helper
# ---------------------------------------------------------------------------

def _ingestion_settings(settings: Any) -> Any | None:
    """settings.ingestion when `settings` is a real Settings object."""
    from rasvcx.config.settings import Settings

    return settings.ingestion if isinstance(settings, Settings) else None


async def _sync_one_source(
    source: Any,
    job_store: Any,
    active_version_path: str,
    corpus_dir: str,
    temp_dir: str,
    lease_tracker: Any,
    app_state: Any,
    settings: Any,
) -> None:
    """Fetch one source, create an ingest job if changed, run pipeline."""
    from .source_sync import fetch_source, build_ingest_job_for_source, compute_next_sync_at
    from .models import SyncState

    source_id = source.source_id
    logger.info("scheduler: syncing source %s (%s)", source_id, source.source_url)

    # Mark sync started
    job_store.mark_source_sync_started(source_id)

    fetch_result = await fetch_source(
        source_id=source_id,
        source_url=source.source_url,
        known_etag=getattr(source, "etag", None),
        known_last_modified=getattr(source, "last_modified", None),
        known_content_hash=getattr(source, "content_hash", None),
        temp_dir=temp_dir,
        settings=_ingestion_settings(settings),
    )
    interval = getattr(source, "sync_interval_seconds", None) or 3600
    job_store.update_source_fields(
        source_id, {"next_sync_at": compute_next_sync_at(int(interval))}
    )

    if fetch_result.error:
        logger.warning("scheduler: source %s fetch error: %s", source_id, fetch_result.error)
        job_store.mark_source_sync_failed(source_id, fetch_result.error)
        return

    if not fetch_result.changed:
        logger.info("scheduler: source %s unchanged — skipping", source_id)
        job_store.mark_source_sync_success(
            source_id,
            etag=fetch_result.etag,
            last_modified=fetch_result.last_modified,
            content_hash=fetch_result.content_hash,
        )
        return

    # Content changed — create job and run pipeline
    job_kwargs = build_ingest_job_for_source(
        source_id=source_id,
        source_url=source.source_url,
        fetch_result=fetch_result,
    )
    if not job_kwargs:
        return

    # Deduplicate by content hash
    served = lease_tracker.current_snapshot() if lease_tracker is not None else None
    already_indexed = (
        served.corpus_store.has_content_hash(fetch_result.content_hash)
        if served is not None and fetch_result.content_hash
        else bool(fetch_result.content_hash)
        and job_store.is_duplicate_content(fetch_result.content_hash)
    )
    if already_indexed:
        logger.info("scheduler: source %s — duplicate content hash, skip", source_id)
        job_store.mark_source_sync_success(
            source_id,
            etag=fetch_result.etag,
            last_modified=fetch_result.last_modified,
            content_hash=fetch_result.content_hash,
        )
        return

    job_id = str(uuid.uuid4())
    job_store.create_job(
        job_id=job_id,
        source_id=source_id,
        source_type="scheduled_url",
        filename=job_kwargs.get("filename", "document"),
        source_url=source.source_url,
        content_hash=fetch_result.content_hash,
    )

    job_store.mark_source_sync_success(
        source_id,
        etag=fetch_result.etag,
        last_modified=fetch_result.last_modified,
        content_hash=fetch_result.content_hash,
    )

    await run_ingestion_pipeline(
        job_id=job_id,
        temp_path=fetch_result.temp_path,
        filename=job_kwargs.get("filename", "document"),
        content_hash=fetch_result.content_hash or "",
        source_url=source.source_url,
        source_id=source_id,
        job_store=job_store,
        active_version_path=active_version_path,
        corpus_dir=corpus_dir,
        lease_tracker=lease_tracker,
        app_state=app_state,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------

async def scheduler_loop(
    job_store: Any,
    active_version_path: str,
    corpus_dir: str,
    temp_dir: str,
    lease_tracker: Any,
    app_state: Any,
    settings: Any,
    poll_interval_seconds: float = 60.0,
) -> None:
    """
    Background asyncio task. Polls for due sources every interval and fires
    _sync_one_source for each. Uses max(60s, interval/10) as check cadence.

    Runs until cancelled.
    """
    check_interval = min(max(5.0, poll_interval_seconds / 10.0), 60.0)
    logger.info(
        "scheduler_loop: started (poll_interval=%.0fs, check_interval=%.0fs)",
        poll_interval_seconds, check_interval,
    )

    while True:
        try:
            due_sources = job_store.get_due_sources()
            for source in due_sources:
                try:
                    await _sync_one_source(
                        source=source,
                        job_store=job_store,
                        active_version_path=active_version_path,
                        corpus_dir=corpus_dir,
                        temp_dir=temp_dir,
                        lease_tracker=lease_tracker,
                        app_state=app_state,
                        settings=settings,
                    )
                except Exception as exc:
                    logger.exception(
                        "scheduler_loop: error syncing source %s: %s",
                        source.source_id, exc,
                    )
        except asyncio.CancelledError:
            logger.info("scheduler_loop: cancelled")
            return
        except Exception as exc:
            logger.exception("scheduler_loop: unexpected error: %s", exc)

        try:
            await asyncio.sleep(check_interval)
        except asyncio.CancelledError:
            logger.info("scheduler_loop: cancelled during sleep")
            return