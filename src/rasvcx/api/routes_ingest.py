# src/rasvcx/api/routes_ingest.py
# ---------------------------------------------------------------------------
# M17 ingestion REST endpoints.
# All routes under prefix /ingest require bearer auth (via verify_auth dep).
# ---------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .dependencies import (
    get_ingest_store,
    get_lease_tracker,
    get_settings,
    ingest_temp_dir,
    rate_limit,
    verify_admin,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class IngestUrlRequest(BaseModel):
    url: str


class CreateSourceRequest(BaseModel):
    display_name: str
    source_url: str
    sync_interval_seconds: int = 3600


# ---------------------------------------------------------------------------
# Auth dependency alias
# ---------------------------------------------------------------------------

# Ingestion writes to the knowledge base: admin credential required
# (RASVCX_ADMIN_TOKEN when set, else the auth token), plus rate limiting.
AuthDep = Annotated[None, Depends(verify_admin)]
RateDep = Annotated[None, Depends(rate_limit)]

# asyncio only keeps weak references to tasks; holding them here stops a
# running ingestion job from being garbage-collected mid-flight.
_BACKGROUND_TASKS: set[asyncio.Task] = set()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temp_dir(settings: Any) -> str:
    d = ingest_temp_dir(settings)
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


def _corpus_dir(settings: Any) -> str:
    return settings.ingestion.corpus_versions_dir


def _active_version_path(settings: Any) -> str:
    return settings.ingestion.active_version_path


def _max_upload_bytes(settings: Any) -> int:
    return int(settings.ingestion.max_upload_bytes)


_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._ ()\-]")


def safe_filename(raw: str | None) -> str:
    """Reduce a client-supplied filename to a harmless display name.

    The uploaded bytes are always stored under a server-generated temp
    name, so this value is never used as a path.  It is still sanitised
    because it is persisted, logged and shown in the UI: directory parts,
    control characters and anything outside a conservative character set
    are removed, and the length is bounded.
    """
    name = (raw or "").replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable())
    name = _FILENAME_SAFE.sub("_", name).strip(" .")
    if len(name) > 150:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 150 - len(ext) - 1] + dot + ext) if dot and len(ext) <= 10 else name[:150]
    return name or "upload"


async def _stream_to_temp(upload: UploadFile, temp_dir: str, max_bytes: int) -> tuple[str, str, int]:
    """Stream UploadFile to a temp file. Returns (path, sha256_hex, size)."""
    import aiofiles

    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=temp_dir, suffix=".upload")
    hasher = hashlib.sha256()
    total = 0

    try:
        import os as _os
        with _os.fdopen(fd, "wb") as fout:
            while True:
                chunk = await upload.read(65536)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"Upload exceeds {max_bytes // 1024 // 1024} MB limit",
                    )
                fout.write(chunk)
                hasher.update(chunk)
    except HTTPException:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise
    except Exception as exc:
        try:
            _os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload stream error: {exc}",
        ) from exc

    return tmp_path, hasher.hexdigest(), total


def _fire_pipeline(
    request: Request,
    job_id: str,
    temp_path: str,
    filename: str,
    content_hash: str,
    source_url: str | None,
    source_id: str | None,
    job_store: Any,
    settings: Any,
    doc_metadata: dict[str, Any] | None = None,
) -> None:
    """Launch run_ingestion_pipeline as a background asyncio task."""
    from rasvcx.ingestion.scheduler import run_ingestion_pipeline

    corpus_dir  = _corpus_dir(settings)
    active_path = _active_version_path(settings)
    lease_tracker = getattr(request.app.state, "rasvcx_lease_tracker", None)

    task = asyncio.create_task(
        run_ingestion_pipeline(
            job_id=job_id,
            temp_path=temp_path,
            filename=filename,
            content_hash=content_hash,
            source_url=source_url,
            source_id=source_id,
            job_store=job_store,
            active_version_path=active_path,
            corpus_dir=corpus_dir,
            lease_tracker=lease_tracker,
            app_state=request.app.state,
            settings=settings,
            doc_metadata=doc_metadata,
        ),
        name=f"pipeline-{job_id[:8]}",
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


# ---------------------------------------------------------------------------
# POST /ingest/upload
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    request: Request,
    _auth: AuthDep,
    _rate: RateDep,
    file: UploadFile,
    job_store: Annotated[Any, Depends(get_ingest_store)],
    settings: Annotated[Any, Depends(get_settings)],
    title: Annotated[str | None, Form()] = None,
    source_type: Annotated[str | None, Form()] = None,
    date: Annotated[str | None, Form()] = None,
    jurisdiction: Annotated[str | None, Form()] = None,
    population: Annotated[str | None, Form()] = None,
    dosage_context: Annotated[str | None, Form()] = None,
    status_: Annotated[str | None, Form(alias="status")] = None,
    effective_date: Annotated[str | None, Form()] = None,
    superseded_by: Annotated[str | None, Form()] = None,
    supersedes: Annotated[str | None, Form()] = None,
    version: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    """
    Stream-upload a document for async ingestion.
    Returns 202 immediately; indexing runs in background.
    `duplicate` is true when identical content WITH identical metadata was
    already indexed -- re-uploading a file to change its metadata (for
    example to mark it withdrawn) is a new version, not a duplicate.

    Optional form fields record the document's provenance and declared
    lifecycle (status: current | superseded | historical | withdrawn;
    supersedes: comma-separated doc_ids).  Anything not supplied is stored
    as unknown -- it is never inferred from the text.
    """
    from rasvcx.ingestion.corpus_builder import (
        CorpusBuildError,
        document_metadata_signature,
        normalise_lifecycle,
        normalise_provenance,
    )

    filename = safe_filename(file.filename)
    doc_metadata = {
        "title": title, "source_type": source_type, "date": date,
        "jurisdiction": jurisdiction, "population": population,
        "dosage_context": dosage_context, "status": status_,
        "effective_date": effective_date, "superseded_by": superseded_by,
        "supersedes": supersedes, "version": version,
    }
    doc_metadata = {k: v.strip() for k, v in doc_metadata.items() if v and v.strip()}
    for key, value in doc_metadata.items():
        if len(value) > 256:
            raise HTTPException(status_code=422, detail=f"{key} exceeds 256 characters")
    try:
        normalise_provenance(doc_metadata)
        normalise_lifecycle(doc_metadata)
    except CorpusBuildError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    tmp_path, content_hash, size = await _stream_to_temp(
        file, _temp_dir(settings), _max_upload_bytes(settings),
    )

    # Duplicate detection: is this exact file already in the knowledge base
    # that is being SERVED?  (Job history alone is not the answer: a
    # document ingested into a version that was later replaced or removed
    # must be ingestible again.)
    lease_tracker = getattr(request.app.state, "rasvcx_lease_tracker", None)
    snapshot = lease_tracker.current_snapshot() if lease_tracker is not None else None
    if snapshot is not None:
        is_dup = document_metadata_signature(doc_metadata, filename) in (
            snapshot.corpus_store.metadata_signatures(content_hash)
        )
    else:
        is_dup = job_store.is_duplicate_content(content_hash)

    job_id = str(uuid.uuid4())
    job_store.create_job(
        job_id=job_id,
        source_id=None,
        source_type="file_upload",
        filename=filename,
        source_url=None,
        content_hash=content_hash,
    )

    if not is_dup:
        _fire_pipeline(
            request=request,
            job_id=job_id,
            temp_path=tmp_path,
            filename=filename,
            content_hash=content_hash,
            source_url=None,
            source_id=None,
            job_store=job_store,
            settings=settings,
            doc_metadata=doc_metadata,
        )
    else:
        # Clean up temp — no processing needed
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        job_store.update_job_status(job_id, "completed")

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "filename": filename,
            "size_bytes": size,
            "message": "Queued for ingestion" if not is_dup else "Duplicate content — already indexed",
            "duplicate": is_dup,
        },
    )


# ---------------------------------------------------------------------------
# POST /ingest/url
# ---------------------------------------------------------------------------

@router.post("/url", status_code=status.HTTP_202_ACCEPTED)
async def ingest_url(
    request: Request,
    _auth: AuthDep,
    _rate: RateDep,
    body: IngestUrlRequest,
    job_store: Annotated[Any, Depends(get_ingest_store)],
    settings: Annotated[Any, Depends(get_settings)],
) -> JSONResponse:
    """Submit a URL for async ingestion."""
    from rasvcx.ingestion.security import validate_url, SecurityError

    url = body.url.strip()

    try:
        validate_url(url, settings.ingestion)
    except SecurityError as exc:
        from rasvcx.security.events import security_event
        security_event("ssrf_rejected", path=request.url.path,
                       request_id=getattr(request.state, "request_id", None),
                       reason=type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"URL rejected: {exc}",
        ) from exc

    if job_store.source_url_exists(url):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A source with this URL already exists.",
        )

    job_id = str(uuid.uuid4())
    filename = safe_filename(urlparse(url).path.rsplit("/", 1)[-1] or "document")

    job_store.create_job(
        job_id=job_id,
        source_id=None,
        source_type="url",
        filename=filename,
        source_url=url,
        content_hash=None,
    )

    # Fetch + pipeline runs async
    from rasvcx.ingestion.source_sync import fetch_source

    async def _fetch_and_run() -> None:
        fetch_result = await fetch_source(
            source_id=job_id,
            source_url=url,
            known_etag=None,
            known_last_modified=None,
            known_content_hash=None,
            temp_dir=_temp_dir(settings),
            settings=settings.ingestion,
        )
        if fetch_result.error or not fetch_result.changed or fetch_result.temp_path is None:
            job_store.update_job_status(
                job_id, "failed",
                error_message=fetch_result.error or "URL fetch returned no content",
            )
            return

        await run_ingestion_pipeline_task(
            request=request,
            job_id=job_id,
            temp_path=fetch_result.temp_path,
            filename=filename,
            content_hash=fetch_result.content_hash or "",
            source_url=url,
            source_id=None,
            job_store=job_store,
            settings=settings,
        )

    task = asyncio.create_task(_fetch_and_run(), name=f"url-ingest-{job_id[:8]}")
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "job_id": job_id,
            "message": "URL queued for ingestion",
            "duplicate": False,
        },
    )


async def run_ingestion_pipeline_task(
    request: Request,
    job_id: str,
    temp_path: str,
    filename: str,
    content_hash: str,
    source_url: str | None,
    source_id: str | None,
    job_store: Any,
    settings: Any,
) -> None:
    from rasvcx.ingestion.scheduler import run_ingestion_pipeline
    corpus_dir  = _corpus_dir(settings)
    active_path = _active_version_path(settings)
    lease_tracker = getattr(request.app.state, "rasvcx_lease_tracker", None)

    await run_ingestion_pipeline(
        job_id=job_id,
        temp_path=temp_path,
        filename=filename,
        content_hash=content_hash,
        source_url=source_url,
        source_id=source_id,
        job_store=job_store,
        active_version_path=active_path,
        corpus_dir=corpus_dir,
        lease_tracker=lease_tracker,
        app_state=request.app.state,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# GET /ingest/jobs
# ---------------------------------------------------------------------------

@router.get("/jobs")
async def list_jobs(
    _auth: AuthDep,
    job_store: Annotated[Any, Depends(get_ingest_store)],
    status_filter: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    jobs = job_store.list_jobs(
        status_filter=status_filter,
        limit=limit,
        offset=offset,
    )
    return {"jobs": [j.__dict__ for j in jobs], "total": len(jobs)}


# ---------------------------------------------------------------------------
# GET /ingest/jobs/{job_id}
# ---------------------------------------------------------------------------

@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    _auth: AuthDep,
    job_store: Annotated[Any, Depends(get_ingest_store)],
) -> dict:
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.__dict__


# ---------------------------------------------------------------------------
# DELETE /ingest/jobs/{job_id}  (cancel)
# ---------------------------------------------------------------------------

@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_job_route(
    job_id: str,
    _auth: AuthDep,
    job_store: Annotated[Any, Depends(get_ingest_store)],
) -> None:
    ok = job_store.cancel_job(job_id)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job cannot be cancelled at its current stage",
        )


# ---------------------------------------------------------------------------
# GET /ingest/sources
# ---------------------------------------------------------------------------

@router.get("/sources")
async def list_sources(
    _auth: AuthDep,
    job_store: Annotated[Any, Depends(get_ingest_store)],
) -> dict:
    sources = job_store.list_sources()
    return {"sources": [s.__dict__ for s in sources]}


# ---------------------------------------------------------------------------
# POST /ingest/sources
# ---------------------------------------------------------------------------

@router.post("/sources", status_code=status.HTTP_201_CREATED)
async def create_source(
    _auth: AuthDep,
    body: CreateSourceRequest,
    job_store: Annotated[Any, Depends(get_ingest_store)],
    settings: Annotated[Any, Depends(get_settings)],
) -> dict:
    from rasvcx.ingestion.security import validate_url, SecurityError

    try:
        validate_url(body.source_url, settings.ingestion)
    except SecurityError as exc:
        raise HTTPException(status_code=400, detail=f"URL rejected: {exc}") from exc
    if body.sync_interval_seconds < 60:
        raise HTTPException(status_code=422, detail="sync_interval_seconds must be >= 60")

    if job_store.source_url_exists(body.source_url):
        raise HTTPException(status_code=409, detail="Source URL already registered")

    source_id = str(uuid.uuid4())
    source = job_store.create_source(
        source_id=source_id,
        display_name=body.display_name,
        source_url=body.source_url,
        sync_interval_seconds=body.sync_interval_seconds,
    )
    # Schedule the first sync immediately; without next_sync_at the
    # scheduler would never pick the source up.
    from rasvcx.ingestion.source_sync import compute_next_sync_at
    job_store.update_source_fields(source_id, {"next_sync_at": compute_next_sync_at(0)})
    return job_store.get_source(source_id).__dict__


# ---------------------------------------------------------------------------
# POST /ingest/sources/{source_id}/sync  (manual trigger)
# ---------------------------------------------------------------------------

@router.post("/sources/{source_id}/sync", status_code=status.HTTP_202_ACCEPTED)
async def sync_source(
    request: Request,
    source_id: str,
    _auth: AuthDep,
    job_store: Annotated[Any, Depends(get_ingest_store)],
    settings: Annotated[Any, Depends(get_settings)],
) -> dict:
    source = job_store.get_source(source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")

    job_id = str(uuid.uuid4())

    from rasvcx.ingestion.scheduler import _sync_one_source
    corpus_dir  = _corpus_dir(settings)
    active_path = _active_version_path(settings)
    lease_tracker = getattr(request.app.state, "rasvcx_lease_tracker", None)

    task = asyncio.create_task(
        _sync_one_source(
            source=source,
            job_store=job_store,
            active_version_path=active_path,
            corpus_dir=corpus_dir,
            temp_dir=_temp_dir(settings),
            lease_tracker=lease_tracker,
            app_state=request.app.state,
            settings=settings,
        ),
        name=f"manual-sync-{source_id[:8]}",
    )
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

    return {"source_id": source_id, "message": "Sync triggered"}


# ---------------------------------------------------------------------------
# DELETE /ingest/sources/{source_id}
# ---------------------------------------------------------------------------

@router.delete("/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(
    source_id: str,
    _auth: AuthDep,
    job_store: Annotated[Any, Depends(get_ingest_store)],
) -> None:
    ok = job_store.delete_source(source_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Source not found")


# ---------------------------------------------------------------------------
# GET /ingest/status
# ---------------------------------------------------------------------------

@router.get("/status")
async def ingest_status(
    request: Request,
    _auth: AuthDep,
    job_store: Annotated[Any, Depends(get_ingest_store)],
    settings: Annotated[Any, Depends(get_settings)],
    lease_tracker: Annotated[Any, Depends(get_lease_tracker)],
) -> dict:
    """Knowledge-base status.

    `active_version` describes the version this process is serving to
    /query (from the lease tracker), not merely what a file on disk says.
    `pointer_version_id` is the durable pointer; the two differ only if
    something is wrong, and `consistent` says so explicitly.
    """
    from rasvcx.ingestion.publisher import read_active_version

    pointer = read_active_version(_active_version_path(settings))
    snapshot = lease_tracker.current_snapshot()
    serving = lease_tracker.current_version()

    active: dict[str, Any] | None = None
    if snapshot is not None:
        active = snapshot.describe()
        if pointer and pointer.get("version_id") == snapshot.version_id:
            active.update({
                "store_path": pointer.get("store_path"),
                "bm25_path": pointer.get("bm25_path"),
                "manifest_path": pointer.get("manifest_path"),
            })
    elif pointer:
        active = dict(pointer)

    counts = job_store.count_jobs_by_status()
    sources = job_store.list_sources()

    return {
        "active_version": active,
        "serving_version_id": serving,
        "pointer_version_id": pointer.get("version_id") if pointer else None,
        # No pointer yet == the bootstrap (seed) version is being served.
        "consistent": pointer is None or pointer.get("version_id") == serving,
        "versions": lease_tracker.list_versions(),
        "pending_jobs": counts.get("queued", 0),
        "active_sources": sum(1 for s in sources if s.is_active),
        "failed_jobs": counts.get("failed", 0),
    }
