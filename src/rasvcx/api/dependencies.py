# src/rasvcx/api/dependencies.py
# ---------------------------------------------------------------------------
# FastAPI dependency injection + application lifecycle.
#
# startup() builds the query runtime (pipeline/factory.build_runtime) and
# registers its initial knowledge-base snapshot with the lease tracker.
# From then on the lease tracker is the single source of truth for which
# KB version is served: /query leases from it, ingestion publishes into it.
# ---------------------------------------------------------------------------

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Request

logger = logging.getLogger(__name__)

_KEY_PIPELINE       = "rasvcx_pipeline"
_KEY_RUNTIME        = "rasvcx_runtime"
_KEY_SETTINGS       = "rasvcx_settings"
_KEY_INGEST_STORE   = "rasvcx_ingest_store"
_KEY_LEASE_TRACKER  = "rasvcx_lease_tracker"
_KEY_SCHEDULER_TASK = "rasvcx_scheduler_task"
_KEY_EXECUTOR       = "rasvcx_executor"
_KEY_SEMAPHORE      = "rasvcx_semaphore"
_KEY_LATENCY        = "rasvcx_latency"


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

def startup(app: Any) -> None:
    """Build the runtime, ingest store, lease tracker, executor + semaphore.

    Any failure here propagates and aborts application startup: a process
    that cannot build its configured pipeline must not come up and serve.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    from rasvcx.api.observability import LatencyRecorder
    from rasvcx.config.loader import load_settings
    from rasvcx.ingestion.job_store import IngestJobStore
    from rasvcx.ingestion.publisher import (
        KBVersionLeaseTracker,
        reconcile_publishing_jobs_on_startup,
    )
    from rasvcx.pipeline.factory import build_runtime

    settings = getattr(app.state, "_settings_override", None) or load_settings()
    app.state.rasvcx_settings = settings

    logger.info("startup: building pipeline (mode=%s)", settings.execution_mode.value)
    runtime = build_runtime(settings)
    app.state.rasvcx_runtime = runtime
    app.state.rasvcx_pipeline = runtime.orchestrator
    logger.info("startup: pipeline ready")

    lease_tracker = KBVersionLeaseTracker()
    snapshot = runtime.initial_snapshot
    lease_tracker.register(
        snapshot.version_id, snapshot=snapshot, published_at=snapshot.published_at
    )
    _register_superseded_versions(lease_tracker, settings, snapshot.version_id)
    app.state.rasvcx_lease_tracker = lease_tracker
    app.state.rasvcx_kb_warnings = kb_configuration_warnings(settings, snapshot)
    for w in app.state.rasvcx_kb_warnings:
        logger.warning("startup: %s", w)
    if settings.ingestion.require_published_kb and snapshot.kb_source != "published_kb":
        from rasvcx.config.settings import ConfigurationError
        raise ConfigurationError(
            f"ingestion.require_published_kb is set but no published KB version exists at "
            f"{settings.ingestion.active_version_path} (would serve {snapshot.kb_source} "
            f"{snapshot.version_id})"
        )
    logger.info(
        "startup: serving KB version %s (%d chunks, %d docs)",
        snapshot.version_id, snapshot.chunk_count, snapshot.doc_count,
    )

    db_path = settings.ingestion.db_path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    job_store = IngestJobStore(db_path=db_path)
    app.state.rasvcx_ingest_store = job_store
    logger.info("startup: ingest store ready at %s", db_path)

    # One worker per admitted request: the semaphore is the admission
    # control, the executor runs the synchronous pipeline off the loop.
    max_workers = settings.api.max_workers
    app.state.rasvcx_executor = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="rasvcx-query"
    )
    app.state.rasvcx_semaphore = asyncio.Semaphore(max_workers)
    app.state.rasvcx_latency = LatencyRecorder()
    from rasvcx.caching.cache import ResponseCache
    app.state.rasvcx_cache = ResponseCache(settings.api.cache_size, settings.api.cache_ttl_seconds)
    from rasvcx.security.rate_limit import TokenBucketLimiter
    app.state.rasvcx_rate_limiter = TokenBucketLimiter(settings.api.rate_limit_per_minute)
    logger.info("startup: query executor + semaphore ready (workers=%d)", max_workers)

    interrupted = job_store.recover_interrupted_jobs()
    if interrupted:
        logger.warning("startup: marked %d interrupted ingestion job(s) failed", interrupted)
    purged = _purge_orphaned_temp_files(ingest_temp_dir(settings))
    if purged:
        logger.info("startup: removed %d orphaned ingestion temp file(s)", purged)
    reconcile_publishing_jobs_on_startup(job_store, settings.ingestion.active_version_path)


def _purge_orphaned_temp_files(temp_dir: str) -> int:
    """Delete upload/download temp files left by a previous process.

    Ingestion jobs do not survive a restart (they are marked failed above),
    so at startup every *.upload / *.download file in the ingestion temp
    directory belongs to a job that can never resume.  Only those two
    suffixes, directly inside that directory, are removed.
    """
    directory = Path(temp_dir)
    if not directory.is_dir():
        return 0
    removed = 0
    for entry in directory.iterdir():
        if entry.is_file() and entry.suffix in (".upload", ".download"):
            try:
                entry.unlink()
                removed += 1
            except OSError:
                logger.warning("startup: could not remove temp file %s", entry.name)
    return removed


def _register_superseded_versions(lease_tracker: Any, settings: Any, current: str) -> None:
    """Tell the tracker about older version dirs on disk so retention GC
    (not a blind startup sweep) decides when they are removed."""
    from rasvcx.ingestion.publisher import read_active_version

    versions_dir = Path(settings.ingestion.corpus_versions_dir)
    if not versions_dir.is_dir():
        return
    import time as _time

    for entry in versions_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("v_") or entry.name == current:
            continue
        manifest = read_active_version(str(entry / "manifest.json")) or {}
        age = max(0.0, _time.time() - entry.stat().st_mtime)
        lease_tracker.register_inactive(
            entry.name, published_at=manifest.get("published_at"), inactive_for_seconds=age,
        )


def kb_configuration_warnings(settings: Any, snapshot: Any) -> list[str]:
    """Configuration states an operator must see (reported by /ready).

    - A pointer file exists in the versions directory but is not the
      configured pointer: a previous configuration published there and the
      current one will never load it.
    - The seed corpus is being served (no published version).
    """
    warnings: list[str] = []
    configured = Path(settings.ingestion.active_version_path).resolve()
    stray = (Path(settings.ingestion.corpus_versions_dir) / "active_version.json").resolve()
    if stray != configured and stray.exists():
        warnings.append(
            f"a KB pointer exists at {stray} but the configured pointer is {configured}; "
            "the stray pointer is ignored (check RASVCX_ACTIVE_VERSION_PATH / RASVCX_CORPUS_DIR)"
        )
    if getattr(snapshot, "kb_source", None) != "published_kb":
        warnings.append(
            f"no published KB version at {configured}: serving the "
            f"{getattr(snapshot, 'kb_source', 'unknown')} corpus {getattr(snapshot, 'version_id', '?')}"
        )
    return warnings


async def start_background_tasks(app: Any) -> None:
    """Launch the source-sync scheduler loop as a background task."""
    import asyncio

    from rasvcx.ingestion.scheduler import scheduler_loop

    settings = app.state.rasvcx_settings
    interval = settings.ingestion.scheduled_sync_interval_seconds
    if interval <= 0:
        app.state.rasvcx_scheduler_task = None
        logger.info("start_background_tasks: scheduled source sync disabled")
        return

    app.state.rasvcx_scheduler_task = asyncio.create_task(
        scheduler_loop(
            job_store=app.state.rasvcx_ingest_store,
            active_version_path=settings.ingestion.active_version_path,
            corpus_dir=settings.ingestion.corpus_versions_dir,
            temp_dir=ingest_temp_dir(settings),
            lease_tracker=app.state.rasvcx_lease_tracker,
            app_state=app.state,
            settings=settings,
            poll_interval_seconds=interval,
        ),
        name="rasvcx-scheduler",
    )
    logger.info("start_background_tasks: scheduler started (interval=%.0fs)", interval)


async def stop_background_tasks(app: Any) -> None:
    """Cancel and await the scheduler task."""
    import asyncio

    task = getattr(app.state, _KEY_SCHEDULER_TASK, None)
    if task and not task.done():
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
    logger.info("stop_background_tasks: scheduler stopped")


def shutdown(app: Any) -> None:
    executor = getattr(app.state, _KEY_EXECUTOR, None)
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
    logger.info("shutdown: complete")


def ingest_temp_dir(settings: Any) -> str:
    return settings.ingestion.temp_dir or ".runtime/ingestion_tmp"


# ---------------------------------------------------------------------------
# Readiness -- every state below is measured, none is assumed
# ---------------------------------------------------------------------------

def check_readiness(app_state: Any) -> list:
    """Probe each component of the running process.

    No billable provider call is made: the LLM is reported as `configured`
    (client constructed, key present) -- its reachability is only proven
    by a real generation, which /query reports per request.
    """
    from rasvcx.api.models import ComponentState, ComponentStatus
    from rasvcx.generation.llm_client import MockLLMClient
    from rasvcx.retrieval.bridge import DisabledRerankingService
    from rasvcx.validation import NullNLIBackend

    runtime = getattr(app_state, _KEY_RUNTIME, None)
    tracker = getattr(app_state, _KEY_LEASE_TRACKER, None)
    comps: list[ComponentStatus] = []
    if runtime is None or tracker is None:
        return [ComponentStatus(
            name="pipeline", state=ComponentState.UNAVAILABLE, detail="runtime not built",
        )]

    snapshot = tracker.current_snapshot()
    if snapshot is None:
        return [ComponentStatus(
            name="knowledge_base", state=ComponentState.UNAVAILABLE,
            detail="no knowledge-base version is loaded",
        )]
    comps.append(ComponentStatus(
        name="knowledge_base", state=ComponentState.LOADED,
        detail=f"version={snapshot.version_id} chunks={snapshot.chunk_count} docs={snapshot.doc_count}",
    ))
    comps.append(ComponentStatus(
        name="bm25_index", state=ComponentState.LOADED,
        detail=f"documents={snapshot.bm25_index.size}",
    ))

    # Pointer, tracker, snapshot, BM25 and store must name ONE version.
    from rasvcx.ingestion.publisher import read_active_version

    problems = []
    if tracker.current_version() != snapshot.version_id:
        problems.append(f"tracker={tracker.current_version()} snapshot={snapshot.version_id}")
    if snapshot.kb_source == "published_kb":
        pointer = (read_active_version(runtime.settings.ingestion.active_version_path) or {}).get("version_id")
        if pointer != snapshot.version_id:
            problems.append(f"pointer={pointer} snapshot={snapshot.version_id}")
    if snapshot.bm25_index.size != snapshot.chunk_count:
        problems.append(f"bm25={snapshot.bm25_index.size} store={snapshot.chunk_count}")
    comps.append(ComponentStatus(
        name="kb_consistency",
        state=ComponentState.UNAVAILABLE if problems else ComponentState.LOADED,
        detail="; ".join(problems) or (
            f"version={snapshot.version_id} source={snapshot.kb_source} "
            f"identity={snapshot.identity_scheme}"
        ),
    ))

    if runtime.settings.retrieval.mode == "hybrid":
        try:
            count = snapshot.dense_retriever.count()
            if count == snapshot.chunk_count:
                comps.append(ComponentStatus(
                    name="dense_index", state=ComponentState.REACHABLE,
                    detail=(
                        f"qdrant_mode={runtime.dense_backend.mode} "
                        f"collection={snapshot.qdrant_collection} points={count}"
                    ),
                ))
            else:
                comps.append(ComponentStatus(
                    name="dense_index", state=ComponentState.UNAVAILABLE,
                    detail=(
                        f"collection={snapshot.qdrant_collection} has {count} points, "
                        f"expected {snapshot.chunk_count}"
                    ),
                ))
        except Exception as exc:  # noqa: BLE001 - a failed probe IS the answer
            comps.append(ComponentStatus(
                name="dense_index", state=ComponentState.UNAVAILABLE,
                detail=f"{type(exc).__name__}: {exc}",
            ))
    else:
        comps.append(ComponentStatus(
            name="dense_index", state=ComponentState.DISABLED,
            detail="retrieval.mode=bm25_only",
        ))

    reranker = runtime.reranking_service
    if isinstance(reranker, DisabledRerankingService):
        comps.append(ComponentStatus(
            name="reranker", state=ComponentState.DISABLED, detail="offline_test: no reranking",
        ))
    else:
        comps.append(ComponentStatus(
            name="reranker", state=ComponentState.LOADED,
            detail=f"model={reranker.config.cross_encoder.model_name}",
        ))

    backend = runtime.nli_service.backend
    if backend is None or isinstance(backend, NullNLIBackend):
        comps.append(ComponentStatus(
            name="nli", state=ComponentState.DISABLED, detail="offline_test: no NLI model",
        ))
    elif getattr(backend, "is_loaded", False):
        comps.append(ComponentStatus(
            name="nli", state=ComponentState.LOADED,
            detail=f"model={getattr(backend, 'model_name', '?')}",
        ))
    else:
        comps.append(ComponentStatus(
            name="nli", state=ComponentState.UNAVAILABLE, detail="model not loaded",
        ))

    client = runtime.llm_client
    if isinstance(client, MockLLMClient):
        comps.append(ComponentStatus(
            name="llm", state=ComponentState.STUB,
            detail="offline_test: deterministic stub, not a real model",
        ))
    elif getattr(client, "is_configured", lambda: False)():
        comps.append(ComponentStatus(
            name="llm", state=ComponentState.CONFIGURED,
            detail=f"provider=gemini model={runtime.settings.llm.model_name} (not probed)",
        ))
    else:
        comps.append(ComponentStatus(
            name="llm", state=ComponentState.UNAVAILABLE, detail="provider client not configured",
        ))

    return comps


def is_ready(components: list) -> bool:
    """Ready unless any component is UNAVAILABLE."""
    from rasvcx.api.models import ComponentState

    if not components:
        return False
    return all(c.state != ComponentState.UNAVAILABLE for c in components)


# ---------------------------------------------------------------------------
# FastAPI dependency getters
# ---------------------------------------------------------------------------

def _require(request: Request, key: str, what: str) -> Any:
    value = getattr(request.app.state, key, None)
    if value is None:
        raise RuntimeError(f"{what} not initialized")
    return value


def get_pipeline(request: Request) -> Any:
    return _require(request, _KEY_PIPELINE, "Pipeline")


def get_runtime(request: Request) -> Any:
    return _require(request, _KEY_RUNTIME, "Runtime")


def get_settings(request: Request) -> Any:
    return _require(request, _KEY_SETTINGS, "Settings")


def get_ingest_store(request: Request) -> Any:
    return _require(request, _KEY_INGEST_STORE, "Ingest store")


def get_lease_tracker(request: Request) -> Any:
    return _require(request, _KEY_LEASE_TRACKER, "Lease tracker")


def get_executor(request: Request) -> Any:
    return _require(request, _KEY_EXECUTOR, "Executor")


def get_semaphore(request: Request) -> Any:
    return _require(request, _KEY_SEMAPHORE, "Semaphore")


def get_latency(request: Request) -> Any:
    return _require(request, _KEY_LATENCY, "Latency recorder")


# ---------------------------------------------------------------------------
# Auth helper — settings-driven (settings.api.require_auth / auth_token)
# ---------------------------------------------------------------------------

def _bearer(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    return auth[7:].strip() if auth.lower().startswith("bearer ") else None


def _check_token(request: Request, accepted: list[str], scope: str) -> None:
    import hmac

    from fastapi import HTTPException, status

    from rasvcx.security.events import pseudonym, security_event

    supplied = _bearer(request)
    rid = getattr(request.state, "request_id", None)
    if supplied is None:
        security_event("auth_failure", reason="missing_bearer", scope=scope,
                       path=request.url.path, request_id=rid)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing or malformed Authorization header")
    if not any(t and hmac.compare_digest(supplied, t) for t in accepted):
        security_event("auth_failure", reason="invalid_token", scope=scope,
                       path=request.url.path, request_id=rid, principal=pseudonym(supplied))
        status_code = status.HTTP_403_FORBIDDEN if scope == "admin" and any(
            t and hmac.compare_digest(supplied, t) for t in accepted_query_tokens(request)
        ) else status.HTTP_401_UNAUTHORIZED
        raise HTTPException(status_code=status_code,
                            detail="Invalid token" if status_code == 401 else "Admin token required")


def accepted_query_tokens(request: Request) -> list[str]:
    api = getattr(getattr(request.app.state, _KEY_SETTINGS, None), "api", None)
    if api is None:
        return []
    return [t for t in {(api.auth_token or "").strip(), (api.admin_token or "").strip()} if t]


def verify_auth(request: Request) -> None:
    """Bearer-token auth for query / feedback / status (health and ready
    omit it).  Accepts the auth token or the admin token."""
    api = getattr(getattr(request.app.state, _KEY_SETTINGS, None), "api", None)
    if api is None or not api.require_auth:
        return
    _check_token(request, accepted_query_tokens(request), "query")


def verify_admin(request: Request) -> None:
    """Auth for knowledge-base writes (ingestion) and /admin.

    When RASVCX_ADMIN_TOKEN is set, ONLY that token is accepted here; a
    query token gets 403.  Without it the auth token is also the admin
    token (single-credential deployments)."""
    api = getattr(getattr(request.app.state, _KEY_SETTINGS, None), "api", None)
    if api is None or not api.require_auth:
        return
    _check_token(request, [(api.admin_token or "").strip()], "admin")


def rate_limit(request: Request) -> None:
    """Per-principal token bucket (api.rate_limit_per_minute; 0 = off)."""
    from fastapi import HTTPException

    from rasvcx.security.events import pseudonym, security_event

    limiter = getattr(request.app.state, "rasvcx_rate_limiter", None)
    if limiter is None or not limiter.enabled:
        return
    token = _bearer(request)
    principal = ("t:" + (pseudonym(token) or "")) if token else (
        "ip:" + (pseudonym(request.client.host if request.client else "unknown") or ""))
    wait = limiter.acquire(principal)
    if wait > 0:
        security_event("rate_limited", principal=principal, path=request.url.path,
                       retry_after_s=round(wait, 1),
                       request_id=getattr(request.state, "request_id", None))
        raise HTTPException(status_code=429, detail="Rate limit exceeded",
                            headers={"Retry-After": str(max(1, int(wait + 0.999)))})
