"""Health, readiness and runtime-status routes for RASVC-X (Module 14).

GET /health
    Liveness probe. Always returns 200 if the process is alive.
    Never requires authentication. No pipeline access.

GET /ready
    Readiness probe. Returns 200 only when every component required by
    the configured execution mode passed its probe. Returns 503 otherwise.
    Never requires authentication (readiness probes must not need auth).
    Does NOT make billable API calls during the probe.

GET /status
    Runtime status for the System Status view: mode, component probes,
    knowledge-base versions with their leases, and measured latency
    percentiles.  Requires authentication when require_auth=True.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse

from rasvcx.api.dependencies import (
    check_readiness,
    get_latency,
    get_lease_tracker,
    get_settings,
    is_ready,
    verify_auth,
)
from rasvcx.api.models import HealthResponse, ReadinessResponse
from rasvcx.config.settings import Settings

router = APIRouter(tags=["health"])

API_VERSION = "0.17.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Always returns 200 if the process is alive. "
        "Never requires authentication."
    ),
)
async def health() -> HealthResponse:
    """Liveness probe -- no state access required."""
    return HealthResponse(status="ok", version=API_VERSION)


@router.get(
    "/ready",
    summary="Readiness probe",
    description=(
        "Returns 200 when all components required by the configured "
        "execution mode are ready. Returns 503 otherwise. "
        "Never requires authentication."
    ),
)
async def ready(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> JSONResponse:
    """Readiness probe -- 200 when ready, 503 when not ready.

    verify_auth is intentionally NOT a dependency of this route so that
    readiness probes always work regardless of the require_auth setting.
    """
    components = check_readiness(request.app.state)
    ready_flag = is_ready(components)
    tracker = getattr(request.app.state, "rasvcx_lease_tracker", None)
    snapshot = tracker.current_snapshot() if tracker is not None else None

    response = ReadinessResponse(
        ready=ready_flag,
        execution_mode=settings.execution_mode.value,
        offline=settings.is_offline,
        llm_provider=settings.llm.provider,
        kb_version_id=tracker.current_version() if tracker is not None else None,
        kb_source=getattr(snapshot, "kb_source", None),
        kb_corpus_hash=getattr(snapshot, "corpus_hash", None),
        kb_doc_count=getattr(snapshot, "doc_count", None),
        kb_chunk_count=getattr(snapshot, "chunk_count", None),
        calibration=getattr(getattr(request.app.state, "rasvcx_runtime", None), "calibration", None),
        kb_warnings=list(getattr(request.app.state, "rasvcx_kb_warnings", []) or []),
        components=components,
    )

    http_status = (
        status.HTTP_200_OK if ready_flag else status.HTTP_503_SERVICE_UNAVAILABLE
    )
    return JSONResponse(
        content=response.model_dump(),
        status_code=http_status,
    )


@router.get(
    "/status",
    summary="Runtime status: components, KB versions, measured latency",
)
async def runtime_status(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    lease_tracker: Annotated[Any, Depends(get_lease_tracker)],
    latency: Annotated[Any, Depends(get_latency)],
    _auth: Annotated[None, Depends(verify_auth)],
) -> dict[str, Any]:
    from rasvcx.api.routes_query import runtime_mode

    components = check_readiness(request.app.state)
    snapshot = lease_tracker.current_snapshot()
    return {
        "ready": is_ready(components),
        "version": API_VERSION,
        "mode": runtime_mode(settings).model_dump(),
        "run_identity": (
            request.app.state.rasvcx_runtime.identity()
            if hasattr(getattr(request.app.state, "rasvcx_runtime", None), "identity") else None
        ),
        "components": [c.model_dump() for c in components],
        "knowledge_base": {
            "active": snapshot.describe() if snapshot is not None else None,
            "versions": lease_tracker.list_versions(),
        },
        "latency": latency.snapshot(),
        "process": __import__("rasvcx.api.observability", fromlist=["process_stats"]).process_stats(),
        "admission": {
            "max_workers": settings.api.max_workers,
            "in_flight": settings.api.max_workers - getattr(
                getattr(request.app.state, "rasvcx_semaphore", None), "_value", settings.api.max_workers),
        },
        "cache": (
            request.app.state.rasvcx_cache.stats()
            if getattr(request.app.state, "rasvcx_cache", None) is not None else None
        ),
    }


__all__ = ["router", "API_VERSION"]
