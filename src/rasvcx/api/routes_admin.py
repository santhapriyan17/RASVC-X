"""Admin routes for RASVC-X (Module 14).

GET /admin/config
    Returns a sanitised snapshot of the active runtime configuration.
    Never includes secrets (api_key, auth_token).
    Includes effective M2 routing defaults for research transparency.
    Gated by verify_auth when require_auth=True.

The configuration is read-only at runtime: Settings is a frozen object
built once at startup, so there is no endpoint that toggles a module or
changes a threshold on a running process.  To change either, edit the
config file and restart.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from rasvcx.api.dependencies import (
    get_settings,
    verify_admin,
)
from rasvcx.api.models import AdminConfigResponse
from rasvcx.config.settings import Settings
from rasvcx.pipeline.factory import get_effective_routing_info

router = APIRouter(tags=["admin"])


@router.get(
    "/admin/config",
    response_model=AdminConfigResponse,
    summary="Sanitised runtime configuration snapshot",
    description=(
        "Returns the active runtime configuration without secrets. "
        "Includes effective M2 routing defaults for research transparency. "
        "Requires authentication when require_auth=True."
    ),
)
async def get_config(
    settings: Annotated[Settings, Depends(get_settings)],
    _auth: Annotated[None, Depends(verify_admin)],
) -> AdminConfigResponse:
    """Return sanitised settings snapshot.

    Secrets (api_key, auth_token) are deliberately excluded.
    Generated text is never present in config responses.
    """
    d = settings.decision
    return AdminConfigResponse(
        execution_mode=settings.execution_mode.value,
        retrieval_mode=settings.retrieval.mode,
        reranker_enabled=settings.reranker.enabled,
        nli_enabled=settings.nli.enabled,
        llm_provider=settings.llm.provider,
        llm_model_name=settings.llm.model_name,
        max_corrective_attempts=settings.pipeline.max_corrective_attempts,
        offline=settings.is_offline,
        qdrant_mode=(
            settings.retrieval.qdrant_mode if settings.retrieval.mode == "hybrid" else None
        ),
        decision_thresholds={
            "answer_min": d.answer_min,
            "warning_min": d.warning_min,
            "regenerate_min": d.regenerate_min,
            "high_risk_answer_min": d.high_risk_answer_min,
            "high_risk_warning_min": d.high_risk_warning_min,
            "high_risk_regenerate_min": d.high_risk_regenerate_min,
            "high_risk_threshold": d.high_risk_threshold,
        },
        effective_routing=get_effective_routing_info(),
    )


@router.post(
    "/admin/kb/gc",
    summary="Retention GC of superseded KB versions (dry run by default)",
)
async def kb_gc(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    _auth: Annotated[None, Depends(verify_admin)],
    dry_run: bool = True,
) -> dict[str, Any]:
    """Apply the retention policy using the LIVE lease tracker.

    A version is eligible only when it is inactive, holds no request lease,
    has been inactive for ingestion.gc_delay_seconds, and is not among the
    ingestion.keep_versions newest inactive versions.  dry_run=true (the
    default) lists what would be deleted and deletes nothing.
    """
    from rasvcx.ingestion.publisher import gc_old_versions

    tracker = request.app.state.rasvcx_lease_tracker
    runtime = getattr(request.app.state, "rasvcx_runtime", None)
    deleted = gc_old_versions(
        tracker,
        corpus_dir=settings.ingestion.corpus_versions_dir,
        dense_backend=getattr(runtime, "dense_backend", None),
        keep_versions=settings.ingestion.keep_versions,
        min_inactive_seconds=settings.ingestion.gc_delay_seconds,
        dry_run=dry_run,
    )
    return {
        "dry_run": dry_run, "current_version": tracker.current_version(),
        "eligible" if dry_run else "deleted": deleted,
        "versions": tracker.list_versions(),
        "policy": {"keep_versions": settings.ingestion.keep_versions,
                   "gc_delay_seconds": settings.ingestion.gc_delay_seconds},
    }


__all__ = ["router"]
