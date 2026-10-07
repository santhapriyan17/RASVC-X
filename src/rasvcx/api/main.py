"""FastAPI application factory for RASVC-X.

`create_app(settings)` builds an application for an explicit Settings
object.  The module-level `app` (used by `uvicorn rasvcx.api.main:app`)
resolves its settings from the environment when the server starts -- see
rasvcx.config.loader.resolve_config_path.  `python -m rasvcx` is the
recommended entrypoint: it also loads `.env`.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI

from rasvcx.api.dependencies import (
    shutdown,
    start_background_tasks,
    startup,
    stop_background_tasks,
)
from rasvcx.api.middleware import add_middleware
from rasvcx.api.routes_admin import router as admin_router
from rasvcx.api.routes_eval import router as eval_router
from rasvcx.api.routes_feedback import router as feedback_router
from rasvcx.api.routes_health import API_VERSION, router as health_router
from rasvcx.api.routes_ingest import router as ingest_router
from rasvcx.api.routes_query import router as query_router

logger = logging.getLogger(__name__)


def create_app(settings: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        startup(app)
        await start_background_tasks(app)
        try:
            yield
        finally:
            await stop_background_tasks(app)
            shutdown(app)

    app = FastAPI(
        title="RASVC-X",
        version=API_VERSION,
        description="Real-Time Medical RAG Evidence Validation System",
        lifespan=lifespan,
    )

    if settings is not None:
        app.state._settings_override = settings

    add_middleware(app)

    app.include_router(health_router)
    app.include_router(admin_router)
    app.include_router(query_router)
    app.include_router(feedback_router)
    app.include_router(eval_router)
    app.include_router(ingest_router)

    # Serve the built chat UI when present.  Mounted last so it never
    # shadows an API route.
    _dist = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "dist")
    if os.path.isdir(_dist):
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")

    return app


app = create_app()
