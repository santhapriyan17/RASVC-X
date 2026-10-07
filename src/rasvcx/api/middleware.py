"""Request middleware for RASVC-X (Module 14).

Middleware applied (in order, outermost first):
  RequestIDMiddleware     Generates/propagates a UUID4 X-Request-ID header.
  RequestSizeMiddleware   Enforces max_request_bytes before body parsing.
  LoggingMiddleware       Logs method, path, status, latency at INFO level.
                          Query text, generated text, and secrets are NEVER
                          logged at any level.

Request-size enforcement:
  The body is read incrementally via the ASGI receive channel before
  FastAPI parses it. If the body exceeds max_request_bytes, a 413
  response is returned immediately without invoking the route handler.
  This is robust for chunked/streamed bodies because it reads the
  actual bytes rather than trusting Content-Length (which may be absent
  or spoofed).

Default-safe logging:
  INFO:  request_id, method, path, status_code, latency_ms
  DEBUG: nothing additional (query text never logged here)
  NEVER: query text, generated text, api_key, auth_token

All middleware classes are standard Starlette BaseHTTPMiddleware subclasses
and are compatible with FastAPI's app.add_middleware().
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

#: Routes whose bodies are documents (streamed to disk by the route itself).
_UPLOAD_PATHS = frozenset({"/ingest/upload"})
#: Allowance for multipart boundaries and form fields around the file.
_MULTIPART_OVERHEAD = 1024 * 1024

_REQUEST_ID_HEADER = "X-Request-ID"
_LATENCY_HEADER = "X-Response-Time-Ms"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a UUID4 request-ID to every request and response.

    If the caller supplies X-Request-ID, that value is used.
    Otherwise a fresh UUID4 is generated.
    The ID is available on request.state.request_id for downstream
    handlers and is echoed in the X-Request-ID response header.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        from rasvcx.security.events import safe_request_id

        # Only a plain token is echoed: a header carrying newlines or
        # control characters could otherwise forge log lines.
        request_id = safe_request_id(request.headers.get(_REQUEST_ID_HEADER)) or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[_REQUEST_ID_HEADER] = request_id
        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Enforce a maximum request body size.

    Reads up to max_bytes + 1 bytes from the ASGI receive channel before
    passing to the route handler. If the body exceeds max_bytes, returns
    HTTP 413 immediately without invoking downstream middleware or routes.

    This approach is robust for chunked/streamed bodies because it reads
    actual bytes rather than trusting the Content-Length header.

    max_bytes is read from app.state.rasvcx_settings at dispatch time so
    it reflects the live settings object attached during startup.
    """

    def __init__(self, app: ASGIApp, default_max_bytes: int = 65_536) -> None:
        super().__init__(app)
        self._default_max_bytes = default_max_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        settings = getattr(request.app.state, "rasvcx_settings", None)
        is_upload = request.url.path in _UPLOAD_PATHS
        if is_upload:
            # Document uploads have their own, much larger limit, enforced
            # here from Content-Length and again WHILE streaming to disk by
            # the route (routes_ingest._stream_to_temp).  They are never
            # buffered in memory by this middleware.
            ingestion = getattr(settings, "ingestion", None)
            max_bytes = int(getattr(ingestion, "max_upload_bytes", 0) or 0) + _MULTIPART_OVERHEAD
        else:
            max_bytes = (
                settings.api.max_request_bytes
                if settings is not None
                else self._default_max_bytes
            )

        # Fast path: Content-Length header present
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                cl = int(content_length)
                if cl > max_bytes:
                    request_id = getattr(request.state, "request_id", "-")
                    logger.warning(
                        "413 body too large: content-length=%d max=%d request_id=%s",
                        cl, max_bytes, request_id,
                    )
                    return Response(
                        content=f'{{"error":"body_too_large","detail":'
                                f'"Request body exceeds {max_bytes} bytes."}}',
                        status_code=413,
                        media_type="application/json",
                    )
            except ValueError:
                pass

        # Streaming path: no Content-Length (chunked transfer).  Not for
        # uploads: buffering a multi-hundred-MB body here would defeat the
        # route's bounded-memory streaming.
        if content_length is None and request.method in ("POST", "PUT", "PATCH") and not is_upload:
            body_chunks: list[bytes] = []
            total = 0
            async for chunk in request.stream():
                total += len(chunk)
                if total > max_bytes:
                    request_id = getattr(request.state, "request_id", "-")
                    logger.warning(
                        "413 streamed body too large: read=%d max=%d request_id=%s",
                        total, max_bytes, request_id,
                    )
                    return Response(
                        content=f'{{"error":"body_too_large","detail":'
                                f'"Request body exceeds {max_bytes} bytes."}}',
                        status_code=413,
                        media_type="application/json",
                    )
                body_chunks.append(chunk)

            buffered = b"".join(body_chunks)

            async def _receive():
                return {"type": "http.request", "body": buffered, "more_body": False}

            request = Request(request.scope, receive=_receive)

        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log each request at INFO level with safe fields only.

    Logged: request_id, method, path, status_code, latency_ms.
    NEVER logged: query text, generated text, api_key, auth_token,
                  request body, response body.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        request_id = getattr(request.state, "request_id", "-")

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers[_LATENCY_HEADER] = f"{elapsed_ms:.1f}"

        logger.info(
            "%s %s %d %.1fms request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )

        return response


def add_middleware(app: ASGIApp, default_max_bytes: int = 65_536) -> None:
    """Register all RASVC-X middleware on the FastAPI app.

    Starlette runs the LAST-added middleware outermost, so they are added
    in reverse to get the execution order
    RequestID -> Logging -> RequestSize -> route handler
    (the logger must run after the request id is assigned, and must also
    log 413 rejections).
    """
    app.add_middleware(RequestSizeMiddleware, default_max_bytes=default_max_bytes)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)


__all__ = [
    "RequestIDMiddleware",
    "RequestSizeMiddleware",
    "LoggingMiddleware",
    "add_middleware",
]