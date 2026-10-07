"""Query route for RASVC-X (Module 14).

POST /query
    Accepts a natural-language query, runs it through the full
    RASVC-X pipeline, and returns a structured response.

Request lifecycle:
  1. Validate the request (length; body size is enforced by middleware).
  2. Bounded admission: reject with 503 when every worker slot is taken.
  3. Lease the current knowledge-base version.  The lease pins one
     immutable KBSnapshot (store + BM25 + Qdrant collection of the same
     version) for the whole request and blocks its garbage collection.
  4. Run the synchronous pipeline in the worker pool against that
     snapshot.  The lease is released when the PIPELINE finishes -- not
     when the HTTP handler returns -- so a request that outlives its
     timeout still holds its version until its thread is done.
  5. Convert the PipelineResult to the response model and record the
     measured stage latencies.

Configuration is immutable per request: Settings is a frozen object built
once at startup, and every pipeline component was constructed from it.

  - Generated text is NEVER logged. Query text only at DEBUG level.
  - Auth: verify_auth is a dependency; no-op when require_auth=False.
  - body.enriched=True returns EnrichedQueryResponse (evidence, claims,
    conflicts, provenance, trace, latencies).  Default returns the thin
    QueryResponse.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse

from rasvcx.api.dependencies import (
    get_executor,
    get_latency,
    get_lease_tracker,
    get_pipeline,
    get_semaphore,
    get_settings,
    rate_limit,
    verify_auth,
)
from rasvcx.api.models import (
    ErrorResponse,
    QueryRequest as APIQueryRequest,
    QueryResponse,
    pipeline_result_to_response,
)
from rasvcx.api.models_enriched import (
    EnrichedQueryResponse,
    RuntimeModeModel,
    pipeline_result_to_enriched,
    stage_latencies_ms,
)
from rasvcx.config.settings import Settings
from rasvcx.pipeline.orchestrator import PipelineOrchestrator
from rasvcx.pipeline.pipeline_result import PipelineResult
from rasvcx.schemas.common import QueryId
from rasvcx.schemas.query import QueryRequest as PipelineQueryRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])


def runtime_mode(settings: Settings) -> RuntimeModeModel:
    """The mode/component summary attached to every response."""
    return RuntimeModeModel(
        execution_mode=settings.execution_mode.value,
        offline=settings.is_offline,
        llm_provider=settings.llm.provider,
        llm_model=None if settings.llm.provider == "stub" else settings.llm.model_name,
        mock_llm=settings.llm.provider == "stub",
        retrieval_mode=settings.retrieval.mode,
        reranker=settings.reranker.enabled,
        nli=settings.nli.enabled,
    )


_PROVIDER_ERROR_CODES = frozenset({"provider_failure", "timeout", "empty_response"})


def _is_degraded(result: PipelineResult) -> bool:
    """A component failed or produced no signal (same rule as the enriched
    `degraded` flag).  A correct safety abstention is not degradation."""
    from rasvcx.schemas.verification import VerificationReasonCode

    vsum = getattr(result, "validation_summary", None)
    ver = getattr(result, "verification_summary", None)
    return (
        getattr(result, "generation_error", None) is not None
        or any(t.status == "failed" for t in getattr(result, "trace", ()))
        or bool(getattr(vsum, "nli_failures", 0))
        or any(
            r.reason_code is VerificationReasonCode.NLI_UNAVAILABLE
            for r in (ver.claim_results if ver is not None else ())
        )
    )


def classify_request(result: PipelineResult, warm: bool) -> str:
    """Latency class of a request that actually ran the pipeline.

    PROVIDER_ERROR  the LLM provider failed (HTTP error, timeout, empty)
    SYSTEM_ERROR    any other stage failed
    COLD_UNCACHED   first pipeline execution of this process
    WARM_UNCACHED   any later pipeline execution
    (CACHE_HIT is assigned by the handler; the pipeline did not run.)
    """
    gen_err = getattr(result, "generation_error", None)
    if gen_err is not None:
        code = getattr(getattr(gen_err, "code", None), "value", None)
        return "PROVIDER_ERROR" if code in _PROVIDER_ERROR_CODES else "SYSTEM_ERROR"
    if any(t.status == "failed" for t in getattr(result, "trace", ())):
        return "SYSTEM_ERROR"
    return "WARM_UNCACHED" if warm else "COLD_UNCACHED"


def _kb_identity(snapshot: Any | None, version_id: str | None) -> dict[str, Any]:
    if snapshot is None:
        return {"version_id": version_id, "kb_source": None}
    d = snapshot.describe()
    return {k: d.get(k) for k in (
        "version_id", "kb_source", "corpus_hash", "index_hash", "identity_scheme",
        "doc_count", "chunk_count", "qdrant_collection", "retrieval_mode",
    )}


def _run_leased(
    pipeline: PipelineOrchestrator,
    lease: Any | None,
    query: PipelineQueryRequest,
    started: dict[str, float] | None = None,
) -> PipelineResult:
    """Worker-thread body: run the pipeline on the leased snapshot, then
    release the lease.  Runs to completion even if the HTTP request has
    already timed out, so the release always matches the real end of use.
    `started["t"]` records when the worker picked the request up (queue
    time = that minus submission)."""
    if started is not None:
        started["t"] = time.perf_counter()
    try:
        snapshot = lease.snapshot if lease is not None else None
        if snapshot is None:
            return pipeline.run(query, kb_version_id=lease.version_id if lease else None)
        return pipeline.run(
            query,
            retrieval_fn=snapshot.retrieval_fn,
            targeted_retrieval_fn=snapshot.targeted_retrieval_fn,
            kb_version_id=snapshot.version_id,
        )
    finally:
        if lease is not None:
            lease.release()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Run evidence-validated query",
    description=(
        "Accepts a natural-language query, runs the full RASVC-X "
        "evidence retrieval and validation pipeline, and returns a "
        "structured response with decision, confidence, and generated text. "
        "Set enriched=true in the request body for evidence, claims, "
        "conflicts, provenance, execution trace and latencies."
    ),
)
async def query(
    request: Request,
    body: APIQueryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    pipeline: Annotated[PipelineOrchestrator, Depends(get_pipeline)],
    executor: Annotated[ThreadPoolExecutor, Depends(get_executor)],
    semaphore: Annotated[asyncio.Semaphore, Depends(get_semaphore)],
    lease_tracker: Annotated[Any, Depends(get_lease_tracker)],
    latency: Annotated[Any, Depends(get_latency)],
    _auth: Annotated[None, Depends(verify_auth)],
    _rate: Annotated[None, Depends(rate_limit)],
) -> JSONResponse:
    """Run the RASVC-X pipeline for one query.

    Returns QueryResponse (or EnrichedQueryResponse when enriched=True).
    HTTP 422 on invalid/overlong query.
    HTTP 503 on timeout or server overload.
    HTTP 500 on unexpected pipeline exception.
    """
    from rasvcx.security.events import safe_request_id, security_event

    # A caller-supplied id is used only if it is a plain token (no control
    # characters that could forge log lines); else the middleware's id.
    request_id = (
        safe_request_id(body.request_id)
        or getattr(request.state, "request_id", None)
        or str(uuid.uuid4())
    )

    # Query length validation (body byte-size enforced by middleware)
    if len(body.query) > settings.api.max_query_chars:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"query length {len(body.query)} exceeds "
                f"max_query_chars={settings.api.max_query_chars}"
            ),
        )

    # Never log query text above DEBUG; here only its length is logged.
    logger.debug(
        "query request: request_id=%s query_len=%d enriched=%s",
        request_id, len(body.query), body.enriched,
    )

    # Answer cache: same question + context against the same KB version,
    # configuration, prompt template and models.
    t_handler = time.perf_counter()
    runtime = getattr(request.app.state, "rasvcx_runtime", None)
    identity = runtime.identity() if hasattr(runtime, "identity") else None
    cache = getattr(request.app.state, "rasvcx_cache", None)
    cache_key = (
        lease_tracker.current_version(),
        (identity or {}).get("config_hash"),
        (identity or {}).get("prompt_version"),
        str((identity or {}).get("model_versions")),
        str((identity or {}).get("calibration")),
        body.enriched,
        " ".join(body.query.lower().split()),
        tuple(sorted((body.context or {}).items())),
    )
    if cache is not None and not body.bypass_cache:
        hit = cache.get(cache_key)
        if hit is not None:
            content = dict(hit)
            content["request_id"] = request_id
            content["cached"] = True
            content["request_class"] = "CACHE_HIT"
            # The stored latencies belong to the request that computed the
            # answer.  Replaying them would report inference time for a
            # lookup, so they move to cached_from_* and the response
            # reports the time this request actually took.
            content["cached_from_total_latency_ms"] = content.get("total_latency_ms")
            content["total_latency_ms"] = round((time.perf_counter() - t_handler) * 1000.0, 3)
            if "stage_latencies_ms" in content:
                content["stage_latencies_ms"] = {}
            logger.info("query served from cache: request_id=%s", request_id)
            return JSONResponse(status_code=status.HTTP_200_OK, content=content)

    # Bounded admission: if all slots occupied, reject immediately.
    if semaphore.locked():
        logger.warning(
            "overload: semaphore exhausted, rejecting request_id=%s", request_id
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="overload",
                detail="Server is at capacity. Try again shortly.",
                request_id=request_id,
            ).model_dump(),
        )

    await semaphore.acquire()

    pipeline_query = PipelineQueryRequest(
        query_id=QueryId(request_id),
        raw_text=body.query,
        normalized_text=body.query.strip().lower(),
        metadata=dict(body.context or {}),
    )

    # Pin one knowledge-base version for the lifetime of this request.
    lease = lease_tracker.lease()

    loop = asyncio.get_running_loop()
    started: dict[str, float] = {}
    t_submit = time.perf_counter()
    try:
        future = loop.run_in_executor(executor, _run_leased, pipeline, lease, pipeline_query, started)
    except BaseException:
        # The worker never started, so it will never release the lease.
        if lease is not None:
            lease.release()
        semaphore.release()
        raise
    # The slot is freed when the WORKER finishes, so a timed-out request
    # keeps occupying its slot (and its KB lease) for as long as its
    # thread is really still running.
    future.add_done_callback(lambda _f: semaphore.release())

    try:
        result = await asyncio.wait_for(
            asyncio.shield(future),
            timeout=settings.api.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "timeout: request_id=%s exceeded %.1fs",
            request_id, settings.api.request_timeout_seconds,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="request_timeout",
                detail=(
                    f"Pipeline did not complete within "
                    f"{settings.api.request_timeout_seconds:.0f}s."
                ),
                request_id=request_id,
            ).model_dump(),
        )
    except Exception as exc:
        logger.exception(
            "pipeline error: request_id=%s error=%s", request_id, type(exc).__name__
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                detail="Pipeline raised an unexpected exception.",
                request_id=request_id,
            ).model_dump(),
        )

    mode = runtime_mode(settings)

    perr = getattr(result, "pipeline_error", None)
    if perr is not None and perr.stage == "input_validation":
        security_event("prompt_injection_rejected", request_id=request_id,
                       rule=perr.message.rsplit("(", 1)[-1].rstrip(")"))

    # Generated text is intentionally NOT logged at any level.
    t_ser = time.perf_counter()
    if body.enriched:
        response: EnrichedQueryResponse | QueryResponse = pipeline_result_to_enriched(
            result, request_id=request_id, mock_llm=mode.mock_llm, mode=mode,
        )
        canonical = response.decision
    else:
        response = pipeline_result_to_response(
            result, request_id=request_id, mock_llm=mode.mock_llm,
            execution_mode=mode.execution_mode,
        )
        canonical = response.decision.decision
    content = response.model_dump()
    serialization_ms = (time.perf_counter() - t_ser) * 1000.0

    warm = bool(getattr(request.app.state, "rasvcx_warm", False))
    request.app.state.rasvcx_warm = True
    request_class = classify_request(result, warm)

    trace = getattr(result, "trace", ())
    total_seconds = getattr(result, "total_seconds", 0.0)
    if (
        trace and isinstance(total_seconds, (int, float))
        and request_class in ("COLD_UNCACHED", "WARM_UNCACHED")
    ):
        # Only completed pipeline executions enter the latency window: a
        # provider error that returned in 200 ms is not inference latency.
        latency.record(
            stage_latencies_ms(trace),
            total_ms=total_seconds * 1000.0 + serialization_ms,
            serialization_ms=serialization_ms,
        )
    if body.enriched:
        content["stage_latencies_ms"]["serialization"] = round(serialization_ms, 3)

    content["cached"] = False
    content["request_class"] = request_class
    # Admission is immediate (or 503); queue_ms is the wait for a worker
    # thread after admission, handler_ms the full server-side time.
    content["queue_ms"] = round((started.get("t", t_submit) - t_submit) * 1000.0, 3)
    content["handler_ms"] = round((time.perf_counter() - t_handler) * 1000.0, 3)
    content["kb"] = _kb_identity(
        lease.snapshot if lease is not None else None, getattr(result, "kb_version_id", None),
    )
    content["run_identity"] = identity
    if cache is not None and not _is_degraded(result):
        # A degraded response (a component failed or produced no signal)
        # is never stored: a transient outage must not be replayed as a
        # cached abstention.
        cache.put(cache_key, content)

    logger.info(
        "query complete: request_id=%s decision=%s success=%s corrective=%d "
        "kb_version=%s enriched=%s",
        request_id, canonical, response.success, response.corrective_attempts,
        getattr(result, "kb_version_id", None), body.enriched,
    )

    return JSONResponse(status_code=status.HTTP_200_OK, content=content)


__all__ = ["router", "runtime_mode"]
