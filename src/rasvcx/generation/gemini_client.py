"""Gemini LLM provider adapter for RASVC-X (Module 14).

Implements the LLMClient Protocol (generation/llm_client.py) using the
current Google Gen AI SDK (google-genai >= 2.0.0).

The previously used google-generativeai SDK is deprecated and no longer
receives updates.  This module uses google.genai exclusively.
See: https://github.com/google-gemini/deprecated-generative-ai-python

Contract (verified against llm_client.py):
  - generate() returns LLMResponse on success.
  - generate() returns LLMResponse(text="") when the provider call
    succeeds but returns empty/whitespace-only text -- the existing M11
    Generator post-flight check classifies this as EMPTY_RESPONSE.
  - generate() raises LLMTimeoutError on timeout.
  - generate() raises LLMClientError on any other provider/API failure.
  - generate() retries ONLY transient provider overload (HTTP 429/500/503),
    at most max_retries times with a short backoff, and never beyond the
    request's timeout budget.  Every other failure is raised immediately.
    The retry count is logged; a request that still fails is reported as
    a provider failure -- it is never replaced by a fabricated answer.
  - SDK imported lazily so the module can be imported without
    google-genai installed; ImportError raised only at
    GeminiLLMClient construction time.

Readiness check (used by /ready endpoint):
  - is_configured(): api_key non-empty and SDK importable.
  - No billable API call is made during readiness checks.

Security:
  - api_key is never logged at any level.
  - api_key is never included in exception messages.
  - Generated text is never logged (only its length).

SDK API surface (verified against google-genai 2.28.0):
  client = genai.Client(api_key=...)
  response = client.models.generate_content(
      model=model_name,
      contents=str,
      config=types.GenerateContentConfig(
          system_instruction=str,
          temperature=float,
          max_output_tokens=int,
          http_options=types.HttpOptions(timeout=int),
      ),
  )
  response.text          -> str | None
  response.usage_metadata.prompt_token_count      -> int | None
  response.usage_metadata.candidates_token_count  -> int | None

Exception types (verified against google.genai.errors):
  google.genai.errors.APIError   -> base; has .code (int) and .status (str)
  google.genai.errors.ClientError -> 4xx errors (including 429 rate limit)
  google.genai.errors.ServerError -> 5xx errors
  TimeoutError (stdlib)           -> HTTP timeout via httpx
  httpx.TimeoutException          -> connection/read timeout
"""

from __future__ import annotations

import logging
import time
from typing import Any

from rasvcx.generation.generation_types import LLMConfig, LLMResponse
from rasvcx.generation.llm_client import LLMClientError, LLMTimeoutError

logger = logging.getLogger(__name__)

_SDK_AVAILABLE: bool | None = None


def _check_sdk() -> None:
    """Raise ImportError if google-genai is not installed."""
    global _SDK_AVAILABLE
    if _SDK_AVAILABLE is True:
        return
    if _SDK_AVAILABLE is False:
        raise ImportError(
            "google-genai is required for GeminiLLMClient. "
            "Install with: pip install google-genai>=2.0.0"
        )
    try:
        import google.genai  # noqa: F401
        _SDK_AVAILABLE = True
    except ImportError:
        _SDK_AVAILABLE = False
        raise ImportError(
            "google-genai is required for GeminiLLMClient. "
            "Install with: pip install google-genai>=2.0.0"
        )


def _is_timeout_exception(exc: BaseException) -> bool:
    """Return True if exc represents a network/HTTP timeout."""
    if isinstance(exc, TimeoutError):
        return True
    try:
        import httpx
        if isinstance(exc, httpx.TimeoutException):
            return True
    except ImportError:
        pass
    exc_type = type(exc).__name__
    exc_str = str(exc).lower()
    if "timeout" in exc_type.lower() or "deadline" in exc_type.lower():
        return True
    if "timed out" in exc_str or "deadline exceeded" in exc_str:
        return True
    try:
        if getattr(exc, "code", None) == 504:
            return True
    except Exception:
        pass
    return False


class GeminiLLMClient:
    """LLMClient adapter for Google Gemini via google-genai SDK (v2.x).

    Replaces the deprecated google-generativeai adapter.

    Empty-response handling:
        If the provider call succeeds but returns empty or whitespace-only
        text, generate() returns LLMResponse(text=""). The Generator's
        existing post-flight check (generator.py) then classifies this as
        EMPTY_RESPONSE. LLMClientError is NOT raised for empty provider
        text -- that would misclassify it as PROVIDER_FAILURE.

    Thread safety:
        google.genai.Client is created once per GeminiLLMClient instance.
        The client uses httpx internally; httpx clients are thread-safe for
        concurrent requests. Each generate() call is independent.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        _check_sdk()

        if not api_key:
            raise ValueError("GeminiLLMClient: api_key must be non-empty")
        if not model_name:
            raise ValueError("GeminiLLMClient: model_name must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError(
                f"GeminiLLMClient: timeout_seconds must be > 0, got {timeout_seconds}"
            )

        if max_retries < 0:
            raise ValueError(f"GeminiLLMClient: max_retries must be >= 0, got {max_retries}")

        self._api_key = api_key           # never logged
        self._model_name = model_name
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

        import google.genai as genai
        import threading
        self._client = genai.Client(api_key=self._api_key)
        # Per-thread record of the last generate() call (a request runs in
        # one worker thread, so this is the request's own provider record).
        self._local = threading.local()

        logger.info(
            "GeminiLLMClient initialised (model=%s timeout=%.1fs sdk=google-genai)",
            self._model_name, self._timeout_seconds,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> LLMResponse:
        """Call the Gemini API and return an LLMResponse.

        Returns LLMResponse(text="") when the provider returns empty
        content -- M11 Generator classifies that as EMPTY_RESPONSE.
        Raises LLMTimeoutError on timeout, LLMClientError on other failures.

        The system_prompt is passed via GenerateContentConfig.system_instruction
        (the correct location in the google-genai SDK). user_prompt is passed
        as the contents argument.
        """
        timeout = (
            config.timeout_seconds
            if config.timeout_seconds > 0
            else self._timeout_seconds
        )

        from google.genai import types as genai_types

        timeout_ms = int(timeout * 1000)

        gen_config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            http_options=genai_types.HttpOptions(timeout=timeout_ms),
            # No tools are passed, so nothing could be called; disabling
            # AFC states that explicitly and keeps the SDK from ever
            # executing a callable on the model's behalf.
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )

        logger.debug(
            "Gemini generate: model=%s temperature=%.2f max_tokens=%d timeout=%.1fs",
            self._model_name, config.temperature, config.max_tokens, timeout,
        )

        start = time.perf_counter()
        deadline = start + timeout
        attempt = 0
        stats = {"attempts": 0, "statuses": [], "retry_wait_seconds": 0.0,
                 "elapsed_seconds": 0.0, "quota": None, "failed": False}
        self._local.last = stats
        while True:
            t_attempt = time.perf_counter()
            stats["attempts"] += 1
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user_prompt,
                    config=gen_config,
                )
                _record_request(self._model_name, 200, time.perf_counter() - t_attempt, None)
                stats["statuses"].append(200)
                break
            except Exception as exc:
                elapsed = time.perf_counter() - start
                code = getattr(exc, "code", None)
                quota = _quota_details(exc)
                _record_request(self._model_name, code if isinstance(code, int) else type(exc).__name__,
                                time.perf_counter() - t_attempt, quota)
                stats["statuses"].append(code if isinstance(code, int) else type(exc).__name__)
                if quota:
                    stats["quota"] = quota
                if _is_timeout_exception(exc):
                    stats.update(failed=True, elapsed_seconds=elapsed)
                    logger.warning(
                        "Gemini timeout after %.2fs (model=%s)",
                        elapsed, self._model_name,
                    )
                    raise LLMTimeoutError(
                        f"Gemini request timed out after {elapsed:.1f}s "
                        f"(model={self._model_name})"
                    ) from exc
                description = _describe_provider_error(exc)
                backoff = _RETRY_BACKOFF_SECONDS * (attempt + 1)
                # The provider tells us when a retry can succeed.  Retrying a
                # rate-limited request before that only adds load and fails
                # again; if the advised delay does not fit in this request's
                # budget, fail now instead of retrying uselessly.
                advised = (quota or {}).get("retry_delay_seconds")
                if isinstance(advised, (int, float)) and advised > 0:
                    backoff = max(backoff, float(advised))
                if (
                    attempt < self._max_retries
                    and _is_transient(exc)
                    and time.perf_counter() + backoff < deadline
                ):
                    attempt += 1
                    logger.warning(
                        "Gemini transient failure (%s%s); retry %d/%d in %.1fs",
                        description,
                        f"; quota={quota.get('quota_id')} limit={quota.get('quota_value')}" if quota else "",
                        attempt, self._max_retries, backoff,
                    )
                    stats["retry_wait_seconds"] += backoff
                    time.sleep(backoff)
                    continue
                stats.update(failed=True, elapsed_seconds=elapsed)
                # Never include api_key or prompt content in error messages
                logger.warning(
                    "Gemini provider failure after %.2fs and %d retr%s: %s%s",
                    elapsed, attempt, "y" if attempt == 1 else "ies", description,
                    f" (quota={quota.get('quota_id')} limit={quota.get('quota_value')} "
                    f"retry_after={quota.get('retry_delay_seconds')}s)" if quota else "",
                )
                raise LLMClientError(
                    f"Gemini provider failure ({description}; model={self._model_name}; "
                    f"retries={attempt})"
                ) from exc
        stats["elapsed_seconds"] = time.perf_counter() - start

        elapsed = time.perf_counter() - start
        text = _extract_text(response)

        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        try:
            usage = response.usage_metadata
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_token_count", None)
                completion_tokens = getattr(usage, "candidates_token_count", None)
        except Exception:
            pass

        finish_reason = _finish_reason(response)
        logger.debug(
            "Gemini response: %.2fs text_len=%d prompt_tok=%s comp_tok=%s finish=%s retries=%d",
            elapsed, len(text), prompt_tokens, completion_tokens, finish_reason, attempt,
        )
        if finish_reason and "MAX_TOKENS" in finish_reason:
            # A truncated answer is still verified claim-by-claim downstream;
            # the truncation itself is surfaced here for the operator.
            logger.warning(
                "Gemini answer hit max_output_tokens=%d (text_len=%d); raise llm.max_tokens",
                config.max_tokens, len(text),
            )

        return LLMResponse(
            text=text,
            model_id=self._model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def reset_call_stats(self) -> None:
        self._local.last = None

    def last_call_stats(self) -> dict[str, Any] | None:
        """Provider record of this thread's most recent generate() call:
        attempts (HTTP requests incl. retries), per-attempt statuses,
        seconds spent waiting between retries, total elapsed, and the
        quota violation the provider reported (if any)."""
        last = getattr(self._local, "last", None)
        return dict(last) if last is not None else None

    def is_configured(self) -> bool:
        """Return True if api_key is non-empty and SDK is importable.
        Does NOT make a billable API call."""
        return bool(self._api_key) and (_SDK_AVAILABLE is True)

    def model_name(self) -> str:
        return self._model_name


# Provider overload / rate limiting: worth one or two quick retries.
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 503})
_RETRY_BACKOFF_SECONDS = 1.5


def _is_transient(exc: BaseException) -> bool:
    """Provider overload (429/500/503), or a connection that was never
    established (the request did not reach the provider, so a retry cannot
    duplicate anything)."""
    if getattr(exc, "code", None) in _TRANSIENT_STATUS_CODES:
        return True
    try:
        import httpx
        return isinstance(exc, httpx.ConnectError)
    except ImportError:  # pragma: no cover - httpx ships with google-genai
        return False


#: Every HTTP attempt this process made (bounded): when, status, duration,
#: quota id.  Evaluation scripts read it to measure the real request rate.
REQUEST_LOG: "collections.deque[dict[str, Any]]"
import collections as _collections  # noqa: E402
import threading as _threading  # noqa: E402

REQUEST_LOG = _collections.deque(maxlen=20000)
_LOG_LOCK = _threading.Lock()


def _record_request(model: str, status: object, seconds: float, quota: dict | None) -> None:
    with _LOG_LOCK:
        REQUEST_LOG.append({"t": time.time(), "model": model, "status": status,
                            "seconds": round(seconds, 3),
                            "quota_id": (quota or {}).get("quota_id"),
                            "quota_value": (quota or {}).get("quota_value"),
                            "retry_delay_seconds": (quota or {}).get("retry_delay_seconds")})


def _quota_details(exc: BaseException) -> dict[str, Any] | None:
    """Structured quota information from a Gemini error, if present.

    Reads only machine fields of google.rpc.QuotaFailure / RetryInfo
    (quota id, metric, limit, retry delay) -- never the free-text message,
    which can echo request content.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    items = (details.get("error") or {}).get("details") or details.get("details") or []
    out: dict[str, Any] = {}
    for d in items if isinstance(items, list) else []:
        kind = str(d.get("@type", ""))
        if kind.endswith("QuotaFailure"):
            v = (d.get("violations") or [{}])[0]
            out.update(quota_id=v.get("quotaId"), quota_metric=v.get("quotaMetric"),
                       quota_value=v.get("quotaValue"))
        elif kind.endswith("RetryInfo"):
            raw = str(d.get("retryDelay", "")).rstrip("s")
            try:
                out["retry_delay_seconds"] = float(raw)
            except ValueError:
                pass
    return out or None


def _describe_provider_error(exc: BaseException) -> str:
    """Short, secret-free description: exception type + HTTP code + status.

    The provider's message text is deliberately NOT included: it can echo
    request content.  Code and status are enough to tell an overloaded
    model (503) from a retired one (404) or an exhausted quota (429).
    """
    parts = [type(exc).__name__]
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    if isinstance(code, int):
        parts.append(str(code))
    if isinstance(status, str) and status:
        parts.append(status)
    return " ".join(parts)


def _finish_reason(response: Any) -> str | None:
    try:
        candidates = response.candidates or []
        if candidates:
            reason = getattr(candidates[0], "finish_reason", None)
            return str(getattr(reason, "name", reason)) if reason is not None else None
    except Exception:  # noqa: BLE001 - observability only
        return None
    return None


def _extract_text(response: Any) -> str:
    """Extract text from a google-genai GenerateContentResponse.

    Returns "" on any failure — caller (Generator) handles empty text
    as EMPTY_RESPONSE, not as PROVIDER_FAILURE.
    """
    try:
        text = response.text
        if isinstance(text, str):
            return text
        if text is None:
            return ""
    except Exception:
        pass
    try:
        for candidate in response.candidates or []:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", []) or []
            fragments = [
                part.text
                for part in parts
                if isinstance(getattr(part, "text", None), str)
            ]
            if fragments:
                return "".join(fragments)
    except Exception:
        pass
    return ""


def sdk_available() -> bool:
    """Return True if google-genai is importable. No side effects."""
    try:
        import google.genai  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = ["GeminiLLMClient", "sdk_available"]