"""LLM provider abstraction (Module 11).

Defines the service boundary between the generation layer and any LLM
provider, mirroring the repository's established pattern in
validation/nli_interface.py (NLIBackend Protocol + NLIService wrapper +
NullNLIBackend safe default).

This module performs no model loading, no network access, and has no
heavy dependencies. Concrete provider adapters (OpenAI, Anthropic,
Ollama, etc.) belong in future adapter modules and are injected into
Generator via this interface -- the generation layer itself never imports
a provider SDK.

LLMClient (Protocol):
    The structural interface any provider implementation must satisfy.
    Uses typing.Protocol (structural subtyping) so adapters do not need
    to inherit from a base class -- matching the NLIBackend convention.

MockLLMClient:
    Test-only client that returns a configurable canned response.
    Analogous to NullNLIBackend: enables full M11 testing without any
    external dependency or network access.

LLMClientError:
    Raised by LLMClient implementations when a call cannot be completed.
    Analogous to NLIUnavailableError. Generator catches this and maps it
    to GenerationError(PROVIDER_FAILURE).

LLMTimeoutError:
    Subclass of LLMClientError for timeout-specific failures, so
    Generator can distinguish TIMEOUT from generic PROVIDER_FAILURE.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rasvcx.generation.generation_types import LLMConfig, LLMResponse


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LLMClientError(Exception):
    """Raised by an LLMClient when a generation call cannot be completed.

    Covers: network failure, authentication error, rate limiting, server
    error, model not found, or any other reason the provider could not
    return a completion. Generator catches this and converts it into a
    structured GenerationError(PROVIDER_FAILURE), so provider-specific
    exception types never leak into the pipeline.
    """


class LLMTimeoutError(LLMClientError):
    """Raised when the provider did not respond within the configured
    timeout. A subclass of LLMClientError so a single except clause
    catches both, while Generator can distinguish TIMEOUT from generic
    PROVIDER_FAILURE when it needs to.
    """


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface a concrete LLM provider adapter must satisfy.

    Implementations wrap a single provider's SDK/API (OpenAI, Anthropic,
    Ollama, a local model, etc.) and translate its response into an
    LLMResponse. The generation layer never calls provider-specific APIs
    directly -- it calls only this protocol.

    Contract:
      - generate() must return an LLMResponse on success.
      - generate() must raise LLMTimeoutError on timeout.
      - generate() must raise LLMClientError on any other failure.
      - generate() must never return a fabricated response on failure.
      - generate() must never perform retries internally; retry policy
        belongs to the caller (Generator or the future orchestrator).
    """

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> LLMResponse:
        """Send a system+user prompt pair to the provider and return the
        completion.

        Must raise LLMClientError (or LLMTimeoutError) if the call
        cannot be completed -- never return a partial or fabricated
        response.
        """
        ...


# ---------------------------------------------------------------------------
# Mock client (test-only)
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Test-only LLM client that returns a configurable canned response.

    Enables full M11 unit and integration testing without any external
    provider dependency, network access, or API key.

    Usage:
        client = MockLLMClient(canned_text="The dosage is 500mg. [E1]")
        response = client.generate(system_prompt, user_prompt, config)
        assert response.text == "The dosage is 500mg. [E1]"

    To simulate provider failure:
        client = MockLLMClient(raise_on_generate=LLMClientError("boom"))

    To simulate timeout:
        client = MockLLMClient(raise_on_generate=LLMTimeoutError("timed out"))

    To simulate empty response:
        client = MockLLMClient(canned_text="")
    """

    def __init__(
        self,
        canned_text: str = "",
        model_id: str = "mock-model",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        raise_on_generate: Exception | None = None,
    ) -> None:
        self._canned_text = canned_text
        self._model_id = model_id
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens
        self._raise_on_generate = raise_on_generate
        self._last_system_prompt: str | None = None
        self._last_user_prompt: str | None = None
        self._last_config: LLMConfig | None = None
        self._call_count: int = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: LLMConfig,
    ) -> LLMResponse:
        self._last_system_prompt = system_prompt
        self._last_user_prompt = user_prompt
        self._last_config = config
        self._call_count += 1

        if self._raise_on_generate is not None:
            raise self._raise_on_generate

        return LLMResponse(
            text=self._canned_text,
            model_id=self._model_id,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )

    # -- test inspection helpers -------------------------------------------

    @property
    def last_system_prompt(self) -> str | None:
        """The system prompt from the most recent generate() call."""
        return self._last_system_prompt

    @property
    def last_user_prompt(self) -> str | None:
        """The user prompt from the most recent generate() call."""
        return self._last_user_prompt

    @property
    def last_config(self) -> LLMConfig | None:
        """The LLMConfig from the most recent generate() call."""
        return self._last_config

    @property
    def call_count(self) -> int:
        """Total number of generate() calls made."""
        return self._call_count


__all__ = [
    "LLMClientError",
    "LLMTimeoutError",
    "LLMClient",
    "MockLLMClient",
]