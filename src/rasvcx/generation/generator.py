"""Top-level generation orchestrator (Module 11).

Composes PromptBuilder, LLMClient, and CitationFormatter into a single
generate() call that produces a GenerationResult. This is the only
public entry point for the generation layer — downstream consumers
(the future M12 orchestrator) call Generator.generate() and receive
a GenerationResult.

Responsibilities:
  - Pre-flight checks (evidence presence, context size)
  - Prompt construction via PromptBuilder
  - LLM invocation via LLMClient
  - Post-generation citation extraction via extract_cited_ids
  - Error mapping (LLMClientError → GenerationError)
  - Timing measurement

Non-responsibilities (belong to other modules):
  - Claim extraction from the answer (M9)
  - Citation correctness verification (M9)
  - Confidence estimation (M10)
  - Decision routing (M10)
  - Retry / regeneration policy (future M12)
  - Evidence selection / truncation (not implemented; fail-safe only)

EvidenceBundle is read-only. No existing M1-M10 contract is modified.
"""

from __future__ import annotations

import time

from rasvcx.generation.citation_formatter import extract_cited_ids
from rasvcx.generation.generation_types import (
    GenerationError,
    GenerationErrorCode,
    GenerationResult,
    LLMConfig,
)
from rasvcx.generation.llm_client import LLMClient, LLMClientError, LLMTimeoutError
from rasvcx.generation.prompt_builder import PromptBuilder
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.query import QueryRequest
from rasvcx.validation.verified_context import ValidationSummary


class Generator:
    """Module 11 orchestrator.

    All collaborators are injected (dependency injection, matching the
    M8 ValidationPipeline and M9 VerificationPipeline conventions):
    no global mutable state, no singleton model objects.

    Constructing a Generator performs no model loading, no network
    access, and no expensive initialization.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._llm_config = llm_config or LLMConfig()

    def last_provider_stats(self) -> dict | None:
        """The LLM client's record of this thread's last provider call
        (attempts, retries, wait, quota), when the client keeps one."""
        getter = getattr(self._llm_client, "last_call_stats", None)
        return getter() if callable(getter) else None

    def generate(
        self,
        query: QueryRequest,
        bundle: EvidenceBundle,
        validation_summary: ValidationSummary | None = None,
        verification_feedback: str | None = None,
    ) -> GenerationResult:
        """Generate an evidence-grounded answer.

        Returns a GenerationResult on every path — success and failure
        alike. Never raises for expected failure modes (no evidence,
        provider failure, timeout, empty response, context too large).
        Unexpected exceptions propagate normally.

        EvidenceBundle is read-only: this method records elapsed time
        via bundle.record_stage_elapsed (the only mutation, matching
        the pattern used by M8 and M9) but never modifies evidence
        items, claims, validation results, or resolution.

        Args:
            query:                 The user's question.
            bundle:                Validated evidence (read-only).
            validation_summary:    M8 conflict/resolution info, or None.
            verification_feedback: Optional repair context derived from
                M9 verification findings on a previous generation
                attempt.  Forwarded to PromptBuilder.build() which
                appends it as a REPAIR INSTRUCTIONS section.  When None
                (the default), behavior is identical to a first-pass
                generation — all existing callers are unaffected.
        """
        start = time.perf_counter()
        reset = getattr(self._llm_client, "reset_call_stats", None)
        if callable(reset):
            reset()  # a pre-flight failure must not inherit the last call's record
        try:
            return self._generate_inner(
                query, bundle, validation_summary, verification_feedback, start
            )
        finally:
            bundle.record_stage_elapsed(
                "generation", time.perf_counter() - start
            )

    def _generate_inner(
        self,
        query: QueryRequest,
        bundle: EvidenceBundle,
        validation_summary: ValidationSummary | None,
        verification_feedback: str | None,
        start: float,
    ) -> GenerationResult:
        # -- Pre-flight: evidence presence --
        if not bundle.evidence_items:
            return GenerationResult(
                generated_text="",
                error=GenerationError(
                    code=GenerationErrorCode.NO_EVIDENCE,
                    message="EvidenceBundle contains no evidence items",
                    is_retryable=False,
                ),
            )

        # -- Prompt construction --
        prompt_result = self._prompt_builder.build(
            query, bundle, validation_summary,
            verification_feedback=verification_feedback,
        )

        # -- Pre-flight: context size --
        if prompt_result.total_chars > self._prompt_builder.max_context_chars:
            return GenerationResult(
                generated_text="",
                error=GenerationError(
                    code=GenerationErrorCode.CONTEXT_TOO_LARGE,
                    message=(
                        f"Prompt size ({prompt_result.total_chars} chars) exceeds "
                        f"max_context_chars ({self._prompt_builder.max_context_chars})"
                    ),
                    is_retryable=False,
                ),
            )

        # -- LLM invocation --
        try:
            response = self._llm_client.generate(
                system_prompt=prompt_result.system_prompt,
                user_prompt=prompt_result.user_prompt,
                config=self._llm_config,
            )
        except LLMTimeoutError as exc:
            return GenerationResult(
                generated_text="",
                error=GenerationError(
                    code=GenerationErrorCode.TIMEOUT,
                    message=str(exc) or "LLM provider timed out",
                    is_retryable=True,
                ),
            )
        except LLMClientError as exc:
            return GenerationResult(
                generated_text="",
                error=GenerationError(
                    code=GenerationErrorCode.PROVIDER_FAILURE,
                    message=str(exc) or "LLM provider call failed",
                    is_retryable=True,
                ),
            )

        # -- Post-flight: empty response --
        if not response.text or not response.text.strip():
            return GenerationResult(
                generated_text="",
                error=GenerationError(
                    code=GenerationErrorCode.EMPTY_RESPONSE,
                    message="LLM provider returned empty or whitespace-only text",
                    is_retryable=True,
                ),
            )

        # -- Success --
        generated_text = response.text
        cited_ids = extract_cited_ids(generated_text, bundle)
        elapsed = time.perf_counter() - start

        return GenerationResult(
            generated_text=generated_text,
            cited_item_ids=cited_ids,
            model_id=response.model_id,
            generation_time_seconds=elapsed,
            prompt_token_estimate=response.prompt_tokens,
            completion_token_estimate=response.completion_tokens,
        )


__all__ = ["Generator"]