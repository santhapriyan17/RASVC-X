"""Generation layer data contracts (Module 11).

Defines the types that flow through the generation layer:

  LLMConfig       — provider configuration (model, temperature, limits)
  LLMResponse     — raw provider output (text + optional token counts)
  GenerationErrorCode — machine-parseable failure taxonomy
  GenerationError — structured failure representation
  GenerationResult — M11 output contract

GenerationResult is the output of Generator.generate(). Its minimum
required fields are:

  generated_text  — the answer string (empty on failure); this is what
                    M9's VerificationPipeline.verify() consumes as its
                    first argument.
  error           — None on success, a GenerationError on failure.

All other fields (cited_item_ids, model_id, generation_time_seconds,
prompt_token_estimate, completion_token_estimate) are optional metadata
for observability, reproducibility, and future evaluation. No existing
downstream module (M9, M10) consumes them.

This module contains no generation logic -- it is a pure contract,
following the repository's established convention (schemas/ modules
define contracts, implementation modules consume them).

No existing M1-M10 contract is redefined, duplicated, or modified here.
EvidenceItemId is reused from rasvcx.schemas.common; no new ID type is
introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from rasvcx.schemas.common import EvidenceItemId


# ---------------------------------------------------------------------------
# LLM provider configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Provider-agnostic generation configuration.

    Carried through the generation layer and recorded in GenerationResult
    metadata for reproducibility. Does NOT claim to represent an actual
    provider context window limit -- max_tokens is the requested completion
    ceiling, not the provider's hard limit. The generation layer's
    max_context_chars safety budget is a separate, application-level
    parameter (see prompt_builder.py).

    temperature=0.0 is the default for maximum reproducibility in a
    research/evidence-grounded context. It does not guarantee
    deterministic output (providers may still vary across calls).
    """

    model_id: str = "unspecified"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("LLMConfig.model_id must be non-empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"LLMConfig.temperature must be in [0, 2], got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(
                f"LLMConfig.max_tokens must be >= 1, got {self.max_tokens}"
            )
        if self.timeout_seconds <= 0:
            raise ValueError(
                f"LLMConfig.timeout_seconds must be > 0, got {self.timeout_seconds}"
            )


# ---------------------------------------------------------------------------
# LLM provider response
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Raw output from an LLM provider call.

    This is the provider-facing contract returned by an LLMClient
    implementation. It is consumed by Generator internally and is NOT
    exposed as part of the pipeline-facing GenerationResult contract.

    text is the generated completion. prompt_tokens and
    completion_tokens are optional provider-reported counts -- None
    when the provider does not report them, never fabricated.
    """

    text: str
    model_id: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# ---------------------------------------------------------------------------
# Generation failure taxonomy
# ---------------------------------------------------------------------------


class GenerationErrorCode(str, Enum):
    """Machine-parseable generation failure codes.

    Each code represents a distinct failure mode with a clear cause:

      PROVIDER_FAILURE  — the LLM provider raised an exception or
                          returned an error status (network, auth, rate
                          limit, server error).
      TIMEOUT           — the provider did not respond within the
                          configured deadline.
      EMPTY_RESPONSE    — the provider returned empty or whitespace-only
                          text. This is not an M9 concern (M9 handles
                          empty answers gracefully) but the generation
                          layer records it as a distinct failure so the
                          future orchestrator can decide whether to retry.
      CONTEXT_TOO_LARGE — the fully-constructed prompt (with ALL
                          evidence included, never truncated) exceeded the
                          configured max_context_chars safety budget. No
                          evidence is silently dropped; the generation
                          fails safely. The future orchestrator may
                          respond with an alternate strategy.
      NO_EVIDENCE       — the EvidenceBundle contained no evidence items.
                          Generation cannot produce a grounded answer
                          from zero evidence.
    """

    PROVIDER_FAILURE = "provider_failure"
    TIMEOUT = "timeout"
    EMPTY_RESPONSE = "empty_response"
    CONTEXT_TOO_LARGE = "context_too_large"
    NO_EVIDENCE = "no_evidence"


@dataclass(frozen=True, slots=True)
class GenerationError:
    """Structured generation failure.

    Preserved in GenerationResult.error so that the future pipeline
    orchestrator (M12, not yet implemented) can inspect and route
    failures without parsing free-text messages. No existing module
    currently consumes this -- it is a forward-looking contract for
    M12 integration.

    is_retryable is advisory: True suggests that the same request
    might succeed on a subsequent attempt (e.g. transient provider
    error). False indicates a structural problem (e.g. no evidence,
    context too large for the configured budget) that will not resolve
    by retrying.
    """

    code: GenerationErrorCode
    message: str
    is_retryable: bool

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("GenerationError.message must be non-empty")


# ---------------------------------------------------------------------------
# Generation result — M11 output contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Immutable output of the generation layer for one pipeline pass.

    Minimum required contract:
      generated_text — the answer string with inline citations, consumed
                       by M9's VerificationPipeline.verify() as its
                       ``generated_answer`` argument. Empty string on
                       failure (M9 handles empty answers gracefully,
                       returning a valid VerificationSummary with no
                       claims).
      error         — None on success; a GenerationError describing the
                       failure mode otherwise. When error is not None,
                       generated_text is empty.

    Optional metadata (not consumed by any existing downstream module):
      cited_item_ids          — EvidenceItemIds that appeared as citation
                                markers in generated_text, extracted by
                                CitationFormatter. Observability only;
                                M9's own GeneratedClaimExtractor is the
                                authoritative claim-to-evidence mapping.
      model_id                — provider/model identifier for
                                reproducibility and future evaluation.
      generation_time_seconds — wall-clock generation latency.
      prompt_token_estimate   — approximate prompt size reported by
                                provider, or None if unavailable.
      completion_token_estimate — approximate completion size reported
                                  by provider, or None if unavailable.
    """

    generated_text: str
    error: GenerationError | None = None

    # -- optional metadata (observability / reproducibility) ----------------
    cited_item_ids: frozenset[EvidenceItemId] = field(default_factory=frozenset)
    model_id: str | None = None
    generation_time_seconds: float | None = None
    prompt_token_estimate: int | None = None
    completion_token_estimate: int | None = None

    def __post_init__(self) -> None:
        if self.error is not None and self.generated_text:
            raise ValueError(
                "GenerationResult: generated_text must be empty when error is set "
                "(a failed generation must not carry a partial answer)"
            )
        if self.generation_time_seconds is not None and self.generation_time_seconds < 0:
            raise ValueError(
                f"GenerationResult.generation_time_seconds must be >= 0, "
                f"got {self.generation_time_seconds}"
            )

    @property
    def success(self) -> bool:
        """True when generation completed without error."""
        return self.error is None