"""evaluation/baselines/interface.py

BaselineAdapter Protocol and BaselineResult for M15 evaluation baselines.

Every baseline must implement BaselineAdapter and be initialised once per
run before any cases are processed (not once per case).

LIMITATIONS:
  - MockLLMClient baselines (B0, B1, B2) do not produce real model output.
    Results must be labelled mock_llm=True and must not be used to claim
    real model accuracy, answer correctness, or calibration.
  - B2 must explicitly skip when Qdrant is unavailable; never substitute BM25.
  - No baseline result constitutes medical advice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from evaluation.schema import CorpusConfig, EvalCase


# ---------------------------------------------------------------------------
# BaselineResult
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    """Typed output from one baseline.run(case) call.

    Fields:
      decision        -- DecisionAction.value string, or None if not produced
      answer_verdict  -- AnswerVerdict.value string, or None
      generated_text  -- generated answer text (empty string if none)
      retrieved_chunk_ids  -- ordered list of retrieved chunk IDs
      conflict_resolutions -- list of {candidate_id, relationship, confidence}
      claim_verifications  -- list of {claim_id, label, confidence}
      confidence      -- decision.confidence float, or None
      mock_llm        -- True if MockLLMClient was the generator
      raw             -- full serialisable dict of additional pipeline output
    """

    decision: Optional[str] = None
    answer_verdict: Optional[str] = None
    generated_text: str = ""
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    conflict_resolutions: list[dict[str, Any]] = field(default_factory=list)
    claim_verifications: list[dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    mock_llm: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "answer_verdict": self.answer_verdict,
            "generated_text": self.generated_text,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "conflict_resolutions": self.conflict_resolutions,
            "claim_verifications": self.claim_verifications,
            "confidence": self.confidence,
            "mock_llm": self.mock_llm,
            "raw": self.raw,
        }


# ---------------------------------------------------------------------------
# BaselineAdapter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BaselineAdapter(Protocol):
    """Protocol that every M15 baseline must implement.

    Lifecycle:
      1. Construct the adapter (once per run, before workers start).
         __init__ receives a CorpusConfig and any baseline-specific settings.
      2. Call initialize() to load indexes, build pipeline, etc.
         Initialisation time is recorded separately from per-case latency.
      3. Call run(case) once per case from a worker thread.
         run() must be thread-safe if called concurrently.
      4. Call close() after all cases are done.

    Properties:
      baseline_id   -- short identifier string (e.g. 'B0_LLM_ONLY')
      mock_llm      -- True if MockLLMClient is the generator
      is_available  -- True after initialize(), False if init failed/skipped
      skip_reason   -- non-None if baseline cannot run (e.g. Qdrant missing)
    """

    @property
    def baseline_id(self) -> str: ...

    @property
    def mock_llm(self) -> bool: ...

    @property
    def is_available(self) -> bool: ...

    @property
    def skip_reason(self) -> Optional[str]: ...

    def initialize(self, corpus_config: CorpusConfig) -> None:
        """Load indexes, build pipeline. Called once before workers start."""
        ...

    def run(self, case: EvalCase) -> BaselineResult:
        """Execute the baseline for one EvalCase. Thread-safe.

        Raises on unexpected errors; the runner converts exceptions to
        CaseResult(status='error').
        """
        ...

    def close(self) -> None:
        """Release resources. Called after all cases are done."""
        ...


__all__ = ["BaselineResult", "BaselineAdapter"]