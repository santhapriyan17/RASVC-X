"""evaluation/baselines/b0_llm_only.py

Baseline B0 — LLM-only, no retrieval, no validation.

Uses MockLLMClient directly (no evidence required).
This baseline represents a raw language model response with no
evidence grounding.  It is the weakest baseline and is used to
measure the minimum bar for any retrieval-augmented system.

LIMITATIONS:
  - MockLLMClient does not perform real language model inference.
  - Results are labelled mock_llm=True.
  - These results must not be used to claim real model accuracy,
    answer correctness, or calibration.
  - No result constitutes medical advice.
"""

from __future__ import annotations

import threading
from typing import Optional

from evaluation.baselines.interface import BaselineAdapter, BaselineResult
from evaluation.schema import CorpusConfig, EvalCase


class LLMOnlyBaseline:
    """B0: raw LLM call with no retrieval or validation.

    Calls MockLLMClient.generate() directly with a minimal system
    prompt and the case query as the user prompt.  No evidence bundle
    is created; the Generator is not used (it requires evidence items).

    Thread-safe: MockLLMClient is stateless; concurrent calls are safe.
    """

    def __init__(
        self,
        canned_text: str = "This is a mock LLM response with no evidence grounding.",
        model_id: str = "mock-llm-b0",
    ) -> None:
        self._canned_text = canned_text
        self._model_id = model_id
        self._llm = None
        self._cfg = None
        self._available = False
        self._skip_reason: Optional[str] = None
        self._lock = threading.Lock()

    # -- BaselineAdapter protocol properties --

    @property
    def baseline_id(self) -> str:
        return "B0_LLM_ONLY"

    @property
    def mock_llm(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def skip_reason(self) -> Optional[str]:
        return self._skip_reason

    # -- Lifecycle --

    def initialize(self, corpus_config: CorpusConfig) -> None:
        """Construct MockLLMClient. No corpus is loaded (LLM-only baseline)."""
        from rasvcx.generation.generation_types import LLMConfig
        from rasvcx.generation.llm_client import MockLLMClient

        self._llm = MockLLMClient(
            canned_text=self._canned_text,
            model_id=self._model_id,
        )
        self._cfg = LLMConfig(
            model_id=self._model_id,
            temperature=0.0,
            max_tokens=256,
            timeout_seconds=10.0,
        )
        self._available = True

    def run(self, case: EvalCase) -> BaselineResult:
        """Call the LLM with the query. No retrieval or validation."""
        if not self._available:
            raise RuntimeError(
                f"B0 baseline not initialised. Call initialize() first."
            )

        system_prompt = (
            "You are a knowledge assistant. Answer the question directly. "
            "No external sources are provided."
        )
        llm_response = self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=case.query,
            config=self._cfg,
        )

        return BaselineResult(
            decision="answer" if llm_response.text else "abstain",
            generated_text=llm_response.text,
            retrieved_chunk_ids=[],
            mock_llm=True,
            raw={
                "baseline_id": self.baseline_id,
                "model_id": self._model_id,
                "llm_tokens": llm_response.completion_tokens,
                "note": (
                    "B0: LLM-only baseline. No retrieval, no validation. "
                    "MockLLMClient; not real model output."
                ),
            },
        )

    def close(self) -> None:
        """No resources to release."""
        self._available = False


__all__ = ["LLMOnlyBaseline"]