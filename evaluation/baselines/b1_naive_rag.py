"""evaluation/baselines/b1_naive_rag.py

Baseline B1 — Naive RAG: BM25 retrieval + MockLLMClient generation,
no M5 sufficiency gate, no M8 validation, no M9 verification.

Represents a minimal retrieval-augmented generation system without
any of RASVC-X's safety or validation components.

LIMITATIONS:
  - MockLLMClient does not perform real language model inference.
  - Results are labelled mock_llm=True.
  - These results must not be used to claim real answer accuracy.
  - No result constitutes medical advice.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

from evaluation.baselines.interface import BaselineAdapter, BaselineResult
from evaluation.schema import CorpusConfig, EvalCase


_CANNED_TEXT = (
    "Based on the retrieved documents, here is a response. "
    "[Mock generation — no real LLM inference performed.]"
)


class NaiveRAGBaseline:
    """B1: BM25 retrieval + direct MockLLMClient generation.

    Retrieves top-K chunks via BM25, formats them into a prompt,
    and generates via MockLLMClient.  No sufficiency gate, no conflict
    detection, no claim validation.

    Initialised once per run; thread-safe for concurrent case execution.
    """

    def __init__(
        self,
        bm25_top_k: int = 5,
        model_id: str = "mock-llm-b1",
        canned_text: str = _CANNED_TEXT,
    ) -> None:
        self._bm25_top_k = bm25_top_k
        self._model_id = model_id
        self._canned_text = canned_text
        self._index = None
        self._store = None
        self._llm = None
        self._cfg = None
        self._available = False
        self._skip_reason: Optional[str] = None

    @property
    def baseline_id(self) -> str:
        return "B1_NAIVE_RAG"

    @property
    def mock_llm(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def skip_reason(self) -> Optional[str]:
        return self._skip_reason

    def initialize(self, corpus_config: CorpusConfig) -> None:
        """Load BM25 index and CorpusStore from corpus_config paths."""
        from rasvcx.generation.generation_types import LLMConfig
        from rasvcx.generation.llm_client import MockLLMClient
        from rasvcx.retrieval.bm25 import BM25Index
        from rasvcx.retrieval.corpus import CorpusStore

        bm25_path = Path(corpus_config.bm25_path)
        store_path = Path(corpus_config.store_path)

        if not bm25_path.exists():
            self._skip_reason = (
                f"BM25 index not found: {bm25_path}"
            )
            return
        if not store_path.exists():
            self._skip_reason = (
                f"CorpusStore not found: {store_path}"
            )
            return

        self._index = BM25Index.load(str(bm25_path))
        self._store = CorpusStore.load(str(store_path))
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
        """Retrieve top-K chunks via BM25 then generate via MockLLMClient."""
        if not self._available:
            raise RuntimeError(
                "B1 not initialised or unavailable. "
                f"skip_reason={self._skip_reason!r}"
            )

        # BM25 retrieval
        hits = self._index.query(case.query, top_k=self._bm25_top_k)
        chunk_ids = [h.chunk_id for h in hits]

        # Build context from retrieved chunks
        context_parts: list[str] = []
        for hit in hits:
            result = self._store.lookup(hit.chunk_id)
            if result is not None:
                text, _ = result
                context_parts.append(f"[{hit.chunk_id}] {text}")

        context = "\n\n".join(context_parts) if context_parts else "(no context)"
        system_prompt = (
            "You are a knowledge assistant. "
            "Use only the provided documents to answer the question."
        )
        user_prompt = (
            f"Documents:\n{context}\n\n"
            f"Question: {case.query}"
        )

        llm_response = self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            config=self._cfg,
        )

        return BaselineResult(
            decision="answer" if llm_response.text else "abstain",
            generated_text=llm_response.text,
            retrieved_chunk_ids=chunk_ids,
            mock_llm=True,
            raw={
                "baseline_id": self.baseline_id,
                "model_id": self._model_id,
                "bm25_top_k": self._bm25_top_k,
                "n_retrieved": len(chunk_ids),
                "note": (
                    "B1: BM25-only naive RAG. No validation. "
                    "MockLLMClient; not real model output."
                ),
            },
        )

    def close(self) -> None:
        self._available = False
        self._index = None
        self._store = None
        self._llm = None


__all__ = ["NaiveRAGBaseline"]