"""evaluation/baselines/b2_hybrid_rag.py

Baseline B2 — Hybrid RAG: BM25 + dense retrieval + MockLLMClient
generation, no M8 validation, no M9 verification.

CRITICAL: If Qdrant is unavailable, B2 is explicitly skipped with
status='skipped', skip_reason='qdrant_unavailable'.
BM25 is NEVER silently substituted for hybrid retrieval.

LIMITATIONS:
  - MockLLMClient does not perform real language model inference.
  - Results are labelled mock_llm=True.
  - Qdrant-unavailable runs record an explicit skip, never a fallback.
  - No result constitutes medical advice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from evaluation.baselines.interface import BaselineAdapter, BaselineResult
from evaluation.schema import CorpusConfig, EvalCase


_CANNED_TEXT = (
    "Based on the hybrid-retrieved documents, here is a response. "
    "[Mock generation — no real LLM inference performed.]"
)
_QDRANT_SKIP = "qdrant_unavailable"


class HybridRAGBaseline:
    """B2: BM25 + dense retrieval + MockLLMClient generation.

    Checks Qdrant availability at initialize() time. If Qdrant is
    unavailable, marks baseline as skipped. Never substitutes BM25
    for hybrid retrieval.

    Thread-safe: all mutable state set in initialize(); run() is read-only.
    """

    def __init__(
        self,
        bm25_top_k: int = 10,
        dense_top_k: int = 10,
        rrf_top_k: int = 5,
        model_id: str = "mock-llm-b2",
        canned_text: str = _CANNED_TEXT,
        qdrant_timeout: float = 2.0,
    ) -> None:
        self._bm25_top_k = bm25_top_k
        self._dense_top_k = dense_top_k
        self._rrf_top_k = rrf_top_k
        self._model_id = model_id
        self._canned_text = canned_text
        self._qdrant_timeout = qdrant_timeout

        self._retrieval_fn = None
        self._store = None
        self._llm = None
        self._cfg = None
        self._available = False
        self._skip_reason: Optional[str] = None

    @property
    def baseline_id(self) -> str:
        return "B2_HYBRID_RAG"

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
        """Load BM25 + attempt DenseRetriever. Skip if Qdrant unavailable."""
        from rasvcx.config.settings import CorpusSettings, RetrievalSettings, Settings
        from rasvcx.generation.generation_types import LLMConfig
        from rasvcx.generation.llm_client import MockLLMClient
        from rasvcx.retrieval.bm25 import BM25Index
        from rasvcx.retrieval.corpus import CorpusStore

        bm25_path = Path(corpus_config.bm25_path)
        store_path = Path(corpus_config.store_path)

        if not bm25_path.exists():
            self._skip_reason = f"BM25 index not found: {bm25_path}"
            return
        if not store_path.exists():
            self._skip_reason = f"CorpusStore not found: {store_path}"
            return

        bm25_index = BM25Index.load(str(bm25_path))
        self._store = CorpusStore.load(str(store_path))

        # Dense retrieval over the collection that belongs to THIS corpus
        # ("<prefix>_<version_id>").  The baseline is skipped -- with the
        # reason recorded -- when Qdrant is unreachable or that collection
        # does not hold exactly one point per chunk; it never degrades to
        # BM25-only while still calling itself a hybrid baseline.
        dense_retriever = None
        try:
            from rasvcx.pipeline.factory import seed_version_id
            from rasvcx.retrieval.dense import DenseRetriever
            from rasvcx.retrieval.knowledge_base import collection_name_for

            r = Settings().retrieval
            collection = collection_name_for(r.qdrant_collection, seed_version_id(self._store))
            dense_retriever = DenseRetriever.build(
                r.dense_model_name, r.qdrant_host, r.qdrant_port, collection,
                mode=r.qdrant_mode, path=r.qdrant_path,
            )
            count = dense_retriever.count()
            if count != len(self._store):
                self._skip_reason = (
                    f"{_QDRANT_SKIP}: collection {collection!r} has {count} points, "
                    f"corpus has {len(self._store)} chunks "
                    f"(build it with scripts/build_index.py --qdrant)"
                )
                return
        except Exception as exc:
            self._skip_reason = (
                f"{_QDRANT_SKIP}: {type(exc).__name__}: {exc}"
            )
            return

        from rasvcx.retrieval.bridge import make_retrieval_fn
        self._retrieval_fn = make_retrieval_fn(
            bm25_index=bm25_index,
            dense_retriever=dense_retriever,
            corpus_store=self._store,
            bm25_top_k=self._bm25_top_k,
            rrf_top_k=self._rrf_top_k,
        )

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
        """Run hybrid retrieval + generate. Raises if called when skipped."""
        if not self._available:
            raise RuntimeError(
                f"B2 is not available. skip_reason={self._skip_reason!r}. "
                "Check is_available before calling run()."
            )

        from rasvcx.schemas.common import QueryId
        from rasvcx.schemas.evidence import EvidenceBundle
        from rasvcx.schemas.query import (
            QueryRequest,
            RiskFeatureScores,
            RiskProfile,
            ValidationDepth,
        )

        qid = QueryId(case.case_id)
        rp = RiskProfile(
            overall_risk_score=0.5,
            feature_scores=RiskFeatureScores(),
            validation_depth=ValidationDepth.STANDARD,
            retrieval_retry_budget=0,
            nli_call_allowance=0,
        )
        query = QueryRequest(
            query_id=qid,
            raw_text=case.query,
            normalized_text=case.query.strip().lower(),
        )
        bundle = EvidenceBundle(query_id=qid, risk_profile=rp)

        self._retrieval_fn(query, rp, bundle)

        chunk_ids = list(bundle.evidence_items.keys())
        context_parts: list[str] = []
        for item_id, item in bundle.evidence_items.items():
            context_parts.append(f"[{item.chunk_id}] {item.text[:300]}")

        context = "\n\n".join(context_parts) if context_parts else "(no context)"
        system_prompt = (
            "You are a knowledge assistant. "
            "Use only the provided documents to answer the question."
        )
        user_prompt = f"Documents:\n{context}\n\nQuestion: {case.query}"

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
                "rrf_top_k": self._rrf_top_k,
                "n_retrieved": len(chunk_ids),
                "note": (
                    "B2: Hybrid RAG baseline. No validation. "
                    "MockLLMClient; not real model output."
                ),
            },
        )

    def close(self) -> None:
        self._available = False
        self._retrieval_fn = None
        self._store = None
        self._llm = None


__all__ = ["HybridRAGBaseline"]