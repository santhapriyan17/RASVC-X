"""evaluation/baselines/b3_rasvcx.py

Baseline B3 — Full RASVC-X pipeline via build_pipeline().

Uses the M13 factory with a Settings object constructed from the
provided CorpusConfig.  This is the primary system under evaluation.

The pipeline is built once at initialize() time and reused across all
cases.  Per-case latency reflects only the pipeline.run() call.

LIMITATIONS:
  - In offline_test mode, MockLLMClient is used; not real model output.
  - Results with mock_llm=True must not be used to claim real accuracy.
  - No result constitutes medical advice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from evaluation.baselines.interface import BaselineAdapter, BaselineResult
from evaluation.schema import CorpusConfig, EvalCase


class RASVCXBaseline:
    """B3: Full RASVC-X pipeline built via M13 factory.

    Corpus paths are injected via Settings(corpus=CorpusSettings(...))
    so the factory reads from the evaluation corpus, not the default
    development corpus.  No M13 factory changes are required.

    Thread-safe: PipelineOrchestrator.run() is designed for concurrent
    use via the M14 ThreadPoolExecutor pattern.
    """

    def __init__(self, model_id: str = "mock-llm-b3") -> None:
        self._model_id = model_id
        self._pipeline = None
        self._settings = None
        self._available = False
        self._skip_reason: Optional[str] = None
        self._mock_llm: bool = True

    @property
    def baseline_id(self) -> str:
        return "B3_RASVCX"

    @property
    def mock_llm(self) -> bool:
        return self._mock_llm

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def skip_reason(self) -> Optional[str]:
        return self._skip_reason

    def initialize(self, corpus_config: CorpusConfig) -> None:
        """Build the full RASVC-X pipeline with the evaluation corpus paths."""
        from rasvcx.config.settings import CorpusSettings, Settings
        from rasvcx.pipeline.factory import build_pipeline

        store_path = Path(corpus_config.store_path)
        bm25_path = Path(corpus_config.bm25_path)

        if not store_path.exists():
            self._skip_reason = f"CorpusStore not found: {store_path}"
            return
        if not bm25_path.exists():
            self._skip_reason = f"BM25 index not found: {bm25_path}"
            return

        try:
            self._settings = Settings(
                corpus=CorpusSettings(
                    store_path=str(store_path),
                    bm25_path=str(bm25_path),
                )
            )
            self._pipeline = build_pipeline(self._settings)
            # Determine if MockLLMClient is in use
            self._mock_llm = (
                self._settings.llm.provider == "stub"
            )
            self._available = True
        except Exception as exc:
            self._skip_reason = (
                f"Pipeline build failed: {type(exc).__name__}: {exc}"
            )

    def run(self, case: EvalCase) -> BaselineResult:
        """Run the full RASVC-X pipeline for one EvalCase."""
        if not self._available:
            raise RuntimeError(
                f"B3 not available. skip_reason={self._skip_reason!r}"
            )

        from rasvcx.schemas.common import QueryId
        from rasvcx.schemas.query import QueryRequest

        query = QueryRequest(
            query_id=QueryId(case.case_id),
            raw_text=case.query,
            normalized_text=case.query.strip().lower(),
        )

        result = self._pipeline.run(query)

        # Extract claim verifications
        claim_verifications: list[dict] = []
        if result.verification_summary is not None:
            for cr in result.verification_summary.claim_results:
                claim_verifications.append({
                    "claim_id": str(cr.claim_id),
                    "label": cr.label.value,
                    "confidence": cr.confidence,
                    "supporting_item_ids": list(cr.supporting_item_ids),
                    "contradicting_item_ids": list(cr.contradicting_item_ids),
                })

        # Extract conflict resolutions
        conflict_resolutions: list[dict] = []
        if result.validation_summary is not None:
            for res in result.validation_summary.resolutions:
                conflict_resolutions.append({
                    "candidate_id": str(res.candidate_id),
                    "relationship": res.relationship.value,
                    "confidence": res.confidence,
                    "rationale": res.rationale,
                })

        answer_verdict = None
        if result.verification_summary is not None:
            answer_verdict = result.verification_summary.answer_verdict.value

        return BaselineResult(
            decision=result.decision.action.value,
            answer_verdict=answer_verdict,
            generated_text=result.generated_text or "",
            # The evidence the decision was based on, in final ranked order.
            retrieved_chunk_ids=[str(item.chunk_id) for item in result.evidence],
            conflict_resolutions=conflict_resolutions,
            claim_verifications=claim_verifications,
            confidence=result.decision.confidence,
            mock_llm=self._mock_llm,
            raw={
                "baseline_id": self.baseline_id,
                "corrective_attempts": result.corrective_attempts,
                "pipeline_error": (
                    {
                        "stage": result.pipeline_error.stage,
                        "message": result.pipeline_error.message,
                        "is_retryable": result.pipeline_error.is_retryable,
                    }
                    if result.pipeline_error else None
                ),
                "success": result.success,
                "note": (
                    "B3: Full RASVC-X pipeline. "
                    f"mock_llm={self._mock_llm}. "
                    "Evidence items not in PipelineResult "
                    "(EvidenceBundle not serialised)."
                ),
            },
        )

    def close(self) -> None:
        self._available = False
        self._pipeline = None
        self._settings = None


__all__ = ["RASVCXBaseline"]