# src/rasvcx/reranking/cross_encoder.py
"""Cross-encoder reranking for RASVC-X.

Loads a cross-encoder model once at construction time and scores
(query, passage) pairs in configurable batches.  The caller controls
the candidate cap; this module enforces it before any inference call
so that the model never sees more candidates than configured.

Design constraints:
- Model is loaded once in __init__; never per request.
- Model is swappable: any sentence-transformers CrossEncoder-compatible
  model name may be supplied.
- Returns a deterministically ordered list (descending score, then
  chunk_id lexicographic as tiebreaker).
- Scores are raw cross-encoder logits/probabilities — not probabilities
  unless the model is explicitly calibrated. Do NOT interpret as
  probabilities downstream.
- No business logic: pure scoring + ordering.
- Thread-safety: CrossEncoder.predict() is stateless given fixed model
  weights; concurrent calls are safe as long as the underlying torch
  model allows it (standard behaviour for inference-only mode).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rasvcx.schemas.common import ChunkId

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_MAX_CANDIDATES: int = 30
_DEFAULT_BATCH_SIZE: int = 32


@dataclass(frozen=True, slots=True)
class CrossEncoderConfig:
    """Configuration for the cross-encoder reranker.

    Attributes:
        model_name:     HuggingFace model identifier for the cross-encoder.
        max_candidates: Hard cap on candidates accepted before inference.
                        Candidates beyond this limit are silently dropped
                        (callers should pre-sort by retrieval score and pass
                        only the top N).
        batch_size:     Inference batch size passed to CrossEncoder.predict().
        device:         Torch device string, e.g. "cpu", "cuda", "mps".
                        None lets sentence-transformers choose automatically.
    """

    model_name: str = _DEFAULT_MODEL_NAME
    max_candidates: int = _DEFAULT_MAX_CANDIDATES
    batch_size: int = _DEFAULT_BATCH_SIZE
    device: str | None = None

    def __post_init__(self) -> None:
        if self.max_candidates < 1:
            raise ValueError(
                f"max_candidates must be >= 1, got {self.max_candidates}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.batch_size}"
            )


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Score produced by the cross-encoder for one (query, passage) pair.

    Attributes:
        chunk_id:      ID of the passage chunk, preserved from input.
        rerank_score:  Raw cross-encoder score. Higher is better.
                       Not a calibrated probability.
        original_text: The passage text that was scored (preserved for
                       downstream EvidenceItem attachment without re-lookup).
    """

    chunk_id: ChunkId
    rerank_score: float
    original_text: str


class CrossEncoderReranker:
    """Wraps a sentence-transformers CrossEncoder for candidate reranking.

    The model is loaded once during __init__.  Subsequent calls to
    ``rerank()`` are pure inference with no model reloading.

    Usage::

        config = CrossEncoderConfig(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                                    max_candidates=30)
        reranker = CrossEncoderReranker(config)
        results = reranker.rerank(query="...", candidates=[...])
    """

    def __init__(self, config: CrossEncoderConfig | None = None) -> None:
        self._config = config or CrossEncoderConfig()
        self._model = self._load_model()

    def _load_model(self) -> object:
        """Load the cross-encoder model once.  Returns the model object."""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        kwargs: dict[str, object] = {}
        if self._config.device is not None:
            kwargs["device"] = self._config.device

        logger.info(
            "Loading cross-encoder model %r (device=%r)",
            self._config.model_name,
            self._config.device,
        )
        model = CrossEncoder(self._config.model_name, **kwargs)
        logger.info("Cross-encoder model loaded successfully.")
        return model

    @property
    def config(self) -> CrossEncoderConfig:
        return self._config

    def rerank(
        self,
        query: str,
        candidates: list[tuple[ChunkId, str]],
    ) -> list[RerankResult]:
        """Score and rank (query, passage) pairs.

        Args:
            query:      The user query string.
            candidates: List of (chunk_id, passage_text) pairs. The order
                        of this list does not affect output ordering; results
                        are always returned sorted descending by rerank_score,
                        with chunk_id as lexicographic tiebreaker.

        Returns:
            Sorted list of RerankResult (descending rerank_score).
            Length = min(len(candidates), config.max_candidates).
            Empty list if candidates is empty or query is empty/whitespace.

        Raises:
            RuntimeError: If the underlying model raises an unexpected error
                          during inference.
        """
        if not query or not query.strip():
            logger.warning("rerank() called with empty/whitespace query; returning empty.")
            return []

        if not candidates:
            return []

        # Enforce candidate cap before inference.
        capped = candidates[: self._config.max_candidates]
        if len(candidates) > self._config.max_candidates:
            logger.warning(
                "rerank(): received %d candidates, capped to %d",
                len(candidates),
                self._config.max_candidates,
            )

        pairs: list[list[str]] = [[query, text] for _, text in capped]
        chunk_ids: list[ChunkId] = [cid for cid, _ in capped]
        texts: list[str] = [text for _, text in capped]

        try:
            scores: list[float] = self._model.predict(  # type: ignore[attr-defined]
                pairs,
                batch_size=self._config.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"CrossEncoder.predict() failed for model "
                f"{self._config.model_name!r}: {exc}"
            ) from exc

        # Build results and sort: descending score, chunk_id as tiebreaker.
        results: list[RerankResult] = [
            RerankResult(
                chunk_id=cid,
                rerank_score=float(score),
                original_text=text,
            )
            for cid, text, score in zip(chunk_ids, texts, scores)
        ]
        results.sort(key=lambda r: (-r.rerank_score, r.chunk_id))
        return results