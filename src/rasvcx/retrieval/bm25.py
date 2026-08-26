"""BM25 sparse retrieval for RASVC-X.

Wraps rank-bm25 (BM25Okapi) with a minimal, strongly-typed interface.
The index is built once at startup from a list of (chunk_id, text) pairs
and is read-only thereafter.  Serialisation (pickle) allows the index to
be persisted to disk and reloaded without re-indexing the corpus.

Design constraints:
- No network I/O, no LLM calls, no global mutable state.
- Index construction is O(N * avg_tokens); query is O(N) after tokenization.
- Tokenization is whitespace + lowercase; callers are responsible for any
  upstream normalization (stopword removal, stemming) if desired.
- Scores returned are raw BM25 scores (not probabilities); do NOT interpret
  them as confidence values.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from rank_bm25 import BM25Okapi

from rasvcx.schemas.common import ChunkId

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True, slots=True)
class BM25Result:
    chunk_id: ChunkId
    score: float


class BM25Index:
    def __init__(self, chunk_ids: list[ChunkId], tokenized: list[list[str]]) -> None:
        if len(chunk_ids) != len(tokenized):
            raise ValueError(f"chunk_ids length ({len(chunk_ids)}) must equal tokenized length ({len(tokenized)})")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_ids must be unique")
        self._chunk_ids: list[ChunkId] = list(chunk_ids)
        self._bm25 = BM25Okapi(tokenized)
        self._size = len(chunk_ids)
        logger.debug("BM25Index built: %d documents", self._size)

    @classmethod
    def build(cls, documents: Sequence[tuple[ChunkId, str]]) -> "BM25Index":
        if not documents:
            raise ValueError("Cannot build BM25Index from empty document list")
        chunk_ids: list[ChunkId] = []
        tokenized: list[list[str]] = []
        for chunk_id, text in documents:
            chunk_ids.append(chunk_id)
            tokenized.append(_tokenize(text))
        return cls(chunk_ids, tokenized)

    def save(self, path: Path) -> None:
        path = Path(path)
        with path.open("wb") as fh:
            pickle.dump({"chunk_ids": self._chunk_ids, "bm25": self._bm25}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25Index saved: %s (%d docs)", path, self._size)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")
        with path.open("rb") as fh:
            data = pickle.load(fh)
        required = {"chunk_ids", "bm25"}
        if not required.issubset(data):
            raise ValueError(f"Malformed BM25 index file {path}: missing keys {required - set(data)}")
        instance = cls.__new__(cls)
        instance._chunk_ids = data["chunk_ids"]
        instance._bm25 = data["bm25"]
        instance._size = len(instance._chunk_ids)
        logger.info("BM25Index loaded: %s (%d docs)", path, instance._size)
        return instance

    def query(self, query_text: str, top_k: int) -> list[BM25Result]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if self._size == 0:
            raise ValueError("Cannot query an empty BM25Index")
        tokens = _tokenize(query_text)
        if not tokens:
            logger.debug("BM25 query produced no tokens; returning empty results")
            return []
        scores: list[float] = self._bm25.get_scores(tokens).tolist()
        k = min(top_k, self._size)
        indexed = sorted(
            ((score, idx) for idx, score in enumerate(scores) if score > 0.0),
            key=lambda t: t[0],
            reverse=True,
        )[:k]
        return [BM25Result(chunk_id=self._chunk_ids[idx], score=score) for score, idx in indexed]

    @property
    def size(self) -> int:
        return self._size