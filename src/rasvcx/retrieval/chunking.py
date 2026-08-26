"""Text chunking strategies for RASVC-X corpus ingestion.

Two strategies are supported:

1. FIXED          — fixed-size token windows with configurable overlap.
2. SENTENCE_WINDOW — sentence-boundary-aware token windows.

Both return a list of Chunk objects ready for indexing.

UNRESOLVED DECISION (see DECISIONS.md):
    Final chunking strategy is unresolved until retrieval benchmarking
    (Recall@K) is complete.  Do not hard-wire one strategy as final.

Design constraints:
- Pure functions: no I/O, no side effects, no global state.
- chunk_id format: "{doc_id}__chunk_{index:04d}" — stable and sortable.
- Overlap measured in whitespace-split tokens, not characters.
- Empty/whitespace-only documents produce zero chunks.
- Overlap clamped to max_tokens - 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from rasvcx.schemas.common import ChunkId


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE_WINDOW = "sentence_window"


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    strategy: ChunkStrategy = ChunkStrategy.FIXED
    max_tokens: int = 256
    overlap: int = 32

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")

    @property
    def effective_overlap(self) -> int:
        return min(self.overlap, self.max_tokens - 1)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: ChunkId
    text: str
    doc_id: str
    index: int


def _make_chunk_id(doc_id: str, index: int) -> ChunkId:
    return ChunkId(f"{doc_id}__chunk_{index:04d}")


def _tokenize(text: str) -> list[str]:
    return text.split()


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


def chunk_fixed(doc_id: str, text: str, config: ChunkConfig) -> list[Chunk]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    max_t = config.max_tokens
    step = max(max_t - config.effective_overlap, 1)
    chunks: list[Chunk] = []
    index = 0
    pos = 0
    while pos < len(tokens):
        window = tokens[pos: pos + max_t]
        chunk_text = " ".join(window).strip()
        if chunk_text:
            chunks.append(Chunk(chunk_id=_make_chunk_id(doc_id, index), text=chunk_text, doc_id=doc_id, index=index))
            index += 1
        pos += step
    return chunks


def chunk_sentence_window(doc_id: str, text: str, config: ChunkConfig) -> list[Chunk]:
    sentences = _split_sentences(text)
    if not sentences:
        return []
    max_t = config.max_tokens
    chunks: list[Chunk] = []
    index = 0
    window_sentences: list[str] = []
    window_token_count = 0
    for sentence in sentences:
        s_tokens = len(_tokenize(sentence))
        if s_tokens == 0:
            continue
        if s_tokens > max_t:
            if window_sentences:
                chunk_text = " ".join(window_sentences).strip()
                if chunk_text:
                    chunks.append(Chunk(chunk_id=_make_chunk_id(doc_id, index), text=chunk_text, doc_id=doc_id, index=index))
                    index += 1
                window_sentences = []
                window_token_count = 0
            chunks.append(Chunk(chunk_id=_make_chunk_id(doc_id, index), text=sentence.strip(), doc_id=doc_id, index=index))
            index += 1
            continue
        if window_token_count + s_tokens > max_t and window_sentences:
            chunk_text = " ".join(window_sentences).strip()
            if chunk_text:
                chunks.append(Chunk(chunk_id=_make_chunk_id(doc_id, index), text=chunk_text, doc_id=doc_id, index=index))
                index += 1
            window_sentences = []
            window_token_count = 0
        window_sentences.append(sentence)
        window_token_count += s_tokens
    if window_sentences:
        chunk_text = " ".join(window_sentences).strip()
        if chunk_text:
            chunks.append(Chunk(chunk_id=_make_chunk_id(doc_id, index), text=chunk_text, doc_id=doc_id, index=index))
    return chunks


def chunk_document(doc_id: str, text: str, config: ChunkConfig) -> list[Chunk]:
    if config.strategy is ChunkStrategy.FIXED:
        return chunk_fixed(doc_id, text, config)
    if config.strategy is ChunkStrategy.SENTENCE_WINDOW:
        return chunk_sentence_window(doc_id, text, config)
    raise ValueError(f"Unknown chunking strategy: {config.strategy!r}")