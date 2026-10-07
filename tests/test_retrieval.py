"""Regression tests for the RASVC-X retrieval package (Module 3).

Covers: BM25Index, DenseRetriever (mocked encoder + mocked Qdrant client),
reciprocal_rank_fusion, chunking strategies, metadata filtering, and
targeted retrieval merge behaviour.

No live external services are used. DenseRetriever/QdrantStore are
exercised against fake SentenceTransformer/QdrantClient objects injected
via monkeypatch, so these tests run without sentence-transformers or
qdrant-client installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from rasvcx.retrieval import dense as dense_mod
from rasvcx.retrieval.bm25 import BM25Index, BM25Result
from rasvcx.retrieval.chunking import (
    ChunkConfig,
    ChunkStrategy,
    chunk_document,
    chunk_fixed,
    chunk_sentence_window,
)
from rasvcx.retrieval.dense import DenseResult, DenseRetriever, QdrantUnavailableError
from rasvcx.retrieval.fusion import FusedResult, reciprocal_rank_fusion
from rasvcx.retrieval.metadata_filter import (
    MetadataFilterCriteria,
    ProvenanceSnapshot,
    filter_by_metadata,
)
from rasvcx.retrieval.targeted_retrieval import (
    TargetedRetrievalConfig,
    run_targeted_retrieval,
)
from rasvcx.schemas.common import UNKNOWN, ChunkId


# ===========================================================================
# BM25
# ===========================================================================

class TestBM25Build:
    def test_build_from_documents(self) -> None:
        index = BM25Index.build(
            [
                (ChunkId("a"), "aspirin reduces fever and pain"),
                (ChunkId("b"), "ibuprofen reduces inflammation"),
            ]
        )
        assert index.size == 2

    def test_build_rejects_empty_document_list(self) -> None:
        with pytest.raises(ValueError):
            BM25Index.build([])

    def test_constructor_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            BM25Index([ChunkId("a"), ChunkId("b")], [["x"]])

    def test_constructor_rejects_duplicate_chunk_ids(self) -> None:
        with pytest.raises(ValueError):
            BM25Index([ChunkId("a"), ChunkId("a")], [["x"], ["y"]])


class TestBM25Query:
    @pytest.fixture()
    def index(self) -> BM25Index:
        return BM25Index.build(
            [
                (ChunkId("aspirin_doc"), "aspirin reduces fever and mild pain in adults"),
                (ChunkId("ibuprofen_doc"), "ibuprofen is an anti inflammatory pain reliever"),
                (ChunkId("unrelated_doc"), "the weather today is sunny and warm"),
            ]
        )

    def test_ranking_orders_by_relevance(self, index: BM25Index) -> None:
        results = index.query("aspirin fever pain", top_k=3)
        assert results, "expected at least one match"
        assert results[0].chunk_id == ChunkId("aspirin_doc")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self, index: BM25Index) -> None:
        assert index.query("", top_k=5) == []

    def test_whitespace_only_query_returns_empty(self, index: BM25Index) -> None:
        assert index.query("   ", top_k=5) == []

    def test_top_k_bound_respected(self, index: BM25Index) -> None:
        results = index.query("pain reliever fever inflammatory", top_k=1)
        assert len(results) <= 1

    def test_top_k_larger_than_corpus_does_not_error(self, index: BM25Index) -> None:
        results = index.query("aspirin", top_k=1000)
        assert len(results) <= index.size

    def test_invalid_top_k_raises(self, index: BM25Index) -> None:
        with pytest.raises(ValueError):
            index.query("aspirin", top_k=0)
        with pytest.raises(ValueError):
            index.query("aspirin", top_k=-1)

    def test_no_matching_terms_returns_empty(self, index: BM25Index) -> None:
        results = index.query("zzzznonexistenttermzzzz", top_k=5)
        assert results == []

    def test_query_is_deterministic(self, index: BM25Index) -> None:
        r1 = index.query("aspirin fever", top_k=3)
        r2 = index.query("aspirin fever", top_k=3)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
        assert [r.score for r in r1] == [r.score for r in r2]

    def test_scores_are_plain_floats_not_probabilities(self, index: BM25Index) -> None:
        results = index.query("aspirin fever pain", top_k=3)
        assert all(isinstance(r.score, float) for r in results)


# ===========================================================================
# Dense retrieval (mocked encoder + mocked Qdrant)
# ===========================================================================

@dataclass
class _FakeEncodedVector:
    values: list

    def tolist(self):
        return self.values


class _FakeEncoder:
    """Fake SentenceTransformer: deterministic, dimension=4, init-once."""

    init_calls = 0

    def __init__(self, model_name: str) -> None:
        _FakeEncoder.init_calls += 1
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts, batch_size=None, show_progress_bar=False, normalize_embeddings=True):
        if isinstance(texts, str):
            return _FakeEncodedVector([1.0, 0.0, 0.0, 0.0])
        return _FakeBatchVectors([[1.0, 0.0, 0.0, 0.0] for _ in texts])


@dataclass
class _FakeBatchVectors:
    rows: list

    def tolist(self):
        return self.rows


class _FakeCollectionDesc:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeCollectionsResponse:
    def __init__(self, names) -> None:
        self.collections = [_FakeCollectionDesc(n) for n in names]


class _FakeScoredPoint:
    def __init__(self, id: int, score: float, payload: dict) -> None:
        self.id = id
        self.score = score
        self.payload = payload


class _FakeQdrantClient:
    """Fake QdrantClient with controllable failure modes.

    Mirrors the qdrant-client API that dense.py actually calls
    (query_points / count / get_collections / create_collection /
    delete_collection / upsert).  qdrant-client removed the old search()
    method, so the fake does not offer one either: code that still called
    it would fail here exactly as it fails against the real client.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self._existing_collections: list = []
        self._search_results: list = []
        self._raise_on_search = False
        self._raise_on_get_collections = False
        self.upserted_points: list = []
        self.query_calls = 0

    def get_collections(self):
        if self._raise_on_get_collections:
            raise RuntimeError("simulated Qdrant connection failure")
        return _FakeCollectionsResponse(self._existing_collections)

    def create_collection(self, collection_name, vectors_config):
        self._existing_collections.append(collection_name)
        self.upserted_points = []

    def delete_collection(self, collection_name):
        if collection_name in self._existing_collections:
            self._existing_collections.remove(collection_name)

    def upsert(self, collection_name, points):
        self.upserted_points.extend(points)

    def count(self, collection_name, exact=True):
        class _Count:
            count = len(self.upserted_points)
        return _Count()

    def query_points(self, collection_name, query, limit, with_payload=True):
        self.query_calls += 1
        if self._raise_on_search:
            raise RuntimeError("simulated Qdrant search failure")

        class _Response:
            points = self._search_results[:limit]
        return _Response()


class _FakeDistance:
    COSINE = "cosine"


class _FakeVectorParams:
    def __init__(self, size, distance) -> None:
        self.size = size
        self.distance = distance


class _FakePointStruct:
    def __init__(self, id, vector, payload) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


@pytest.fixture()
def fake_qdrant_client(monkeypatch: pytest.MonkeyPatch) -> _FakeQdrantClient:
    """Patch dense.py's module-level names so DenseRetriever can be built
    without real sentence-transformers / qdrant-client installed."""
    monkeypatch.setattr(dense_mod, "_ST_AVAILABLE", True, raising=False)
    monkeypatch.setattr(dense_mod, "_QDRANT_AVAILABLE", True, raising=False)
    monkeypatch.setattr(dense_mod, "SentenceTransformer", _FakeEncoder, raising=False)
    monkeypatch.setattr(dense_mod, "Distance", _FakeDistance, raising=False)
    monkeypatch.setattr(dense_mod, "VectorParams", _FakeVectorParams, raising=False)
    monkeypatch.setattr(dense_mod, "PointStruct", _FakePointStruct, raising=False)

    created = {}

    class _Capturing(_FakeQdrantClient):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            created["client"] = self

    monkeypatch.setattr(dense_mod, "QdrantClient", _Capturing, raising=False)
    retriever = DenseRetriever.build(
        "fake-model", "localhost", 6333, "rasvcx_chunks",
    )
    client = created["client"]
    client._owning_retriever = retriever
    return client


@pytest.fixture()
def dense_retriever(fake_qdrant_client: _FakeQdrantClient) -> DenseRetriever:
    return fake_qdrant_client._owning_retriever


class TestDenseRetriever:
    def test_encoder_initialized_once_at_construction(
        self, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        before = _FakeEncoder.init_calls
        retriever = fake_qdrant_client._owning_retriever
        fake_qdrant_client._search_results = [
            _FakeScoredPoint(id=1, score=0.9, payload={"chunk_id": "x"})
        ]
        retriever.query("test", top_k=1)
        retriever.query("test again", top_k=1)
        assert _FakeEncoder.init_calls == before  # no new encoder built per query

    def test_query_propagates_scores_and_chunk_ids(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        fake_qdrant_client._search_results = [
            _FakeScoredPoint(id=1, score=0.87, payload={"chunk_id": "chunk_a"}),
            _FakeScoredPoint(id=2, score=0.55, payload={"chunk_id": "chunk_b"}),
        ]
        results = dense_retriever.query("aspirin dosage", top_k=2)
        assert results == [
            DenseResult(chunk_id=ChunkId("chunk_a"), score=0.87),
            DenseResult(chunk_id=ChunkId("chunk_b"), score=0.55),
        ]

    def test_query_skips_hits_missing_chunk_id_payload(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        fake_qdrant_client._search_results = [
            _FakeScoredPoint(id=1, score=0.9, payload={}),
            _FakeScoredPoint(id=2, score=0.5, payload={"chunk_id": "chunk_b"}),
        ]
        results = dense_retriever.query("q", top_k=2)
        assert len(results) == 1
        assert results[0].chunk_id == ChunkId("chunk_b")

    def test_empty_query_returns_empty_without_calling_qdrant(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        fake_qdrant_client._raise_on_search = True  # would blow up if called
        assert dense_retriever.query("   ", top_k=5) == []
        assert fake_qdrant_client.query_calls == 0

    def test_query_uses_the_current_qdrant_api(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        dense_retriever.query("aspirin", top_k=3)
        assert fake_qdrant_client.query_calls == 1

    def test_retrievers_for_other_collections_share_one_encoder(
        self, dense_retriever: DenseRetriever
    ) -> None:
        before = _FakeEncoder.init_calls
        other = dense_retriever.backend.retriever("rasvcx_chunks_v_other")
        other.query("q", top_k=1)
        assert other.collection_name == "rasvcx_chunks_v_other"
        assert _FakeEncoder.init_calls == before

    def test_build_collection_verifies_point_count(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        backend = dense_retriever.backend
        assert backend.build_collection("c1", [(ChunkId("a"), "t1"), (ChunkId("b"), "t2")]) == 2
        assert backend.collection_count("c1") == 2
        assert backend.collection_count("missing") is None

    def test_build_collection_rejects_partial_index(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A backend that silently drops points must not yield a publishable index.
        monkeypatch.setattr(fake_qdrant_client, "upsert", lambda collection_name, points: None)
        with pytest.raises(QdrantUnavailableError, match="expected 2"):
            dense_retriever.backend.build_collection(
                "c1", [(ChunkId("a"), "t1"), (ChunkId("b"), "t2")]
            )

    def test_invalid_top_k_raises(self, dense_retriever: DenseRetriever) -> None:
        with pytest.raises(ValueError):
            dense_retriever.query("x", top_k=0)

    def test_qdrant_unavailable_on_search_raises_typed_error(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        fake_qdrant_client._raise_on_search = True
        with pytest.raises(QdrantUnavailableError):
            dense_retriever.query("aspirin", top_k=3)

    def test_qdrant_unavailable_on_connect_raises_typed_error(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        fake_qdrant_client._raise_on_get_collections = True
        with pytest.raises(QdrantUnavailableError):
            dense_retriever.ensure_collection()

    def test_upsert_requires_nonempty_documents(self, dense_retriever: DenseRetriever) -> None:
        with pytest.raises(ValueError):
            dense_retriever.upsert([])

    def test_upsert_produces_deterministic_point_ids_across_calls(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        dense_retriever.upsert([(ChunkId("stable_chunk"), "some evidence text")])
        first_id = fake_qdrant_client.upserted_points[0].id
        fake_qdrant_client.upserted_points.clear()
        dense_retriever.upsert([(ChunkId("stable_chunk"), "some evidence text")])
        second_id = fake_qdrant_client.upserted_points[0].id
        assert first_id == second_id, (
            "Qdrant point IDs must be deterministic (digest-based), not "
            "derived from Python's randomized hash()"
        )

    def test_upsert_point_ids_are_distinct_and_nonnegative(
        self, dense_retriever: DenseRetriever, fake_qdrant_client: _FakeQdrantClient
    ) -> None:
        dense_retriever.upsert(
            [(ChunkId("chunk_one"), "text one"), (ChunkId("chunk_two"), "text two")]
        )
        ids = [p.id for p in fake_qdrant_client.upserted_points]
        assert len(ids) == len(set(ids)), "distinct chunk_ids must not collide"
        assert all(isinstance(i, int) and i >= 0 for i in ids)


# ===========================================================================
# RRF fusion
# ===========================================================================

class TestReciprocalRankFusion:
    def test_rank_correctness_top_result_present_in_both_lists(self) -> None:
        bm25 = [BM25Result(chunk_id=ChunkId("a"), score=5.0), BM25Result(chunk_id=ChunkId("b"), score=2.0)]
        dense = [DenseResult(chunk_id=ChunkId("a"), score=0.9), DenseResult(chunk_id=ChunkId("c"), score=0.8)]
        fused = reciprocal_rank_fusion(bm25, dense, top_k=10)
        assert fused[0].chunk_id == ChunkId("a")

    def test_overlap_scores_are_additive(self) -> None:
        bm25 = [BM25Result(chunk_id=ChunkId("a"), score=1.0)]
        dense = [DenseResult(chunk_id=ChunkId("a"), score=1.0)]
        fused = reciprocal_rank_fusion(bm25, dense, top_k=10, rrf_k=60)
        expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
        assert math.isclose(fused[0].rrf_score, expected)

    def test_disjoint_lists_both_present(self) -> None:
        bm25 = [BM25Result(chunk_id=ChunkId("a"), score=1.0)]
        dense = [DenseResult(chunk_id=ChunkId("b"), score=1.0)]
        fused = reciprocal_rank_fusion(bm25, dense, top_k=10)
        chunk_ids = {r.chunk_id for r in fused}
        assert chunk_ids == {ChunkId("a"), ChunkId("b")}

    def test_top_k_bound_respected(self) -> None:
        bm25 = [BM25Result(chunk_id=ChunkId(f"c{i}"), score=float(10 - i)) for i in range(10)]
        fused = reciprocal_rank_fusion(bm25, [], top_k=3)
        assert len(fused) == 3

    def test_invalid_top_k_raises(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([], [], top_k=0)

    def test_invalid_rrf_k_raises(self) -> None:
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([], [], top_k=5, rrf_k=0)

    def test_both_empty_returns_empty(self) -> None:
        assert reciprocal_rank_fusion([], [], top_k=5) == []

    def test_deterministic_ordering_for_equal_scores(self) -> None:
        bm25 = [BM25Result(chunk_id=ChunkId("a"), score=1.0), BM25Result(chunk_id=ChunkId("b"), score=1.0)]
        f1 = reciprocal_rank_fusion(bm25, [], top_k=10)
        f2 = reciprocal_rank_fusion(bm25, [], top_k=10)
        assert [r.chunk_id for r in f1] == [r.chunk_id for r in f2]

    def test_preserves_original_bm25_and_dense_scores(self) -> None:
        bm25 = [BM25Result(chunk_id=ChunkId("a"), score=3.3)]
        dense = [DenseResult(chunk_id=ChunkId("a"), score=0.42)]
        fused = reciprocal_rank_fusion(bm25, dense, top_k=10)
        assert fused[0].bm25_score == 3.3
        assert fused[0].dense_score == 0.42

    def test_result_missing_from_one_list_has_none_score(self) -> None:
        bm25 = [BM25Result(chunk_id=ChunkId("a"), score=3.3)]
        fused = reciprocal_rank_fusion(bm25, [], top_k=10)
        assert fused[0].dense_score is None


# ===========================================================================
# Chunking
# ===========================================================================

class TestChunkingFixed:
    def test_fixed_strategy_basic(self) -> None:
        text = " ".join(f"word{i}" for i in range(50))
        config = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=20, overlap=0)
        chunks = chunk_fixed("doc1", text, config)
        assert len(chunks) == 3  # 20 + 20 + 10
        assert chunks[0].text.split() == [f"word{i}" for i in range(20)]

    def test_fixed_strategy_overlap_repeats_tokens(self) -> None:
        text = " ".join(f"w{i}" for i in range(30))
        config = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=10, overlap=5)
        chunks = chunk_fixed("doc1", text, config)
        first_tail = chunks[0].text.split()[-5:]
        second_head = chunks[1].text.split()[:5]
        assert first_tail == second_head

    def test_empty_document_produces_zero_chunks(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.FIXED)
        assert chunk_fixed("doc1", "", config) == []
        assert chunk_fixed("doc1", "   ", config) == []

    def test_stable_chunk_ids(self) -> None:
        text = " ".join(f"w{i}" for i in range(25))
        config = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=10, overlap=0)
        chunks1 = chunk_fixed("doc1", text, config)
        chunks2 = chunk_fixed("doc1", text, config)
        assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]
        assert chunks1[0].chunk_id == "doc1__chunk_0000"

    def test_overlap_clamped_to_max_tokens_minus_one(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=5, overlap=100)
        assert config.effective_overlap == 4

    def test_invalid_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError):
            ChunkConfig(max_tokens=0)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError):
            ChunkConfig(overlap=-1)


class TestChunkingSentenceWindow:
    def test_sentence_window_basic(self) -> None:
        text = "First sentence here. Second sentence here. Third one too."
        config = ChunkConfig(strategy=ChunkStrategy.SENTENCE_WINDOW, max_tokens=6)
        chunks = chunk_sentence_window("doc2", text, config)
        assert len(chunks) >= 1
        assert all(c.doc_id == "doc2" for c in chunks)

    def test_long_sentence_exceeding_max_tokens_becomes_its_own_chunk(self) -> None:
        long_sentence = " ".join(f"tok{i}" for i in range(50)) + "."
        config = ChunkConfig(strategy=ChunkStrategy.SENTENCE_WINDOW, max_tokens=10)
        chunks = chunk_sentence_window("doc3", long_sentence, config)
        assert any(len(c.text.split()) > 10 for c in chunks), (
            "an over-long sentence must still be preserved as a single chunk, "
            "not silently truncated or dropped"
        )

    def test_empty_document_produces_zero_chunks(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.SENTENCE_WINDOW)
        assert chunk_sentence_window("doc4", "", config) == []

    def test_stable_chunk_ids(self) -> None:
        text = "Sentence one. Sentence two. Sentence three."
        config = ChunkConfig(strategy=ChunkStrategy.SENTENCE_WINDOW, max_tokens=4)
        c1 = chunk_sentence_window("doc5", text, config)
        c2 = chunk_sentence_window("doc5", text, config)
        assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]


class TestChunkDocumentDispatch:
    def test_dispatches_to_fixed(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=5)
        result = chunk_document("d", "a b c d e f g h", config)
        assert result == chunk_fixed("d", "a b c d e f g h", config)

    def test_dispatches_to_sentence_window(self) -> None:
        config = ChunkConfig(strategy=ChunkStrategy.SENTENCE_WINDOW, max_tokens=5)
        text = "One sentence. Another sentence."
        assert chunk_document("d", text, config) == chunk_sentence_window("d", text, config)


# ===========================================================================
# Metadata filtering
# ===========================================================================

def _fused(chunk_id: str, score: float = 1.0) -> FusedResult:
    return FusedResult(chunk_id=ChunkId(chunk_id), rrf_score=score, bm25_score=None, dense_score=None)


class TestMetadataFilter:
    def test_no_criteria_returns_input_unchanged(self) -> None:
        results = [_fused("a"), _fused("b")]
        out = filter_by_metadata(results, {}, MetadataFilterCriteria())
        assert out == results

    def test_allowlist_excludes_non_matching_known_value(self) -> None:
        results = [_fused("a"), _fused("b")]
        prov = {
            "a": ProvenanceSnapshot(chunk_id="a", jurisdiction="US", population=UNKNOWN, date=UNKNOWN, dosage_context=UNKNOWN),
            "b": ProvenanceSnapshot(chunk_id="b", jurisdiction="EU", population=UNKNOWN, date=UNKNOWN, dosage_context=UNKNOWN),
        }
        criteria = MetadataFilterCriteria(allowed_jurisdictions=("US",))
        out = filter_by_metadata(results, prov, criteria)
        assert [r.chunk_id for r in out] == [ChunkId("a")]

    def test_unknown_provenance_always_passes_allowlist(self) -> None:
        results = [_fused("a")]
        prov = {"a": ProvenanceSnapshot(chunk_id="a", jurisdiction=UNKNOWN, population=UNKNOWN, date=UNKNOWN, dosage_context=UNKNOWN)}
        criteria = MetadataFilterCriteria(allowed_jurisdictions=("US",))
        out = filter_by_metadata(results, prov, criteria)
        assert [r.chunk_id for r in out] == [ChunkId("a")], (
            "UNKNOWN must never be silently excluded by a filter -- it is "
            "conservative, not a mismatch"
        )

    def test_date_range_filters_known_dates(self) -> None:
        results = [_fused("old"), _fused("new")]
        prov = {
            "old": ProvenanceSnapshot(chunk_id="old", jurisdiction=UNKNOWN, population=UNKNOWN, date="2010-01-01", dosage_context=UNKNOWN),
            "new": ProvenanceSnapshot(chunk_id="new", jurisdiction=UNKNOWN, population=UNKNOWN, date="2023-01-01", dosage_context=UNKNOWN),
        }
        criteria = MetadataFilterCriteria(min_date="2020-01-01")
        out = filter_by_metadata(results, prov, criteria)
        assert [r.chunk_id for r in out] == [ChunkId("new")]

    def test_unknown_date_always_passes_date_range(self) -> None:
        results = [_fused("a")]
        prov = {"a": ProvenanceSnapshot(chunk_id="a", jurisdiction=UNKNOWN, population=UNKNOWN, date=UNKNOWN, dosage_context=UNKNOWN)}
        criteria = MetadataFilterCriteria(min_date="2020-01-01", max_date="2025-01-01")
        out = filter_by_metadata(results, prov, criteria)
        assert [r.chunk_id for r in out] == [ChunkId("a")]

    def test_missing_provenance_entry_passes_through_fail_open(self) -> None:
        results = [_fused("a"), _fused("b")]
        prov = {"a": ProvenanceSnapshot(chunk_id="a", jurisdiction="US", population=UNKNOWN, date=UNKNOWN, dosage_context=UNKNOWN)}
        criteria = MetadataFilterCriteria(allowed_jurisdictions=("US",))
        out = filter_by_metadata(results, prov, criteria)
        assert ChunkId("b") in {r.chunk_id for r in out}

    def test_order_preserved(self) -> None:
        results = [_fused("a", 3.0), _fused("b", 2.0), _fused("c", 1.0)]
        out = filter_by_metadata(results, {}, MetadataFilterCriteria())
        assert [r.chunk_id for r in out] == [ChunkId("a"), ChunkId("b"), ChunkId("c")]


# ===========================================================================
# Targeted retrieval
# ===========================================================================

class _StubBM25Index:
    def __init__(self, results) -> None:
        self._results = results
        self.queries = []

    def query(self, query_text: str, top_k: int):
        self.queries.append((query_text, top_k))
        return self._results[:top_k]


class _StubDenseRetriever:
    def __init__(self, results=None, raise_unavailable: bool = False) -> None:
        self._results = results or []
        self._raise = raise_unavailable
        self.queries = []

    def query(self, query_text: str, top_k: int):
        self.queries.append((query_text, top_k))
        if self._raise:
            raise QdrantUnavailableError("simulated Qdrant outage")
        return self._results[:top_k]


class TestTargetedRetrieval:
    def test_returns_existing_unchanged_when_no_new_results(self) -> None:
        existing = [_fused("a", 1.0)]
        bm25 = _StubBM25Index(results=[])
        result = run_targeted_retrieval(
            "some query", existing, bm25, None, TargetedRetrievalConfig(augment_query=False)
        )
        assert result.merged_results == existing
        assert result.new_chunk_ids == frozenset()

    def test_merges_new_bm25_chunks_into_results(self) -> None:
        existing = [_fused("a", 1.0)]
        bm25 = _StubBM25Index(results=[BM25Result(chunk_id=ChunkId("b"), score=5.0)])
        result = run_targeted_retrieval(
            "query text", existing, bm25, None, TargetedRetrievalConfig(augment_query=False)
        )
        chunk_ids = {r.chunk_id for r in result.merged_results}
        assert ChunkId("a") in chunk_ids
        assert ChunkId("b") in chunk_ids
        assert ChunkId("b") in result.new_chunk_ids

    def test_no_duplicate_explosion_for_overlapping_chunk(self) -> None:
        existing = [_fused("a", 1.0)]
        bm25 = _StubBM25Index(results=[BM25Result(chunk_id=ChunkId("a"), score=9.0)])
        result = run_targeted_retrieval(
            "query", existing, bm25, None, TargetedRetrievalConfig(augment_query=False)
        )
        chunk_ids = [r.chunk_id for r in result.merged_results]
        assert chunk_ids.count(ChunkId("a")) == 1
        assert result.new_chunk_ids == frozenset()  # "a" was already present

    def test_preserves_bm25_and_dense_score_provenance_no_conflation(self) -> None:
        # Regression test for the RRF/BM25 score-conflation bug: existing
        # results' rrf_score must never be re-labelled as a fresh BM25 score.
        existing = [
            FusedResult(chunk_id=ChunkId("a"), rrf_score=0.5, bm25_score=None, dense_score=0.77)
        ]
        bm25 = _StubBM25Index(results=[])
        dense = _StubDenseRetriever(results=[])
        result = run_targeted_retrieval(
            "query", existing, bm25, dense, TargetedRetrievalConfig(augment_query=False)
        )
        merged_a = next(r for r in result.merged_results if r.chunk_id == ChunkId("a"))
        assert merged_a.bm25_score is None, "must not fabricate a bm25_score from a prior rrf_score"
        assert merged_a.dense_score == 0.77, "original dense provenance must be preserved"

    def test_dense_unavailable_is_reported_not_hidden(self) -> None:
        existing = []
        bm25 = _StubBM25Index(results=[BM25Result(chunk_id=ChunkId("x"), score=2.0)])
        dense = _StubDenseRetriever(raise_unavailable=True)
        result = run_targeted_retrieval(
            "query", existing, bm25, dense, TargetedRetrievalConfig(augment_query=False)
        )
        assert ChunkId("x") in {r.chunk_id for r in result.merged_results}
        assert result.targeted_dense_count == 0
        # The BM25-only supplementary pass is labelled as degraded.
        assert result.dense_error is not None
        assert "QdrantUnavailableError" in result.dense_error

    def test_dense_available_reports_no_error(self) -> None:
        bm25 = _StubBM25Index(results=[BM25Result(chunk_id=ChunkId("x"), score=2.0)])
        dense = _StubDenseRetriever(results=[])
        result = run_targeted_retrieval(
            "query", [], bm25, dense, TargetedRetrievalConfig(augment_query=False)
        )
        assert result.dense_error is None

    def test_conditional_usage_is_caller_responsibility(self) -> None:
        # targeted_retrieval itself has no sufficiency-gate awareness; it
        # simply executes when called. This documents that contract.
        bm25 = _StubBM25Index(results=[BM25Result(chunk_id=ChunkId("y"), score=1.0)])
        run_targeted_retrieval("q", [], bm25, None, TargetedRetrievalConfig(augment_query=False))
        assert len(bm25.queries) == 1

    def test_empty_retrieval_both_sources_returns_existing(self) -> None:
        existing = [_fused("a"), _fused("b")]
        bm25 = _StubBM25Index(results=[])
        dense = _StubDenseRetriever(results=[])
        result = run_targeted_retrieval(
            "q", existing, bm25, dense, TargetedRetrievalConfig(augment_query=False)
        )
        assert result.merged_results == existing

    def test_bounded_result_size_respects_top_k_budget(self) -> None:
        existing = [_fused(f"e{i}", score=float(100 - i)) for i in range(5)]
        bm25 = _StubBM25Index(
            results=[BM25Result(chunk_id=ChunkId(f"b{i}"), score=float(50 - i)) for i in range(20)]
        )
        config = TargetedRetrievalConfig(bm25_top_k=5, dense_top_k=5, augment_query=False)
        result = run_targeted_retrieval("q", existing, bm25, None, config)
        assert len(result.merged_results) <= len(existing) + config.bm25_top_k + config.dense_top_k

    def test_invalid_config_rejected(self) -> None:
        with pytest.raises(ValueError):
            TargetedRetrievalConfig(bm25_top_k=0)
        with pytest.raises(ValueError):
            TargetedRetrievalConfig(dense_top_k=0)
        with pytest.raises(ValueError):
            TargetedRetrievalConfig(rrf_k=0)

    def test_query_augmentation_is_recorded_in_result(self) -> None:
        bm25 = _StubBM25Index(results=[])
        result = run_targeted_retrieval(
            "Does Aspirin 500mg interact with Warfarin",
            [], bm25, None, TargetedRetrievalConfig(augment_query=True),
        )
        assert bm25.queries[0][0] == result.augmented_query