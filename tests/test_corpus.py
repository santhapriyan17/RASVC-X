"""Tests for corpus.py and bridge.py (Module 13)."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import pytest
from rasvcx.retrieval.corpus import (
    CorpusDocument, CorpusStore, CorpusManifest, StaleIndexError,
    build_corpus_store, compute_corpus_fingerprint, load_corpus_json,
)
from rasvcx.retrieval.chunking import ChunkConfig, ChunkStrategy
from rasvcx.retrieval.bridge import make_retrieval_fn, DisabledRerankingService
from rasvcx.retrieval.bm25 import BM25Index
from rasvcx.schemas.common import QueryId, UNKNOWN
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth

_CFG = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=50, overlap=5)


def _doc(doc_id="d1", text="Drug A is recommended at 500mg daily for adults.", **kw):
    return CorpusDocument(doc_id=doc_id, title=f"T-{doc_id}", text=text,
                          source_type="clinical_guideline", is_synthetic=True, **kw)


def _rp():
    return RiskProfile(overall_risk_score=0.1, feature_scores=RiskFeatureScores(),
                       validation_depth=ValidationDepth.SHALLOW,
                       retrieval_retry_budget=0, nli_call_allowance=0)


def _query(text="dosage"):
    return QueryRequest(query_id=QueryId("q1"), raw_text=text, normalized_text=text.lower())


class TestCorpusDocument:
    def test_valid_doc(self):
        _doc()

    def test_empty_doc_id_raises(self):
        with pytest.raises(ValueError):
            CorpusDocument(doc_id="", title="T", text="text", source_type="clinical_guideline")

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            CorpusDocument(doc_id="d1", title="T", text="", source_type="clinical_guideline")

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValueError):
            CorpusDocument(doc_id="d1", title="T", text="text", source_type="bad_type")

    def test_provenance_none_becomes_unknown(self):
        doc = _doc(date=None, jurisdiction=None, population=None)
        prov = doc.to_provenance()
        assert prov.date is UNKNOWN
        assert prov.jurisdiction is UNKNOWN
        assert prov.population is UNKNOWN

    def test_provenance_values_preserved(self):
        doc = _doc(date="2023-01-01", jurisdiction="US", population="adults")
        prov = doc.to_provenance()
        assert prov.date == "2023-01-01"
        assert prov.jurisdiction == "US"
        assert prov.population == "adults"

    def test_serialise_round_trip(self):
        doc = _doc(date="2023-01-01")
        d = doc.to_dict()
        doc2 = CorpusDocument.from_dict(d)
        assert doc2.doc_id == doc.doc_id
        assert doc2.date == doc.date


class TestCorpusStore:
    def test_add_and_lookup(self):
        store, _ = build_corpus_store([_doc("d1")], _CFG)
        cid = list(store.chunk_ids())[0]
        text, prov = store.lookup(cid)
        assert text
        assert prov is not None

    def test_missing_lookup_returns_none(self):
        from rasvcx.schemas.common import ChunkId
        store = CorpusStore()
        assert store.lookup(ChunkId("nonexistent")) is None

    def test_duplicate_chunk_id_raises(self):
        from rasvcx.schemas.common import ChunkId
        from rasvcx.schemas.evidence import Provenance
        from rasvcx.schemas.common import SourceType
        store = CorpusStore()
        prov = Provenance(source_type=SourceType.CLINICAL_GUIDELINE, date=UNKNOWN,
                          jurisdiction=UNKNOWN, population=UNKNOWN, dosage_context=UNKNOWN)
        cid = ChunkId("c1")
        store.add(cid, "text a", prov)
        with pytest.raises(ValueError):
            store.add(cid, "text b", prov)

    def test_json_round_trip_unknown(self):
        doc = _doc("d1", text="Dosing not established.")
        store, _ = build_corpus_store([doc], _CFG)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "store.json"
            store.save(p)
            loaded = CorpusStore.load(p)
        cid = list(store.chunk_ids())[0]
        _, orig_prov = store.lookup(cid)
        _, load_prov = loaded.lookup(cid)
        assert load_prov.date is UNKNOWN
        assert load_prov.source_type == orig_prov.source_type

    def test_json_round_trip_known_values(self):
        doc = _doc("d1", date="2023-01-01", jurisdiction="US", population="adults")
        store, _ = build_corpus_store([doc], _CFG)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "store.json"
            store.save(p)
            loaded = CorpusStore.load(p)
        cid = list(store.chunk_ids())[0]
        _, prov = loaded.lookup(cid)
        assert prov.date == "2023-01-01"
        assert prov.jurisdiction == "US"

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            CorpusStore.load(Path("/nonexistent/store.json"))


class TestBuildCorpusStore:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_corpus_store([], _CFG)

    def test_duplicate_doc_id_raises(self):
        with pytest.raises(ValueError):
            build_corpus_store([_doc("d1"), _doc("d1")], _CFG)

    def test_chunk_ids_stable(self):
        doc = _doc("d1")
        _, pairs1 = build_corpus_store([doc], _CFG)
        _, pairs2 = build_corpus_store([doc], _CFG)
        assert [cid for cid, _ in pairs1] == [cid for cid, _ in pairs2]

    def test_all_chunks_in_store(self):
        store, pairs = build_corpus_store(
            [_doc("d1"), _doc("d2", text="Another drug B 100mg.")], _CFG)
        for cid, _ in pairs:
            assert store.lookup(cid) is not None


class TestFingerprint:
    def test_deterministic(self):
        docs = [_doc("d1"), _doc("d2", text="Second doc.")]
        assert compute_corpus_fingerprint(docs, _CFG) == compute_corpus_fingerprint(docs, _CFG)

    def test_different_content_different_fingerprint(self):
        f1 = compute_corpus_fingerprint([_doc("d1")], _CFG)
        f2 = compute_corpus_fingerprint([_doc("d1", text="Different text entirely.")], _CFG)
        assert f1 != f2

    def test_different_chunking_different_fingerprint(self):
        doc = [_doc("d1")]
        cfg2 = ChunkConfig(strategy=ChunkStrategy.FIXED, max_tokens=128, overlap=10)
        assert compute_corpus_fingerprint(doc, _CFG) != compute_corpus_fingerprint(doc, cfg2)


class TestManifest:
    def test_save_load(self):
        m = CorpusManifest(corpus_fingerprint="abc123", chunk_count=10,
                           doc_count=3, store_path="s.json", bm25_path="b.pkl",
                           qdrant_collection=None)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "manifest.json"
            m.save(p)
            m2 = CorpusManifest.load(p)
        assert m2.corpus_fingerprint == "abc123"
        assert m2.chunk_count == 10

    def test_validate_matching(self):
        m = CorpusManifest(corpus_fingerprint="abc", chunk_count=1,
                           doc_count=1, store_path="s", bm25_path="b",
                           qdrant_collection=None)
        m.validate_against("abc")

    def test_validate_mismatch_raises(self):
        m = CorpusManifest(corpus_fingerprint="abc", chunk_count=1,
                           doc_count=1, store_path="s", bm25_path="b",
                           qdrant_collection=None)
        with pytest.raises(StaleIndexError):
            m.validate_against("different")


class TestSmokeCorpus:
    def test_loads(self):
        docs = load_corpus_json(Path("corpus/smoke_corpus.json"))
        assert len(docs) == 10

    def test_all_synthetic(self):
        docs = load_corpus_json(Path("corpus/smoke_corpus.json"))
        assert all(d.is_synthetic for d in docs)

    def test_conflict_pair_present(self):
        docs = load_corpus_json(Path("corpus/smoke_corpus.json"))
        texts = " ".join(d.text for d in docs)
        assert "75mg" in texts and "50mg" in texts


class TestBridge:
    def _make(self):
        docs = load_corpus_json(Path("corpus/smoke_corpus.json"))
        store, pairs = build_corpus_store(docs, ChunkConfig(
            strategy=ChunkStrategy.FIXED, max_tokens=256, overlap=32))
        index = BM25Index.build(pairs)
        return store, index

    def test_retrieval_fn_populates_bundle(self):
        store, index = self._make()
        fn = make_retrieval_fn(index, None, store, bm25_top_k=10, rrf_top_k=5)
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_rp())
        fn(_query("visiting hours"), _rp(), bundle)
        assert len(bundle.evidence_items) > 0

    def test_retrieval_records_retrieval_call(self):
        store, index = self._make()
        fn = make_retrieval_fn(index, None, store, bm25_top_k=10, rrf_top_k=5)
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_rp())
        fn(_query("visiting hours"), _rp(), bundle)
        assert bundle.metadata.retrieval_calls == 1

    def test_rerank_score_is_none(self):
        store, index = self._make()
        fn = make_retrieval_fn(index, None, store, bm25_top_k=10, rrf_top_k=5)
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_rp())
        fn(_query("drug dosage"), _rp(), bundle)
        for item in bundle.evidence_items.values():
            assert item.rerank_score is None

    def test_provenance_preserved(self):
        store, index = self._make()
        fn = make_retrieval_fn(index, None, store, bm25_top_k=10, rrf_top_k=5)
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_rp())
        fn(_query("visiting hours"), _rp(), bundle)
        for item in bundle.evidence_items.values():
            assert item.provenance is not None
            assert item.text

    def test_empty_query_returns_empty(self):
        store, index = self._make()
        fn = make_retrieval_fn(index, None, store, bm25_top_k=10, rrf_top_k=5)
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_rp())
        fn(_query("xyzqwerty12345nonexistent"), _rp(), bundle)
        assert bundle.metadata.retrieval_calls == 1

    def test_disabled_reranking_leaves_none(self):
        store, index = self._make()
        fn = make_retrieval_fn(index, None, store)
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_rp())
        fn(_query("drug"), _rp(), bundle)
        svc = DisabledRerankingService()
        svc.rerank_bundle(bundle, "drug")
        for item in bundle.evidence_items.values():
            assert item.rerank_score is None

    def test_disabled_reranking_records_elapsed(self):
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_rp())
        svc = DisabledRerankingService()
        svc.rerank_bundle(bundle, "test")
        assert "reranking" in bundle.metadata.elapsed_per_stage