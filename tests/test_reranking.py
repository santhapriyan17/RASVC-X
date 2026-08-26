"""Tests for Module 4 — Reranking.

Covers:
  cross_encoder.py  — CrossEncoderConfig, RerankResult, CrossEncoderReranker
  reranker.py       — RerankingServiceConfig, RerankingService
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from rasvcx.schemas.common import (
    ChunkId,
    EvidenceItemId,
    QueryId,
    SourceType,
    UNKNOWN,
)
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.reranking.cross_encoder import (
    CrossEncoderConfig,
    CrossEncoderReranker,
    RerankResult,
    _DEFAULT_MAX_CANDIDATES,
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_MODEL_NAME,
)
from rasvcx.reranking.reranker import RerankingService, RerankingServiceConfig
from rasvcx.reranking import (
    CrossEncoderConfig as _CECfg,
    CrossEncoderReranker as _CER,
    RerankResult as _RR,
    RerankingService as _RS,
    RerankingServiceConfig as _RSC,
)


# ---------------------------------------------------------------------------
# Shared factories
# ---------------------------------------------------------------------------

def _provenance() -> Provenance:
    return Provenance(
        source_type=SourceType.OTHER,
        date=UNKNOWN,
        jurisdiction=UNKNOWN,
        population=UNKNOWN,
        dosage_context=UNKNOWN,
    )


def _item(n: int, retrieval_score: float = 1.0) -> EvidenceItem:
    return EvidenceItem(
        item_id=EvidenceItemId(f"item_{n}"),
        chunk_id=ChunkId(f"chunk_{n}"),
        text=f"evidence text {n}",
        retrieval_score=retrieval_score,
        provenance=_provenance(),
    )


def _bundle(*items: EvidenceItem) -> EvidenceBundle:
    b = EvidenceBundle(query_id=QueryId("q-test"), risk_profile=object())
    for it in items:
        b.evidence_items[it.item_id] = it
    return b


def _service_with_scores(
    scores: list[float],
    max_candidates: int = 30,
    pre_sort: bool = True,
) -> RerankingService:
    cfg = RerankingServiceConfig(
        cross_encoder=CrossEncoderConfig(max_candidates=max_candidates),
        pre_sort_by_retrieval=pre_sort,
    )
    with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        mock_load.return_value = mock_model
        svc = RerankingService(cfg)
    return svc


def _mock_reranker(
    scores: list[float],
    config: CrossEncoderConfig | None = None,
) -> CrossEncoderReranker:
    cfg = config or CrossEncoderConfig()
    with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        mock_load.return_value = mock_model
        return CrossEncoderReranker(cfg)


# ===========================================================================
# CrossEncoderConfig
# ===========================================================================

class TestCrossEncoderConfig:
    def test_defaults(self):
        cfg = CrossEncoderConfig()
        assert cfg.model_name == _DEFAULT_MODEL_NAME
        assert cfg.max_candidates == _DEFAULT_MAX_CANDIDATES
        assert cfg.batch_size == _DEFAULT_BATCH_SIZE
        assert cfg.device is None

    def test_custom_values(self):
        cfg = CrossEncoderConfig(
            model_name="my-model", max_candidates=10, batch_size=8, device="cuda"
        )
        assert cfg.model_name == "my-model"
        assert cfg.max_candidates == 10
        assert cfg.batch_size == 8
        assert cfg.device == "cuda"

    def test_max_candidates_zero_raises(self):
        with pytest.raises(ValueError, match="max_candidates"):
            CrossEncoderConfig(max_candidates=0)

    def test_max_candidates_negative_raises(self):
        with pytest.raises(ValueError, match="max_candidates"):
            CrossEncoderConfig(max_candidates=-1)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            CrossEncoderConfig(batch_size=0)

    def test_batch_size_negative_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            CrossEncoderConfig(batch_size=-5)

    def test_frozen(self):
        cfg = CrossEncoderConfig()
        with pytest.raises(Exception):
            cfg.max_candidates = 5  # type: ignore[misc]

    def test_max_candidates_one_valid(self):
        cfg = CrossEncoderConfig(max_candidates=1)
        assert cfg.max_candidates == 1

    def test_batch_size_one_valid(self):
        cfg = CrossEncoderConfig(batch_size=1)
        assert cfg.batch_size == 1


# ===========================================================================
# RerankResult
# ===========================================================================

class TestRerankResult:
    def test_fields_preserved(self):
        r = RerankResult(chunk_id=ChunkId("c1"), rerank_score=0.9, original_text="hello")
        assert r.chunk_id == ChunkId("c1")
        assert r.rerank_score == pytest.approx(0.9)
        assert r.original_text == "hello"

    def test_frozen(self):
        r = RerankResult(chunk_id=ChunkId("c1"), rerank_score=0.5, original_text="x")
        with pytest.raises(Exception):
            r.rerank_score = 1.0  # type: ignore[misc]

    def test_negative_score_stored(self):
        r = RerankResult(chunk_id=ChunkId("c1"), rerank_score=-0.3, original_text="t")
        assert r.rerank_score == pytest.approx(-0.3)


# ===========================================================================
# CrossEncoderReranker — initialisation
# ===========================================================================

class TestCrossEncoderRerankerInit:
    def test_model_loaded_once_at_init(self):
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_load.return_value = MagicMock()
            CrossEncoderReranker()
            assert mock_load.call_count == 1

    def test_default_config_used_when_none(self):
        with patch.object(CrossEncoderReranker, "_load_model", return_value=MagicMock()):
            r = CrossEncoderReranker()
            assert r.config == CrossEncoderConfig()

    def test_supplied_config_preserved(self):
        cfg = CrossEncoderConfig(max_candidates=5)
        with patch.object(CrossEncoderReranker, "_load_model", return_value=MagicMock()):
            r = CrossEncoderReranker(cfg)
            assert r.config.max_candidates == 5

    def test_missing_sentence_transformers_raises_importerror(self):
        with patch.object(CrossEncoderReranker, "_load_model") as mock:
            mock.side_effect = ImportError("sentence-transformers")
            with pytest.raises(ImportError):
                CrossEncoderReranker()


# ===========================================================================
# CrossEncoderReranker — rerank()
# ===========================================================================

class TestCrossEncoderRerankerRerank:
    def test_empty_candidates_returns_empty(self):
        r = _mock_reranker([])
        assert r.rerank("query", []) == []

    def test_empty_query_returns_empty(self):
        r = _mock_reranker([0.5])
        assert r.rerank("", [(ChunkId("c1"), "text")]) == []

    def test_whitespace_query_returns_empty(self):
        r = _mock_reranker([0.5])
        assert r.rerank("   ", [(ChunkId("c1"), "text")]) == []

    def test_single_candidate(self):
        r = _mock_reranker([0.75])
        results = r.rerank("query", [(ChunkId("c1"), "text one")])
        assert len(results) == 1
        assert results[0].chunk_id == ChunkId("c1")
        assert results[0].rerank_score == pytest.approx(0.75)
        assert results[0].original_text == "text one"

    def test_results_sorted_descending_by_score(self):
        scores = [0.2, 0.9, 0.5]
        candidates = [(ChunkId(f"c{i}"), f"text {i}") for i in range(3)]
        r = _mock_reranker(scores)
        results = r.rerank("query", candidates)
        returned_scores = [res.rerank_score for res in results]
        assert returned_scores == sorted(returned_scores, reverse=True)

    def test_tiebreaker_lexicographic_chunk_id(self):
        scores = [0.5, 0.5, 0.5]
        candidates = [
            (ChunkId("chunk_z"), "text z"),
            (ChunkId("chunk_a"), "text a"),
            (ChunkId("chunk_m"), "text m"),
        ]
        r = _mock_reranker(scores)
        results = r.rerank("query", candidates)
        assert [res.chunk_id for res in results] == ["chunk_a", "chunk_m", "chunk_z"]

    def test_chunk_ids_preserved(self):
        cids = [ChunkId("alpha"), ChunkId("beta"), ChunkId("gamma")]
        candidates = [(cid, f"text {cid}") for cid in cids]
        r = _mock_reranker([0.3, 0.7, 0.5])
        results = r.rerank("query", candidates)
        assert {res.chunk_id for res in results} == set(cids)

    def test_texts_preserved(self):
        candidates = [(ChunkId("c1"), "hello world"), (ChunkId("c2"), "foo bar")]
        r = _mock_reranker([0.6, 0.4])
        results = r.rerank("query", candidates)
        assert {res.original_text for res in results} == {"hello world", "foo bar"}

    def test_candidate_cap_enforced(self):
        cfg = CrossEncoderConfig(max_candidates=3)
        r = _mock_reranker([0.1, 0.5, 0.9], config=cfg)
        candidates = [(ChunkId(f"c{i}"), f"t{i}") for i in range(5)]
        results = r.rerank("query", candidates)
        assert len(results) == 3
        pairs_passed = r._model.predict.call_args[0][0]
        assert len(pairs_passed) == 3

    def test_cap_of_one(self):
        cfg = CrossEncoderConfig(max_candidates=1)
        r = _mock_reranker([0.8], config=cfg)
        candidates = [(ChunkId(f"c{i}"), f"t{i}") for i in range(10)]
        results = r.rerank("query", candidates)
        assert len(results) == 1

    def test_fewer_candidates_than_cap(self):
        r = _mock_reranker([float(i) * 0.1 for i in range(5)])
        candidates = [(ChunkId(f"c{i}"), f"t{i}") for i in range(5)]
        results = r.rerank("query", candidates)
        assert len(results) == 5

    def test_model_predict_called_with_correct_pairs(self):
        r = _mock_reranker([0.5, 0.6])
        candidates = [(ChunkId("c1"), "text one"), (ChunkId("c2"), "text two")]
        r.rerank("my query", candidates)
        pairs = r._model.predict.call_args[0][0]
        assert pairs == [["my query", "text one"], ["my query", "text two"]]

    def test_batch_size_propagated(self):
        cfg = CrossEncoderConfig(batch_size=4)
        r = _mock_reranker([0.5], config=cfg)
        r.rerank("q", [(ChunkId("c1"), "text")])
        kwargs = r._model.predict.call_args[1]
        assert kwargs.get("batch_size") == 4

    def test_model_error_raises_runtime_error(self):
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception("CUDA OOM")
            mock_load.return_value = mock_model
            r = CrossEncoderReranker()
        with pytest.raises(RuntimeError, match="CrossEncoder.predict()"):
            r.rerank("query", [(ChunkId("c1"), "text")])

    def test_deterministic_equal_inputs(self):
        scores = [0.7, 0.3, 0.5]
        candidates = [(ChunkId(f"c{i}"), f"text {i}") for i in range(3)]
        r = _mock_reranker(scores)
        r1 = r.rerank("query", candidates)
        r._model.predict.return_value = scores
        r2 = r.rerank("query", candidates)
        assert [res.chunk_id for res in r1] == [res.chunk_id for res in r2]

    def test_negative_scores_sorted_correctly(self):
        scores = [-0.5, -0.1, -0.9]
        candidates = [(ChunkId(f"c{i}"), f"t{i}") for i in range(3)]
        r = _mock_reranker(scores)
        results = r.rerank("q", candidates)
        assert results[0].rerank_score == pytest.approx(-0.1)
        assert results[-1].rerank_score == pytest.approx(-0.9)

    def test_score_cast_to_float(self):
        r = _mock_reranker([1, 0, 2])
        candidates = [(ChunkId(f"c{i}"), f"t{i}") for i in range(3)]
        results = r.rerank("q", candidates)
        for res in results:
            assert isinstance(res.rerank_score, float)


# ===========================================================================
# RerankingServiceConfig
# ===========================================================================

class TestRerankingServiceConfig:
    def test_defaults(self):
        cfg = RerankingServiceConfig()
        assert isinstance(cfg.cross_encoder, CrossEncoderConfig)
        assert cfg.pre_sort_by_retrieval is True

    def test_frozen(self):
        cfg = RerankingServiceConfig()
        with pytest.raises(Exception):
            cfg.pre_sort_by_retrieval = False  # type: ignore[misc]

    def test_custom_cross_encoder_config(self):
        ce = CrossEncoderConfig(max_candidates=5)
        cfg = RerankingServiceConfig(cross_encoder=ce)
        assert cfg.cross_encoder.max_candidates == 5


# ===========================================================================
# RerankingService — initialisation
# ===========================================================================

class TestRerankingServiceInit:
    def test_model_loaded_once(self):
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_load.return_value = MagicMock()
            RerankingService()
            assert mock_load.call_count == 1

    def test_config_accessible(self):
        with patch.object(CrossEncoderReranker, "_load_model", return_value=MagicMock()):
            svc = RerankingService()
            assert isinstance(svc.config, RerankingServiceConfig)


# ===========================================================================
# RerankingService.rerank_bundle — degenerate inputs
# ===========================================================================

class TestRerankBundleDegenerate:
    def test_empty_bundle_noop(self):
        svc = _service_with_scores([])
        bundle = _bundle()
        svc.rerank_bundle(bundle, "query")
        assert bundle.evidence_items == {}
        assert "reranking" in bundle.metadata.elapsed_per_stage

    def test_empty_query_leaves_scores_none(self):
        svc = _service_with_scores([0.5])
        bundle = _bundle(_item(1))
        svc.rerank_bundle(bundle, "")
        assert bundle.evidence_items[EvidenceItemId("item_1")].rerank_score is None

    def test_whitespace_query_leaves_scores_none(self):
        svc = _service_with_scores([0.5])
        bundle = _bundle(_item(1))
        svc.rerank_bundle(bundle, "   ")
        assert bundle.evidence_items[EvidenceItemId("item_1")].rerank_score is None

    def test_elapsed_recorded_on_empty_bundle(self):
        svc = _service_with_scores([])
        bundle = _bundle()
        svc.rerank_bundle(bundle, "query")
        assert bundle.metadata.elapsed_per_stage["reranking"] >= 0.0

    def test_elapsed_recorded_on_empty_query(self):
        svc = _service_with_scores([])
        bundle = _bundle(_item(1))
        svc.rerank_bundle(bundle, "")
        assert bundle.metadata.elapsed_per_stage["reranking"] >= 0.0


# ===========================================================================
# RerankingService.rerank_bundle — score attachment
# ===========================================================================

class TestRerankBundleScoreAttachment:
    def test_rerank_score_attached(self):
        svc = _service_with_scores([0.8])
        bundle = _bundle(_item(1))
        svc.rerank_bundle(bundle, "query")
        assert bundle.evidence_items[EvidenceItemId("item_1")].rerank_score == pytest.approx(0.8)

    def test_all_other_fields_preserved(self):
        original = _item(1, retrieval_score=0.42)
        svc = _service_with_scores([0.9])
        bundle = _bundle(original)
        svc.rerank_bundle(bundle, "query")
        scored = bundle.evidence_items[EvidenceItemId("item_1")]
        assert scored.item_id == original.item_id
        assert scored.chunk_id == original.chunk_id
        assert scored.text == original.text
        assert scored.retrieval_score == pytest.approx(0.42)
        assert scored.provenance == original.provenance
        assert scored.extracted_claim_ids == original.extracted_claim_ids

    def test_multiple_items_all_scored(self):
        items = [_item(i, retrieval_score=float(i)) for i in range(1, 4)]
        svc = _service_with_scores([0.3, 0.9, 0.6])
        bundle = _bundle(*items)
        svc.rerank_bundle(bundle, "query")
        for iid in [EvidenceItemId(f"item_{i}") for i in range(1, 4)]:
            assert bundle.evidence_items[iid].rerank_score is not None

    def test_rerank_score_is_float(self):
        svc = _service_with_scores([1])
        bundle = _bundle(_item(1))
        svc.rerank_bundle(bundle, "query")
        score = bundle.evidence_items[EvidenceItemId("item_1")].rerank_score
        assert isinstance(score, float)


# ===========================================================================
# RerankingService.rerank_bundle — ordering
# ===========================================================================

class TestRerankBundleOrdering:
    def test_bundle_ordered_descending_by_rerank_score(self):
        items = [_item(i, retrieval_score=float(i)) for i in range(1, 4)]
        svc = _service_with_scores([0.1, 0.9, 0.5])
        bundle = _bundle(*items)
        svc.rerank_bundle(bundle, "query")
        scores = [
            it.rerank_score
            for it in bundle.evidence_items.values()
            if it.rerank_score is not None
        ]
        assert scores == sorted(scores, reverse=True)

    def test_items_beyond_cap_appended_after_scored(self):
        items = [_item(i, retrieval_score=float(i)) for i in range(1, 6)]
        svc = _service_with_scores([0.8, 0.6, 0.4], max_candidates=3)
        bundle = _bundle(*items)
        svc.rerank_bundle(bundle, "query")
        all_items = list(bundle.evidence_items.values())
        for it in all_items[:3]:
            assert it.rerank_score is not None
        for it in all_items[3:]:
            assert it.rerank_score is None

    def test_all_items_retained_when_some_beyond_cap(self):
        items = [_item(i) for i in range(1, 6)]
        svc = _service_with_scores([0.5, 0.3, 0.8], max_candidates=3)
        bundle = _bundle(*items)
        svc.rerank_bundle(bundle, "query")
        assert len(bundle.evidence_items) == 5


# ===========================================================================
# RerankingService.rerank_bundle — candidate cap
# ===========================================================================

class TestRerankBundleCap:
    def test_cap_limits_candidates_sent_to_model(self):
        items = [_item(i, retrieval_score=float(i)) for i in range(1, 11)]
        svc = _service_with_scores([float(i) * 0.1 for i in range(5)], max_candidates=5)
        bundle = _bundle(*items)
        svc.rerank_bundle(bundle, "query")
        pairs = svc._reranker._model.predict.call_args[0][0]
        assert len(pairs) == 5

    def test_pre_sort_keeps_highest_retrieval_for_reranking(self):
        items = [_item(i, retrieval_score=float(i)) for i in range(1, 6)]
        svc = _service_with_scores([0.7, 0.5, 0.3], max_candidates=3)
        bundle = _bundle(*items)
        svc.rerank_bundle(bundle, "query")
        pairs = svc._reranker._model.predict.call_args[0][0]
        sent_texts = {p[1] for p in pairs}
        assert "evidence text 5" in sent_texts
        assert "evidence text 4" in sent_texts
        assert "evidence text 3" in sent_texts
        assert "evidence text 1" not in sent_texts
        assert "evidence text 2" not in sent_texts


# ===========================================================================
# RerankingService.rerank_bundle — graceful degradation
# ===========================================================================

class TestRerankBundleGracefulDegradation:
    def _failing_service(self) -> RerankingService:
        with patch.object(CrossEncoderReranker, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.side_effect = Exception("GPU error")
            mock_load.return_value = mock_model
            return RerankingService()

    def test_model_error_does_not_raise(self):
        svc = self._failing_service()
        svc.rerank_bundle(_bundle(_item(1)), "query")

    def test_model_error_leaves_scores_none(self):
        svc = self._failing_service()
        bundle = _bundle(_item(1), _item(2))
        svc.rerank_bundle(bundle, "query")
        for it in bundle.evidence_items.values():
            assert it.rerank_score is None

    def test_model_error_records_elapsed(self):
        svc = self._failing_service()
        bundle = _bundle(_item(1))
        svc.rerank_bundle(bundle, "query")
        assert "reranking" in bundle.metadata.elapsed_per_stage


# ===========================================================================
# RerankingService.rerank_bundle — metadata
# ===========================================================================

class TestRerankBundleMetadata:
    def test_elapsed_recorded(self):
        svc = _service_with_scores([0.5])
        bundle = _bundle(_item(1))
        svc.rerank_bundle(bundle, "query")
        assert "reranking" in bundle.metadata.elapsed_per_stage
        assert bundle.metadata.elapsed_per_stage["reranking"] >= 0.0

    def test_retrieval_call_count_unchanged(self):
        svc = _service_with_scores([0.5])
        bundle = _bundle(_item(1))
        before = bundle.metadata.retrieval_calls
        svc.rerank_bundle(bundle, "query")
        assert bundle.metadata.retrieval_calls == before

    def test_nli_call_count_unchanged(self):
        svc = _service_with_scores([0.5])
        bundle = _bundle(_item(1))
        before = bundle.metadata.nli_calls
        svc.rerank_bundle(bundle, "query")
        assert bundle.metadata.nli_calls == before


# ===========================================================================
# EvidenceItem immutability
# ===========================================================================

class TestEvidenceItemImmutabilityRespected:
    def test_original_python_object_not_mutated(self):
        original = _item(1)
        svc = _service_with_scores([0.9])
        bundle = _bundle(original)
        svc.rerank_bundle(bundle, "query")
        assert original.rerank_score is None

    def test_bundle_item_is_new_object(self):
        original = _item(1)
        svc = _service_with_scores([0.9])
        bundle = _bundle(original)
        svc.rerank_bundle(bundle, "query")
        bundle_item = bundle.evidence_items[EvidenceItemId("item_1")]
        assert bundle_item is not original
        assert bundle_item.rerank_score == pytest.approx(0.9)


# ===========================================================================
# Package __init__ exports
# ===========================================================================

class TestPackageExports:
    def test_all_public_names_importable(self):
        assert _CECfg is CrossEncoderConfig
        assert _CER is CrossEncoderReranker
        assert _RR is RerankResult
        assert _RS is RerankingService
        assert _RSC is RerankingServiceConfig

    def test_all_declared_in_dunder_all(self):
        import rasvcx.reranking as pkg
        for name in pkg.__all__:
            assert hasattr(pkg, name)