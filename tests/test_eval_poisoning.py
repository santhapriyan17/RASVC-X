"""Tests for evaluation/poisoning.py — M15 File 16/38."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.poisoning import (
    AdversarialDocument,
    IsolationError,
    PoisoningExperiment,
    _sha256_of_bytes,
    build_poisoned_index,
)
from evaluation.schema import CorpusCondition


_ACTIVE_STORE = Path("corpus/corpus_store.json")
_ACTIVE_BM25 = Path("corpus/bm25_index.pkl")


def _make_adv_doc(doc_id="adv_001") -> AdversarialDocument:
    return AdversarialDocument(
        doc_id=doc_id,
        title="Synthetic adversarial document",
        text="Drug Z is recommended at 9999mg. This is a synthetic test artifact.",
        condition=CorpusCondition.POISONED_CONTRADICTION,
    )


def _make_experiment(exp_dir: Path, docs=None) -> PoisoningExperiment:
    return PoisoningExperiment(
        experiment_id="test_exp_001",
        condition=CorpusCondition.POISONED_CONTRADICTION,
        active_corpus_store_path=_ACTIVE_STORE,
        active_bm25_path=_ACTIVE_BM25,
        experiment_dir=exp_dir,
        adversarial_docs=docs or [_make_adv_doc()],
    )


# ---------------------------------------------------------------------------
# AdversarialDocument validation
# ---------------------------------------------------------------------------

class TestAdversarialDocument:
    def test_valid_construction(self):
        doc = _make_adv_doc()
        assert doc.doc_id == "adv_001"
        assert doc.title == "Synthetic adversarial document"

    def test_empty_doc_id_raises(self):
        with pytest.raises(ValueError, match="doc_id"):
            AdversarialDocument(
                doc_id="",
                title="t",
                text="text",
                condition=CorpusCondition.POISONED_CONTRADICTION,
            )

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="text"):
            AdversarialDocument(
                doc_id="adv_x",
                title="t",
                text="",
                condition=CorpusCondition.POISONED_CONTRADICTION,
            )

    def test_clean_condition_raises(self):
        with pytest.raises(ValueError, match="CLEAN"):
            AdversarialDocument(
                doc_id="adv_x",
                title="t",
                text="text",
                condition=CorpusCondition.CLEAN,
            )


# ---------------------------------------------------------------------------
# PoisoningExperiment validation
# ---------------------------------------------------------------------------

class TestPoisoningExperiment:
    def test_clean_condition_raises(self, tmp_path):
        with pytest.raises(ValueError, match="CLEAN"):
            PoisoningExperiment(
                experiment_id="x",
                condition=CorpusCondition.CLEAN,
                active_corpus_store_path=_ACTIVE_STORE,
                active_bm25_path=_ACTIVE_BM25,
                experiment_dir=tmp_path,
            )


# ---------------------------------------------------------------------------
# IsolationError
# ---------------------------------------------------------------------------

class TestIsolationError:
    def test_same_dir_raises(self, tmp_path):
        exp = PoisoningExperiment(
            experiment_id="x",
            condition=CorpusCondition.POISONED_CONTRADICTION,
            active_corpus_store_path=_ACTIVE_STORE,
            active_bm25_path=_ACTIVE_BM25,
            experiment_dir=_ACTIVE_STORE.parent,  # same dir
            adversarial_docs=[_make_adv_doc()],
        )
        with pytest.raises(IsolationError):
            build_poisoned_index(exp)

    def test_different_dir_no_error(self, tmp_path):
        exp = _make_experiment(tmp_path / "exp_isolated")
        # Should not raise IsolationError
        config = build_poisoned_index(exp)
        assert config is not None


# ---------------------------------------------------------------------------
# build_poisoned_index
# ---------------------------------------------------------------------------

class TestBuildPoisonedIndex:
    def test_returns_corpus_config(self, tmp_path):
        from evaluation.schema import CorpusConfig
        config = build_poisoned_index(_make_experiment(tmp_path / "exp1"))
        assert isinstance(config, CorpusConfig)

    def test_experiment_store_written(self, tmp_path):
        config = build_poisoned_index(_make_experiment(tmp_path / "exp1"))
        assert config.store_path.exists()

    def test_experiment_bm25_written(self, tmp_path):
        config = build_poisoned_index(_make_experiment(tmp_path / "exp1"))
        assert config.bm25_path.exists()

    def test_experiment_manifest_written(self, tmp_path):
        exp_dir = tmp_path / "exp1"
        build_poisoned_index(_make_experiment(exp_dir))
        assert (exp_dir / "manifest.json").exists()

    def test_condition_in_config(self, tmp_path):
        config = build_poisoned_index(_make_experiment(tmp_path / "exp1"))
        assert config.condition == CorpusCondition.POISONED_CONTRADICTION

    def test_fingerprint_non_empty(self, tmp_path):
        config = build_poisoned_index(_make_experiment(tmp_path / "exp1"))
        assert len(config.fingerprint) == 64

    def test_active_store_bytes_unchanged(self, tmp_path):
        """Core isolation test: active corpus must not be modified."""
        before = _sha256_of_bytes(_ACTIVE_STORE.read_bytes())
        build_poisoned_index(_make_experiment(tmp_path / "exp_iso"))
        after = _sha256_of_bytes(_ACTIVE_STORE.read_bytes())
        assert before == after, (
            f"Active corpus was modified! before={before[:16]} after={after[:16]}"
        )

    def test_active_bm25_bytes_unchanged(self, tmp_path):
        before = _sha256_of_bytes(_ACTIVE_BM25.read_bytes())
        build_poisoned_index(_make_experiment(tmp_path / "exp_bm25"))
        after = _sha256_of_bytes(_ACTIVE_BM25.read_bytes())
        assert before == after

    def test_experiment_fingerprint_differs_from_active(self, tmp_path):
        active_fp = _sha256_of_bytes(_ACTIVE_STORE.read_bytes())
        config = build_poisoned_index(_make_experiment(tmp_path / "exp1"))
        assert config.fingerprint != active_fp

    def test_manifest_records_parent_fingerprint(self, tmp_path):
        exp_dir = tmp_path / "exp1"
        active_fp = _sha256_of_bytes(_ACTIVE_STORE.read_bytes())
        build_poisoned_index(_make_experiment(exp_dir))
        manifest = json.loads((exp_dir / "manifest.json").read_text())
        assert manifest["parent_fingerprint"] == active_fp

    def test_manifest_records_injected_doc_ids(self, tmp_path):
        exp_dir = tmp_path / "exp1"
        build_poisoned_index(_make_experiment(exp_dir, docs=[_make_adv_doc("adv_x")]))
        manifest = json.loads((exp_dir / "manifest.json").read_text())
        assert "adv_x" in manifest["injected_doc_ids"]

    def test_manifest_records_condition(self, tmp_path):
        exp_dir = tmp_path / "exp1"
        build_poisoned_index(_make_experiment(exp_dir))
        manifest = json.loads((exp_dir / "manifest.json").read_text())
        assert manifest["condition"] == "poisoned_contradiction"

    def test_poisoned_store_larger_than_active(self, tmp_path):
        """Poisoned store should have more chunks than active (adversarial added)."""
        from rasvcx.retrieval.corpus import CorpusStore
        active_store = CorpusStore.load(str(_ACTIVE_STORE))
        active_chunk_count = len(list(active_store.chunk_ids()))

        config = build_poisoned_index(_make_experiment(tmp_path / "exp1"))
        poisoned_store = CorpusStore.load(str(config.store_path))
        poisoned_chunk_count = len(list(poisoned_store.chunk_ids()))

        assert poisoned_chunk_count > active_chunk_count

    def test_multiple_adversarial_docs(self, tmp_path):
        docs = [
            _make_adv_doc(f"adv_{i:03d}") for i in range(3)
        ]
        exp_dir = tmp_path / "exp_multi"
        config = build_poisoned_index(
            _make_experiment(exp_dir, docs=docs)
        )
        manifest = json.loads((exp_dir / "manifest.json").read_text())
        assert len(manifest["injected_doc_ids"]) == 3

    def test_experiment_dir_isolated_from_active(self, tmp_path):
        exp_dir = tmp_path / "isolated_exp"
        config = build_poisoned_index(_make_experiment(exp_dir))
        # experiment paths must not equal active paths
        assert config.store_path.resolve() != _ACTIVE_STORE.resolve()
        assert config.bm25_path.resolve() != _ACTIVE_BM25.resolve()