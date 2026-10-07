"""Tests for evaluation/dataset.py — M15 File 7/38."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.dataset import (
    DatasetError,
    LeakageReport,
    check_leakage,
    compute_dataset_fingerprint,
    load_dataset,
    save_dataset,
)
from evaluation.schema import (
    SCHEMA_VERSION,
    CaseCategory,
    CorpusCondition,
    DatasetSplit,
    EvalCase,
    EvalDataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(case_id="c1", split=DatasetSplit.TEST, query="test query",
               chunk_ids=(), **kw) -> EvalCase:
    kw.setdefault("expected_chunk_ids", tuple(chunk_ids))
    return EvalCase(
        case_id=case_id,
        dataset_id="ds1",
        dataset_version="1.0.0",
        query=query,
        split=split,
        corpus_condition=CorpusCondition.CLEAN,
        category=CaseCategory.CLEAN,
        **kw,
    )


def _make_dataset(*cases) -> EvalDataset:
    return EvalDataset(dataset_id="ds1", version="1.0.0", cases=tuple(cases))


def _save_and_load(dataset: EvalDataset, tmp_path: Path) -> EvalDataset:
    p = tmp_path / "dataset.json"
    save_dataset(dataset, p)
    return load_dataset(p)


# ---------------------------------------------------------------------------
# compute_dataset_fingerprint
# ---------------------------------------------------------------------------

class TestDatasetFingerprint:
    def test_returns_hex_string(self):
        ds = _make_dataset(_make_case())
        fp = compute_dataset_fingerprint(ds)
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hex

    def test_deterministic(self):
        ds = _make_dataset(_make_case())
        assert compute_dataset_fingerprint(ds) == compute_dataset_fingerprint(ds)

    def test_different_query_different_fingerprint(self):
        ds1 = _make_dataset(_make_case(query="query one"))
        ds2 = _make_dataset(_make_case(query="query two"))
        assert compute_dataset_fingerprint(ds1) != compute_dataset_fingerprint(ds2)

    def test_different_case_id_different_fingerprint(self):
        ds1 = _make_dataset(_make_case(case_id="c1"))
        ds2 = _make_dataset(_make_case(case_id="c2"))
        assert compute_dataset_fingerprint(ds1) != compute_dataset_fingerprint(ds2)

    def test_order_independent(self):
        c1 = _make_case(case_id="c1", query="alpha")
        c2 = _make_case(case_id="c2", query="beta")
        ds1 = _make_dataset(c1, c2)
        ds2 = EvalDataset(dataset_id="ds1", version="1.0.0", cases=(c2, c1))
        # fingerprint sorts by case_id, so order doesn't matter
        assert compute_dataset_fingerprint(ds1) == compute_dataset_fingerprint(ds2)


# ---------------------------------------------------------------------------
# check_leakage
# ---------------------------------------------------------------------------

class TestCheckLeakage:
    def test_clean_dataset(self):
        ds = _make_dataset(
            _make_case(case_id="c1", split=DatasetSplit.DEV,
                       query="dev query unique"),
            _make_case(case_id="c2", split=DatasetSplit.TEST,
                       query="test query unique"),
        )
        report = check_leakage(ds)
        assert report.is_clean

    def test_query_overlap_warning(self):
        ds = _make_dataset(
            _make_case(case_id="c1", split=DatasetSplit.DEV, query="same query"),
            _make_case(case_id="c2", split=DatasetSplit.TEST, query="same query"),
        )
        report = check_leakage(ds)
        assert not report.has_fatal_errors
        assert any("Query text duplication" in w for w in report.warnings)

    def test_chunk_id_overlap_warning(self):
        ds = _make_dataset(
            _make_case(case_id="c1", split=DatasetSplit.DEV,
                       chunk_ids=["chunk_a"]),
            _make_case(case_id="c2", split=DatasetSplit.TEST,
                       chunk_ids=["chunk_a"]),
        )
        report = check_leakage(ds)
        assert not report.has_fatal_errors
        assert any("chunk" in w.lower() for w in report.warnings)

    def test_no_cross_split_overlap_in_clean(self):
        ds = _make_dataset(
            _make_case(case_id="c1", split=DatasetSplit.DEV),
            _make_case(case_id="c2", split=DatasetSplit.VAL),
            _make_case(case_id="c3", split=DatasetSplit.TEST),
        )
        report = check_leakage(ds)
        assert not report.has_fatal_errors

    def test_leakage_report_has_fatal_errors_property(self):
        report = LeakageReport(fatal_errors=["an error"])
        assert report.has_fatal_errors
        assert not report.is_clean

    def test_empty_dataset_is_clean(self):
        ds = EvalDataset(dataset_id="ds1", version="1.0.0", cases=())
        report = check_leakage(ds)
        assert report.is_clean


# ---------------------------------------------------------------------------
# save_dataset / load_dataset
# ---------------------------------------------------------------------------

class TestSaveAndLoad:
    def test_roundtrip_single_case(self, tmp_path):
        c = _make_case(
            case_id="c1",
            query="What are visiting hours?",
            split=DatasetSplit.TEST,
            expected_decision="answer",
            expected_chunk_ids=("chunk1", "chunk2"),
        )
        ds = _make_dataset(c)
        loaded = _save_and_load(ds, tmp_path)

        assert len(loaded) == 1
        lc = loaded.cases[0]
        assert lc.case_id == "c1"
        assert lc.query == "What are visiting hours?"
        assert lc.split == DatasetSplit.TEST
        assert lc.expected_decision == "answer"
        assert set(lc.expected_chunk_ids) == {"chunk1", "chunk2"}

    def test_roundtrip_all_optional_fields(self, tmp_path):
        c = _make_case(
            expected_conflict_labels={"p1": "genuine-conflict"},
            expected_claim_labels={"cl1": "supported"},
            expected_answer_verdict="verified",
            tags=("tag1",),
            notes="a note",
        )
        loaded = _save_and_load(_make_dataset(c), tmp_path)
        lc = loaded.cases[0]
        assert lc.expected_conflict_labels == {"p1": "genuine-conflict"}
        assert lc.expected_claim_labels == {"cl1": "supported"}
        assert lc.expected_answer_verdict == "verified"
        assert "tag1" in lc.tags
        assert lc.notes == "a note"

    def test_no_overwrite(self, tmp_path):
        ds = _make_dataset(_make_case())
        p = tmp_path / "dataset.json"
        save_dataset(ds, p)
        with pytest.raises(FileExistsError):
            save_dataset(ds, p)

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.json")

    def test_load_invalid_json_raises(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not valid json {{{{", encoding="utf-8")
        with pytest.raises(DatasetError, match="Invalid JSON"):
            load_dataset(p)

    def test_load_wrong_schema_version_raises(self, tmp_path):
        p = tmp_path / "ds.json"
        data = {
            "dataset_id": "ds1", "version": "1.0.0",
            "schema_version": SCHEMA_VERSION + 99, "cases": [],
        }
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(DatasetError, match="schema_version"):
            load_dataset(p)

    def test_load_missing_dataset_id_raises(self, tmp_path):
        p = tmp_path / "ds.json"
        p.write_text(json.dumps({"version": "1.0.0",
                                 "schema_version": SCHEMA_VERSION,
                                 "cases": []}), encoding="utf-8")
        with pytest.raises(DatasetError):
            load_dataset(p)

    def test_fingerprint_stable_after_roundtrip(self, tmp_path):
        c = _make_case(query="stable query")
        ds = _make_dataset(c)
        fp_before = compute_dataset_fingerprint(ds)
        loaded = _save_and_load(ds, tmp_path)
        fp_after = compute_dataset_fingerprint(loaded)
        assert fp_before == fp_after

    def test_multiple_cases_roundtrip(self, tmp_path):
        cases = [_make_case(case_id=f"c{i}") for i in range(5)]
        ds = _make_dataset(*cases)
        loaded = _save_and_load(ds, tmp_path)
        assert len(loaded) == 5
        loaded_ids = {c.case_id for c in loaded.cases}
        assert loaded_ids == {f"c{i}" for i in range(5)}

    def test_corpus_condition_preserved(self, tmp_path):
        c2 = EvalCase(
            case_id="c1", dataset_id="ds1", dataset_version="1.0.0",
            query="test", split=DatasetSplit.TEST,
            corpus_condition=CorpusCondition.POISONED_CONTRADICTION,
            category=CaseCategory.CONTRADICTION,
        )
        ds = EvalDataset(dataset_id="ds1", version="1.0.0", cases=(c2,))
        loaded = _save_and_load(ds, tmp_path)
        assert loaded.cases[0].corpus_condition == CorpusCondition.POISONED_CONTRADICTION

    def test_load_triggers_leakage_check(self, tmp_path):
        """Load should reject datasets with fatal leakage errors."""
        case_dict = {
            "case_id": "c1", "dataset_id": "ds1",
            "dataset_version": "1.0.0", "query": "test",
            "split": "test",
            "corpus_condition": "clean",
            "category": "clean",
            "schema_version": SCHEMA_VERSION,
        }
        data = {
            "dataset_id": "ds1", "version": "1.0.0",
            "schema_version": SCHEMA_VERSION,
            "cases": [case_dict, {**case_dict, "split": "dev"}],
        }
        p = tmp_path / "dup.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises((DatasetError, ValueError)):
            load_dataset(p)