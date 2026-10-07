"""Tests for evaluation/results.py — M15 File 17/38."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.results import (
    CaseResult,
    ResultWriter,
    RunRecord,
    load_case_results,
    load_run_record,
    list_run_records,
    make_run_id,
)
from evaluation.schema import CorpusCondition, DatasetSplit


# ---------------------------------------------------------------------------
# CaseResult
# ---------------------------------------------------------------------------

class TestCaseResult:
    def test_valid_completed(self):
        r = CaseResult(
            case_id="c1", status="completed", baseline_id="B3",
            corpus_condition=CorpusCondition.CLEAN,
            latency_seconds=0.5, data={"action": "answer"},
        )
        assert r.status == "completed"
        assert r.latency_seconds == 0.5

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="status"):
            CaseResult(case_id="c1", status="bad_status",
                       baseline_id="B3", corpus_condition=CorpusCondition.CLEAN)

    def test_error_requires_error_field(self):
        with pytest.raises(ValueError, match="error"):
            CaseResult(case_id="c1", status="error",
                       baseline_id="B3", corpus_condition=CorpusCondition.CLEAN)

    def test_error_with_error_field(self):
        r = CaseResult(case_id="c1", status="error", baseline_id="B3",
                       corpus_condition=CorpusCondition.CLEAN,
                       error="RuntimeError(test)")
        assert r.error == "RuntimeError(test)"

    def test_skipped_requires_skip_reason(self):
        with pytest.raises(ValueError, match="skip_reason"):
            CaseResult(case_id="c1", status="skipped",
                       baseline_id="B2", corpus_condition=CorpusCondition.CLEAN)

    def test_skipped_with_reason(self):
        r = CaseResult(case_id="c1", status="skipped", baseline_id="B2",
                       corpus_condition=CorpusCondition.CLEAN,
                       skip_reason="qdrant_unavailable")
        assert r.skip_reason == "qdrant_unavailable"

    def test_negative_latency_raises(self):
        with pytest.raises(ValueError, match="latency"):
            CaseResult(case_id="c1", status="completed", baseline_id="B3",
                       corpus_condition=CorpusCondition.CLEAN,
                       latency_seconds=-0.1)

    def test_empty_case_id_raises(self):
        with pytest.raises(ValueError, match="case_id"):
            CaseResult(case_id="", status="completed", baseline_id="B3",
                       corpus_condition=CorpusCondition.CLEAN)

    def test_to_dict_serialisable(self):
        r = CaseResult(case_id="c1", status="completed", baseline_id="B3",
                       corpus_condition=CorpusCondition.CLEAN)
        d = r.to_dict()
        assert json.dumps(d)  # must be JSON-serialisable

    def test_to_dict_corpus_condition_value(self):
        r = CaseResult(case_id="c1", status="completed", baseline_id="B3",
                       corpus_condition=CorpusCondition.POISONED_STALE)
        d = r.to_dict()
        assert d["corpus_condition"] == "poisoned_stale"

    def test_all_valid_statuses(self):
        for status in ("completed", "error", "rejected_overload",
                       "cancelled_deadline", "not_offered_deadline"):
            extra = {"error": "e"} if status == "error" else {}
            r = CaseResult(case_id="c1", status=status, baseline_id="B3",
                           corpus_condition=CorpusCondition.CLEAN, **extra)
            assert r.status == status

    def test_timestamp_set(self):
        r = CaseResult(case_id="c1", status="completed", baseline_id="B3",
                       corpus_condition=CorpusCondition.CLEAN)
        assert r.timestamp_utc and "T" in r.timestamp_utc

    def test_mock_llm_default_true(self):
        r = CaseResult(case_id="c1", status="completed", baseline_id="B3",
                       corpus_condition=CorpusCondition.CLEAN)
        assert r.mock_llm is True


# ---------------------------------------------------------------------------
# ResultWriter
# ---------------------------------------------------------------------------

class TestResultWriter:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "results.jsonl"
        w = ResultWriter(p)
        w.close()
        assert p.exists()

    def test_append_writes_line(self, tmp_path):
        p = tmp_path / "results.jsonl"
        with ResultWriter(p) as w:
            w.append(CaseResult(case_id="c1", status="completed",
                                baseline_id="B3",
                                corpus_condition=CorpusCondition.CLEAN))
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 1
        assert json.loads(lines[0])["case_id"] == "c1"

    def test_append_multiple(self, tmp_path):
        p = tmp_path / "results.jsonl"
        with ResultWriter(p) as w:
            for i in range(5):
                w.append(CaseResult(case_id=f"c{i}", status="completed",
                                    baseline_id="B3",
                                    corpus_condition=CorpusCondition.CLEAN))
        assert w.count == 5

    def test_no_overwrite_raises(self, tmp_path):
        p = tmp_path / "results.jsonl"
        ResultWriter(p).close()
        with pytest.raises(FileExistsError):
            ResultWriter(p)

    def test_append_after_close_raises(self, tmp_path):
        p = tmp_path / "results.jsonl"
        w = ResultWriter(p)
        w.close()
        with pytest.raises(RuntimeError, match="close"):
            w.append(CaseResult(case_id="c1", status="completed",
                                baseline_id="B3",
                                corpus_condition=CorpusCondition.CLEAN))

    def test_close_idempotent(self, tmp_path):
        p = tmp_path / "results.jsonl"
        w = ResultWriter(p)
        w.close()
        w.close()  # should not raise

    def test_count_property(self, tmp_path):
        p = tmp_path / "results.jsonl"
        with ResultWriter(p) as w:
            assert w.count == 0
            w.append(CaseResult(case_id="c1", status="completed",
                                baseline_id="B3",
                                corpus_condition=CorpusCondition.CLEAN))
            assert w.count == 1

    def test_path_property(self, tmp_path):
        p = tmp_path / "results.jsonl"
        w = ResultWriter(p)
        assert w.path == p
        w.close()


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------

class TestRunRecord:
    def _make_valid_record(self, run_id=None) -> RunRecord:
        return RunRecord(
            run_id=run_id or make_run_id(),
            baseline_id="B3",
            dataset_id="ds1",
            dataset_version="1.0.0",
            split=DatasetSplit.TEST,
            corpus_fingerprint="abc123def456abcd",
            corpus_condition=CorpusCondition.CLEAN,
            execution_mode="offline_test",
            mock_llm=True,
            total_cases=5,
            offered=5,
            not_offered_deadline=0,
            accepted=4,
            rejected_overload=1,
            cancelled_deadline=0,
            completed=4,
            error=0,
        )

    def test_check_accounting_valid(self):
        rec = self._make_valid_record()
        assert rec.check_accounting() is None

    def test_check_accounting_total_mismatch(self):
        rec = self._make_valid_record()
        object.__setattr__(rec, 'total_cases', 99)  # deliberately break
        assert rec.check_accounting() is not None

    def test_check_accounting_accepted_mismatch(self):
        rec = self._make_valid_record()
        object.__setattr__(rec, 'completed', 99)
        assert rec.check_accounting() is not None

    def test_to_dict_has_required_keys(self):
        rec = self._make_valid_record()
        d = rec.to_dict()
        assert "run_id" in d
        assert "accounting" in d
        assert "timing" in d
        assert "limitations" in d

    def test_limitations_present(self):
        rec = self._make_valid_record()
        d = rec.to_dict()
        assert "MockLLMClient" in d["limitations"]

    def test_split_serialised_as_value(self):
        rec = self._make_valid_record()
        d = rec.to_dict()
        assert d["split"] == "test"

    def test_corpus_condition_serialised_as_value(self):
        rec = self._make_valid_record()
        d = rec.to_dict()
        assert d["corpus_condition"] == "clean"

    def test_save_creates_file(self, tmp_path):
        rec = self._make_valid_record()
        path = rec.save(tmp_path)
        assert path.exists()

    def test_save_no_overwrite(self, tmp_path):
        rec = self._make_valid_record()
        rec.save(tmp_path)
        with pytest.raises(FileExistsError):
            rec.save(tmp_path)

    def test_save_filename_contains_run_id(self, tmp_path):
        run_id = make_run_id()
        rec = self._make_valid_record(run_id=run_id)
        path = rec.save(tmp_path)
        assert run_id in path.name

    def test_calibration_proxy_true_by_default(self):
        rec = self._make_valid_record()
        assert rec.calibration_proxy is True

    def test_json_serialisable(self, tmp_path):
        rec = self._make_valid_record()
        path = rec.save(tmp_path)
        loaded = json.loads(path.read_text())
        assert loaded["run_id"] == rec.run_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_make_run_id_unique(self):
        ids = {make_run_id() for _ in range(100)}
        assert len(ids) == 100

    def test_make_run_id_is_hex(self):
        rid = make_run_id()
        assert all(c in "0123456789abcdef" for c in rid)
        assert len(rid) == 32

    def test_load_case_results(self, tmp_path):
        p = tmp_path / "r.jsonl"
        with ResultWriter(p) as w:
            w.append(CaseResult(case_id="c1", status="completed",
                                baseline_id="B3",
                                corpus_condition=CorpusCondition.CLEAN))
            w.append(CaseResult(case_id="c2", status="error",
                                baseline_id="B3",
                                corpus_condition=CorpusCondition.CLEAN,
                                error="err"))
        rows = load_case_results(p)
        assert len(rows) == 2
        assert rows[0]["case_id"] == "c1"
        assert rows[1]["status"] == "error"

    def test_load_run_record(self, tmp_path):
        rec = RunRecord(
            run_id=make_run_id(), baseline_id="B3", dataset_id="ds1",
            dataset_version="1.0.0", split=DatasetSplit.TEST,
            corpus_fingerprint="abc123def456abcd",
            corpus_condition=CorpusCondition.CLEAN,
            execution_mode="offline_test", mock_llm=True,
        )
        path = rec.save(tmp_path)
        loaded = load_run_record(path)
        assert loaded["run_id"] == rec.run_id
        assert loaded["limitations"].startswith("MockLLMClient")

    def test_list_run_records_empty(self, tmp_path):
        assert list_run_records(tmp_path) == []

    def test_list_run_records_returns_paths(self, tmp_path):
        for _ in range(3):
            RunRecord(
                run_id=make_run_id(), baseline_id="B3", dataset_id="ds1",
                dataset_version="1.0.0", split=DatasetSplit.TEST,
                corpus_fingerprint="abc123def456abcd",
                corpus_condition=CorpusCondition.CLEAN,
                execution_mode="offline_test", mock_llm=True,
            ).save(tmp_path)
        records = list_run_records(tmp_path)
        assert len(records) == 3
        assert all(p.suffix == ".json" for p in records)

    def test_list_run_records_nonexistent_dir(self, tmp_path):
        assert list_run_records(tmp_path / "nonexistent") == []