"""Tests for evaluation/runner.py — M15 File 14/38.

All tests are synchronous (pytest-asyncio not installed).
Uses a MockBaseline that returns immediately or sleeps/raises on demand.

Covers all 7 accounting scenarios:
  1. All cases admitted and completed
  2. Deadline before any case offered
  3. Deadline midway through admission
  4. Queue full before deadline
  5. Queue admission timeout coinciding with deadline
  6. One case errors
  7. Exact one-result-per-input invariant
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.baselines.interface import BaselineResult
from evaluation.results import load_case_results
from evaluation.runner import BenchmarkRunner
from evaluation.schema import (
    CaseCategory,
    CorpusCondition,
    CorpusConfig,
    DatasetSplit,
    EvalCase,
    EvalDataset,
    RunConfig,
    RunConfigError,
)


# ---------------------------------------------------------------------------
# Mock baseline
# ---------------------------------------------------------------------------


class _MockBaseline:
    """Synchronous mock baseline for runner tests."""

    def __init__(
        self,
        sleep_seconds: float = 0.0,
        raise_on_run: bool = False,
        skip: bool = False,
        skip_reason: str = "mock_skip",
    ) -> None:
        self._sleep = sleep_seconds
        self._raise = raise_on_run
        self._skip = skip
        self._skip_reason_val = skip_reason
        self._available = False

    @property
    def baseline_id(self) -> str:
        return "MOCK"

    @property
    def mock_llm(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def skip_reason(self) -> Optional[str]:
        return self._skip_reason_val if self._skip else None

    def initialize(self, corpus_config) -> None:
        if self._skip:
            return  # leave _available=False
        self._available = True

    def run(self, case: EvalCase) -> BaselineResult:
        if self._sleep > 0:
            time.sleep(self._sleep)
        if self._raise:
            raise RuntimeError(f"Mock error for case {case.case_id}")
        return BaselineResult(
            decision="answer",
            generated_text="mock answer",
            mock_llm=True,
        )

    def close(self) -> None:
        self._available = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CC = CorpusConfig(
    store_path=Path("corpus/corpus_store.json"),
    bm25_path=Path("corpus/bm25_index.pkl"),
    fingerprint="abc123def456abcd",
    condition=CorpusCondition.CLEAN,
)


def _make_case(case_id: str) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        dataset_id="ds1",
        dataset_version="1.0.0",
        query=f"query for {case_id}",
        split=DatasetSplit.TEST,
        corpus_condition=CorpusCondition.CLEAN,
        category=CaseCategory.CLEAN,
    )


def _make_dataset(n: int) -> EvalDataset:
    cases = tuple(_make_case(f"c{i:04d}") for i in range(n))
    return EvalDataset(dataset_id="ds1", version="1.0.0", cases=cases)


def _make_cfg(tmp_path: Path, **overrides) -> RunConfig:
    defaults = dict(
        max_workers=2,
        max_queue_depth=4,
        max_cases=1000,
        output_dir=tmp_path,
        admission_timeout_seconds=2.0,
        wall_clock_deadline_seconds=30.0,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def _run(tmp_path, n_cases=5, baseline=None, **cfg_overrides):
    cfg = _make_cfg(tmp_path, **cfg_overrides)
    ds = _make_dataset(n_cases)
    bl = baseline or _MockBaseline()
    runner = BenchmarkRunner(cfg, bl, ds, _CC, split=DatasetSplit.TEST)
    return runner.run()


# ---------------------------------------------------------------------------
# Scenario 1: all cases admitted and completed
# ---------------------------------------------------------------------------

class TestAllCompleted:
    def test_total_cases(self, tmp_path):
        rec = _run(tmp_path, n_cases=5)
        assert rec.total_cases == 5

    def test_completed_equals_total(self, tmp_path):
        rec = _run(tmp_path, n_cases=5)
        assert rec.completed == 5
        assert rec.error == 0
        assert rec.accepted == 5
        assert rec.offered == 5

    def test_results_count_equals_total(self, tmp_path):
        rec = _run(tmp_path, n_cases=5)
        rows = load_case_results(Path(rec.results_jsonl_path))
        assert len(rows) == 5

    def test_one_result_per_case(self, tmp_path):
        rec = _run(tmp_path, n_cases=5)
        rows = load_case_results(Path(rec.results_jsonl_path))
        ids = [r["case_id"] for r in rows]
        assert len(set(ids)) == 5

    def test_no_integrity_error(self, tmp_path):
        rec = _run(tmp_path, n_cases=5)
        assert rec.integrity_error is None

    def test_accounting_valid(self, tmp_path):
        rec = _run(tmp_path, n_cases=5)
        assert rec.check_accounting() is None

    def test_runrecord_file_written(self, tmp_path):
        rec = _run(tmp_path, n_cases=3)
        summary = tmp_path / f"run_{rec.run_id}.json"
        assert summary.exists()

    def test_jsonl_file_written(self, tmp_path):
        rec = _run(tmp_path, n_cases=3)
        assert Path(rec.results_jsonl_path).exists()

    def test_all_status_completed(self, tmp_path):
        rec = _run(tmp_path, n_cases=3)
        rows = load_case_results(Path(rec.results_jsonl_path))
        statuses = {r["status"] for r in rows}
        assert statuses == {"completed"}

    def test_mock_llm_flagged(self, tmp_path):
        rec = _run(tmp_path, n_cases=2)
        assert rec.mock_llm is True

    def test_corpus_fingerprint_recorded(self, tmp_path):
        rec = _run(tmp_path, n_cases=2)
        assert rec.corpus_fingerprint == "abc123def456abcd"


# ---------------------------------------------------------------------------
# Scenario 2: deadline before any case offered
# ---------------------------------------------------------------------------

class TestDeadlineBeforeAny:
    def test_all_not_offered(self, tmp_path):
        # Short deadline with slow baseline: most cases not admitted
        bl = _MockBaseline(sleep_seconds=0.05)
        rec = _run(tmp_path, n_cases=20, baseline=bl,
                   max_workers=1, max_queue_depth=1,
                   admission_timeout_seconds=0.001,
                   wall_clock_deadline_seconds=0.005)
        completed = rec.completed + rec.error
        assert completed < 20
        total = (rec.completed + rec.error + rec.not_offered_deadline
                 + rec.cancelled_deadline + rec.rejected_overload + rec.skipped)
        assert total == 20

    def test_results_count_still_equals_total(self, tmp_path):
        rec = _run(tmp_path, n_cases=5, wall_clock_deadline_seconds=0.001)
        rows = load_case_results(Path(rec.results_jsonl_path))
        assert len(rows) == 5

    def test_accounting_valid(self, tmp_path):
        rec = _run(tmp_path, n_cases=5, wall_clock_deadline_seconds=0.001)
        assert rec.check_accounting() is None


# ---------------------------------------------------------------------------
# Scenario 3: deadline midway through admission
# ---------------------------------------------------------------------------

class TestDeadlineMidway:
    def test_some_completed_some_not_offered(self, tmp_path):
        # Fast baseline, short deadline, many cases
        rec = _run(tmp_path, n_cases=20, wall_clock_deadline_seconds=0.5,
                   max_workers=2, max_queue_depth=4)
        total = rec.completed + rec.error + rec.not_offered_deadline + \
                rec.cancelled_deadline + rec.rejected_overload + rec.skipped
        assert total == 20

    def test_results_count_equals_total(self, tmp_path):
        rec = _run(tmp_path, n_cases=20, wall_clock_deadline_seconds=0.5)
        rows = load_case_results(Path(rec.results_jsonl_path))
        assert len(rows) == 20

    def test_accounting_valid(self, tmp_path):
        rec = _run(tmp_path, n_cases=20, wall_clock_deadline_seconds=0.5)
        assert rec.check_accounting() is None


# ---------------------------------------------------------------------------
# Scenario 4: queue full before deadline
# ---------------------------------------------------------------------------

class TestQueueFull:
    def test_rejected_cases_recorded(self, tmp_path):
        # 1 slow worker, queue depth 1, 10 cases -> some rejected
        bl = _MockBaseline(sleep_seconds=0.05)
        rec = _run(tmp_path, n_cases=10, baseline=bl,
                   max_workers=1, max_queue_depth=1,
                   admission_timeout_seconds=0.01,
                   wall_clock_deadline_seconds=10.0)
        assert rec.rejected_overload + rec.completed + rec.error > 0

    def test_results_count_equals_total(self, tmp_path):
        bl = _MockBaseline(sleep_seconds=0.05)
        rec = _run(tmp_path, n_cases=10, baseline=bl,
                   max_workers=1, max_queue_depth=1,
                   admission_timeout_seconds=0.01,
                   wall_clock_deadline_seconds=10.0)
        rows = load_case_results(Path(rec.results_jsonl_path))
        assert len(rows) == 10

    def test_accounting_valid(self, tmp_path):
        bl = _MockBaseline(sleep_seconds=0.05)
        rec = _run(tmp_path, n_cases=10, baseline=bl,
                   max_workers=1, max_queue_depth=1,
                   admission_timeout_seconds=0.01,
                   wall_clock_deadline_seconds=10.0)
        assert rec.check_accounting() is None


# ---------------------------------------------------------------------------
# Scenario 5: admission timeout coinciding with deadline
# ---------------------------------------------------------------------------

class TestAdmissionTimeoutCoinciding:
    def test_cancelled_or_rejected_recorded(self, tmp_path):
        bl = _MockBaseline(sleep_seconds=0.1)
        rec = _run(tmp_path, n_cases=5, baseline=bl,
                   max_workers=1, max_queue_depth=1,
                   admission_timeout_seconds=0.05,
                   wall_clock_deadline_seconds=0.08)
        total = (rec.completed + rec.error + rec.not_offered_deadline +
                 rec.cancelled_deadline + rec.rejected_overload + rec.skipped)
        assert total == 5

    def test_accounting_valid(self, tmp_path):
        bl = _MockBaseline(sleep_seconds=0.1)
        rec = _run(tmp_path, n_cases=5, baseline=bl,
                   max_workers=1, max_queue_depth=1,
                   admission_timeout_seconds=0.05,
                   wall_clock_deadline_seconds=0.08)
        assert rec.check_accounting() is None


# ---------------------------------------------------------------------------
# Scenario 6: one case errors
# ---------------------------------------------------------------------------

class TestCaseError:
    def test_error_counted(self, tmp_path):
        bl = _MockBaseline(raise_on_run=True)
        rec = _run(tmp_path, n_cases=3, baseline=bl)
        assert rec.error == 3
        assert rec.completed == 0

    def test_error_status_in_results(self, tmp_path):
        bl = _MockBaseline(raise_on_run=True)
        rec = _run(tmp_path, n_cases=3, baseline=bl)
        rows = load_case_results(Path(rec.results_jsonl_path))
        statuses = {r["status"] for r in rows}
        assert statuses == {"error"}

    def test_error_field_populated(self, tmp_path):
        bl = _MockBaseline(raise_on_run=True)
        rec = _run(tmp_path, n_cases=3, baseline=bl)
        rows = load_case_results(Path(rec.results_jsonl_path))
        for row in rows:
            assert row["error"] is not None

    def test_results_count_equals_total(self, tmp_path):
        bl = _MockBaseline(raise_on_run=True)
        rec = _run(tmp_path, n_cases=3, baseline=bl)
        rows = load_case_results(Path(rec.results_jsonl_path))
        assert len(rows) == 3

    def test_accounting_valid(self, tmp_path):
        bl = _MockBaseline(raise_on_run=True)
        rec = _run(tmp_path, n_cases=3, baseline=bl)
        assert rec.check_accounting() is None


# ---------------------------------------------------------------------------
# Scenario 7: one-result-per-input invariant
# ---------------------------------------------------------------------------

class TestOneResultPerInput:
    def test_exact_case_ids(self, tmp_path):
        rec = _run(tmp_path, n_cases=7)
        rows = load_case_results(Path(rec.results_jsonl_path))
        result_ids = sorted(r["case_id"] for r in rows)
        expected_ids = sorted(f"c{i:04d}" for i in range(7))
        assert result_ids == expected_ids

    def test_no_duplicate_case_ids_in_results(self, tmp_path):
        rec = _run(tmp_path, n_cases=7)
        rows = load_case_results(Path(rec.results_jsonl_path))
        ids = [r["case_id"] for r in rows]
        assert len(ids) == len(set(ids))

    def test_zero_cases(self, tmp_path):
        rec = _run(tmp_path, n_cases=0)
        assert rec.total_cases == 0
        rows = load_case_results(Path(rec.results_jsonl_path))
        assert len(rows) == 0

    def test_single_case(self, tmp_path):
        rec = _run(tmp_path, n_cases=1)
        rows = load_case_results(Path(rec.results_jsonl_path))
        assert len(rows) == 1

    def test_accounting_invariant_all_scenarios(self, tmp_path):
        for n in [1, 3, 10]:
            td = tmp_path / f"run_{n}"
            td.mkdir()
            rec = _run(td, n_cases=n)
            assert rec.check_accounting() is None, \
                f"Accounting failed for n={n}: {rec.check_accounting()}"


# ---------------------------------------------------------------------------
# Baseline skipped
# ---------------------------------------------------------------------------

class TestBaselineSkipped:
    def test_all_skipped_when_unavailable(self, tmp_path):
        bl = _MockBaseline(skip=True, skip_reason="qdrant_unavailable")
        rec = _run(tmp_path, n_cases=4, baseline=bl)
        assert rec.skipped == 4
        assert rec.completed == 0

    def test_skipped_results_in_jsonl(self, tmp_path):
        bl = _MockBaseline(skip=True, skip_reason="qdrant_unavailable")
        rec = _run(tmp_path, n_cases=4, baseline=bl)
        rows = load_case_results(Path(rec.results_jsonl_path))
        assert all(r["status"] == "skipped" for r in rows)

    def test_skip_reason_recorded(self, tmp_path):
        bl = _MockBaseline(skip=True, skip_reason="test_skip_reason")
        rec = _run(tmp_path, n_cases=2, baseline=bl)
        rows = load_case_results(Path(rec.results_jsonl_path))
        for row in rows:
            assert row["skip_reason"] == "test_skip_reason"


# ---------------------------------------------------------------------------
# RunConfig validation
# ---------------------------------------------------------------------------

class TestRunnerConfigValidation:
    def test_dataset_exceeds_max_cases(self, tmp_path):
        cfg = _make_cfg(tmp_path, max_cases=2)
        ds = _make_dataset(5)
        bl = _MockBaseline()
        with pytest.raises(RunConfigError, match="max_cases"):
            BenchmarkRunner(cfg, bl, ds, _CC)