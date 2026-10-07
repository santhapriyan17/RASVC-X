"""evaluation/runner.py

Bounded fixed-worker producer/consumer evaluation runner for M15.

Design:
  work_queue = queue.Queue(maxsize=max_queue_depth)
    Holds WAITING cases only. Active workers have already called get().
    Capacity: max_queue_depth (not max_workers + max_queue_depth).

  active workers <= max_workers         (ThreadPoolExecutor)
  queued cases   <= max_queue_depth     (work_queue.maxsize)
  total outstanding <= max_workers + max_queue_depth

Case accounting (all invariants enforced before RunRecord is written):
  total_cases           = offered + not_offered_deadline
  offered               = accepted + rejected_overload + cancelled_deadline
  accepted              = completed + error
  results_count         = total_cases  (every case has exactly one result)

Admission decision logic:
  - Check deadline BEFORE put(): not_offered_deadline if expired
  - put(timeout=admission_timeout): accepted on success
  - queue.Full and deadline expired: cancelled_deadline
  - queue.Full and deadline not expired: rejected_overload

Sentinel insertion:
  After all cases offered, enqueue max_workers SENTINELs with
  block=True (no timeout). Workers drain the queue; sentinel insertion
  cannot deadlock as long as workers are running. Documented limitation:
  a slow baseline blocks shutdown until its call returns.

Timeout limitation:
  Python threads cannot be forcibly cancelled. A wall-clock deadline only
  prevents new admissions; already-running cases complete naturally.
  Their results are recorded as completed or error, never cancelled.

Thread safety:
  ResultWriter is protected by result_lock (threading.Lock).
  RunRecord counters are updated only by the main thread after all
  workers have completed (post work_queue.join()).

LIMITATIONS:
  - MockLLMClient results must not be used to claim real model accuracy.
  - No result constitutes medical advice.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from evaluation.baselines.interface import BaselineAdapter, BaselineResult
from evaluation.results import CaseResult, ResultWriter, RunRecord, make_run_id
from evaluation.schema import (
    CorpusConfig,
    DatasetSplit,
    EvalDataset,
    RunConfig,
    RunConfigError,
)


# Unique sentinel object; identity check only
_SENTINEL = object()


class BenchmarkRunner:
    """Bounded fixed-worker evaluation runner.

    Usage:
        runner = BenchmarkRunner(run_config, baseline, dataset, corpus_config)
        run_record = runner.run()
    """

    def __init__(
        self,
        run_config: RunConfig,
        baseline: BaselineAdapter,
        dataset: EvalDataset,
        corpus_config: CorpusConfig,
        split: DatasetSplit = DatasetSplit.TEST,
    ) -> None:
        # Validate dataset size against max_cases
        cases = dataset.by_split(split)
        if len(cases) > run_config.max_cases:
            raise RunConfigError(
                f"Dataset has {len(cases)} cases in split {split.value!r} "
                f"but RunConfig.max_cases={run_config.max_cases}. "
                f"Increase max_cases or reduce the dataset."
            )
        self._run_config = run_config
        self._baseline = baseline
        self._cases = cases
        self._corpus_config = corpus_config
        self._split = split

    def run(self) -> RunRecord:
        """Execute the benchmark run. Returns a completed RunRecord.

        Lifecycle:
          1. Validate RunConfig (done in __init__)
          2. Initialize baseline (once; measured separately)
          3. Create work_queue, ResultWriter, result_lock
          4. Submit max_workers worker-loop tasks to executor
          5. Main thread performs admission
          6. Enqueue max_workers SENTINELs (block=True)
          7. work_queue.join() -- waits for all task_done() calls
          8. executor.shutdown(wait=True, cancel_futures=False)
          9. Assert results_count == total_cases
          10. result_writer.close()
          11. Write RunRecord summary
        """
        cfg = self._run_config
        run_id = make_run_id()
        start_wall = time.monotonic()
        start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Baseline initialisation (measured separately from per-case latency)
        init_start = time.monotonic()
        self._baseline.initialize(self._corpus_config)
        baseline_init_seconds = time.monotonic() - init_start

        # If baseline is unavailable after init, record all cases as skipped
        if not self._baseline.is_available:
            return self._all_skipped(
                run_id=run_id,
                start_utc=start_utc,
                baseline_init_seconds=baseline_init_seconds,
                start_wall=start_wall,
                skip_reason=self._baseline.skip_reason or "baseline_unavailable",
            )

        # Storage setup
        output_dir = Path(cfg.output_dir)
        jsonl_path = output_dir / f"cases_{run_id}.jsonl"
        result_writer = ResultWriter(jsonl_path)
        result_lock = threading.Lock()

        # Shared counters (updated by workers via _record, then by main thread)
        _results_written = [0]  # list for mutability in nested scope

        def _record(result: CaseResult) -> None:
            try:
                with result_lock:
                    result_writer.append(result)
                    _results_written[0] += 1
            except Exception as exc:
                # Write failure is non-fatal; logged in RunRecord
                _write_errors[0] += 1

        _write_errors = [0]
        _worker_errors: list[str] = []

        # Work queue: holds WAITING cases only
        # maxsize=max_queue_depth; active workers have removed their items
        work_queue: queue.Queue = queue.Queue(maxsize=cfg.max_queue_depth)

        # Worker function
        def worker() -> None:
            while True:
                item = work_queue.get(block=True)  # removes item; frees slot
                if item is _SENTINEL:
                    work_queue.task_done()
                    return
                case_id, case = item
                t_start = time.monotonic_ns()
                try:
                    outcome: BaselineResult = self._baseline.run(case)
                    elapsed = (time.monotonic_ns() - t_start) / 1e9
                    _record(CaseResult(
                        case_id=case_id,
                        status="completed",
                        baseline_id=self._baseline.baseline_id,
                        corpus_condition=self._corpus_config.condition,
                        latency_seconds=elapsed,
                        data=outcome.to_dict(),
                        mock_llm=self._baseline.mock_llm,
                    ))
                except Exception as exc:
                    elapsed = (time.monotonic_ns() - t_start) / 1e9
                    _record(CaseResult(
                        case_id=case_id,
                        status="error",
                        baseline_id=self._baseline.baseline_id,
                        corpus_condition=self._corpus_config.condition,
                        latency_seconds=elapsed,
                        error=repr(exc),
                        mock_llm=self._baseline.mock_llm,
                    ))
                finally:
                    work_queue.task_done()  # always; updates join() counter

        # Step 4: submit exactly max_workers worker tasks
        executor = ThreadPoolExecutor(
            max_workers=cfg.max_workers,
            thread_name_prefix="rasvcx-eval-worker",
        )
        worker_futures = [
            executor.submit(worker) for _ in range(cfg.max_workers)
        ]

        # Step 5: admission
        total_cases = len(self._cases)
        offered = 0
        not_offered_deadline = 0
        accepted = 0
        rejected_overload = 0
        cancelled_deadline = 0

        deadline = time.monotonic() + cfg.wall_clock_deadline_seconds

        for case in self._cases:
            # Check global deadline before attempting put()
            if time.monotonic() >= deadline:
                _record(CaseResult(
                    case_id=case.case_id,
                    status="not_offered_deadline",
                    baseline_id=self._baseline.baseline_id,
                    corpus_condition=self._corpus_config.condition,
                    mock_llm=self._baseline.mock_llm,
                ))
                not_offered_deadline += 1
                continue

            offered += 1
            admitted = False

            while not admitted:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _record(CaseResult(
                        case_id=case.case_id,
                        status="cancelled_deadline",
                        baseline_id=self._baseline.baseline_id,
                        corpus_condition=self._corpus_config.condition,
                        mock_llm=self._baseline.mock_llm,
                    ))
                    cancelled_deadline += 1
                    break

                wait = min(cfg.admission_timeout_seconds, remaining)
                try:
                    work_queue.put(
                        (case.case_id, case),
                        block=True,
                        timeout=wait,
                    )
                    admitted = True
                    accepted += 1
                except queue.Full:
                    # Determine which limit was hit
                    if time.monotonic() >= deadline:
                        _record(CaseResult(
                            case_id=case.case_id,
                            status="cancelled_deadline",
                            baseline_id=self._baseline.baseline_id,
                            corpus_condition=self._corpus_config.condition,
                            mock_llm=self._baseline.mock_llm,
                        ))
                        cancelled_deadline += 1
                    else:
                        _record(CaseResult(
                            case_id=case.case_id,
                            status="rejected_overload",
                            baseline_id=self._baseline.baseline_id,
                            corpus_condition=self._corpus_config.condition,
                            mock_llm=self._baseline.mock_llm,
                        ))
                        rejected_overload += 1
                    break

        # Step 6: enqueue exactly max_workers SENTINELs
        # block=True, no timeout: workers drain the queue, cannot deadlock
        for _ in range(cfg.max_workers):
            work_queue.put(_SENTINEL, block=True)

        # Step 7: wait until all get()/task_done() pairs are matched
        work_queue.join()

        # Step 8: clean executor shutdown
        executor.shutdown(wait=True, cancel_futures=False)

        # Collect worker-level errors (exceptions outside per-case handler)
        for f in worker_futures:
            exc = f.exception()
            if exc is not None:
                _worker_errors.append(repr(exc))

        # Finalise baseline
        try:
            self._baseline.close()
        except Exception as exc:
            _worker_errors.append(f"baseline.close() raised: {repr(exc)}")

        # Step 10: close result writer
        result_writer.close()

        end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        total_run_wall = time.monotonic() - start_wall

        # Derive completed/error from accepted minus the accounting above
        # Actual counts come from what was written; we use accepted for now
        # and trust _record was called for every accepted case
        completed = accepted - (_results_written[0] - (
            rejected_overload + cancelled_deadline + not_offered_deadline
        ))
        # Simpler: count statuses from written results
        # Recount from JSONL for accuracy
        from evaluation.results import load_case_results
        try:
            written_results = load_case_results(jsonl_path)
            status_counts: dict[str, int] = {}
            for r in written_results:
                s = r.get("status", "unknown")
                status_counts[s] = status_counts.get(s, 0) + 1
        except Exception:
            status_counts = {}

        n_completed = status_counts.get("completed", 0)
        n_error = status_counts.get("error", 0)
        n_skipped = status_counts.get("skipped", 0)

        # Step 11: build RunRecord
        record = RunRecord(
            run_id=run_id,
            baseline_id=self._baseline.baseline_id,
            dataset_id=self._cases[0].dataset_id if self._cases else "unknown",
            dataset_version=self._cases[0].dataset_version if self._cases else "unknown",
            split=self._split,
            corpus_fingerprint=self._corpus_config.fingerprint,
            corpus_condition=self._corpus_config.condition,
            execution_mode=_detect_execution_mode(),
            mock_llm=self._baseline.mock_llm,
            total_cases=total_cases,
            offered=offered,
            not_offered_deadline=not_offered_deadline,
            accepted=accepted,
            rejected_overload=rejected_overload,
            cancelled_deadline=cancelled_deadline,
            completed=n_completed,
            error=n_error,
            skipped=n_skipped,
            baseline_init_seconds=baseline_init_seconds,
            total_run_wall_seconds=total_run_wall,
            start_utc=start_utc,
            end_utc=end_utc,
            results_jsonl_path=str(jsonl_path),
            worker_errors=_worker_errors,
            write_errors=_write_errors[0],
        )

        # Step 9: check accounting invariants
        accounting_err = record.check_accounting()
        if accounting_err:
            record.integrity_error = accounting_err

        # Step 12: save RunRecord summary
        record.save(output_dir)

        return record

    def _all_skipped(
        self,
        run_id: str,
        start_utc: str,
        baseline_init_seconds: float,
        start_wall: float,
        skip_reason: str,
    ) -> RunRecord:
        """Return a RunRecord where all cases are skipped."""
        cfg = self._run_config
        output_dir = Path(cfg.output_dir)
        jsonl_path = output_dir / f"cases_{run_id}.jsonl"

        result_writer = ResultWriter(jsonl_path)
        for case in self._cases:
            result_writer.append(CaseResult(
                case_id=case.case_id,
                status="skipped",
                baseline_id=self._baseline.baseline_id,
                corpus_condition=self._corpus_config.condition,
                skip_reason=skip_reason,
                mock_llm=self._baseline.mock_llm,
            ))
        result_writer.close()

        total = len(self._cases)
        record = RunRecord(
            run_id=run_id,
            baseline_id=self._baseline.baseline_id,
            dataset_id=self._cases[0].dataset_id if self._cases else "unknown",
            dataset_version=self._cases[0].dataset_version if self._cases else "unknown",
            split=self._split,
            corpus_fingerprint=self._corpus_config.fingerprint,
            corpus_condition=self._corpus_config.condition,
            execution_mode=_detect_execution_mode(),
            mock_llm=self._baseline.mock_llm,
            total_cases=total,
            offered=0,
            not_offered_deadline=0,
            accepted=0,
            rejected_overload=0,
            cancelled_deadline=0,
            completed=0,
            error=0,
            skipped=total,
            baseline_init_seconds=baseline_init_seconds,
            total_run_wall_seconds=time.monotonic() - start_wall,
            start_utc=start_utc,
            end_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            results_jsonl_path=str(jsonl_path),
            skipped_baselines=[
                f"{self._baseline.baseline_id}: {skip_reason}"
            ],
        )
        record.save(output_dir)
        return record


def _detect_execution_mode() -> str:
    """Detect the current execution mode from Settings."""
    try:
        from rasvcx.config.settings import Settings
        return Settings().execution_mode.value
    except Exception:
        return "unknown"


__all__ = ["BenchmarkRunner"]