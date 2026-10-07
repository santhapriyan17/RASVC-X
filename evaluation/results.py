"""evaluation/results.py

Per-case results, run records, and incremental JSONL storage for the
RASVC-X M15 evaluation framework.

Public types:
  CaseResult   -- exactly one per dataset case; carries status and data
  RunRecord    -- summary of a completed benchmark run
  ResultWriter -- incremental append-only JSONL writer (open for run duration)

Storage design:
  - ResultWriter writes one JSON line per CaseResult to a .jsonl file.
  - The file is flushed after every write; never buffered across cases.
  - result_lock (threading.Lock) must be held by the caller before
    calling ResultWriter.append().  The writer itself is not thread-safe.
  - RunRecord summary is written as a separate .json file only AFTER
    ResultWriter.close() is called; never before.
  - Files are identified by run_id (UUID4); no silent overwrite.

Accounting invariants enforced before RunRecord is written:
  total_cases = offered + not_offered_deadline
  offered     = accepted + rejected_overload + cancelled_deadline
  accepted    = completed + error
  results_count (all statuses) == total_cases

Status values (mutually exclusive per case):
  completed          -- baseline.run() returned normally
  error              -- baseline.run() raised an exception
  rejected_overload  -- queue.put() admission timeout expired; deadline not expired
  cancelled_deadline -- global deadline expired before successful queue insertion
  not_offered_deadline -- main thread never attempted put(); deadline already expired
  skipped            -- baseline explicitly skipped (e.g. Qdrant unavailable)

LIMITATIONS:
  - MockLLMClient results labelled mock_llm=True must not be used to
    claim real model accuracy.
  - Calibration fields are operational proxies, not clinical correctness.
  - No secrets, API keys, or credentials are stored in any result file.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from evaluation.schema import CorpusCondition, DatasetSplit, RunConfig


# ---------------------------------------------------------------------------
# CaseResult
# ---------------------------------------------------------------------------

# Valid status values — enforced in CaseResult.__post_init__
_VALID_STATUSES = frozenset({
    "completed",
    "error",
    "rejected_overload",
    "cancelled_deadline",
    "not_offered_deadline",
    "skipped",
})


@dataclass
class CaseResult:
    """Exactly one result per dataset case, regardless of status.

    Every dataset case must produce a CaseResult before the RunRecord
    is finalised.  The runner enforces: results_count == total_cases.

    Fields:
      case_id           -- must match the EvalCase.case_id
      status            -- one of _VALID_STATUSES
      baseline_id       -- which baseline produced this result
      corpus_condition  -- CorpusCondition of the corpus used
      latency_seconds   -- per-case execution time (None if not executed)
      data              -- serialisable dict of baseline output (None if
                          not executed or errored without output)
      error             -- repr(exception) for status=error; None otherwise
      skip_reason       -- human-readable reason for status=skipped
      mock_llm          -- True when MockLLMClient was the generator;
                          answer-accuracy metrics must not be claimed
      timestamp_utc     -- ISO-8601 UTC timestamp of result creation
    """

    case_id: str
    status: str
    baseline_id: str
    corpus_condition: CorpusCondition

    latency_seconds: Optional[float] = None
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    skip_reason: Optional[str] = None
    mock_llm: bool = True
    timestamp_utc: str = field(
        default_factory=lambda: time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
    )

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("CaseResult.case_id must be non-empty")
        if self.status not in _VALID_STATUSES:
            raise ValueError(
                f"CaseResult.status {self.status!r} is not valid. "
                f"Must be one of: {sorted(_VALID_STATUSES)}"
            )
        if self.status == "error" and self.error is None:
            raise ValueError(
                "CaseResult.error must be set when status='error'"
            )
        if self.status == "skipped" and self.skip_reason is None:
            raise ValueError(
                "CaseResult.skip_reason must be set when status='skipped'"
            )
        if self.latency_seconds is not None and self.latency_seconds < 0:
            raise ValueError(
                f"CaseResult.latency_seconds must be ≥ 0, "
                f"got {self.latency_seconds}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        d = {
            "case_id": self.case_id,
            "status": self.status,
            "baseline_id": self.baseline_id,
            "corpus_condition": self.corpus_condition.value,
            "latency_seconds": self.latency_seconds,
            "data": self.data,
            "error": self.error,
            "skip_reason": self.skip_reason,
            "mock_llm": self.mock_llm,
            "timestamp_utc": self.timestamp_utc,
        }
        return d


# ---------------------------------------------------------------------------
# RunRecord
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """Summary of a completed benchmark run.

    Written as a .json file only after ResultWriter.close() is called.
    Never overwrites an existing file (enforced by save()).

    Timing fields:
      baseline_init_seconds -- time to construct/load the baseline
                               (excluded from per-case latency)
      total_run_wall_seconds -- total wall-clock time including init
      per-case latency is stored in each CaseResult.latency_seconds

    Accounting:
      total_cases = offered + not_offered_deadline
      offered     = accepted + rejected_overload + cancelled_deadline
      accepted    = completed + error
      results_count must equal total_cases before save() is called.

    Limitations stored inline:
      mock_llm=True  → answer accuracy/calibration metrics not applicable
      calibration_proxy=True → not clinical correctness
    """

    run_id: str
    baseline_id: str
    dataset_id: str
    dataset_version: str
    split: DatasetSplit
    corpus_fingerprint: str
    corpus_condition: CorpusCondition
    execution_mode: str            # e.g. 'offline_test', 'research_bm25'
    mock_llm: bool
    calibration_proxy: bool = True

    # Accounting
    total_cases: int = 0
    offered: int = 0
    not_offered_deadline: int = 0
    accepted: int = 0
    rejected_overload: int = 0
    cancelled_deadline: int = 0
    completed: int = 0
    error: int = 0
    skipped: int = 0

    # Timing
    baseline_init_seconds: float = 0.0
    total_run_wall_seconds: float = 0.0

    # Timestamps
    start_utc: str = ""
    end_utc: str = ""

    # Paths
    results_jsonl_path: str = ""

    # Skipped baselines
    skipped_baselines: list[str] = field(default_factory=list)

    # Worker errors (unexpected exceptions outside per-case handler)
    worker_errors: list[str] = field(default_factory=list)

    # Write errors
    write_errors: int = 0

    # Timing precision note
    timing_note: str = (
        "Per-case latency excludes baseline initialisation time. "
        "Total run time includes initialisation. "
        "Deadline precision is subject to OS scheduler latency."
    )

    # Limitations note — must not be removed
    limitations: str = (
        "MockLLMClient results (mock_llm=True) are not evidence of real "
        "model accuracy. Calibration fields are operational proxies, not "
        "clinical correctness labels. This system is a research prototype; "
        "no medical advice or clinical deployment claims are made."
    )

    integrity_error: Optional[str] = None
    """Set when results_count != total_cases at finalisation."""

    def check_accounting(self) -> Optional[str]:
        """Return an error string if accounting invariants are violated."""
        errors = []
        if self.offered + self.not_offered_deadline != self.total_cases:
            errors.append(
                f"offered({self.offered}) + "
                f"not_offered_deadline({self.not_offered_deadline}) "
                f"!= total_cases({self.total_cases})"
            )
        if (self.accepted + self.rejected_overload
                + self.cancelled_deadline) != self.offered:
            errors.append(
                f"accepted({self.accepted}) + "
                f"rejected_overload({self.rejected_overload}) + "
                f"cancelled_deadline({self.cancelled_deadline}) "
                f"!= offered({self.offered})"
            )
        if self.completed + self.error != self.accepted:
            errors.append(
                f"completed({self.completed}) + error({self.error}) "
                f"!= accepted({self.accepted})"
            )
        total_results = (
            self.completed + self.error + self.rejected_overload
            + self.cancelled_deadline + self.not_offered_deadline
            + self.skipped
        )
        if total_results != self.total_cases:
            errors.append(
                f"results_count({total_results}) != total_cases({self.total_cases})"
            )
        return "; ".join(errors) if errors else None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return {
            "run_id": self.run_id,
            "baseline_id": self.baseline_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "split": self.split.value,
            "corpus_fingerprint": self.corpus_fingerprint,
            "corpus_condition": self.corpus_condition.value,
            "execution_mode": self.execution_mode,
            "mock_llm": self.mock_llm,
            "calibration_proxy": self.calibration_proxy,
            "accounting": {
                "total_cases": self.total_cases,
                "offered": self.offered,
                "not_offered_deadline": self.not_offered_deadline,
                "accepted": self.accepted,
                "rejected_overload": self.rejected_overload,
                "cancelled_deadline": self.cancelled_deadline,
                "completed": self.completed,
                "error": self.error,
                "skipped": self.skipped,
            },
            "timing": {
                "baseline_init_seconds": self.baseline_init_seconds,
                "total_run_wall_seconds": self.total_run_wall_seconds,
                "start_utc": self.start_utc,
                "end_utc": self.end_utc,
                "note": self.timing_note,
            },
            "paths": {
                "results_jsonl": self.results_jsonl_path,
            },
            "skipped_baselines": self.skipped_baselines,
            "worker_errors": self.worker_errors,
            "write_errors": self.write_errors,
            "integrity_error": self.integrity_error,
            "limitations": self.limitations,
        }

    def save(self, output_dir: Path) -> Path:
        """Write summary JSON to output_dir/run_{run_id}.json.

        Raises FileExistsError if the file already exists.
        Never overwrites silently.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"run_{self.run_id}.json"
        if path.exists():
            raise FileExistsError(
                f"RunRecord already exists at {path}. "
                f"Each run_id must be unique. Got run_id={self.run_id!r}"
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path


# ---------------------------------------------------------------------------
# ResultWriter
# ---------------------------------------------------------------------------


class ResultWriter:
    """Incremental append-only JSONL writer for CaseResult objects.

    Usage:
      writer = ResultWriter(path)
      try:
          with result_lock:
              writer.append(case_result)
      finally:
          writer.close()

    Design:
      - File is opened once at construction; kept open for the run duration.
      - Each append() writes one JSON line and flushes immediately.
      - close() flushes and closes the file handle.
      - ResultWriter is NOT thread-safe; callers must hold result_lock.
      - Raises FileExistsError if the path already exists (no overwrite).
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if self._path.exists():
            raise FileExistsError(
                f"ResultWriter path already exists: {self._path}. "
                f"Each run must use a unique path."
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w", encoding="utf-8")
        self._count = 0
        self._closed = False

    def append(self, result: CaseResult) -> None:
        """Write one CaseResult as a JSON line. Flush immediately.

        Must be called with result_lock held.
        Raises if already closed.
        """
        if self._closed:
            raise RuntimeError(
                "ResultWriter.append() called after close()"
            )
        line = json.dumps(result.to_dict(), ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        self._count += 1

    def close(self) -> None:
        """Flush and close the file handle. Idempotent."""
        if not self._closed:
            self._fh.flush()
            self._fh.close()
            self._closed = True

    @property
    def count(self) -> int:
        """Number of CaseResults written so far."""
        return self._count

    @property
    def path(self) -> Path:
        return self._path

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_run_id() -> str:
    """Generate a unique run identifier (UUID4 hex string)."""
    return uuid.uuid4().hex


def load_run_record(summary_path: Path) -> dict[str, Any]:
    """Load a RunRecord summary JSON from disk. Returns raw dict."""
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def load_case_results(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load all CaseResult dicts from a JSONL file."""
    results = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def list_run_records(output_dir: Path) -> list[Path]:
    """Return all run summary JSON paths in output_dir, newest first."""
    if not output_dir.exists():
        return []
    paths = sorted(
        output_dir.glob("run_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return paths


__all__ = [
    "CaseResult",
    "RunRecord",
    "ResultWriter",
    "make_run_id",
    "load_run_record",
    "load_case_results",
    "list_run_records",
]