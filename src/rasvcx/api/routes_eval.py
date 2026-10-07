"""src/rasvcx/api/routes_eval.py

Evaluation API routes for RASVC-X M16.

Endpoints:
  GET  /eval/runs            -- list completed evaluation run summaries
  GET  /eval/runs/{run_id}   -- get one run summary by run_id
  GET  /eval/runs/{run_id}/cases -- stream JSONL case results for one run

Design:
  - Routes are read-only; no evaluation run is triggered via the API.
    Runs are executed offline via evaluation/runner.py and stored as
    JSONL + JSON summary files in an output directory.
  - output_dir defaults to "eval_output" relative to the working directory.
    Override via the RASVCX_EVAL_OUTPUT_DIR environment variable.
  - All routes return 404 when the requested run or file does not exist.
  - StreamingResponse is used for /cases to avoid loading large JSONL
    files into memory.

LIMITATIONS:
  - RunRecord files are read-only; the API never writes evaluation data.
  - mock_llm=True results must not be used to claim real model accuracy.
  - No result constitutes medical advice.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterator

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from rasvcx.api.models_enriched import (
    EvalRunListModel,
    EvalRunSummaryModel,
    run_record_to_summary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/eval", tags=["eval"])

_DEFAULT_EVAL_OUTPUT_DIR = "eval_output"


def _get_output_dir() -> Path:
    """Return eval output directory from env var or default."""
    return Path(
        os.environ.get("RASVCX_EVAL_OUTPUT_DIR", _DEFAULT_EVAL_OUTPUT_DIR)
    )


def _list_run_json_files(output_dir: Path) -> list[Path]:
    """Return all run_*.json files in output_dir, newest first."""
    if not output_dir.exists():
        return []
    files = sorted(
        output_dir.glob("run_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


def _load_run_summary(path: Path) -> EvalRunSummaryModel:
    """Load and parse one run summary JSON file into EvalRunSummaryModel."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read run file {path.name}: {exc}",
        )

    # Extract top-level accounting block if nested
    accounting = raw.get("accounting", {})
    timing = raw.get("timing", {})

    return EvalRunSummaryModel(
        run_id=raw.get("run_id", ""),
        baseline_id=raw.get("baseline_id", ""),
        dataset_id=raw.get("dataset_id", ""),
        dataset_version=raw.get("dataset_version", ""),
        split=raw.get("split", ""),
        corpus_fingerprint=raw.get("corpus_fingerprint", ""),
        corpus_condition=raw.get("corpus_condition", ""),
        execution_mode=raw.get("execution_mode", ""),
        mock_llm=raw.get("mock_llm", True),
        total_cases=accounting.get("total_cases", raw.get("total_cases", 0)),
        offered=accounting.get("offered", raw.get("offered", 0)),
        accepted=accounting.get("accepted", raw.get("accepted", 0)),
        completed=accounting.get("completed", raw.get("completed", 0)),
        error=accounting.get("error", raw.get("error", 0)),
        skipped=accounting.get("skipped", raw.get("skipped", 0)),
        rejected_overload=accounting.get(
            "rejected_overload", raw.get("rejected_overload", 0)
        ),
        cancelled_deadline=accounting.get(
            "cancelled_deadline", raw.get("cancelled_deadline", 0)
        ),
        not_offered_deadline=accounting.get(
            "not_offered_deadline", raw.get("not_offered_deadline", 0)
        ),
        baseline_init_seconds=timing.get(
            "baseline_init_seconds", raw.get("baseline_init_seconds", 0.0)
        ),
        total_run_wall_seconds=timing.get(
            "total_run_wall_seconds", raw.get("total_run_wall_seconds", 0.0)
        ),
        start_utc=timing.get("start_utc", raw.get("start_utc", "")),
        end_utc=timing.get("end_utc", raw.get("end_utc", "")),
        results_jsonl_path=raw.get("results_jsonl_path", ""),
        integrity_error=raw.get("integrity_error"),
    )


def _find_run_file(output_dir: Path, run_id: str) -> Path:
    """Find the run summary file for run_id; raise 404 if not found."""
    candidate = output_dir / f"run_{run_id}.json"
    if candidate.exists():
        return candidate
    # Fallback: scan for any file containing run_id
    for path in output_dir.glob("run_*.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("run_id") == run_id:
                return path
        except (OSError, json.JSONDecodeError):
            continue
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Run {run_id!r} not found in {output_dir}",
    )


def _stream_jsonl(path: Path) -> Iterator[str]:
    """Yield lines from a JSONL file one at a time."""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                yield line
    except OSError as exc:
        logger.error("Failed to stream JSONL %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/runs",
    response_model=EvalRunListModel,
    summary="List evaluation run summaries",
    description=(
        "Returns a list of completed evaluation run summaries from the "
        "eval output directory. Sorted newest first. "
        "LIMITATION: mock_llm=True results must not be used to claim real "
        "model accuracy. No result constitutes medical advice."
    ),
)
def list_eval_runs(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> EvalRunListModel:
    """GET /eval/runs — list run summaries, newest first."""
    output_dir = _get_output_dir()
    all_files = _list_run_json_files(output_dir)
    total = len(all_files)
    page = all_files[offset: offset + limit]

    summaries: list[EvalRunSummaryModel] = []
    for path in page:
        try:
            summaries.append(_load_run_summary(path))
        except HTTPException:
            logger.warning("Skipping unreadable run file: %s", path.name)

    return EvalRunListModel(runs=summaries, total=total)


@router.get(
    "/runs/{run_id}",
    response_model=EvalRunSummaryModel,
    summary="Get one evaluation run summary",
    description=(
        "Returns the summary for one evaluation run by run_id. "
        "Returns 404 if the run does not exist. "
        "LIMITATION: mock_llm=True results must not be used to claim real "
        "model accuracy."
    ),
)
def get_eval_run(run_id: str) -> EvalRunSummaryModel:
    """GET /eval/runs/{run_id} — get one run summary."""
    output_dir = _get_output_dir()
    path = _find_run_file(output_dir, run_id)
    return _load_run_summary(path)


@router.get(
    "/runs/{run_id}/cases",
    summary="Stream case results for one evaluation run",
    description=(
        "Streams the JSONL case results file for one run as "
        "application/x-ndjson. Each line is one JSON object. "
        "Returns 404 if the run or its JSONL file does not exist. "
        "LIMITATION: results from mock_llm=True runs must not be used "
        "to claim real model accuracy. No result constitutes medical advice."
    ),
    response_class=StreamingResponse,
)
def stream_eval_run_cases(run_id: str) -> StreamingResponse:
    """GET /eval/runs/{run_id}/cases — stream JSONL case results."""
    output_dir = _get_output_dir()
    run_path = _find_run_file(output_dir, run_id)

    try:
        summary_raw = json.loads(run_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read run file: {exc}",
        )

    jsonl_path_str = summary_raw.get("results_jsonl_path", "")
    if not jsonl_path_str:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id!r} has no results_jsonl_path recorded.",
        )

    jsonl_path = Path(jsonl_path_str)
    if not jsonl_path.is_absolute():
        jsonl_path = output_dir / jsonl_path

    if not jsonl_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"JSONL file for run {run_id!r} not found: {jsonl_path}"
            ),
        )

    return StreamingResponse(
        _stream_jsonl(jsonl_path),
        media_type="application/x-ndjson",
        headers={
            "X-Run-Id": run_id,
            "X-Mock-LLM": str(summary_raw.get("mock_llm", True)).lower(),
            "X-Limitations": (
                "mock_llm results are not evidence of real model accuracy; "
                "no result constitutes medical advice"
            ),
        },
    )


__all__ = ["router"]