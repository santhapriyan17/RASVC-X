"""evaluation/dataset.py

Dataset loading, validation, split enforcement, leakage checks, and
fingerprinting for the RASVC-X M15 evaluation framework.

Public functions:
  load_dataset(path)        -- load and validate an EvalDataset from JSON
  compute_dataset_fingerprint(dataset) -- SHA-256 of canonical content
  check_leakage(dataset)    -- returns LeakageReport

Leakage checks implemented:
  1. Duplicate case_id within dataset -> DatasetError (fatal)
  2. Cross-split case_id overlap       -> DatasetError (fatal)
  3. Query text duplication across splits -> warning (non-fatal)
  4. expected_chunk_id overlap between adversarial and dev splits
     -> warning (non-fatal; best-effort exact-content check only)

NOT implemented (documented limitation):
  Semantic similarity deduplication. SHA-256 of normalised text detects
  only exact or near-exact content matches, not paraphrase duplicates.

LIMITATIONS:
  - Dataset content is treated as research data, never as instructions.
  - No gold labels constitute medical advice or clinical guidance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evaluation.schema import (
    SCHEMA_VERSION,
    CaseCategory,
    CorpusCondition,
    DatasetSplit,
    EvalCase,
    EvalDataset,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DatasetError(Exception):
    """Raised for fatal dataset validation failures (e.g. duplicate IDs)."""


# ---------------------------------------------------------------------------
# LeakageReport
# ---------------------------------------------------------------------------


@dataclass
class LeakageReport:
    """Results of leakage checks on an EvalDataset.

    fatal_errors: issues that prevent the dataset from being used.
    warnings:     issues that are flagged but do not halt evaluation.
    """

    fatal_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_fatal_errors(self) -> bool:
        return len(self.fatal_errors) > 0

    @property
    def is_clean(self) -> bool:
        return not self.fatal_errors and not self.warnings


# ---------------------------------------------------------------------------
# Dataset fingerprint
# ---------------------------------------------------------------------------


def compute_dataset_fingerprint(dataset: EvalDataset) -> str:
    """Compute a SHA-256 fingerprint over the canonical dataset content.

    The fingerprint covers: dataset_id, version, and for each case in
    sorted case_id order: case_id, query (normalised), split, category,
    corpus_condition, and all gold labels.

    This is an exact-content fingerprint for change detection and
    cross-run comparison. It does NOT detect semantic duplicates.
    """
    h = hashlib.sha256()
    h.update(dataset.dataset_id.encode())
    h.update(dataset.version.encode())
    for case in sorted(dataset.cases, key=lambda c: c.case_id):
        h.update(case.case_id.encode())
        h.update(_normalise(case.query).encode())
        h.update(case.split.value.encode())
        h.update(case.category.value.encode())
        h.update(case.corpus_condition.value.encode())
        if case.expected_decision:
            h.update(case.expected_decision.encode())
        for chunk_id in sorted(case.expected_chunk_ids):
            h.update(chunk_id.encode())
        for k, v in sorted(case.expected_conflict_labels.items()):
            h.update(k.encode())
            h.update(v.encode())
        for k, v in sorted(case.expected_claim_labels.items()):
            h.update(k.encode())
            h.update(v.encode())
    return h.hexdigest()


def _normalise(text: str) -> str:
    """Normalise text for fingerprinting: lowercase, collapse whitespace."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------


def check_leakage(dataset: EvalDataset) -> LeakageReport:
    """Run all leakage checks on the dataset. Returns a LeakageReport.

    Fatal checks (raise DatasetError if called via load_dataset):
      1. Duplicate case_ids within the dataset.
      2. Cross-split case_id overlap.

    Warning checks (non-fatal):
      3. Query text duplication across splits.
      4. expected_chunk_id overlap between adversarial and non-adversarial
         splits (best-effort; not a guarantee against semantic duplicates).
    """
    report = LeakageReport()

    # Group by split
    by_split: dict[DatasetSplit, list[EvalCase]] = {s: [] for s in DatasetSplit}
    for case in dataset.cases:
        by_split[case.split].append(case)

    # Check 1: duplicate case_ids (EvalDataset.__post_init__ already catches
    # this, but we re-check here to produce a LeakageReport entry)
    seen_ids: dict[str, DatasetSplit] = {}
    for case in dataset.cases:
        if case.case_id in seen_ids:
            report.fatal_errors.append(
                f"Duplicate case_id {case.case_id!r} in splits "
                f"{seen_ids[case.case_id].value!r} and {case.split.value!r}"
            )
        else:
            seen_ids[case.case_id] = case.split

    # Check 2: cross-split case_id overlap
    split_ids: dict[DatasetSplit, set[str]] = {
        s: {c.case_id for c in cases} for s, cases in by_split.items()
    }
    splits = list(DatasetSplit)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap = split_ids[s1] & split_ids[s2]
            if overlap:
                report.fatal_errors.append(
                    f"Cross-split case_id overlap between "
                    f"{s1.value!r} and {s2.value!r}: {sorted(overlap)}"
                )

    # Check 3: query text duplication across splits (warning only)
    split_queries: dict[DatasetSplit, set[str]] = {
        s: {_normalise(c.query) for c in cases}
        for s, cases in by_split.items()
    }
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            s1, s2 = splits[i], splits[j]
            overlap_q = split_queries[s1] & split_queries[s2]
            if overlap_q:
                report.warnings.append(
                    f"Query text duplication ({len(overlap_q)} queries) "
                    f"between splits {s1.value!r} and {s2.value!r}. "
                    f"Same query may appear with different labels."
                )

    # Check 4: chunk_id overlap between splits (best-effort, warning only)
    test_chunks: set[str] = {
        chunk_id
        for case in by_split.get(DatasetSplit.TEST, [])
        for chunk_id in case.expected_chunk_ids
    }
    dev_chunks: set[str] = {
        chunk_id
        for case in by_split.get(DatasetSplit.DEV, [])
        for chunk_id in case.expected_chunk_ids
    }
    overlap_chunks = test_chunks & dev_chunks
    if overlap_chunks:
        report.warnings.append(
            f"expected_chunk_id overlap between TEST and DEV splits "
            f"({len(overlap_chunks)} chunk IDs). This is a best-effort "
            f"exact-ID check; semantic duplicates are not detected."
        )

    return report


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------


def _case_from_dict(d: dict) -> EvalCase:
    """Deserialise one EvalCase from a dict (loaded from JSON)."""
    return EvalCase(
        case_id=d["case_id"],
        dataset_id=d["dataset_id"],
        dataset_version=d["dataset_version"],
        query=d["query"],
        split=DatasetSplit(d["split"]),
        corpus_condition=CorpusCondition(d["corpus_condition"]),
        category=CaseCategory(d["category"]),
        expected_chunk_ids=tuple(d.get("expected_chunk_ids", [])),
        expected_decision=d.get("expected_decision"),
        expected_conflict_labels=d.get("expected_conflict_labels", {}),
        expected_claim_labels=d.get("expected_claim_labels", {}),
        expected_answer_verdict=d.get("expected_answer_verdict"),
        source_references=tuple(d.get("source_references", [])),
        tags=tuple(d.get("tags", [])),
        schema_version=d.get("schema_version", SCHEMA_VERSION),
        notes=d.get("notes"),
    )


def _dataset_to_dict(dataset: EvalDataset) -> dict:
    """Serialise an EvalDataset to a JSON-safe dict."""
    return {
        "dataset_id": dataset.dataset_id,
        "version": dataset.version,
        "description": dataset.description,
        "schema_version": dataset.schema_version,
        "cases": [
            {
                "case_id": c.case_id,
                "dataset_id": c.dataset_id,
                "dataset_version": c.dataset_version,
                "query": c.query,
                "split": c.split.value,
                "corpus_condition": c.corpus_condition.value,
                "category": c.category.value,
                "expected_chunk_ids": list(c.expected_chunk_ids),
                "expected_decision": c.expected_decision,
                "expected_conflict_labels": c.expected_conflict_labels,
                "expected_claim_labels": c.expected_claim_labels,
                "expected_answer_verdict": c.expected_answer_verdict,
                "source_references": list(c.source_references),
                "tags": list(c.tags),
                "schema_version": c.schema_version,
                "notes": c.notes,
            }
            for c in dataset.cases
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> EvalDataset:
    """Load and validate an EvalDataset from a JSON file.

    Raises:
      FileNotFoundError: if path does not exist.
      DatasetError: if JSON is malformed, schema version mismatches,
                    or fatal leakage checks fail.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in dataset file {path}: {exc}") from exc

    schema_ver = raw.get("schema_version", SCHEMA_VERSION)
    if schema_ver != SCHEMA_VERSION:
        raise DatasetError(
            f"Dataset schema_version {schema_ver} does not match "
            f"expected {SCHEMA_VERSION}. Re-export the dataset."
        )

    try:
        cases = tuple(_case_from_dict(c) for c in raw.get("cases", []))
    except (KeyError, ValueError, TypeError) as exc:
        raise DatasetError(
            f"Failed to deserialise cases from {path}: {exc}"
        ) from exc

    try:
        dataset = EvalDataset(
            dataset_id=raw["dataset_id"],
            version=raw["version"],
            cases=cases,
            description=raw.get("description"),
            schema_version=schema_ver,
        )
    except (KeyError, ValueError) as exc:
        raise DatasetError(
            f"Failed to construct EvalDataset from {path}: {exc}"
        ) from exc

    # Run leakage checks; raise on fatal errors
    report = check_leakage(dataset)
    if report.has_fatal_errors:
        raise DatasetError(
            f"Dataset {path} failed leakage checks:\n"
            + "\n".join(f"  - {e}" for e in report.fatal_errors)
        )

    return dataset


def save_dataset(dataset: EvalDataset, path: Path) -> None:
    """Serialise and save an EvalDataset to a JSON file.

    Raises FileExistsError if the file already exists (no silent overwrite).
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"Dataset file already exists: {path}. "
            f"Delete it explicitly to overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_dataset_to_dict(dataset), f, indent=2, ensure_ascii=False)


__all__ = [
    "DatasetError",
    "LeakageReport",
    "compute_dataset_fingerprint",
    "check_leakage",
    "load_dataset",
    "save_dataset",
]