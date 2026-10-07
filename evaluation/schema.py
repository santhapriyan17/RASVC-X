"""Evaluation schema for RASVC-X M15 research evaluation framework.

Defines the versioned, typed data contracts used throughout the evaluation
layer.  Nothing in this file modifies the M1-M14 pipeline; these are
purely additive research artefacts.

Key types:
  CorpusCondition  -- labelled condition for controlled experiments
  DatasetSplit     -- strict three-way separation of evaluation data
  CaseCategory     -- edge-case taxonomy for adversarial testing
  CorpusConfig     -- corpus paths and fingerprint for one evaluation run
  RunConfig        -- validated runner parameters; raises RunConfigError
  EvalCase         -- single evaluation input with optional gold labels
  EvalDataset      -- versioned, split-labelled collection of EvalCases

Design rules:
  - All fields that are genuinely optional carry explicit None defaults.
  - No field is silently ignored; missing required fields raise at
    construction time via __post_init__.
  - CorpusCondition is carried in every EvalCase and CaseResult so that
    downstream consumers never need to re-join against run metadata.
  - RunConfig validation raises RunConfigError (not ValueError) so callers
    can catch configuration errors separately from logic errors.
  - Schema version is a module-level constant checked during dataset load.

LIMITATIONS (must not be removed):
  - MockLLMClient results are not evidence of real model accuracy.
  - Calibration metrics derived from these labels are an operational proxy,
    not a measure of clinical correctness.
  - No field in this schema constitutes medical advice or clinical guidance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
"""Increment when EvalCase fields are added or removed in a breaking way."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CorpusCondition(str, Enum):
    """Labelled condition for a corpus used in a controlled experiment."""

    CLEAN = "clean"
    POISONED_CONTRADICTION = "poisoned_contradiction"
    POISONED_STALE = "poisoned_stale"
    POISONED_PROVENANCE = "poisoned_provenance"
    POISONED_COMBINED = "poisoned_combined"


class DatasetSplit(str, Enum):
    """Strict separation of evaluation data.

    DEV               development: inspect, debug, tune policy freely
    CALIBRATION       the ONLY data a confidence calibrator may be fitted on
    TEST              frozen held-out test: evaluated once, never tuned on
    SAFETY_REGRESSION purpose-built safety cases (e.g. the 18 demo cases):
                      must-pass regression checks, never used for
                      calibration or as a general accuracy estimate
    VAL               legacy name kept for existing datasets
    """

    DEV = "dev"
    CALIBRATION = "calibration"
    VAL = "val"
    TEST = "test"
    SAFETY_REGRESSION = "safety_regression"


class CaseCategory(str, Enum):
    """Edge-case taxonomy for adversarial and systematic test coverage."""

    CLEAN = "clean"
    CONTRADICTION = "contradiction"
    TEMPORAL_CONFLICT = "temporal_conflict"
    POPULATION_MISMATCH = "population_mismatch"
    JURISDICTION_MISMATCH = "jurisdiction_mismatch"
    DOSAGE_CONFLICT = "dosage_conflict"
    MISSING_EVIDENCE = "missing_evidence"
    WEAK_PROVENANCE = "weak_provenance"
    STALE_DOCUMENT = "stale_document"
    PROMPT_INJECTION = "prompt_injection"
    CORRUPTED_TEXT = "corrupted_text"
    DUPLICATE_DOCUMENT = "duplicate_document"
    PROVIDER_FAILURE = "provider_failure"


# ---------------------------------------------------------------------------
# CorpusConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusConfig:
    """Corpus paths and fingerprint for one evaluation run."""

    store_path: Path
    bm25_path: Path
    fingerprint: str
    condition: CorpusCondition

    def __post_init__(self) -> None:
        if not self.fingerprint or len(self.fingerprint) < 16:
            raise ValueError(
                f"CorpusConfig.fingerprint must be a non-empty hex string, "
                f"got: {self.fingerprint!r}"
            )
        if self.condition not in CorpusCondition.__members__.values():
            raise ValueError(f"Unknown CorpusCondition: {self.condition!r}")


# ---------------------------------------------------------------------------
# RunConfigError and RunConfig
# ---------------------------------------------------------------------------


class RunConfigError(Exception):
    """Raised when RunConfig validation fails.

    Distinct from ValueError so callers can catch configuration errors
    separately from logic errors.  Raised before any thread is created.
    """


@dataclass(frozen=True)
class RunConfig:
    """Validated runner parameters.

    Raises RunConfigError (not ValueError) for any invalid value.
    All validation happens in __post_init__ before workers start.

    Capacity invariant:
      active_workers <= max_workers         (ThreadPoolExecutor)
      queued_cases   <= max_queue_depth     (work_queue.maxsize)
      outstanding    <= max_workers + max_queue_depth

    Note: queue.Queue(maxsize=0) is unbounded in Python.
    max_queue_depth=0 is therefore explicitly prohibited.
    """

    max_workers: int
    max_queue_depth: int
    max_cases: int
    output_dir: Path
    admission_timeout_seconds: float
    wall_clock_deadline_seconds: float
    baseline_init_timeout_seconds: float = 120.0
    warmup_cases: int = 5

    def __post_init__(self) -> None:
        errors: list[str] = []

        if self.max_workers < 1:
            errors.append(
                f"max_workers must be >= 1, got {self.max_workers}"
            )
        if self.max_queue_depth < 1:
            errors.append(
                f"max_queue_depth must be >= 1 (0 would be unbounded), "
                f"got {self.max_queue_depth}"
            )
        if self.max_cases < 1:
            errors.append(f"max_cases must be >= 1, got {self.max_cases}")
        if self.admission_timeout_seconds <= 0:
            errors.append(
                f"admission_timeout_seconds must be > 0, "
                f"got {self.admission_timeout_seconds}"
            )
        if self.wall_clock_deadline_seconds <= 0:
            errors.append(
                f"wall_clock_deadline_seconds must be > 0, "
                f"got {self.wall_clock_deadline_seconds}"
            )
        if self.warmup_cases < 0:
            errors.append(
                f"warmup_cases must be >= 0, got {self.warmup_cases}"
            )

        # output_dir: explicit is_file() check first (cross-platform),
        # then attempt mkdir + probe write
        out = Path(self.output_dir)
        if out.is_file():
            errors.append(
                f"output_dir {self.output_dir!r} exists as a file, "
                f"not a directory"
            )
        else:
            try:
                out.mkdir(parents=True, exist_ok=True)
                probe = out / ".rasvcx_write_probe"
                probe.touch()
                probe.unlink()
            except Exception as exc:
                errors.append(
                    f"output_dir {self.output_dir!r} is not writable: {exc}"
                )

        if errors:
            raise RunConfigError(
                "Invalid RunConfig:\n" + "\n".join(f"  - {e}" for e in errors)
            )


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalCase:
    """One evaluation input with optional gold labels."""

    # Required
    case_id: str
    dataset_id: str
    dataset_version: str
    query: str
    split: DatasetSplit
    corpus_condition: CorpusCondition
    category: CaseCategory

    # Optional gold labels
    expected_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    expected_decision: Optional[str] = None
    expected_conflict_labels: dict[str, str] = field(default_factory=dict)
    expected_claim_labels: dict[str, str] = field(default_factory=dict)
    expected_answer_verdict: Optional[str] = None

    # Metadata
    source_references: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("EvalCase.case_id must be non-empty")
        if not self.dataset_id.strip():
            raise ValueError("EvalCase.dataset_id must be non-empty")
        if not self.dataset_version.strip():
            raise ValueError("EvalCase.dataset_version must be non-empty")
        if not self.query.strip():
            raise ValueError("EvalCase.query must be non-empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"EvalCase schema_version mismatch: "
                f"expected {SCHEMA_VERSION}, got {self.schema_version}"
            )
        if self.expected_decision is not None:
            _VALID_DECISIONS = {
                "answer", "warning", "repair", "regenerate", "abstain"
            }
            if self.expected_decision not in _VALID_DECISIONS:
                raise ValueError(
                    f"EvalCase.expected_decision {self.expected_decision!r} "
                    f"is not a valid DecisionAction value. "
                    f"Valid: {sorted(_VALID_DECISIONS)}"
                )
        if self.expected_answer_verdict is not None:
            _VALID_VERDICTS = {
                "verified", "partially_verified", "unverified",
                "unsafe", "insufficient_evidence"
            }
            if self.expected_answer_verdict not in _VALID_VERDICTS:
                raise ValueError(
                    f"EvalCase.expected_answer_verdict "
                    f"{self.expected_answer_verdict!r} is not a valid "
                    f"AnswerVerdict value. Valid: {sorted(_VALID_VERDICTS)}"
                )
        for pair_id, rel in self.expected_conflict_labels.items():
            _VALID_RELS = {
                "compatible", "population-diff", "temporal-diff",
                "jurisdiction-diff", "dosage-diff", "genuine-conflict",
                "unresolved",
            }
            if rel not in _VALID_RELS:
                raise ValueError(
                    f"EvalCase.expected_conflict_labels[{pair_id!r}] = "
                    f"{rel!r} is not a valid EvidenceRelationship value."
                )
        for cid, label in self.expected_claim_labels.items():
            _VALID_LABELS = {
                "supported", "partially_supported", "contradicted",
                "unsupported", "uncertain", "not_verifiable",
            }
            if label not in _VALID_LABELS:
                raise ValueError(
                    f"EvalCase.expected_claim_labels[{cid!r}] = "
                    f"{label!r} is not a valid SupportLabel value."
                )


# ---------------------------------------------------------------------------
# EvalDataset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalDataset:
    """Versioned, split-labelled collection of EvalCases."""

    dataset_id: str
    version: str
    cases: tuple[EvalCase, ...]
    description: Optional[str] = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("EvalDataset.dataset_id must be non-empty")
        if not self.version.strip():
            raise ValueError("EvalDataset.version must be non-empty")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"EvalDataset schema_version mismatch: "
                f"expected {SCHEMA_VERSION}, got {self.schema_version}"
            )
        for c in self.cases:
            if c.dataset_id != self.dataset_id:
                raise ValueError(
                    f"EvalCase {c.case_id!r} has dataset_id "
                    f"{c.dataset_id!r} but EvalDataset.dataset_id is "
                    f"{self.dataset_id!r}"
                )
            if c.dataset_version != self.version:
                raise ValueError(
                    f"EvalCase {c.case_id!r} has dataset_version "
                    f"{c.dataset_version!r} but EvalDataset.version is "
                    f"{self.version!r}"
                )
        seen: set[str] = set()
        for c in self.cases:
            if c.case_id in seen:
                raise ValueError(
                    f"Duplicate case_id {c.case_id!r} in EvalDataset "
                    f"{self.dataset_id!r}"
                )
            seen.add(c.case_id)

    def by_split(self, split: DatasetSplit) -> tuple[EvalCase, ...]:
        """Return all cases in the given split."""
        return tuple(c for c in self.cases if c.split == split)

    def by_category(self, category: CaseCategory) -> tuple[EvalCase, ...]:
        """Return all cases with the given category."""
        return tuple(c for c in self.cases if c.category == category)

    def __len__(self) -> int:
        return len(self.cases)


__all__ = [
    "SCHEMA_VERSION",
    "CorpusCondition",
    "DatasetSplit",
    "CaseCategory",
    "CorpusConfig",
    "RunConfigError",
    "RunConfig",
    "EvalCase",
    "EvalDataset",
]