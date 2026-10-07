"""Tests for evaluation/schema.py — M15 File 4/38."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.schema import (
    SCHEMA_VERSION,
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
# Helpers
# ---------------------------------------------------------------------------

def _make_case(**overrides) -> EvalCase:
    defaults = dict(
        case_id="c1",
        dataset_id="ds1",
        dataset_version="1.0.0",
        query="What are visiting hours?",
        split=DatasetSplit.TEST,
        corpus_condition=CorpusCondition.CLEAN,
        category=CaseCategory.CLEAN,
    )
    defaults.update(overrides)
    return EvalCase(**defaults)


def _make_run_config(tmp_path: Path, **overrides) -> RunConfig:
    defaults = dict(
        max_workers=2,
        max_queue_depth=4,
        max_cases=100,
        output_dir=tmp_path,
        admission_timeout_seconds=1.0,
        wall_clock_deadline_seconds=60.0,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


# ---------------------------------------------------------------------------
# SCHEMA_VERSION
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    def test_version_is_int(self):
        assert isinstance(SCHEMA_VERSION, int)

    def test_version_is_positive(self):
        assert SCHEMA_VERSION >= 1


# ---------------------------------------------------------------------------
# CorpusCondition
# ---------------------------------------------------------------------------

class TestCorpusCondition:
    def test_all_values_present(self):
        values = {e.value for e in CorpusCondition}
        assert "clean" in values
        assert "poisoned_contradiction" in values
        assert "poisoned_stale" in values
        assert "poisoned_provenance" in values
        assert "poisoned_combined" in values

    def test_count(self):
        assert len(CorpusCondition) == 5


# ---------------------------------------------------------------------------
# DatasetSplit
# ---------------------------------------------------------------------------

class TestDatasetSplit:
    def test_all_values_present(self):
        values = {e.value for e in DatasetSplit}
        assert "dev" in values
        assert "val" in values
        assert "test" in values

    def test_count(self):
        # dev / calibration / test, legacy val, and the safety-regression suite
        assert {e.value for e in DatasetSplit} == {
            "dev", "calibration", "val", "test", "safety_regression",
        }


# ---------------------------------------------------------------------------
# CaseCategory
# ---------------------------------------------------------------------------

class TestCaseCategory:
    def test_clean_present(self):
        assert CaseCategory.CLEAN.value == "clean"

    def test_contradiction_present(self):
        assert CaseCategory.CONTRADICTION.value == "contradiction"

    def test_prompt_injection_present(self):
        assert CaseCategory.PROMPT_INJECTION.value == "prompt_injection"

    def test_at_least_ten_categories(self):
        assert len(CaseCategory) >= 10


# ---------------------------------------------------------------------------
# CorpusConfig
# ---------------------------------------------------------------------------

class TestCorpusConfig:
    def test_valid_construction(self):
        cc = CorpusConfig(
            store_path=Path("corpus/store.json"),
            bm25_path=Path("corpus/bm25.pkl"),
            fingerprint="abc123def456abcd",
            condition=CorpusCondition.CLEAN,
        )
        assert cc.condition == CorpusCondition.CLEAN
        assert cc.fingerprint == "abc123def456abcd"

    def test_empty_fingerprint_raises(self):
        with pytest.raises(ValueError, match="fingerprint"):
            CorpusConfig(
                store_path=Path("x"),
                bm25_path=Path("y"),
                fingerprint="",
                condition=CorpusCondition.CLEAN,
            )

    def test_short_fingerprint_raises(self):
        with pytest.raises(ValueError, match="fingerprint"):
            CorpusConfig(
                store_path=Path("x"),
                bm25_path=Path("y"),
                fingerprint="abc",
                condition=CorpusCondition.CLEAN,
            )

    def test_poisoned_condition(self):
        cc = CorpusConfig(
            store_path=Path("x"),
            bm25_path=Path("y"),
            fingerprint="abc123def456abcd",
            condition=CorpusCondition.POISONED_CONTRADICTION,
        )
        assert cc.condition == CorpusCondition.POISONED_CONTRADICTION

    def test_is_frozen(self):
        cc = CorpusConfig(
            store_path=Path("x"),
            bm25_path=Path("y"),
            fingerprint="abc123def456abcd",
            condition=CorpusCondition.CLEAN,
        )
        with pytest.raises(Exception):
            cc.fingerprint = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RunConfig validation
# ---------------------------------------------------------------------------

class TestRunConfig:
    def test_valid_construction(self, tmp_path):
        cfg = _make_run_config(tmp_path)
        assert cfg.max_workers == 2
        assert cfg.max_queue_depth == 4

    def test_max_workers_zero_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="max_workers"):
            _make_run_config(tmp_path, max_workers=0)

    def test_max_workers_negative_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="max_workers"):
            _make_run_config(tmp_path, max_workers=-1)

    def test_max_queue_depth_zero_raises(self, tmp_path):
        """queue.Queue(maxsize=0) is unbounded — explicitly prohibited."""
        with pytest.raises(RunConfigError, match="max_queue_depth"):
            _make_run_config(tmp_path, max_queue_depth=0)

    def test_max_queue_depth_negative_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="max_queue_depth"):
            _make_run_config(tmp_path, max_queue_depth=-1)

    def test_max_cases_zero_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="max_cases"):
            _make_run_config(tmp_path, max_cases=0)

    def test_max_cases_negative_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="max_cases"):
            _make_run_config(tmp_path, max_cases=-1)

    def test_admission_timeout_zero_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="admission_timeout"):
            _make_run_config(tmp_path, admission_timeout_seconds=0.0)

    def test_admission_timeout_negative_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="admission_timeout"):
            _make_run_config(tmp_path, admission_timeout_seconds=-1.0)

    def test_deadline_zero_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="wall_clock_deadline"):
            _make_run_config(tmp_path, wall_clock_deadline_seconds=0.0)

    def test_deadline_negative_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="wall_clock_deadline"):
            _make_run_config(tmp_path, wall_clock_deadline_seconds=-5.0)

    def test_warmup_negative_raises(self, tmp_path):
        with pytest.raises(RunConfigError, match="warmup"):
            _make_run_config(tmp_path, warmup_cases=-1)

    def test_invalid_output_dir_raises(self, tmp_path):
        """Pass an existing file as output_dir — is_file() check fires."""
        # Write a real file; schema now does explicit is_file() check
        # before mkdir, so this works identically on Windows and Linux.
        file_path = tmp_path / "i_am_a_file.txt"
        file_path.write_text("not a directory")
        with pytest.raises(RunConfigError, match="output_dir"):
            RunConfig(
                max_workers=1,
                max_queue_depth=1,
                max_cases=10,
                output_dir=file_path,
                admission_timeout_seconds=1.0,
                wall_clock_deadline_seconds=60.0,
            )

    def test_multiple_errors_reported(self, tmp_path):
        with pytest.raises(RunConfigError) as exc_info:
            _make_run_config(tmp_path, max_workers=0, max_queue_depth=0)
        msg = str(exc_info.value)
        assert "max_workers" in msg
        assert "max_queue_depth" in msg

    def test_is_frozen(self, tmp_path):
        cfg = _make_run_config(tmp_path)
        with pytest.raises(Exception):
            cfg.max_workers = 99  # type: ignore[misc]

    def test_raises_run_config_error_not_value_error(self, tmp_path):
        with pytest.raises(RunConfigError):
            _make_run_config(tmp_path, max_workers=0)


# ---------------------------------------------------------------------------
# EvalCase
# ---------------------------------------------------------------------------

class TestEvalCase:
    def test_valid_minimal(self):
        c = _make_case()
        assert c.case_id == "c1"
        assert c.schema_version == SCHEMA_VERSION

    def test_empty_case_id_raises(self):
        with pytest.raises(ValueError, match="case_id"):
            _make_case(case_id="")

    def test_whitespace_case_id_raises(self):
        with pytest.raises(ValueError, match="case_id"):
            _make_case(case_id="   ")

    def test_empty_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            _make_case(query="")

    def test_whitespace_query_raises(self):
        with pytest.raises(ValueError, match="query"):
            _make_case(query="   ")

    def test_wrong_schema_version_raises(self):
        with pytest.raises(ValueError, match="schema_version"):
            _make_case(schema_version=SCHEMA_VERSION + 99)

    def test_invalid_expected_decision_raises(self):
        with pytest.raises(ValueError, match="DecisionAction"):
            _make_case(expected_decision="invalid_action")

    def test_valid_expected_decisions(self):
        for action in ("answer", "warning", "repair", "regenerate", "abstain"):
            c = _make_case(expected_decision=action)
            assert c.expected_decision == action

    def test_none_expected_decision_allowed(self):
        c = _make_case(expected_decision=None)
        assert c.expected_decision is None

    def test_invalid_conflict_label_raises(self):
        with pytest.raises(ValueError, match="EvidenceRelationship"):
            _make_case(expected_conflict_labels={"p1": "not_a_relationship"})

    def test_valid_conflict_labels(self):
        for rel in ("genuine-conflict", "compatible", "population-diff",
                    "temporal-diff", "jurisdiction-diff", "dosage-diff",
                    "unresolved"):
            c = _make_case(expected_conflict_labels={"p1": rel})
            assert c.expected_conflict_labels["p1"] == rel

    def test_invalid_claim_label_raises(self):
        with pytest.raises(ValueError, match="SupportLabel"):
            _make_case(expected_claim_labels={"c1": "invalid_label"})

    def test_valid_claim_labels(self):
        for lbl in ("supported", "partially_supported", "contradicted",
                    "unsupported", "uncertain", "not_verifiable"):
            c = _make_case(expected_claim_labels={"c1": lbl})
            assert c.expected_claim_labels["c1"] == lbl

    def test_invalid_answer_verdict_raises(self):
        with pytest.raises(ValueError, match="AnswerVerdict"):
            _make_case(expected_answer_verdict="bad_verdict")

    def test_valid_answer_verdicts(self):
        for v in ("verified", "partially_verified", "unverified",
                  "unsafe", "insufficient_evidence"):
            c = _make_case(expected_answer_verdict=v)
            assert c.expected_answer_verdict == v

    def test_optional_fields_default_empty(self):
        c = _make_case()
        assert c.expected_chunk_ids == ()
        assert c.expected_conflict_labels == {}
        assert c.expected_claim_labels == {}
        assert c.expected_decision is None
        assert c.expected_answer_verdict is None
        assert c.tags == ()
        assert c.notes is None

    def test_corpus_condition_stored(self):
        c = _make_case(corpus_condition=CorpusCondition.POISONED_STALE)
        assert c.corpus_condition == CorpusCondition.POISONED_STALE

    def test_all_splits_accepted(self):
        for split in DatasetSplit:
            c = _make_case(split=split)
            assert c.split == split

    def test_all_categories_accepted(self):
        for cat in CaseCategory:
            c = _make_case(category=cat)
            assert c.category == cat

    def test_is_frozen(self):
        c = _make_case()
        with pytest.raises(Exception):
            c.case_id = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvalDataset
# ---------------------------------------------------------------------------

class TestEvalDataset:
    def test_valid_single_case(self):
        c = _make_case()
        ds = EvalDataset(dataset_id="ds1", version="1.0.0", cases=(c,))
        assert len(ds) == 1

    def test_empty_cases_allowed(self):
        ds = EvalDataset(dataset_id="ds1", version="1.0.0", cases=())
        assert len(ds) == 0

    def test_duplicate_case_id_raises(self):
        c = _make_case()
        with pytest.raises(ValueError, match="Duplicate"):
            EvalDataset(dataset_id="ds1", version="1.0.0", cases=(c, c))

    def test_mismatched_dataset_id_raises(self):
        c = _make_case(dataset_id="other_ds")
        with pytest.raises(ValueError, match="dataset_id"):
            EvalDataset(dataset_id="ds1", version="1.0.0", cases=(c,))

    def test_mismatched_version_raises(self):
        c = _make_case(dataset_version="2.0.0")
        with pytest.raises(ValueError, match="dataset_version"):
            EvalDataset(dataset_id="ds1", version="1.0.0", cases=(c,))

    def test_empty_dataset_id_raises(self):
        with pytest.raises(ValueError, match="dataset_id"):
            EvalDataset(dataset_id="", version="1.0.0", cases=())

    def test_wrong_schema_version_raises(self):
        with pytest.raises(ValueError, match="schema_version"):
            EvalDataset(
                dataset_id="ds1", version="1.0.0", cases=(),
                schema_version=SCHEMA_VERSION + 99,
            )

    def test_by_split(self):
        c_dev  = _make_case(case_id="c1", split=DatasetSplit.DEV)
        c_test = _make_case(case_id="c2", split=DatasetSplit.TEST)
        ds = EvalDataset(dataset_id="ds1", version="1.0.0",
                         cases=(c_dev, c_test))
        assert len(ds.by_split(DatasetSplit.DEV)) == 1
        assert len(ds.by_split(DatasetSplit.TEST)) == 1
        assert len(ds.by_split(DatasetSplit.VAL)) == 0

    def test_by_category(self):
        c1 = _make_case(case_id="c1", category=CaseCategory.CLEAN)
        c2 = _make_case(case_id="c2", category=CaseCategory.CONTRADICTION)
        ds = EvalDataset(dataset_id="ds1", version="1.0.0", cases=(c1, c2))
        assert len(ds.by_category(CaseCategory.CLEAN)) == 1
        assert len(ds.by_category(CaseCategory.CONTRADICTION)) == 1
        assert len(ds.by_category(CaseCategory.STALE_DOCUMENT)) == 0

    def test_len(self):
        cases = tuple(_make_case(case_id=f"c{i}") for i in range(5))
        ds = EvalDataset(dataset_id="ds1", version="1.0.0", cases=cases)
        assert len(ds) == 5

    def test_is_frozen(self):
        ds = EvalDataset(dataset_id="ds1", version="1.0.0", cases=())
        with pytest.raises(Exception):
            ds.dataset_id = "other"  # type: ignore[misc]