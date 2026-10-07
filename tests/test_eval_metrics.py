"""Tests for evaluation/metrics.py — M15 File 5/38."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evaluation.metrics import (
    abstention_rate,
    brier_score,
    claim_confusion_matrix,
    claim_macro_f1,
    claim_per_class_prf,
    conflict_confusion_matrix,
    conflict_precision_recall_f1,
    decision_accuracy,
    error_rate,
    expected_calibration_error,
    false_abstention_rate,
    latency_percentiles,
    mean_ndcg_at_k,
    mean_reciprocal_rank,
    mean_recall_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    rejection_rate,
    reliability_data,
    throughput,
    unsafe_answer_rate,
)


# ---------------------------------------------------------------------------
# A. Retrieval
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_zero_recall(self):
        assert recall_at_k(["x", "y"], {"a", "b"}, 2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "x"], {"a", "b"}, 2) == 0.5

    def test_k_truncates(self):
        assert recall_at_k(["x", "a"], {"a"}, 1) == 0.0
        assert recall_at_k(["a", "x"], {"a"}, 1) == 1.0

    def test_empty_relevant_returns_none(self):
        assert recall_at_k(["a", "b"], set(), 2) is None

    def test_empty_retrieved_zero_recall(self):
        assert recall_at_k([], {"a"}, 3) == 0.0

    def test_k_larger_than_retrieved(self):
        assert recall_at_k(["a"], {"a", "b"}, 10) == 0.5


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a"], {"a"}) == 0.5

    def test_not_found_returns_zero(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_empty_retrieved(self):
        assert reciprocal_rank([], {"a"}) == 0.0

    def test_empty_relevant(self):
        assert reciprocal_rank(["a"], set()) == 0.0


class TestNdcgAtK:
    def test_perfect(self):
        v = ndcg_at_k(["a", "b"], {"a", "b"}, 2)
        assert v is not None
        assert abs(v - 1.0) < 1e-9

    def test_empty_relevant_returns_none(self):
        assert ndcg_at_k(["a"], set(), 3) is None

    def test_zero_hits(self):
        v = ndcg_at_k(["x", "y"], {"a", "b"}, 2)
        assert v == 0.0

    def test_partial(self):
        v = ndcg_at_k(["a", "x"], {"a", "b"}, 2)
        assert v is not None
        assert 0 < v < 1


class TestMeanRetrievalMetrics:
    def test_mean_recall_empty_cases(self):
        assert mean_recall_at_k([], 5) is None

    def test_mean_recall_all_none(self):
        assert mean_recall_at_k([( ["a"], set()), (["b"], set())], 3) is None

    def test_mean_recall_mixed(self):
        cases = [(["a"], {"a"}), (["x"], set())]
        v = mean_recall_at_k(cases, 1)
        assert v == 1.0

    def test_mrr_empty(self):
        assert mean_reciprocal_rank([]) is None

    def test_mrr_includes_zero(self):
        cases = [(["a"], {"a"}), (["x"], {"y"})]
        v = mean_reciprocal_rank(cases)
        assert v == 0.5

    def test_mean_ndcg_empty(self):
        assert mean_ndcg_at_k([], 5) is None


# ---------------------------------------------------------------------------
# B. Conflict
# ---------------------------------------------------------------------------

class TestConflictPRF:
    def _preds_gold(self):
        preds = {"p1": "genuine-conflict", "p2": "compatible",
                 "p3": "genuine-conflict"}
        gold  = {"p1": "genuine-conflict", "p2": "genuine-conflict",
                 "p3": "compatible"}
        return preds, gold

    def test_tp_fp_fn(self):
        preds, gold = self._preds_gold()
        r = conflict_precision_recall_f1(preds, gold)
        assert r["tp"] == 1
        assert r["fp"] == 1
        assert r["fn"] == 1

    def test_precision(self):
        preds, gold = self._preds_gold()
        r = conflict_precision_recall_f1(preds, gold)
        assert r["precision"] == 0.5

    def test_recall(self):
        preds, gold = self._preds_gold()
        r = conflict_precision_recall_f1(preds, gold)
        assert r["recall"] == 0.5

    def test_f1(self):
        preds, gold = self._preds_gold()
        r = conflict_precision_recall_f1(preds, gold)
        assert r["f1"] == 0.5

    def test_no_positive_gold_recall_none(self):
        r = conflict_precision_recall_f1(
            {"p1": "compatible"}, {"p1": "compatible"}
        )
        assert r["recall"] is None

    def test_no_positive_pred_precision_none(self):
        r = conflict_precision_recall_f1(
            {"p1": "compatible"}, {"p1": "genuine-conflict"}
        )
        assert r["precision"] is None
        assert r["f1"] is None

    def test_empty_inputs(self):
        r = conflict_precision_recall_f1({}, {})
        assert r["precision"] is None
        assert r["recall"] is None
        assert r["f1"] is None

    def test_perfect_precision_recall(self):
        r = conflict_precision_recall_f1(
            {"p1": "genuine-conflict"},
            {"p1": "genuine-conflict"},
        )
        assert r["precision"] == 1.0
        assert r["recall"] == 1.0
        assert r["f1"] == 1.0

    def test_no_overlap_in_pair_ids(self):
        r = conflict_precision_recall_f1(
            {"p1": "genuine-conflict"}, {"p2": "genuine-conflict"}
        )
        assert r["tp"] == 0
        assert r["precision"] is None
        assert r["recall"] is None


class TestConflictConfusionMatrix:
    def test_returns_dict(self):
        cm = conflict_confusion_matrix(
            {"p1": "genuine-conflict"},
            {"p1": "genuine-conflict"},
        )
        assert isinstance(cm, dict)
        assert "genuine-conflict" in cm

    def test_correct_count(self):
        cm = conflict_confusion_matrix(
            {"p1": "genuine-conflict"},
            {"p1": "genuine-conflict"},
        )
        assert cm["genuine-conflict"]["genuine-conflict"] == 1

    def test_unknown_label(self):
        cm = conflict_confusion_matrix(
            {"p1": "unknown_label"},
            {"p1": "genuine-conflict"},
        )
        assert cm["genuine-conflict"]["unknown"] == 1


# ---------------------------------------------------------------------------
# C. Claim
# ---------------------------------------------------------------------------

class TestClaimPRF:
    def test_supported_per_class(self):
        r = claim_per_class_prf(
            {"c1": "supported", "c2": "contradicted"},
            {"c1": "supported", "c2": "supported"},
            "supported",
        )
        assert r["tp"] == 1
        assert r["fn"] == 1
        assert r["precision"] == 1.0
        assert r["recall"] == 0.5

    def test_precision_none_when_no_positive_pred(self):
        r = claim_per_class_prf(
            {"c1": "unsupported"},
            {"c1": "supported"},
            "supported",
        )
        assert r["precision"] is None

    def test_recall_none_when_no_positive_gold(self):
        r = claim_per_class_prf(
            {"c1": "supported"},
            {"c1": "unsupported"},
            "supported",
        )
        assert r["recall"] is None

    def test_perfect(self):
        r = claim_per_class_prf(
            {"c1": "supported"},
            {"c1": "supported"},
            "supported",
        )
        assert r["precision"] == 1.0
        assert r["recall"] == 1.0
        assert r["f1"] == 1.0

    def test_no_overlap(self):
        r = claim_per_class_prf(
            {"c1": "supported"},
            {"c2": "supported"},
            "supported",
        )
        assert r["tp"] == 0
        assert r["precision"] is None
        assert r["recall"] is None


class TestClaimMacroF1:
    def test_empty_returns_none(self):
        assert claim_macro_f1({}, {}) is None

    def test_single_class(self):
        v = claim_macro_f1(
            {"c1": "supported"},
            {"c1": "supported"},
        )
        assert v == 1.0

    def test_absent_gold_class_excluded(self):
        v = claim_macro_f1(
            {"c1": "supported"},
            {"c1": "supported"},
        )
        assert v is not None

    def test_no_defined_f1_returns_none(self):
        v = claim_macro_f1(
            {"c1": "supported"},
            {"c2": "supported"},
        )
        assert v is None


class TestClaimConfusionMatrix:
    def test_returns_dict(self):
        cm = claim_confusion_matrix(
            {"c1": "supported"},
            {"c1": "supported"},
        )
        assert "supported" in cm

    def test_correct_cell(self):
        cm = claim_confusion_matrix(
            {"c1": "supported"},
            {"c1": "supported"},
        )
        assert cm["supported"]["supported"] == 1


# ---------------------------------------------------------------------------
# D. Decision
# ---------------------------------------------------------------------------

class TestDecisionAccuracy:
    def test_all_correct(self):
        assert decision_accuracy(["answer", "abstain"], ["answer", "abstain"],
                                 "answer") == 1.0

    def test_half_correct(self):
        v = decision_accuracy(
            ["answer", "abstain"],
            ["answer", "answer"],
            "answer",
        )
        assert v == 0.5

    def test_action_absent_from_gold(self):
        assert decision_accuracy(["answer"], ["abstain"], "warning") is None

    def test_empty_lists(self):
        assert decision_accuracy([], [], "answer") is None

    def test_all_wrong(self):
        assert decision_accuracy(["abstain"], ["answer"], "answer") == 0.0


class TestAbstentionRate:
    def test_empty_returns_none(self):
        assert abstention_rate([]) is None

    def test_all_abstain(self):
        assert abstention_rate(["abstain", "abstain"]) == 1.0

    def test_none_abstain(self):
        assert abstention_rate(["answer", "warning"]) == 0.0

    def test_partial(self):
        assert abs(abstention_rate(["abstain", "answer", "abstain"]) - 2/3) < 1e-9


class TestFalseAbstentionRate:
    def test_all_gold_abstain_returns_none(self):
        assert false_abstention_rate(["answer"], ["abstain"]) is None

    def test_no_false_abstentions(self):
        assert false_abstention_rate(["answer", "warning"],
                                     ["answer", "warning"]) == 0.0

    def test_all_false_abstentions(self):
        assert false_abstention_rate(["abstain", "abstain"],
                                     ["answer", "warning"]) == 1.0

    def test_partial(self):
        v = false_abstention_rate(
            ["abstain", "answer", "abstain"],
            ["answer", "answer", "warning"],
        )
        assert abs(v - 2/3) < 1e-9

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            false_abstention_rate(["answer"], ["answer", "abstain"])


class TestUnsafeAnswerRate:
    def test_mock_llm_returns_none(self):
        assert unsafe_answer_rate(["answer"], ["unsafe"], mock_llm=True) is None

    def test_no_answer_or_warning_returns_none(self):
        assert unsafe_answer_rate(["abstain"], [None], mock_llm=False) is None

    def test_all_none_verdicts_returns_none(self):
        assert unsafe_answer_rate(["answer", "warning"],
                                  [None, None], mock_llm=False) is None

    def test_half_unsafe(self):
        v = unsafe_answer_rate(
            ["answer", "warning", "answer"],
            ["unsafe", "verified", None],
            mock_llm=False,
        )
        assert v == 0.5

    def test_zero_unsafe(self):
        v = unsafe_answer_rate(
            ["answer", "warning"],
            ["verified", "partially_verified"],
            mock_llm=False,
        )
        assert v == 0.0

    def test_all_unsafe(self):
        v = unsafe_answer_rate(
            ["answer", "warning"],
            ["unsafe", "unsafe"],
            mock_llm=False,
        )
        assert v == 1.0

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            unsafe_answer_rate(["answer"], ["unsafe", "verified"],
                               mock_llm=False)

    def test_abstain_excluded_from_denominator(self):
        v = unsafe_answer_rate(
            ["answer", "abstain", "warning"],
            ["unsafe", "unsafe", "verified"],
            mock_llm=False,
        )
        assert v == 0.5


# ---------------------------------------------------------------------------
# E. Calibration
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_mock_llm_returns_none(self):
        assert brier_score([0.8], ["verified"], mock_llm=True) is None

    def test_below_min_cases_returns_none(self):
        assert brier_score([0.8] * 20, ["verified"] * 20,
                           mock_llm=False) is None

    def test_all_none_verdicts_returns_none(self):
        assert brier_score([0.8] * 30, [None] * 30, mock_llm=False) is None

    def test_valid_result_in_range(self):
        confs = [0.9] * 15 + [0.1] * 15
        verd  = ["verified"] * 15 + ["unverified"] * 15
        v = brier_score(confs, verd, mock_llm=False)
        assert v is not None
        assert 0.0 <= v <= 1.0

    def test_perfect_calibration(self):
        confs = [1.0] * 30
        verd  = ["verified"] * 30
        v = brier_score(confs, verd, mock_llm=False)
        assert v is not None
        assert abs(v) < 1e-9

    def test_worst_calibration(self):
        confs = [0.0] * 30
        verd  = ["verified"] * 30
        v = brier_score(confs, verd, mock_llm=False)
        assert v is not None
        assert abs(v - 1.0) < 1e-9

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError):
            brier_score([0.5, 0.5], ["verified"], mock_llm=False)

    def test_none_verdicts_excluded(self):
        confs = [0.8] * 50
        verd  = [None] * 20 + ["verified"] * 30
        v = brier_score(confs, verd, mock_llm=False)
        assert v is not None

    def test_proxy_correct_verdicts(self):
        confs = [1.0] * 15 + [1.0] * 15
        verd  = ["verified"] * 15 + ["partially_verified"] * 15
        v = brier_score(confs, verd, mock_llm=False)
        assert v is not None
        assert abs(v) < 1e-9


class TestECE:
    def test_mock_llm_returns_none(self):
        assert expected_calibration_error([0.5] * 30, ["verified"] * 30,
                                          mock_llm=True) is None

    def test_below_min_returns_none(self):
        assert expected_calibration_error([0.5] * 20, ["verified"] * 20,
                                          mock_llm=False) is None

    def test_valid_result_in_range(self):
        confs = [0.9] * 15 + [0.1] * 15
        verd  = ["verified"] * 15 + ["unverified"] * 15
        v = expected_calibration_error(confs, verd, mock_llm=False)
        assert v is not None
        assert 0.0 <= v <= 1.0

    def test_perfect_calibration_low_ece(self):
        confs = [0.95] * 30
        verd  = ["verified"] * 30
        v = expected_calibration_error(confs, verd, mock_llm=False)
        assert v is not None
        assert v < 0.1


class TestReliabilityData:
    def test_mock_llm_returns_none(self):
        assert reliability_data([0.5] * 30, ["verified"] * 30,
                                mock_llm=True) is None

    def test_below_min_returns_none(self):
        assert reliability_data([0.5] * 20, ["verified"] * 20,
                                mock_llm=False) is None

    def test_returns_list_of_dicts(self):
        confs = [0.9] * 15 + [0.1] * 15
        verd  = ["verified"] * 15 + ["unverified"] * 15
        rd = reliability_data(confs, verd, mock_llm=False)
        assert rd is not None
        assert isinstance(rd, list)
        assert len(rd) > 0

    def test_dict_keys(self):
        confs = [0.9] * 30
        verd  = ["verified"] * 30
        rd = reliability_data(confs, verd, mock_llm=False)
        assert rd is not None
        for entry in rd:
            assert "bin_lo" in entry
            assert "bin_hi" in entry
            assert "mean_confidence" in entry
            assert "fraction_correct" in entry
            assert "n_cases" in entry

    def test_n_cases_sums_to_included(self):
        confs = [0.9] * 15 + [0.1] * 15
        verd  = ["verified"] * 15 + ["unverified"] * 15
        rd = reliability_data(confs, verd, mock_llm=False)
        assert rd is not None
        assert sum(e["n_cases"] for e in rd) == 30


# ---------------------------------------------------------------------------
# F. Performance
# ---------------------------------------------------------------------------

class TestLatencyPercentiles:
    def test_empty_returns_none(self):
        p = latency_percentiles([])
        assert p["p50"] is None
        assert p["p95"] is None
        assert p["p99"] is None

    def test_warmup_excluded(self):
        lats = [100.0, 0.1, 0.2]
        p = latency_percentiles(lats, warmup_cases=1)
        assert p["p50"] is not None
        assert p["p50"] <= 0.2

    def test_single_value(self):
        p = latency_percentiles([1.5])
        assert p["p50"] == 1.5
        assert p["p95"] == 1.5
        assert p["p99"] == 1.5

    def test_n_field(self):
        p = latency_percentiles([0.1, 0.2, 0.3], warmup_cases=1)
        assert p["n"] == 2


class TestThroughput:
    def test_normal(self):
        assert abs(throughput(10, 5.0) - 2.0) < 1e-9

    def test_zero_wall_clock_returns_none(self):
        assert throughput(10, 0.0) is None

    def test_negative_wall_clock_returns_none(self):
        assert throughput(10, -1.0) is None

    def test_zero_completed(self):
        assert throughput(0, 5.0) == 0.0


class TestErrorRate:
    def test_normal(self):
        assert abs(error_rate(2, 10) - 0.2) < 1e-9

    def test_zero_submitted_returns_none(self):
        assert error_rate(0, 0) is None

    def test_zero_errors(self):
        assert error_rate(0, 10) == 0.0

    def test_all_errors(self):
        assert error_rate(10, 10) == 1.0


class TestRejectionRate:
    def test_normal(self):
        assert abs(rejection_rate(3, 10) - 0.3) < 1e-9

    def test_zero_submitted_returns_none(self):
        assert rejection_rate(0, 0) is None

    def test_zero_rejections(self):
        assert rejection_rate(0, 10) == 0.0