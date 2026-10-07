"""evaluation/metrics.py

Pure metric functions for the RASVC-X M15 research evaluation framework.

All functions are pure (no side effects, no I/O, no global state).
Every function returns None for undefined denominators rather than raising.
No function imports from the RASVC-X pipeline; inputs are plain Python
types derived from CaseResult.data dicts.

Metric groups:
  A. Retrieval     -- Recall@K, MRR, nDCG@K
  B. Conflict      -- Precision, Recall, F1, confusion matrix
  C. Claim         -- Per-class P/R/F1, macro-F1, confusion matrix
  D. Decision      -- Per-class accuracy, abstention rate, false-abstention
                      rate, unsafe-answer rate
  E. Calibration   -- Brier score, ECE (10 equal-width bins),
                      reliability data (proxy only; not clinical correctness)
  F. Performance   -- p50/p95/p99 latency, throughput, error rate

VERIFIED ENUM VALUES (from rasvcx source, this session):
  AnswerVerdict: verified, partially_verified, unverified, unsafe,
                 insufficient_evidence
  SupportLabel:  supported, partially_supported, contradicted,
                 unsupported, uncertain, not_verifiable
  EvidenceRelationship: compatible, population-diff, temporal-diff,
                        jurisdiction-diff, dosage-diff, genuine-conflict,
                        unresolved
  DecisionAction: answer, warning, repair, regenerate, abstain

LIMITATIONS (must not be removed):
  - MockLLMClient results must not be used to compute claim verification
    accuracy or calibration metrics.  Call sites must set mock_llm=True
    and must not interpret None results as zero.
  - Calibration correctness proxy: answer_verdict (verified/partially_verified
    -> 1, others -> 0) is a pipeline output, not an independent clinical
    correctness label.  Reported Brier score and ECE measure operational
    proxy calibration only.
  - Minimum 30 cases for calibration is a reporting threshold chosen for
    this project, not a statistical reliability guarantee.
  - No metric in this module implies clinical validity or deployment
    readiness.
"""

from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Type aliases (plain Python; no pipeline imports)
# ---------------------------------------------------------------------------

RetrievedIds = list[str]
RelevantIds = set[str]

ConflictPredictions = dict[str, str]
ConflictGold        = dict[str, str]

ClaimPredictions = dict[str, str]
ClaimGold        = dict[str, str]

DecisionActions = list

ConfidenceScores  = list[float]
VerdictsForProxy  = list[Optional[str]]

LatencySeconds = list[float]

# ---------------------------------------------------------------------------
# Verified constant sets (from source, this session)
# ---------------------------------------------------------------------------

_SUPPORT_LABELS = frozenset({
    "supported", "partially_supported", "contradicted",
    "unsupported", "uncertain", "not_verifiable",
})
_EVIDENCE_RELATIONSHIPS = frozenset({
    "compatible", "population-diff", "temporal-diff",
    "jurisdiction-diff", "dosage-diff", "genuine-conflict", "unresolved",
})
_DECISION_ACTIONS = frozenset({
    "answer", "warning", "repair", "regenerate", "abstain",
})
_ANSWER_VERDICTS = frozenset({
    "verified", "partially_verified", "unverified",
    "unsafe", "insufficient_evidence",
})
_GENUINE_CONFLICT = "genuine-conflict"

_PROXY_CORRECT_VERDICTS = frozenset({"verified", "partially_verified"})

_CALIBRATION_MIN_CASES = 30
_ECE_N_BINS = 10


# ---------------------------------------------------------------------------
# A. Retrieval metrics
# ---------------------------------------------------------------------------


def recall_at_k(retrieved: RetrievedIds, relevant: RelevantIds, k: int) -> Optional[float]:
    """Recall@K: fraction of relevant items found in top-K retrieved."""
    if not relevant:
        return None
    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: RetrievedIds, relevant: RelevantIds) -> float:
    """Reciprocal rank of the first relevant item in retrieved list."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: RetrievedIds, relevant: RelevantIds, k: int) -> Optional[float]:
    """nDCG@K with binary relevance."""
    if not relevant:
        return None

    def dcg(items: list[str], rel: set[str], k_: int) -> float:
        return sum(
            (1.0 / math.log2(i + 2))
            for i, item in enumerate(items[:k_])
            if item in rel
        )

    actual_dcg = dcg(retrieved, relevant, k)
    ideal_items = list(relevant)[:k]
    ideal_dcg = dcg(ideal_items, relevant, k)

    if ideal_dcg == 0.0:
        return None
    return actual_dcg / ideal_dcg


def mean_recall_at_k(
    cases: list[tuple[RetrievedIds, RelevantIds]], k: int
) -> Optional[float]:
    """Mean Recall@K over cases with at least one relevant item."""
    values = [recall_at_k(r, rel, k) for r, rel in cases]
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


def mean_reciprocal_rank(
    cases: list[tuple[RetrievedIds, RelevantIds]]
) -> Optional[float]:
    """MRR over all cases; includes 0.0 for no-hit cases."""
    if not cases:
        return None
    return sum(reciprocal_rank(r, rel) for r, rel in cases) / len(cases)


def mean_ndcg_at_k(
    cases: list[tuple[RetrievedIds, RelevantIds]], k: int
) -> Optional[float]:
    """Mean nDCG@K over cases with at least one relevant item."""
    values = [ndcg_at_k(r, rel, k) for r, rel in cases]
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None


# ---------------------------------------------------------------------------
# B. Conflict detection metrics
# ---------------------------------------------------------------------------


def conflict_confusion_matrix(
    predictions: ConflictPredictions,
    gold: ConflictGold,
) -> dict[str, dict[str, int]]:
    """7x7 confusion matrix over EvidenceRelationship values."""
    matrix: dict[str, dict[str, int]] = {}
    labels = _EVIDENCE_RELATIONSHIPS | {"unknown"}
    for lbl in labels:
        matrix[lbl] = {l2: 0 for l2 in labels}

    for pair_id, gold_label in gold.items():
        if pair_id not in predictions:
            continue
        pred_label = predictions[pair_id]
        g = gold_label if gold_label in _EVIDENCE_RELATIONSHIPS else "unknown"
        p = pred_label if pred_label in _EVIDENCE_RELATIONSHIPS else "unknown"
        matrix[g][p] += 1

    return matrix


def conflict_precision_recall_f1(
    predictions: ConflictPredictions,
    gold: ConflictGold,
) -> dict[str, Optional[float]]:
    """Binary conflict detection P/R/F1. Positive class: genuine-conflict."""
    tp = fp = fn = 0
    common = set(predictions) & set(gold)

    for pair_id in common:
        pred_pos = predictions[pair_id] == _GENUINE_CONFLICT
        gold_pos = gold[pair_id] == _GENUINE_CONFLICT
        if pred_pos and gold_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif not pred_pos and gold_pos:
            fn += 1

    precision: Optional[float] = tp / (tp + fp) if (tp + fp) > 0 else None
    recall: Optional[float] = tp / (tp + fn) if (tp + fn) > 0 else None

    f1: Optional[float] = None
    if precision is not None and recall is not None:
        denom = precision + recall
        if denom > 0:
            f1 = 2 * precision * recall / denom

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# C. Claim verification metrics
# ---------------------------------------------------------------------------


def claim_confusion_matrix(
    predictions: ClaimPredictions,
    gold: ClaimGold,
) -> dict[str, dict[str, int]]:
    """6x6 confusion matrix over SupportLabel values."""
    matrix: dict[str, dict[str, int]] = {}
    labels = _SUPPORT_LABELS | {"unknown"}
    for lbl in labels:
        matrix[lbl] = {l2: 0 for l2 in labels}

    for cid, gold_label in gold.items():
        if cid not in predictions:
            continue
        pred_label = predictions[cid]
        g = gold_label if gold_label in _SUPPORT_LABELS else "unknown"
        p = pred_label if pred_label in _SUPPORT_LABELS else "unknown"
        matrix[g][p] += 1

    return matrix


def claim_per_class_prf(
    predictions: ClaimPredictions,
    gold: ClaimGold,
    label: str,
) -> dict[str, Optional[float]]:
    """One-vs-rest P/R/F1 for a single SupportLabel value."""
    common = set(predictions) & set(gold)
    tp = fp = fn = 0

    for cid in common:
        pred_pos = predictions[cid] == label
        gold_pos = gold[cid] == label
        if pred_pos and gold_pos:
            tp += 1
        elif pred_pos and not gold_pos:
            fp += 1
        elif not pred_pos and gold_pos:
            fn += 1

    precision: Optional[float] = tp / (tp + fp) if (tp + fp) > 0 else None
    recall: Optional[float] = tp / (tp + fn) if (tp + fn) > 0 else None
    f1: Optional[float] = None
    if precision is not None and recall is not None:
        denom = precision + recall
        if denom > 0:
            f1 = 2 * precision * recall / denom

    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "label": label}


def claim_macro_f1(
    predictions: ClaimPredictions,
    gold: ClaimGold,
) -> Optional[float]:
    """Macro-F1 over SupportLabel values present in gold set."""
    gold_classes = set(gold.values()) & _SUPPORT_LABELS
    f1_values: list[float] = []

    for label in gold_classes:
        result = claim_per_class_prf(predictions, gold, label)
        if result["f1"] is not None:
            f1_values.append(result["f1"])

    return sum(f1_values) / len(f1_values) if f1_values else None


# ---------------------------------------------------------------------------
# D. Decision quality metrics
# ---------------------------------------------------------------------------


def decision_accuracy(
    predictions: list,
    gold: list,
    action: str,
) -> Optional[float]:
    """Fraction of cases where system action matches expected for one action."""
    gold_indices = [i for i, g in enumerate(gold) if g == action]
    if not gold_indices:
        return None
    correct = sum(1 for i in gold_indices if predictions[i] == action)
    return correct / len(gold_indices)


def abstention_rate(predictions: list[str]) -> Optional[float]:
    """Fraction of cases with action='abstain'."""
    if not predictions:
        return None
    return sum(1 for p in predictions if p == "abstain") / len(predictions)


def false_abstention_rate(
    predictions: list[str],
    gold: list[str],
) -> Optional[float]:
    """Fraction of cases where system abstained but expected not to."""
    if len(predictions) != len(gold):
        raise ValueError("predictions and gold must have the same length")
    expected_non_abstain = [i for i, g in enumerate(gold) if g != "abstain"]
    if not expected_non_abstain:
        return None
    false_abstains = sum(
        1 for i in expected_non_abstain if predictions[i] == "abstain"
    )
    return false_abstains / len(expected_non_abstain)


def unsafe_answer_rate(
    actions: list[str],
    answer_verdicts: list[Optional[str]],
    mock_llm: bool,
) -> Optional[float]:
    """Fraction of ANSWER/WARNING decisions where answer_verdict is 'unsafe'."""
    if mock_llm:
        return None
    if len(actions) != len(answer_verdicts):
        raise ValueError("actions and answer_verdicts must have same length")

    answering = [
        (actions[i], answer_verdicts[i])
        for i in range(len(actions))
        if actions[i] in ("answer", "warning")
        and answer_verdicts[i] is not None
    ]
    if not answering:
        return None

    unsafe = sum(1 for _, v in answering if v == "unsafe")
    return unsafe / len(answering)


# ---------------------------------------------------------------------------
# E. Calibration metrics
# ---------------------------------------------------------------------------


def _calibration_proxy_correct(verdict: Optional[str]) -> Optional[int]:
    """Convert answer_verdict to proxy correctness label."""
    if verdict is None:
        return None
    if verdict in _PROXY_CORRECT_VERDICTS:
        return 1
    return 0


def brier_score(
    confidences: ConfidenceScores,
    verdicts: VerdictsForProxy,
    mock_llm: bool,
) -> Optional[float]:
    """Brier score: mean((confidence - proxy_correct)^2)."""
    if mock_llm:
        return None
    if len(confidences) != len(verdicts):
        raise ValueError("confidences and verdicts must have same length")

    pairs = [
        (confidences[i], _calibration_proxy_correct(verdicts[i]))
        for i in range(len(confidences))
        if _calibration_proxy_correct(verdicts[i]) is not None
    ]
    if len(pairs) < _CALIBRATION_MIN_CASES:
        return None

    return sum((c - y) ** 2 for c, y in pairs) / len(pairs)


def expected_calibration_error(
    confidences: ConfidenceScores,
    verdicts: VerdictsForProxy,
    mock_llm: bool,
    n_bins: int = _ECE_N_BINS,
) -> Optional[float]:
    """Expected Calibration Error (ECE) with equal-width bins."""
    if mock_llm:
        return None
    if len(confidences) != len(verdicts):
        raise ValueError("confidences and verdicts must have same length")

    pairs = [
        (confidences[i], _calibration_proxy_correct(verdicts[i]))
        for i in range(len(confidences))
        if _calibration_proxy_correct(verdicts[i]) is not None
    ]
    n_included = len(pairs)
    if n_included < _CALIBRATION_MIN_CASES:
        return None

    bin_width = 1.0 / n_bins
    ece = 0.0
    for b in range(n_bins):
        lo = b * bin_width
        hi = lo + bin_width
        in_bin = [
            (c, y) for c, y in pairs
            if lo <= c < hi or (b == n_bins - 1 and c == 1.0)
        ]
        if not in_bin:
            continue
        acc_b = sum(y for _, y in in_bin) / len(in_bin)
        conf_b = sum(c for c, _ in in_bin) / len(in_bin)
        ece += (len(in_bin) / n_included) * abs(acc_b - conf_b)

    return ece


def reliability_data(
    confidences: ConfidenceScores,
    verdicts: VerdictsForProxy,
    mock_llm: bool,
    n_bins: int = _ECE_N_BINS,
) -> Optional[list[dict]]:
    """Per-bin reliability data for calibration plots."""
    if mock_llm:
        return None
    if len(confidences) != len(verdicts):
        raise ValueError("confidences and verdicts must have same length")

    pairs = [
        (confidences[i], _calibration_proxy_correct(verdicts[i]))
        for i in range(len(confidences))
        if _calibration_proxy_correct(verdicts[i]) is not None
    ]
    if len(pairs) < _CALIBRATION_MIN_CASES:
        return None

    bin_width = 1.0 / n_bins
    result = []
    for b in range(n_bins):
        lo = b * bin_width
        hi = lo + bin_width
        in_bin = [
            (c, y) for c, y in pairs
            if lo <= c < hi or (b == n_bins - 1 and c == 1.0)
        ]
        if not in_bin:
            continue
        result.append({
            "bin_lo": round(lo, 4),
            "bin_hi": round(hi, 4),
            "mean_confidence": sum(c for c, _ in in_bin) / len(in_bin),
            "fraction_correct": sum(y for _, y in in_bin) / len(in_bin),
            "n_cases": len(in_bin),
        })
    return result


# ---------------------------------------------------------------------------
# F. Performance metrics
# ---------------------------------------------------------------------------


def latency_percentiles(
    latencies: LatencySeconds,
    warmup_cases: int = 0,
) -> dict[str, Optional[float]]:
    """p50, p95, p99 latency in seconds."""
    measured = sorted(latencies[warmup_cases:])
    n = len(measured)

    def pct(p: float) -> Optional[float]:
        if n == 0:
            return None
        idx = int(math.ceil(p / 100.0 * n)) - 1
        idx = max(0, min(idx, n - 1))
        return measured[idx]

    return {"p50": pct(50), "p95": pct(95), "p99": pct(99), "n": n}


def throughput(n_completed: int, wall_clock_seconds: float) -> Optional[float]:
    """Completed cases per second."""
    if wall_clock_seconds <= 0:
        return None
    return n_completed / wall_clock_seconds


def error_rate(n_error: int, n_submitted: int) -> Optional[float]:
    """Fraction of submitted cases that errored."""
    if n_submitted == 0:
        return None
    return n_error / n_submitted


def rejection_rate(
    n_rejected_or_cancelled: int,
    n_submitted: int,
) -> Optional[float]:
    """Fraction of submitted cases that were rejected or deadline-cancelled."""
    if n_submitted == 0:
        return None
    return n_rejected_or_cancelled / n_submitted


__all__ = [
    "recall_at_k", "reciprocal_rank", "ndcg_at_k",
    "mean_recall_at_k", "mean_reciprocal_rank", "mean_ndcg_at_k",
    "conflict_confusion_matrix", "conflict_precision_recall_f1",
    "claim_confusion_matrix", "claim_per_class_prf", "claim_macro_f1",
    "decision_accuracy", "abstention_rate", "false_abstention_rate",
    "unsafe_answer_rate",
    "brier_score", "expected_calibration_error", "reliability_data",
    "latency_percentiles", "throughput", "error_rate", "rejection_rate",
]