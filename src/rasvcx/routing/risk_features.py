"""Deterministic feature extraction for risk/complexity routing.

Every feature here is a pure function over the query text: no LLM calls, no
network I/O, no randomness. This keeps risk scoring fast, reproducible, and
auditable, and keeps it clearly separated from the safety floor (which is
owned by validation/selective_router.py, not by this module).

Risk weights/thresholds used to combine these scores are configurable and
owned by risk_router.py / config -- this module only produces the raw,
bounded [0, 1] signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rasvcx.schemas.query import RiskFeatureScores

_NUMERIC_PATTERN = re.compile(r"\d+(\.\d+)?")
_UNIT_PATTERN = re.compile(
    r"\b(mg|mcg|g|kg|ml|mL|l|L|iu|IU|mmol|mol|units?|tablets?|capsules?|"
    r"%|percent|hours?|hrs?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
_SAFETY_STRUCTURE_PATTERN = re.compile(
    r"\b(dose|dosage|dosing|overdose|contraindicat\w*|interaction|"
    r"maximum|minimum|max\b|min\b|threshold|limit|toxic\w*|lethal)\b",
    re.IGNORECASE,
)
_AMBIGUITY_MARKERS = re.compile(
    r"\b(it|this|that|they|these|those|the drug|the medication)\b",
    re.IGNORECASE,
)


def _bounded(value: float) -> float:
    """Clamp to [0, 1]; scores are bounded signals, never raw counts."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _density_score(pattern: re.Pattern[str], text: str, saturate_at: int) -> float:
    """Fraction of `saturate_at` matches found, capped at 1.0."""
    if not text:
        return 0.0
    match_count = len(pattern.findall(text))
    return _bounded(match_count / saturate_at)


@dataclass(frozen=True, slots=True)
class QueryComplexitySignals:
    """Intermediate signals used to derive query_complexity_score."""

    token_count: int
    clause_count: int
    question_mark_count: int


def _query_complexity_signals(text: str) -> QueryComplexitySignals:
    tokens = text.split()
    clauses = re.split(r"[,;]|\band\b|\bor\b", text, flags=re.IGNORECASE)
    clauses = [c for c in clauses if c.strip()]
    question_marks = text.count("?")
    return QueryComplexitySignals(
        token_count=len(tokens),
        clause_count=len(clauses),
        question_mark_count=question_marks,
    )


def compute_numeric_content_score(text: str) -> float:
    """Density of numeric tokens in the query."""
    return _density_score(_NUMERIC_PATTERN, text, saturate_at=3)


def compute_unit_sensitive_score(text: str) -> float:
    """Density of unit-sensitive tokens (dosage units, time units, etc.)."""
    return _density_score(_UNIT_PATTERN, text, saturate_at=2)


def compute_safety_critical_structure_score(text: str) -> float:
    """Density of explicit safety-critical structural markers."""
    return _density_score(_SAFETY_STRUCTURE_PATTERN, text, saturate_at=2)


def compute_query_complexity_score(text: str) -> float:
    """Combines token count, clause count, and question count into [0, 1]."""
    signals = _query_complexity_signals(text)
    token_score = _bounded(signals.token_count / 40)
    clause_score = _bounded((signals.clause_count - 1) / 3)
    question_score = _bounded((signals.question_mark_count - 1) / 2)
    return _bounded((token_score + clause_score + question_score) / 3)


def compute_ambiguity_score(text: str) -> float:
    """Density of unresolved-referent markers (pronouns without antecedents)."""
    return _density_score(_AMBIGUITY_MARKERS, text, saturate_at=2)


def extract_risk_features(text: str) -> RiskFeatureScores:
    """Compute the full RiskFeatureScores contract for one query.

    Pure function: identical input text always produces identical output,
    with no dependency on prior calls or external state.
    """
    return RiskFeatureScores(
        numeric_content_score=compute_numeric_content_score(text),
        unit_sensitive_score=compute_unit_sensitive_score(text),
        safety_critical_structure_score=compute_safety_critical_structure_score(text),
        query_complexity_score=compute_query_complexity_score(text),
        ambiguity_score=compute_ambiguity_score(text),
    )