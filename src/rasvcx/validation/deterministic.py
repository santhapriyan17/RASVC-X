"""Deterministic validation (Section 8).

Compares claim text pairwise using regex-based numeric/dosage/temporal
extraction and exact/approximate-equality rules. No LLM, no network, no
model initialization. Reproducible, explainable, low latency.

This module never mutates the EvidenceBundle; callers are responsible for
recording the returned ValidationResult (see validation/verified_context.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rasvcx.schemas.claims import Claim, ClaimType
from rasvcx.schemas.common import CandidateId, EvidenceItemId
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.validation import ValidationLabel, ValidationResult, ValidationStage
from rasvcx.validation.validation_types import (
    CandidateClaimPair,
    DeterministicConfig,
    NumericExtraction,
)

# ---------------------------------------------------------------------------
# Precompiled patterns (module-level; never compiled inside a loop)
# ---------------------------------------------------------------------------

# Number optionally preceded by a sign, optionally with a decimal part, and
# optionally followed by a unit token or "%". Ranges ("5-10 mg") are matched
# as two separate numbers sharing a unit; the range-hyphen itself is not
# consumed by this pattern so both endpoints are extracted independently.
_NUMBER_RE = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>%|mcg|mg|ml|g|kg|iu|units?|mmol|mol)?\b",
    re.IGNORECASE,
)

# A 4-digit year, used as a light-weight temporal signal distinct from full
# date parsing (that lives in provenance/context_extractor.py and operates
# on Provenance.date, not on claim text).
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on",
        "for", "and", "or", "with", "by", "at", "as", "be", "this", "that",
        "it", "its", "may", "should", "can", "not", "than", "per",
    }
)

# Unit -> (dimension, multiplier to canonical base unit within that dimension)
_UNIT_TO_BASE: dict[str, tuple[str, float]] = {
    "mcg": ("mass", 1e-6),
    "mg": ("mass", 1e-3),
    "g": ("mass", 1.0),
    "kg": ("mass", 1e3),
    "ml": ("volume", 1e-3),
    "l": ("volume", 1.0),
    "%": ("percent", 1.0),
    "iu": ("iu", 1.0),
    "unit": ("iu", 1.0),
    "units": ("iu", 1.0),
    "mmol": ("mmol", 1.0),
    "mol": ("mmol", 1e3),
}


def significant_tokens(text: str) -> frozenset[str]:
    """Lowercase, tokenize, and drop stopwords/short tokens.

    Pure function; used both for candidate generation (subject bucketing)
    and here (deciding whether two claims are "about the same thing"
    before attempting numeric comparison).
    """
    return frozenset(
        tok for tok in _TOKEN_RE.findall(text.lower())
        if tok not in _STOPWORDS and len(tok) > 2
    )


def extract_numbers(text: str) -> list[NumericExtraction]:
    """Extract all numeric tokens (with adjacent unit, if any) from text.

    Returns extractions in left-to-right order. Never raises on malformed
    input; text without numbers yields an empty list.
    """
    results: list[NumericExtraction] = []
    for match in _NUMBER_RE.finditer(text):
        raw_unit = match.group("unit")
        unit = raw_unit.lower() if raw_unit else None
        # Normalize "unit"/"units" to a single canonical spelling so the
        # comparison table lookup below is uniform.
        if unit in ("unit", "units"):
            unit = "iu"
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        results.append(
            NumericExtraction(value=value, unit=unit, span=match.span())
        )
    return results


def extract_years(text: str) -> list[int]:
    """Extract 4-digit years mentioned directly in claim text (lightweight,
    distinct from Provenance.date parsing in provenance/context_extractor.py).
    """
    return [int(m.group(0)) for m in _YEAR_RE.finditer(text)]


@dataclass(frozen=True, slots=True)
class _NumericComparison:
    label: ValidationLabel
    confidence: float
    reason: str


def _compare_numeric_pair(
    numbers_a: list[NumericExtraction],
    numbers_b: list[NumericExtraction],
    config: DeterministicConfig,
) -> _NumericComparison | None:
    """Compare the numeric content of two claim texts.

    Returns None when the two claims do not carry comparable numeric
    content (e.g. neither has numbers, or the only numbers present have
    incompatible/unknown units) -- callers must not force a verdict in
    that case; absence of comparable numbers is not evidence of anything.
    """
    if not numbers_a or not numbers_b:
        return None

    best: _NumericComparison | None = None

    for na in numbers_a:
        for nb in numbers_b:
            comparison = _compare_two_numbers(na, nb, config)
            if comparison is None:
                continue
            # Prefer the most decisive (highest-confidence) comparison
            # across all number pairs; a single clear contradiction or
            # match is more informative than an average over unrelated
            # numbers appearing in the same sentence.
            if best is None or comparison.confidence > best.confidence:
                best = comparison

    return best


def _compare_two_numbers(
    a: NumericExtraction, b: NumericExtraction, config: DeterministicConfig
) -> _NumericComparison | None:
    if a.unit is None and b.unit is None:
        # Two bare numbers with no unit context: comparing them would risk
        # treating unrelated quantities (e.g. a year vs. a dosage count) as
        # if they were the same measurement. Too ambiguous to compare.
        return None

    if a.unit is not None and b.unit is not None and a.unit != b.unit:
        dim_a = _UNIT_TO_BASE.get(a.unit)
        dim_b = _UNIT_TO_BASE.get(b.unit)
        if dim_a is None or dim_b is None or dim_a[0] != dim_b[0]:
            # Different, non-convertible units (e.g. "mg" vs "%"): the
            # numbers are not comparable. Do not classify as contradiction.
            return None
        # Same dimension, different unit spelling (e.g. mg vs g): convert
        # to a shared base unit before comparing.
        value_a = a.value * dim_a[1]
        value_b = b.value * dim_b[1]
        return _numeric_equality_verdict(value_a, value_b, config, note=f"{a.unit}->base vs {b.unit}->base")

    # Same unit (or both bare-but-one-sided None handled above as
    # incomparable), direct comparison.
    return _numeric_equality_verdict(a.value, b.value, config, note=f"unit={a.unit or b.unit}")


def _numeric_equality_verdict(
    value_a: float, value_b: float, config: DeterministicConfig, note: str
) -> _NumericComparison:
    if value_a == value_b:
        return _NumericComparison(
            label=ValidationLabel.SUPPORTED,
            confidence=0.98,
            reason=f"Exact numeric match ({value_a!r}, {note})",
        )

    tolerance = config.numeric_relative_tolerance * max(abs(value_a), abs(value_b), 1e-9)
    if abs(value_a - value_b) <= tolerance:
        return _NumericComparison(
            label=ValidationLabel.SUPPORTED,
            confidence=0.85,
            reason=f"Numeric values approximately equal ({value_a!r} vs {value_b!r}, {note})",
        )

    return _NumericComparison(
        label=ValidationLabel.CONTRADICTION,
        confidence=0.9,
        reason=f"Numeric values differ beyond tolerance ({value_a!r} vs {value_b!r}, {note})",
    )


def _compare_year_pair(text_a: str, text_b: str) -> _NumericComparison | None:
    years_a = extract_years(text_a)
    years_b = extract_years(text_b)
    if not years_a or not years_b:
        return None
    # Conservative: only decisive when each side mentions exactly one year;
    # multiple years on either side make "the" year being compared
    # ambiguous, so we abstain rather than guess which one is meant.
    if len(years_a) != 1 or len(years_b) != 1:
        return None
    if years_a[0] == years_b[0]:
        return _NumericComparison(
            label=ValidationLabel.SUPPORTED,
            confidence=0.6,
            reason=f"Same year mentioned ({years_a[0]})",
        )
    return _NumericComparison(
        label=ValidationLabel.CONTRADICTION,
        confidence=0.55,
        reason=f"Different years mentioned ({years_a[0]} vs {years_b[0]})",
    )


class DeterministicValidator:
    """Deterministic-first validation of a CandidatePair's linked claims.

    Given a CandidatePair (two EvidenceItems flagged as worth checking),
    compares the numeric/dosage/temporal content of their linked claims and
    produces a single ValidationResult for that candidate. Never invokes an
    NLI model.
    """

    def __init__(self, config: DeterministicConfig | None = None) -> None:
        self._config = config or DeterministicConfig()

    def validate(
        self,
        bundle: EvidenceBundle,
        candidate_id: CandidateId,
        item_id_a: EvidenceItemId,
        item_id_b: EvidenceItemId,
    ) -> ValidationResult:
        """Validate one candidate pair deterministically.

        Never raises for well-formed bundles; evidence items with no
        linked claims, or claims with no comparable numeric/temporal
        content, produce an UNCERTAIN result (absence of comparable
        deterministic signal, not evidence of contradiction).
        """
        item_a = bundle.evidence_items[item_id_a]
        item_b = bundle.evidence_items[item_id_b]

        claims_a = [bundle.claims.get(cid) for cid in sorted(item_a.extracted_claim_ids)]
        claims_b = [bundle.claims.get(cid) for cid in sorted(item_b.extracted_claim_ids)]

        pairs = self._pair_related_claims(claims_a, claims_b, item_id_a, item_id_b)

        if not pairs:
            return ValidationResult(
                candidate_id=candidate_id,
                stage=ValidationStage.DETERMINISTIC,
                label=ValidationLabel.UNCERTAIN,
                confidence=0.0,
                rationale="No claim pair shares enough subject overlap for deterministic comparison",
            )

        best_result: ValidationResult | None = None
        for pair, claim_a, claim_b in pairs[: self._config.max_claim_pairs_per_candidate]:
            candidate_result = self._compare_claim_pair(candidate_id, claim_a, claim_b)
            if candidate_result is None:
                continue
            if best_result is None or candidate_result.confidence > best_result.confidence:
                best_result = candidate_result

        if best_result is None:
            return ValidationResult(
                candidate_id=candidate_id,
                stage=ValidationStage.DETERMINISTIC,
                label=ValidationLabel.UNCERTAIN,
                confidence=0.0,
                rationale="Claims share subject overlap but no comparable numeric/temporal content",
            )

        return best_result

    # -- internal helpers -----------------------------------------------

    def _pair_related_claims(
        self,
        claims_a: list[Claim],
        claims_b: list[Claim],
        item_id_a: EvidenceItemId,
        item_id_b: EvidenceItemId,
    ) -> list[tuple[CandidateClaimPair, Claim, Claim]]:
        """Pair claims from each side that plausibly discuss the same
        subject, ranked by shared-token overlap (most related first).
        """
        eligible_types = {
            ClaimType.NUMERIC,
            ClaimType.DOSAGE,
            ClaimType.TEMPORAL,
        }
        pairs: list[tuple[CandidateClaimPair, Claim, Claim]] = []

        for claim_a in claims_a:
            if claim_a.claim_type not in eligible_types:
                continue
            tokens_a = significant_tokens(claim_a.normalized_text or claim_a.text)
            for claim_b in claims_b:
                if claim_b.claim_type not in eligible_types:
                    continue
                tokens_b = significant_tokens(claim_b.normalized_text or claim_b.text)
                shared = len(tokens_a & tokens_b)
                if shared < self._config.min_shared_tokens_for_comparison:
                    continue
                pairs.append(
                    (
                        CandidateClaimPair(
                            claim_id_a=claim_a.claim_id,
                            claim_id_b=claim_b.claim_id,
                            item_id_a=item_id_a,
                            item_id_b=item_id_b,
                            shared_token_count=shared,
                        ),
                        claim_a,
                        claim_b,
                    )
                )

        pairs.sort(key=lambda p: p[0].shared_token_count, reverse=True)
        return pairs

    def _compare_claim_pair(
        self, candidate_id: CandidateId, claim_a: Claim, claim_b: Claim
    ) -> ValidationResult | None:
        text_a = claim_a.normalized_text or claim_a.text
        text_b = claim_b.normalized_text or claim_b.text

        numbers_a = extract_numbers(text_a)
        numbers_b = extract_numbers(text_b)

        numeric_comparison = _compare_numeric_pair(numbers_a, numbers_b, self._config)
        if numeric_comparison is not None:
            return ValidationResult(
                candidate_id=candidate_id,
                stage=ValidationStage.DETERMINISTIC,
                label=numeric_comparison.label,
                confidence=numeric_comparison.confidence,
                rationale=(
                    f"{numeric_comparison.reason} "
                    f"[claim_a={claim_a.claim_id!r} claim_b={claim_b.claim_id!r}]"
                ),
            )

        year_comparison = _compare_year_pair(text_a, text_b)
        if year_comparison is not None:
            return ValidationResult(
                candidate_id=candidate_id,
                stage=ValidationStage.DETERMINISTIC,
                label=year_comparison.label,
                confidence=year_comparison.confidence,
                rationale=(
                    f"{year_comparison.reason} "
                    f"[claim_a={claim_a.claim_id!r} claim_b={claim_b.claim_id!r}]"
                ),
            )

        return None