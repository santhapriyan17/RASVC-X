# src/rasvcx/provenance/context_extractor.py
"""Deterministic provenance context extraction and comparison (Module 6).

Responsibilities:
  1. Parse temporal metadata from Provenance.date strings into a structured
     representation (year, optional month/day, whether the parse succeeded).
  2. Compute temporal relevance of evidence relative to a reference date
     and a query's time-sensitivity level.
  3. Compare population, jurisdiction, and dosage_context fields between
     evidence items, or between evidence and query context.
  4. Detect provenance-level conflicts (date disagreements, population/
     jurisdiction/dosage mismatches) across a pair of evidence items.

Design constraints:
  - Pure functions: no I/O, no ML, no retrieval calls.
  - UNKNOWN is never fabricated into a concrete value.
  - UNKNOWN != mismatch: a field being UNKNOWN means "insufficient
    information to rule out conflict," not "definitely different."
  - Temporal age alone is not treated as invalidity: old evidence may still
    be relevant; stale determination is query/domain dependent.
  - All outputs are explicitly structured (named fields), never a single
    opaque float.
  - Standard library only: datetime, re, dataclasses, enum, typing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Mapping

from rasvcx.schemas.common import UNKNOWN, _UnknownType
from rasvcx.schemas.evidence import EvidenceItem, Provenance
from rasvcx.schemas.query import QueryRequest


# ---------------------------------------------------------------------------
# Temporal parsing
# ---------------------------------------------------------------------------


class DateParseStatus(str, Enum):
    """Outcome of attempting to parse a date string from provenance metadata."""

    PARSED_FULL = "parsed_full"         # Year, month, and day successfully parsed
    PARSED_YEAR_MONTH = "parsed_year_month"  # Year and month only
    PARSED_YEAR_ONLY = "parsed_year_only"    # Year only
    UNPARSEABLE = "unparseable"         # String present but could not be parsed
    UNKNOWN = "unknown"                 # Source field was the UNKNOWN sentinel


@dataclass(frozen=True, slots=True)
class ParsedDate:
    """Structured result of parsing a Provenance.date field.

    Attributes:
        status:        How much temporal precision was recovered.
        year:          Calendar year, or None if UNKNOWN/unparseable.
        month:         Calendar month (1–12), or None if unavailable.
        day:           Calendar day (1–31), or None if unavailable.
        as_date:       A ``datetime.date`` object when at least year+month+day
                       are known; None otherwise. Used for arithmetic comparison.
        original:      The raw string value from provenance, for audit.
                       None when the source field was the UNKNOWN sentinel.
    """

    status: DateParseStatus
    year: int | None
    month: int | None
    day: int | None
    as_date: date | None
    original: str | None

    def __post_init__(self) -> None:
        if self.year is not None and not 1000 <= self.year <= 9999:
            raise ValueError(
                f"ParsedDate.year must be in [1000, 9999], got {self.year}"
            )
        if self.month is not None and not 1 <= self.month <= 12:
            raise ValueError(
                f"ParsedDate.month must be in [1, 12], got {self.month}"
            )
        if self.day is not None and not 1 <= self.day <= 31:
            raise ValueError(
                f"ParsedDate.day must be in [1, 31], got {self.day}"
            )


# Ordered list of (regex_pattern, strptime_format) pairs tried in sequence.
# More precise formats are attempted before less precise ones.
_DATE_FORMAT_CANDIDATES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "%Y/%m/%d"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%d/%m/%Y"),
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "%d-%m-%Y"),
    (
        re.compile(
            r"^(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\s+(\d{4})$",
            re.IGNORECASE,
        ),
        "%d %b %Y",
    ),
    (
        re.compile(
            r"^(January|February|March|April|May|June|July|August|September|"
            r"October|November|December)\s+(\d{1,2}),?\s+(\d{4})$",
            re.IGNORECASE,
        ),
        "%B %d %Y",
    ),
]

_YEAR_MONTH_RE = re.compile(r"^(\d{4})[/-](\d{2})$")
_YEAR_ONLY_RE = re.compile(r"^(\d{4})$")


def parse_date(raw: "str | object") -> ParsedDate:
    """Parse a Provenance.date field into a structured ParsedDate.

    Args:
        raw: The raw Provenance.date value. May be the UNKNOWN sentinel,
             a string, or (defensively) any other type which is treated as
             unparseable.

    Returns:
        ParsedDate with the best available precision. Never raises.
    """
    # UNKNOWN sentinel: explicit, conservative path.
    if isinstance(raw, _UnknownType):
        return ParsedDate(
            status=DateParseStatus.UNKNOWN,
            year=None,
            month=None,
            day=None,
            as_date=None,
            original=None,
        )

    if not isinstance(raw, str):
        # Defensive: unexpected type treated as unparseable, not UNKNOWN.
        return ParsedDate(
            status=DateParseStatus.UNPARSEABLE,
            year=None,
            month=None,
            day=None,
            as_date=None,
            original=str(raw) if raw is not None else "",
        )

    cleaned = raw.strip()
    if not cleaned:
        return ParsedDate(
            status=DateParseStatus.UNPARSEABLE,
            year=None,
            month=None,
            day=None,
            as_date=None,
            original=raw,
        )

    # Attempt full-date formats.
    for _pattern, fmt in _DATE_FORMAT_CANDIDATES:
        # Normalize multi-space to single space for month-name formats.
        normalized = " ".join(cleaned.split())
        try:
            parsed = datetime.strptime(normalized, fmt)
            return ParsedDate(
                status=DateParseStatus.PARSED_FULL,
                year=parsed.year,
                month=parsed.month,
                day=parsed.day,
                as_date=parsed.date(),
                original=raw,
            )
        except ValueError:
            continue

    # Attempt year-month.
    m = _YEAR_MONTH_RE.match(cleaned)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1000 <= y <= 9999 and 1 <= mo <= 12:
            return ParsedDate(
                status=DateParseStatus.PARSED_YEAR_MONTH,
                year=y,
                month=mo,
                day=None,
                as_date=None,
                original=raw,
            )

    # Attempt year-only.
    m = _YEAR_ONLY_RE.match(cleaned)
    if m:
        y = int(m.group(1))
        if 1000 <= y <= 9999:
            return ParsedDate(
                status=DateParseStatus.PARSED_YEAR_ONLY,
                year=y,
                month=None,
                day=None,
                as_date=None,
                original=raw,
            )

    return ParsedDate(
        status=DateParseStatus.UNPARSEABLE,
        year=None,
        month=None,
        day=None,
        as_date=None,
        original=raw,
    )


# ---------------------------------------------------------------------------
# Temporal relevance
# ---------------------------------------------------------------------------


class TimeSensitivity(str, Enum):
    """How strongly a query's answer depends on recency of evidence.

    TIMELESS:   The correct answer does not meaningfully change over time
                (e.g. anatomy, established mechanism of action).
    MODERATE:   The answer is unlikely to change drastically in a few years
                but may be updated by new evidence (e.g. standard dosing).
    HIGH:       The answer depends on current guidelines, regulations, or
                rapidly evolving evidence (e.g. drug approvals, outbreak data).
    """

    TIMELESS = "timeless"
    MODERATE = "moderate"
    HIGH = "high"


class TemporalRelevanceLabel(str, Enum):
    """Coarse temporal relevance label assigned to an evidence item.

    CURRENT:    Evidence is recent relative to the query's time sensitivity.
    AGING:      Evidence is older but still potentially applicable; warrants
                a downstream warning, not outright rejection.
    STALE:      Evidence is old enough that its applicability is doubtful
                for a time-sensitive query; downstream validation should treat
                this as a risk signal.
    FUTURE:     Evidence date is after the reference date (clock skew, typo,
                or legitimately future-dated material such as upcoming guidelines).
    UNCERTAIN:  Temporal precision is insufficient to classify (year-only,
                year-month, or unparseable date with non-UNKNOWN status).
    UNKNOWN:    The date field was the UNKNOWN sentinel; no temporal label
                can be assigned.
    """

    CURRENT = "current"
    AGING = "aging"
    STALE = "stale"
    FUTURE = "future"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TemporalRelevanceConfig:
    """Thresholds (in days) for temporal relevance classification.

    Thresholds are separated by time_sensitivity so that a TIMELESS query
    does not penalize older but still-valid foundational evidence.

    Attributes:
        current_threshold_days:  Evidence published within this many days of
                                  reference_date is CURRENT.
        aging_threshold_days:    Evidence older than current but within this
                                  many days is AGING. Beyond this → STALE.
        timeless_multiplier:     Multiplier applied to both thresholds for
                                  TIMELESS queries (relaxes staleness judgment).
        high_sensitivity_divisor: Divisor applied to both thresholds for HIGH
                                  sensitivity queries (tightens staleness).
    """

    current_threshold_days: int = 730       # ~2 years
    aging_threshold_days: int = 1825        # ~5 years
    timeless_multiplier: float = 3.0
    high_sensitivity_divisor: float = 2.0

    def __post_init__(self) -> None:
        if self.current_threshold_days <= 0:
            raise ValueError("current_threshold_days must be positive")
        if self.aging_threshold_days <= self.current_threshold_days:
            raise ValueError(
                "aging_threshold_days must be greater than current_threshold_days"
            )
        if self.timeless_multiplier <= 0:
            raise ValueError("timeless_multiplier must be positive")
        if self.high_sensitivity_divisor <= 0:
            raise ValueError("high_sensitivity_divisor must be positive")


DEFAULT_TEMPORAL_CONFIG = TemporalRelevanceConfig()


@dataclass(frozen=True, slots=True)
class TemporalSignal:
    """Auditable temporal relevance signal for one evidence item.

    Attributes:
        parsed_date:        Structured parse result for Provenance.date.
        label:              Coarse relevance label.
        age_days:           Approximate age in days from reference_date to
                            parsed_date.as_date. None when as_date is None.
        reference_date:     The date used as "now" for comparison.
        time_sensitivity:   The query's time-sensitivity tier.
        reason:             Human-readable justification for the label.
    """

    parsed_date: ParsedDate
    label: TemporalRelevanceLabel
    age_days: int | None
    reference_date: date
    time_sensitivity: TimeSensitivity
    reason: str


def compute_temporal_signal(
    provenance_date: "str | object",
    time_sensitivity: TimeSensitivity = TimeSensitivity.MODERATE,
    reference_date: date | None = None,
    config: TemporalRelevanceConfig = DEFAULT_TEMPORAL_CONFIG,
) -> TemporalSignal:
    """Compute the temporal relevance signal for one evidence date field.

    Args:
        provenance_date:   The raw Provenance.date value (string or UNKNOWN).
        time_sensitivity:  How time-sensitive the query is.
        reference_date:    Date to treat as "now". Defaults to today (UTC).
                           Supplied explicitly in tests for determinism.
        config:            Threshold configuration.

    Returns:
        TemporalSignal with parsed date, label, age, and reason. Never raises.
    """
    if reference_date is None:
        reference_date = datetime.now(timezone.utc).date()

    parsed = parse_date(provenance_date)

    # UNKNOWN date → cannot classify.
    if parsed.status == DateParseStatus.UNKNOWN:
        return TemporalSignal(
            parsed_date=parsed,
            label=TemporalRelevanceLabel.UNKNOWN,
            age_days=None,
            reference_date=reference_date,
            time_sensitivity=time_sensitivity,
            reason="Provenance.date is UNKNOWN; temporal relevance cannot be determined",
        )

    # Unparseable date → uncertain.
    if parsed.status == DateParseStatus.UNPARSEABLE:
        return TemporalSignal(
            parsed_date=parsed,
            label=TemporalRelevanceLabel.UNCERTAIN,
            age_days=None,
            reference_date=reference_date,
            time_sensitivity=time_sensitivity,
            reason=(
                f"Provenance.date {parsed.original!r} could not be parsed; "
                f"temporal relevance is uncertain"
            ),
        )

    # Year-only or year-month: insufficient precision for day-level arithmetic.
    if parsed.status in (DateParseStatus.PARSED_YEAR_ONLY,
                         DateParseStatus.PARSED_YEAR_MONTH):
        # Use conservative year-level estimate only.
        assert parsed.year is not None
        year_age = reference_date.year - parsed.year
        label = _classify_year_age(year_age, time_sensitivity, config)
        return TemporalSignal(
            parsed_date=parsed,
            label=label,
            age_days=None,   # day-level precision not available
            reference_date=reference_date,
            time_sensitivity=time_sensitivity,
            reason=(
                f"Provenance.date parsed to year-level precision only "
                f"(~{year_age} year(s) old); label {label.value!r} assigned "
                f"with reduced confidence"
            ),
        )

    # Full date: day-level arithmetic.
    assert parsed.as_date is not None
    delta = reference_date - parsed.as_date
    age_days = delta.days

    if age_days < 0:
        return TemporalSignal(
            parsed_date=parsed,
            label=TemporalRelevanceLabel.FUTURE,
            age_days=age_days,
            reference_date=reference_date,
            time_sensitivity=time_sensitivity,
            reason=(
                f"Provenance.date {parsed.as_date} is {abs(age_days)} day(s) "
                f"after reference_date {reference_date}; flagged as FUTURE "
                f"(possible clock skew, typo, or upcoming guideline)"
            ),
        )

    # Apply sensitivity adjustment to thresholds.
    current_thr, aging_thr = _adjusted_thresholds(time_sensitivity, config)
    label = _classify_age_days(age_days, current_thr, aging_thr)

    return TemporalSignal(
        parsed_date=parsed,
        label=label,
        age_days=age_days,
        reference_date=reference_date,
        time_sensitivity=time_sensitivity,
        reason=(
            f"Evidence is {age_days} day(s) old; sensitivity={time_sensitivity.value}; "
            f"thresholds current≤{current_thr}d, aging≤{aging_thr}d; "
            f"label={label.value!r}"
        ),
    )


def _adjusted_thresholds(
    sensitivity: TimeSensitivity,
    config: TemporalRelevanceConfig,
) -> tuple[int, int]:
    """Return (current_days, aging_days) adjusted for time sensitivity."""
    c = config.current_threshold_days
    a = config.aging_threshold_days
    if sensitivity == TimeSensitivity.TIMELESS:
        return int(c * config.timeless_multiplier), int(a * config.timeless_multiplier)
    if sensitivity == TimeSensitivity.HIGH:
        return (
            max(1, int(c / config.high_sensitivity_divisor)),
            max(2, int(a / config.high_sensitivity_divisor)),
        )
    return c, a   # MODERATE: use base thresholds


def _classify_age_days(
    age_days: int,
    current_threshold: int,
    aging_threshold: int,
) -> TemporalRelevanceLabel:
    if age_days <= current_threshold:
        return TemporalRelevanceLabel.CURRENT
    if age_days <= aging_threshold:
        return TemporalRelevanceLabel.AGING
    return TemporalRelevanceLabel.STALE


def _classify_year_age(
    year_age: int,
    sensitivity: TimeSensitivity,
    config: TemporalRelevanceConfig,
) -> TemporalRelevanceLabel:
    """Coarse year-level classification when day precision is unavailable."""
    current_thr, aging_thr = _adjusted_thresholds(sensitivity, config)
    # Convert thresholds from days to approximate years for comparison.
    current_years = current_thr / 365.25
    aging_years = aging_thr / 365.25
    if year_age < 0:
        return TemporalRelevanceLabel.FUTURE
    if year_age <= current_years:
        return TemporalRelevanceLabel.CURRENT
    if year_age <= aging_years:
        return TemporalRelevanceLabel.AGING
    return TemporalRelevanceLabel.STALE


# ---------------------------------------------------------------------------
# Applicability comparison (population, jurisdiction, dosage_context)
# ---------------------------------------------------------------------------


class ApplicabilityLabel(str, Enum):
    """Outcome of comparing a contextual provenance field between two items
    or between an evidence item and a query context.

    MATCH:    Both values are known and identical (or semantically equivalent
              by normalized comparison).
    MISMATCH: Both values are known and differ.
    UNKNOWN:  At least one value is the UNKNOWN sentinel; cannot rule out
              conflict.
    NOT_COMPARED: Comparison was not attempted (e.g. neither side provided
                  a value to compare against).
    """

    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"
    NOT_COMPARED = "not_compared"


@dataclass(frozen=True, slots=True)
class ApplicabilitySignal:
    """Auditable applicability comparison signal for one contextual field.

    Attributes:
        field_name:    Which provenance field was compared (e.g. "population").
        label:         Comparison outcome.
        value_a:       First value in the comparison (string or None for UNKNOWN).
        value_b:       Second value in the comparison (string or None for UNKNOWN).
        reason:        Human-readable justification.
    """

    field_name: str
    label: ApplicabilityLabel
    value_a: str | None   # None when that side was UNKNOWN sentinel
    value_b: str | None
    reason: str


def _normalize_context_field(value: "str | object") -> "str | _UnknownType":
    """Normalize a provenance context field for comparison.

    - UNKNOWN sentinel → returned as UNKNOWN (singleton).
    - Non-string → treated as UNKNOWN (conservative; no fabrication).
    - String: stripped, lowercased, internal whitespace collapsed.
    - Empty string after stripping → UNKNOWN (treat missing-as-empty as unknown).
    """
    if isinstance(value, _UnknownType):
        return UNKNOWN
    if not isinstance(value, str):
        return UNKNOWN
    normalized = " ".join(value.strip().lower().split())
    if not normalized:
        return UNKNOWN
    return normalized


def compare_context_field(
    field_name: str,
    value_a: "str | object",
    value_b: "str | object",
) -> ApplicabilitySignal:
    """Compare one provenance context field between two evidence items.

    UNKNOWN on either side → UNKNOWN (cannot rule out conflict).
    Both known and equal (after normalization) → MATCH.
    Both known and different → MISMATCH.

    Args:
        field_name: Name of the field being compared (for audit).
        value_a:    Value from the first evidence item.
        value_b:    Value from the second evidence item.

    Returns:
        ApplicabilitySignal. Never raises.
    """
    norm_a = _normalize_context_field(value_a)
    norm_b = _normalize_context_field(value_b)

    a_unknown = isinstance(norm_a, _UnknownType)
    b_unknown = isinstance(norm_b, _UnknownType)

    if a_unknown or b_unknown:
        sides = []
        if a_unknown:
            sides.append("value_a")
        if b_unknown:
            sides.append("value_b")
        return ApplicabilitySignal(
            field_name=field_name,
            label=ApplicabilityLabel.UNKNOWN,
            value_a=None if a_unknown else str(norm_a),
            value_b=None if b_unknown else str(norm_b),
            reason=(
                f"{field_name}: {' and '.join(sides)} is UNKNOWN; "
                f"cannot rule out conflict"
            ),
        )

    # Both known strings.
    a_str = str(norm_a)
    b_str = str(norm_b)
    if a_str == b_str:
        return ApplicabilitySignal(
            field_name=field_name,
            label=ApplicabilityLabel.MATCH,
            value_a=a_str,
            value_b=b_str,
            reason=f"{field_name}: values match ({a_str!r})",
        )

    return ApplicabilitySignal(
        field_name=field_name,
        label=ApplicabilityLabel.MISMATCH,
        value_a=a_str,
        value_b=b_str,
        reason=f"{field_name}: mismatch ({a_str!r} vs {b_str!r})",
    )


def compare_applicability(
    prov_a: Provenance,
    prov_b: Provenance,
) -> list[ApplicabilitySignal]:
    """Compare all three applicability fields between two Provenance objects.

    Returns one ApplicabilitySignal per field:
      - population
      - jurisdiction
      - dosage_context

    Order is stable (alphabetical by field name) for deterministic output.

    Args:
        prov_a: Provenance of the first evidence item.
        prov_b: Provenance of the second evidence item.

    Returns:
        List of three ApplicabilitySignals, one per contextual field.
    """
    return [
        compare_context_field("dosage_context", prov_a.dosage_context, prov_b.dosage_context),
        compare_context_field("jurisdiction", prov_a.jurisdiction, prov_b.jurisdiction),
        compare_context_field("population", prov_a.population, prov_b.population),
    ]


def compare_against_query_context(
    provenance: Provenance,
    query_context: dict[str, str],
) -> list[ApplicabilitySignal]:
    """Compare an evidence item's provenance against query-supplied context.

    Args:
        provenance:    Provenance of one evidence item.
        query_context: Dict with optional keys "population", "jurisdiction",
                       "dosage_context" from the query's metadata. Missing
                       keys are treated as NOT_COMPARED (the query did not
                       constrain that field).

    Returns:
        List of ApplicabilitySignals for the fields present in query_context.
        Fields absent from query_context produce a NOT_COMPARED signal.
    """
    results: list[ApplicabilitySignal] = []

    field_map: dict[str, "str | object"] = {
        "dosage_context": provenance.dosage_context,
        "jurisdiction": provenance.jurisdiction,
        "population": provenance.population,
    }

    for field_name in sorted(field_map.keys()):
        evidence_value = field_map[field_name]
        if field_name not in query_context:
            results.append(
                ApplicabilitySignal(
                    field_name=field_name,
                    label=ApplicabilityLabel.NOT_COMPARED,
                    value_a=None,
                    value_b=None,
                    reason=(
                        f"{field_name}: not present in query context; "
                        f"comparison not attempted"
                    ),
                )
            )
        else:
            results.append(
                compare_context_field(
                    field_name,
                    evidence_value,
                    query_context[field_name],
                )
            )

    return results


# ---------------------------------------------------------------------------
# Pairwise temporal conflict detection
# ---------------------------------------------------------------------------


class TemporalConflictLabel(str, Enum):
    """Outcome of comparing publication dates between two evidence items.

    NO_CONFLICT:         Both dates are known and within an acceptable gap.
    TEMPORAL_DIVERGENCE: Both dates are known and differ by more than the
                         configured threshold (may warrant downstream attention).
    UNKNOWN:             At least one date is UNKNOWN/unparseable; conflict
                         cannot be ruled out.
    NOT_COMPARABLE:      Dates were parsed but lack sufficient precision for
                         numeric comparison (e.g. year-only on one side).
    """

    NO_CONFLICT = "no_conflict"
    TEMPORAL_DIVERGENCE = "temporal_divergence"
    UNKNOWN = "unknown"
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True, slots=True)
class TemporalConflictSignal:
    """Auditable temporal conflict signal between two evidence items.

    Attributes:
        label:         Outcome of comparing the two dates.
        gap_days:      Absolute day difference between as_date values,
                       or None if comparison was not possible.
        date_a:        ParsedDate for the first item.
        date_b:        ParsedDate for the second item.
        reason:        Human-readable justification.
    """

    label: TemporalConflictLabel
    gap_days: int | None
    date_a: ParsedDate
    date_b: ParsedDate
    reason: str


def detect_temporal_conflict(
    prov_a: Provenance,
    prov_b: Provenance,
    divergence_threshold_days: int = 730,
) -> TemporalConflictSignal:
    """Detect whether two evidence items have a meaningful publication date gap.

    This is a provenance-level signal only. It does NOT determine whether
    the *content* of the two items conflicts; that belongs to M8 validation.

    Args:
        prov_a:                    Provenance of the first evidence item.
        prov_b:                    Provenance of the second evidence item.
        divergence_threshold_days: Gap in days above which dates are flagged
                                   as TEMPORAL_DIVERGENCE. Default ~2 years.

    Returns:
        TemporalConflictSignal. Never raises.
    """
    if divergence_threshold_days <= 0:
        raise ValueError("divergence_threshold_days must be positive")

    date_a = parse_date(prov_a.date)
    date_b = parse_date(prov_b.date)

    # Either date UNKNOWN or unparseable → cannot rule out conflict.
    a_usable = date_a.status == DateParseStatus.PARSED_FULL
    b_usable = date_b.status == DateParseStatus.PARSED_FULL

    if not a_usable or not b_usable:
        which = []
        if not a_usable:
            which.append(f"date_a ({date_a.status.value})")
        if not b_usable:
            which.append(f"date_b ({date_b.status.value})")
        # Distinguish UNKNOWN sentinel from merely low-precision/unparseable.
        has_unknown = (
            date_a.status == DateParseStatus.UNKNOWN
            or date_b.status == DateParseStatus.UNKNOWN
        )
        label = (
            TemporalConflictLabel.UNKNOWN
            if has_unknown
            else TemporalConflictLabel.NOT_COMPARABLE
        )
        return TemporalConflictSignal(
            label=label,
            gap_days=None,
            date_a=date_a,
            date_b=date_b,
            reason=(
                f"Temporal conflict cannot be assessed: "
                f"{'; '.join(which)} lacks full-date precision"
            ),
        )

    assert date_a.as_date is not None
    assert date_b.as_date is not None
    gap_days = abs((date_a.as_date - date_b.as_date).days)

    if gap_days > divergence_threshold_days:
        return TemporalConflictSignal(
            label=TemporalConflictLabel.TEMPORAL_DIVERGENCE,
            gap_days=gap_days,
            date_a=date_a,
            date_b=date_b,
            reason=(
                f"Publication dates differ by {gap_days} day(s) "
                f"(threshold={divergence_threshold_days}d): "
                f"{date_a.as_date} vs {date_b.as_date}"
            ),
        )

    return TemporalConflictSignal(
        label=TemporalConflictLabel.NO_CONFLICT,
        gap_days=gap_days,
        date_a=date_a,
        date_b=date_b,
        reason=(
            f"Publication dates within acceptable gap: "
            f"{gap_days} day(s) ≤ threshold {divergence_threshold_days}d"
        ),
    )


# ---------------------------------------------------------------------------
# ContextExtractor — thin orchestration adapter (analyzer.py collaborator)
# ---------------------------------------------------------------------------
#
# The analyzer's documented collaborator contract (see analyzer.py module
# docstring) expects:
#
#     ContextExtractor.extract(evidence_item, query) -> ContextCompatibilityResult
#
#     where ContextCompatibilityResult exposes:
#         .temporal, .jurisdiction, .population, .dosage_context
#
# Everything below is a stateless adapter over the pure functions already
# defined in this module (compute_temporal_signal, compare_against_query_
# context). No comparison/parsing logic is duplicated here, and this module
# does NOT import analyzer.py (avoids a circular import — analyzer.py is
# the one importing this module).
#
# Design notes:
#   - The four output fields use ApplicabilityLabel (MATCH / MISMATCH /
#     UNKNOWN), not a temporal-specific enum, so the analyzer's adapter
#     layer can treat all four fields uniformly.
#   - ApplicabilityLabel.NOT_COMPARED (query did not supply a constraint
#     for that field) is folded into UNKNOWN at this boundary. The absence
#     of a query-side constraint is not evidence of a verified match, so it
#     must not be reported as one.
#   - Temporal "compatibility" is derived from TemporalRelevanceLabel via an
#     explicit, documented mapping (see _TEMPORAL_LABEL_TO_APPLICABILITY):
#     CURRENT/AGING -> MATCH (AGING is "still potentially applicable" per
#     TemporalRelevanceLabel's own docstring, not a rejection), STALE/FUTURE
#     -> MISMATCH (risk signal), UNCERTAIN/UNKNOWN -> UNKNOWN (insufficient
#     precision is not the same as a known mismatch).
#   - time_sensitivity is resolved from QueryRequest.metadata (QueryRequest
#     has no dedicated field for it); an unrecognized or missing value
#     defaults to MODERATE. This is a query-side configuration default, not
#     a fabrication of evidence-side UNKNOWN data.


_TIME_SENSITIVITY_BY_METADATA_VALUE: dict[str, TimeSensitivity] = {
    "timeless": TimeSensitivity.TIMELESS,
    "moderate": TimeSensitivity.MODERATE,
    "high": TimeSensitivity.HIGH,
}

_TEMPORAL_LABEL_TO_APPLICABILITY: dict[TemporalRelevanceLabel, ApplicabilityLabel] = {
    TemporalRelevanceLabel.CURRENT: ApplicabilityLabel.MATCH,
    TemporalRelevanceLabel.AGING: ApplicabilityLabel.MATCH,
    TemporalRelevanceLabel.STALE: ApplicabilityLabel.MISMATCH,
    TemporalRelevanceLabel.FUTURE: ApplicabilityLabel.MISMATCH,
    TemporalRelevanceLabel.UNCERTAIN: ApplicabilityLabel.UNKNOWN,
    TemporalRelevanceLabel.UNKNOWN: ApplicabilityLabel.UNKNOWN,
}


def _resolve_time_sensitivity(query_metadata: Mapping[str, str]) -> TimeSensitivity:
    """Resolve TimeSensitivity from QueryRequest.metadata.

    Missing or unrecognized values default to MODERATE. This is a
    deliberate, documented query-side default — distinct from the
    evidence-side UNKNOWN sentinel, which this function never returns.
    """
    raw = query_metadata.get("time_sensitivity")
    if raw is None:
        return TimeSensitivity.MODERATE
    return _TIME_SENSITIVITY_BY_METADATA_VALUE.get(
        raw.strip().lower(), TimeSensitivity.MODERATE
    )


@dataclass(frozen=True, slots=True)
class ContextCompatibilityResult:
    """Per-item context-compatibility result consumed by ProvenanceAnalyzer.

    Attributes:
        temporal:       ApplicabilityLabel derived from temporal relevance
                        (see _TEMPORAL_LABEL_TO_APPLICABILITY mapping above).
        jurisdiction:   ApplicabilityLabel comparing evidence vs. query
                        context (NOT_COMPARED already folded into UNKNOWN).
        population:     Same, for the population field.
        dosage_context: Same, for the dosage_context field.
        temporal_signal:        Full underlying TemporalSignal, retained for
                                audit/debug. Not required by the analyzer's
                                minimal collaborator contract.
        applicability_signals:  Full underlying ApplicabilitySignal tuple
                                (dosage_context, jurisdiction, population, in
                                that order), retained for audit/debug.
    """

    temporal: ApplicabilityLabel
    jurisdiction: ApplicabilityLabel
    population: ApplicabilityLabel
    dosage_context: ApplicabilityLabel
    temporal_signal: TemporalSignal
    applicability_signals: tuple[ApplicabilitySignal, ...]


class ContextExtractor:
    """Stateless orchestration adapter fulfilling the analyzer's collaborator
    contract: ``.extract(evidence_item, query) -> ContextCompatibilityResult``.

    Delegates entirely to compute_temporal_signal() and
    compare_against_query_context(); duplicates no comparison/parsing logic.

    Complexity: O(1) per call (bounded work: one date parse, three field
    comparisons). The analyzer is responsible for the O(n) loop over items.
    """

    def __init__(
        self,
        temporal_config: TemporalRelevanceConfig = DEFAULT_TEMPORAL_CONFIG,
        reference_date: date | None = None,
    ) -> None:
        """
        Args:
            temporal_config: Threshold configuration, injected for
                determinism/testability. Defaults to DEFAULT_TEMPORAL_CONFIG.
            reference_date: Fixed "now" for deterministic testing. When
                None (production default), compute_temporal_signal() uses
                the real current UTC date at call time — so leaving this
                None means results are NOT identical across calendar days,
                which is expected/correct behavior for temporal relevance,
                not a violation of per-call determinism.
        """
        self._temporal_config = temporal_config
        self._reference_date = reference_date

    def extract(
        self,
        evidence_item: EvidenceItem,
        query: "QueryRequest",
    ) -> ContextCompatibilityResult:
        """Extract context-compatibility signals for one evidence item.

        Args:
            evidence_item: The evidence item to analyze. Not mutated.
            query: The active QueryRequest; `query.metadata` supplies both
                the optional "time_sensitivity" hint and the optional
                "population" / "jurisdiction" / "dosage_context" constraints.

        Returns:
            ContextCompatibilityResult. Never raises for well-formed inputs;
            malformed EvidenceItem/QueryRequest types are a caller contract
            violation and will surface as AttributeError, not be silently
            absorbed into UNKNOWN.
        """
        provenance = evidence_item.provenance
        time_sensitivity = _resolve_time_sensitivity(query.metadata)

        temporal_signal = compute_temporal_signal(
            provenance.date,
            time_sensitivity=time_sensitivity,
            reference_date=self._reference_date,
            config=self._temporal_config,
        )
        temporal_label = _TEMPORAL_LABEL_TO_APPLICABILITY[temporal_signal.label]

        applicability_signals = tuple(
            compare_against_query_context(provenance, query.metadata)
        )
        by_field = {signal.field_name: signal for signal in applicability_signals}

        def _resolved(field_name: str) -> ApplicabilityLabel:
            label = by_field[field_name].label
            if label == ApplicabilityLabel.NOT_COMPARED:
                return ApplicabilityLabel.UNKNOWN
            return label

        return ContextCompatibilityResult(
            temporal=temporal_label,
            jurisdiction=_resolved("jurisdiction"),
            population=_resolved("population"),
            dosage_context=_resolved("dosage_context"),
            temporal_signal=temporal_signal,
            applicability_signals=applicability_signals,
        )
