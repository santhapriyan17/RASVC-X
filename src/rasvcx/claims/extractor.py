"""Deterministic claim extraction from evidence text (Module 7).

Responsibilities:
  1. Split evidence text into candidate sentences (deterministic, lightweight).
  2. Classify each sentence into a ClaimType using pattern-based rules.
  3. Detect safety-critical claims using conservative deterministic rules.
  4. Generate stable, deterministic claim IDs (SHA-256 based).
  5. Construct immutable Claim objects ready for normalization and registration.

Design constraints:
  - Pure functions over text + config: no I/O, no ML, no network calls.
  - All regexes compiled at module import time (never inside loops).
  - O(n) per evidence text where n = text length.
  - Deterministic: identical input always produces identical output, regardless
    of Python hash seed, process, or platform.
  - Conservative classification: false negatives are safer than false positives
    for safety-critical detection.  This is a deterministic baseline, not a
    semantic understanding engine.

Claim classification precedence (highest to lowest):
  DOSAGE > TEMPORAL > NUMERIC > RECOMMENDATION > FACTUAL

A claim is assigned the highest-priority type whose pattern matches.  If no
pattern matches, the claim is FACTUAL (the conservative default).  ClaimType.OTHER
is reserved for future use and is never assigned by the deterministic extractor.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rasvcx.schemas.claims import Claim, ClaimSource, ClaimType
from rasvcx.schemas.common import ChunkId, ClaimId, EvidenceItemId


# ═══════════════════════════════════════════════════════════════════════════
# SENTENCE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Abbreviation protection
# ---------------------------------------------------------------------------
# Common abbreviations whose trailing period must NOT trigger a sentence
# break.  The list is intentionally conservative (medical/scientific focus)
# and case-insensitive.

_ABBREVIATIONS: frozenset[str] = frozenset({
    # Titles
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    # Latin / academic
    "e.g", "i.e", "vs", "etc", "al", "approx", "dept",
    "fig", "ref", "vol", "no",
    # Medical abbreviations (NOT measurement units — mg, ml, kg, etc.
    # are unit suffixes, not abbreviations.  "5 mg." should split at
    # the period because it's a sentence terminal, not an abbreviation.)
    "pt", "pts", "rx", "dx", "tx", "hx", "fx",
    "inc", "corp", "ltd", "org",
    # Geographic / organizational
    "st", "ave", "blvd", "u.s", "u.k",
})

# Pre-build a set of lowercased abbreviations for O(1) lookup.  For
# multi-part abbreviations like "e.g" we also store the dotted form
# "e.g." so that "e.g. this" is correctly protected.
_ABBREV_LOOKUP: frozenset[str] = frozenset(
    a.lower() for a in _ABBREVIATIONS
) | frozenset(
    f"{a.lower()}." for a in _ABBREVIATIONS if "." not in a
)


def _is_abbreviation(token: str) -> bool:
    """Check if *token* (with its trailing period) is a known abbreviation.

    Args:
        token: The word immediately preceding a period, e.g. "Dr" from "Dr.".

    Returns:
        True if the token (case-insensitive) is a known abbreviation.
    """
    lower = token.lower().rstrip(".")
    return lower in _ABBREV_LOOKUP or f"{lower}." in _ABBREV_LOOKUP


# ---------------------------------------------------------------------------
# Decimal / numeric protection
# ---------------------------------------------------------------------------

# A period preceded by a digit and followed by a digit is a decimal point,
# not a sentence boundary.  Precompiled for efficiency.
_DECIMAL_DOT_RE = re.compile(r"\d\.\d")


# ---------------------------------------------------------------------------
# Sentence-boundary detection
# ---------------------------------------------------------------------------

# Potential sentence boundaries: a sentence-terminal mark (. ! ?) followed
# by whitespace and then an uppercase letter or digit.  This is deliberately
# conservative — it may under-split rather than over-split, which is the
# safer direction for claim extraction (preserving context > splitting
# aggressively).
_SENT_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])"     # lookbehind: sentence terminal
    r"\s+"             # at least one whitespace character
    r"(?=[A-Z0-9\"'])" # lookahead: next sentence starts with upper/digit/quote
)


def split_sentences(text: str) -> list[str]:
    """Split evidence text into candidate sentences.

    Strategy:
      1. Normalize internal whitespace (collapse runs, strip).
      2. Identify potential sentence boundaries using ``_SENT_BOUNDARY_RE``.
      3. For each candidate boundary, verify it is NOT:
         a. Inside a decimal number (e.g. "5.5 mg").
         b. After a known abbreviation (e.g. "Dr. Smith").
      4. Only split at verified boundaries.

    If no valid boundary is found, the entire (whitespace-normalized) text
    is returned as a single sentence.

    Empty or whitespace-only input produces an empty list.

    Args:
        text: Raw evidence text.

    Returns:
        List of sentence strings, each stripped of leading/trailing
        whitespace.  Order is preserved from the original text.
    """
    if not text or not text.strip():
        return []

    # Normalize whitespace: collapse all runs (including newlines/tabs) to
    # single spaces, then strip.
    normalized = re.sub(r"\s+", " ", text.strip())

    if not normalized:
        return []

    # Find all potential boundary positions
    sentences: list[str] = []
    last_end = 0

    for match in _SENT_BOUNDARY_RE.finditer(normalized):
        boundary_start = match.start()  # position right after the terminal mark

        # The terminal mark is at boundary_start - 1 (the character before
        # the whitespace).  We need the text before the boundary to check
        # for abbreviations and decimals.
        pre_boundary = normalized[:boundary_start]

        # Check 1: Is this a decimal point?  Look for digit.digit pattern
        # spanning the boundary position.
        # The period is at boundary_start - 1 in the normalized string.
        period_pos = boundary_start - 1
        if period_pos >= 0 and normalized[period_pos] == ".":
            # Check if this period is part of a decimal number
            if period_pos > 0 and normalized[period_pos - 1].isdigit():
                # Look ahead past the whitespace: if next non-space char is
                # a digit, this IS a decimal... but the regex already requires
                # whitespace between, so "5.5" wouldn't match the boundary
                # regex.  However, we still protect abbreviation-like cases.
                pass  # Not a decimal boundary (whitespace present), proceed

            # Check 2: Is this after an abbreviation?
            # Extract the word before the period.
            word_before = _extract_word_before_period(pre_boundary)
            if word_before and _is_abbreviation(word_before):
                continue  # Skip this boundary — it's an abbreviation

        sentences.append(normalized[last_end:boundary_start].strip())
        last_end = match.end()

    # Append the remaining text
    remainder = normalized[last_end:].strip()
    if remainder:
        sentences.append(remainder)

    # Filter out empty strings (defensive)
    return [s for s in sentences if s]


def _extract_word_before_period(text: str) -> str:
    """Extract the word immediately before the trailing period in *text*.

    Example: "prescribed by Dr." → "Dr"
    """
    if not text or text[-1] != ".":
        return ""

    # Walk backwards from the period to find the word
    i = len(text) - 2  # skip the period
    while i >= 0 and text[i] == ".":
        # Handle multi-dot abbreviations like "U.S."
        i -= 1

    end = i + 1
    while i >= 0 and (text[i].isalpha() or text[i] == "."):
        i -= 1

    word = text[i + 1:end]
    return word


# ═══════════════════════════════════════════════════════════════════════════
# CLAIM CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# DOSAGE patterns
# ---------------------------------------------------------------------------
# Matches a numeric value followed by a medical dosage unit.  The numeric
# value must be adjacent or whitespace-separated from the unit.
# Anchored with word boundaries to avoid matching inside larger words.

_DOSAGE_UNITS = (
    r"mg|g|kg|mcg|µg|μg|ml|ml|l|iu|units?|"
    r"mg/kg|mg/day|mg/kg/day|mcg/kg|"
    r"mg/ml|g/l|meq|mmol|"
    r"tablets?|capsules?|drops?|puffs?|"
    r"mg\s*/\s*kg|mg\s*/\s*day"
)

# A dosage expression: number + unit, with the unit being medically specific.
_DOSAGE_RE = re.compile(
    rf"\b(\d+(?:[.,]\d+)?)\s*(?:{_DOSAGE_UNITS})\b",
    re.IGNORECASE,
)

# Dosage action phrases (prescriptive dosing context, not merely mentioning
# a number).  These help distinguish "take 5 mg" from "5 mg was measured".
_DOSAGE_CONTEXT_RE = re.compile(
    r"\b(?:dose|doses|dosage|dosing|administer(?:ed)?|"
    r"prescrib(?:e|ed|ing)|titrat(?:e|ed|ing)|"
    r"maximum\s+dose|minimum\s+dose|"
    r"loading\s+dose|maintenance\s+dose|"
    r"dose\s+adjustment|adjusted?\s+dose)\b",
    re.IGNORECASE,
)

# Negative dosage context: mentioning dosage in a negated or absent sense.
# "No dosage data available" should NOT be classified as DOSAGE.
# Covers both "no dosage ..." and "dosage ... unavailable" patterns.
_DOSAGE_NEGATION_RE = re.compile(
    r"\b(?:no|not|without|absence\s+of|lack\s+of|unavailable|unknown)\s+"
    r"(?:dosage|dose|dosing)\b"
    r"|\b(?:dosage|dose|dosing)\s+(?:\w+\s+)*?"
    r"(?:unavailable|unknown|not\s+available|not\s+reported|"
    r"not\s+established|not\s+determined)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# TEMPORAL patterns
# ---------------------------------------------------------------------------

_TEMPORAL_RE = re.compile(
    r"\b(?:"
    # Explicit years (4-digit year not preceded by a decimal or unit context)
    r"(?:in|since|from|before|after|during|until|by)\s+\d{4}"
    r"|(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2})?"
    # Temporal keywords
    r"|(?:before|after|since|until|during|previously|"
    r"recently|historically|formerly|currently|"
    r"prior\s+to|subsequent(?:ly)?|"
    r"(?:year|month|week|day|hour)s?\s+(?:ago|later|earlier|before|after)|"
    r"(?:first|second|third|initial|final|baseline|follow[- ]?up)\s+"
    r"(?:visit|assessment|evaluation|examination|study|trial|phase)|"
    r"(?:at|over|within|for)\s+\d+\s+(?:year|month|week|day|hour)s?)"
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# NUMERIC patterns
# ---------------------------------------------------------------------------
# Matches quantitative statements (percentages, counts, measurements) that
# are NOT dosage.  Applied only after DOSAGE check has failed.

_NUMERIC_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*%"                       # percentages
    r"|\b(?:p\s*[<>=≤≥]\s*0?\.\d+)"                # p-values
    r"|\b(?:ci|confidence\s+interval)\s*[:=]?\s*"   # confidence intervals
    r"(?:\[?\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\]?)"
    r"|\b\d+(?:[.,]\d+)?\s*(?:patients?|subjects?|participants?|"
    r"cases?|individuals?|samples?|events?|deaths?|"
    r"trials?|studies|episodes?)\b"                  # counts of entities
    r"|\b(?:mean|median|average|range|ratio|rate|"
    r"sensitivity|specificity|prevalence|incidence|"
    r"odds\s+ratio|hazard\s+ratio|relative\s+risk|"
    r"absolute\s+risk|number\s+needed\s+to\s+treat)\b"
    r"\s*(?:[:=]|(?:was|were|is|of))\s*"
    r"\d+(?:[.,]\d+)?",                              # stat = value
    re.IGNORECASE,
)

# Standalone numeric: a number present in the text that doesn't match
# dosage or temporal but still represents quantitative information.
_STANDALONE_NUMERIC_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*%"   # percentage
    r"|\b\d{2,}\b",            # multi-digit number (excludes single digits)
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# RECOMMENDATION patterns
# ---------------------------------------------------------------------------

_RECOMMENDATION_RE = re.compile(
    r"\b(?:"
    r"(?:is|are)\s+(?:recommended|advised|suggested|indicated|contraindicated)"
    r"|(?:should|must|shall)\s+(?:be\s+)?(?:\w+)"
    r"|(?:recommend(?:s|ed|ing|ation)?|"
    r"advis(?:e|ed|es|ing)|"
    r"suggest(?:s|ed|ing)?|"
    r"contraindicated?|"
    r"(?:do|does)\s+not\s+(?:recommend|advise|use)|"
    r"not\s+recommended|"
    r"(?:should|must)\s+(?:not|never)|"
    r"(?:should|must)\s+avoid|"
    r"avoid(?:ed|ing|s)?(?:\s+(?:in|for|during|when))?|"
    r"prefer(?:red|ably)?|"
    r"consider(?:ed|ing)?\s+(?:as|for)|"
    r"(?:first|second)[- ]line\s+(?:treatment|therapy|option))"
    r")\b",
    re.IGNORECASE,
)


def classify_claim_type(text: str) -> ClaimType:
    """Classify a claim sentence into a ClaimType.

    Classification uses deterministic pattern matching with the following
    precedence (highest to lowest):

        DOSAGE > TEMPORAL > NUMERIC > RECOMMENDATION > FACTUAL

    ClaimType.OTHER is never assigned by the deterministic extractor.

    Args:
        text: The claim sentence text.

    Returns:
        One of the six ClaimType enum values.
    """
    if not text or not text.strip():
        return ClaimType.FACTUAL

    # --- DOSAGE (highest priority) ---
    # Check for negated dosage context first: "No dosage data available"
    # should NOT be classified as DOSAGE.
    if _DOSAGE_NEGATION_RE.search(text):
        pass  # Fall through to other classifiers
    elif _DOSAGE_RE.search(text) or _DOSAGE_CONTEXT_RE.search(text):
        # Verify this is an actual dosage, not just a mention of a unit
        # in a non-dosage context (e.g., "the sample weighed 5 kg" is
        # technically a measurement, but if the unit is dosage-specific
        # like mg, mcg, IU, we treat it as dosage).
        if _DOSAGE_RE.search(text):
            return ClaimType.DOSAGE
        # Context-only dosage (mentions dose/dosing without a numeric value)
        # is still DOSAGE if it's clearly prescriptive.
        if _DOSAGE_CONTEXT_RE.search(text):
            return ClaimType.DOSAGE

    # --- TEMPORAL ---
    if _TEMPORAL_RE.search(text):
        return ClaimType.TEMPORAL

    # --- NUMERIC ---
    if _NUMERIC_RE.search(text) or _STANDALONE_NUMERIC_RE.search(text):
        return ClaimType.NUMERIC

    # --- RECOMMENDATION ---
    if _RECOMMENDATION_RE.search(text):
        return ClaimType.RECOMMENDATION

    # --- FACTUAL (default) ---
    return ClaimType.FACTUAL


# ═══════════════════════════════════════════════════════════════════════════
# SAFETY-CRITICAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Keywords/phrases indicating high-risk medical content that warrants
# downstream safety-floor enforcement.
_SAFETY_CRITICAL_KEYWORDS_RE = re.compile(
    r"\b(?:"
    r"contraindicated?|contraindication|"
    r"overdose|over[- ]?dosage|"
    r"maximum\s+(?:dose|dosage|daily\s+dose)|"
    r"minimum\s+(?:dose|dosage)|"
    r"lethal\s+dose|toxic\s+dose|"
    r"(?:drug|medication)\s+interaction|"
    r"adverse\s+(?:event|effect|reaction)|"
    r"black\s+box\s+warning|"
    r"(?:stop|discontinue|withhold|withdraw)\s+(?:the\s+)?(?:medication|drug|treatment|therapy)|"
    r"(?:life|immediately)\s*[-\s]?\s*threatening|"
    r"anaphyla(?:xis|ctic)|"
    r"(?:do\s+not|never)\s+(?:administer|prescribe|give|use)|"
    r"(?:dose|dosage)\s+adjustment|"
    r"renal\s+(?:impairment|failure|insufficiency)|"
    r"hepatic\s+(?:impairment|failure|insufficiency)|"
    r"(?:severe|serious)\s+(?:adverse|side)\s+(?:effect|reaction)"
    r")\b",
    re.IGNORECASE,
)


def is_safety_critical(text: str, claim_type: ClaimType) -> bool:
    """Determine whether a claim should be flagged as safety-critical.

    A safety-critical flag routes the claim through the downstream safety
    floor (forced contextual validation + NLI eligibility).  It is a
    deterministic routing signal, NOT a medical truth judgment.

    Rules:
      1. DOSAGE claims with an actual numeric dosage value → always True.
      2. Any claim matching explicit high-risk keyword patterns → True.
      3. Everything else → False.

    The logic is intentionally conservative: it is better to under-flag
    (and let downstream validation decide) than to over-flag and cause
    excessive safety-floor invocations that degrade throughput.

    Args:
        text: The claim sentence text.
        claim_type: The ClaimType assigned by ``classify_claim_type()``.

    Returns:
        True if the claim warrants safety-critical treatment.
    """
    if not text:
        return False

    # Rule 1: DOSAGE claims with actual numeric dosage values
    if claim_type == ClaimType.DOSAGE and _DOSAGE_RE.search(text):
        return True

    # Rule 2: Explicit high-risk keywords in any claim type
    if _SAFETY_CRITICAL_KEYWORDS_RE.search(text):
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# CLAIM ID GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def generate_claim_id(
    item_id: EvidenceItemId,
    claim_index: int,
    claim_text: str,
) -> ClaimId:
    """Generate a deterministic, SHA-256-based claim ID.

    The identity of a claim is determined by:
      - The evidence item it was extracted from (item_id).
      - Its position within that item's extracted sentences (claim_index).
      - The exact text of the claim (claim_text).

    This ensures:
      - Same input → same ID (deterministic, reproducible).
      - Different item or position → different ID (collision-resistant).
      - Independent of Python's ``hash()`` seed randomization.
      - Stable across processes and platforms.

    Args:
        item_id: The EvidenceItemId of the source evidence item.
        claim_index: 0-based index of this claim within the item's sentences.
        claim_text: The exact claim text (pre-normalization).

    Returns:
        A ClaimId of the form ``claim_<hex_digest_prefix>``.
        Uses a 16-character hex prefix (64 bits) which provides sufficient
        collision resistance for practical repository usage.
    """
    identity = f"{item_id}:{claim_index}:{claim_text}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return ClaimId(f"claim_{digest}")


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACTION CONFIG + EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Configuration for the deterministic claim extractor.

    Attributes:
        min_claim_length: Minimum character length for a sentence to be
            treated as a claim.  Very short fragments (< 10 characters)
            are unlikely to be independently verifiable statements and
            are skipped to reduce noise.
    """

    min_claim_length: int = 10

    def __post_init__(self) -> None:
        if self.min_claim_length < 0:
            raise ValueError(
                f"min_claim_length must be non-negative, "
                f"got {self.min_claim_length}"
            )


DEFAULT_EXTRACTION_CONFIG = ExtractionConfig()


class ClaimExtractor:
    """Deterministic claim extractor from evidence text.

    Extracts sentence-level claims from ``EvidenceItem.text``, classifies
    them, detects safety-critical content, normalizes text, and generates
    stable claim IDs.  No LLM, no network calls, no global mutable state.

    Dependency-injectable via ``ExtractionConfig``.

    Design decision — sentence-level atomicity:
      Claims are extracted at the sentence level.  Sub-sentence clause
      splitting is intentionally NOT performed because:
        1. Deterministic clause splitting without an NLP parser risks
           destroying numeric/dosage context (e.g. splitting "Take 5 mg
           daily for 7 days" would separate the dosage from its duration).
        2. Sentence-level claims preserve the original evidence wording,
           making them auditable and traceable.
        3. Sub-sentence decomposition can be added as a future enhancement
           (a separate abstraction layer) without changing M7's contract.
    """

    def __init__(
        self, config: ExtractionConfig = DEFAULT_EXTRACTION_CONFIG
    ) -> None:
        if not isinstance(config, ExtractionConfig):
            raise TypeError(
                f"ClaimExtractor config must be ExtractionConfig, "
                f"got {type(config).__name__!r}"
            )
        self._config = config

    @property
    def config(self) -> ExtractionConfig:
        """Read-only view of the active extraction config."""
        return self._config

    def extract(
        self,
        item_id: EvidenceItemId,
        chunk_id: ChunkId,
        text: str,
    ) -> list[Claim]:
        """Extract claims from a single evidence item's text.

        Args:
            item_id: The EvidenceItemId of the source item.
            chunk_id: The ChunkId of the source item (becomes
                ``origin_chunk_id`` on each extracted Claim).
            text: The evidence text to segment and classify.

        Returns:
            Ordered list of Claim objects.  Order corresponds to sentence
            order in the original text.  May be empty if the text yields
            no valid claims.
        """
        from rasvcx.claims.claim_normalizer import normalize_claim_text

        sentences = split_sentences(text)
        claims: list[Claim] = []

        for index, sentence in enumerate(sentences):
            # Skip very short fragments
            if len(sentence.strip()) < self._config.min_claim_length:
                continue

            claim_type = classify_claim_type(sentence)
            safety = is_safety_critical(sentence, claim_type)
            claim_id = generate_claim_id(item_id, index, sentence)
            normalized = normalize_claim_text(sentence)

            # Skip if normalization produces empty text
            if not normalized:
                continue

            claim = Claim(
                claim_id=claim_id,
                text=sentence,
                source=ClaimSource.EVIDENCE_EXTRACTION,
                claim_type=claim_type,
                origin_chunk_id=chunk_id,
                normalized_text=normalized,
                is_safety_critical=safety,
            )
            claims.append(claim)

        return claims