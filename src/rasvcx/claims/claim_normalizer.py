"""Deterministic claim text normalization (Module 7).

Normalization is applied to extracted claim text to produce a canonical
form suitable for downstream matching, deduplication, and NLI premise/
hypothesis construction.

Design constraints:
  - Pure function: no I/O, no ML, no state.
  - Deterministic and idempotent: normalize(normalize(x)) == normalize(x).
  - Meaning-preserving: numbers, units, negation, and medically significant
    symbols are never altered.  Only representational noise (extra whitespace,
    inconsistent casing, trailing sentence punctuation) is removed.
  - Standard library only.
"""

from __future__ import annotations

import dataclasses
import re

from rasvcx.schemas.claims import Claim

# ---------------------------------------------------------------------------
# Precompiled patterns (module-level, compiled once)
# ---------------------------------------------------------------------------

# Collapse any run of whitespace (spaces, tabs, newlines, etc.) into a
# single ASCII space.  Applied after stripping leading/trailing whitespace.
_WHITESPACE_RUN_RE = re.compile(r"\s+")

# Safe trailing punctuation: only sentence-terminal marks that do not carry
# semantic meaning inside the claim.  Periods, exclamation marks, and
# question marks at the very end of the normalized string are removed.
# Colons, semicolons, commas, parentheses, hyphens, slashes, and percent
# signs are NOT removed because they may be part of dosage/numeric content.
_TRAILING_PUNCT_RE = re.compile(r"[.!?]+$")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_claim_text(text: str) -> str:
    """Normalize a single claim text string.

    Operations (in order):
      1. Strip leading/trailing whitespace.
      2. Collapse internal whitespace runs to a single space.
      3. Lowercase (English text normalization; safe because claim matching
         is case-insensitive by convention, and medically meaningful tokens
         like unit abbreviations — mg, mL, IU — retain their semantic
         identity after lowercasing).
      4. Remove safe trailing sentence punctuation (. ! ?).

    Args:
        text: Raw claim text.  May contain arbitrary whitespace.

    Returns:
        Normalized text.  May be empty if the input was whitespace-only or
        consisted solely of punctuation characters.
    """
    if not text:
        return ""

    # Step 1+2: strip + collapse whitespace
    normalized = _WHITESPACE_RUN_RE.sub(" ", text.strip())

    if not normalized:
        return ""

    # Step 3: lowercase
    normalized = normalized.lower()

    # Step 4: remove safe trailing punctuation
    normalized = _TRAILING_PUNCT_RE.sub("", normalized)

    # Final strip in case punctuation removal left trailing space
    return normalized.strip()


def apply_normalization(claim: Claim) -> Claim:
    """Return a new Claim with ``normalized_text`` populated.

    If the claim already has a ``normalized_text`` that equals the result
    of normalizing ``claim.text``, the original claim is returned unchanged
    (avoids unnecessary object creation).

    Args:
        claim: An immutable Claim instance.

    Returns:
        A (potentially new) Claim with ``normalized_text`` set.  Never
        mutates the input.

    Raises:
        No exceptions beyond those raised by ``Claim.__init__`` if the
        normalized text somehow violates Claim invariants (should not
        happen for well-formed claims).
    """
    normalized = normalize_claim_text(claim.text)

    # Avoid unnecessary replacement if already populated with same value
    if claim.normalized_text == normalized:
        return claim

    return dataclasses.replace(claim, normalized_text=normalized)