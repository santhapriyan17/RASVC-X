"""Post-generation claim extraction and citation alignment (Sections 9-13, 32-34, 67-70).

Extracts atomic, independently-verifiable claims from a *generated* answer
(as opposed to claims/extractor.py, which extracts claims from *evidence*
text during Module 7) and aligns each claim with any citation markers it
carries.

Design decisions, explicit and documented rather than silently assumed:

  - Sentence-level atomicity, matching M7's own documented decision (see
    claims/extractor.py's ClaimExtractor docstring): full sub-sentence
    compositional decomposition (Section 21) is NOT attempted, because
    deterministic clause splitting without an NLP parser risks destroying
    numeric/qualifier context exactly as M7 already reasoned. This module
    reuses ``split_sentences`` from claims/extractor.py directly rather
    than re-implementing sentence segmentation.
  - Citation marker syntax: no citation-rendering convention exists
    elsewhere in this repository (generation/*.py is a set of empty
    stubs), so this module recognizes two conventions and treats anything
    else as unparseable-but-preserved (Section 32: "Do not delete
    citations merely because they are difficult to parse"):
      1. Direct-ID citations, e.g. ``[E12]`` / ``[evid-42]`` -- the
         bracketed text is matched case-sensitively against
         ``EvidenceBundle.evidence_items`` keys.
      2. Numeric citations, e.g. ``[1]`` -- resolved as a 1-based index
         into evidence items sorted by ``EvidenceItemId`` (deterministic
         ordering). This is an explicit, documented assumption about
         citation numbering, not a repository-wide contract.
  - Fenced code blocks (```...```) and inline code spans (`...`) are
    stripped before citation-marker scanning, so citation-like text inside
    code is never misread as a real citation (Section 68).
"""

from __future__ import annotations

import hashlib
import re

from rasvcx.claims.claim_normalizer import normalize_claim_text
from rasvcx.claims.extractor import classify_claim_type, is_safety_critical, split_sentences
from rasvcx.schemas.common import EvidenceItemId
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.verification import GeneratedClaim, GeneratedClaimId

# ---------------------------------------------------------------------------
# Precompiled patterns
# ---------------------------------------------------------------------------

_FENCED_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
_CITATION_MARKER_RE = re.compile(
    r"\[([A-Za-z0-9_\-]{1,64}(?:\s*,\s*[A-Za-z0-9_\-]{1,64}){0,31})\]"
)
"""Matches a single bracketed citation, e.g. "[E1]" or "[1]", AND a
comma-separated multi-citation bracket, e.g. "[1,2]" or "[E1, E2]". The
{0,31} bound caps a single bracket at 32 comma-separated ids -- generous
for any realistic answer, but keeps the regex non-pathological."""


def _mask_spans(text: str, pattern: re.Pattern[str]) -> str:
    """Replace all regex matches with same-length whitespace, preserving
    character offsets and newlines (so downstream span math stays valid).
    """

    def _blank(match: re.Match[str]) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return pattern.sub(_blank, text)


_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[*+\-\u2022]|\d{1,3}[.)]|#{1,6})[ \t]+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"\*{1,3}|_{2,3}|^[ \t]*>+", re.MULTILINE)
_TERMINAL = ".!?"


def _mask_markdown_structure(text: str) -> str:
    """Turn markdown layout into sentence boundaries, preserving offsets.

    A model that answers with headings and bullet lists produces lines, not
    sentences: without this, a whole list ("... include: * A ... * B ...")
    is one "claim", and its many unrelated numbers and qualifiers are then
    verified as if they were a single statement.

    Every substitution is same-length, so claim spans still index into the
    original answer:
      - list / heading markers and emphasis characters become spaces;
      - a line break that ends a line without terminal punctuation, and is
        followed by a list item or a blank line, becomes a full stop, so
        each list item and each paragraph is its own sentence.
    """
    chars = list(text)
    marker_starts: set[int] = set()
    for match in _LIST_MARKER_RE.finditer(text):
        marker_starts.add(match.start())
        for i in range(match.start(), match.end()):
            if chars[i] not in "\r\n":
                chars[i] = " "
    for match in _EMPHASIS_RE.finditer(text):
        for i in range(match.start(), match.end()):
            if chars[i] not in "\r\n":
                chars[i] = " "

    for i, ch in enumerate(text):
        if ch != "\n":
            continue
        next_line_start = i + 1
        starts_item = next_line_start in marker_starts
        blank_follows = next_line_start >= len(text) or text[next_line_start] in "\r\n"
        if not (starts_item or blank_follows):
            continue  # soft wrap inside a paragraph: same sentence
        j = i - 1
        while j >= 0 and chars[j] in " \t\r":
            j -= 1
        if j >= 0 and chars[j] not in _TERMINAL and chars[j] != "\n":
            chars[i] = "."
    return "".join(chars)


def _mask_code_spans(text: str) -> str:
    """Replace fenced/inline code content with same-length whitespace.

    Preserves character offsets (so extracted claim spans still index into
    the *original* text) while preventing citation-marker matches inside
    code (Section 68).
    """
    text = _mask_spans(text, _FENCED_CODE_RE)
    text = _mask_spans(text, _INLINE_CODE_RE)
    return text


def generate_generated_claim_id(answer_digest: str, index: int, claim_text: str) -> GeneratedClaimId:
    """Deterministic, SHA-256-based ID for a generated claim.

    Mirrors claims/extractor.py's ``generate_claim_id`` strategy (content-
    derived, reproducible, no UUIDs) but keyed on the generated answer's
    own digest + position, since generated claims have no origin_chunk_id.
    """
    identity = f"{answer_digest}:{index}:{claim_text}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return GeneratedClaimId(f"gclaim_{digest}")


def _resolve_citation_tokens(
    tokens: list[str], bundle: EvidenceBundle
) -> tuple[frozenset[EvidenceItemId], frozenset[str]]:
    """Resolve raw bracket tokens (e.g. "E12", "1") to EvidenceItemIds.

    Returns ``(resolved_ids, unresolved_tokens)``: unresolvable tokens are
    NOT silently discarded (Section 13 distinguishes MISSING citation from
    INCORRECT citation, which requires knowing that a marker was present
    but invalid, not merely that no evidence was found).
    """
    if not tokens:
        return frozenset(), frozenset()

    known_ids = bundle.evidence_items.keys()
    sorted_ids = sorted(known_ids)  # deterministic index for numeric citations

    resolved: set[EvidenceItemId] = set()
    unresolved: set[str] = set()
    for token in tokens:
        candidate = EvidenceItemId(token)
        if candidate in bundle.evidence_items:
            resolved.add(candidate)
            continue
        if token.isdigit():
            position = int(token) - 1  # 1-based numeric citation convention
            if 0 <= position < len(sorted_ids):
                resolved.add(sorted_ids[position])
                continue
        unresolved.add(token)
    return frozenset(resolved), frozenset(unresolved)


def _collapse_whitespace_with_offsets(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to single spaces and strip, mirroring
    ``split_sentences``'s own internal normalization, while returning a
    parallel list mapping each character index in the collapsed string to
    its original index in *text*.
    """
    out_chars: list[str] = []
    index_map: list[int] = []
    in_whitespace = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_whitespace:
                out_chars.append(" ")
                index_map.append(i)
                in_whitespace = True
            continue
        out_chars.append(ch)
        index_map.append(i)
        in_whitespace = False

    # Strip leading/trailing space, trimming the index map in lockstep.
    start = 0
    end = len(out_chars)
    while start < end and out_chars[start] == " ":
        start += 1
    while end > start and out_chars[end - 1] == " ":
        end -= 1

    return "".join(out_chars[start:end]), index_map[start:end]


class GeneratedClaimExtractor:
    """Extracts GeneratedClaims (with resolved citations) from an answer.

    Complexity: O(n) in answer length for sentence splitting + citation
    scanning; O(k) per claim for citation resolution where k = markers on
    that claim (bounded, not proportional to bundle size beyond a single
    dict lookup / sorted-index lookup per marker).
    """

    def __init__(self, min_claim_length: int = 5) -> None:
        self._min_claim_length = min_claim_length

    def extract(self, generated_answer: str, bundle: EvidenceBundle) -> list[GeneratedClaim]:
        """Extract claims from *generated_answer* with citations resolved
        against *bundle*.

        Empty or whitespace-only input returns an empty list rather than
        raising (Section 58) -- callers (verdict_aggregator) are
        responsible for treating "no claims" as its own explicit case, not
        this function forcing a verdict.
        """
        if not generated_answer or not generated_answer.strip():
            return []

        answer_digest = hashlib.sha256(generated_answer.encode("utf-8")).hexdigest()[:16]
        code_masked = _mask_code_spans(generated_answer)

        # Citation markers (e.g. "[E1]") are found on the code-masked text
        # -- so a marker-like token inside code was never a real marker --
        # but must be masked out *before* sentence splitting: a marker
        # sitting between two sentences (e.g. "...accuracy. [E1] It is...")
        # otherwise defeats split_sentences()'s boundary regex, which
        # requires an uppercase/digit/quote character immediately after the
        # terminal punctuation and whitespace -- "[" fails that check and
        # silently merges what should be two separate claims.
        markers = [
            (match.span(), token.strip())
            for match in _CITATION_MARKER_RE.finditer(code_masked)
            for token in match.group(1).split(",")
            if token.strip()
        ]
        citation_masked = _mask_spans(code_masked, _CITATION_MARKER_RE)
        citation_masked = _mask_markdown_structure(citation_masked)

        sentences_with_spans = self._split_with_spans(citation_masked)
        marker_assignments = self._assign_markers_to_sentences(markers, sentences_with_spans)

        claims: list[GeneratedClaim] = []
        for index, (sentence, span) in enumerate(sentences_with_spans):
            claim_text_only = re.sub(r"[ \t]{2,}", " ", sentence).strip()
            # A line break promoted to a full stop can leave ":." or a
            # space before the stop; tidy the text that is shown and verified.
            claim_text_only = re.sub(r"\s+([.!?])$", r"\1", claim_text_only)
            if len(claim_text_only) < self._min_claim_length:
                continue
            if claim_text_only.rstrip(".").rstrip().endswith(":"):
                # A lead-in such as "The indications include:" introduces a
                # list; it asserts nothing by itself and is not a claim.
                continue

            normalized = normalize_claim_text(claim_text_only)
            if not normalized:
                continue

            marker_tokens = marker_assignments[index]
            cited_ids, unresolved_tokens = _resolve_citation_tokens(marker_tokens, bundle)
            claim_type = classify_claim_type(claim_text_only)
            safety = is_safety_critical(claim_text_only, claim_type)

            claim_id = generate_generated_claim_id(answer_digest, index, claim_text_only)
            claims.append(
                GeneratedClaim(
                    claim_id=claim_id,
                    text=claim_text_only,
                    normalized_text=normalized,
                    span=span,
                    cited_item_ids=cited_ids,
                    unresolved_citation_tokens=unresolved_tokens,
                    is_safety_critical=safety,
                )
            )

        return claims

    def _assign_markers_to_sentences(
        self,
        markers: list[tuple[tuple[int, int], str]],
        sentences_with_spans: list[tuple[str, tuple[int, int]]],
    ) -> dict[int, list[str]]:
        """Attach each citation marker to a sentence index.

        A marker inside a sentence's own span (rare, e.g. mid-clause
        citations) attaches to that sentence. A marker in the *gap*
        between two sentences -- the common case, e.g.
        "...accuracy. [E1] It is..." -- attaches to the nearest
        *preceding* sentence, matching the natural reading convention that
        a citation follows the claim it supports. A marker appearing
        before any sentence (no preceding sentence exists) attaches to the
        first sentence instead, rather than being silently dropped.
        """
        assignments: dict[int, list[str]] = {i: [] for i in range(len(sentences_with_spans))}
        if not sentences_with_spans:
            return assignments

        for (marker_start, _marker_end), token in markers:
            target_index = 0
            for index, (_sentence, (start, end)) in enumerate(sentences_with_spans):
                if start <= marker_start < end:
                    target_index = index
                    break
                if end <= marker_start:
                    target_index = index  # nearest preceding sentence so far
                else:
                    break
            assignments[target_index].append(token)

        return assignments

    def _split_with_spans(self, masked_text: str) -> list[tuple[str, tuple[int, int]]]:
        """Split into sentences while recovering each sentence's original
        character span.

        ``split_sentences`` internally collapses whitespace runs before
        splitting, so a naive substring search for its returned sentences
        against *masked_text* silently fails whenever masking introduced
        multi-space runs (from blanked-out citation markers or code
        spans) -- exactly the case this module produces. Instead, this
        collapses whitespace itself while recording an explicit
        collapsed-index -> original-index map, so each returned sentence's
        position in the (now single-spaced) collapsed text can be
        translated back to a true original-text span.
        """
        collapsed, index_map = _collapse_whitespace_with_offsets(masked_text)
        sentences = split_sentences(collapsed)

        spans: list[tuple[str, tuple[int, int]]] = []
        cursor = 0
        for sentence in sentences:
            start_in_collapsed = collapsed.find(sentence, cursor)
            if start_in_collapsed == -1 or not index_map:
                # Defensive fallback: should not occur given collapsed is
                # exactly what split_sentences operated on, but never raise.
                spans.append((sentence, (cursor, cursor)))
                continue
            end_in_collapsed = start_in_collapsed + len(sentence)
            original_start = index_map[start_in_collapsed]
            original_end = (
                index_map[end_in_collapsed - 1] + 1
                if end_in_collapsed - 1 < len(index_map)
                else index_map[-1] + 1
            )
            spans.append((sentence, (original_start, original_end)))
            cursor = end_in_collapsed
        return spans


def orphan_citation_item_ids(
    claims: list[GeneratedClaim], bundle: EvidenceBundle
) -> frozenset[EvidenceItemId]:
    """Evidence items present in the bundle but never cited by any claim.

    Section 33: distinct from an *invalid* citation marker (which never
    resolves to an EvidenceItemId at all) -- this is the inverse case,
    evidence that exists but was never referenced.
    """
    cited: set[EvidenceItemId] = set()
    for claim in claims:
        cited |= claim.cited_item_ids
    return frozenset(bundle.evidence_items.keys()) - cited