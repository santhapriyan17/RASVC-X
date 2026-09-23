"""Evidence formatting and citation extraction (Module 11).

Provides two concerns:

1. format_evidence_block() — renders EvidenceItems into prompt-ready
   text blocks with citation labels, provenance metadata, and explicit
   data delimiters that prevent retrieved content from being interpreted
   as instructions.

2. extract_cited_ids() — scans a generated answer string and returns
   the set of EvidenceItemIds that appeared as citation markers. This
   is observability metadata only; M9's GeneratedClaimExtractor is the
   authoritative per-claim citation mapping.

Citation label convention:
   Each evidence item is labeled by its EvidenceItemId (e.g. "E1"),
   matching the two citation syntaxes M9's claim_extraction_post.py
   already recognizes:
     - Direct-ID: [E1] — matched case-sensitively against bundle keys
     - Numeric:   [1]  — 1-based index into sorted EvidenceItemIds
   Using the actual EvidenceItemId as the label means direct-ID
   resolution works with zero ambiguity.

Security boundary (minimal, generation-level only):
   Evidence text is wrapped in <evidence> XML-style tags. The system
   prompt instructs the model to treat content within these tags as
   retrieved data, never as instructions. This is not a complete prompt
   injection defense (that belongs to M15) — it is the minimum
   architectural separation required at the generation layer.

No existing M1-M10 contract is modified. EvidenceBundle is read-only.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from rasvcx.schemas.common import EvidenceItemId, UNKNOWN, _UnknownType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance


# ---------------------------------------------------------------------------
# Evidence block formatting
# ---------------------------------------------------------------------------


def _render_provenance_attr(name: str, value: object) -> str:
    """Render a single provenance attribute for the evidence tag.

    UNKNOWN fields are rendered explicitly as 'unknown' — never omitted,
    never fabricated into a concrete value.
    """
    if isinstance(value, _UnknownType):
        return f'{name}="unknown"'
    return f'{name}="{value}"'


def format_evidence_block(item: EvidenceItem) -> str:
    """Render one EvidenceItem as a prompt-ready evidence block.

    Format:
        <evidence id="E1" source_type="clinical_guideline" date="2023-01-15"
                  jurisdiction="US" population="adults" dosage_context="unknown">
        The recommended dosage is 500mg twice daily for adults.
        </evidence>

    The XML-style tags are data delimiters, not executable markup.
    The system prompt instructs the model to treat their content as
    retrieved data only.
    """
    prov = item.provenance
    attrs = " ".join([
        f'id="{item.item_id}"',
        _render_provenance_attr("source_type", prov.source_type.value
                                 if hasattr(prov.source_type, "value") else prov.source_type),
        _render_provenance_attr("date", prov.date),
        _render_provenance_attr("jurisdiction", prov.jurisdiction),
        _render_provenance_attr("population", prov.population),
        _render_provenance_attr("dosage_context", prov.dosage_context),
    ])
    return f"<evidence {attrs}>\n{item.text}\n</evidence>"


def format_evidence_blocks(
    bundle: EvidenceBundle,
) -> tuple[str, list[EvidenceItemId]]:
    """Render ALL evidence items from the bundle into a single evidence
    section string.

    Returns:
        (evidence_section, ordered_item_ids)

    evidence_section: concatenated evidence blocks separated by blank
        lines, ready for inclusion in the user prompt.
    ordered_item_ids: the EvidenceItemIds in the order they appear in
        the section (sorted by EvidenceItemId for deterministic ordering).

    ALL evidence items are included. No evidence is silently dropped,
    truncated, or selected. The bundle is read-only — this function
    never modifies it.
    """
    sorted_ids = sorted(bundle.evidence_items.keys())
    blocks: list[str] = []
    for item_id in sorted_ids:
        item = bundle.evidence_items[item_id]
        blocks.append(format_evidence_block(item))
    return "\n\n".join(blocks), sorted_ids


# ---------------------------------------------------------------------------
# Citation extraction from generated text
# ---------------------------------------------------------------------------

# Reuse the same regex pattern M9 uses (claim_extraction_post.py line 51),
# so citation_formatter and M9 agree on what constitutes a citation marker.
_CITATION_MARKER_RE = re.compile(
    r"\[([A-Za-z0-9_\-]{1,64}(?:\s*,\s*[A-Za-z0-9_\-]{1,64}){0,31})\]"
)


def extract_cited_ids(
    generated_text: str,
    bundle: EvidenceBundle,
) -> frozenset[EvidenceItemId]:
    """Extract EvidenceItemIds cited in generated_text.

    Scans for citation markers matching M9's convention and resolves
    them against the bundle's evidence items. Unresolvable markers are
    silently ignored here (they are NOT the authoritative mapping — M9
    handles unresolved citations per-claim with full traceability).

    This is observability metadata for GenerationResult.cited_item_ids.
    """
    if not generated_text:
        return frozenset()

    known_ids = bundle.evidence_items.keys()
    sorted_ids = sorted(known_ids)

    cited: set[EvidenceItemId] = set()
    for match in _CITATION_MARKER_RE.finditer(generated_text):
        tokens = [t.strip() for t in match.group(1).split(",") if t.strip()]
        for token in tokens:
            candidate = EvidenceItemId(token)
            if candidate in bundle.evidence_items:
                cited.add(candidate)
            elif token.isdigit():
                position = int(token) - 1
                if 0 <= position < len(sorted_ids):
                    cited.add(sorted_ids[position])

    return frozenset(cited)


__all__ = [
    "format_evidence_block",
    "format_evidence_blocks",
    "extract_cited_ids",
]