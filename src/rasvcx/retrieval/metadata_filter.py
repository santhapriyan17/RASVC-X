"""Metadata-based filtering of retrieval results for RASVC-X.

Filters fused retrieval results against caller-supplied provenance criteria.
Applied AFTER RRF fusion and BEFORE reranking.

UNKNOWN semantics (architectural invariant):
    A result with UNKNOWN provenance for a filtered dimension is NEVER
    excluded.  UNKNOWN means "insufficient information to rule out relevance."
    Excluding UNKNOWN results would silently discard potentially valid
    evidence — the more dangerous failure mode for a medical RAG system.

Design constraints:
- Pure functions: no I/O, no side effects, no global mutable state.
- All filter criteria are optional; omitting = no filter on that dimension.
- UNKNOWN provenance always passes any filter criterion.
- Results missing from provenance_map are passed through (fail-open).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rasvcx.retrieval.fusion import FusedResult
from rasvcx.schemas.common import UNKNOWN, Unknown

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MetadataFilterCriteria:
    """Criteria for filtering retrieval results by provenance metadata.

    Each field is an optional allowlist. A result passes a dimension if:
      - The criterion list is empty (no filter), OR
      - The result's provenance value is UNKNOWN, OR
      - The result's provenance value is in the criterion list.

    Attributes:
        allowed_jurisdictions:   Allowlist of jurisdiction strings.
        allowed_populations:     Allowlist of population strings.
        min_date:                Minimum date string (ISO 8601, inclusive).
        max_date:                Maximum date string (ISO 8601, inclusive).
        allowed_dosage_contexts: Allowlist of dosage context strings.
    """

    allowed_jurisdictions: tuple[str, ...] = field(default_factory=tuple)
    allowed_populations: tuple[str, ...] = field(default_factory=tuple)
    min_date: str | None = None
    max_date: str | None = None
    allowed_dosage_contexts: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class ProvenanceSnapshot:
    """Provenance values for a single result, extracted before filtering.

    Decouples filter logic from EvidenceItem/Provenance schema for testability.
    """

    chunk_id: str
    jurisdiction: str | Unknown
    population: str | Unknown
    date: str | Unknown
    dosage_context: str | Unknown


def _passes_allowlist(value: str | Unknown, allowlist: tuple[str, ...]) -> bool:
    if not allowlist:
        return True
    if isinstance(value, Unknown):
        return True
    return value in allowlist


def _passes_date_range(date: str | Unknown, min_date: str | None, max_date: str | None) -> bool:
    if min_date is None and max_date is None:
        return True
    if isinstance(date, Unknown):
        return True
    if min_date is not None and date < min_date:
        return False
    if max_date is not None and date > max_date:
        return False
    return True


def _passes_criteria(prov: ProvenanceSnapshot, criteria: MetadataFilterCriteria) -> bool:
    if not _passes_allowlist(prov.jurisdiction, criteria.allowed_jurisdictions):
        logger.debug("chunk '%s' filtered: jurisdiction '%s' not in allowlist %s",
                     prov.chunk_id, prov.jurisdiction, criteria.allowed_jurisdictions)
        return False
    if not _passes_allowlist(prov.population, criteria.allowed_populations):
        logger.debug("chunk '%s' filtered: population '%s' not in allowlist %s",
                     prov.chunk_id, prov.population, criteria.allowed_populations)
        return False
    if not _passes_date_range(prov.date, criteria.min_date, criteria.max_date):
        logger.debug("chunk '%s' filtered: date '%s' outside [%s, %s]",
                     prov.chunk_id, prov.date, criteria.min_date, criteria.max_date)
        return False
    if not _passes_allowlist(prov.dosage_context, criteria.allowed_dosage_contexts):
        logger.debug("chunk '%s' filtered: dosage_context '%s' not in allowlist %s",
                     prov.chunk_id, prov.dosage_context, criteria.allowed_dosage_contexts)
        return False
    return True


def filter_by_metadata(
    results: list[FusedResult],
    provenance_map: dict[str, ProvenanceSnapshot],
    criteria: MetadataFilterCriteria,
) -> list[FusedResult]:
    """Filter fused retrieval results by provenance metadata.

    Results not present in provenance_map are passed through (fail-open).
    Order is preserved.

    Args:
        results:        Fused results in descending rrf_score order.
        provenance_map: Dict mapping chunk_id → ProvenanceSnapshot.
        criteria:       Filter criteria to apply.

    Returns:
        Filtered list of FusedResult preserving original order.
        Returns input as-is (O(1)) when no criteria are active.
    """
    no_filter = (
        not criteria.allowed_jurisdictions
        and not criteria.allowed_populations
        and criteria.min_date is None
        and criteria.max_date is None
        and not criteria.allowed_dosage_contexts
    )
    if no_filter:
        return results

    filtered: list[FusedResult] = []
    excluded = 0

    for result in results:
        prov = provenance_map.get(result.chunk_id)
        if prov is None:
            logger.warning("chunk '%s' has no provenance snapshot; passing through (fail-open)",
                           result.chunk_id)
            filtered.append(result)
            continue
        if _passes_criteria(prov, criteria):
            filtered.append(result)
        else:
            excluded += 1

    if excluded:
        logger.debug("metadata_filter: %d / %d results excluded by criteria",
                     excluded, len(results))
    return filtered