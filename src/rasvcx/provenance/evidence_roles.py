"""Evidence roles and temporal states (deterministic, auditable).

Retrieval returns candidates; it does not establish that a passage supports
the answer.  This module assigns every evidence item exactly one ROLE and
one TEMPORAL STATE from structured signals only:

TemporalStatus -- from DECLARED lifecycle metadata (SourceLifecycle), never
from the publication date alone and never from document text:
    WITHDRAWN    status == withdrawn
    SUPERSEDED   status == superseded, or superseded_by is set (ingestion
                 fills superseded_by KB-wide from other documents'
                 `supersedes` declarations)
    HISTORICAL   status == historical
    CURRENT      status == current
    UNKNOWN      nothing declared

EvidenceRole -- precedence, first match wins:
    CONTRADICTORY  M9 lists the item as contradicting a generated claim, or
                   it is in a genuine-conflict / unresolved pair with an item
                   the answer relies on
    SUPERSEDED     temporal state WITHDRAWN or SUPERSEDED (whether or not the
                   answer used it -- see supports_answer)
    SUPPORTING     M9 lists the item as supporting a supported or partially
                   supported generated claim
    RELEVANT       reranker score >= min_relevance_score
    IRRELEVANT     reranker score <  min_relevance_score
    RETRIEVED      none of the above could be established (no reranker
                   score, no verification)

supports_answer / contradicts_answer are kept separately so a superseded
item the answer nevertheless cited stays visible as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from rasvcx.schemas.common import EvidenceRelationship


class EvidenceRole(str, Enum):
    RETRIEVED = "RETRIEVED"
    RELEVANT = "RELEVANT"
    SUPPORTING = "SUPPORTING"
    CONTRADICTORY = "CONTRADICTORY"
    IRRELEVANT = "IRRELEVANT"
    SUPERSEDED = "SUPERSEDED"


class TemporalStatus(str, Enum):
    CURRENT = "CURRENT"
    SUPERSEDED = "SUPERSEDED"
    HISTORICAL = "HISTORICAL"
    WITHDRAWN = "WITHDRAWN"
    FUTURE = "FUTURE"        # declared effective date (or recorded date) is after today
    UNDATED = "UNDATED"      # no lifecycle declared AND no publication date recorded
    UNKNOWN = "UNKNOWN"      # dated, but no lifecycle declared


#: Temporal states that must never ground an unqualified answer about the
#: present: replaced, withdrawn, or not yet in effect.
NOT_CURRENT = frozenset({TemporalStatus.SUPERSEDED, TemporalStatus.WITHDRAWN, TemporalStatus.FUTURE})
#: Temporal states that explain a disagreement with current evidence.
STALE_OR_PENDING = NOT_CURRENT | {TemporalStatus.HISTORICAL}


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    role: EvidenceRole
    temporal_status: TemporalStatus
    supports_answer: bool
    contradicts_answer: bool
    reason: str
    temporal_reason: str


def _iso_date(value: Any) -> "date | None":
    """A recorded ISO date (YYYY-MM-DD, YYYY-MM, YYYY), else None.  Partial
    dates take their earliest day, so a future check is never premature."""
    from datetime import date

    if not isinstance(value, str):
        return None
    parts = value.strip()[:10].split("-")
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        return date(y, m, d)
    except (ValueError, IndexError):
        return None


def temporal_status(item: Any, today: "date | None" = None) -> tuple[TemporalStatus, str]:
    """Temporal state of one evidence item.

    Only DECLARED or RECORDED values are used -- never inferred from text,
    never fabricated: the declared lifecycle, its effective date, and the
    recorded publication date.  Precedence: withdrawn > superseded >
    historical > future (effective/publication date after today) >
    current > undated / unknown.
    """
    from datetime import date as _date

    today = today or _date.today()
    lc = getattr(getattr(item, "source", None), "lifecycle", None)
    if lc is not None:
        if lc.status == "withdrawn":
            return TemporalStatus.WITHDRAWN, "source declared withdrawn"
        if lc.status == "superseded" or lc.superseded_by:
            by = f" by {lc.superseded_by}" if lc.superseded_by else ""
            return TemporalStatus.SUPERSEDED, f"source superseded{by}"
        if lc.status == "historical":
            return TemporalStatus.HISTORICAL, "source declared historical"
    effective = _iso_date(getattr(lc, "effective_date", None)) if lc is not None else None
    published = _iso_date(getattr(getattr(item, "provenance", None), "date", None))
    if effective is not None and effective > today:
        return TemporalStatus.FUTURE, f"declared effective date {effective.isoformat()} is in the future"
    if published is not None and published > today:
        return TemporalStatus.FUTURE, f"recorded date {published.isoformat()} is in the future"
    if lc is not None and lc.status == "current":
        return TemporalStatus.CURRENT, "source declared current"
    if published is None and effective is None:
        return TemporalStatus.UNDATED, "no lifecycle declared and no date recorded"
    return TemporalStatus.UNKNOWN, "dated, but no lifecycle declared for this source"


_SUPPORT_LABELS = ("supported", "partially_supported")
_CONFLICT_RELATIONSHIPS = (EvidenceRelationship.GENUINE_CONFLICT, EvidenceRelationship.UNRESOLVED)


def assess_evidence(
    items: Iterable[Any],
    verification_summary: Any | None = None,
    validation_summary: Any | None = None,
    candidate_pairs: Mapping[str, tuple[str, str]] | None = None,
    min_relevance_score: float = 0.0,
) -> dict[str, EvidenceAssessment]:
    """Role + temporal state for every item, keyed by item_id (str)."""
    items = list(items)
    supporting: set[str] = set()
    contradicting: set[str] = set()
    if verification_summary is not None:
        for r in getattr(verification_summary, "claim_results", ()):
            label = getattr(r.label, "value", r.label)
            reason = getattr(getattr(r, "reason_code", None), "value", None)
            # A claim reported as history ("the earlier 20 mg schedule was
            # withdrawn") cites its non-current source as history, not as
            # support for the present answer.
            if label in _SUPPORT_LABELS and reason != "historical_statement":
                supporting.update(str(i) for i in r.supporting_item_ids)
            contradicting.update(str(i) for i in r.contradicting_item_ids)

    # Items in an unresolved / genuine conflict with evidence the answer uses.
    conflict_with_support: set[str] = set()
    if validation_summary is not None and candidate_pairs:
        for res in getattr(validation_summary, "resolutions", ()):
            if res.relationship not in _CONFLICT_RELATIONSHIPS:
                continue
            pair = candidate_pairs.get(str(res.candidate_id))
            if not pair:
                continue
            a, b = (str(x) for x in pair)
            if a in supporting and b not in supporting:
                conflict_with_support.add(b)
            if b in supporting and a not in supporting:
                conflict_with_support.add(a)

    out: dict[str, EvidenceAssessment] = {}
    for item in items:
        iid = str(item.item_id)
        tstatus, treason = temporal_status(item)
        supports = iid in supporting
        contradicts = iid in contradicting or iid in conflict_with_support
        score = getattr(item, "rerank_score", None)
        if contradicts:
            role, why = EvidenceRole.CONTRADICTORY, (
                "contradicts a generated claim" if iid in contradicting
                else "in an unresolved/genuine conflict with supporting evidence"
            )
        elif tstatus in NOT_CURRENT:
            role, why = EvidenceRole.SUPERSEDED, treason
        elif supports:
            role, why = EvidenceRole.SUPPORTING, "supports a verified generated claim"
        elif score is not None and score >= min_relevance_score:
            role, why = EvidenceRole.RELEVANT, f"rerank_score {score:.2f} >= {min_relevance_score}"
        elif score is not None:
            role, why = EvidenceRole.IRRELEVANT, f"rerank_score {score:.2f} < {min_relevance_score}"
        else:
            role, why = EvidenceRole.RETRIEVED, "retrieved; relevance and support not established"
        out[iid] = EvidenceAssessment(
            role=role, temporal_status=tstatus, supports_answer=supports,
            contradicts_answer=contradicts, reason=why, temporal_reason=treason,
        )
    return out


def answer_basis(items: list[Any], verification_summary: Any | None) -> list[Any]:
    """The items an answer's confidence should be computed from.

    The items M9 found supporting a generated claim, excluding superseded
    and withdrawn sources; all items when verification identified none
    (no answer yet, or nothing verifiable) -- never an empty basis when
    evidence exists.
    """
    if verification_summary is None:
        return items
    assessed = assess_evidence(items, verification_summary)
    basis = [
        it for it in items
        if assessed[str(it.item_id)].supports_answer
        and assessed[str(it.item_id)].temporal_status not in NOT_CURRENT
    ]
    return basis or items


__all__ = [
    "EvidenceAssessment",
    "EvidenceRole",
    "NOT_CURRENT",
    "STALE_OR_PENDING",
    "TemporalStatus",
    "answer_basis",
    "assess_evidence",
    "temporal_status",
]
