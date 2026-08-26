# src/rasvcx/sufficiency/scorer.py
"""Sufficiency signal computation (Module 5).

This module computes *auditable, individually-inspectable* signals about
the evidence currently sitting in an EvidenceBundle. It makes no decision
about whether that evidence is "enough" -- that judgment belongs to
sufficiency/gate.py, which combines these signals with the request's
RiskProfile and configurable thresholds.

Design constraints (see architecture rule for Module 5):
  - Pure functions over EvidenceBundle / RiskProfile: no mutation, no I/O,
    no retrieval calls, no budget tracking.
  - Every signal is a named, typed field -- never collapsed into a single
    opaque score, so later stages (provenance, evaluation/ablation) can
    consume individual signals.
  - Read-only over EvidenceItem/Provenance; never fabricates a concrete
    value for a field that is UNKNOWN, and never treats UNKNOWN as
    "compatible" or "known" when computing coverage/diversity signals.
  - Deterministic: given the same bundle contents, results are identical
    across calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from rasvcx.schemas.common import UNKNOWN
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem
from rasvcx.schemas.query import RiskProfile


@dataclass(frozen=True, slots=True)
class SufficiencySignals:
    """Individually-inspectable evidence signals for one sufficiency check.

    All score-like fields are normalized to [0, 1] where the underlying
    quantity is meaningfully bounded; count-like fields are raw integers.
    A signal being 0.0 does NOT necessarily mean "bad" -- e.g. an empty
    ``unknown_provenance_ratio`` means "no contextual fields are UNKNOWN,"
    which is distinct from "evidence is consistent" (see architecture
    note: absence of contradiction is not proof of correctness).
    Downstream consumers must not treat any single field as dispositive.

    Attributes:
        evidence_item_count:       Raw count of evidence items in the bundle.
        scored_item_count:         Count of items with a non-None rerank_score.
        top_rerank_score:          Highest rerank_score present, or None if
                                    no item has been scored.
        rerank_score_margin:       top_rerank_score minus the next-highest
                                    rerank_score, or None if fewer than 2
                                    scored items exist. A small margin means
                                    the top result is not clearly separated
                                    from its closest competitor.
        mean_retrieval_score:      Mean retrieval_score across all items, or
                                    None if the bundle is empty.
        source_type_diversity:     Count of distinct SourceType values
                                    present across evidence items.
        unknown_provenance_ratio:  Fraction of (item, provenance-field) pairs
                                    across {date, jurisdiction, population,
                                    dosage_context} that are UNKNOWN, in
                                    [0, 1]. Higher means less contextual
                                    information is available to rule out
                                    conflicts later in the pipeline.
        targeted_retrieval_used:   Whether targeted retrieval has already
                                    run for this bundle (from bundle
                                    metadata) -- signals that a second
                                    retrieval pass has already been spent.
        unscored_item_ratio:       Fraction of items with rerank_score=None,
                                    in [0, 1]. 0.0 for an empty bundle.
    """

    evidence_item_count: int
    scored_item_count: int
    top_rerank_score: float | None
    rerank_score_margin: float | None
    mean_retrieval_score: float | None
    source_type_diversity: int
    unknown_provenance_ratio: float
    targeted_retrieval_used: bool
    unscored_item_ratio: float

    def __post_init__(self) -> None:
        if self.evidence_item_count < 0:
            raise ValueError("evidence_item_count must be non-negative")
        if self.scored_item_count < 0:
            raise ValueError("scored_item_count must be non-negative")
        if self.scored_item_count > self.evidence_item_count:
            raise ValueError("scored_item_count cannot exceed evidence_item_count")
        if not 0.0 <= self.unknown_provenance_ratio <= 1.0:
            raise ValueError(
                f"unknown_provenance_ratio must be in [0, 1], got "
                f"{self.unknown_provenance_ratio}"
            )
        if not 0.0 <= self.unscored_item_ratio <= 1.0:
            raise ValueError(
                f"unscored_item_ratio must be in [0, 1], got {self.unscored_item_ratio}"
            )
        if self.source_type_diversity < 0:
            raise ValueError("source_type_diversity must be non-negative")


_PROVENANCE_CONTEXT_FIELDS = ("date", "jurisdiction", "population", "dosage_context")


def _unknown_provenance_ratio(items: list[EvidenceItem]) -> float:
    """Fraction of context provenance fields that are UNKNOWN.

    Counts across all four contextual fields (date, jurisdiction,
    population, dosage_context) on every item -- source_type is excluded
    because it is never UNKNOWN by contract (it is a concrete SourceType
    enum value, not ``str | Unknown``).
    """
    if not items:
        return 0.0
    total_fields = 0
    unknown_fields = 0
    for item in items:
        for field_name in _PROVENANCE_CONTEXT_FIELDS:
            total_fields += 1
            if getattr(item.provenance, field_name) is UNKNOWN:
                unknown_fields += 1
    if total_fields == 0:
        return 0.0
    return unknown_fields / total_fields


def compute_sufficiency_signals(
    bundle: EvidenceBundle,
    risk_profile: RiskProfile,
) -> SufficiencySignals:
    """Compute sufficiency signals from the current state of ``bundle``.

    ``risk_profile`` is accepted for interface symmetry with the gate (and
    because future signals -- e.g. risk-scaled coverage targets -- may need
    it) but is currently not read; all signals here are evidence-derived
    only. It is validated for presence but not otherwise inspected.

    Args:
        bundle:        Current EvidenceBundle (post-retrieval, post-rerank).
        risk_profile:  This request's RiskProfile (read-only).

    Returns:
        SufficiencySignals computed from ``bundle.evidence_items``.
    """
    if risk_profile is None:
        raise ValueError("risk_profile must not be None")

    items = list(bundle.evidence_items.values())
    evidence_item_count = len(items)

    scored = [it for it in items if it.rerank_score is not None]
    scored_item_count = len(scored)

    top_rerank_score: float | None = None
    rerank_score_margin: float | None = None
    if scored:
        sorted_scores = sorted(
            (it.rerank_score for it in scored), reverse=True  # type: ignore[arg-type]
        )
        top_rerank_score = sorted_scores[0]
        if len(sorted_scores) >= 2:
            rerank_score_margin = sorted_scores[0] - sorted_scores[1]

    mean_retrieval_score: float | None = None
    if items:
        mean_retrieval_score = sum(it.retrieval_score for it in items) / len(items)

    source_type_diversity = len({it.provenance.source_type for it in items})

    unknown_provenance_ratio = _unknown_provenance_ratio(items)

    unscored_item_ratio = (
        0.0
        if evidence_item_count == 0
        else (evidence_item_count - scored_item_count) / evidence_item_count
    )

    return SufficiencySignals(
        evidence_item_count=evidence_item_count,
        scored_item_count=scored_item_count,
        top_rerank_score=top_rerank_score,
        rerank_score_margin=rerank_score_margin,
        mean_retrieval_score=mean_retrieval_score,
        source_type_diversity=source_type_diversity,
        unknown_provenance_ratio=unknown_provenance_ratio,
        targeted_retrieval_used=bundle.metadata.targeted_retrieval_used,
        unscored_item_ratio=unscored_item_ratio,
    )