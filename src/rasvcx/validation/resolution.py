"""Conflict detection / resolution (Sections 13-14).

Combines the deterministic, contextual, and (if available) NLI validation
results for a CandidatePair into a single, resolved
``EvidenceRelationshipResult`` using the fixed ``EvidenceRelationship``
taxonomy already defined in ``rasvcx.schemas.common``. Never invents a new
taxonomy value.

Resolution re-derives the underlying M6 applicability/temporal signals
(cheap, O(1), pure functions) rather than parsing the free-text
``rationale`` strings on upstream ValidationResults, so the mapping to a
specific EvidenceRelationship (POPULATION_DIFF, TEMPORAL_DIFF, ...) is
based on structured data, not string matching.

Source quality (M6's ``SourceQualityScorer``) is used as *supporting*
traceability context on GENUINE_CONFLICT / UNRESOLVED outcomes -- it is
recorded in the rationale for downstream Confidence/Decision modules to
weigh, but never used by M8 itself to unilaterally pick a "winning" side
of a conflict (see Section 14: no fabricated certainty).
"""

from __future__ import annotations

from typing import Mapping

from rasvcx.provenance.context_extractor import (
    ApplicabilityLabel,
    ApplicabilitySignal,
    TemporalConflictLabel,
    TemporalConflictSignal,
    compare_applicability,
    detect_temporal_conflict,
)
from rasvcx.provenance.evidence_roles import STALE_OR_PENDING, TemporalStatus, temporal_status
from rasvcx.provenance.source_quality import SourceQualityScorer
from rasvcx.schemas.common import CandidateId, ClaimId, EvidenceRelationship
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, EvidenceItemId
from rasvcx.schemas.validation import (
    EvidenceRelationshipResult,
    ValidationLabel,
    ValidationResult,
    ValidationStage,
)


#: superseded / withdrawn / historical / not-yet-effective (FUTURE)
_NON_CURRENT = STALE_OR_PENDING


def _contributing_claim_ids(
    bundle: EvidenceBundle, item_id_a: EvidenceItemId, item_id_b: EvidenceItemId
) -> frozenset[ClaimId]:
    item_a = bundle.evidence_items[item_id_a]
    item_b = bundle.evidence_items[item_id_b]
    return frozenset(item_a.extracted_claim_ids) | frozenset(item_b.extracted_claim_ids)


def _is_no_signal(result: ValidationResult) -> bool:
    """True for the deterministic stage's explicit "nothing comparable"
    outcome (see DeterministicValidator.validate)."""
    return (
        result.stage is ValidationStage.DETERMINISTIC
        and result.label is ValidationLabel.UNCERTAIN
        and result.confidence == 0.0
    )


class EvidenceResolver:
    """Resolves a candidate pair's accumulated validation signal into an
    ``EvidenceRelationshipResult``.

    Resolution never chooses "the newest text" or "the highest retrieval
    score" as a tiebreaker in isolation -- source quality, temporal
    applicability, and validation-stage agreement are combined, and any
    genuinely unresolved case maps to ``EvidenceRelationship.UNRESOLVED``
    rather than a fabricated verdict.
    """

    def __init__(self, source_quality_scorer: SourceQualityScorer | None = None) -> None:
        self._source_quality_scorer = source_quality_scorer

    def resolve(
        self,
        bundle: EvidenceBundle,
        candidate_id: CandidateId,
        item_id_a: EvidenceItemId,
        item_id_b: EvidenceItemId,
        deterministic_result: ValidationResult,
        contextual_result: ValidationResult | None,
        nli_result: ValidationResult | None,
    ) -> EvidenceRelationshipResult:
        item_a = bundle.evidence_items[item_id_a]
        item_b = bundle.evidence_items[item_id_b]
        contributing = _contributing_claim_ids(bundle, item_id_a, item_id_b)

        applicability_signals = compare_applicability(item_a.provenance, item_b.provenance)
        temporal_signal = detect_temporal_conflict(item_a.provenance, item_b.provenance)

        mismatched_by_field: dict[str, ApplicabilitySignal] = {
            s.field_name: s for s in applicability_signals if s.label is ApplicabilityLabel.MISMATCH
        }
        any_unknown = any(s.label is ApplicabilityLabel.UNKNOWN for s in applicability_signals)
        temporal_diverges = temporal_signal.label is TemporalConflictLabel.TEMPORAL_DIVERGENCE
        temporal_unknown = temporal_signal.label is TemporalConflictLabel.UNKNOWN

        # A genuine disagreement can be raised by *any* stage that ran --
        # e.g. deterministic numeric comparison finding a contradiction
        # must not be silently overridden by a contextual stage that only
        # found the provenance metadata itself unremarkable. Conversely,
        # "compatible" requires every stage that ran to agree; one
        # contradicting stage is enough to withhold COMPATIBLE.
        results = [r for r in (deterministic_result, contextual_result, nli_result) if r is not None]
        indicates_conflict = any(r.label is ValidationLabel.CONTRADICTION for r in results)

        # The deterministic stage reports "nothing comparable" (UNCERTAIN at
        # confidence 0.0) when the two items make no statement about the
        # same quantity.  That is the ABSENCE of a signal, not a doubtful
        # one: there is no disagreement to resolve.  Such a result must not
        # veto COMPATIBLE when every stage that did produce a signal agrees,
        # and it must not be counted as an unresolved conflict either.
        informative = [r for r in results if not _is_no_signal(r)]
        indicates_support = bool(informative) and all(
            r.label is ValidationLabel.SUPPORTED for r in informative
        )
        indicates_uncertain = any(
            r.label in (ValidationLabel.UNCERTAIN, ValidationLabel.PARTIAL) for r in informative
        )

        # 1. A known contextual mismatch, on its own, is not a genuine
        #    conflict -- it explains *why* the underlying claims might
        #    differ. Prefer the most specific known-mismatch explanation.
        if mismatched_by_field:
            confidence = self._confidence_for(results)
            if "dosage_context" in mismatched_by_field:
                relationship = EvidenceRelationship.DOSAGE_DIFF
            elif "jurisdiction" in mismatched_by_field:
                relationship = EvidenceRelationship.JURISDICTION_DIFF
            else:
                relationship = EvidenceRelationship.POPULATION_DIFF
            return self._build(
                candidate_id, relationship, confidence, contributing,
                mismatched_by_field, temporal_signal,
            )

        # 1b. Declared lifecycle decides temporal disagreements.  A
        #     contradiction is explained by time ONLY when one side is
        #     declared superseded / withdrawn / historical and the other is
        #     not.  Two sources both declared current that disagree are a
        #     genuine conflict (conflicting effective guidance), whatever
        #     their dates.  Publication-date distance alone never resolves
        #     a contradiction: recency is not evidence of supersession.
        ts_a, _ = temporal_status(item_a)
        ts_b, _ = temporal_status(item_b)
        stale_a, stale_b = ts_a in _NON_CURRENT, ts_b in _NON_CURRENT
        if indicates_conflict and stale_a != stale_b:
            old = item_a if stale_a else item_b
            return self._build(
                candidate_id, EvidenceRelationship.TEMPORAL_DIFF, self._confidence_for(results),
                contributing, mismatched_by_field, temporal_signal,
                extra_note=(
                    f"resolved by declared lifecycle: {old.item_id} is "
                    f"{(ts_a if stale_a else ts_b).value.lower()}"
                ),
            )
        if indicates_conflict and ts_a is TemporalStatus.CURRENT and ts_b is TemporalStatus.CURRENT:
            return self._build(
                candidate_id, EvidenceRelationship.GENUINE_CONFLICT,
                max(r.confidence for r in results if r.label is ValidationLabel.CONTRADICTION),
                contributing, mismatched_by_field, temporal_signal,
                extra_note="both sources are declared current",
                item_a=item_a, item_b=item_b,
            )
        if indicates_conflict and temporal_diverges:
            return self._build(
                candidate_id, EvidenceRelationship.UNRESOLVED,
                min(max(r.confidence for r in results if r.label is ValidationLabel.CONTRADICTION), 0.5),
                contributing, mismatched_by_field, temporal_signal,
                extra_note=(
                    "publication dates differ but no source declares supersession; "
                    "recency alone does not resolve a contradiction"
                ),
                item_a=item_a, item_b=item_b,
            )

        if temporal_diverges:
            # Dates differ but nothing disagrees: a benign temporal difference.
            confidence = self._confidence_for(results)
            return self._build(
                candidate_id, EvidenceRelationship.TEMPORAL_DIFF, confidence,
                contributing, mismatched_by_field, temporal_signal,
            )

        # 2. A disagreement was found (CONTRADICTION), but at least one
        #    contextual field (or the publication date) is UNKNOWN, so a
        #    context-based explanation cannot be ruled out. Do not
        #    fabricate certainty by calling this a genuine conflict --
        #    the honest answer is that it is unresolved.
        if indicates_conflict and (any_unknown or temporal_unknown):
            conflict_confidence = max(
                r.confidence for r in results if r.label is ValidationLabel.CONTRADICTION
            )
            unknown_fields = sorted(
                s.field_name for s in applicability_signals if s.label is ApplicabilityLabel.UNKNOWN
            )
            if temporal_unknown:
                unknown_fields.append("date")
            return self._build(
                candidate_id,
                EvidenceRelationship.UNRESOLVED,
                # Confidence in "this is genuinely unresolved" is bounded
                # below the raw conflict confidence: we are confident
                # something disagrees, but not confident it is a *genuine*
                # conflict versus an explainable context difference.
                min(conflict_confidence, 0.5),
                contributing,
                mismatched_by_field,
                temporal_signal,
                extra_note=f"conflict found but context unknown ({', '.join(unknown_fields)})",
                item_a=item_a,
                item_b=item_b,
            )

        # 3. No contextual explanation for a disagreement, and context is
        #    fully known. A CONTRADICTION signal from any stage that ran
        #    is a genuine conflict.
        if indicates_conflict:
            conflict_confidence = max(
                r.confidence for r in results if r.label is ValidationLabel.CONTRADICTION
            )
            return self._build(
                candidate_id,
                EvidenceRelationship.GENUINE_CONFLICT,
                conflict_confidence,
                contributing,
                mismatched_by_field,
                temporal_signal,
                item_a=item_a,
                item_b=item_b,
            )

        # 4. Every stage that ran agrees on support, and no unresolved
        #    unknowns remain anywhere: compatible.
        if indicates_support and not any_unknown and not temporal_unknown:
            return self._build(
                candidate_id,
                EvidenceRelationship.COMPATIBLE,
                min(r.confidence for r in informative),
                contributing,
                mismatched_by_field,
                temporal_signal,
                extra_note=(
                    "no comparable conflicting content"
                    if len(informative) < len(results) else None
                ),
            )

        # 5. Everything else -- insufficient information (UNKNOWN
        #    provenance fields, uncertain/partial validation results, or
        #    NLI unavailable with no other conclusive signal) -- is
        #    UNRESOLVED. Never fabricated as COMPATIBLE or GENUINE_CONFLICT.
        low_confidence = min((r.confidence for r in informative), default=0.0)
        return self._build(
            candidate_id,
            EvidenceRelationship.UNRESOLVED,
            low_confidence if (indicates_uncertain or any_unknown or temporal_unknown) else 0.0,
            contributing,
            mismatched_by_field,
            temporal_signal,
        )

    # -- internal helpers -----------------------------------------------

    def _confidence_for(self, results: list[ValidationResult]) -> float:
        return max((r.confidence for r in results), default=0.0)

    def _source_quality_note(self, item_a: EvidenceItem, item_b: EvidenceItem) -> str | None:
        """Traceability-only note on relative source quality.

        Never used to decide the relationship itself -- only appended to
        the rationale so a downstream module (e.g. Decision) can weigh it
        when choosing how to present conflicting evidence to a user.
        """
        if self._source_quality_scorer is None:
            return None
        score_a = self._source_quality_scorer.score(item_a)
        score_b = self._source_quality_scorer.score(item_b)
        if not score_a.is_known or not score_b.is_known:
            return "source_quality=unknown"
        return f"source_quality=({score_a.quality_score:.2f} vs {score_b.quality_score:.2f})"

    def _build(
        self,
        candidate_id: CandidateId,
        relationship: EvidenceRelationship,
        confidence: float,
        contributing: frozenset[ClaimId],
        mismatched_by_field: Mapping[str, ApplicabilitySignal],
        temporal_signal: TemporalConflictSignal,
        extra_note: str | None = None,
        item_a: EvidenceItem | None = None,
        item_b: EvidenceItem | None = None,
    ) -> EvidenceRelationshipResult:
        parts = [f"relationship={relationship.value}"]
        if mismatched_by_field:
            parts.append("mismatched_fields=" + ",".join(sorted(mismatched_by_field)))
        parts.append(f"temporal={temporal_signal.label.value}")
        if extra_note:
            parts.append(extra_note)
        if item_a is not None and item_b is not None:
            quality_note = self._source_quality_note(item_a, item_b)
            if quality_note:
                parts.append(quality_note)
        return EvidenceRelationshipResult(
            candidate_id=candidate_id,
            relationship=relationship,
            confidence=max(0.0, min(1.0, confidence)),
            contributing_claim_ids=contributing,
            rationale="; ".join(parts),
        )