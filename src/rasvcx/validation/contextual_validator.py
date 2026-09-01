"""Contextual validation (Section 10).

Wraps the existing Module 6 (provenance) public API -- specifically
``compare_applicability`` and ``detect_temporal_conflict`` from
``rasvcx.provenance.context_extractor`` -- to decide whether two evidence
items' *provenance context* (population, jurisdiction, dosage_context,
publication date) is compatible enough for their claims to be compared at
face value.

This module never duplicates M6's comparison/parsing logic; it only
interprets M6's outputs into a ValidationResult for the M8 pipeline. It
also never converts UNKNOWN into a match: any UNKNOWN applicability field
downgrades the result to UNCERTAIN, never SUPPORTED.
"""

from __future__ import annotations

from rasvcx.provenance.context_extractor import (
    ApplicabilityLabel,
    TemporalConflictLabel,
    compare_applicability,
    detect_temporal_conflict,
)
from rasvcx.schemas.common import CandidateId
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItemId
from rasvcx.schemas.validation import ValidationLabel, ValidationResult, ValidationStage


class ContextualValidator:
    """Validates a CandidatePair's provenance-context compatibility.

    Complexity: O(1) per candidate (three field comparisons + one date
    comparison, all delegated to M6).
    """

    def __init__(self, temporal_divergence_threshold_days: int = 730) -> None:
        self._temporal_divergence_threshold_days = temporal_divergence_threshold_days

    def validate(
        self,
        bundle: EvidenceBundle,
        candidate_id: CandidateId,
        item_id_a: EvidenceItemId,
        item_id_b: EvidenceItemId,
    ) -> ValidationResult:
        item_a = bundle.evidence_items[item_id_a]
        item_b = bundle.evidence_items[item_id_b]

        applicability_signals = compare_applicability(item_a.provenance, item_b.provenance)
        temporal_signal = detect_temporal_conflict(
            item_a.provenance,
            item_b.provenance,
            divergence_threshold_days=self._temporal_divergence_threshold_days,
        )

        mismatched = [s for s in applicability_signals if s.label is ApplicabilityLabel.MISMATCH]
        unknown = [s for s in applicability_signals if s.label is ApplicabilityLabel.UNKNOWN]

        temporal_diverges = temporal_signal.label is TemporalConflictLabel.TEMPORAL_DIVERGENCE
        temporal_unknown = temporal_signal.label is TemporalConflictLabel.UNKNOWN

        reasons = [s.reason for s in mismatched]
        if temporal_diverges:
            reasons.append(temporal_signal.reason)

        if mismatched or temporal_diverges:
            # A known, concrete difference in context: this is not
            # necessarily a genuine contradiction (it may fully explain an
            # apparent numeric conflict as e.g. a population difference),
            # but it is also not a clean confirmation. Evidence resolution
            # (resolution.py) re-derives the same M6 signals to choose the
            # specific EvidenceRelationship (POPULATION_DIFF, etc.).
            return ValidationResult(
                candidate_id=candidate_id,
                stage=ValidationStage.CONTEXTUAL,
                label=ValidationLabel.PARTIAL,
                confidence=0.7,
                rationale="; ".join(reasons) or "Context mismatch detected",
            )

        if unknown or temporal_unknown:
            # UNKNOWN must never be silently treated as compatible.
            unknown_fields = [s.field_name for s in unknown]
            if temporal_unknown:
                unknown_fields.append("date")
            return ValidationResult(
                candidate_id=candidate_id,
                stage=ValidationStage.CONTEXTUAL,
                label=ValidationLabel.UNCERTAIN,
                confidence=0.3,
                rationale=(
                    f"Insufficient provenance to rule out context conflict: "
                    f"{', '.join(unknown_fields)} unknown"
                ),
            )

        # All known fields match and dates (if comparable) do not diverge.
        return ValidationResult(
            candidate_id=candidate_id,
            stage=ValidationStage.CONTEXTUAL,
            label=ValidationLabel.SUPPORTED,
            confidence=0.8,
            rationale="Provenance context is compatible across all known fields",
        )