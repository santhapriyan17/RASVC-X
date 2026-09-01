from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from rasvcx.schemas.common import EvidenceItemId, SourceType, UNKNOWN
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem
from rasvcx.schemas.query import QueryRequest, RiskProfile

from rasvcx.provenance.source_quality import SourceQualityScorer
from rasvcx.provenance.context_extractor import ApplicabilityLabel, ContextExtractor


class CompatibilityVerdict(Enum):
    KNOWN_MATCH = "known_match"
    KNOWN_MISMATCH = "known_mismatch"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ItemProvenanceAnalysis:
    item_id: EvidenceItemId
    source_type: SourceType | None
    is_known: bool
    quality_score: float
    temporal: CompatibilityVerdict
    jurisdiction: CompatibilityVerdict
    population: CompatibilityVerdict
    dosage_context: CompatibilityVerdict

    def __post_init__(self) -> None:
        if not (0.0 <= self.quality_score <= 1.0):
            raise ValueError(
                f"quality_score out of bounds for {self.item_id!r}: {self.quality_score}"
            )
        if self.is_known and self.source_type is None:
            raise ValueError(
                f"Contract violation for {self.item_id!r}: is_known=True but source_type is None"
            )
        if not self.is_known and self.source_type is not None:
            raise ValueError(
                f"Contract violation for {self.item_id!r}: is_known=False but source_type is {self.source_type!r}"
            )

    @property
    def unknown_field_count(self) -> int:
        return sum(
            1
            for v in (self.temporal, self.jurisdiction, self.population, self.dosage_context)
            if v is CompatibilityVerdict.UNKNOWN
        )

    @property
    def mismatch_field_count(self) -> int:
        return sum(
            1
            for v in (self.temporal, self.jurisdiction, self.population, self.dosage_context)
            if v is CompatibilityVerdict.KNOWN_MISMATCH
        )


@dataclass(frozen=True, slots=True)
class ProvenanceAnalysisResult:
    per_item: Mapping[EvidenceItemId, ItemProvenanceAnalysis]
    unique_source_count: int
    unknown_provenance_ratio: float
    mismatch_present: bool
    strict_context_required: bool

    def __post_init__(self) -> None:
        if not (0.0 <= self.unknown_provenance_ratio <= 1.0):
            raise ValueError(
                f"unknown_provenance_ratio out of bounds: {self.unknown_provenance_ratio}"
            )
        if self.unique_source_count < 0:
            raise ValueError(f"unique_source_count cannot be negative: {self.unique_source_count}")


def _source_identity_key(item: EvidenceItem) -> tuple:
    provenance = item.provenance
    source_id = getattr(provenance, "source_id", UNKNOWN)
    if source_id is UNKNOWN:
        return (provenance.source_type, UNKNOWN)
    return (provenance.source_type, source_id)


_APPLICABILITY_TO_VERDICT: dict[ApplicabilityLabel, CompatibilityVerdict] = {
    ApplicabilityLabel.MATCH: CompatibilityVerdict.KNOWN_MATCH,
    ApplicabilityLabel.MISMATCH: CompatibilityVerdict.KNOWN_MISMATCH,
    ApplicabilityLabel.UNKNOWN: CompatibilityVerdict.UNKNOWN,
    ApplicabilityLabel.NOT_COMPARED: CompatibilityVerdict.UNKNOWN,
}


def _coerce_verdict(value: object) -> CompatibilityVerdict:
    if isinstance(value, CompatibilityVerdict):
        return value
    if isinstance(value, ApplicabilityLabel):
        return _APPLICABILITY_TO_VERDICT[value]
    if value is UNKNOWN:
        return CompatibilityVerdict.UNKNOWN
    raise ValueError(f"Unrecognized compatibility verdict value: {value!r}")


class ProvenanceAnalyzer:
    def __init__(
        self,
        source_quality_scorer: SourceQualityScorer,
        context_extractor: ContextExtractor,
    ) -> None:
        self._source_quality_scorer = source_quality_scorer
        self._context_extractor = context_extractor

    def analyze(
        self,
        bundle: EvidenceBundle,
        query: QueryRequest,
        risk_profile: RiskProfile,
    ) -> ProvenanceAnalysisResult:
        import time

        start = time.perf_counter()

        per_item: dict[EvidenceItemId, ItemProvenanceAnalysis] = {}
        for item_id, item in bundle.evidence_items.items():
            per_item[item_id] = self._analyze_item(item, query)

        unique_source_count = len(
            {_source_identity_key(item) for item in bundle.evidence_items.values()}
        )

        total_items = len(per_item)
        if total_items == 0:
            unknown_provenance_ratio = 0.0
            mismatch_present = False
        else:
            unknown_count = sum(
                1
                for analysis in per_item.values()
                if analysis.unknown_field_count > 0 or not analysis.is_known
            )
            unknown_provenance_ratio = unknown_count / total_items
            mismatch_present = any(
                analysis.mismatch_field_count > 0 for analysis in per_item.values()
            )

        result = ProvenanceAnalysisResult(
            per_item=per_item,
            unique_source_count=unique_source_count,
            unknown_provenance_ratio=unknown_provenance_ratio,
            mismatch_present=mismatch_present,
            strict_context_required=bool(risk_profile.safety_floor_forced),
        )

        elapsed = time.perf_counter() - start
        bundle.record_stage_elapsed("provenance_context", elapsed)

        return result

    def _analyze_item(self, item: EvidenceItem, query: QueryRequest) -> ItemProvenanceAnalysis:
        quality_result = self._source_quality_scorer.score(item)
        context_result = self._context_extractor.extract(item, query)

        return ItemProvenanceAnalysis(
            item_id=item.item_id,
            source_type=quality_result.source_type,
            is_known=quality_result.is_known,
            quality_score=quality_result.quality_score,
            temporal=_coerce_verdict(context_result.temporal),
            jurisdiction=_coerce_verdict(context_result.jurisdiction),
            population=_coerce_verdict(context_result.population),
            dosage_context=_coerce_verdict(context_result.dosage_context),
        )