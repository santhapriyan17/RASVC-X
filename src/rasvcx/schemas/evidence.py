"""EvidenceBundle: the canonical, request-scoped evidence data contract.

EvidenceBundle is the single shared state object that flows through the
entire RASVC-X pipeline. It does not interpret domain semantics -- it is a
typed container with controlled, incremental mutation methods. Claims are
never duplicated: EvidenceItem references claims by ClaimId only, and the
actual Claim objects live exclusively in EvidenceBundle.claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet

from rasvcx.schemas.claims import Claim, ClaimRegistry
from rasvcx.schemas.common import (
    CandidateId,
    ChunkId,
    ClaimId,
    EvidenceItemId,
    PipelineStage,
    QueryId,
    SourceType,
    Unknown,
)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Contextual metadata for a single evidence chunk.

    Every field is either a concrete value or the UNKNOWN sentinel.
    UNKNOWN must never be fabricated into a concrete value, and must never
    be treated as compatible with a concrete value during conflict
    resolution -- it means "insufficient information to rule out conflict."
    """

    source_type: SourceType
    date: str | Unknown
    jurisdiction: str | Unknown
    population: str | Unknown
    dosage_context: str | Unknown


#: Declared lifecycle status of a source document (knowledge-plane metadata).
LIFECYCLE_STATUSES = ("current", "superseded", "historical", "withdrawn")


@dataclass(frozen=True, slots=True)
class SourceLifecycle:
    """Document lifecycle as DECLARED by whoever registered the source.

    Temporal evidence states (CURRENT / SUPERSEDED / HISTORICAL / WITHDRAWN)
    are derived only from these declarations -- never guessed from a
    publication date or from the document text.  None = not declared.

      status          one of LIFECYCLE_STATUSES
      effective_date  ISO date the content takes effect
      superseded_by   doc_id of the document that replaces this one
      supersedes      doc_ids this document replaces
      version         publisher's version label
    """

    status: str | None = None
    effective_date: str | None = None
    superseded_by: str | None = None
    supersedes: tuple[str, ...] = ()
    version: str | None = None

    def __post_init__(self) -> None:
        if self.status is not None and self.status not in LIFECYCLE_STATUSES:
            raise ValueError(
                f"SourceLifecycle.status must be one of {LIFECYCLE_STATUSES}, got {self.status!r}"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status, "effective_date": self.effective_date,
            "superseded_by": self.superseded_by, "supersedes": list(self.supersedes),
            "version": self.version,
        }

    @classmethod
    def from_record(cls, rec: object) -> "SourceLifecycle | None":
        """Parse a record's lifecycle block; None when nothing is declared."""
        if not isinstance(rec, dict):
            return None

        def _s(key: str) -> str | None:
            v = rec.get(key)
            return str(v).strip() or None if v not in (None, "") else None

        raw_sup = rec.get("supersedes") or ()
        supersedes = tuple(str(s).strip() for s in (raw_sup if isinstance(raw_sup, (list, tuple)) else [raw_sup]) if str(s).strip())
        status = _s("status")
        lc = cls(
            status=status.lower() if status else None,
            effective_date=_s("effective_date"), superseded_by=_s("superseded_by"),
            supersedes=supersedes, version=_s("version"),
        )
        return None if lc == cls() else lc


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Where an evidence chunk came from inside the knowledge base.

    Attribution metadata (document identity and location) plus the
    document's declared lifecycle.  Contextual semantics live in
    Provenance.  None means the knowledge base did not record the field;
    it is never fabricated.
    """

    doc_id: str
    title: str | None = None
    source_url: str | None = None
    filename: str | None = None
    heading: str | None = None
    page: int | None = None
    lifecycle: SourceLifecycle | None = None


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """A single retrieved-and-reranked evidence chunk.

    extracted_claim_ids references Claim objects stored in
    EvidenceBundle.claims -- it never holds Claim objects directly.
    """

    item_id: EvidenceItemId
    chunk_id: ChunkId
    text: str
    retrieval_score: float
    provenance: Provenance
    rerank_score: float | None = None
    extracted_claim_ids: FrozenSet[ClaimId] = field(default_factory=frozenset)
    source: SourceRef | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("EvidenceItem.text must be non-empty")


class ConflictLabel(str, Enum):
    """Preliminary label attached to a candidate pair at generation time.

    This is distinct from the final EvidenceRelationship taxonomy resolved
    later in the pipeline; it only records why the pair was flagged as a
    candidate worth validating.
    """

    POTENTIAL_NUMERIC_CONFLICT = "potential_numeric_conflict"
    POTENTIAL_CONTEXTUAL_CONFLICT = "potential_contextual_conflict"
    POTENTIAL_SEMANTIC_CONFLICT = "potential_semantic_conflict"


@dataclass(frozen=True, slots=True)
class CandidatePair:
    """A bounded conflict candidate: two evidence items worth validating."""

    candidate_id: CandidateId
    item_id_a: EvidenceItemId
    item_id_b: EvidenceItemId
    label: ConflictLabel
    priority: float = 0.0


@dataclass(frozen=True, slots=True)
class BundleMetadata:
    """Observability counters and bounded-loop tracking for one request.

    Immutable snapshot; EvidenceBundle produces a new BundleMetadata on each
    update rather than mutating counters in place, so the bundle's
    incremental-mutation methods stay the single point of state change.
    """

    retrieval_calls: int = 0
    nli_calls: int = 0
    targeted_retrieval_used: bool = False
    corrective_attempts_used: int = 0
    elapsed_per_stage: dict[PipelineStage, float] = field(default_factory=dict)

    def with_retrieval_call(self) -> "BundleMetadata":
        return self._replace(retrieval_calls=self.retrieval_calls + 1)

    def with_nli_calls(self, count: int) -> "BundleMetadata":
        return self._replace(nli_calls=self.nli_calls + count)

    def with_targeted_retrieval_used(self) -> "BundleMetadata":
        return self._replace(targeted_retrieval_used=True)

    def with_corrective_attempt(self) -> "BundleMetadata":
        return self._replace(
            corrective_attempts_used=self.corrective_attempts_used + 1
        )

    def with_stage_elapsed(self, stage: PipelineStage, seconds: float) -> "BundleMetadata":
        updated = dict(self.elapsed_per_stage)
        updated[stage] = seconds
        return self._replace(elapsed_per_stage=updated)

    def _replace(self, **changes: object) -> "BundleMetadata":
        current = {
            "retrieval_calls": self.retrieval_calls,
            "nli_calls": self.nli_calls,
            "targeted_retrieval_used": self.targeted_retrieval_used,
            "corrective_attempts_used": self.corrective_attempts_used,
            "elapsed_per_stage": dict(self.elapsed_per_stage),
        }
        current.update(changes)
        return BundleMetadata(**current)  # type: ignore[arg-type]


@dataclass(slots=True)
class EvidenceBundle:
    """Canonical, request-scoped evidence state passed through the pipeline.

    Mutability note: the bundle itself is a mutable container (pipeline
    stages accumulate results into it incrementally), but every field that
    represents "facts about the world" is immutable once inserted:
      - EvidenceItem, Claim, CandidatePair are frozen dataclasses.
      - Claims are stored exactly once (enforced by ClaimRegistry).
      - validation_results and resolution accumulate additively; existing
        entries are never silently overwritten.

    There is no global state: every EvidenceBundle instance is scoped to a
    single request and must not be shared across concurrent requests.
    """

    query_id: QueryId
    risk_profile: "object"  # rasvcx.schemas.query.RiskProfile; see note below
    evidence_items: dict[EvidenceItemId, EvidenceItem] = field(default_factory=dict)
    claims: ClaimRegistry = field(default_factory=ClaimRegistry)
    conflict_candidates: list[CandidatePair] = field(default_factory=list)
    validation_results: dict[CandidateId, "object"] = field(default_factory=dict)
    resolution: list["object"] = field(default_factory=list)
    metadata: BundleMetadata = field(default_factory=BundleMetadata)
    # Observability only: what the retrieval stage actually executed for this
    # request (per-retriever hit counts and timings, KB version). Written by
    # the retrieval bridge; never read by validation or decision logic.
    retrieval_trace: dict[str, object] = field(default_factory=dict)

    # -- evidence item insertion -------------------------------------------------

    def add_evidence_item(self, item: EvidenceItem) -> None:
        if item.item_id in self.evidence_items:
            raise ValueError(
                f"EvidenceItem {item.item_id!r} already present in bundle"
            )
        for claim_id in item.extracted_claim_ids:
            if claim_id not in self.claims:
                raise ValueError(
                    f"EvidenceItem {item.item_id!r} references unregistered "
                    f"claim_id {claim_id!r}; register the Claim first"
                )
        self.evidence_items[item.item_id] = item

    # -- claim registration --------------------------------------------------

    def register_claim(self, claim: Claim) -> None:
        self.claims.register(claim)

    # -- conflict candidates ---------------------------------------------------

    def add_conflict_candidate(self, candidate: CandidatePair, max_candidates: int) -> None:
        if len(self.conflict_candidates) >= max_candidates:
            raise ValueError(
                f"conflict_candidates bound exceeded: max_candidates={max_candidates}"
            )
        for existing in self.conflict_candidates:
            if existing.candidate_id == candidate.candidate_id:
                raise ValueError(
                    f"CandidatePair {candidate.candidate_id!r} already present"
                )
        self.conflict_candidates.append(candidate)

    # -- validation results ---------------------------------------------------

    def add_validation_result(self, candidate_id: CandidateId, result: "object") -> None:
        if candidate_id in self.validation_results:
            raise ValueError(
                f"ValidationResult for {candidate_id!r} already recorded; "
                f"validation results accumulate, they do not overwrite"
            )
        self.validation_results[candidate_id] = result

    # -- resolution ---------------------------------------------------

    def add_resolution(self, relationship: "object") -> None:
        self.resolution.append(relationship)

    # -- metadata ---------------------------------------------------

    def record_retrieval_call(self) -> None:
        self.metadata = self.metadata.with_retrieval_call()

    def record_nli_calls(self, count: int) -> None:
        if count < 0:
            raise ValueError("nli call count must be non-negative")
        self.metadata = self.metadata.with_nli_calls(count)

    def record_targeted_retrieval_used(self) -> None:
        self.metadata = self.metadata.with_targeted_retrieval_used()

    def record_corrective_attempt(self) -> None:
        self.metadata = self.metadata.with_corrective_attempt()

    def record_stage_elapsed(self, stage: PipelineStage, seconds: float) -> None:
        self.metadata = self.metadata.with_stage_elapsed(stage, seconds)