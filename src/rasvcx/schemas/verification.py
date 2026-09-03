"""Verification contract (Module 9).

Owned exclusively by verification/*.py -- this module only defines the
immutable public contract; it contains no verification logic itself (see
architecture rule: schemas contain no business decisions, mirrored from
schemas/decision.py and schemas/validation.py).

Module 9 answers a different question than Module 8:
  M8: "How do retrieved evidence items relate to each other?"
  M9: "Does the generated answer accurately represent the validated
       evidence?"

SupportLabel is a distinct taxonomy from M8's ValidationLabel -- it is
never collapsed into it, and UNSUPPORTED is never conflated with
CONTRADICTED (see module docstring in verification/deterministic_check.py
for the full rationale).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import NewType

from rasvcx.schemas.common import EvidenceItemId

GeneratedClaimId = NewType("GeneratedClaimId", str)
"""Distinct ID space from M7's ClaimId: these identify propositions found
in the *generated answer*, not claims extracted from *evidence*."""


class SupportLabel(str, Enum):
    """Per-claim verification verdict (Section 14).

    UNSUPPORTED and CONTRADICTED are never conflated: UNSUPPORTED means no
    evidence was found either confirming or denying the claim; CONTRADICTED
    means specific evidence establishes an incompatible proposition.
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    UNCERTAIN = "uncertain"
    NOT_VERIFIABLE = "not_verifiable"


class CitationStatus(str, Enum):
    """Outcome of aligning a claim with its cited evidence (Section 13)."""

    CORRECT = "correct"
    WEAK = "weak"
    INCORRECT = "incorrect"
    MISSING = "missing"
    CONTRADICTORY = "contradictory"
    ORPHAN = "orphan"
    AMBIGUOUS = "ambiguous"


class VerificationReasonCode(str, Enum):
    """Stable, machine-readable reason codes (Section 46).

    Human-readable rationale strings are still produced, but callers
    (evaluation tooling, Module 10) must be able to branch on this enum
    rather than parsing free text.
    """

    NO_EVIDENCE = "no_evidence"
    DIRECT_EVIDENCE_SUPPORT = "direct_evidence_support"
    PARTIAL_EVIDENCE = "partial_evidence"
    EVIDENCE_CONTRADICTION = "evidence_contradiction"
    NUMERIC_MISMATCH = "numeric_mismatch"
    TEMPORAL_MISMATCH = "temporal_mismatch"
    JURISDICTION_MISMATCH = "jurisdiction_mismatch"
    POPULATION_MISMATCH = "population_mismatch"
    DOSAGE_MISMATCH = "dosage_mismatch"
    QUALIFIER_STRENGTHENING = "qualifier_strengthening"
    NEGATION_MISMATCH = "negation_mismatch"
    COMPARATIVE_REVERSAL = "comparative_reversal"
    CITATION_MISSING = "citation_missing"
    CITATION_INVALID = "citation_invalid"
    CITATION_MISMATCH = "citation_mismatch"
    NLI_UNAVAILABLE = "nli_unavailable"
    NLI_UNCERTAIN = "nli_uncertain"
    EVIDENCE_CONFLICT = "evidence_conflict"
    PROVENANCE_UNKNOWN = "provenance_unknown"
    BUDGET_EXHAUSTED = "budget_exhausted"
    EMPTY_ANSWER = "empty_answer"


class VerificationStage(str, Enum):
    """Which layer of the deterministic-first pipeline produced a verdict."""

    DETERMINISTIC = "deterministic"
    PROVENANCE = "provenance"
    SELECTIVE_NLI = "selective_nli"


class AnswerVerdict(str, Enum):
    """Answer-level aggregate verdict (Section 27)."""

    VERIFIED = "verified"
    PARTIALLY_VERIFIED = "partially_verified"
    UNVERIFIED = "unverified"
    UNSAFE = "unsafe"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True, slots=True)
class GeneratedClaim:
    """One atomic proposition extracted from a generated answer.

    Sentence-level atomicity, matching M7's ``ClaimExtractor`` design
    decision (see claims/extractor.py docstring) for consistency and to
    avoid destroying qualifier context via unreliable sub-sentence
    splitting. Full compositional (sub-sentence) decomposition is a
    documented limitation, not silently attempted.
    """

    claim_id: GeneratedClaimId
    text: str
    normalized_text: str
    span: tuple[int, int]
    """Character offsets into the original generated_answer string."""
    cited_item_ids: frozenset[EvidenceItemId] = field(default_factory=frozenset)
    unresolved_citation_tokens: frozenset[str] = field(default_factory=frozenset)
    """Raw bracket tokens (e.g. "E999") that carried citation syntax but
    did not resolve to any known EvidenceItemId. Kept distinct from simply
    having no citation at all, so downstream citation verification can
    tell MISSING (Section 13) apart from INCORRECT."""
    is_safety_critical: bool = False

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("GeneratedClaim.text must be non-empty")


@dataclass(frozen=True, slots=True)
class CitationResult:
    """Result of verifying one claim's citation(s) against evidence."""

    claim_id: GeneratedClaimId
    status: CitationStatus
    cited_item_ids: frozenset[EvidenceItemId]
    rationale: str


@dataclass(frozen=True, slots=True)
class ClaimVerificationResult:
    """Full, traceable verification outcome for one generated claim.

    Answers Section 47's explainability requirement: which claim, against
    which evidence, at which stage, with what confidence, and why.
    """

    claim_id: GeneratedClaimId
    label: SupportLabel
    confidence: float
    stage: VerificationStage
    reason_code: VerificationReasonCode
    rationale: str
    supporting_item_ids: frozenset[EvidenceItemId] = field(default_factory=frozenset)
    contradicting_item_ids: frozenset[EvidenceItemId] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"ClaimVerificationResult.confidence must be in [0, 1], got {self.confidence}"
            )


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    """Immutable, auditable result of running Module 9 on one answer.

    Retains full per-claim and per-citation traceability (Section 27: "Do
    not hide individual failed claims behind an aggregate score") -- the
    answer-level verdict is a derived summary, never the sole output.
    """

    answer_verdict: AnswerVerdict
    overall_confidence: float
    claim_results: tuple[ClaimVerificationResult, ...]
    citation_results: tuple[CitationResult, ...]
    orphan_citation_ids: frozenset[EvidenceItemId] = field(default_factory=frozenset)
    semantic_verification_calls: int = 0
    safety_critical_failure_count: int = 0
    budget_exhausted: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.overall_confidence <= 1.0:
            raise ValueError(
                f"VerificationSummary.overall_confidence must be in [0, 1], "
                f"got {self.overall_confidence}"
            )
        if self.semantic_verification_calls < 0:
            raise ValueError("VerificationSummary.semantic_verification_calls must be >= 0")

    @property
    def supported_count(self) -> int:
        return sum(1 for r in self.claim_results if r.label is SupportLabel.SUPPORTED)

    @property
    def contradicted_count(self) -> int:
        return sum(1 for r in self.claim_results if r.label is SupportLabel.CONTRADICTED)

    @property
    def unsupported_count(self) -> int:
        return sum(1 for r in self.claim_results if r.label is SupportLabel.UNSUPPORTED)

    @property
    def partial_count(self) -> int:
        return sum(1 for r in self.claim_results if r.label is SupportLabel.PARTIALLY_SUPPORTED)

    @property
    def uncertain_count(self) -> int:
        return sum(
            1
            for r in self.claim_results
            if r.label in (SupportLabel.UNCERTAIN, SupportLabel.NOT_VERIFIABLE)
        )