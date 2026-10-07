"""Answer-level verdict aggregation (Sections 27-29, 63-64 of master prompt).

Combines per-claim ``ClaimVerificationResult``s into a single, auditable
``VerificationSummary``. The aggregation policy is deterministic,
documented, and configurable via ``VerificationConfig`` -- no threshold is
hardcoded inline (Section 28: "Do NOT invent arbitrary thresholds. Put
thresholds into configuration objects.").

Per-claim traceability is never discarded: the answer-level verdict is a
derived summary over ``claim_results``, which the summary always retains
in full (Section 27).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rasvcx.schemas.common import EvidenceItemId, EvidenceRelationship
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.verification import (
    VerificationReasonCode,
    AnswerVerdict,
    CitationResult,
    ClaimVerificationResult,
    SupportLabel,
    VerificationSummary,
)
from rasvcx.validation.verified_context import ValidationSummary


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    """Tunable thresholds for answer-level aggregation. All defaults are
    documented here rather than hardcoded in aggregation logic.
    """

    max_claims: int = 200
    """Section 38 budget: claims beyond this index are still extracted and
    reported, but excluded from semantic escalation to bound latency."""

    max_nli_calls_per_answer: int = 20
    """Section 38 budget: hard cap on SelectiveSemanticVerifier invocations
    for a single ``verify()`` call."""

    max_fallback_evidence_candidates: int = 5
    """When a claim has no valid citation, at most this many lexically
    related evidence items are offered to semantic verification."""

    unsupported_fraction_for_partial: float = 0.2
    """If the fraction of UNSUPPORTED/UNCERTAIN/NOT_VERIFIABLE claims meets
    or exceeds this threshold (and there is no contradiction), the answer
    is downgraded from VERIFIED to PARTIALLY_VERIFIED."""

    unsupported_fraction_for_unverified: float = 0.5
    """Above this fraction, the answer is UNVERIFIED rather than merely
    PARTIALLY_VERIFIED."""


def _label_of(result: ClaimVerificationResult) -> SupportLabel:
    return result.label


class VerdictAggregator:
    """Deterministic aggregation of claim/citation results into a
    ``VerificationSummary``.

    Policy (Section 28), evaluated in order:
      1. Any safety-critical claim CONTRADICTED => UNSAFE. A contradicted
         safety-critical claim is never downgraded by other passing
         claims (Section 29: "uncertainty must not silently become
         support"; the inverse holds too -- a genuine safety contradiction
         is never diluted by an average).
      2. No claims extracted at all => INSUFFICIENT_EVIDENCE (distinct
         from an empty *answer*, which the caller handles before claim
         extraction even runs -- see verification/__init__.py).
      3. Any (non-safety-critical) claim CONTRADICTED => UNVERIFIED.
      4. Fraction of UNSUPPORTED/UNCERTAIN/NOT_VERIFIABLE claims >=
         ``unsupported_fraction_for_unverified`` => UNVERIFIED.
      5. That fraction >= ``unsupported_fraction_for_partial`` => 
         PARTIALLY_VERIFIED.
      6. Any PARTIALLY_SUPPORTED claim present => PARTIALLY_VERIFIED.
      7. Otherwise (every claim SUPPORTED) => VERIFIED.
    """

    def __init__(self, config: VerificationConfig | None = None) -> None:
        self._config = config or VerificationConfig()

    def aggregate(
        self,
        claim_results: list[ClaimVerificationResult],
        citation_results: list[CitationResult],
        orphan_citation_ids: frozenset[EvidenceItemId],
        semantic_verification_calls: int,
        budget_exhausted: bool,
        claim_is_safety_critical: dict[str, bool],
    ) -> VerificationSummary:
        if not claim_results:
            return VerificationSummary(
                answer_verdict=AnswerVerdict.INSUFFICIENT_EVIDENCE,
                overall_confidence=0.0,
                claim_results=tuple(claim_results),
                citation_results=tuple(citation_results),
                orphan_citation_ids=orphan_citation_ids,
                semantic_verification_calls=semantic_verification_calls,
                safety_critical_failure_count=0,
                budget_exhausted=budget_exhausted,
            )

        safety_critical_failures = [
            r
            for r in claim_results
            if claim_is_safety_critical.get(r.claim_id, False)
            and _label_of(r) is SupportLabel.CONTRADICTED
        ]

        total = len(claim_results)
        contradicted = [r for r in claim_results if _label_of(r) is SupportLabel.CONTRADICTED]
        weak = [
            r
            for r in claim_results
            if _label_of(r)
            in (SupportLabel.UNSUPPORTED, SupportLabel.UNCERTAIN, SupportLabel.NOT_VERIFIABLE)
        ]
        partial = [r for r in claim_results if _label_of(r) is SupportLabel.PARTIALLY_SUPPORTED]
        weak_fraction = len(weak) / total

        if safety_critical_failures:
            verdict = AnswerVerdict.UNSAFE
        elif contradicted:
            verdict = AnswerVerdict.UNVERIFIED
        elif weak_fraction >= self._config.unsupported_fraction_for_unverified:
            verdict = AnswerVerdict.UNVERIFIED
        elif weak_fraction >= self._config.unsupported_fraction_for_partial:
            verdict = AnswerVerdict.PARTIALLY_VERIFIED
        elif partial:
            verdict = AnswerVerdict.PARTIALLY_VERIFIED
        else:
            verdict = AnswerVerdict.VERIFIED

        overall_confidence = self._compute_overall_confidence(claim_results, verdict)

        return VerificationSummary(
            answer_verdict=verdict,
            overall_confidence=overall_confidence,
            claim_results=tuple(claim_results),
            citation_results=tuple(citation_results),
            orphan_citation_ids=orphan_citation_ids,
            semantic_verification_calls=semantic_verification_calls,
            safety_critical_failure_count=len(safety_critical_failures),
            budget_exhausted=budget_exhausted,
        )

    def _compute_overall_confidence(
        self, claim_results: list[ClaimVerificationResult], verdict: AnswerVerdict
    ) -> float:
        """Mean per-claim confidence, weighted down for any answer-level
        verdict short of VERIFIED.

        This is an explicit, simple, documented policy -- not a claim of
        statistically precise confidence (Section 25: "Do not manufacture
        mathematically precise confidence from arbitrary heuristics").
        UNSAFE/UNVERIFIED answers are deliberately capped low regardless of
        individual claim confidences, so a few high-confidence supported
        claims can never mask a genuine contradiction (Invariant 6: "High
        confidence != truth").
        """
        mean_confidence = sum(r.confidence for r in claim_results) / len(claim_results)
        if verdict in (AnswerVerdict.UNSAFE, AnswerVerdict.UNVERIFIED):
            return min(mean_confidence, 0.3)
        if verdict is AnswerVerdict.PARTIALLY_VERIFIED:
            return min(mean_confidence, 0.7)
        return mean_confidence


def _is_newer(bundle: EvidenceBundle, newer: EvidenceItemId, older: EvidenceItemId) -> bool:
    """True when `older` is DECLARED non-current (superseded / withdrawn /
    historical) and `newer` is not.

    Publication dates are not compared: recency is not evidence that a
    source was replaced (see resolution.EvidenceResolver rule 1b).
    """
    from rasvcx.provenance.evidence_roles import STALE_OR_PENDING as stale, temporal_status

    return (
        temporal_status(bundle.evidence_items[older])[0] in stale
        and temporal_status(bundle.evidence_items[newer])[0] not in stale
    )


def _resolve_superseded_sources(
    claim_results: list[ClaimVerificationResult],
    validation_summary: ValidationSummary,
    bundle: EvidenceBundle,
    candidate_items: dict[str, tuple[EvidenceItemId, EvidenceItemId]],
) -> list[ClaimVerificationResult]:
    """A claim that follows the CURRENT source of a superseded pair is supported.

    The deterministic verifier marks a claim CONTRADICTED when any cited
    source states a different value, even if another cited source states
    exactly the claim's value.  When M8 resolved that pair of sources as a
    temporal difference (same statement, publication dates far apart) and
    the supporting source is the more recent one, the "contradiction" is an
    old recommendation that has been replaced: the claim reports the
    current one and is SUPPORTED.

    Deliberately narrow:
      - only temporal-diff, and only when the SUPPORTING source is newer --
        a claim that follows the older source stays CONTRADICTED;
      - every contradicting source must be explained this way;
      - population / jurisdiction / dosage differences are not handled
        here: which side applies depends on the question, not on a date.
    """
    temporal_pairs: set[frozenset[EvidenceItemId]] = set()
    for resolution in validation_summary.resolutions:
        if resolution.relationship is EvidenceRelationship.TEMPORAL_DIFF:
            pair = candidate_items.get(resolution.candidate_id)
            if pair is not None:
                temporal_pairs.add(frozenset(pair))
    if not temporal_pairs:
        return claim_results

    out: list[ClaimVerificationResult] = []
    for result in claim_results:
        explained = (
            result.label is SupportLabel.CONTRADICTED
            and result.supporting_item_ids
            and result.contradicting_item_ids
            and all(
                any(
                    frozenset((sup, con)) in temporal_pairs and _is_newer(bundle, sup, con)
                    for sup in result.supporting_item_ids
                )
                for con in result.contradicting_item_ids
            )
        )
        if not explained:
            out.append(result)
            continue
        older = ", ".join(sorted(str(i) for i in result.contradicting_item_ids))
        out.append(
            ClaimVerificationResult(
                claim_id=result.claim_id,
                label=SupportLabel.SUPPORTED,
                confidence=min(result.confidence, 0.8),
                stage=result.stage,
                reason_code=VerificationReasonCode.DIRECT_EVIDENCE_SUPPORT,
                rationale=(
                    f"Supported by the current source; the differing value in {older} "
                    f"is from a source declared superseded/withdrawn/historical (temporal-diff)"
                ),
                supporting_item_ids=result.supporting_item_ids,
                contradicting_item_ids=frozenset(),
            )
        )
    return out


_HISTORICAL_MARKER_RE = re.compile(
    r"\b(?:earlier|previous(?:ly)?|former(?:ly)?|prior|no\s+longer|used\s+to|in\s+the\s+past|"
    r"historical(?:ly)?|withdrawn|superseded|replaced|discontinued|retired|obsolete)\b",
    re.IGNORECASE,
)
_PRESENT_ASSERTION_RE = re.compile(
    r"\b(?:current(?:ly)?|now|today|still|at\s+present|presently|is\s+recommended|"
    r"are\s+recommended|should\s+be|must\s+be)\b",
    re.IGNORECASE,
)


def resolve_historical_statements(
    claim_results: list[ClaimVerificationResult],
    claims: list,
    bundle: EvidenceBundle,
) -> list[ClaimVerificationResult]:
    """A claim that REPORTS superseded content as history is not a
    contradiction of the current evidence.

    Example: "An earlier dose of 20 mg weekly was previously used, but that
    schedule has been withdrawn [E1, E2]" -- E2 (declared withdrawn) says
    20 mg weekly, E1 (current) says 10 mg.  The numeric check marks it
    CONTRADICTED although it is a faithful statement about the past.

    Re-labelled SUPPORTED (reason HISTORICAL_STATEMENT) only when ALL hold:
      - the claim text carries an explicit past/withdrawn marker and makes
        no present-tense or recommendation assertion;
      - it has supporting evidence, and EVERY supporting item is declared
        non-current (withdrawn / superseded / historical / future);
      - it has contradicting evidence, and EVERY contradicting item is NOT
        declared non-current.
    Otherwise the result is unchanged -- a claim asserting the old value as
    the current one stays CONTRADICTED.  Lifecycle comes only from KB
    metadata; nothing here reads dates out of text.
    """
    from rasvcx.provenance.evidence_roles import STALE_OR_PENDING, temporal_status

    texts = {str(getattr(c, "claim_id", "")): getattr(c, "text", "") for c in claims}
    items = bundle.evidence_items

    def stale(iid: EvidenceItemId) -> bool | None:
        item = items.get(iid)
        return None if item is None else temporal_status(item)[0] in STALE_OR_PENDING

    out: list[ClaimVerificationResult] = []
    for r in claim_results:
        text = texts.get(str(r.claim_id), "")
        if (
            r.label is SupportLabel.CONTRADICTED
            and r.supporting_item_ids and r.contradicting_item_ids
            and _HISTORICAL_MARKER_RE.search(text)
            and not _PRESENT_ASSERTION_RE.search(text)
            and all(stale(i) is True for i in r.supporting_item_ids)
            and all(stale(i) is False for i in r.contradicting_item_ids)
        ):
            sup = ", ".join(sorted(str(i) for i in r.supporting_item_ids))
            con = ", ".join(sorted(str(i) for i in r.contradicting_item_ids))
            out.append(ClaimVerificationResult(
                claim_id=r.claim_id, label=SupportLabel.SUPPORTED,
                confidence=min(r.confidence, 0.8), stage=r.stage,
                reason_code=VerificationReasonCode.HISTORICAL_STATEMENT,
                rationale=(
                    f"Historical statement: matches {sup} (declared non-current) and is "
                    f"reported as past; current evidence {con} differs as expected"
                ),
                supporting_item_ids=r.supporting_item_ids,
                contradicting_item_ids=frozenset(),
            ))
        else:
            out.append(r)
    return out


def apply_m8_conflict_context(
    claim_results: list[ClaimVerificationResult],
    validation_summary: ValidationSummary | None,
    bundle: EvidenceBundle | None = None,
) -> list[ClaimVerificationResult]:
    """Integration point with Module 8 (Section 50).

    If an evidence item a claim relied on was itself flagged by M8 as part
    of an unresolved or genuine conflict between evidence items, that
    context is surfaced in the claim's rationale and its confidence is
    capped -- this never changes the SupportLabel itself (M8's evidence-vs-
    evidence conflict is a different question from M9's claim-vs-evidence
    verification; M9 does not silently re-decide M8's question), but a
    claim resting on contested evidence should not report undiluted
    confidence.

    ``EvidenceRelationshipResult`` (M8) carries only a ``candidate_id``, not
    the underlying evidence item ids directly -- those live on the
    ``CandidatePair`` objects in ``bundle.conflict_candidates``. This
    function cross-references the two, so ``bundle`` must be the same
    bundle M8's ``ValidationPipeline`` ran against. If ``bundle`` is not
    supplied, this is a documented no-op (rather than a fabricated
    guess at which items were involved).
    """
    if validation_summary is None or bundle is None:
        return claim_results

    candidate_items: dict[str, tuple[EvidenceItemId, EvidenceItemId]] = {
        c.candidate_id: (c.item_id_a, c.item_id_b) for c in bundle.conflict_candidates
    }

    claim_results = _resolve_superseded_sources(
        claim_results, validation_summary, bundle, candidate_items
    )

    conflicted_items: set[EvidenceItemId] = set()
    for resolution in validation_summary.resolutions:
        if resolution.relationship in (
            EvidenceRelationship.GENUINE_CONFLICT,
            EvidenceRelationship.UNRESOLVED,
        ):
            pair = candidate_items.get(resolution.candidate_id)
            if pair is not None:
                conflicted_items.update(pair)

    if not conflicted_items:
        return claim_results

    adjusted: list[ClaimVerificationResult] = []
    for result in claim_results:
        if result.supporting_item_ids & conflicted_items or result.contradicting_item_ids & conflicted_items:
            adjusted.append(
                ClaimVerificationResult(
                    claim_id=result.claim_id,
                    label=result.label,
                    confidence=min(result.confidence, 0.5),
                    stage=result.stage,
                    reason_code=result.reason_code,
                    rationale=result.rationale + "; evidence involved in an unresolved M8 conflict",
                    supporting_item_ids=result.supporting_item_ids,
                    contradicting_item_ids=result.contradicting_item_ids,
                )
            )
        else:
            adjusted.append(result)
    return adjusted