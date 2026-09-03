"""Deterministic-first post-generation verification (Sections 8, 14-22, 85-87
of the M9 master prompt; negation/qualifier/comparative/temporal guidance
from the M9 continuation prompt).

Runs before any semantic/NLI escalation. No LLM, no network, no model
initialization. Reproducible, explainable, low latency -- mirrors M8's
``validation/deterministic.py`` design philosophy exactly, but answers a
different question: not "do these two evidence items agree with each
other" (M8), but "does this specific generated claim accurately represent
its cited/aligned evidence" (M9).

Reuse, not duplication: numeric extraction and significant-token
tokenization are imported directly from ``rasvcx.validation.deterministic``
(M8) rather than re-implemented.

Hard invariants enforced here (Section 41 of the master prompt):
  - Missing evidence => UNSUPPORTED, never SUPPORTED.
  - Silence (evidence simply doesn't mention something) => UNSUPPORTED or
    UNCERTAIN, never CONTRADICTED (Section 15's "do not mark contradiction
    merely because evidence is silent").
  - A negation/qualifier/comparative/numeric mismatch always overrides a
    same-claim lexical overlap: no accidental "supported" from shared
    words alone (this file never returns SUPPORTED on lexical similarity
    once a genuine mismatch signal is found).
  - Ambiguous negation is never guessed at: if the deterministic negation
    check cannot reliably tell whether the polarity differs, it returns
    None (inconclusive) so the caller can escalate to semantic
    verification, rather than picking a side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rasvcx.schemas.common import EvidenceItemId, _UnknownType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem
from rasvcx.schemas.verification import (
    CitationResult,
    CitationStatus,
    ClaimVerificationResult,
    GeneratedClaim,
    SupportLabel,
    VerificationReasonCode,
    VerificationStage,
)
from rasvcx.validation.deterministic import extract_numbers, extract_years, significant_tokens

# ---------------------------------------------------------------------------
# Precompiled patterns / vocabularies
# ---------------------------------------------------------------------------

_NEGATION_TOKENS: frozenset[str] = frozenset({
    "not", "no", "never", "cannot", "without", "unless", "prohibited",
    "unavailable", "excludes", "excluded", "except",
})
# Deliberately excludes "rarely"/"only" -- the negation section explicitly
# warns against naive substring negation logic around such words; a
# frequency qualifier is handled separately (qualifier strengthening), not
# conflated with true negation.
_NEGATION_PREFIX_RE = re.compile(r"\b(?:non|un)-?[a-z]{3,}", re.IGNORECASE)

_STRONG_QUALIFIERS: frozenset[str] = frozenset({
    "always", "never", "exactly", "guaranteed", "must", "only", "all",
    "every", "everyone", "everywhere", "worldwide", "globally",
})
_WEAK_QUALIFIERS: frozenset[str] = frozenset({
    "may", "might", "can", "could", "typically", "usually", "often",
    "sometimes", "approximately", "roughly", "generally", "up",  # "up to"
    "at",  # "at least" / "at most"
})

_COMPARATIVE_OPPOSITES: tuple[frozenset[str], ...] = (
    frozenset({"higher", "greater", "more", "increased", "increase", "faster", "better", "largest", "most"}),
    frozenset({"lower", "less", "fewer", "decreased", "decrease", "slower", "worse", "smallest", "least"}),
)
_COMPARATIVE_WORDS: frozenset[str] = frozenset().union(*_COMPARATIVE_OPPOSITES)

_TENSE_PRESENT_RE = re.compile(
    r"\b(?:currently|today|now|at\s+present|as\s+of\s+today)\b", re.IGNORECASE
)

_JURISDICTIONS: dict[str, frozenset[str]] = {
    "US": frozenset({"united states", "us", "u.s.", "usa", "america"}),
    "EU": frozenset({"european union", "eu", "europe"}),
    "UK": frozenset({"united kingdom", "uk", "britain"}),
    "India": frozenset({"india"}),
    "China": frozenset({"china"}),
    "Canada": frozenset({"canada"}),
    "Australia": frozenset({"australia"}),
    "worldwide": frozenset({"worldwide", "globally", "everywhere", "all countries"}),
}
_POPULATIONS: dict[str, frozenset[str]] = {
    "adults": frozenset({"adults", "adult"}),
    "pediatric": frozenset({"children", "child", "pediatric", "kids"}),
    "elderly": frozenset({"elderly", "seniors", "older adults"}),
    "all patients": frozenset({"all patients", "everyone", "all"}),
}


def _raw_tokens(text: str) -> frozenset[str]:
    """Lowercase word tokens with NO stopword filtering.

    Deliberately distinct from ``significant_tokens`` (M8): that function's
    stopword list -- reasonable for M8's numeric/temporal comparison use
    case -- includes exactly the modal/negation words ("can", "may",
    "should", "not") this module needs to detect qualifier strengthening
    and negation. Reusing ``significant_tokens`` here would silently strip
    the signal this check exists to find.
    """
    return frozenset(re.findall(r"[a-z']+", text.lower()))


def _find_scope_keyword(text: str, vocab: dict[str, frozenset[str]]) -> str | None:
    lowered = text.lower()
    for canonical, synonyms in vocab.items():
        for synonym in synonyms:
            if re.search(rf"\b{re.escape(synonym)}\b", lowered):
                return canonical
    return None


@dataclass(frozen=True, slots=True)
class _Verdict:
    label: SupportLabel
    reason_code: VerificationReasonCode
    confidence: float
    rationale: str
    supporting_item_ids: frozenset[EvidenceItemId] = frozenset()
    contradicting_item_ids: frozenset[EvidenceItemId] = frozenset()


def check_citation(claim: GeneratedClaim, bundle: EvidenceBundle) -> CitationResult:
    """Verify a claim's citation(s), independent of whether the claim
    content itself turns out to be supported (Sections 13, 32).

    Citation *presence* is checked here; citation *correctness* (whether
    the cited evidence actually supports the claim) is layered on top by
    ``DeterministicClaimVerifier.verify``, per Invariant 5: "Citation
    existence != citation correctness."
    """
    if not claim.cited_item_ids:
        if claim.unresolved_citation_tokens:
            return CitationResult(
                claim_id=claim.claim_id,
                status=CitationStatus.INCORRECT,
                cited_item_ids=frozenset(),
                rationale=(
                    f"Citation marker(s) {sorted(claim.unresolved_citation_tokens)} did not "
                    f"resolve to any evidence item in the bundle"
                ),
            )
        return CitationResult(
            claim_id=claim.claim_id,
            status=CitationStatus.MISSING,
            cited_item_ids=frozenset(),
            rationale="Claim carries no citation marker",
        )

    resolved = frozenset(iid for iid in claim.cited_item_ids if iid in bundle.evidence_items)
    if not resolved:
        return CitationResult(
            claim_id=claim.claim_id,
            status=CitationStatus.INCORRECT,
            cited_item_ids=claim.cited_item_ids,
            rationale="Citation marker did not resolve to any evidence item in the bundle",
        )

    return CitationResult(
        claim_id=claim.claim_id,
        status=CitationStatus.CORRECT,
        cited_item_ids=resolved,
        rationale=f"Citation resolved to {len(resolved)} evidence item(s)",
    )


def find_lexically_relevant_items(
    claim: GeneratedClaim, bundle: EvidenceBundle, min_shared_tokens: int = 2
) -> list[EvidenceItem]:
    """Best-effort fallback alignment when a claim carries no (valid)
    citation: find evidence items whose text shares enough significant
    tokens with the claim to be worth deterministic comparison.

    This does NOT constitute proof of support by itself (Invariant 4:
    "Semantic similarity != factual proof") -- it only selects *candidates*
    for the numeric/negation/qualifier/temporal checks below.
    """
    claim_tokens = significant_tokens(claim.normalized_text)
    if not claim_tokens:
        return []

    scored: list[tuple[int, EvidenceItem]] = []
    for item in bundle.evidence_items.values():
        shared = len(claim_tokens & significant_tokens(item.text))
        if shared >= min_shared_tokens:
            scored.append((shared, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _shared, item in scored]


class DeterministicClaimVerifier:
    """Deterministic-first verification of one GeneratedClaim.

    Returns a ``ClaimVerificationResult`` when a deterministic signal is
    conclusive; returns ``None`` when the claim requires semantic
    escalation (inconclusive lexical match, no numeric/negation/temporal/
    comparative signal either way).
    """

    def __init__(
        self,
        numeric_relative_tolerance: float = 0.01,
        max_fallback_candidates: int = 10,
    ) -> None:
        self._numeric_relative_tolerance = numeric_relative_tolerance
        self._max_fallback_candidates = max_fallback_candidates

    def verify(
        self, claim: GeneratedClaim, bundle: EvidenceBundle, citation: CitationResult
    ) -> ClaimVerificationResult | None:
        candidate_items = self._candidate_items(claim, bundle, citation)

        if not candidate_items:
            return self._result(
                claim,
                _Verdict(
                    label=SupportLabel.UNSUPPORTED,
                    reason_code=VerificationReasonCode.NO_EVIDENCE,
                    confidence=0.9,
                    rationale="No cited or lexically related evidence found in the bundle",
                ),
            )

        best: _Verdict | None = None
        for item in candidate_items[: self._max_fallback_candidates]:
            verdict = self._compare_claim_to_item(claim, item)
            if verdict is None:
                continue
            if best is None or _severity(verdict.label) > _severity(best.label) or (
                _severity(verdict.label) == _severity(best.label)
                and verdict.confidence > best.confidence
            ):
                best = verdict

        if best is None:
            return None  # inconclusive -- escalate to semantic verification

        return self._result(claim, best)

    # -- internal helpers -----------------------------------------------

    def _candidate_items(
        self, claim: GeneratedClaim, bundle: EvidenceBundle, citation: CitationResult
    ) -> list[EvidenceItem]:
        if citation.status is CitationStatus.CORRECT:
            return [bundle.evidence_items[iid] for iid in sorted(citation.cited_item_ids)]
        # No valid citation: fall back to lexical candidate search so an
        # unsupported/uncited claim can still be checked against the
        # bundle rather than defaulting straight to NO_EVIDENCE when
        # relevant evidence does, in fact, exist uncited.
        return find_lexically_relevant_items(claim, bundle)

    def _compare_claim_to_item(self, claim: GeneratedClaim, item: EvidenceItem) -> _Verdict | None:
        # Order matters: mismatch signals (numeric, negation, comparative,
        # temporal, scope) all take precedence over a same-claim lexical
        # SUPPORTED verdict, per this module's docstring invariant.
        checks = (
            self._check_numeric(claim, item),
            self._check_negation(claim, item),
            self._check_comparative(claim, item),
            self._check_temporal(claim, item),
            self._check_scope(claim, item, _JURISDICTIONS, "jurisdiction",
                               VerificationReasonCode.JURISDICTION_MISMATCH),
            self._check_scope(claim, item, _POPULATIONS, "population",
                               VerificationReasonCode.POPULATION_MISMATCH),
            self._check_qualifier_strengthening(claim, item),
        )
        for verdict in checks:
            if verdict is not None:
                return verdict

        return self._check_lexical_support(claim, item)

    def _check_numeric(self, claim: GeneratedClaim, item: EvidenceItem) -> _Verdict | None:
        claim_numbers = extract_numbers(claim.normalized_text)
        evidence_numbers = extract_numbers(item.text.lower())
        if not claim_numbers or not evidence_numbers:
            return None

        for cn in claim_numbers:
            for en in evidence_numbers:
                if cn.unit is None or en.unit is None or cn.unit != en.unit:
                    continue
                if cn.value == en.value:
                    return _Verdict(
                        label=SupportLabel.SUPPORTED,
                        reason_code=VerificationReasonCode.DIRECT_EVIDENCE_SUPPORT,
                        confidence=0.95,
                        rationale=f"Exact numeric match ({cn.value}{cn.unit} == {en.value}{en.unit})",
                        supporting_item_ids=frozenset({item.item_id}),
                    )
                tolerance = self._numeric_relative_tolerance * max(abs(cn.value), abs(en.value), 1e-9)
                if abs(cn.value - en.value) > tolerance:
                    return _Verdict(
                        label=SupportLabel.CONTRADICTED,
                        reason_code=VerificationReasonCode.NUMERIC_MISMATCH,
                        confidence=0.95,
                        rationale=(
                            f"Claim states {cn.value}{cn.unit}, evidence states "
                            f"{en.value}{en.unit}"
                        ),
                        contradicting_item_ids=frozenset({item.item_id}),
                    )
        return None

    def _check_negation(self, claim: GeneratedClaim, item: EvidenceItem) -> _Verdict | None:
        claim_tokens = significant_tokens(claim.normalized_text)
        evidence_tokens = significant_tokens(item.text.lower())
        shared = claim_tokens & evidence_tokens
        # Only judge negation polarity when the claim and evidence are
        # plausibly about the same proposition -- otherwise a "not"
        # anywhere in unrelated evidence could produce a false mismatch.
        if len(shared) < 2:
            return None

        claim_negated = self._is_negated(claim.normalized_text)
        evidence_negated = self._is_negated(item.text.lower())
        if claim_negated is None or evidence_negated is None:
            # Deterministic negation detection wasn't reliable here (e.g.
            # a "non-"/"un-" prefix was found, which is ambiguous without
            # semantic understanding) -- defer rather than guess.
            return None

        if claim_negated != evidence_negated:
            return _Verdict(
                label=SupportLabel.CONTRADICTED,
                reason_code=VerificationReasonCode.NEGATION_MISMATCH,
                confidence=0.85,
                rationale="Claim and evidence assert opposite polarity on the same proposition",
                contradicting_item_ids=frozenset({item.item_id}),
            )
        return None

    def _is_negated(self, text: str) -> bool | None:
        tokens = set(re.findall(r"[a-z']+", text))
        if tokens & _NEGATION_TOKENS:
            return True
        if _NEGATION_PREFIX_RE.search(text):
            # A non-/un- prefix exists but explicit negation words do not:
            # ambiguous (e.g. "unavailable" is covered explicitly above,
            # but "unique" or "underway" are not negations at all) --
            # deliberately inconclusive rather than a naive substring guess.
            return None
        return False

    def _check_comparative(self, claim: GeneratedClaim, item: EvidenceItem) -> _Verdict | None:
        claim_tokens = significant_tokens(claim.normalized_text)
        evidence_tokens = significant_tokens(item.text.lower())
        shared_subject = claim_tokens & evidence_tokens
        if len(shared_subject) < 1:
            return None

        claim_comp = claim_tokens & _COMPARATIVE_WORDS
        evidence_comp = evidence_tokens & _COMPARATIVE_WORDS
        if not claim_comp or not evidence_comp:
            return None

        for group_a, group_b in (_COMPARATIVE_OPPOSITES, tuple(reversed(_COMPARATIVE_OPPOSITES))):
            if claim_comp & group_a and evidence_comp & group_b:
                return _Verdict(
                    label=SupportLabel.CONTRADICTED,
                    reason_code=VerificationReasonCode.COMPARATIVE_REVERSAL,
                    confidence=0.75,
                    rationale=(
                        f"Claim uses comparative term(s) {sorted(claim_comp & group_a)} while "
                        f"evidence uses opposite term(s) {sorted(evidence_comp & group_b)}"
                    ),
                    contradicting_item_ids=frozenset({item.item_id}),
                )
        return None

    def _check_temporal(self, claim: GeneratedClaim, item: EvidenceItem) -> _Verdict | None:
        claim_years = extract_years(claim.normalized_text)
        evidence_years = extract_years(item.text)
        if len(claim_years) == 1 and len(evidence_years) == 1 and claim_years[0] != evidence_years[0]:
            return _Verdict(
                label=SupportLabel.CONTRADICTED,
                reason_code=VerificationReasonCode.TEMPORAL_MISMATCH,
                confidence=0.8,
                rationale=f"Claim states {claim_years[0]}, evidence states {evidence_years[0]}",
                contradicting_item_ids=frozenset({item.item_id}),
            )

        # Present-tense claim ("currently", "now") backed only by evidence
        # with an old/parseable historical date: cannot be silently
        # promoted into a current fact (Invariant 9).
        if _TENSE_PRESENT_RE.search(claim.text) and not claim_years:
            if not isinstance(item.provenance.date, _UnknownType):
                return _Verdict(
                    label=SupportLabel.UNSUPPORTED,
                    reason_code=VerificationReasonCode.TEMPORAL_MISMATCH,
                    confidence=0.6,
                    rationale=(
                        f"Claim asserts a present-tense fact; evidence is dated "
                        f"{item.provenance.date!r} and cannot establish a current value"
                    ),
                )
        return None

    def _check_scope(
        self,
        claim: GeneratedClaim,
        item: EvidenceItem,
        vocab: dict[str, frozenset[str]],
        field_name: str,
        reason_code: VerificationReasonCode,
    ) -> _Verdict | None:
        evidence_value = getattr(item.provenance, field_name)
        if isinstance(evidence_value, _UnknownType):
            return None  # UNKNOWN provenance never becomes a mismatch signal

        claim_keyword = _find_scope_keyword(claim.text, vocab)
        if claim_keyword is None:
            return None

        if claim_keyword in ("worldwide", "all patients"):
            # Overgeneralization: evidence supports a narrower scope than
            # the claim asserts.
            return _Verdict(
                label=SupportLabel.PARTIALLY_SUPPORTED,
                reason_code=reason_code,
                confidence=0.7,
                rationale=(
                    f"Claim generalizes to '{claim_keyword}' but evidence "
                    f"{field_name} is narrower ({evidence_value!r})"
                ),
                supporting_item_ids=frozenset({item.item_id}),
            )

        if str(evidence_value).lower() != claim_keyword.lower():
            return _Verdict(
                label=SupportLabel.CONTRADICTED,
                reason_code=reason_code,
                confidence=0.7,
                rationale=f"Claim asserts {field_name}={claim_keyword!r}, evidence states {evidence_value!r}",
                contradicting_item_ids=frozenset({item.item_id}),
            )
        return None

    def _check_qualifier_strengthening(
        self, claim: GeneratedClaim, item: EvidenceItem
    ) -> _Verdict | None:
        # Shared-subject gating still uses significant_tokens (stopword
        # filtering is fine there -- it's only judging topical overlap);
        # the qualifier words themselves must come from _raw_tokens, since
        # significant_tokens would strip "can"/"may"/"should" outright.
        claim_subject = significant_tokens(claim.normalized_text)
        evidence_subject = significant_tokens(item.text.lower())
        if len(claim_subject & evidence_subject) < 2:
            return None

        evidence_weak = _raw_tokens(item.text) & _WEAK_QUALIFIERS
        claim_strong = _raw_tokens(claim.text) & _STRONG_QUALIFIERS
        if evidence_weak and claim_strong:
            return _Verdict(
                label=SupportLabel.PARTIALLY_SUPPORTED,
                reason_code=VerificationReasonCode.QUALIFIER_STRENGTHENING,
                confidence=0.65,
                rationale=(
                    f"Evidence uses hedged qualifier(s) {sorted(evidence_weak)} but claim "
                    f"asserts certainty via {sorted(claim_strong)}"
                ),
                supporting_item_ids=frozenset({item.item_id}),
            )
        return None

    def _check_lexical_support(self, claim: GeneratedClaim, item: EvidenceItem) -> _Verdict | None:
        claim_tokens = significant_tokens(claim.normalized_text)
        evidence_tokens = significant_tokens(item.text.lower())
        if not claim_tokens:
            return None
        overlap_ratio = len(claim_tokens & evidence_tokens) / len(claim_tokens)
        # Conservative threshold: only a near-total token overlap is
        # treated as deterministic lexical support. This is NOT semantic
        # entailment (Invariant 4); it is a narrow, explainable special
        # case for claims that closely paraphrase or restate evidence text.
        if overlap_ratio >= 0.85:
            return _Verdict(
                label=SupportLabel.SUPPORTED,
                reason_code=VerificationReasonCode.DIRECT_EVIDENCE_SUPPORT,
                confidence=0.7,
                rationale=f"Claim tokens overlap {overlap_ratio:.0%} with evidence text",
                supporting_item_ids=frozenset({item.item_id}),
            )
        return None  # inconclusive -- let selective NLI decide

    def _result(self, claim: GeneratedClaim, verdict: _Verdict) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            label=verdict.label,
            confidence=verdict.confidence,
            stage=VerificationStage.DETERMINISTIC,
            reason_code=verdict.reason_code,
            rationale=verdict.rationale,
            supporting_item_ids=verdict.supporting_item_ids,
            contradicting_item_ids=verdict.contradicting_item_ids,
        )


_LABEL_SEVERITY: dict[SupportLabel, int] = {
    SupportLabel.CONTRADICTED: 3,
    SupportLabel.PARTIALLY_SUPPORTED: 2,
    SupportLabel.SUPPORTED: 1,
    SupportLabel.UNSUPPORTED: 0,
    SupportLabel.UNCERTAIN: 0,
    SupportLabel.NOT_VERIFIABLE: 0,
}


def _severity(label: SupportLabel) -> int:
    """Ranks verdict labels so that, across multiple candidate evidence
    items, a CONTRADICTED signal always wins over a SUPPORTED one found
    against a *different* item (Invariant 8: a real contradiction must not
    be downgraded merely because another, weaker signal says support).
    """
    return _LABEL_SEVERITY[label]