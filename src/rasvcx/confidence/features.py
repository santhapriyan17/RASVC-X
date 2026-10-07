"""Extracts ConfidenceFeatures (schemas/confidence.py) from upstream
pipeline signals. Reuses M6 SourceQualityScorer, M8 ValidationSummary,
M9 VerificationSummary directly -- no re-derivation of their logic.

Every feature is bounded to [0, 1] by construction. Missing upstream data
never becomes a positive signal: see FeatureCompleteness, which records
which of the 8 features were actually derived from real signals versus a
documented conservative default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rasvcx.provenance.source_quality import SourceQualityScorer
from rasvcx.schemas.confidence import ConfidenceFeatures
from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.validation.verified_context import ValidationSummary
from rasvcx.verification import SupportLabel, VerificationSummary


@dataclass(frozen=True, slots=True)
class FeatureCompleteness:
    """Which ConfidenceFeatures fields came from real signals vs a
    conservative default because the upstream signal was absent.
    """

    had_evidence: bool
    had_validation: bool
    had_verification: bool
    had_rerank_scores: bool
    had_provenance: bool


@dataclass(frozen=True, slots=True)
class FeatureExtractionResult:
    features: ConfidenceFeatures
    completeness: FeatureCompleteness


# Resolutions that reflect a real disagreement between evidence items, as
# opposed to a context-explained divergence (population/jurisdiction/
# dosage/temporal diff) which is not itself evidence disagreement.
_CONFLICT_RELATIONSHIPS = frozenset({"genuine-conflict"})
_UNRESOLVED_RELATIONSHIPS = frozenset({"unresolved"})
_COMPATIBLE_RELATIONSHIPS = frozenset({"compatible"})


class FeatureExtractor:
    def __init__(self, source_quality_scorer: SourceQualityScorer | None = None) -> None:
        self._source_quality_scorer = source_quality_scorer or SourceQualityScorer()

    def extract(
        self,
        bundle: EvidenceBundle,
        validation_summary: ValidationSummary | None,
        verification_summary: VerificationSummary | None,
    ) -> FeatureExtractionResult:
        items = list(bundle.evidence_items.values())
        had_evidence = len(items) > 0

        retrieval_quality = self._retrieval_quality(bundle, items) if had_evidence else 0.0

        rerank_scores = [it.rerank_score for it in items if it.rerank_score is not None]
        had_rerank_scores = len(rerank_scores) > 0
        rerank_quality = self._rerank_quality(rerank_scores) if had_rerank_scores else retrieval_quality

        # Provenance quality and corroboration describe the evidence the
        # ANSWER rests on (M9's supporting items, minus withdrawn/superseded
        # sources) -- an irrelevant guideline in the candidate list must not
        # raise source diversity.  All items when no support is identified.
        from rasvcx.provenance.evidence_roles import answer_basis

        basis = answer_basis(items, verification_summary)
        provenance_quality, had_provenance = self._provenance_quality(basis)
        source_diversity = self._source_diversity(basis)

        evidence_agreement, contradiction_penalty, resolution_uncertainty_penalty, had_validation = (
            self._validation_signals(validation_summary, had_evidence)
        )

        claim_verification_support, had_verification = self._verification_signal(verification_summary)

        features = ConfidenceFeatures(
            evidence_agreement=evidence_agreement,
            claim_verification_support=claim_verification_support,
            contradiction_penalty=contradiction_penalty,
            provenance_quality=provenance_quality,
            source_diversity=source_diversity,
            retrieval_quality=retrieval_quality,
            rerank_quality=rerank_quality,
            resolution_uncertainty_penalty=resolution_uncertainty_penalty,
        )
        completeness = FeatureCompleteness(
            had_evidence=had_evidence,
            had_validation=had_validation,
            had_verification=had_verification,
            had_rerank_scores=had_rerank_scores,
            had_provenance=had_provenance,
        )
        return FeatureExtractionResult(features=features, completeness=completeness)

    def _retrieval_quality(self, bundle: EvidenceBundle, items: list) -> float:
        """How well retrieval agreed with itself, in [0, 1].

        In hybrid mode this is the fraction of fused results that BM25 and
        dense retrieval both returned: lexical and semantic search
        independently pointing at the same passages.

        The fused RRF score is not used for this.  RRF scores are bounded by
        about 2 / (rrf_k + 1) (~0.03 for rrf_k = 60), so a mean of them is
        ~0.02 for every query -- a constant, not a signal -- and it capped
        the achievable confidence of a perfect answer.  Without a hybrid
        trace (BM25-only, or a caller-supplied retrieval function) the
        clamped mean of the retrieval scores is kept.
        """
        trace = getattr(bundle, "retrieval_trace", None) or {}
        fused = trace.get("rrf_fused")
        both = trace.get("rrf_from_both")
        if (
            trace.get("mode") == "hybrid"
            and isinstance(fused, int) and fused > 0
            and isinstance(both, int)
        ):
            return max(0.0, min(1.0, both / fused))
        return self._mean_clamped([it.retrieval_score for it in items])

    @staticmethod
    def _rerank_quality(scores: list[float]) -> float:
        """Relevance of the best evidence, in [0, 1].

        Cross-encoder scores are logits (roughly -11 .. +11), not
        probabilities.  Each is mapped through the logistic function and the
        TOP three are averaged: what matters is whether strong evidence was
        found, not how irrelevant the tail of the candidate list is.
        (Clamping the raw mean of all logits to [0, 1] -- the previous
        behaviour -- made the feature 0 or 1 almost at random.)
        """
        finite = sorted((v for v in scores if math.isfinite(v)), reverse=True)[:3]
        if not finite:
            return 0.0
        relevance = [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, v)))) for v in finite]
        return sum(relevance) / len(relevance)

    @staticmethod
    def _mean_clamped(values: list[float]) -> float:
        # NaN/+-inf must never reach a mean: min()/max() do not reject NaN
        # (Python's total-order comparisons make NaN silently "win" against
        # 1.0 in min(1.0, nan) == 1.0), so a single malformed upstream score
        # could otherwise become the maximum possible confidence value.
        # Non-finite values are dropped rather than propagated or coerced.
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return 0.0
        mean = sum(finite) / len(finite)
        return max(0.0, min(1.0, mean))

    def _provenance_quality(self, items: list) -> tuple[float, bool]:
        if not items:
            return 0.0, False
        scores = [self._source_quality_scorer.score(it) for it in items]
        known = [s.quality_score for s in scores if s.is_known]
        if not known:
            return 0.0, True
        return self._mean_clamped(known), True

    @staticmethod
    def _source_diversity(items: list) -> float:
        """Independent corroboration, in [0, 1].

            0.0  everything comes from one document
            0.5  several documents of a single source type
            1.0  at least two source types (e.g. a label and a guideline)

        The previous formula, (types - 1) / (items - 1), divided by the
        NUMBER OF CHUNKS: ten chunks from a label and a guideline scored
        0.11, and the maximum was unreachable for any realistic evidence
        set.  Documents are identified by SourceRef.doc_id; items without
        one count as a single unknown document per source type.
        """
        if len(items) <= 1:
            return 0.0
        types = {it.provenance.source_type for it in items}
        if len(types) >= 2:
            return 1.0
        documents = {
            getattr(getattr(it, "source", None), "doc_id", None) for it in items
        }
        documents.discard(None)
        return 0.5 if len(documents) >= 2 else 0.0

    @staticmethod
    def _validation_signals(
        validation_summary: ValidationSummary | None, had_evidence: bool
    ) -> tuple[float, float, float, bool]:
        if validation_summary is None:
            # No validation ran: cannot claim agreement, cannot claim
            # contradiction. Both stay at a conservative midpoint-free
            # zero contribution rather than being invented from silence.
            return (0.0, 0.0, 1.0 if had_evidence else 0.0, False)

        resolutions = validation_summary.resolutions
        if not resolutions:
            if validation_summary.candidates_generated > 0:
                # Candidates existed but none were flagged conflicting:
                # mild positive signal, not proof.
                return (0.6, 0.0, 0.3, True)
            # Too few evidence items for M8 to even attempt conflict
            # detection (e.g. a single source): this is NOT corroborated
            # agreement between independent sources, only "nothing to
            # disagree with." A neutral value avoids both fabricating
            # multi-source agreement and penalizing single-source evidence
            # as if it were contradictory.
            return (0.5, 0.0, 0.0, True)

        total = len(resolutions)
        compatible = sum(1 for r in resolutions if r.relationship.value in _COMPATIBLE_RELATIONSHIPS)
        conflict = sum(1 for r in resolutions if r.relationship.value in _CONFLICT_RELATIONSHIPS)
        unresolved = sum(1 for r in resolutions if r.relationship.value in _UNRESOLVED_RELATIONSHIPS)

        agreement = max(0.0, min(1.0, (compatible - conflict - 0.5 * unresolved) / total + 0.5))
        contradiction_penalty = max(0.0, min(1.0, conflict / total))
        uncertainty_penalty = max(0.0, min(1.0, unresolved / total))
        return (agreement, contradiction_penalty, uncertainty_penalty, True)

    @staticmethod
    def _verification_signal(
        verification_summary: VerificationSummary | None,
    ) -> tuple[float, bool]:
        if verification_summary is None or not verification_summary.claim_results:
            return 0.0, verification_summary is not None

        total = len(verification_summary.claim_results)
        supported = sum(
            1 for r in verification_summary.claim_results if r.label is SupportLabel.SUPPORTED
        )
        partial = sum(
            1
            for r in verification_summary.claim_results
            if r.label is SupportLabel.PARTIALLY_SUPPORTED
        )
        score = (supported + 0.5 * partial) / total
        return max(0.0, min(1.0, score)), True