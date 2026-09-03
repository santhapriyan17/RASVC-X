"""Tests for Module 9 -- Post-Generation Verification.

Encodes the master prompt's invariants directly as assertions (Section 53:
"Tests must encode invariants") rather than merely re-running demo
scenarios for a pass/fail check.
"""

from __future__ import annotations

from rasvcx.schemas.common import ChunkId, EvidenceItemId, NLILabel, QueryId, SourceType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.schemas.validation import NLISignal
from rasvcx.schemas.verification import AnswerVerdict, CitationStatus, SupportLabel, VerificationReasonCode
from rasvcx.validation.nli_interface import NLIService
from rasvcx.validation.nli_model import NullNLIBackend
from rasvcx.verification import VerificationPipeline
from rasvcx.verification.claim_extraction_post import GeneratedClaimExtractor
from rasvcx.verification.deterministic_check import DeterministicClaimVerifier, check_citation

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _risk_profile() -> RiskProfile:
    return RiskProfile(
        overall_risk_score=0.3, feature_scores=RiskFeatureScores(),
        validation_depth=ValidationDepth.STANDARD, retrieval_retry_budget=1, nli_call_allowance=5,
    )


def _query() -> QueryRequest:
    return QueryRequest(query_id=QueryId("q1"), raw_text="test", normalized_text="test")


def _provenance(**overrides) -> Provenance:
    base = dict(source_type=SourceType.PEER_REVIEWED_LITERATURE, date="2022-01-01",
                jurisdiction="US", population="adults", dosage_context="general")
    base.update(overrides)
    return Provenance(**base)


def _bundle_with_evidence(item_id: str, text: str, **prov_overrides) -> EvidenceBundle:
    bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
    bundle.add_evidence_item(
        EvidenceItem(
            item_id=EvidenceItemId(item_id), chunk_id=ChunkId(f"ch-{item_id}"), text=text,
            retrieval_score=0.9, provenance=_provenance(**prov_overrides),
        )
    )
    return bundle


def _empty_bundle() -> EvidenceBundle:
    return EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())


class _FakeEntailmentBackend:
    def is_available(self) -> bool:
        return True

    def predict(self, premise: str, hypothesis: str) -> NLISignal:
        return NLISignal(label=NLILabel.ENTAILMENT, confidence=0.9)


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


class TestClaimExtraction:
    def test_empty_answer_yields_no_claims(self):
        assert GeneratedClaimExtractor().extract("", _empty_bundle()) == []

    def test_whitespace_only_answer_yields_no_claims(self):
        assert GeneratedClaimExtractor().extract("   \n\t  ", _empty_bundle()) == []

    def test_citation_marker_does_not_merge_adjacent_sentences(self):
        bundle = _bundle_with_evidence("E1", "x")
        claims = GeneratedClaimExtractor().extract("First fact. [E1] Second fact.", bundle)
        assert len(claims) == 2

    def test_span_is_byte_exact_against_original_answer(self):
        bundle = _bundle_with_evidence("E1", "x")
        answer = "First fact. [E1] Second fact."
        claims = GeneratedClaimExtractor().extract(answer, bundle)
        for claim in claims:
            start, end = claim.span
            assert answer[start:end] == claim.text

    def test_claim_ids_are_deterministic(self):
        bundle = _bundle_with_evidence("E1", "x")
        answer = "The sky is blue today."
        first = GeneratedClaimExtractor().extract(answer, bundle)
        second = GeneratedClaimExtractor().extract(answer, bundle)
        assert [c.claim_id for c in first] == [c.claim_id for c in second]

    def test_numeric_citation_resolves_by_sorted_position(self):
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
        bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("E1"), chunk_id=ChunkId("c1"),
                                               text="a", retrieval_score=0.9, provenance=_provenance()))
        bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("E2"), chunk_id=ChunkId("c2"),
                                               text="b", retrieval_score=0.5, provenance=_provenance()))
        claims = GeneratedClaimExtractor().extract("Some fact stated here. [1]", bundle)
        assert claims[0].cited_item_ids == frozenset({EvidenceItemId("E1")})

    def test_comma_separated_multi_citation(self):
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
        bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("E1"), chunk_id=ChunkId("c1"),
                                               text="a", retrieval_score=0.9, provenance=_provenance()))
        bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("E2"), chunk_id=ChunkId("c2"),
                                               text="b", retrieval_score=0.5, provenance=_provenance()))
        claims = GeneratedClaimExtractor().extract("Revenue grew. [1,2]", bundle)
        assert claims[0].cited_item_ids == frozenset({EvidenceItemId("E1"), EvidenceItemId("E2")})

    def test_citation_inside_fenced_code_block_is_ignored(self):
        bundle = _bundle_with_evidence("E1", "x")
        answer = 'See below. ```print("[E1]")``` Also confirmed. [E1]'
        claims = GeneratedClaimExtractor().extract(answer, bundle)
        code_claim = next(c for c in claims if "print" in c.text or "See below" in c.text)
        assert code_claim.cited_item_ids == frozenset()

    def test_invalid_citation_id_never_becomes_valid(self):
        bundle = _bundle_with_evidence("E1", "x")
        claims = GeneratedClaimExtractor().extract("A fact is stated. [E999]", bundle)
        assert claims[0].cited_item_ids == frozenset()
        assert "E999" in claims[0].unresolved_citation_tokens

    def test_unresolved_citation_distinguished_from_no_citation(self):
        bundle = _bundle_with_evidence("E1", "x")
        with_bad_citation = GeneratedClaimExtractor().extract("A fact. [E999]", bundle)[0]
        without_citation = GeneratedClaimExtractor().extract("A fact.", bundle)[0]
        assert with_bad_citation.unresolved_citation_tokens != without_citation.unresolved_citation_tokens


# ---------------------------------------------------------------------------
# Citation verification (Section 13, Invariant 5)
# ---------------------------------------------------------------------------


class TestCitationVerification:
    def test_missing_citation_status(self):
        bundle = _bundle_with_evidence("E1", "x")
        claim = GeneratedClaimExtractor().extract("A fact with no citation.", bundle)[0]
        assert check_citation(claim, bundle).status is CitationStatus.MISSING

    def test_invalid_citation_status_distinct_from_missing(self):
        bundle = _bundle_with_evidence("E1", "x")
        claim = GeneratedClaimExtractor().extract("A fact. [E999]", bundle)[0]
        result = check_citation(claim, bundle)
        assert result.status is CitationStatus.INCORRECT
        assert result.status is not CitationStatus.MISSING

    def test_citation_existence_does_not_imply_correctness(self):
        """Invariant 5: a claim can carry a valid, resolvable citation
        while the claim CONTENT is still contradicted by that evidence."""
        bundle = _bundle_with_evidence("E1", "Accuracy was 98.7%.")
        claim = GeneratedClaimExtractor().extract("Accuracy was 99.7%. [E1]", bundle)[0]
        citation = check_citation(claim, bundle)
        result = DeterministicClaimVerifier().verify(claim, bundle, citation)
        assert citation.status is CitationStatus.CORRECT
        assert result.label is SupportLabel.CONTRADICTED


# ---------------------------------------------------------------------------
# Deterministic checks (Sections 14-22)
# ---------------------------------------------------------------------------


class TestDeterministicVerification:
    def test_numeric_mismatch_is_contradicted(self):
        bundle = _bundle_with_evidence("E1", "RASVC-X achieved 98.7% accuracy.")
        claim = GeneratedClaimExtractor().extract("RASVC-X achieved 99.7% accuracy. [E1]", bundle)[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is SupportLabel.CONTRADICTED
        assert result.reason_code is VerificationReasonCode.NUMERIC_MISMATCH

    def test_exact_numeric_match_is_supported(self):
        bundle = _bundle_with_evidence("E1", "Applications increased by 20%.")
        claim = GeneratedClaimExtractor().extract("Applications increased by 20%. [E1]", bundle)[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is SupportLabel.SUPPORTED

    def test_negation_reversal_is_contradicted(self):
        bundle = _bundle_with_evidence("E1", "The policy does not apply to contractors.")
        claim = GeneratedClaimExtractor().extract("The policy applies to contractors. [E1]", bundle)[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is SupportLabel.CONTRADICTED
        assert result.reason_code is VerificationReasonCode.NEGATION_MISMATCH

    def test_qualifier_strengthening_is_partially_supported_not_supported(self):
        bundle = _bundle_with_evidence("E1", "The system can process up to 100 requests per second.")
        claim = GeneratedClaimExtractor().extract(
            "The system processes exactly 100 requests per second. [E1]", bundle
        )[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is SupportLabel.PARTIALLY_SUPPORTED
        assert result.label is not SupportLabel.SUPPORTED

    def test_jurisdiction_mismatch_is_contradicted(self):
        bundle = _bundle_with_evidence("E1", "The study was conducted in India.", jurisdiction="India")
        claim = GeneratedClaimExtractor().extract(
            "The study was conducted in the United States. [E1]", bundle
        )[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is SupportLabel.CONTRADICTED
        assert result.reason_code is VerificationReasonCode.JURISDICTION_MISMATCH

    def test_population_overgeneralization_is_partial_not_full_support(self):
        bundle = _bundle_with_evidence("E1", "Drug X is approved for adults.", population="adults")
        claim = GeneratedClaimExtractor().extract("Drug X is approved for all patients. [E1]", bundle)[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is SupportLabel.PARTIALLY_SUPPORTED

    def test_temporal_year_mismatch_is_contradicted(self):
        bundle = _bundle_with_evidence("E1", "In 2022, revenue was $5 billion.")
        claim = GeneratedClaimExtractor().extract("In 2023, revenue was $5 billion. [E1]", bundle)[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is SupportLabel.CONTRADICTED
        assert result.reason_code is VerificationReasonCode.TEMPORAL_MISMATCH

    def test_historical_evidence_does_not_become_current_fact(self):
        """Invariant 9: historical evidence must not automatically
        establish a present-tense claim."""
        bundle = _bundle_with_evidence("E1", "In 2022, revenue was $5 billion.")
        claim = GeneratedClaimExtractor().extract("Revenue is currently $5 billion. [E1]", bundle)[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is not SupportLabel.SUPPORTED

    def test_no_evidence_is_unsupported_never_supported(self):
        """Invariant 1: missing evidence != supported."""
        bundle = _bundle_with_evidence("E1", "Something completely unrelated.")
        claim = GeneratedClaimExtractor().extract("RASVC-X achieved 99.7% accuracy.", bundle)[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is SupportLabel.UNSUPPORTED
        assert result.reason_code is VerificationReasonCode.NO_EVIDENCE

    def test_no_evidence_at_all_in_bundle_is_unsupported(self):
        bundle = _empty_bundle()
        claim = GeneratedClaimExtractor().extract("A claim with no evidence anywhere.", bundle)[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        assert result.label is SupportLabel.UNSUPPORTED

    def test_ambiguous_negation_prefix_defers_rather_than_guesses(self):
        """An un-/non- prefix without an explicit negation word must not
        be treated as a confident negation signal (over naive substring
        matching, per the negation-handling guidance)."""
        bundle = _bundle_with_evidence("E1", "The results were unique across all trials and metrics run.")
        claim = GeneratedClaimExtractor().extract(
            "The results were unique across every trial and metric tested. [E1]", bundle
        )[0]
        result = DeterministicClaimVerifier().verify(claim, bundle, check_citation(claim, bundle))
        # Must not be a false CONTRADICTED from the "un-" prefix heuristic.
        assert result is None or result.reason_code is not VerificationReasonCode.NEGATION_MISMATCH


# ---------------------------------------------------------------------------
# Semantic (NLI) verification
# ---------------------------------------------------------------------------


class TestSemanticVerification:
    def test_nli_unavailable_never_becomes_supported(self):
        """Invariant 2: NLI unavailable != supported."""
        bundle = _bundle_with_evidence("E1", "Python is a general-purpose programming language.")
        pipeline = VerificationPipeline(nli_service=NLIService(NullNLIBackend()))
        summary = pipeline.verify(
            "Python is primarily used for building operating system kernels.", bundle, query=_query()
        )
        assert all(r.label is not SupportLabel.SUPPORTED for r in summary.claim_results)

    def test_nli_entailment_can_produce_supported_when_deterministic_is_inconclusive(self):
        bundle = _bundle_with_evidence("E1", "Python is a general-purpose, high-level programming language.")
        pipeline = VerificationPipeline(nli_service=NLIService(_FakeEntailmentBackend()))
        summary = pipeline.verify(
            "Python is broadly applicable across many kinds of software projects. [E1]",
            bundle,
            query=_query(),
        )
        assert summary.claim_results[0].stage.value == "selective_nli"
        assert summary.claim_results[0].label is SupportLabel.SUPPORTED


# ---------------------------------------------------------------------------
# Answer-level aggregation / pipeline
# ---------------------------------------------------------------------------


class TestVerificationPipeline:
    def test_import_performs_no_model_loading(self):
        import importlib

        import rasvcx.verification as pkg

        importlib.reload(pkg)
        assert hasattr(pkg, "VerificationPipeline")

    def test_empty_answer_returns_valid_result_never_verified(self):
        bundle = _bundle_with_evidence("E1", "x")
        pipeline = VerificationPipeline()
        summary = pipeline.verify("", bundle, query=_query())
        assert summary.answer_verdict is not AnswerVerdict.VERIFIED
        assert summary.claim_results == ()

    def test_contradiction_yields_unverified(self):
        bundle = _bundle_with_evidence("E1", "RASVC-X achieved 98.7% accuracy.")
        pipeline = VerificationPipeline()
        summary = pipeline.verify("RASVC-X achieved 99.7% accuracy. [E1]", bundle, query=_query())
        assert summary.answer_verdict is AnswerVerdict.UNVERIFIED

    def test_full_support_yields_verified(self):
        bundle = _bundle_with_evidence("E1", "RASVC-X achieved 98.7% accuracy.")
        pipeline = VerificationPipeline()
        summary = pipeline.verify("RASVC-X achieved 98.7% accuracy. [E1]", bundle, query=_query())
        assert summary.answer_verdict is AnswerVerdict.VERIFIED

    def test_confidence_always_in_valid_range(self):
        bundle = _bundle_with_evidence("E1", "RASVC-X achieved 98.7% accuracy.")
        pipeline = VerificationPipeline()
        summary = pipeline.verify("RASVC-X achieved 99.7% accuracy. [E1]", bundle, query=_query())
        assert 0.0 <= summary.overall_confidence <= 1.0
        for r in summary.claim_results:
            assert 0.0 <= r.confidence <= 1.0

    def test_high_confidence_contradiction_is_not_diluted_by_average(self):
        """Invariant 8/6: a genuine contradiction must not be washed out
        by an aggregate confidence average, and high confidence elsewhere
        must not imply truth."""
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
        bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("E1"), chunk_id=ChunkId("c1"),
                                               text="Accuracy was 98.7%.", retrieval_score=0.9,
                                               provenance=_provenance()))
        bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("E2"), chunk_id=ChunkId("c2"),
                                               text="The model uses a transformer architecture.",
                                               retrieval_score=0.9, provenance=_provenance()))
        pipeline = VerificationPipeline()
        answer = (
            "The model uses a transformer architecture. [E2] "
            "Accuracy was 99.7%. [E1]"
        )
        summary = pipeline.verify(answer, bundle, query=_query())
        assert summary.answer_verdict is AnswerVerdict.UNVERIFIED
        assert summary.overall_confidence <= 0.3

    def test_safety_critical_contradiction_yields_unsafe(self):
        bundle = _bundle_with_evidence(
            "E1", "The recommended dose is 500 mg twice daily.",
        )
        pipeline = VerificationPipeline()
        summary = pipeline.verify(
            "The recommended dose is 250 mg twice daily. [E1]", bundle, query=_query()
        )
        assert summary.answer_verdict is AnswerVerdict.UNSAFE
        assert summary.safety_critical_failure_count >= 1

    def test_repeated_verification_is_deterministic(self):
        bundle = _bundle_with_evidence("E1", "RASVC-X achieved 98.7% accuracy.")
        pipeline = VerificationPipeline()
        answer = "RASVC-X achieved 99.7% accuracy. [E1]"
        first = pipeline.verify(answer, bundle, query=_query())
        second = pipeline.verify(answer, bundle, query=_query())
        assert first.answer_verdict == second.answer_verdict
        assert [r.claim_id for r in first.claim_results] == [r.claim_id for r in second.claim_results]
        assert [r.label for r in first.claim_results] == [r.label for r in second.claim_results]

    def test_timing_is_recorded_under_the_correct_pipeline_stage(self):
        bundle = _bundle_with_evidence("E1", "x")
        pipeline = VerificationPipeline()
        pipeline.verify("A fact. [E1]", bundle, query=_query())
        assert "post_generation_verification" in bundle.metadata.elapsed_per_stage

    def test_orphan_citation_detection(self):
        bundle = EvidenceBundle(query_id=QueryId("q1"), risk_profile=_risk_profile())
        bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("E1"), chunk_id=ChunkId("c1"),
                                               text="Cited fact.", retrieval_score=0.9,
                                               provenance=_provenance()))
        bundle.add_evidence_item(EvidenceItem(item_id=EvidenceItemId("E2"), chunk_id=ChunkId("c2"),
                                               text="Never cited fact.", retrieval_score=0.5,
                                               provenance=_provenance()))
        pipeline = VerificationPipeline()
        summary = pipeline.verify("Cited fact stated here. [E1]", bundle, query=_query())
        assert EvidenceItemId("E2") in summary.orphan_citation_ids
        assert EvidenceItemId("E1") not in summary.orphan_citation_ids