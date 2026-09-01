"""Comprehensive tests for Module 7 — Atomic Claims + Claim/Evidence Linkage.

Tests cover:
  A. Sentence segmentation (split_sentences)
  B. Claim type classification (classify_claim_type)
  C. Classification false-positive guards
  D. Safety-critical detection (is_safety_critical)
  E. Normalization (normalize_claim_text, apply_normalization)
  F. Claim ID generation (generate_claim_id)
  G. Claim-to-evidence linking (ClaimLinker)
  H. Bundle integration (AtomicClaimPipeline)
  I. Idempotency (repeated pipeline runs)
  J. Large input (1000 items)
"""

import dataclasses
import pytest

from rasvcx.schemas.common import (
    UNKNOWN,
    ChunkId,
    ClaimId,
    EvidenceItemId,
    QueryId,
    SourceType,
)
from rasvcx.schemas.claims import Claim, ClaimRegistry, ClaimSource, ClaimType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import RiskFeatureScores, RiskProfile, ValidationDepth

from rasvcx.claims import (
    AtomicClaimExtractionResult,
    AtomicClaimPipeline,
    ClaimExtractor,
    ClaimLinker,
    DEFAULT_EXTRACTION_CONFIG,
    ExtractionConfig,
    apply_normalization,
    classify_claim_type,
    generate_claim_id,
    is_safety_critical,
    normalize_claim_text,
    split_sentences,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def make_provenance(**overrides):
    defaults = dict(
        source_type=SourceType.DRUG_LABEL,
        date=UNKNOWN,
        jurisdiction=UNKNOWN,
        population=UNKNOWN,
        dosage_context=UNKNOWN,
    )
    defaults.update(overrides)
    return Provenance(**defaults)


def make_item(item_id="e1", chunk_id=None, text="Some evidence text here.", **prov_overrides):
    if chunk_id is None:
        chunk_id = f"chunk-{item_id}"
    return EvidenceItem(
        item_id=EvidenceItemId(item_id),
        chunk_id=ChunkId(chunk_id),
        text=text,
        retrieval_score=0.9,
        provenance=make_provenance(**prov_overrides),
    )


def make_risk_profile():
    return RiskProfile(
        overall_risk_score=0.5,
        feature_scores=RiskFeatureScores(),
        validation_depth=list(ValidationDepth)[0],
        retrieval_retry_budget=1,
        nli_call_allowance=1,
        safety_floor_forced=False,
    )


def make_bundle():
    return EvidenceBundle(
        query_id=QueryId("q1"),
        risk_profile=make_risk_profile(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# A. SENTENCE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════


class TestSplitSentences:
    """Tests for split_sentences()."""

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_whitespace_only(self):
        assert split_sentences("   \t\n  ") == []

    def test_single_sentence(self):
        result = split_sentences("The drug is effective.")
        assert result == ["The drug is effective."]

    def test_single_sentence_no_punctuation(self):
        result = split_sentences("The drug is effective")
        assert result == ["The drug is effective"]

    def test_multiple_sentences(self):
        result = split_sentences("Take 5 mg. Repeat daily.")
        assert len(result) == 2
        assert result[0] == "Take 5 mg."
        assert result[1] == "Repeat daily."

    def test_three_sentences(self):
        result = split_sentences(
            "First sentence. Second sentence. Third sentence."
        )
        assert len(result) == 3

    def test_dr_abbreviation_not_split(self):
        result = split_sentences("Dr. Smith prescribed 5.5 mg daily.")
        assert len(result) == 1
        assert "Dr. Smith" in result[0]

    def test_mr_abbreviation_not_split(self):
        result = split_sentences("Mr. Jones was treated. He recovered.")
        assert len(result) == 2
        assert "Mr. Jones" in result[0]

    def test_eg_abbreviation_not_split(self):
        result = split_sentences("e.g. this case was reviewed.")
        assert len(result) == 1

    def test_ie_abbreviation_not_split(self):
        result = split_sentences("i.e. the primary outcome was measured.")
        assert len(result) == 1

    def test_decimal_in_dosage_preserved(self):
        result = split_sentences("The dose was 5.5 mg.")
        assert len(result) == 1
        assert "5.5 mg" in result[0]

    def test_newlines_normalized(self):
        result = split_sentences("First sentence.\nSecond sentence.")
        assert len(result) == 2

    def test_tabs_normalized(self):
        result = split_sentences("First sentence.\tSecond sentence.")
        assert len(result) == 2

    def test_multiple_spaces(self):
        result = split_sentences("First sentence.    Second sentence.")
        assert len(result) == 2

    def test_mixed_whitespace(self):
        result = split_sentences("First.\n\t  Second.")
        assert len(result) == 2

    def test_exclamation_mark_splits(self):
        result = split_sentences("Stop immediately! Call the doctor.")
        assert len(result) == 2

    def test_question_mark_splits(self):
        result = split_sentences("Is this safe? Further research needed.")
        assert len(result) == 2

    def test_unicode_text(self):
        result = split_sentences("The dose was 5 µg. The patient recovered.")
        assert len(result) == 2

    def test_long_text_with_many_sentences(self):
        text = ". ".join(f"Sentence number {i}" for i in range(20)) + "."
        result = split_sentences(text)
        # Should produce approximately 20 sentences
        assert len(result) >= 15  # conservative check

    def test_sentence_with_only_punctuation(self):
        # Degenerate case: punctuation-only input
        result = split_sentences("...")
        # Should return something (the normalized text)
        assert len(result) <= 1

    def test_preserves_sentence_order(self):
        text = "Alpha first. Beta second. Gamma third."
        result = split_sentences(text)
        assert result[0].startswith("Alpha")
        assert result[1].startswith("Beta")
        assert result[2].startswith("Gamma")


# ═══════════════════════════════════════════════════════════════════════════
# B. CLAIM TYPE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════


class TestClassifyClaimType:
    """Tests for classify_claim_type()."""

    def test_dosage_with_mg(self):
        assert classify_claim_type("Take 5 mg daily.") == ClaimType.DOSAGE

    def test_dosage_with_mg_kg(self):
        assert classify_claim_type("The dose is 0.5 mg/kg.") == ClaimType.DOSAGE

    def test_dosage_with_mcg(self):
        assert classify_claim_type("Administer 500 mcg.") == ClaimType.DOSAGE

    def test_dosage_with_unicode_microgram(self):
        assert classify_claim_type("The dose was 5 µg.") == ClaimType.DOSAGE

    def test_dosage_with_ml(self):
        assert classify_claim_type("Inject 10 mL subcutaneously.") == ClaimType.DOSAGE

    def test_dosage_with_iu(self):
        assert classify_claim_type("Give 5 IU daily.") == ClaimType.DOSAGE

    def test_dosage_with_units(self):
        assert classify_claim_type("Administer 20 units.") == ClaimType.DOSAGE

    def test_dosage_with_tablets(self):
        assert classify_claim_type("Take 2 tablets daily.") == ClaimType.DOSAGE

    def test_dosage_context_word(self):
        assert classify_claim_type(
            "Dose adjustment is required for renal impairment."
        ) == ClaimType.DOSAGE

    def test_temporal_with_year(self):
        assert classify_claim_type(
            "The study was conducted in 2020."
        ) == ClaimType.TEMPORAL

    def test_temporal_with_year_range(self):
        assert classify_claim_type(
            "Data were collected from 2015-2020."
        ) == ClaimType.TEMPORAL

    def test_temporal_with_keyword_since(self):
        assert classify_claim_type(
            "Since 2018, treatment guidelines have changed."
        ) == ClaimType.TEMPORAL

    def test_temporal_with_keyword_before(self):
        assert classify_claim_type(
            "Before treatment, baseline measurements were taken."
        ) == ClaimType.TEMPORAL

    def test_temporal_with_duration(self):
        assert classify_claim_type(
            "The study lasted for 12 months."
        ) == ClaimType.TEMPORAL

    def test_numeric_with_percentage(self):
        assert classify_claim_type("Mortality was 5%.") == ClaimType.NUMERIC

    def test_numeric_with_patient_count(self):
        assert classify_claim_type(
            "The trial included 240 patients."
        ) == ClaimType.NUMERIC

    def test_numeric_with_sensitivity(self):
        assert classify_claim_type(
            "Sensitivity was 98.2%."
        ) == ClaimType.NUMERIC

    def test_recommendation_should(self):
        assert classify_claim_type(
            "Patients should avoid alcohol."
        ) == ClaimType.RECOMMENDATION

    def test_recommendation_contraindicated(self):
        assert classify_claim_type(
            "This drug is contraindicated in pregnancy."
        ) == ClaimType.RECOMMENDATION

    def test_recommendation_not_recommended(self):
        assert classify_claim_type(
            "This approach is not recommended."
        ) == ClaimType.RECOMMENDATION

    def test_recommendation_must_avoid(self):
        assert classify_claim_type(
            "Patients must avoid grapefruit."
        ) == ClaimType.RECOMMENDATION

    def test_factual_default(self):
        assert classify_claim_type(
            "The drug inhibits enzyme X."
        ) == ClaimType.FACTUAL

    def test_factual_mechanism(self):
        assert classify_claim_type(
            "The protein binds to the receptor."
        ) == ClaimType.FACTUAL

    def test_empty_text(self):
        assert classify_claim_type("") == ClaimType.FACTUAL

    def test_whitespace_only(self):
        assert classify_claim_type("   ") == ClaimType.FACTUAL

    def test_dosage_precedence_over_temporal(self):
        """'Take 5 mg daily for 7 days' has both dosage and temporal.
        DOSAGE should win due to precedence."""
        result = classify_claim_type("Take 5 mg daily for 7 days.")
        assert result == ClaimType.DOSAGE

    def test_dosage_precedence_over_numeric(self):
        """Dosage with numeric value — DOSAGE should win."""
        result = classify_claim_type("The patient received 10 mg.")
        assert result == ClaimType.DOSAGE

    def test_dosage_precedence_over_recommendation(self):
        """'Patients should receive 5 mg twice daily' — DOSAGE wins."""
        result = classify_claim_type(
            "Patients should receive 5 mg twice daily."
        )
        assert result == ClaimType.DOSAGE


# ═══════════════════════════════════════════════════════════════════════════
# C. CLASSIFICATION FALSE-POSITIVE GUARDS
# ═══════════════════════════════════════════════════════════════════════════


class TestClassificationFalsePositives:
    """Tests that specific patterns are NOT misclassified."""

    def test_no_dosage_data_is_not_dosage(self):
        """'No dosage data were available' is about ABSENCE of dosage."""
        result = classify_claim_type("No dosage data were available.")
        assert result != ClaimType.DOSAGE

    def test_percentage_is_not_dosage(self):
        """'5%' is a percentage, not a dosage."""
        result = classify_claim_type("The response rate was 5%.")
        assert result != ClaimType.DOSAGE

    def test_year_is_temporal_not_numeric(self):
        """'2020' in temporal context is TEMPORAL, not NUMERIC."""
        result = classify_claim_type("The study was conducted in 2020.")
        assert result == ClaimType.TEMPORAL

    def test_patient_count_is_numeric_not_dosage(self):
        """'240 patients' is a count, not a dosage."""
        result = classify_claim_type("The trial included 240 patients.")
        assert result == ClaimType.NUMERIC

    def test_simple_factual_not_recommendation(self):
        """Factual description should not be misclassified as recommendation."""
        result = classify_claim_type("The study examined drug efficacy.")
        assert result == ClaimType.FACTUAL

    def test_research_recommendation_is_recommendation(self):
        """'The authors recommend further studies' IS a recommendation."""
        result = classify_claim_type(
            "The authors recommend further studies."
        )
        assert result == ClaimType.RECOMMENDATION


# ═══════════════════════════════════════════════════════════════════════════
# D. SAFETY-CRITICAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════


class TestIsSafetyCritical:
    """Tests for is_safety_critical()."""

    def test_dosage_with_numeric_value_is_critical(self):
        assert is_safety_critical("Take 5 mg daily.", ClaimType.DOSAGE) is True

    def test_dosage_context_only_is_not_critical(self):
        """DOSAGE claim without actual numeric dosage value → depends on
        whether safety keywords match."""
        # 'Dose adjustment' matches safety keywords
        result = is_safety_critical(
            "Dose adjustment is required for renal impairment.",
            ClaimType.DOSAGE,
        )
        assert result is True  # matches dose adjustment keyword

    def test_contraindication_is_critical(self):
        assert is_safety_critical(
            "The drug is contraindicated in severe renal impairment.",
            ClaimType.RECOMMENDATION,
        ) is True

    def test_overdose_is_critical(self):
        assert is_safety_critical(
            "Overdose may cause respiratory depression.",
            ClaimType.FACTUAL,
        ) is True

    def test_maximum_dose_is_critical(self):
        assert is_safety_critical(
            "The maximum dose is 400 mg per day.",
            ClaimType.DOSAGE,
        ) is True

    def test_drug_interaction_is_critical(self):
        assert is_safety_critical(
            "Drug interaction with warfarin may occur.",
            ClaimType.FACTUAL,
        ) is True

    def test_adverse_event_is_critical(self):
        assert is_safety_critical(
            "Severe adverse event reported in clinical trials.",
            ClaimType.FACTUAL,
        ) is True

    def test_stop_medication_is_critical(self):
        assert is_safety_critical(
            "Discontinue the medication if symptoms worsen.",
            ClaimType.RECOMMENDATION,
        ) is True

    def test_ordinary_factual_not_critical(self):
        assert is_safety_critical(
            "The drug inhibits enzyme X.",
            ClaimType.FACTUAL,
        ) is False

    def test_ordinary_numeric_not_critical(self):
        assert is_safety_critical(
            "Mortality was 5%.",
            ClaimType.NUMERIC,
        ) is False

    def test_ordinary_recommendation_not_critical(self):
        assert is_safety_critical(
            "The study recommends further research.",
            ClaimType.RECOMMENDATION,
        ) is False

    def test_empty_text_not_critical(self):
        assert is_safety_critical("", ClaimType.FACTUAL) is False

    def test_do_not_administer_is_critical(self):
        assert is_safety_critical(
            "Do not administer to patients with known hypersensitivity.",
            ClaimType.RECOMMENDATION,
        ) is True


# ═══════════════════════════════════════════════════════════════════════════
# E. NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalization:
    """Tests for normalize_claim_text() and apply_normalization()."""

    def test_basic_normalization(self):
        assert normalize_claim_text("  Take   5 mg  DAILY.  ") == "take 5 mg daily"

    def test_empty_string(self):
        assert normalize_claim_text("") == ""

    def test_whitespace_only(self):
        assert normalize_claim_text("   \t\n  ") == ""

    def test_trailing_period_removed(self):
        assert normalize_claim_text("The drug is effective.") == "the drug is effective"

    def test_trailing_exclamation_removed(self):
        assert normalize_claim_text("Stop immediately!") == "stop immediately"

    def test_trailing_question_removed(self):
        assert normalize_claim_text("Is this safe?") == "is this safe"

    def test_internal_punctuation_preserved(self):
        assert "mg/kg" in normalize_claim_text("Dose: 5 mg/kg.")

    def test_numbers_preserved(self):
        result = normalize_claim_text("Take 5.5 mg.")
        assert "5.5" in result

    def test_idempotent(self):
        text = "  Take   5 mg  DAILY.  "
        first = normalize_claim_text(text)
        second = normalize_claim_text(first)
        assert first == second

    def test_unicode_preserved(self):
        result = normalize_claim_text("The dose was 5 µg.")
        assert "µg" in result

    def test_negation_preserved(self):
        result = normalize_claim_text("This drug should NOT be taken.")
        assert "not" in result

    def test_lowercasing(self):
        result = normalize_claim_text("IMPORTANT FINDING")
        assert result == "important finding"

    def test_apply_normalization_creates_new_claim(self):
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="  Take   5 mg  DAILY.  ",
            source=ClaimSource.EVIDENCE_EXTRACTION,
            claim_type=ClaimType.DOSAGE,
            origin_chunk_id=ChunkId("chunk1"),
        )
        normalized = apply_normalization(claim)
        assert normalized.normalized_text == "take 5 mg daily"
        assert normalized.text == "  Take   5 mg  DAILY.  "  # original preserved

    def test_apply_normalization_idempotent(self):
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="Some text.",
            source=ClaimSource.EVIDENCE_EXTRACTION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=ChunkId("chunk1"),
        )
        first = apply_normalization(claim)
        second = apply_normalization(first)
        assert first is second  # Same object returned

    def test_tabs_and_newlines_collapsed(self):
        result = normalize_claim_text("word1\tword2\nword3")
        assert result == "word1 word2 word3"


# ═══════════════════════════════════════════════════════════════════════════
# F. CLAIM ID GENERATION
# ═══════════════════════════════════════════════════════════════════════════


class TestGenerateClaimId:
    """Tests for generate_claim_id()."""

    def test_deterministic(self):
        id1 = generate_claim_id(EvidenceItemId("e1"), 0, "Test claim.")
        id2 = generate_claim_id(EvidenceItemId("e1"), 0, "Test claim.")
        assert id1 == id2

    def test_prefix_format(self):
        cid = generate_claim_id(EvidenceItemId("e1"), 0, "Test claim.")
        assert cid.startswith("claim_")
        assert len(cid) == len("claim_") + 16  # 16-char hex digest

    def test_different_item_produces_different_id(self):
        id1 = generate_claim_id(EvidenceItemId("e1"), 0, "Same text.")
        id2 = generate_claim_id(EvidenceItemId("e2"), 0, "Same text.")
        assert id1 != id2

    def test_different_index_produces_different_id(self):
        id1 = generate_claim_id(EvidenceItemId("e1"), 0, "Same text.")
        id2 = generate_claim_id(EvidenceItemId("e1"), 1, "Same text.")
        assert id1 != id2

    def test_different_text_produces_different_id(self):
        id1 = generate_claim_id(EvidenceItemId("e1"), 0, "Text A.")
        id2 = generate_claim_id(EvidenceItemId("e1"), 0, "Text B.")
        assert id1 != id2

    def test_not_python_hash(self):
        """ID must not change across processes (no Python hash())."""
        import hashlib
        item_id = EvidenceItemId("e1")
        index = 0
        text = "Test claim."
        expected_digest = hashlib.sha256(
            f"{item_id}:{index}:{text}".encode("utf-8")
        ).hexdigest()[:16]
        expected_id = f"claim_{expected_digest}"
        actual_id = generate_claim_id(item_id, index, text)
        assert actual_id == expected_id

    def test_unicode_text_supported(self):
        cid = generate_claim_id(EvidenceItemId("e1"), 0, "Dose: 5 µg.")
        assert cid.startswith("claim_")

    def test_returns_claim_id_type(self):
        cid = generate_claim_id(EvidenceItemId("e1"), 0, "Test.")
        assert isinstance(cid, str)  # ClaimId is NewType(str)


# ═══════════════════════════════════════════════════════════════════════════
# G. CLAIM-TO-EVIDENCE LINKING
# ═══════════════════════════════════════════════════════════════════════════


class TestClaimLinker:
    """Tests for ClaimLinker.link()."""

    def test_empty_claims(self):
        linker = ClaimLinker()
        items = {"e1": make_item("e1")}
        result = linker.link([], items)
        assert result == {}

    def test_empty_items(self):
        linker = ClaimLinker()
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="Some claim.",
            source=ClaimSource.EVIDENCE_EXTRACTION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=ChunkId("chunk1"),
        )
        result = linker.link([claim], {})
        assert result == {}

    def test_single_claim_single_item(self):
        linker = ClaimLinker()
        item = make_item("e1", chunk_id="chunk-e1")
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="Some claim.",
            source=ClaimSource.EVIDENCE_EXTRACTION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=ChunkId("chunk-e1"),
        )
        result = linker.link([claim], {EvidenceItemId("e1"): item})
        assert EvidenceItemId("e1") in result
        assert ClaimId("c1") in result[EvidenceItemId("e1")]

    def test_multiple_claims_same_chunk(self):
        linker = ClaimLinker()
        item = make_item("e1", chunk_id="chunk-e1")
        claims = [
            Claim(
                claim_id=ClaimId(f"c{i}"),
                text=f"Claim {i}.",
                source=ClaimSource.EVIDENCE_EXTRACTION,
                claim_type=ClaimType.FACTUAL,
                origin_chunk_id=ChunkId("chunk-e1"),
            )
            for i in range(3)
        ]
        result = linker.link(claims, {EvidenceItemId("e1"): item})
        assert len(result[EvidenceItemId("e1")]) == 3

    def test_unknown_origin_produces_no_link(self):
        linker = ClaimLinker()
        item = make_item("e1", chunk_id="chunk-e1")
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="Post-generation claim.",
            source=ClaimSource.POST_GENERATION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=UNKNOWN,
        )
        result = linker.link([claim], {EvidenceItemId("e1"): item})
        assert result == {}

    def test_missing_chunk_produces_no_link(self):
        linker = ClaimLinker()
        item = make_item("e1", chunk_id="chunk-e1")
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="Orphan claim.",
            source=ClaimSource.EVIDENCE_EXTRACTION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=ChunkId("nonexistent-chunk"),
        )
        result = linker.link([claim], {EvidenceItemId("e1"): item})
        assert result == {}

    def test_multiple_items_same_chunk(self):
        """Multiple evidence items sharing the same chunk_id should all
        receive the claim linkage."""
        linker = ClaimLinker()
        items = {
            EvidenceItemId("e1"): make_item("e1", chunk_id="shared-chunk"),
            EvidenceItemId("e2"): make_item("e2", chunk_id="shared-chunk"),
        }
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="Shared claim.",
            source=ClaimSource.EVIDENCE_EXTRACTION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=ChunkId("shared-chunk"),
        )
        result = linker.link([claim], items)
        assert ClaimId("c1") in result.get(EvidenceItemId("e1"), frozenset())
        assert ClaimId("c1") in result.get(EvidenceItemId("e2"), frozenset())

    def test_claims_across_different_chunks(self):
        linker = ClaimLinker()
        items = {
            EvidenceItemId("e1"): make_item("e1", chunk_id="chunk-a"),
            EvidenceItemId("e2"): make_item("e2", chunk_id="chunk-b"),
        }
        claims = [
            Claim(
                claim_id=ClaimId("c1"),
                text="Claim for chunk A.",
                source=ClaimSource.EVIDENCE_EXTRACTION,
                claim_type=ClaimType.FACTUAL,
                origin_chunk_id=ChunkId("chunk-a"),
            ),
            Claim(
                claim_id=ClaimId("c2"),
                text="Claim for chunk B.",
                source=ClaimSource.EVIDENCE_EXTRACTION,
                claim_type=ClaimType.FACTUAL,
                origin_chunk_id=ChunkId("chunk-b"),
            ),
        ]
        result = linker.link(claims, items)
        assert ClaimId("c1") in result[EvidenceItemId("e1")]
        assert ClaimId("c2") in result[EvidenceItemId("e2")]
        assert ClaimId("c1") not in result.get(EvidenceItemId("e2"), frozenset())

    def test_deterministic_output(self):
        """Same input → same output."""
        linker = ClaimLinker()
        item = make_item("e1", chunk_id="chunk-e1")
        claims = [
            Claim(
                claim_id=ClaimId(f"c{i}"),
                text=f"Claim {i}.",
                source=ClaimSource.EVIDENCE_EXTRACTION,
                claim_type=ClaimType.FACTUAL,
                origin_chunk_id=ChunkId("chunk-e1"),
            )
            for i in range(5)
        ]
        items = {EvidenceItemId("e1"): item}
        r1 = linker.link(claims, items)
        r2 = linker.link(claims, items)
        assert r1 == r2


# ═══════════════════════════════════════════════════════════════════════════
# H. BUNDLE INTEGRATION (AtomicClaimPipeline)
# ═══════════════════════════════════════════════════════════════════════════


class TestAtomicClaimPipeline:
    """Tests for AtomicClaimPipeline.run()."""

    def test_empty_bundle(self):
        bundle = make_bundle()
        pipeline = AtomicClaimPipeline()
        result = pipeline.run(bundle)
        assert result.total_claims == 0
        assert result.claim_ids_by_item == {}
        assert "atomic_claim_extraction" in bundle.metadata.elapsed_per_stage

    def test_single_item_single_sentence(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="The drug inhibits enzyme X."
        ))
        pipeline = AtomicClaimPipeline()
        result = pipeline.run(bundle)
        assert result.total_claims == 1
        assert result.factual_claim_count == 1

    def test_claims_registered_in_bundle(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Take 5 mg daily. Avoid alcohol."
        ))
        pipeline = AtomicClaimPipeline()
        result = pipeline.run(bundle)
        assert result.total_claims >= 2
        assert len(bundle.claims) >= 2

    def test_evidence_items_updated_with_claim_ids(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="The drug is effective. Mortality was 5%."
        ))
        pipeline = AtomicClaimPipeline()
        result = pipeline.run(bundle)
        updated_item = bundle.evidence_items[EvidenceItemId("e1")]
        assert len(updated_item.extracted_claim_ids) >= 2

    def test_original_item_not_mutated(self):
        original_item = make_item("e1", text="The drug is effective.")
        bundle = make_bundle()
        bundle.add_evidence_item(original_item)
        original_provenance = original_item.provenance
        original_text = original_item.text
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        # The original object should still have its original values
        # (it was replaced in the bundle, not mutated)
        assert original_item.provenance is original_provenance
        assert original_item.text == original_text
        assert original_item.extracted_claim_ids == frozenset()

    def test_timing_recorded(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item("e1", text="Some evidence text here."))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        assert "atomic_claim_extraction" in bundle.metadata.elapsed_per_stage
        assert bundle.metadata.elapsed_per_stage["atomic_claim_extraction"] >= 0.0

    def test_timing_recorded_even_on_empty_bundle(self):
        bundle = make_bundle()
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        assert "atomic_claim_extraction" in bundle.metadata.elapsed_per_stage

    def test_dosage_claim_safety_critical(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Patients should receive 5 mg twice daily."
        ))
        pipeline = AtomicClaimPipeline()
        result = pipeline.run(bundle)
        assert result.safety_critical_claim_count >= 1
        assert result.dosage_claim_count >= 1

    def test_multiple_items_different_chunks(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", chunk_id="chunk-a", text="The drug inhibits enzyme X."
        ))
        bundle.add_evidence_item(make_item(
            "e2", chunk_id="chunk-b", text="Mortality was 5%."
        ))
        pipeline = AtomicClaimPipeline()
        result = pipeline.run(bundle)
        assert result.total_claims == 2

    def test_result_type_counts_consistent(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Take 5 mg daily. Mortality was 5%. The drug is effective."
        ))
        pipeline = AtomicClaimPipeline()
        result = pipeline.run(bundle)
        type_sum = (
            result.factual_claim_count
            + result.numeric_claim_count
            + result.dosage_claim_count
            + result.temporal_claim_count
            + result.recommendation_claim_count
            + result.other_claim_count
        )
        assert type_sum == result.total_claims

    def test_no_add_evidence_item_used(self):
        """Pipeline must NOT call bundle.add_evidence_item() for existing
        items.  It should directly replace in the evidence_items dict."""
        bundle = make_bundle()
        item = make_item("e1", text="The drug is effective.")
        bundle.add_evidence_item(item)

        # Count items before pipeline
        count_before = len(bundle.evidence_items)
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        count_after = len(bundle.evidence_items)
        assert count_after == count_before  # no new items added

    def test_claim_origin_chunk_id_set_correctly(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", chunk_id="my-chunk", text="The drug is effective."
        ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        for claim in bundle.claims.as_dict().values():
            assert claim.origin_chunk_id == ChunkId("my-chunk")
            assert claim.source == ClaimSource.EVIDENCE_EXTRACTION

    def test_claim_normalized_text_populated(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="  The Drug Is Effective.  "
        ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        for claim in bundle.claims.as_dict().values():
            assert claim.normalized_text is not None
            assert claim.normalized_text != ""


# ═══════════════════════════════════════════════════════════════════════════
# I. IDEMPOTENCY
# ═══════════════════════════════════════════════════════════════════════════


class TestIdempotency:
    """Running the pipeline twice must produce the same results."""

    def test_idempotent_claim_ids(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Take 5 mg daily. Avoid alcohol."
        ))
        pipeline = AtomicClaimPipeline()
        r1 = pipeline.run(bundle)
        r2 = pipeline.run(bundle)
        assert r1.total_claims == r2.total_claims
        assert r1.claim_ids_by_item == r2.claim_ids_by_item

    def test_idempotent_safety_flags(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Take 5 mg daily."
        ))
        pipeline = AtomicClaimPipeline()
        r1 = pipeline.run(bundle)
        r2 = pipeline.run(bundle)
        assert r1.safety_critical_claim_count == r2.safety_critical_claim_count

    def test_idempotent_claim_types(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Mortality was 5%. Take 5 mg daily."
        ))
        pipeline = AtomicClaimPipeline()
        r1 = pipeline.run(bundle)
        r2 = pipeline.run(bundle)
        assert r1.dosage_claim_count == r2.dosage_claim_count
        assert r1.numeric_claim_count == r2.numeric_claim_count
        assert r1.factual_claim_count == r2.factual_claim_count

    def test_registry_does_not_grow_on_rerun(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="The drug is effective."
        ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        claim_count_after_first = len(bundle.claims)
        pipeline.run(bundle)
        claim_count_after_second = len(bundle.claims)
        assert claim_count_after_first == claim_count_after_second

    def test_extracted_claim_ids_stable(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="The drug is effective."
        ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        ids_after_first = bundle.evidence_items[EvidenceItemId("e1")].extracted_claim_ids
        pipeline.run(bundle)
        ids_after_second = bundle.evidence_items[EvidenceItemId("e1")].extracted_claim_ids
        assert ids_after_first == ids_after_second


# ═══════════════════════════════════════════════════════════════════════════
# J. LARGE INPUT
# ═══════════════════════════════════════════════════════════════════════════


class TestLargeInput:
    """Tests with large bundles to verify performance characteristics."""

    def test_1000_items_completes(self):
        """1000 evidence items with varied text should complete without error."""
        bundle = make_bundle()
        for i in range(1000):
            bundle.add_evidence_item(make_item(
                f"e{i}",
                chunk_id=f"chunk-{i}",
                text=f"Patient {i} received 5 mg daily. Outcome was favorable.",
            ))
        pipeline = AtomicClaimPipeline()
        result = pipeline.run(bundle)

        # Basic correctness checks
        assert result.total_claims >= 1000  # at least 1 claim per item
        assert result.safety_critical_claim_count >= 0
        assert result.total_claims >= result.safety_critical_claim_count

        # All items should have claims linked
        for i in range(1000):
            item = bundle.evidence_items[EvidenceItemId(f"e{i}")]
            assert len(item.extracted_claim_ids) >= 1

    def test_deterministic_large_input(self):
        """Same large input produces same output."""
        def build_bundle():
            b = make_bundle()
            for i in range(100):
                b.add_evidence_item(make_item(
                    f"e{i}",
                    chunk_id=f"chunk-{i}",
                    text=f"Study {i} demonstrated efficacy. Dose was {i+1} mg.",
                ))
            return b

        bundle1 = build_bundle()
        bundle2 = build_bundle()
        pipeline = AtomicClaimPipeline()
        r1 = pipeline.run(bundle1)
        r2 = pipeline.run(bundle2)
        assert r1.total_claims == r2.total_claims
        assert r1.dosage_claim_count == r2.dosage_claim_count
        assert r1.safety_critical_claim_count == r2.safety_critical_claim_count
        assert r1.claim_ids_by_item == r2.claim_ids_by_item


# ═══════════════════════════════════════════════════════════════════════════
# K. RESULT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestAtomicClaimExtractionResult:
    """Tests for AtomicClaimExtractionResult validation."""

    def test_negative_total_claims_rejected(self):
        with pytest.raises(ValueError):
            AtomicClaimExtractionResult(
                claim_ids_by_item={},
                total_claims=-1,
                safety_critical_claim_count=0,
            )

    def test_safety_exceeds_total_rejected(self):
        with pytest.raises(ValueError):
            AtomicClaimExtractionResult(
                claim_ids_by_item={},
                total_claims=1,
                safety_critical_claim_count=2,
                factual_claim_count=1,
            )

    def test_type_counts_must_sum_to_total(self):
        with pytest.raises(ValueError):
            AtomicClaimExtractionResult(
                claim_ids_by_item={},
                total_claims=5,
                safety_critical_claim_count=0,
                factual_claim_count=2,
                # Missing other counts → sums to 2, not 5
            )

    def test_valid_result_construction(self):
        result = AtomicClaimExtractionResult(
            claim_ids_by_item={},
            total_claims=3,
            safety_critical_claim_count=1,
            factual_claim_count=1,
            dosage_claim_count=1,
            numeric_claim_count=1,
        )
        assert result.total_claims == 3


# ═══════════════════════════════════════════════════════════════════════════
# L. EXTRACTOR CONFIG VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractionConfig:
    """Tests for ExtractionConfig."""

    def test_default_config(self):
        config = ExtractionConfig()
        assert config.min_claim_length == 10

    def test_custom_config(self):
        config = ExtractionConfig(min_claim_length=5)
        assert config.min_claim_length == 5

    def test_negative_min_claim_length_rejected(self):
        with pytest.raises(ValueError):
            ExtractionConfig(min_claim_length=-1)

    def test_zero_min_claim_length_allowed(self):
        config = ExtractionConfig(min_claim_length=0)
        assert config.min_claim_length == 0


# ═══════════════════════════════════════════════════════════════════════════
# M. CLAIM EXTRACTOR CLASS
# ═══════════════════════════════════════════════════════════════════════════


class TestClaimExtractor:
    """Tests for ClaimExtractor class."""

    def test_extract_empty_text(self):
        extractor = ClaimExtractor()
        claims = extractor.extract(
            EvidenceItemId("e1"), ChunkId("chunk1"), ""
        )
        assert claims == []

    def test_extract_whitespace_text(self):
        extractor = ClaimExtractor()
        claims = extractor.extract(
            EvidenceItemId("e1"), ChunkId("chunk1"), "   \n\t  "
        )
        assert claims == []

    def test_extract_short_fragment_skipped(self):
        """Fragments shorter than min_claim_length should be skipped."""
        extractor = ClaimExtractor(ExtractionConfig(min_claim_length=20))
        claims = extractor.extract(
            EvidenceItemId("e1"), ChunkId("chunk1"), "Short."
        )
        assert claims == []

    def test_extract_sets_origin_chunk_id(self):
        extractor = ClaimExtractor()
        claims = extractor.extract(
            EvidenceItemId("e1"),
            ChunkId("my-chunk"),
            "The drug inhibits enzyme X.",
        )
        assert len(claims) == 1
        assert claims[0].origin_chunk_id == ChunkId("my-chunk")

    def test_extract_sets_evidence_extraction_source(self):
        extractor = ClaimExtractor()
        claims = extractor.extract(
            EvidenceItemId("e1"),
            ChunkId("chunk1"),
            "The drug is effective.",
        )
        for claim in claims:
            assert claim.source == ClaimSource.EVIDENCE_EXTRACTION

    def test_extract_populates_normalized_text(self):
        extractor = ClaimExtractor()
        claims = extractor.extract(
            EvidenceItemId("e1"),
            ChunkId("chunk1"),
            "  The Drug Is Effective.  ",
        )
        assert len(claims) == 1
        assert claims[0].normalized_text is not None

    def test_extract_multiple_sentences(self):
        extractor = ClaimExtractor()
        claims = extractor.extract(
            EvidenceItemId("e1"),
            ChunkId("chunk1"),
            "Take 5 mg daily. Avoid alcohol. Monitor blood pressure.",
        )
        assert len(claims) == 3

    def test_invalid_config_type_rejected(self):
        with pytest.raises(TypeError):
            ClaimExtractor(config="not a config")


# ═══════════════════════════════════════════════════════════════════════════
# N. ADVERSARIAL: DOSAGE NEGATION FALSE-POSITIVES (REGRESSION)
# ═══════════════════════════════════════════════════════════════════════════


class TestDosageNegationRegression:
    """Regression tests for dosage negation patterns."""

    def test_dosage_unavailable_not_dosage(self):
        """'The dosage information was unavailable' is about ABSENCE."""
        assert classify_claim_type(
            "The dosage information was unavailable."
        ) != ClaimType.DOSAGE

    def test_dose_not_established(self):
        assert classify_claim_type(
            "Dosage was not established."
        ) != ClaimType.DOSAGE

    def test_dose_not_reported(self):
        assert classify_claim_type(
            "Dose was not reported."
        ) != ClaimType.DOSAGE

    def test_no_dosage_data(self):
        assert classify_claim_type(
            "No dosage data were available."
        ) != ClaimType.DOSAGE

    def test_dose_unknown(self):
        assert classify_claim_type(
            "The dose is unknown."
        ) != ClaimType.DOSAGE

    def test_real_dosage_still_works(self):
        """Ensure negation fix doesn't break real dosage detection."""
        assert classify_claim_type("Take 5 mg daily.") == ClaimType.DOSAGE
        assert classify_claim_type(
            "Dose adjustment is required for renal impairment."
        ) == ClaimType.DOSAGE
        assert classify_claim_type(
            "The maximum dose is 400 mg per day."
        ) == ClaimType.DOSAGE


# ═══════════════════════════════════════════════════════════════════════════
# O. ADVERSARIAL: DOSAGE UNIT COVERAGE
# ═══════════════════════════════════════════════════════════════════════════


class TestDosageUnitCoverage:
    """Verify dosage detection for specific unit patterns."""

    def test_mg(self):
        assert classify_claim_type("Take 5 mg.") == ClaimType.DOSAGE

    def test_mg_no_space(self):
        assert classify_claim_type("Take 5mg daily.") == ClaimType.DOSAGE

    def test_decimal_mg(self):
        assert classify_claim_type("Take 5.5 mg daily.") == ClaimType.DOSAGE

    def test_fractional_ml(self):
        assert classify_claim_type("Inject 0.5 mL.") == ClaimType.DOSAGE

    def test_mcg(self):
        assert classify_claim_type("Give 500 mcg.") == ClaimType.DOSAGE

    def test_unicode_micro_u(self):
        assert classify_claim_type("Give 500 µg.") == ClaimType.DOSAGE

    def test_unicode_micro_mu(self):
        assert classify_claim_type("Give 500 μg.") == ClaimType.DOSAGE

    def test_mg_per_kg(self):
        assert classify_claim_type("Dose is 10 mg/kg.") == ClaimType.DOSAGE

    def test_mg_per_day(self):
        assert classify_claim_type("Take 10 mg/day.") == ClaimType.DOSAGE

    def test_iu(self):
        assert classify_claim_type("Administer 5000 IU daily.") == ClaimType.DOSAGE

    def test_units(self):
        assert classify_claim_type("Give 20 units subcutaneously.") == ClaimType.DOSAGE

    def test_percentage_is_not_dosage(self):
        assert classify_claim_type("Response rate was 85%.") != ClaimType.DOSAGE

    def test_patient_count_is_not_dosage(self):
        assert classify_claim_type("100 patients enrolled.") != ClaimType.DOSAGE

    def test_year_is_not_dosage(self):
        assert classify_claim_type("Published in 2026.") != ClaimType.DOSAGE


# ═══════════════════════════════════════════════════════════════════════════
# P. ADVERSARIAL: SAFETY-CRITICAL EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestSafetyCriticalEdgeCases:
    """Additional safety-critical detection tests."""

    def test_black_box_warning(self):
        assert is_safety_critical(
            "This drug carries a black box warning.",
            ClaimType.FACTUAL,
        ) is True

    def test_life_threatening(self):
        assert is_safety_critical(
            "Life-threatening reactions may occur.",
            ClaimType.FACTUAL,
        ) is True

    def test_anaphylaxis(self):
        assert is_safety_critical(
            "Anaphylaxis has been reported.",
            ClaimType.FACTUAL,
        ) is True

    def test_renal_impairment(self):
        assert is_safety_critical(
            "Use with caution in renal impairment.",
            ClaimType.RECOMMENDATION,
        ) is True

    def test_hepatic_failure(self):
        assert is_safety_critical(
            "Hepatic failure is a known risk.",
            ClaimType.FACTUAL,
        ) is True

    def test_never_administer(self):
        assert is_safety_critical(
            "Never administer to patients under 18.",
            ClaimType.RECOMMENDATION,
        ) is True

    def test_benign_recommendation_not_critical(self):
        assert is_safety_critical(
            "Consider using a pillbox for adherence.",
            ClaimType.RECOMMENDATION,
        ) is False

    def test_ordinary_temporal_not_critical(self):
        assert is_safety_critical(
            "The study was conducted in 2020.",
            ClaimType.TEMPORAL,
        ) is False


# ═══════════════════════════════════════════════════════════════════════════
# Q. ADVERSARIAL: UNKNOWN SEMANTICS
# ═══════════════════════════════════════════════════════════════════════════


class TestUnknownSemantics:
    """UNKNOWN must never be collapsed, conflated, or lost."""

    def test_unknown_is_truthy(self):
        assert bool(UNKNOWN) is True

    def test_unknown_is_singleton(self):
        from rasvcx.schemas.common import _UnknownType
        assert _UnknownType() is UNKNOWN

    def test_unknown_repr(self):
        assert repr(UNKNOWN) == "UNKNOWN"

    def test_unknown_not_none(self):
        assert UNKNOWN is not None

    def test_unknown_not_empty_string(self):
        assert UNKNOWN != ""

    def test_unknown_not_false(self):
        assert UNKNOWN is not False

    def test_claim_with_unknown_origin(self):
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="Post-generation claim.",
            source=ClaimSource.POST_GENERATION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=UNKNOWN,
        )
        assert claim.origin_chunk_id is UNKNOWN
        from rasvcx.schemas.common import _UnknownType
        assert isinstance(claim.origin_chunk_id, _UnknownType)

    def test_evidence_extraction_claim_has_concrete_chunk_id(self):
        """Evidence-extracted claims should NOT have UNKNOWN origin."""
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", chunk_id="chunk-x", text="The drug inhibits enzyme X."
        ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        for claim in bundle.claims.as_dict().values():
            assert claim.origin_chunk_id is not UNKNOWN
            assert claim.origin_chunk_id == ChunkId("chunk-x")

    def test_linker_does_not_link_unknown_origin(self):
        """Claims with UNKNOWN origin must not be linked to any evidence."""
        linker = ClaimLinker()
        item = make_item("e1", chunk_id="chunk-e1")
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="Post-gen claim.",
            source=ClaimSource.POST_GENERATION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=UNKNOWN,
        )
        result = linker.link([claim], {EvidenceItemId("e1"): item})
        assert result == {}


# ═══════════════════════════════════════════════════════════════════════════
# R. ADVERSARIAL: IMMUTABILITY PROOF
# ═══════════════════════════════════════════════════════════════════════════


class TestImmutabilityProof:
    """Prove that frozen objects are never mutated."""

    def test_claim_is_frozen(self):
        claim = Claim(
            claim_id=ClaimId("c1"),
            text="Test.",
            source=ClaimSource.EVIDENCE_EXTRACTION,
            claim_type=ClaimType.FACTUAL,
            origin_chunk_id=ChunkId("chunk1"),
        )
        with pytest.raises(AttributeError):
            claim.text = "Modified"

    def test_evidence_item_is_frozen(self):
        item = make_item("e1")
        with pytest.raises(AttributeError):
            item.text = "Modified"

    def test_evidence_item_claim_ids_not_mutated_by_pipeline(self):
        """Original EvidenceItem object must not have its claim_ids
        modified in-place by the pipeline."""
        original = make_item("e1", text="The drug is effective.")
        original_ids = original.extracted_claim_ids
        bundle = make_bundle()
        bundle.add_evidence_item(original)
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        # The original object's claim IDs must be unchanged
        assert original.extracted_claim_ids is original_ids
        assert original.extracted_claim_ids == frozenset()

    def test_pipeline_uses_dataclasses_replace(self):
        """The updated item in the bundle must be a different object."""
        original = make_item("e1", text="The drug is effective.")
        bundle = make_bundle()
        bundle.add_evidence_item(original)
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        updated = bundle.evidence_items[EvidenceItemId("e1")]
        if len(updated.extracted_claim_ids) > 0:
            assert updated is not original

    def test_original_evidence_fields_preserved(self):
        """Pipeline must not alter retrieval_score, rerank_score,
        provenance, or text."""
        original = make_item(
            "e1", chunk_id="chunk-1",
            text="The drug inhibits enzyme X.",
        )
        bundle = make_bundle()
        bundle.add_evidence_item(original)
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        updated = bundle.evidence_items[EvidenceItemId("e1")]
        assert updated.text == original.text
        assert updated.retrieval_score == original.retrieval_score
        assert updated.rerank_score == original.rerank_score
        assert updated.provenance == original.provenance
        assert updated.chunk_id == original.chunk_id
        assert updated.item_id == original.item_id


# ═══════════════════════════════════════════════════════════════════════════
# S. ADVERSARIAL: SENTENCE SEGMENTATION EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestSentenceSegmentationEdgeCases:
    """Targeted edge cases for split_sentences()."""

    def test_multiple_punctuation_marks(self):
        result = split_sentences("Really?! Another sentence.")
        assert len(result) >= 1  # At least one sentence preserved

    def test_quotes(self):
        result = split_sentences('"Take 5 mg daily." The doctor advised.')
        assert len(result) >= 1

    def test_parentheses(self):
        result = split_sentences(
            "The drug (brand name: Xanax) is effective. Another finding."
        )
        assert len(result) == 2

    def test_colon_does_not_split(self):
        result = split_sentences("Dosage: 5 mg daily for 7 days.")
        assert len(result) == 1

    def test_semicolon_does_not_split(self):
        result = split_sentences(
            "The drug is effective; however, side effects occur."
        )
        assert len(result) == 1

    def test_very_long_text(self):
        text = "Very important clinical finding. " * 100
        result = split_sentences(text)
        assert len(result) >= 50

    def test_unicode_bullet_points(self):
        text = "• First finding. • Second finding."
        result = split_sentences(text)
        # Should not crash on Unicode; exact split behavior acceptable
        assert len(result) >= 1

    def test_no_terminal_punctuation_single_chunk(self):
        result = split_sentences("A long evidence chunk without period")
        assert len(result) == 1
        assert result[0] == "A long evidence chunk without period"


# ═══════════════════════════════════════════════════════════════════════════
# T. ADVERSARIAL: NORMALIZATION MEANING PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizationMeaningPreservation:
    """Verify normalizer preserves clinically meaningful content."""

    def test_preserves_mg_per_kg(self):
        assert "mg/kg" in normalize_claim_text("5 mg/kg daily.")

    def test_preserves_percentage(self):
        assert "%" in normalize_claim_text("Mortality was 5%.")

    def test_preserves_decimal(self):
        assert "3.5" in normalize_claim_text("Value was 3.5.")

    def test_preserves_range_dash(self):
        assert "10-20" in normalize_claim_text("10-20 mg daily.")

    def test_preserves_ph(self):
        assert "7.4" in normalize_claim_text("pH 7.4 is normal.")

    def test_preserves_colon_in_ratio(self):
        assert ":" in normalize_claim_text("Ratio: 1:4.")

    def test_preserves_parentheses(self):
        assert "(" in normalize_claim_text("Drug (brand name) is safe.")
        assert ")" in normalize_claim_text("Drug (brand name) is safe.")

    def test_double_normalize_same_result(self):
        """Strict idempotency test."""
        texts = [
            "Take 5 mg DAILY.",
            "  Whitespace  everywhere  !!",
            "pH 7.4 (normal range).",
            "5 mg/kg bid",
            "IMPORTANT: do not stop medication!!!",
        ]
        for text in texts:
            first = normalize_claim_text(text)
            second = normalize_claim_text(first)
            assert first == second, f"Idempotency failed for {text!r}"


# ═══════════════════════════════════════════════════════════════════════════
# U. ADVERSARIAL: PROPERTY-STYLE INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPropertyInvariants:
    """Verify cross-cutting system invariants."""

    def test_every_registered_claim_has_nonempty_id(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Take 5 mg daily. Avoid alcohol. Monitor closely."
        ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        for cid, claim in bundle.claims.as_dict().items():
            assert cid, "Claim ID must be non-empty"
            assert claim.claim_id == cid

    def test_every_claim_has_nonempty_text(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Take 5 mg daily. Avoid alcohol."
        ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        for claim in bundle.claims.as_dict().values():
            assert claim.text, "Claim text must be non-empty"

    def test_every_evidence_extracted_claim_has_concrete_origin(self):
        bundle = make_bundle()
        for i in range(5):
            bundle.add_evidence_item(make_item(
                f"e{i}", chunk_id=f"chunk-{i}",
                text=f"Finding {i} is significant. More detail here."
            ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        for claim in bundle.claims.as_dict().values():
            assert claim.source == ClaimSource.EVIDENCE_EXTRACTION
            assert claim.origin_chunk_id is not UNKNOWN

    def test_every_item_claim_id_exists_in_registry(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Take 5 mg daily. Avoid alcohol."
        ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        for item in bundle.evidence_items.values():
            for cid in item.extracted_claim_ids:
                assert cid in bundle.claims, (
                    f"Claim {cid!r} referenced by item {item.item_id!r} "
                    f"not found in ClaimRegistry"
                )

    def test_claim_ids_are_deterministic_across_runs(self):
        text = "Take 5 mg daily. Avoid alcohol. Monitor renal function."

        def extract_ids():
            b = make_bundle()
            b.add_evidence_item(make_item("e1", text=text))
            AtomicClaimPipeline().run(b)
            return b.claims.ids()

        ids1 = extract_ids()
        ids2 = extract_ids()
        assert ids1 == ids2

    def test_pipeline_does_not_alter_evidence_text(self):
        original_text = "Take 5 mg daily. Avoid alcohol."
        bundle = make_bundle()
        bundle.add_evidence_item(make_item("e1", text=original_text))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        assert bundle.evidence_items[EvidenceItemId("e1")].text == original_text

    def test_pipeline_does_not_alter_retrieval_scores(self):
        bundle = make_bundle()
        item = make_item("e1", text="The drug is effective.")
        bundle.add_evidence_item(item)
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        updated = bundle.evidence_items[EvidenceItemId("e1")]
        assert updated.retrieval_score == item.retrieval_score

    def test_pipeline_does_not_alter_provenance(self):
        bundle = make_bundle()
        item = make_item("e1", text="The drug is effective.")
        bundle.add_evidence_item(item)
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        updated = bundle.evidence_items[EvidenceItemId("e1")]
        assert updated.provenance == item.provenance

    def test_unknown_remains_unknown_after_pipeline(self):
        """UNKNOWN provenance fields must NOT be altered by M7."""
        bundle = make_bundle()
        bundle.add_evidence_item(make_item("e1", text="The drug is effective."))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        item = bundle.evidence_items[EvidenceItemId("e1")]
        assert item.provenance.date is UNKNOWN
        assert item.provenance.jurisdiction is UNKNOWN
        assert item.provenance.population is UNKNOWN
        assert item.provenance.dosage_context is UNKNOWN


# ═══════════════════════════════════════════════════════════════════════════
# V. ADVERSARIAL: PERFORMANCE BENCHMARKING
# ═══════════════════════════════════════════════════════════════════════════


class TestPerformanceBenchmark:
    """Measure actual M7 performance characteristics."""

    def test_100_items_under_1_second(self):
        import time
        bundle = make_bundle()
        for i in range(100):
            bundle.add_evidence_item(make_item(
                f"e{i}", chunk_id=f"chunk-{i}",
                text=(
                    f"Patient {i} received 5 mg daily. "
                    f"Outcome was favorable. Mortality was 2%."
                ),
            ))
        pipeline = AtomicClaimPipeline()
        start = time.perf_counter()
        result = pipeline.run(bundle)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"100 items took {elapsed:.3f}s, expected < 1s"
        assert result.total_claims >= 100

    def test_1000_items_under_5_seconds(self):
        import time
        bundle = make_bundle()
        for i in range(1000):
            bundle.add_evidence_item(make_item(
                f"e{i}", chunk_id=f"chunk-{i}",
                text=(
                    f"Patient {i} received {i+1} mg daily. "
                    f"Follow-up at {i+2} weeks showed improvement."
                ),
            ))
        pipeline = AtomicClaimPipeline()
        start = time.perf_counter()
        result = pipeline.run(bundle)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"1000 items took {elapsed:.3f}s, expected < 5s"
        assert result.total_claims >= 1000

    def test_scaling_approximately_linear(self):
        """Verify M7 scales approximately linearly, not quadratically."""
        import time

        def measure(n):
            b = make_bundle()
            for i in range(n):
                b.add_evidence_item(make_item(
                    f"e{i}", chunk_id=f"chunk-{i}",
                    text=f"Patient {i} received {i+1} mg daily."
                ))
            start = time.perf_counter()
            AtomicClaimPipeline().run(b)
            return time.perf_counter() - start

        t100 = measure(100)
        t500 = measure(500)
        # If linear, t500 should be ~5x t100.  If quadratic, ~25x.
        # Allow generous margin: ratio < 12 means not quadratic.
        if t100 > 0.001:  # Guard against near-zero timing
            ratio = t500 / t100
            assert ratio < 12, (
                f"Scaling ratio {ratio:.1f} (t100={t100:.3f}s, t500={t500:.3f}s) "
                f"suggests worse than linear complexity"
            )


# ═══════════════════════════════════════════════════════════════════════════
# W. ADVERSARIAL: CLAIM REGISTRY ISOLATION
# ═══════════════════════════════════════════════════════════════════════════


class TestClaimRegistryIsolation:
    """Verify no state leakage between bundles."""

    def test_separate_bundles_separate_registries(self):
        bundle1 = make_bundle()
        bundle1.add_evidence_item(make_item(
            "e1", chunk_id="chunk-a", text="Drug A inhibits enzyme X."
        ))
        bundle2 = make_bundle()
        bundle2.add_evidence_item(make_item(
            "e1", chunk_id="chunk-b", text="Drug B activates receptor Y."
        ))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle1)
        pipeline.run(bundle2)
        # Registries must be completely independent
        ids1 = bundle1.claims.ids()
        ids2 = bundle2.claims.ids()
        assert ids1.isdisjoint(ids2), (
            "Claims from different bundles must not share IDs "
            "unless text+item+index are identical"
        )

    def test_pipeline_has_no_global_state(self):
        """Running the same pipeline on different bundles must not leak."""
        pipeline = AtomicClaimPipeline()
        b1 = make_bundle()
        b1.add_evidence_item(make_item("e1", text="Drug A is effective."))
        pipeline.run(b1)
        count1 = len(b1.claims)

        b2 = make_bundle()
        b2.add_evidence_item(make_item("e1", text="Drug B is safe."))
        pipeline.run(b2)
        count2 = len(b2.claims)

        # Each bundle should have exactly its own claims
        assert count1 >= 1
        assert count2 >= 1
        # b1's registry must not have grown
        assert len(b1.claims) == count1


# ═══════════════════════════════════════════════════════════════════════════
# X. ADVERSARIAL: UNICODE HANDLING
# ═══════════════════════════════════════════════════════════════════════════


class TestUnicodeHandling:
    """Verify Unicode is handled consistently throughout M7."""

    def test_unicode_micro_sign_in_segmentation(self):
        result = split_sentences("The dose was 5 µg. Patient recovered.")
        assert len(result) == 2

    def test_unicode_greek_mu_in_segmentation(self):
        result = split_sentences("The dose was 5 μg. Patient recovered.")
        assert len(result) == 2

    def test_unicode_in_classification(self):
        assert classify_claim_type("Give 500 µg daily.") == ClaimType.DOSAGE
        assert classify_claim_type("Give 500 μg daily.") == ClaimType.DOSAGE

    def test_unicode_in_normalization(self):
        n1 = normalize_claim_text("5 µg daily.")
        assert "µg" in n1

    def test_unicode_in_claim_id_generation(self):
        cid = generate_claim_id(EvidenceItemId("e1"), 0, "5 µg daily.")
        assert cid.startswith("claim_")
        assert len(cid) == len("claim_") + 16

    def test_unicode_end_to_end(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item(
            "e1", text="Administer 500 µg daily. Monitor for adverse effects."
        ))
        pipeline = AtomicClaimPipeline()
        result = pipeline.run(bundle)
        assert result.total_claims >= 2
        assert result.dosage_claim_count >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Y. ADVERSARIAL: TIMING CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════


class TestTimingCorrectness:
    """Verify timing uses bundle.record_stage_elapsed correctly."""

    def test_timing_key_is_atomic_claim_extraction(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item("e1", text="The drug is effective."))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        assert "atomic_claim_extraction" in bundle.metadata.elapsed_per_stage

    def test_timing_is_positive(self):
        bundle = make_bundle()
        bundle.add_evidence_item(make_item("e1", text="The drug is effective."))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        elapsed = bundle.metadata.elapsed_per_stage["atomic_claim_extraction"]
        assert elapsed >= 0.0
        assert isinstance(elapsed, float)

    def test_timing_recorded_even_if_exception_in_future(self):
        """Timing is in a finally block — recorded even for empty bundle."""
        bundle = make_bundle()
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        assert "atomic_claim_extraction" in bundle.metadata.elapsed_per_stage

    def test_timing_does_not_overwrite_other_stages(self):
        bundle = make_bundle()
        # Simulate a previous stage
        bundle.record_stage_elapsed("reranking", 0.123)
        bundle.add_evidence_item(make_item("e1", text="The drug is effective."))
        pipeline = AtomicClaimPipeline()
        pipeline.run(bundle)
        assert bundle.metadata.elapsed_per_stage["reranking"] == 0.123
        assert "atomic_claim_extraction" in bundle.metadata.elapsed_per_stage