"""Tests for the M6 provenance/context analysis package."""

import pytest

from rasvcx.schemas.common import UNKNOWN, ChunkId, EvidenceItemId, QueryId, SourceType
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import QueryRequest, RiskFeatureScores, RiskProfile, ValidationDepth

from rasvcx.provenance import (
    ApplicabilityLabel,
    CompatibilityVerdict,
    ContextExtractor,
    ProvenanceAnalyzer,
    SourceQualityScorer,
    TemporalRelevanceLabel,
    compute_temporal_signal,
    compare_context_field,
)


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


def make_item(item_id="e1", **prov_overrides):
    return EvidenceItem(
        item_id=EvidenceItemId(item_id),
        chunk_id=ChunkId(f"chunk-{item_id}"),
        text="some evidence text",
        retrieval_score=0.9,
        provenance=make_provenance(**prov_overrides),
    )


def make_risk_profile(safety_floor_forced=False):
    return RiskProfile(
        overall_risk_score=0.5,
        feature_scores=RiskFeatureScores(),
        validation_depth=list(ValidationDepth)[0],
        retrieval_retry_budget=1,
        nli_call_allowance=1,
        safety_floor_forced=safety_floor_forced,
    )


def make_query(**metadata):
    return QueryRequest(
        query_id=QueryId("q1"),
        raw_text="test query",
        normalized_text="test query",
        metadata=metadata,
    )


def make_bundle(risk_profile=None):
    return EvidenceBundle(
        query_id=QueryId("q1"),
        risk_profile=risk_profile or make_risk_profile(),
    )


def make_analyzer():
    return ProvenanceAnalyzer(SourceQualityScorer(), ContextExtractor())


def test_compare_context_field_unknown_vs_unknown():
    signal = compare_context_field("population", UNKNOWN, UNKNOWN)
    assert signal.label == ApplicabilityLabel.UNKNOWN


def test_compare_context_field_known_match():
    signal = compare_context_field("population", "adults", "Adults")
    assert signal.label == ApplicabilityLabel.MATCH


def test_compare_context_field_known_mismatch():
    signal = compare_context_field("population", "adults", "pediatric")
    assert signal.label == ApplicabilityLabel.MISMATCH


def test_compare_context_field_one_side_unknown_is_unknown_not_mismatch():
    signal = compare_context_field("population", "adults", UNKNOWN)
    assert signal.label == ApplicabilityLabel.UNKNOWN


def test_compare_context_field_empty_string_treated_as_unknown():
    signal = compare_context_field("population", "   ", "adults")
    assert signal.label == ApplicabilityLabel.UNKNOWN


def test_extract_unknown_evidence_field_stays_unknown_regardless_of_query():
    item = make_item(jurisdiction=UNKNOWN)
    query = make_query(jurisdiction="US")
    result = ContextExtractor().extract(item, query)
    assert result.jurisdiction == ApplicabilityLabel.UNKNOWN


def test_extract_query_has_no_constraint_is_unknown_not_match():
    item = make_item(population="adults")
    query = make_query()
    result = ContextExtractor().extract(item, query)
    assert result.population == ApplicabilityLabel.UNKNOWN


def test_extract_known_match():
    item = make_item(population="adults")
    query = make_query(population="adults")
    result = ContextExtractor().extract(item, query)
    assert result.population == ApplicabilityLabel.MATCH


def test_extract_known_mismatch():
    item = make_item(population="adults")
    query = make_query(population="pediatric")
    result = ContextExtractor().extract(item, query)
    assert result.population == ApplicabilityLabel.MISMATCH


def test_extract_temporal_unknown_date():
    item = make_item(date=UNKNOWN)
    query = make_query()
    result = ContextExtractor().extract(item, query)
    assert result.temporal == ApplicabilityLabel.UNKNOWN


def test_extract_temporal_current_date_is_match():
    from datetime import date
    item = make_item(date="2024-06-01")
    query = make_query()
    extractor = ContextExtractor(reference_date=date(2024, 7, 1))
    result = extractor.extract(item, query)
    assert result.temporal == ApplicabilityLabel.MATCH


def test_extract_temporal_stale_date_is_mismatch():
    from datetime import date
    item = make_item(date="2010-01-01")
    query = make_query()
    extractor = ContextExtractor(reference_date=date(2024, 1, 1))
    result = extractor.extract(item, query)
    assert result.temporal == ApplicabilityLabel.MISMATCH


def test_extract_temporal_uncertain_precision_is_unknown_not_mismatch():
    item = make_item(date="not-a-date")
    query = make_query()
    result = ContextExtractor().extract(item, query)
    assert result.temporal == ApplicabilityLabel.UNKNOWN


def test_extract_does_not_mutate_item():
    item = make_item(population="adults")
    query = make_query(population="pediatric")
    before = item.provenance
    ContextExtractor().extract(item, query)
    assert item.provenance is before


def test_empty_bundle():
    bundle = make_bundle()
    result = make_analyzer().analyze(bundle, make_query(), make_risk_profile())
    assert result.per_item == {}
    assert result.unique_source_count == 0
    assert result.unknown_provenance_ratio == 0.0
    assert result.mismatch_present is False


def test_single_item_no_dropped_items():
    bundle = make_bundle()
    bundle.add_evidence_item(make_item("e1"))
    result = make_analyzer().analyze(bundle, make_query(), make_risk_profile())
    assert set(result.per_item.keys()) == {"e1"}


def test_all_input_items_represented():
    bundle = make_bundle()
    for i in range(5):
        bundle.add_evidence_item(make_item(f"e{i}"))
    result = make_analyzer().analyze(bundle, make_query(), make_risk_profile())
    assert set(result.per_item.keys()) == {f"e{i}" for i in range(5)}


def test_known_source_type_preserved_and_is_known_true():
    bundle = make_bundle()
    bundle.add_evidence_item(make_item("e1", source_type=SourceType.PEER_REVIEWED_LITERATURE))
    result = make_analyzer().analyze(bundle, make_query(), make_risk_profile())
    analysis = result.per_item["e1"]
    assert analysis.is_known is True
    assert analysis.source_type == SourceType.PEER_REVIEWED_LITERATURE


def test_mismatch_present_true_when_any_field_mismatches():
    bundle = make_bundle()
    bundle.add_evidence_item(make_item("e1", population="adults"))
    result = make_analyzer().analyze(bundle, make_query(population="pediatric"), make_risk_profile())
    assert result.mismatch_present is True
    assert result.per_item["e1"].population == CompatibilityVerdict.KNOWN_MISMATCH


def test_mismatch_present_false_when_all_unknown():
    bundle = make_bundle()
    bundle.add_evidence_item(make_item("e1"))
    result = make_analyzer().analyze(bundle, make_query(), make_risk_profile())
    assert result.mismatch_present is False
    assert result.per_item["e1"].unknown_field_count == 4


def test_unknown_provenance_ratio_counts_items_with_any_unknown_field():
    bundle = make_bundle()
    bundle.add_evidence_item(
        make_item("e1", date="2024-01-01", population="adults", jurisdiction="US", dosage_context="oral")
    )
    bundle.add_evidence_item(make_item("e2"))
    result = make_analyzer().analyze(
        bundle,
        make_query(population="adults", jurisdiction="US", dosage_context="oral"),
        make_risk_profile(),
    )
    assert result.unknown_provenance_ratio == 0.5


def test_strict_context_required_reflects_safety_floor_forced():
    bundle = make_bundle(risk_profile=make_risk_profile(safety_floor_forced=True))
    result = make_analyzer().analyze(bundle, make_query(), make_risk_profile(safety_floor_forced=True))
    assert result.strict_context_required is True


def test_unique_source_count_groups_by_source_type_fallback():
    bundle = make_bundle()
    bundle.add_evidence_item(make_item("e1", source_type=SourceType.DRUG_LABEL))
    bundle.add_evidence_item(make_item("e2", source_type=SourceType.DRUG_LABEL))
    bundle.add_evidence_item(make_item("e3", source_type=SourceType.PEER_REVIEWED_LITERATURE))
    result = make_analyzer().analyze(bundle, make_query(), make_risk_profile())
    assert result.unique_source_count == 2


def test_timing_recorded_on_bundle():
    bundle = make_bundle()
    bundle.add_evidence_item(make_item("e1"))
    make_analyzer().analyze(bundle, make_query(), make_risk_profile())
    assert "provenance_context" in bundle.metadata.elapsed_per_stage
    assert bundle.metadata.elapsed_per_stage["provenance_context"] >= 0.0


def test_analyze_does_not_mutate_evidence_items():
    bundle = make_bundle()
    item = make_item("e1", population="adults")
    bundle.add_evidence_item(item)
    before = item.provenance
    make_analyzer().analyze(bundle, make_query(population="pediatric"), make_risk_profile())
    assert bundle.evidence_items["e1"].provenance is before


def test_deterministic_repeated_calls():
    bundle = make_bundle()
    bundle.add_evidence_item(make_item("e1", population="adults", jurisdiction="US"))
    query = make_query(population="adults", jurisdiction="EU")
    analyzer = make_analyzer()
    r1 = analyzer.analyze(bundle, query, make_risk_profile())
    r2 = analyzer.analyze(bundle, query, make_risk_profile())
    assert r1.per_item["e1"].population == r2.per_item["e1"].population
    assert r1.per_item["e1"].jurisdiction == r2.per_item["e1"].jurisdiction
    assert r1.mismatch_present == r2.mismatch_present


def test_large_bundle_all_items_present():
    bundle = make_bundle()
    for i in range(500):
        bundle.add_evidence_item(make_item(f"e{i}"))
    result = make_analyzer().analyze(bundle, make_query(), make_risk_profile())
    assert len(result.per_item) == 500


def test_item_provenance_analysis_rejects_quality_score_out_of_bounds():
    with pytest.raises(ValueError):
        from rasvcx.provenance.analyzer import ItemProvenanceAnalysis
        ItemProvenanceAnalysis(
            item_id=EvidenceItemId("e1"),
            source_type=SourceType.DRUG_LABEL,
            is_known=True,
            quality_score=1.5,
            temporal=CompatibilityVerdict.UNKNOWN,
            jurisdiction=CompatibilityVerdict.UNKNOWN,
            population=CompatibilityVerdict.UNKNOWN,
            dosage_context=CompatibilityVerdict.UNKNOWN,
        )


def test_item_provenance_analysis_rejects_is_known_true_with_none_source_type():
    with pytest.raises(ValueError):
        from rasvcx.provenance.analyzer import ItemProvenanceAnalysis
        ItemProvenanceAnalysis(
            item_id=EvidenceItemId("e1"),
            source_type=None,
            is_known=True,
            quality_score=0.5,
            temporal=CompatibilityVerdict.UNKNOWN,
            jurisdiction=CompatibilityVerdict.UNKNOWN,
            population=CompatibilityVerdict.UNKNOWN,
            dosage_context=CompatibilityVerdict.UNKNOWN,
        )


def test_item_provenance_analysis_rejects_is_known_false_with_concrete_source_type():
    with pytest.raises(ValueError):
        from rasvcx.provenance.analyzer import ItemProvenanceAnalysis
        ItemProvenanceAnalysis(
            item_id=EvidenceItemId("e1"),
            source_type=SourceType.DRUG_LABEL,
            is_known=False,
            quality_score=0.5,
            temporal=CompatibilityVerdict.UNKNOWN,
            jurisdiction=CompatibilityVerdict.UNKNOWN,
            population=CompatibilityVerdict.UNKNOWN,
            dosage_context=CompatibilityVerdict.UNKNOWN,
        )


def test_coerce_verdict_rejects_unrecognized_value():
    from rasvcx.provenance.analyzer import _coerce_verdict
    with pytest.raises(ValueError):
        _coerce_verdict("not_a_verdict")


def test_provenance_analysis_result_rejects_ratio_out_of_bounds():
    from rasvcx.provenance.analyzer import ProvenanceAnalysisResult
    with pytest.raises(ValueError):
        ProvenanceAnalysisResult(
            per_item={},
            unique_source_count=0,
            unknown_provenance_ratio=1.5,
            mismatch_present=False,
            strict_context_required=False,
        )