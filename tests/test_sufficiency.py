"""Tests for Module 5 — Sufficiency Gate.

Covers:
  - scorer.py: Pure signal extraction, UNKNOWN ratio calc, margins.
  - gate.py: Config bounds, verdict branching (SUFFICIENT vs INSUFFICIENT vs CONSERVATIVE).
  - High-risk vs Low-risk threshold scaling.
  - Pipeline timing recording.
"""
from __future__ import annotations

import pytest

from rasvcx.schemas.common import ChunkId, EvidenceItemId, QueryId, SourceType, UNKNOWN
from rasvcx.schemas.evidence import EvidenceBundle, EvidenceItem, Provenance
from rasvcx.schemas.query import RiskFeatureScores, RiskProfile, ValidationDepth
from rasvcx.sufficiency.gate import (
    SufficiencyGateConfig,
    SufficiencyResult,
    SufficiencyVerdict,
    evaluate_sufficiency,
)
from rasvcx.sufficiency.scorer import SufficiencySignals, compute_sufficiency_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _risk(
    depth: ValidationDepth = ValidationDepth.STANDARD,
    score: float = 0.5,
    safety_forced: bool = False
) -> RiskProfile:
    return RiskProfile(
        overall_risk_score=score,
        feature_scores=RiskFeatureScores(),
        validation_depth=depth,
        retrieval_retry_budget=1,
        nli_call_allowance=5,
        safety_floor_forced=safety_forced,
    )

def _prov(
    date: str | type[UNKNOWN] = "2023-01-01",
    jur: str | type[UNKNOWN] = "US",
    pop: str | type[UNKNOWN] = "Adults",
    dos: str | type[UNKNOWN] = "10mg",
    src: SourceType = SourceType.CLINICAL_GUIDELINE
) -> Provenance:
    return Provenance(
        source_type=src, date=date, jurisdiction=jur, population=pop, dosage_context=dos
    )

def _item(
    id_str: str,
    ret_score: float = 1.0,
    rr_score: float | None = 0.9,
    prov: Provenance | None = None
) -> EvidenceItem:
    return EvidenceItem(
        item_id=EvidenceItemId(id_str),
        chunk_id=ChunkId(id_str),
        text="text",
        retrieval_score=ret_score,
        rerank_score=rr_score,
        provenance=prov or _prov(),
    )

def _bundle(*items: EvidenceItem, risk: RiskProfile | None = None, targeted_used: bool = False) -> EvidenceBundle:
    b = EvidenceBundle(query_id=QueryId("q1"), risk_profile=risk or _risk())
    for it in items:
        b.add_evidence_item(it)
    if targeted_used:
        b.record_targeted_retrieval_used()
    return b


# ===========================================================================
# 1. Scorer Tests
# ===========================================================================

class TestScorer:
    def test_empty_bundle(self):
        b = _bundle()
        sig = compute_sufficiency_signals(b, b.risk_profile)
        assert sig.evidence_item_count == 0
        assert sig.scored_item_count == 0
        assert sig.top_rerank_score is None
        assert sig.rerank_score_margin is None
        assert sig.mean_retrieval_score is None
        assert sig.source_type_diversity == 0
        assert sig.unknown_provenance_ratio == 0.0
        assert sig.unscored_item_ratio == 0.0

    def test_single_item(self):
        b = _bundle(_item("1", ret_score=0.8, rr_score=0.9))
        sig = compute_sufficiency_signals(b, b.risk_profile)
        assert sig.evidence_item_count == 1
        assert sig.scored_item_count == 1
        assert sig.top_rerank_score == 0.9
        assert sig.rerank_score_margin is None
        assert sig.mean_retrieval_score == 0.8
        assert sig.unscored_item_ratio == 0.0

    def test_margin_calculation(self):
        b = _bundle(
            _item("1", rr_score=0.9),
            _item("2", rr_score=0.7),
            _item("3", rr_score=0.8)
        )
        sig = compute_sufficiency_signals(b, b.risk_profile)
        assert sig.top_rerank_score == 0.9
        assert sig.rerank_score_margin == pytest.approx(0.1) # 0.9 - 0.8

    def test_unknown_provenance_ratio(self):
        # 4 fields per item. 
        # Item 1: 0 unknowns
        # Item 2: 2 unknowns (date, population)
        # Total fields = 8, Unknowns = 2 -> Ratio = 0.25
        i1 = _item("1", prov=_prov())
        i2 = _item("2", prov=_prov(date=UNKNOWN, pop=UNKNOWN))
        
        b = _bundle(i1, i2)
        sig = compute_sufficiency_signals(b, b.risk_profile)
        assert sig.unknown_provenance_ratio == 0.25

    def test_source_diversity(self):
        b = _bundle(
            _item("1", prov=_prov(src=SourceType.CLINICAL_GUIDELINE)),
            _item("2", prov=_prov(src=SourceType.DRUG_LABEL)),
            _item("3", prov=_prov(src=SourceType.CLINICAL_GUIDELINE))
        )
        sig = compute_sufficiency_signals(b, b.risk_profile)
        assert sig.source_type_diversity == 2

    def test_unscored_items(self):
        b = _bundle(
            _item("1", rr_score=0.9),
            _item("2", rr_score=None),
            _item("3", rr_score=None)
        )
        sig = compute_sufficiency_signals(b, b.risk_profile)
        assert sig.scored_item_count == 1
        assert sig.unscored_item_ratio == pytest.approx(2/3)


# ===========================================================================
# 2. Gate Configuration Tests
# ===========================================================================

class TestGateConfig:
    def test_valid_config(self):
        cfg = SufficiencyGateConfig()
        assert cfg.min_evidence_items == 2

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            SufficiencyGateConfig(min_evidence_items=-1)
        with pytest.raises(ValueError):
            SufficiencyGateConfig(high_risk_score_threshold=1.5)


class TestSufficiencyResult:
    def test_sufficient_cannot_recommend_retrieval(self):
        sig = compute_sufficiency_signals(_bundle(), _risk())
        with pytest.raises(ValueError, match="must not recommend"):
            SufficiencyResult(
                verdict=SufficiencyVerdict.SUFFICIENT,
                signals=sig,
                recommend_targeted_retrieval=True
            )

    def test_insufficient_must_have_reasons(self):
        sig = compute_sufficiency_signals(_bundle(), _risk())
        with pytest.raises(ValueError, match="requires at least one reason"):
            SufficiencyResult(
                verdict=SufficiencyVerdict.INSUFFICIENT,
                signals=sig,
                reasons=()
            )


# ===========================================================================
# 3. Gate Logic Tests
# ===========================================================================

class TestGateLogic:
    def setup_method(self):
        self.cfg = SufficiencyGateConfig(
            min_evidence_items=2,
            min_scored_items=1,
            min_top_rerank_score=0.5,
            high_risk_score_threshold=0.7,
            max_unknown_provenance_ratio_high_risk=0.5
        )

    def test_low_risk_sufficient(self):
        b = _bundle(_item("1", rr_score=0.8), _item("2", rr_score=0.6), risk=_risk(score=0.1))
        res = evaluate_sufficiency(b, b.risk_profile, self.cfg)
        assert res.verdict == SufficiencyVerdict.SUFFICIENT
        assert res.is_high_risk is False
        assert res.recommend_targeted_retrieval is False
        assert "sufficiency_gate" in b.metadata.elapsed_per_stage

    def test_low_risk_insufficient_due_to_count(self):
        b = _bundle(_item("1", rr_score=0.8), risk=_risk(score=0.1))
        res = evaluate_sufficiency(b, b.risk_profile, self.cfg)
        assert res.verdict == SufficiencyVerdict.INSUFFICIENT
        assert res.is_high_risk is False
        assert res.recommend_targeted_retrieval is True
        assert any("evidence_item_count" in r for r in res.reasons)

    def test_high_risk_sufficient(self):
        # Meets all coverage AND risk limits
        b = _bundle(
            _item("1", rr_score=0.8, prov=_prov(src=SourceType.OTHER)),
            _item("2", rr_score=0.6, prov=_prov(src=SourceType.DRUG_LABEL)),
            risk=_risk(score=0.9)
        )
        res = evaluate_sufficiency(b, b.risk_profile, self.cfg)
        assert res.verdict == SufficiencyVerdict.SUFFICIENT
        assert res.is_high_risk is True

    def test_high_risk_conservative_due_to_coverage(self):
        # Missing items -> coverage fails. Because it's high risk, it returns CONSERVATIVE, not INSUFFICIENT
        b = _bundle(_item("1", rr_score=0.8), risk=_risk(score=0.9))
        res = evaluate_sufficiency(b, b.risk_profile, self.cfg)
        assert res.verdict == SufficiencyVerdict.CONSERVATIVE
        assert res.is_high_risk is True
        assert res.recommend_targeted_retrieval is True

    def test_high_risk_conservative_due_to_unknown_provenance(self):
        # Good coverage, but too much UNKNOWN provenance for a high-risk query
        prov_unknown = _prov(date=UNKNOWN, jur=UNKNOWN, pop=UNKNOWN, dos=UNKNOWN)
        b = _bundle(
            _item("1", rr_score=0.8, prov=prov_unknown),
            _item("2", rr_score=0.6, prov=prov_unknown),
            risk=_risk(score=0.9)
        )
        res = evaluate_sufficiency(b, b.risk_profile, self.cfg)
        assert res.verdict == SufficiencyVerdict.CONSERVATIVE
        assert res.is_high_risk is True
        assert any("unknown_provenance_ratio" in r for r in res.reasons)

    def test_targeted_retrieval_used_flag_prevents_recommendation(self):
        # If targeted retrieval was already used, the gate flags insufficient/conservative
        # but does NOT recommend targeted_retrieval (prevents infinite loops in orchestrator)
        b = _bundle(_item("1"), risk=_risk(score=0.1), targeted_used=True)
        res = evaluate_sufficiency(b, b.risk_profile, self.cfg)
        assert res.verdict == SufficiencyVerdict.INSUFFICIENT
        assert res.recommend_targeted_retrieval is False

    def test_safety_floor_forced_triggers_high_risk(self):
        # Score is low (0.1), but safety_floor_forced is True.
        # It must trigger the CONSERVATIVE branch for risk issues.
        prov_unknown = _prov(date=UNKNOWN, jur=UNKNOWN, pop=UNKNOWN, dos=UNKNOWN)
        b = _bundle(
            _item("1", rr_score=0.8, prov=prov_unknown),
            _item("2", rr_score=0.6, prov=prov_unknown),
            risk=_risk(score=0.1, safety_forced=True)
        )
        res = evaluate_sufficiency(b, b.risk_profile, self.cfg)
        assert res.is_high_risk is True
        assert res.verdict == SufficiencyVerdict.CONSERVATIVE