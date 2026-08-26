
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pytest

from rasvcx.routing.risk_features import (
    compute_ambiguity_score,
    compute_numeric_content_score,
    compute_query_complexity_score,
    compute_safety_critical_structure_score,
    compute_unit_sensitive_score,
    extract_risk_features,
)
from rasvcx.routing.risk_router import (
    RiskRouterThresholds,
    RiskRouterWeights,
    _SAFETY_FLOOR_FEATURE_THRESHOLD,
    _budgets_for_depth,
    _validation_depth_for_score,
    _weighted_overall_score,
    route_query,
)
from rasvcx.schemas.query import RiskFeatureScores, RiskProfile, ValidationDepth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_features() -> RiskFeatureScores:
    return RiskFeatureScores(
        numeric_content_score=0.0,
        unit_sensitive_score=0.0,
        safety_critical_structure_score=0.0,
        query_complexity_score=0.0,
        ambiguity_score=0.0,
    )


def _max_features() -> RiskFeatureScores:
    return RiskFeatureScores(
        numeric_content_score=1.0,
        unit_sensitive_score=1.0,
        safety_critical_structure_score=1.0,
        query_complexity_score=1.0,
        ambiguity_score=1.0,
    )


# ===========================================================================
# 1. Individual feature computation — pure functions
# ===========================================================================


class TestComputeNumericContentScore:
    def test_empty_string_returns_zero(self) -> None:
        assert compute_numeric_content_score("") == 0.0

    def test_no_numbers_returns_zero(self) -> None:
        assert compute_numeric_content_score("what is the recommended dose") == 0.0

    def test_single_integer(self) -> None:
        score = compute_numeric_content_score("give 5 mg")
        assert 0.0 < score <= 1.0

    def test_saturates_at_max_one(self) -> None:
        text = " ".join(str(i) for i in range(100))
        assert compute_numeric_content_score(text) == 1.0

    def test_decimal_detected(self) -> None:
        score = compute_numeric_content_score("0.5 mg twice daily")
        assert score > 0.0

    def test_deterministic(self) -> None:
        text = "dose 10 mg for 3 days"
        assert compute_numeric_content_score(text) == compute_numeric_content_score(text)


class TestComputeUnitSensitiveScore:
    def test_empty_string_returns_zero(self) -> None:
        assert compute_unit_sensitive_score("") == 0.0

    def test_no_units_returns_zero(self) -> None:
        assert compute_unit_sensitive_score("what is the recommended dose") == 0.0

    def test_mg_detected(self) -> None:
        assert compute_unit_sensitive_score("give 500 mg once") > 0.0

    def test_ml_detected(self) -> None:
        assert compute_unit_sensitive_score("administer 10 mL IV") > 0.0

    def test_percent_word_detected(self) -> None:
        # The unit pattern uses \b word boundaries; "%" alone does not form a
        # word boundary match.  The word "percent" does match.
        assert compute_unit_sensitive_score("5 percent solution") > 0.0

    def test_percent_symbol_not_matched(self) -> None:
        # "%" is not matched by the \b-bounded regex — document this explicitly
        # so future changes to the pattern are noticed by this test.
        assert compute_unit_sensitive_score("5% solution") == 0.0

    def test_saturates(self) -> None:
        text = "mg mcg g kg ml"
        assert compute_unit_sensitive_score(text) == 1.0

    def test_case_insensitive(self) -> None:
        lower = compute_unit_sensitive_score("10 mg daily")
        upper = compute_unit_sensitive_score("10 MG daily")
        assert lower == upper

    def test_deterministic(self) -> None:
        text = "500 mg twice per day for 7 days"
        assert compute_unit_sensitive_score(text) == compute_unit_sensitive_score(text)


class TestComputeSafetyCriticalStructureScore:
    def test_empty_string_returns_zero(self) -> None:
        assert compute_safety_critical_structure_score("") == 0.0

    def test_no_safety_keywords_returns_zero(self) -> None:
        assert compute_safety_critical_structure_score("what color is aspirin") == 0.0

    def test_dose_keyword(self) -> None:
        assert compute_safety_critical_structure_score("what is the recommended dose") > 0.0

    def test_contraindicated_keyword(self) -> None:
        assert compute_safety_critical_structure_score("is it contraindicated in pregnancy") > 0.0

    def test_overdose_keyword(self) -> None:
        assert compute_safety_critical_structure_score("signs of overdose include") > 0.0

    def test_toxic_keyword(self) -> None:
        assert compute_safety_critical_structure_score("toxic threshold for this drug") > 0.0

    def test_lethal_keyword(self) -> None:
        assert compute_safety_critical_structure_score("lethal dose in animals") > 0.0

    def test_saturates(self) -> None:
        text = "maximum dose dosage contraindicated interaction threshold lethal"
        assert compute_safety_critical_structure_score(text) == 1.0

    def test_case_insensitive(self) -> None:
        lower = compute_safety_critical_structure_score("dose limit")
        upper = compute_safety_critical_structure_score("DOSE LIMIT")
        assert lower == upper

    def test_deterministic(self) -> None:
        text = "what is the maximum dose before toxicity"
        assert compute_safety_critical_structure_score(text) == compute_safety_critical_structure_score(text)


class TestComputeQueryComplexityScore:
    def test_empty_string_returns_zero(self) -> None:
        assert compute_query_complexity_score("") == 0.0

    def test_single_word_near_zero(self) -> None:
        # token_count=1 contributes (1/40)/3 ≈ 0.0083; clause/question scores
        # are 0 — result is very small but non-zero by the formula.
        score = compute_query_complexity_score("aspirin")
        assert 0.0 <= score < 0.05

    def test_short_simple_query_bounded(self) -> None:
        score = compute_query_complexity_score("what is aspirin")
        assert 0.0 <= score <= 1.0

    def test_long_multi_clause_query_positive(self) -> None:
        text = (
            "What is the recommended maximum daily dose of metformin for type 2 diabetes "
            "in elderly patients with renal impairment, and how should it be adjusted "
            "based on eGFR, and are there any contraindications?"
        )
        score = compute_query_complexity_score(text)
        assert score > 0.0

    def test_bounded_for_very_long_input(self) -> None:
        very_long = " ".join(["word"] * 200) + "? ? ?"
        score = compute_query_complexity_score(very_long)
        assert 0.0 <= score <= 1.0

    def test_deterministic(self) -> None:
        text = "what is the dose of X and should it be taken with food or without food?"
        assert compute_query_complexity_score(text) == compute_query_complexity_score(text)


class TestComputeAmbiguityScore:
    def test_empty_string_returns_zero(self) -> None:
        assert compute_ambiguity_score("") == 0.0

    def test_no_ambiguity_markers(self) -> None:
        assert compute_ambiguity_score("what is the dose of aspirin") == 0.0

    def test_pronoun_it(self) -> None:
        assert compute_ambiguity_score("is it safe for children") > 0.0

    def test_pronoun_they(self) -> None:
        assert compute_ambiguity_score("do they interact with warfarin") > 0.0

    def test_the_drug(self) -> None:
        assert compute_ambiguity_score("what is the drug used for") > 0.0

    def test_saturates(self) -> None:
        text = "it this that they these those the drug the medication"
        assert compute_ambiguity_score(text) == 1.0

    def test_deterministic(self) -> None:
        text = "is it safe and does it interact with this medication"
        assert compute_ambiguity_score(text) == compute_ambiguity_score(text)


# ===========================================================================
# 2. extract_risk_features — contract and properties
# ===========================================================================


class TestExtractRiskFeatures:
    def test_returns_risk_feature_scores(self) -> None:
        result = extract_risk_features("what is the dose of aspirin")
        assert isinstance(result, RiskFeatureScores)

    def test_all_scores_bounded_0_1(self) -> None:
        text = "is 500mg dose safe? maximum dose contraindicated for it and they"
        features = extract_risk_features(text)
        for score in (
            features.numeric_content_score,
            features.unit_sensitive_score,
            features.safety_critical_structure_score,
            features.query_complexity_score,
            features.ambiguity_score,
        ):
            assert 0.0 <= score <= 1.0, f"Score out of bounds: {score}"

    def test_empty_query_all_zeros(self) -> None:
        features = extract_risk_features("")
        assert features.numeric_content_score == 0.0
        assert features.unit_sensitive_score == 0.0
        assert features.safety_critical_structure_score == 0.0
        assert features.query_complexity_score == 0.0
        assert features.ambiguity_score == 0.0

    def test_pure_deterministic(self) -> None:
        text = "what is the maximum daily dose of metformin 500 mg for elderly patients"
        f1 = extract_risk_features(text)
        f2 = extract_risk_features(text)
        assert f1 == f2

    def test_safety_critical_query_raises_safety_score(self) -> None:
        text = "what is the lethal dose threshold for digoxin"
        features = extract_risk_features(text)
        assert features.safety_critical_structure_score > 0.0

    def test_numeric_and_unit_present(self) -> None:
        text = "administer 250 mg twice daily for 7 days"
        features = extract_risk_features(text)
        assert features.numeric_content_score > 0.0
        assert features.unit_sensitive_score > 0.0

    def test_immutable(self) -> None:
        features = extract_risk_features("dose of aspirin")
        with pytest.raises((AttributeError, TypeError)):
            features.numeric_content_score = 0.99  # type: ignore[misc]


# ===========================================================================
# 3. RiskRouterWeights
# ===========================================================================


class TestRiskRouterWeights:
    def test_default_construction_succeeds(self) -> None:
        w = RiskRouterWeights()
        assert w.numeric_content_weight == 0.25
        assert w.unit_sensitive_weight == 0.25
        assert w.safety_critical_structure_weight == 0.30
        assert w.query_complexity_weight == 0.10
        assert w.ambiguity_weight == 0.10

    def test_custom_weights_summing_to_one(self) -> None:
        w = RiskRouterWeights(
            numeric_content_weight=0.20,
            unit_sensitive_weight=0.20,
            safety_critical_structure_weight=0.20,
            query_complexity_weight=0.20,
            ambiguity_weight=0.20,
        )
        total = (
            w.numeric_content_weight + w.unit_sensitive_weight
            + w.safety_critical_structure_weight + w.query_complexity_weight
            + w.ambiguity_weight
        )
        assert abs(total - 1.0) < 1e-9

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            RiskRouterWeights(
                numeric_content_weight=0.30,
                unit_sensitive_weight=0.30,
                safety_critical_structure_weight=0.30,
                query_complexity_weight=0.10,
                ambiguity_weight=0.10,
            )

    def test_frozen(self) -> None:
        w = RiskRouterWeights()
        with pytest.raises((AttributeError, TypeError)):
            w.numeric_content_weight = 0.5  # type: ignore[misc]

    def test_equality(self) -> None:
        assert RiskRouterWeights() == RiskRouterWeights()


# ===========================================================================
# 4. RiskRouterThresholds
# ===========================================================================


class TestRiskRouterThresholds:
    def test_default_construction_succeeds(self) -> None:
        t = RiskRouterThresholds()
        assert t.standard_threshold == 0.35
        assert t.deep_threshold == 0.65
        assert t.shallow_retrieval_retry_budget == 0
        assert t.standard_retrieval_retry_budget == 1
        assert t.deep_retrieval_retry_budget == 2
        assert t.shallow_nli_call_allowance == 0
        assert t.standard_nli_call_allowance == 5
        assert t.deep_nli_call_allowance == 15

    def test_deep_equal_standard_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly greater"):
            RiskRouterThresholds(standard_threshold=0.65, deep_threshold=0.65)

    def test_deep_less_than_standard_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly greater"):
            RiskRouterThresholds(standard_threshold=0.70, deep_threshold=0.50)

    def test_standard_threshold_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="standard_threshold"):
            RiskRouterThresholds(standard_threshold=-0.1, deep_threshold=0.65)

    def test_negative_retry_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskRouterThresholds(shallow_retrieval_retry_budget=-1)

    def test_negative_nli_allowance_raises(self) -> None:
        with pytest.raises(ValueError):
            RiskRouterThresholds(shallow_nli_call_allowance=-1)

    def test_frozen(self) -> None:
        t = RiskRouterThresholds()
        with pytest.raises((AttributeError, TypeError)):
            t.standard_threshold = 0.5  # type: ignore[misc]

    def test_custom_thresholds_accepted(self) -> None:
        t = RiskRouterThresholds(standard_threshold=0.40, deep_threshold=0.70)
        assert t.standard_threshold == 0.40
        assert t.deep_threshold == 0.70


# ===========================================================================
# 5. _weighted_overall_score
# ===========================================================================


class TestWeightedOverallScore:
    def test_all_zero_features_gives_zero(self) -> None:
        score = _weighted_overall_score(_zero_features(), RiskRouterWeights())
        assert score == 0.0

    def test_all_max_features_gives_one(self) -> None:
        score = _weighted_overall_score(_max_features(), RiskRouterWeights())
        assert score == 1.0

    def test_result_bounded(self) -> None:
        score = _weighted_overall_score(_max_features(), RiskRouterWeights())
        assert 0.0 <= score <= 1.0

    def test_partial_features_numeric_only(self) -> None:
        features = RiskFeatureScores(
            numeric_content_score=1.0,
            unit_sensitive_score=0.0,
            safety_critical_structure_score=0.0,
            query_complexity_score=0.0,
            ambiguity_score=0.0,
        )
        # 1.0 * 0.25 = 0.25
        score = _weighted_overall_score(features, RiskRouterWeights())
        assert abs(score - 0.25) < 1e-9

    def test_deterministic(self) -> None:
        features = _max_features()
        weights = RiskRouterWeights()
        assert _weighted_overall_score(features, weights) == _weighted_overall_score(features, weights)


# ===========================================================================
# 6. _validation_depth_for_score
# ===========================================================================


class TestValidationDepthForScore:
    def setup_method(self) -> None:
        self.t = RiskRouterThresholds(standard_threshold=0.35, deep_threshold=0.65)

    def test_score_zero_is_shallow(self) -> None:
        assert _validation_depth_for_score(0.0, self.t) is ValidationDepth.SHALLOW

    def test_score_just_below_standard_is_shallow(self) -> None:
        assert _validation_depth_for_score(0.34, self.t) is ValidationDepth.SHALLOW

    def test_score_at_standard_is_standard(self) -> None:
        assert _validation_depth_for_score(0.35, self.t) is ValidationDepth.STANDARD

    def test_score_mid_range_is_standard(self) -> None:
        assert _validation_depth_for_score(0.50, self.t) is ValidationDepth.STANDARD
        assert _validation_depth_for_score(0.64, self.t) is ValidationDepth.STANDARD

    def test_score_at_deep_threshold_is_deep(self) -> None:
        assert _validation_depth_for_score(0.65, self.t) is ValidationDepth.DEEP

    def test_score_one_is_deep(self) -> None:
        assert _validation_depth_for_score(1.0, self.t) is ValidationDepth.DEEP

    def test_custom_thresholds_respected(self) -> None:
        t = RiskRouterThresholds(standard_threshold=0.20, deep_threshold=0.80)
        assert _validation_depth_for_score(0.19, t) is ValidationDepth.SHALLOW
        assert _validation_depth_for_score(0.20, t) is ValidationDepth.STANDARD
        assert _validation_depth_for_score(0.79, t) is ValidationDepth.STANDARD
        assert _validation_depth_for_score(0.80, t) is ValidationDepth.DEEP


# ===========================================================================
# 7. _budgets_for_depth
# ===========================================================================


class TestBudgetsForDepth:
    def setup_method(self) -> None:
        self.t = RiskRouterThresholds()

    def test_shallow_budgets(self) -> None:
        retry, nli = _budgets_for_depth(ValidationDepth.SHALLOW, self.t)
        assert retry == self.t.shallow_retrieval_retry_budget
        assert nli == self.t.shallow_nli_call_allowance

    def test_standard_budgets(self) -> None:
        retry, nli = _budgets_for_depth(ValidationDepth.STANDARD, self.t)
        assert retry == self.t.standard_retrieval_retry_budget
        assert nli == self.t.standard_nli_call_allowance

    def test_deep_budgets(self) -> None:
        retry, nli = _budgets_for_depth(ValidationDepth.DEEP, self.t)
        assert retry == self.t.deep_retrieval_retry_budget
        assert nli == self.t.deep_nli_call_allowance

    def test_all_budgets_non_negative(self) -> None:
        for depth in ValidationDepth:
            retry, nli = _budgets_for_depth(depth, self.t)
            assert retry >= 0
            assert nli >= 0

    def test_deep_at_least_as_large_as_standard(self) -> None:
        retry_std, nli_std = _budgets_for_depth(ValidationDepth.STANDARD, self.t)
        retry_deep, nli_deep = _budgets_for_depth(ValidationDepth.DEEP, self.t)
        assert retry_deep >= retry_std
        assert nli_deep >= nli_std


# ===========================================================================
# 8. route_query — integration
# ===========================================================================


class TestRouteQuery:
    def test_returns_risk_profile(self) -> None:
        assert isinstance(route_query("what is aspirin"), RiskProfile)

    def test_risk_profile_is_frozen(self) -> None:
        profile = route_query("what is aspirin")
        with pytest.raises((AttributeError, TypeError)):
            profile.overall_risk_score = 0.99  # type: ignore[misc]

    def test_overall_risk_score_bounded(self) -> None:
        profile = route_query("what is the maximum dose of warfarin 5 mg daily")
        assert 0.0 <= profile.overall_risk_score <= 1.0

    def test_feature_scores_type(self) -> None:
        profile = route_query("give 500 mg aspirin")
        assert isinstance(profile.feature_scores, RiskFeatureScores)

    def test_validation_depth_is_enum(self) -> None:
        profile = route_query("what is aspirin used for")
        assert profile.validation_depth in ValidationDepth

    def test_budgets_non_negative(self) -> None:
        profile = route_query("what is aspirin")
        assert profile.retrieval_retry_budget >= 0
        assert profile.nli_call_allowance >= 0

    def test_safety_floor_forced_is_bool(self) -> None:
        profile = route_query("what is the dosage")
        assert isinstance(profile.safety_floor_forced, bool)

    def test_benign_query_shallow_or_standard(self) -> None:
        profile = route_query("what color is aspirin")
        assert profile.validation_depth in (ValidationDepth.SHALLOW, ValidationDepth.STANDARD)

    def test_high_risk_query_standard_or_deep(self) -> None:
        # A query with multiple safety-critical keywords scores in the upper range.
        # With default thresholds (deep >= 0.65) it may land in STANDARD or DEEP
        # depending on exact token density — the important assertion is that it is
        # NOT SHALLOW.
        text = (
            "What is the maximum lethal dose threshold for digoxin 0.25 mg daily "
            "and is it contraindicated with amiodarone and what is the toxic interaction?"
        )
        profile = route_query(text)
        assert profile.validation_depth in (ValidationDepth.STANDARD, ValidationDepth.DEEP)

    def test_maximum_signal_query_is_deep(self) -> None:
        # This text is crafted to reliably exceed the 0.65 deep threshold:
        # numeric (0.25mg, 10ml) + units (mg, ml) + safety keywords (dose,
        # threshold, contraindicated, toxic, lethal) + moderate complexity.
        text = (
            "what is the maximum dose threshold of digoxin 0.25 mg 10 ml "
            "contraindicated with amiodarone toxic lethal"
        )
        profile = route_query(text)
        assert profile.validation_depth is ValidationDepth.DEEP

    def test_safety_floor_true_for_safety_critical_query(self) -> None:
        text = "what is the maximum dose threshold and is it contraindicated"
        profile = route_query(text)
        assert profile.safety_floor_forced is True

    def test_safety_floor_false_for_benign_query(self) -> None:
        assert route_query("what color is aspirin").safety_floor_forced is False

    def test_safety_floor_threshold_constant_valid(self) -> None:
        assert 0.0 < _SAFETY_FLOOR_FEATURE_THRESHOLD <= 1.0

    def test_safety_floor_consistent_with_feature_score(self) -> None:
        text = "what is the dosage limit"
        profile = route_query(text)
        expected = profile.feature_scores.safety_critical_structure_score >= _SAFETY_FLOOR_FEATURE_THRESHOLD
        assert profile.safety_floor_forced is expected

    def test_shallow_depth_budget_matches_thresholds(self) -> None:
        thresholds = RiskRouterThresholds()
        profile = route_query("what color is aspirin", thresholds=thresholds)
        if profile.validation_depth is ValidationDepth.SHALLOW:
            assert profile.retrieval_retry_budget == thresholds.shallow_retrieval_retry_budget
            assert profile.nli_call_allowance == thresholds.shallow_nli_call_allowance

    def test_deep_depth_budget_matches_thresholds(self) -> None:
        text = (
            "what is the maximum lethal threshold dose of digoxin 0.25 mg daily "
            "contraindicated with amiodarone and toxic interaction"
        )
        thresholds = RiskRouterThresholds()
        profile = route_query(text, thresholds=thresholds)
        if profile.validation_depth is ValidationDepth.DEEP:
            assert profile.retrieval_retry_budget == thresholds.deep_retrieval_retry_budget
            assert profile.nli_call_allowance == thresholds.deep_nli_call_allowance

    def test_custom_weights_accepted(self) -> None:
        weights = RiskRouterWeights(
            numeric_content_weight=0.20,
            unit_sensitive_weight=0.20,
            safety_critical_structure_weight=0.20,
            query_complexity_weight=0.20,
            ambiguity_weight=0.20,
        )
        assert isinstance(route_query("500 mg daily", weights=weights), RiskProfile)

    def test_custom_thresholds_accepted(self) -> None:
        thresholds = RiskRouterThresholds(standard_threshold=0.10, deep_threshold=0.20)
        assert isinstance(route_query("what is aspirin", thresholds=thresholds), RiskProfile)

    def test_low_deep_threshold_forces_deep(self) -> None:
        thresholds = RiskRouterThresholds(standard_threshold=0.01, deep_threshold=0.02)
        profile = route_query("dose of aspirin", thresholds=thresholds)
        assert profile.validation_depth is ValidationDepth.DEEP

    def test_very_high_thresholds_give_shallow(self) -> None:
        thresholds = RiskRouterThresholds(standard_threshold=0.95, deep_threshold=0.99)
        profile = route_query("what is aspirin used for", thresholds=thresholds)
        assert profile.validation_depth is ValidationDepth.SHALLOW

    def test_deterministic(self) -> None:
        text = "what is the maximum dose of metformin 500 mg in elderly patients"
        assert route_query(text) == route_query(text)

    def test_higher_risk_text_higher_or_equal_score(self) -> None:
        p_low = route_query("what color is aspirin")
        p_high = route_query(
            "what is the lethal dose threshold for digoxin 0.25 mg "
            "contraindicated with amiodarone"
        )
        assert p_low.overall_risk_score <= p_high.overall_risk_score

    def test_router_has_no_action_field(self) -> None:
        profile = route_query("what is the dose of aspirin")
        assert not hasattr(profile, "action")
        assert not hasattr(profile, "decision")

    def test_safety_floor_forced_is_plain_bool(self) -> None:
        profile = route_query("what is the maximum lethal dose threshold")
        assert type(profile.safety_floor_forced) is bool


# ===========================================================================
# 9. Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_query(self) -> None:
        profile = route_query("")
        assert profile.overall_risk_score == 0.0
        assert profile.validation_depth is ValidationDepth.SHALLOW
        assert profile.safety_floor_forced is False

    def test_whitespace_only_query(self) -> None:
        profile = route_query("   ")
        assert profile.overall_risk_score == 0.0
        assert profile.validation_depth is ValidationDepth.SHALLOW

    def test_single_numeric_token_bounded(self) -> None:
        profile = route_query("500")
        assert 0.0 <= profile.overall_risk_score <= 1.0

    def test_very_long_query_bounded(self) -> None:
        text = " ".join(["dose", "mg", "500", "contraindicated"] * 50)
        profile = route_query(text)
        assert 0.0 <= profile.overall_risk_score <= 1.0

    def test_all_risk_signals_score_at_most_one(self) -> None:
        text = (
            "maximum lethal dose threshold 500 mg 10 ml contraindicated "
            "interaction toxic overdose it this that they these"
        )
        profile = route_query(text)
        assert profile.overall_risk_score <= 1.0

    def test_no_global_state_between_calls(self) -> None:
        text_a = "what is aspirin"
        text_b = "lethal dose threshold 0.25 mg contraindicated"
        profile_a1 = route_query(text_a)
        _ = route_query(text_b)
        profile_a2 = route_query(text_a)
        assert profile_a1 == profile_a2

    def test_unicode_query_does_not_raise(self) -> None:
        profile = route_query("dosis máxima de 500 mg diario")
        assert isinstance(profile, RiskProfile)

    def test_numeric_saturated_query(self) -> None:
        profile = route_query(" ".join(str(i) for i in range(100)))
        assert profile.feature_scores.numeric_content_score == 1.0

    def test_unit_saturated_query(self) -> None:
        profile = route_query("mg mcg g ml L")
        assert profile.feature_scores.unit_sensitive_score == 1.0