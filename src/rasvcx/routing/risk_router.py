"""Risk/complexity router: produces the request-scoped RiskProfile.

CRITICAL INVARIANT: this router is advisory for computational optimization
only. It allocates validation effort (depth, retry budgets, NLI allowance)
but is NEVER authoritative over minimum validation safety. Deterministic
validation always runs regardless of what this router decides -- that
safety floor is enforced downstream in validation/selective_router.py, not
here. This module must not be able to prevent that floor from applying.
"""

from __future__ import annotations

from dataclasses import dataclass

from rasvcx.routing.risk_features import extract_risk_features
from rasvcx.schemas.query import RiskFeatureScores, RiskProfile, ValidationDepth


@dataclass(frozen=True, slots=True)
class RiskRouterWeights:
    """Configurable weights combining feature scores into overall_risk_score.

    Weights must sum to 1.0 so overall_risk_score stays bounded in [0, 1]
    given each feature score is itself bounded in [0, 1]. These are initial
    heuristics (see architecture DECISIONS.md item L: risk weights are
    initial heuristics, not empirically tuned) and are expected to be
    revisited after ablation studies.
    """

    numeric_content_weight: float = 0.25
    unit_sensitive_weight: float = 0.25
    safety_critical_structure_weight: float = 0.30
    query_complexity_weight: float = 0.10
    ambiguity_weight: float = 0.10

    def __post_init__(self) -> None:
        total = (
            self.numeric_content_weight
            + self.unit_sensitive_weight
            + self.safety_critical_structure_weight
            + self.query_complexity_weight
            + self.ambiguity_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"RiskRouterWeights must sum to 1.0, got {total}")


@dataclass(frozen=True, slots=True)
class RiskRouterThresholds:
    """Configurable thresholds mapping overall_risk_score to a ValidationDepth
    and to retrieval/NLI budget allocations.

    deep_threshold must be strictly greater than standard_threshold.
    """

    standard_threshold: float = 0.35
    deep_threshold: float = 0.65

    shallow_retrieval_retry_budget: int = 0
    standard_retrieval_retry_budget: int = 1
    deep_retrieval_retry_budget: int = 2

    shallow_nli_call_allowance: int = 0
    standard_nli_call_allowance: int = 5
    deep_nli_call_allowance: int = 15

    def __post_init__(self) -> None:
        if not 0.0 <= self.standard_threshold <= 1.0:
            raise ValueError("standard_threshold must be in [0, 1]")
        if not 0.0 <= self.deep_threshold <= 1.0:
            raise ValueError("deep_threshold must be in [0, 1]")
        if self.deep_threshold <= self.standard_threshold:
            raise ValueError("deep_threshold must be strictly greater than standard_threshold")
        for name, value in (
            ("shallow_retrieval_retry_budget", self.shallow_retrieval_retry_budget),
            ("standard_retrieval_retry_budget", self.standard_retrieval_retry_budget),
            ("deep_retrieval_retry_budget", self.deep_retrieval_retry_budget),
            ("shallow_nli_call_allowance", self.shallow_nli_call_allowance),
            ("standard_nli_call_allowance", self.standard_nli_call_allowance),
            ("deep_nli_call_allowance", self.deep_nli_call_allowance),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


# Safety-critical structure alone is enough to force the safety floor
# eligibility flag on the RiskProfile, independent of the overall score.
# This threshold is intentionally low: any meaningful presence of
# safety-critical structure (dose/contraindication/threshold language)
# should force eligibility, not just a high aggregate score.
_SAFETY_FLOOR_FEATURE_THRESHOLD = 0.3


def _weighted_overall_score(features: RiskFeatureScores, weights: RiskRouterWeights) -> float:
    score = (
        features.numeric_content_score * weights.numeric_content_weight
        + features.unit_sensitive_score * weights.unit_sensitive_weight
        + features.safety_critical_structure_score * weights.safety_critical_structure_weight
        + features.query_complexity_score * weights.query_complexity_weight
        + features.ambiguity_score * weights.ambiguity_weight
    )
    return min(1.0, max(0.0, score))


def _validation_depth_for_score(score: float, thresholds: RiskRouterThresholds) -> ValidationDepth:
    if score >= thresholds.deep_threshold:
        return ValidationDepth.DEEP
    if score >= thresholds.standard_threshold:
        return ValidationDepth.STANDARD
    return ValidationDepth.SHALLOW


def _budgets_for_depth(
    depth: ValidationDepth, thresholds: RiskRouterThresholds
) -> tuple[int, int]:
    if depth is ValidationDepth.DEEP:
        return thresholds.deep_retrieval_retry_budget, thresholds.deep_nli_call_allowance
    if depth is ValidationDepth.STANDARD:
        return thresholds.standard_retrieval_retry_budget, thresholds.standard_nli_call_allowance
    return thresholds.shallow_retrieval_retry_budget, thresholds.shallow_nli_call_allowance


def route_query(
    query_text: str,
    weights: RiskRouterWeights | None = None,
    thresholds: RiskRouterThresholds | None = None,
) -> RiskProfile:
    """Compute the RiskProfile for one normalized query.

    This function is advisory-only: its output allocates validation effort
    (validation_depth, retrieval_retry_budget, nli_call_allowance) but never
    decides whether deterministic validation runs (it always does, enforced
    elsewhere) and never decides the final ANSWER/WARNING/REPAIR/REGENERATE/
    ABSTAIN action (owned by decision/engine.py).

    safety_floor_forced is set here as an advisory signal only; the actual
    enforcement of the safety floor -- ensuring such queries cannot skip
    contextual validation / NLI eligibility -- is implemented in
    validation/selective_router.py, which must not rely solely on this flag
    (it re-derives safety-critical eligibility from claims/evidence too).
    """
    weights = weights or RiskRouterWeights()
    thresholds = thresholds or RiskRouterThresholds()

    features = extract_risk_features(query_text)
    overall_score = _weighted_overall_score(features, weights)
    depth = _validation_depth_for_score(overall_score, thresholds)
    retrieval_retry_budget, nli_call_allowance = _budgets_for_depth(depth, thresholds)

    safety_floor_forced = features.safety_critical_structure_score >= _SAFETY_FLOOR_FEATURE_THRESHOLD

    return RiskProfile(
        overall_risk_score=overall_score,
        feature_scores=features,
        validation_depth=depth,
        retrieval_retry_budget=retrieval_retry_budget,
        nli_call_allowance=nli_call_allowance,
        safety_floor_forced=safety_floor_forced,
    )