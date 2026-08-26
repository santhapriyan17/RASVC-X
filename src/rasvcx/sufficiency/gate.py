# src/rasvcx/sufficiency/gate.py
"""Sufficiency gate (Module 5): SUFFICIENT / INSUFFICIENT / CONSERVATIVE.

Combines SufficiencySignals (evidence-derived, from scorer.py) with the
request's RiskProfile (routing-derived) and a set of explicit, configurable
thresholds to decide whether the pipeline may safely proceed toward
provenance/claims/validation/generation, should request additional
(targeted) retrieval, or -- for high-risk queries whose evidence is
insufficient or ambiguous -- should take the conservative path toward
eventual abstention.

Architectural boundaries:
  - This module does NOT call retrieval. It only recommends whether
    targeted retrieval would help; the orchestrator owns the global
    corrective-attempt budget and decides whether to act on that
    recommendation (see retrieval/targeted_retrieval.py docstring).
  - This module does NOT perform final truth validation or generation.
  - "No contradiction found" is never treated as "evidence is correct":
    the gate only reasons about coverage/consistency-adjacent *signals*,
    not ground truth.
  - UNKNOWN provenance is never silently treated as compatible; a high
    unknown_provenance_ratio pushes the verdict toward CONSERVATIVE for
    high-risk queries rather than being ignored.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from rasvcx.schemas.evidence import EvidenceBundle
from rasvcx.schemas.query import RiskProfile
from rasvcx.sufficiency.scorer import SufficiencySignals, compute_sufficiency_signals


class SufficiencyVerdict(str, Enum):
    """Outcome of the sufficiency gate for one pipeline pass.

    SUFFICIENT:   Evidence supports proceeding toward provenance/claims/
                  validation/generation.
    INSUFFICIENT: Evidence does not meet the coverage/quality bar; targeted
                  retrieval is recommended (bounded-loop budget permitting).
    CONSERVATIVE: High-risk query combined with insufficient or ambiguous
                  evidence -- the pipeline should prefer the conservative
                  path (further validation, or eventual abstention) rather
                  than treating this as an ordinary retry case.
    """

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONSERVATIVE = "conservative"


@dataclass(frozen=True, slots=True)
class SufficiencyGateConfig:
    """Explicit, inspectable thresholds for the sufficiency gate.

    All thresholds are deliberately named per-signal rather than folded
    into one composite score, so each can be tuned/ablated independently
    during evaluation experiments.

    Attributes:
        min_evidence_items:        Minimum evidence_item_count to consider
                                    coverage adequate at all.
        min_scored_items:          Minimum scored_item_count (items that
                                    survived reranking) required.
        min_top_rerank_score:      Minimum top_rerank_score required when a
                                    top score is available. Ignored (not
                                    evaluated) if no item has been scored --
                                    that case is already caught by
                                    min_scored_items.
        min_rerank_score_margin:   Minimum acceptable rerank_score_margin.
                                    A margin below this is treated as
                                    "ambiguous top result," not as failing
                                    coverage outright -- it only affects the
                                    CONSERVATIVE branch for high-risk
                                    queries, since ambiguity alone is not a
                                    coverage failure for low-risk queries.
        max_unknown_provenance_ratio_high_risk:
                                    For high-risk queries only: maximum
                                    tolerable unknown_provenance_ratio
                                    before the gate prefers CONSERVATIVE.
        min_source_diversity_high_risk:
                                    For high-risk queries only: minimum
                                    source_type_diversity required to avoid
                                    the CONSERVATIVE path.
        high_risk_score_threshold: overall_risk_score at or above which a
                                    query is treated as high-risk for the
                                    purposes of this gate, independent of
                                    safety_floor_forced.
    """

    min_evidence_items: int = 2
    min_scored_items: int = 1
    min_top_rerank_score: float = 0.0
    min_rerank_score_margin: float = 0.05
    max_unknown_provenance_ratio_high_risk: float = 0.5
    min_source_diversity_high_risk: int = 2
    high_risk_score_threshold: float = 0.7

    def __post_init__(self) -> None:
        if self.min_evidence_items < 0:
            raise ValueError("min_evidence_items must be non-negative")
        if self.min_scored_items < 0:
            raise ValueError("min_scored_items must be non-negative")
        if self.min_rerank_score_margin < 0.0:
            raise ValueError("min_rerank_score_margin must be non-negative")
        if not 0.0 <= self.max_unknown_provenance_ratio_high_risk <= 1.0:
            raise ValueError(
                "max_unknown_provenance_ratio_high_risk must be in [0, 1]"
            )
        if self.min_source_diversity_high_risk < 0:
            raise ValueError("min_source_diversity_high_risk must be non-negative")
        if not 0.0 <= self.high_risk_score_threshold <= 1.0:
            raise ValueError("high_risk_score_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SufficiencyResult:
    """Immutable, auditable output of the sufficiency gate.

    Attributes:
        verdict:                    The gate's decision.
        signals:                    The SufficiencySignals the verdict was
                                     computed from (for logging/evaluation).
        reasons:                    Ordered, human-readable reasons that
                                     contributed to the verdict. Never
                                     empty for INSUFFICIENT or CONSERVATIVE;
                                     may be empty for SUFFICIENT.
        recommend_targeted_retrieval:
                                     True if the gate believes another
                                     retrieval pass would plausibly help.
                                     Advisory only -- the orchestrator must
                                     still check the corrective-attempt
                                     budget before acting on this.
        is_high_risk:               Whether this request was evaluated as
                                     high-risk (feeds CONSERVATIVE branch).
    """

    verdict: SufficiencyVerdict
    signals: SufficiencySignals
    reasons: tuple[str, ...] = field(default_factory=tuple)
    recommend_targeted_retrieval: bool = False
    is_high_risk: bool = False

    def __post_init__(self) -> None:
        if self.verdict == SufficiencyVerdict.SUFFICIENT and self.recommend_targeted_retrieval:
            raise ValueError(
                "SufficiencyResult.verdict == SUFFICIENT must not "
                "recommend_targeted_retrieval"
            )
        if self.verdict != SufficiencyVerdict.SUFFICIENT and not self.reasons:
            raise ValueError(
                f"SufficiencyResult.verdict == {self.verdict.value!r} "
                f"requires at least one reason"
            )


def _is_high_risk(risk_profile: RiskProfile, config: SufficiencyGateConfig) -> bool:
    return (
        risk_profile.safety_floor_forced
        or risk_profile.overall_risk_score >= config.high_risk_score_threshold
    )


def _evaluate_coverage(
    signals: SufficiencySignals, config: SufficiencyGateConfig
) -> list[str]:
    """Baseline coverage failures, applicable regardless of risk level."""
    reasons: list[str] = []
    if signals.evidence_item_count < config.min_evidence_items:
        reasons.append(
            f"evidence_item_count={signals.evidence_item_count} < "
            f"min_evidence_items={config.min_evidence_items}"
        )
    if signals.scored_item_count < config.min_scored_items:
        reasons.append(
            f"scored_item_count={signals.scored_item_count} < "
            f"min_scored_items={config.min_scored_items}"
        )
    if (
        signals.top_rerank_score is not None
        and signals.top_rerank_score < config.min_top_rerank_score
    ):
        reasons.append(
            f"top_rerank_score={signals.top_rerank_score:.4f} < "
            f"min_top_rerank_score={config.min_top_rerank_score}"
        )
    return reasons


def _evaluate_high_risk_concerns(
    signals: SufficiencySignals, config: SufficiencyGateConfig
) -> list[str]:
    """Additional concerns evaluated only for high-risk queries.

    These do not by themselves make evidence "insufficient" for a
    low-risk query -- they only push a high-risk query toward the
    CONSERVATIVE path, since ambiguity and missing context matter more
    when the downstream stakes are higher.
    """
    reasons: list[str] = []
    if (
        signals.rerank_score_margin is not None
        and signals.rerank_score_margin < config.min_rerank_score_margin
    ):
        reasons.append(
            f"rerank_score_margin={signals.rerank_score_margin:.4f} < "
            f"min_rerank_score_margin={config.min_rerank_score_margin} "
            f"(ambiguous top result)"
        )
    if signals.unknown_provenance_ratio > config.max_unknown_provenance_ratio_high_risk:
        reasons.append(
            f"unknown_provenance_ratio={signals.unknown_provenance_ratio:.4f} > "
            f"max_unknown_provenance_ratio_high_risk="
            f"{config.max_unknown_provenance_ratio_high_risk} "
            f"(insufficient context to rule out conflict)"
        )
    if signals.source_type_diversity < config.min_source_diversity_high_risk:
        reasons.append(
            f"source_type_diversity={signals.source_type_diversity} < "
            f"min_source_diversity_high_risk="
            f"{config.min_source_diversity_high_risk}"
        )
    return reasons


def evaluate_sufficiency(
    bundle: EvidenceBundle,
    risk_profile: RiskProfile,
    config: SufficiencyGateConfig | None = None,
) -> SufficiencyResult:
    """Run the sufficiency gate for the current state of ``bundle``.

    Always records elapsed time under the ``"sufficiency_gate"`` pipeline
    stage, including on the empty-evidence and every-verdict path, mirroring
    the timing convention established by RerankingService.rerank_bundle.

    Verdict logic:
      1. Compute baseline coverage reasons (item counts, top score).
      2. If risk is high (safety_floor_forced or overall_risk_score above
         threshold): also compute high-risk-only concerns (margin
         ambiguity, unknown provenance, source diversity). If EITHER the
         baseline coverage reasons OR the high-risk concerns are non-empty,
         verdict is CONSERVATIVE.
      3. Otherwise (not high-risk, or high-risk with no concerns at all):
         verdict is INSUFFICIENT if baseline coverage reasons are
         non-empty, else SUFFICIENT.

    Args:
        bundle:        Current EvidenceBundle (post-retrieval, post-rerank).
        risk_profile:  This request's RiskProfile (read-only).
        config:        Gate thresholds; defaults to SufficiencyGateConfig().

    Returns:
        SufficiencyResult with verdict, signals, and rationale.
    """
    cfg = config or SufficiencyGateConfig()
    start = time.perf_counter()
    try:
        signals = compute_sufficiency_signals(bundle, risk_profile)
        high_risk = _is_high_risk(risk_profile, cfg)

        coverage_reasons = _evaluate_coverage(signals, cfg)

        if high_risk:
            risk_reasons = _evaluate_high_risk_concerns(signals, cfg)
            all_reasons = coverage_reasons + risk_reasons
            if all_reasons:
                return SufficiencyResult(
                    verdict=SufficiencyVerdict.CONSERVATIVE,
                    signals=signals,
                    reasons=tuple(all_reasons),
                    recommend_targeted_retrieval=not signals.targeted_retrieval_used,
                    is_high_risk=True,
                )
            return SufficiencyResult(
                verdict=SufficiencyVerdict.SUFFICIENT,
                signals=signals,
                is_high_risk=True,
            )

        if coverage_reasons:
            return SufficiencyResult(
                verdict=SufficiencyVerdict.INSUFFICIENT,
                signals=signals,
                reasons=tuple(coverage_reasons),
                recommend_targeted_retrieval=not signals.targeted_retrieval_used,
                is_high_risk=False,
            )

        return SufficiencyResult(
            verdict=SufficiencyVerdict.SUFFICIENT,
            signals=signals,
            is_high_risk=False,
        )
    finally:
        elapsed = time.perf_counter() - start
        bundle.record_stage_elapsed("sufficiency_gate", elapsed)