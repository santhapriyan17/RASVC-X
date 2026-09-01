# src/rasvcx/provenance/source_quality.py
"""Source quality assessment for Module 6 — Provenance & Temporal Evidence.

Maps a SourceType to a configurable numeric quality tier and produces an
auditable SourceQualitySignal. The default tier assignments reflect
general-purpose document authority conventions (higher-tier sources have
stronger methodological grounding or regulatory standing), but they are
NOT hard-coded medical ground truth: every tier is overridable via
SourceQualityConfig.

Design constraints:
  - Pure function over SourceType + config: no mutation, no I/O, no ML.
  - UNKNOWN source_type → explicit SourceQualitySignal with is_known=False.
    Missing source type is uncertainty, not invalidity.
  - No tier is treated as "proof of correctness": tiers are signals for
    downstream validation priority, not authority to skip validation.
  - Configurable: the default tiers are initial heuristics; ablation studies
    may replace them without touching the logic in analyzer.py.
  - SourceQualityScorer is a thin stateless class that wraps
    assess_source_quality() so analyzer.py can use dependency injection and
    tests can replace the scorer without monkey-patching module-level
    functions.
  - SourceQualityScoreResult explicitly carries is_known so that
    ItemProvenanceAnalysis (and every downstream consumer) can distinguish
    UNKNOWN source_type from SourceType.OTHER without losing epistemic
    information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from rasvcx.schemas.common import SourceType
from rasvcx.schemas.evidence import EvidenceItem


# ---------------------------------------------------------------------------
# Quality tier enumeration
# ---------------------------------------------------------------------------


class SourceTier(str, Enum):
    """Ordinal authority tier for a source type.

    TIER_1 represents the strongest methodological grounding or regulatory
    standing available in the default policy. TIER_4 is weakest or unknown.

    These tiers are policy-configurable (see SourceQualityConfig.tier_map)
    and are advisory signals, NOT a claim of absolute medical authority.
    """

    TIER_1 = "tier_1"   # Strongest: regulatory, authoritative guidelines
    TIER_2 = "tier_2"   # Strong: peer-reviewed literature, systematic evidence
    TIER_3 = "tier_3"   # Moderate: institutional, drug labeling
    TIER_4 = "tier_4"   # Weakest / unknown / fallback


# Numeric value associated with each tier for arithmetic downstream.
# Intentionally separated from the enum so the enum stays a pure label.
_TIER_VALUES: dict[SourceTier, float] = {
    SourceTier.TIER_1: 1.0,
    SourceTier.TIER_2: 0.75,
    SourceTier.TIER_3: 0.50,
    SourceTier.TIER_4: 0.25,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceQualityConfig:
    """Configurable mapping from SourceType to SourceTier.

    The default tier assignments are initial heuristics grounded in the
    general principle that regulatory documents and clinical guidelines
    carry stronger methodological accountability than institutional policies
    or uncategorized sources. These assignments are NOT hard-coded medical
    truth and MUST be revisited against domain-expert review and ablation
    results.

    tier_map overrides specific SourceType → SourceTier assignments.
    Any SourceType absent from tier_map falls through to default_tier.

    Attributes:
        tier_map:      Optional per-SourceType overrides.
        default_tier:  Tier assigned when no override exists for a given type.
        unknown_tier:  Tier assigned when source_type is UNKNOWN.
                       Kept separate from default_tier so the caller can
                       distinguish "unknown source" from "known but uncategorized."
    """

    tier_map: dict[SourceType, SourceTier] = field(default_factory=dict)
    default_tier: SourceTier = SourceTier.TIER_3
    unknown_tier: SourceTier = SourceTier.TIER_4

    def __post_init__(self) -> None:
        # Validate that any provided tier_map values are SourceTier instances.
        for src_type, tier in self.tier_map.items():
            if not isinstance(src_type, SourceType):
                raise TypeError(
                    f"tier_map key must be SourceType, got {type(src_type).__name__!r}"
                )
            if not isinstance(tier, SourceTier):
                raise TypeError(
                    f"tier_map value must be SourceTier, got {type(tier).__name__!r}"
                )


def _build_default_tier_map() -> dict[SourceType, SourceTier]:
    """Return the default tier map.

    Separated into a named function for testability (the map can be
    inspected and compared without constructing a full config object).

    Default reasoning (heuristic, not clinical authority):
      TIER_1: REGULATORY_DOCUMENT, CLINICAL_GUIDELINE — formal review /
              regulatory mandate provides the strongest accountability.
      TIER_2: PEER_REVIEWED_LITERATURE — methodological peer review.
      TIER_3: DRUG_LABEL, INSTITUTIONAL_POLICY — context-specific authority,
              may be jurisdiction- or institution-scoped.
      TIER_4: OTHER — catch-all; insufficient information to assign a tier.
    """
    return {
        SourceType.REGULATORY_DOCUMENT: SourceTier.TIER_1,
        SourceType.CLINICAL_GUIDELINE: SourceTier.TIER_1,
        SourceType.PEER_REVIEWED_LITERATURE: SourceTier.TIER_2,
        SourceType.DRUG_LABEL: SourceTier.TIER_3,
        SourceType.INSTITUTIONAL_POLICY: SourceTier.TIER_3,
        SourceType.OTHER: SourceTier.TIER_4,
    }


DEFAULT_SOURCE_QUALITY_CONFIG = SourceQualityConfig(
    tier_map=_build_default_tier_map(),
    default_tier=SourceTier.TIER_4,
    unknown_tier=SourceTier.TIER_4,
)


# ---------------------------------------------------------------------------
# Output signal (low-level: one assessment, raw inputs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceQualitySignal:
    """Auditable source quality signal for a single evidence item.

    Attributes:
        source_type:   The SourceType as recorded in the item's provenance.
                       May be UNKNOWN (is_known=False) if the type is the
                       UNKNOWN sentinel. This field holds the SourceType
                       enum value when known.
        is_known:      False when the provenance source_type was UNKNOWN.
                       UNKNOWN is never fabricated into a concrete SourceType.
        tier:          The assigned SourceTier, or unknown_tier when
                       is_known=False.
        tier_value:    Numeric representation of the tier in [0, 1].
                       Higher = stronger authority signal.
        reason:        Human-readable justification for the assignment,
                       for audit/logging purposes.
    """

    source_type: SourceType | None   # None iff is_known is False
    is_known: bool
    tier: SourceTier
    tier_value: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.tier_value <= 1.0:
            raise ValueError(
                f"SourceQualitySignal.tier_value must be in [0, 1], "
                f"got {self.tier_value}"
            )
        if self.is_known and self.source_type is None:
            raise ValueError(
                "SourceQualitySignal.source_type must not be None when is_known=True"
            )
        if not self.is_known and self.source_type is not None:
            raise ValueError(
                "SourceQualitySignal.source_type must be None when is_known=False"
            )


# ---------------------------------------------------------------------------
# High-level result: what SourceQualityScorer.score() returns
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceQualityScoreResult:
    """Result of scoring one EvidenceItem's source quality.

    This is the type consumed by analyzer.py (ItemProvenanceAnalysis
    construction). It wraps SourceQualitySignal with an item_id reference
    so the result is traceable back to its origin.

    Attributes:
        item_id:       The EvidenceItemId of the scored item.
        quality_score: Numeric quality score in [0, 1] (= signal.tier_value).
                       Higher = stronger authority signal.
        source_type:   The SourceType enum value, or None if is_known=False.
                       NEVER fabricated from UNKNOWN to SourceType.OTHER.
        is_known:      Explicitly preserved epistemic state. False means the
                       source_type was the UNKNOWN sentinel; the item must
                       contribute to the unknown_provenance_ratio, not be
                       treated as SourceType.OTHER.
        signal:        The full SourceQualitySignal for audit/logging.
    """

    item_id: object   # EvidenceItemId (NewType str); typed as object to avoid
                      # circular import — callers access it only for logging.
    quality_score: float
    source_type: SourceType | None   # None iff is_known is False
    is_known: bool
    signal: SourceQualitySignal

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(
                f"SourceQualityScoreResult.quality_score must be in [0, 1], "
                f"got {self.quality_score}"
            )
        # Mirror the signal invariants at the result level for belt-and-
        # suspenders safety: any discrepancy is a programming error.
        if self.is_known and self.source_type is None:
            raise ValueError(
                "SourceQualityScoreResult.source_type must not be None "
                "when is_known=True"
            )
        if not self.is_known and self.source_type is not None:
            raise ValueError(
                "SourceQualityScoreResult.source_type must be None "
                "when is_known=False"
            )
        if self.quality_score != self.signal.tier_value:
            raise ValueError(
                f"SourceQualityScoreResult.quality_score ({self.quality_score}) "
                f"must equal signal.tier_value ({self.signal.tier_value})"
            )


# ---------------------------------------------------------------------------
# Core computation — module-level pure functions
# ---------------------------------------------------------------------------


def assess_source_quality(
    source_type: "SourceType | object",   # accepts UNKNOWN sentinel
    config: SourceQualityConfig = DEFAULT_SOURCE_QUALITY_CONFIG,
) -> SourceQualitySignal:
    """Assess the authority tier of a single source_type field.

    Args:
        source_type: A SourceType enum value, or the UNKNOWN sentinel
                     (rasvcx.schemas.common.UNKNOWN). Any other type raises
                     TypeError to prevent silent misclassification.
        config:      Quality policy (tier_map, default_tier, unknown_tier).

    Returns:
        SourceQualitySignal with explicit is_known flag, tier, tier_value,
        and human-readable reason. Never raises on UNKNOWN; always returns
        a populated signal.

    Raises:
        TypeError: if source_type is neither SourceType nor the UNKNOWN
                   sentinel (catches programming errors early).
    """
    from rasvcx.schemas.common import _UnknownType

    # UNKNOWN sentinel path: explicit, conservative, no fabrication.
    if isinstance(source_type, _UnknownType):
        tier = config.unknown_tier
        return SourceQualitySignal(
            source_type=None,
            is_known=False,
            tier=tier,
            tier_value=_TIER_VALUES[tier],
            reason="source_type is UNKNOWN; no authority tier can be assigned",
        )

    if not isinstance(source_type, SourceType):
        raise TypeError(
            f"assess_source_quality expects SourceType or UNKNOWN, "
            f"got {type(source_type).__name__!r}"
        )

    # Known SourceType path: tier_map first, then default_tier.
    tier = config.tier_map.get(source_type, config.default_tier)
    if source_type in config.tier_map:
        reason = (
            f"source_type {source_type.value!r} mapped to {tier.value!r} "
            f"by policy tier_map"
        )
    else:
        reason = (
            f"source_type {source_type.value!r} has no explicit tier_map entry; "
            f"assigned default_tier {tier.value!r}"
        )

    return SourceQualitySignal(
        source_type=source_type,
        is_known=True,
        tier=tier,
        tier_value=_TIER_VALUES[tier],
        reason=reason,
    )


def aggregate_source_quality(
    signals: list[SourceQualitySignal],
) -> tuple[float, int, int]:
    """Aggregate source quality signals across an evidence bundle.

    Args:
        signals: One SourceQualitySignal per evidence item.

    Returns:
        Tuple of:
          mean_tier_value   — mean tier_value across all signals in [0, 1],
                              or 0.0 if signals is empty.
          known_count       — number of signals with is_known=True.
          unknown_count     — number of signals with is_known=False.

    Pure function; does not mutate inputs.
    """
    if not signals:
        return 0.0, 0, 0

    known_count = sum(1 for s in signals if s.is_known)
    unknown_count = len(signals) - known_count
    mean_tier_value = sum(s.tier_value for s in signals) / len(signals)

    return mean_tier_value, known_count, unknown_count


# ---------------------------------------------------------------------------
# SourceQualityScorer — injectable class wrapper
# ---------------------------------------------------------------------------


class SourceQualityScorer:
    """Stateless scorer that maps an EvidenceItem to a SourceQualityScoreResult.

    This class exists to support dependency injection in ProvenanceAnalyzer
    and to make unit tests replaceable via a simple mock or subclass. It
    carries no mutable state; every call to score() is a pure function over
    the item's provenance and the injected config.

    Design invariants:
      - Never mutates the input EvidenceItem.
      - Never fabricates a known SourceType from an UNKNOWN sentinel.
      - is_known=False in the result is the explicit signal that the item
        contributes to the unknown-provenance accounting in ProvenanceAnalyzer,
        NOT to SourceType.OTHER accounting.
      - Raises TypeError on malformed source_type (programming errors are loud).
    """

    def __init__(
        self,
        config: SourceQualityConfig = DEFAULT_SOURCE_QUALITY_CONFIG,
    ) -> None:
        """
        Args:
            config: Quality policy. Defaults to DEFAULT_SOURCE_QUALITY_CONFIG.
                    Injected at construction so the scorer can be swapped in
                    tests and ablation experiments without global mutation.
        """
        if not isinstance(config, SourceQualityConfig):
            raise TypeError(
                f"SourceQualityScorer config must be SourceQualityConfig, "
                f"got {type(config).__name__!r}"
            )
        self._config = config

    @property
    def config(self) -> SourceQualityConfig:
        """Read-only view of the active quality policy."""
        return self._config

    def score(self, item: EvidenceItem) -> SourceQualityScoreResult:
        """Score one EvidenceItem's source quality.

        Reads item.provenance.source_type; applies the tier policy; returns
        a SourceQualityScoreResult that explicitly preserves is_known so
        the caller can account for UNKNOWN sources separately from OTHER.

        Args:
            item: A (frozen) EvidenceItem. Not mutated.

        Returns:
            SourceQualityScoreResult with quality_score, source_type,
            is_known, and the underlying SourceQualitySignal for audit.

        Raises:
            TypeError: if item.provenance.source_type is not a SourceType
                       or the UNKNOWN sentinel (programming error guard).
        """
        if not isinstance(item, EvidenceItem):
            raise TypeError(
                f"SourceQualityScorer.score expects EvidenceItem, "
                f"got {type(item).__name__!r}"
            )

        signal = assess_source_quality(item.provenance.source_type, self._config)

        return SourceQualityScoreResult(
            item_id=item.item_id,
            quality_score=signal.tier_value,
            source_type=signal.source_type,   # None iff is_known=False
            is_known=signal.is_known,
            signal=signal,
        )
