"""Combines ConfidenceFeatures into a raw scalar. This is explicitly NOT a
calibrated probability -- see calibration.py for the layer that may
legitimately claim calibration. Constructing ConfidenceScore here always
sets is_calibrated=False (the schema forbids doing otherwise without a
calibration_method).

Mathematical note — positive-ceiling normalization
───────────────────────────────────────────────────
The estimator formula is:

    pre_norm = Σ wᵢ·fᵢ − Σ wⱼ·fⱼ      (i ∈ positive, j ∈ penalty)
    raw      = clamp(pre_norm / W_P, 0, 1)

where W_P = Σ wᵢ (i ∈ positive features only).

EstimatorWeights.__post_init__ enforces Σ(all 8 weights) = 1.0, but
positive and penalty weights operate asymmetrically: positives ADD
while penalties SUBTRACT.  Without normalization the maximum achievable
pre_norm (all positives = 1, all penalties = 0) equals W_P = 1 − W_N,
which is strictly less than 1.0 whenever any penalty weight is nonzero.
With the default weights W_P = 0.70, making thresholds above 0.70
structurally unreachable regardless of evidence quality.

Dividing by W_P rescales the score so that perfect positive evidence
with zero penalties produces raw = 1.0, and thresholds retain their
intended [0, 1] semantics.  Penalty deductions still subtract from the
pre_norm before division, so their effect scales proportionally.
Monotonicity is preserved (division by a positive constant).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rasvcx.schemas.confidence import ConfidenceFeatures, ConfidenceScore


@dataclass(frozen=True, slots=True)
class EstimatorWeights:
    evidence_agreement: float = 0.20
    claim_verification_support: float = 0.25
    contradiction_penalty: float = 0.20
    provenance_quality: float = 0.10
    source_diversity: float = 0.05
    retrieval_quality: float = 0.05
    rerank_quality: float = 0.05
    resolution_uncertainty_penalty: float = 0.10

    def __post_init__(self) -> None:
        weights = (
            self.evidence_agreement,
            self.claim_verification_support,
            self.contradiction_penalty,
            self.provenance_quality,
            self.source_diversity,
            self.retrieval_quality,
            self.rerank_quality,
            self.resolution_uncertainty_penalty,
        )
        for w in weights:
            if not math.isfinite(w):
                raise ValueError(f"EstimatorWeights must be finite, got {w}")
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"EstimatorWeights must each be in [0, 1], got {w}")

        total = sum(weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"EstimatorWeights must sum to 1.0, got {total}")


# Penalty-type features subtract from the score instead of adding.
_PENALTY_FIELDS = frozenset({"contradiction_penalty", "resolution_uncertainty_penalty"})


class RawReliabilityEstimator:
    def __init__(self, weights: EstimatorWeights | None = None) -> None:
        self._weights = weights or EstimatorWeights()
        # W_P: sum of positive (non-penalty) weights, used to normalize
        # the raw score so that perfect positive evidence yields 1.0.
        # Computed once here rather than per-estimate call.
        self._w_positive = sum(
            getattr(self._weights, name)
            for name in (
                "evidence_agreement",
                "claim_verification_support",
                "provenance_quality",
                "source_diversity",
                "retrieval_quality",
                "rerank_quality",
            )
        )

    def estimate(self, features: ConfidenceFeatures) -> ConfidenceScore:
        raw = 0.0
        for name in (
            "evidence_agreement",
            "claim_verification_support",
            "contradiction_penalty",
            "provenance_quality",
            "source_diversity",
            "retrieval_quality",
            "rerank_quality",
            "resolution_uncertainty_penalty",
        ):
            weight = getattr(self._weights, name)
            value = getattr(features, name)
            raw += -weight * value if name in _PENALTY_FIELDS else weight * value

        # Normalize by the positive-weight ceiling so that perfect positive
        # evidence with zero penalties produces raw = 1.0 (see module docstring).
        # Penalty terms can still drive the result below 0; clamp to [0, 1].
        normalized = raw / self._w_positive if self._w_positive > 0 else 0.0
        value = max(0.0, min(1.0, normalized))
        return ConfidenceScore(value=value, features=features, is_calibrated=False)