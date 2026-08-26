"""Confidence score contract.

Confidence must combine multiple signals (evidence agreement, claim
verification support, contradiction, provenance quality, source diversity,
retrieval/rerank quality, resolution uncertainty) -- it must never be a
single retrieval/rerank score treated as a probability. Calibration must
not be claimed automatically: is_calibrated defaults to False and may only
be set True by a caller that has run labelled-data calibration evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ConfidenceFeatures:
    """The individual signals combined into a final confidence score.

    Each field is a named, bounded [0, 1] signal so the contribution of
    each factor is inspectable and testable in isolation, rather than
    folded into an opaque scalar.
    """

    evidence_agreement: float
    claim_verification_support: float
    contradiction_penalty: float
    provenance_quality: float
    source_diversity: float
    retrieval_quality: float
    rerank_quality: float
    resolution_uncertainty_penalty: float

    def __post_init__(self) -> None:
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
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"ConfidenceFeatures.{name} must be in [0, 1], got {value}")


@dataclass(frozen=True, slots=True)
class ConfidenceScore:
    """Final estimated confidence for a generated response.

    is_calibrated is False by default and must only be set True by a caller
    that has actually run ECE/Brier evaluation against labelled held-out
    data (see confidence/calibration.py). Constructing this with
    is_calibrated=True is a claim that must be backed by that evaluation --
    it is not enforced at the type level, only at the call-site contract.
    """

    value: float
    features: ConfidenceFeatures
    is_calibrated: bool = False
    calibration_method: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"ConfidenceScore.value must be in [0, 1], got {self.value}")
        if self.is_calibrated and self.calibration_method is None:
            raise ValueError(
                "ConfidenceScore.is_calibrated=True requires a calibration_method"
            )
        if not self.is_calibrated and self.calibration_method is not None:
            raise ValueError(
                "ConfidenceScore.calibration_method must be None when is_calibrated=False"
            )