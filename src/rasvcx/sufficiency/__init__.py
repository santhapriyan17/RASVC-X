"""Module 5: Sufficiency Gate."""

from __future__ import annotations

from rasvcx.sufficiency.scorer import (
    SufficiencySignals,
    compute_sufficiency_signals,
)
from rasvcx.sufficiency.gate import (
    SufficiencyGateConfig,
    SufficiencyResult,
    SufficiencyVerdict,
    evaluate_sufficiency,
)

__all__ = [
    "SufficiencySignals",
    "compute_sufficiency_signals",
    "SufficiencyGateConfig",
    "SufficiencyResult",
    "SufficiencyVerdict",
    "evaluate_sufficiency",
]