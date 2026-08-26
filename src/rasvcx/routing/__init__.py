"""RASVC-X risk/complexity routing: advisory validation-effort allocation.

Re-exports the public surface so downstream packages can do e.g.
``from rasvcx.routing import route_query`` instead of reaching into
individual files.
"""

from __future__ import annotations

from rasvcx.routing.risk_features import (
    QueryComplexitySignals,
    compute_ambiguity_score,
    compute_numeric_content_score,
    compute_query_complexity_score,
    compute_safety_critical_structure_score,
    compute_unit_sensitive_score,
    extract_risk_features,
)
from rasvcx.routing.risk_router import RiskRouterThresholds, RiskRouterWeights, route_query

__all__ = [
    "QueryComplexitySignals",
    "compute_ambiguity_score",
    "compute_numeric_content_score",
    "compute_query_complexity_score",
    "compute_safety_critical_structure_score",
    "compute_unit_sensitive_score",
    "extract_risk_features",
    "RiskRouterThresholds",
    "RiskRouterWeights",
    "route_query",
]