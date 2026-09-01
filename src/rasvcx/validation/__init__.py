"""Module 8 -- Validation.

Public surface for downstream modules (Generation/Verification/Confidence/
Decision). Importing this package performs no model loading, no network
access, and no expensive initialization -- see validation/nli_model.py for
the lazy-loading contract.
"""

from __future__ import annotations

from rasvcx.validation.candidate_generation import CandidateGenerator
from rasvcx.validation.contextual_validator import ContextualValidator
from rasvcx.validation.deterministic import (
    DeterministicValidator,
    extract_numbers,
    extract_years,
    significant_tokens,
)
from rasvcx.validation.nli_interface import NLIBackend, NLIService, NLIUnavailableError
from rasvcx.validation.nli_model import NullNLIBackend, TransformersNLIBackend
from rasvcx.validation.resolution import EvidenceResolver
from rasvcx.validation.selective_router import (
    RoutingAction,
    RoutingDecision,
    SelectiveValidationRouter,
    candidate_is_safety_critical,
)
from rasvcx.validation.validation_types import (
    CandidateClaimPair,
    CandidateGenerationConfig,
    DeterministicConfig,
    NumericExtraction,
    SelectiveRoutingConfig,
    ValidationConfig,
)
from rasvcx.validation.verified_context import ValidationPipeline, ValidationSummary

__all__ = [
    "CandidateGenerator",
    "ContextualValidator",
    "DeterministicValidator",
    "extract_numbers",
    "extract_years",
    "significant_tokens",
    "NLIBackend",
    "NLIService",
    "NLIUnavailableError",
    "NullNLIBackend",
    "TransformersNLIBackend",
    "EvidenceResolver",
    "RoutingAction",
    "RoutingDecision",
    "SelectiveValidationRouter",
    "candidate_is_safety_critical",
    "CandidateClaimPair",
    "CandidateGenerationConfig",
    "DeterministicConfig",
    "NumericExtraction",
    "SelectiveRoutingConfig",
    "ValidationConfig",
    "ValidationPipeline",
    "ValidationSummary",
]