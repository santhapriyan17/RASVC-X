"""Tests for rasvcx.validation.nli_interface / nli_model.

Covers: NullNLIBackend is always unavailable; NLIService.predict_safe never
raises and never returns a signal from an unavailable/broken backend; a
broken backend (raising an unexpected exception) degrades to None rather
than propagating; importing the validation package performs no model
loading.
"""

from __future__ import annotations

from rasvcx.schemas.common import NLILabel
from rasvcx.schemas.validation import NLISignal
from rasvcx.validation.nli_interface import NLIService, NLIUnavailableError
from rasvcx.validation.nli_model import NullNLIBackend, TransformersNLIBackend


class _AlwaysAvailableGoodBackend:
    def is_available(self) -> bool:
        return True

    def predict(self, premise: str, hypothesis: str) -> NLISignal:
        return NLISignal(label=NLILabel.CONTRADICTION, confidence=0.77)


class _RaisesUnavailableBackend:
    def is_available(self) -> bool:
        return True

    def predict(self, premise: str, hypothesis: str) -> NLISignal:
        raise NLIUnavailableError("simulated model failure")


class _RaisesUnexpectedBackend:
    def is_available(self) -> bool:
        return True

    def predict(self, premise: str, hypothesis: str) -> NLISignal:
        raise RuntimeError("simulated unexpected crash, e.g. OOM")


class _BrokenAvailabilityCheckBackend:
    def is_available(self) -> bool:
        raise RuntimeError("availability probe itself is broken")

    def predict(self, premise: str, hypothesis: str) -> NLISignal:  # pragma: no cover
        raise AssertionError("predict should never be reached")


def test_null_backend_is_never_available():
    backend = NullNLIBackend()
    assert backend.is_available() is False


def test_null_backend_predict_raises_typed_error():
    backend = NullNLIBackend()
    try:
        backend.predict("a", "b")
        raised = False
    except NLIUnavailableError:
        raised = True
    assert raised


def test_service_with_no_backend_is_unavailable_and_returns_none():
    service = NLIService(backend=None)
    assert service.is_available() is False
    assert service.predict_safe("premise", "hypothesis") is None


def test_service_with_null_backend_returns_none_never_a_fabricated_signal():
    service = NLIService(backend=NullNLIBackend())
    result = service.predict_safe("premise", "hypothesis")
    assert result is None


def test_service_with_good_backend_returns_the_real_signal():
    service = NLIService(backend=_AlwaysAvailableGoodBackend())
    result = service.predict_safe("premise", "hypothesis")
    assert result is not None
    assert result.label is NLILabel.CONTRADICTION
    assert result.confidence == 0.77


def test_service_converts_nli_unavailable_error_to_none():
    service = NLIService(backend=_RaisesUnavailableBackend())
    assert service.predict_safe("premise", "hypothesis") is None


def test_service_never_propagates_unexpected_backend_exceptions():
    service = NLIService(backend=_RaisesUnexpectedBackend())
    # Must degrade gracefully -- never let a third-party backend crash the
    # pipeline, and never turn a crash into a fabricated positive result.
    result = service.predict_safe("premise", "hypothesis")
    assert result is None


def test_service_treats_broken_availability_check_as_unavailable():
    service = NLIService(backend=_BrokenAvailabilityCheckBackend())
    assert service.is_available() is False
    assert service.predict_safe("premise", "hypothesis") is None


def test_transformers_backend_is_available_reports_false_without_dependency_or_true_if_present():
    # Only asserts the *contract* (a bool, never raises) -- whether
    # `transformers` happens to be installed in this environment is not
    # asserted either way; this repository declares it as optional.
    backend = TransformersNLIBackend()
    result = backend.is_available()
    assert isinstance(result, bool)


def test_constructing_transformers_backend_does_not_load_a_model():
    # Model loading must be lazy: constructing the backend alone must not
    # touch the network or load model weights.
    backend = TransformersNLIBackend(model_name="some/nonexistent-model-xyz")
    assert backend._pipeline is None


def test_importing_validation_package_performs_no_model_loading():
    import importlib

    import rasvcx.validation as validation_pkg

    importlib.reload(validation_pkg)
    # No assertion beyond "import completes quickly and without side
    # effects" -- if model loading happened at import time this would be
    # slow and/or require network access, which the sandboxed test run
    # does not have.
    assert hasattr(validation_pkg, "ValidationPipeline")