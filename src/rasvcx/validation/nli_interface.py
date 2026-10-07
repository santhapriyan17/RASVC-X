"""NLI interface (Section 11).

Defines the service boundary between M8 and any semantic/NLI model, so the
rest of the validation package (selective routing, resolution) never
depends on a concrete model implementation. This module performs no model
loading and has no heavy dependencies; concrete backends live in
``validation/nli_model.py`` and are imported lazily.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rasvcx.schemas.common import NLILabel
from rasvcx.schemas.validation import NLISignal


class NLIUnavailableError(Exception):
    """Raised by an NLIBackend when it cannot service a prediction request.

    Covers: model not installed, model failed to load, request timed out,
    or any other reason inference could not be completed. Callers (see
    ``NLIService.predict``) must treat this identically regardless of
    cause -- graceful degradation, never a crash, and never fabricated
    certainty.
    """


@runtime_checkable
class NLIBackend(Protocol):
    """Minimal interface a concrete NLI model implementation must satisfy.

    Implementations may wrap any underlying model family (DeBERTa-style,
    RoBERTa-style, or otherwise) without requiring changes to this
    protocol or to any M8 caller.
    """

    def predict(self, premise: str, hypothesis: str) -> NLISignal:
        """Return the model's entailment/contradiction/neutral judgment.

        Must raise ``NLIUnavailableError`` (not return a fabricated
        signal) if a genuine prediction cannot be produced.
        """
        ...

    def is_available(self) -> bool:
        """Cheap, side-effect-free check of whether ``predict`` is likely
        to succeed right now (e.g. model weights present/loaded).

        Used by selective routing to avoid attempting -- and paying the
        latency cost of -- calls that are known to be unserviceable.
        """
        ...


class NLIService:
    """Thin, defensive wrapper around a single injected NLIBackend.

    - Never loads a model itself.
    - Never raises out of ``predict_safe``: failures are converted into
      ``None``, and callers must treat ``None`` as "no NLI signal
      available" -- never as entailment or contradiction.
    - Bounded: does not retry internally: retry/timeout policy is a
      concern of the concrete backend or of the caller's budget, not of
      this thin wrapper.
    """

    def __init__(self, backend: NLIBackend | None) -> None:
        self._backend = backend

    @property
    def backend(self) -> NLIBackend | None:
        """The injected backend (for readiness reporting; never call
        predict on it directly -- use predict_safe)."""
        return self._backend

    def is_available(self) -> bool:
        if self._backend is None:
            return False
        try:
            return self._backend.is_available()
        except Exception:
            # A broken availability check is itself unavailability.
            return False

    def predict_safe(self, premise: str, hypothesis: str) -> NLISignal | None:
        """Attempt an NLI prediction; return None on any failure.

        This is the only method the rest of M8 should call -- it converts
        NLIUnavailableError, and any other unexpected exception raised by
        a third-party backend, into a clean "no signal" outcome rather
        than propagating an implementation-specific error.
        """
        if self._backend is None:
            return None
        import time as _time

        t0 = _time.perf_counter()
        try:
            if not self._backend.is_available():
                return None
            return self._backend.predict(premise, hypothesis)
        except NLIUnavailableError:
            return None
        except Exception:
            # Defensive: a third-party backend raising something other
            # than NLIUnavailableError must still degrade gracefully.
            return None
        finally:
            # Per-thread accounting (one request = one worker thread) so the
            # orchestrator can report NLI inference time as its own stage.
            acc = _NLI_TIME.__dict__
            acc["seconds"] = acc.get("seconds", 0.0) + (_time.perf_counter() - t0)
            acc["calls"] = acc.get("calls", 0) + 1

    @staticmethod
    def take_thread_timing() -> tuple[float, int]:
        """(seconds, calls) of NLI inference on this thread since the last
        take, then reset."""
        acc = _NLI_TIME.__dict__
        out = (acc.get("seconds", 0.0), acc.get("calls", 0))
        acc["seconds"], acc["calls"] = 0.0, 0
        return out


import threading as _threading  # noqa: E402

_NLI_TIME = _threading.local()


__all__ = [
    "NLIUnavailableError",
    "NLIBackend",
    "NLIService",
    "NLILabel",
    "NLISignal",
]