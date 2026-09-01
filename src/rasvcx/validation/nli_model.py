"""Concrete NLI backends (Section 11).

No backend in this module performs any work at import time: no model is
downloaded, loaded, or otherwise initialized merely because
``rasvcx.validation`` (or this module) was imported. Model loading happens
lazily, on first use, and only for the backend actually constructed and
used by the caller.
"""

from __future__ import annotations

import threading
from typing import Any

from rasvcx.schemas.common import NLILabel
from rasvcx.schemas.validation import NLISignal
from rasvcx.validation.nli_interface import NLIBackend, NLIUnavailableError


class NullNLIBackend(NLIBackend):
    """Always-unavailable backend.

    The safe default: used whenever no semantic model is configured (e.g.
    no GPU, offline environment, or the operator has not opted into
    NLI-backed validation). Deterministic and contextual validation must
    remain fully usable with this backend in place.
    """

    def is_available(self) -> bool:
        return False

    def predict(self, premise: str, hypothesis: str) -> NLISignal:
        raise NLIUnavailableError("NullNLIBackend has no model configured")


class TransformersNLIBackend(NLIBackend):
    """Optional backend wrapping a HuggingFace ``transformers`` NLI model.

    ``transformers``/``torch`` are imported lazily inside ``_ensure_loaded``
    -- never at module import time -- so importing ``rasvcx.validation``
    never requires these (optional) dependencies to be installed, and never
    pays model-load latency until a prediction is actually requested.

    Thread-safe lazy initialization: concurrent first calls to ``predict``
    will not race to load the model twice.
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-large-mnli",
        device: str = "cpu",
        label_map: dict[str, NLILabel] | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        # Most MNLI-style checkpoints emit "contradiction" / "neutral" /
        # "entailment"; this map lets callers adapt a differently-labeled
        # checkpoint without subclassing.
        self._label_map = label_map or {
            "contradiction": NLILabel.CONTRADICTION,
            "neutral": NLILabel.NEUTRAL,
            "entailment": NLILabel.ENTAILMENT,
        }
        self._lock = threading.Lock()
        self._pipeline: Any | None = None
        self._load_failed = False

    def is_available(self) -> bool:
        if self._load_failed:
            return False
        if self._pipeline is not None:
            return True
        # Cheap capability probe: only checks importability, does not
        # trigger the (expensive) actual model download/load.
        try:
            import importlib

            importlib.import_module("transformers")
        except ImportError:
            return False
        return True

    def predict(self, premise: str, hypothesis: str) -> NLISignal:
        self._ensure_loaded()
        assert self._pipeline is not None  # narrowed by _ensure_loaded

        try:
            raw = self._pipeline({"text": premise, "text_pair": hypothesis}, top_k=None)
        except Exception as exc:  # pragma: no cover - third-party failure path
            raise NLIUnavailableError(f"NLI inference failed: {exc}") from exc

        best = self._select_best(raw)
        raw_label = str(best.get("label", "")).lower()
        mapped = self._label_map.get(raw_label)
        if mapped is None:
            raise NLIUnavailableError(f"Unrecognized NLI label from model: {raw_label!r}")

        score = float(best.get("score", 0.0))
        score = min(1.0, max(0.0, score))
        return NLISignal(label=mapped, confidence=score)

    # -- internal helpers -----------------------------------------------

    def _select_best(self, raw: object) -> dict:
        # transformers pipelines with top_k=None return either a flat list
        # of {"label", "score"} dicts, or (batched) a list-of-lists.
        candidates = raw
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], list):
            candidates = candidates[0]
        if not isinstance(candidates, list) or not candidates:
            raise NLIUnavailableError("NLI model returned no predictions")
        return max(candidates, key=lambda item: float(item.get("score", 0.0)))

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        with self._lock:
            if self._pipeline is not None:
                return
            if self._load_failed:
                raise NLIUnavailableError(f"Model {self._model_name!r} previously failed to load")
            try:
                from transformers import pipeline as hf_pipeline

                self._pipeline = hf_pipeline(
                    task="text-classification",
                    model=self._model_name,
                    device=-1 if self._device == "cpu" else 0,
                )
            except Exception as exc:
                self._load_failed = True
                raise NLIUnavailableError(
                    f"Failed to load NLI model {self._model_name!r}: {exc}"
                ) from exc


__all__ = ["NullNLIBackend", "TransformersNLIBackend"]