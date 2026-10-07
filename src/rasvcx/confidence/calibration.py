"""Calibration turns a raw reliability score into a calibrated probability
estimate -- but only when backed by a fitted CalibrationArtifact evaluated
against held-out labelled data. Runtime application is O(1)/O(log n)
(binary search for isotonic); fitting (fit_platt/fit_isotonic) is an
explicitly offline operation, never invoked per-request.

is_calibrated=True is a claim (schemas/confidence.py enforces
calibration_method must accompany it); this module never sets it True
without a successfully-applied, structurally valid CalibrationArtifact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from rasvcx.schemas.confidence import ConfidenceScore

FEATURE_SCHEMA_VERSION = "confidence_features_v1"


class CalibrationMethod(str, Enum):
    IDENTITY = "identity"
    PLATT = "platt"
    ISOTONIC = "isotonic"


class CalibrationArtifactError(Exception):
    """Structurally invalid artifact (version mismatch, malformed params,
    non-monotonic isotonic table). Callers must treat this as
    "calibration unavailable", never as a reason to fabricate a value.
    """


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    method: CalibrationMethod
    version: str
    feature_schema_version: str
    params: tuple[float, ...]
    fitted_at: str
    dataset_id: str | None = None

    def validate(self) -> None:
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise CalibrationArtifactError(
                f"feature_schema_version mismatch: artifact={self.feature_schema_version!r} "
                f"runtime={FEATURE_SCHEMA_VERSION!r}"
            )
        if self.method is CalibrationMethod.PLATT:
            if len(self.params) != 2:
                raise CalibrationArtifactError("platt artifact requires exactly 2 params (a, b)")
            if any(math.isnan(v) or math.isinf(v) for v in self.params):
                raise CalibrationArtifactError("platt artifact params must be finite")
        elif self.method is CalibrationMethod.ISOTONIC:
            if len(self.params) < 4 or len(self.params) % 2 != 0:
                raise CalibrationArtifactError(
                    "isotonic artifact requires an even number of params >= 4 "
                    "(x0, y0, x1, y1, ...)"
                )
            xs = self.params[0::2]
            ys = self.params[1::2]
            if any(math.isnan(v) or math.isinf(v) for v in self.params):
                raise CalibrationArtifactError("isotonic artifact contains NaN/inf")
            if list(xs) != sorted(xs):
                raise CalibrationArtifactError("isotonic artifact x-breakpoints must be sorted")
            if list(ys) != sorted(ys):
                raise CalibrationArtifactError("isotonic artifact y-values must be non-decreasing")
            if any(not 0.0 <= y <= 1.0 for y in ys):
                raise CalibrationArtifactError("isotonic artifact y-values must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    score: ConfidenceScore
    status: str  # "calibrated" | "uncalibrated" | "invalidated" | "unavailable"
    reason: str | None = None
    calibration_version: str | None = None
    calibration_dataset_hash: str | None = None


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _apply_platt(raw: float, params: tuple[float, ...]) -> float:
    a, b = params
    return max(0.0, min(1.0, _sigmoid(a * raw + b)))


def _apply_isotonic(raw: float, params: tuple[float, ...]) -> float:
    xs = params[0::2]
    ys = params[1::2]
    if raw <= xs[0]:
        return ys[0]
    if raw >= xs[-1]:
        return ys[-1]
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= raw:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    if x1 == x0:
        return y0
    t = (raw - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


class Calibrator:
    """Applies a CalibrationArtifact at inference time. Never fits; see
    fit_platt/fit_isotonic for the offline counterpart.
    """

    def __init__(
        self,
        artifact: CalibrationArtifact | None,
        *,
        invalid_reason: str | None = None,
        kb_version_id: str | None = None,
        calibration_version: str | None = None,
        calibration_dataset_hash: str | None = None,
    ) -> None:
        """invalid_reason: the artifact exists but does not apply to this
        runtime (config / prompt / model changed) -> every outcome is
        "invalidated".  kb_version_id: the KB the artifact was fitted
        against; a request served from another KB version is "invalidated".
        """
        self._artifact = artifact
        self._invalid_reason = invalid_reason
        self._kb_version_id = kb_version_id
        self._version = calibration_version
        self._dataset_hash = calibration_dataset_hash

    def calibrate(
        self, raw_score: ConfidenceScore, kb_version_id: str | None = None,
    ) -> CalibrationOutcome:
        raw = raw_score.value
        if math.isnan(raw) or math.isinf(raw):
            raise ValueError(f"raw score must be finite, got {raw}")

        if self._artifact is None and self._invalid_reason is None:
            return CalibrationOutcome(score=raw_score, status="uncalibrated", reason="no_artifact")
        meta = {"calibration_version": self._version, "calibration_dataset_hash": self._dataset_hash}
        if self._invalid_reason is not None:
            return CalibrationOutcome(score=raw_score, status="invalidated",
                                      reason=self._invalid_reason, **meta)
        if (
            self._kb_version_id is not None and kb_version_id is not None
            and kb_version_id != self._kb_version_id
        ):
            return CalibrationOutcome(
                score=raw_score, status="invalidated",
                reason=f"calibration fitted on KB {self._kb_version_id}, request served from {kb_version_id}",
                **meta,
            )
        assert self._artifact is not None

        try:
            self._artifact.validate()
        except CalibrationArtifactError as exc:
            return CalibrationOutcome(score=raw_score, status="unavailable", reason=str(exc))

        if self._artifact.method is CalibrationMethod.IDENTITY:
            return CalibrationOutcome(score=raw_score, status="uncalibrated", reason="identity_method")

        if self._artifact.method is CalibrationMethod.PLATT:
            calibrated_value = _apply_platt(raw, self._artifact.params)
        else:
            calibrated_value = _apply_isotonic(raw, self._artifact.params)

        calibrated = ConfidenceScore(
            value=calibrated_value,
            features=raw_score.features,
            is_calibrated=True,
            calibration_method=self._artifact.method.value,
        )
        return CalibrationOutcome(score=calibrated, status="calibrated", **meta)


def fit_platt(
    raw_scores: list[float], labels: list[int], iterations: int = 200, learning_rate: float = 0.1
) -> CalibrationArtifact:
    """Offline: 1-D logistic regression (a*x + b) via gradient descent.
    Deterministic (fixed init a=1, b=0, no randomness); pure Python, no ML
    dependency, since this fits only 2 parameters.
    """
    if len(raw_scores) != len(labels):
        raise ValueError("raw_scores and labels must have equal length")
    if not raw_scores:
        raise ValueError("cannot fit on empty data")
    if any(math.isnan(x) or math.isinf(x) for x in raw_scores):
        raise ValueError("raw_scores must be finite")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary (0 or 1)")
    if len(set(labels)) < 2:
        raise ValueError("labels must contain both classes to fit Platt scaling")

    a, b = 1.0, 0.0
    n = len(raw_scores)
    for _ in range(iterations):
        grad_a = 0.0
        grad_b = 0.0
        for x, y in zip(raw_scores, labels):
            p = _sigmoid(a * x + b)
            error = p - y
            grad_a += error * x
            grad_b += error
        a -= learning_rate * grad_a / n
        b -= learning_rate * grad_b / n

    return CalibrationArtifact(
        method=CalibrationMethod.PLATT,
        version="fit_platt_v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        params=(a, b),
        fitted_at="",
    )


def fit_isotonic(raw_scores: list[float], labels: list[int]) -> CalibrationArtifact:
    """Offline: pool-adjacent-violators algorithm (PAVA), O(n log n)."""
    if len(raw_scores) != len(labels):
        raise ValueError("raw_scores and labels must have equal length")
    if not raw_scores:
        raise ValueError("cannot fit on empty data")
    if any(math.isnan(x) or math.isinf(x) for x in raw_scores):
        raise ValueError("raw_scores must be finite")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be binary (0 or 1)")

    pairs = sorted(zip(raw_scores, labels))
    xs = [p[0] for p in pairs]
    ys = [float(p[1]) for p in pairs]

    blocks: list[list[float]] = [[y, 1.0] for y in ys]
    i = 0
    while i < len(blocks) - 1:
        mean_i = blocks[i][0] / blocks[i][1]
        mean_next = blocks[i + 1][0] / blocks[i + 1][1]
        if mean_i > mean_next:
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1

    out_xs: list[float] = []
    out_ys: list[float] = []
    idx = 0
    for total, count in blocks:
        mean = total / count
        block_size = int(count)
        out_xs.append(xs[idx])
        out_ys.append(max(0.0, min(1.0, mean)))
        idx += block_size
        if idx < len(xs) and block_size > 1:
            out_xs.append(xs[idx - 1])
            out_ys.append(max(0.0, min(1.0, mean)))

    params: list[float] = []
    for x, y in zip(out_xs, out_ys):
        params.extend([x, y])

    return CalibrationArtifact(
        method=CalibrationMethod.ISOTONIC,
        version="fit_isotonic_v1",
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        params=tuple(params),
        fitted_at="",
    )