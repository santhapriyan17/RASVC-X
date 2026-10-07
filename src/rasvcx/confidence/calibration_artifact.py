"""On-disk calibration artifacts and their binding to the runtime.

A calibrator is only valid for the system that produced the scores it was
fitted on.  Every artifact therefore records the identity of that system --
KB version + corpus hash, config hash, prompt version, model versions -- and
the hash of the calibration split it was fitted on.  At startup the
artifact is compared with the running process; per request its KB version
is compared with the leased KB.  Any difference marks the confidence
INVALIDATED (the raw score is returned, labelled as such): a calibration
can never silently go stale.

Artifacts are written only by scripts/calibrate.py, which fits on the
calibration split and refuses to run without a frozen held-out test split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rasvcx.confidence.calibration import CalibrationArtifact, CalibrationMethod

ARTIFACT_FORMAT = "rasvcx-calibration-v1"

#: Runtime-identity fields an artifact must match to be applied.
_BOUND_FIELDS = ("config_hash", "prompt_version", "model_versions")


@dataclass(frozen=True)
class BoundCalibration:
    """A loaded artifact plus the identity it is valid for."""

    artifact: CalibrationArtifact
    calibration_version: str
    calibration_dataset_hash: str
    kb_version_id: str
    corpus_hash: str
    runtime_identity: dict[str, Any]
    n_calibration: int


def artifact_version(params: tuple[float, ...], method: str, dataset_hash: str) -> str:
    blob = json.dumps({"m": method, "p": list(params), "d": dataset_hash}, sort_keys=True)
    return "cal-" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def save_artifact(path: str | Path, artifact: CalibrationArtifact, *, dataset_hash: str,
                  kb: dict[str, Any], runtime_identity: dict[str, Any], n_calibration: int,
                  extra: dict[str, Any] | None = None) -> str:
    version = artifact_version(artifact.params, artifact.method.value, dataset_hash)
    record = {
        "format": ARTIFACT_FORMAT,
        "calibration_version": version,
        "method": artifact.method.value,
        "params": list(artifact.params),
        "feature_schema_version": artifact.feature_schema_version,
        "fitted_at": artifact.fitted_at,
        "calibration_dataset_hash": dataset_hash,
        "n_calibration": n_calibration,
        "kb_version_id": kb.get("version_id"),
        "corpus_hash": kb.get("corpus_hash"),
        "runtime_identity": {k: runtime_identity.get(k) for k in _BOUND_FIELDS},
        **(extra or {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return version


def load_artifact(path: str | Path) -> BoundCalibration:
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    if rec.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"{path}: not a {ARTIFACT_FORMAT} artifact")
    art = CalibrationArtifact(
        method=CalibrationMethod(rec["method"]),
        version=rec["calibration_version"],
        feature_schema_version=rec["feature_schema_version"],
        params=tuple(float(p) for p in rec["params"]),
        fitted_at=rec.get("fitted_at") or "",
        dataset_id=rec.get("calibration_dataset_hash"),
    )
    art.validate()
    return BoundCalibration(
        artifact=art,
        calibration_version=rec["calibration_version"],
        calibration_dataset_hash=rec["calibration_dataset_hash"],
        kb_version_id=rec["kb_version_id"],
        corpus_hash=rec["corpus_hash"],
        runtime_identity=rec.get("runtime_identity") or {},
        n_calibration=int(rec.get("n_calibration") or 0),
    )


def runtime_mismatch(bound: BoundCalibration, runtime_identity: dict[str, Any]) -> str | None:
    """Why this artifact does not apply to the running process (or None)."""
    diffs = [
        f for f in _BOUND_FIELDS
        if bound.runtime_identity.get(f) != runtime_identity.get(f)
    ]
    if diffs:
        return "calibration fitted under a different " + ", ".join(diffs)
    return None


__all__ = [
    "ARTIFACT_FORMAT",
    "BoundCalibration",
    "artifact_version",
    "load_artifact",
    "runtime_mismatch",
    "save_artifact",
]
