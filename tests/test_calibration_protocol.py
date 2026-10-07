"""Calibration protocol and artifact binding.

These tests exercise the MACHINERY with synthetic scores.  They make no
claim that RASVC-X confidence is calibrated: no artifact ships with the
repository and the runtime reports "uncalibrated" until one is fitted on a
real calibration split.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from evaluation.protocol import (
    ProtocolError,
    apply,
    brier,
    calibration_report,
    check_frozen,
    choose_and_fit,
    dataset_hash,
    ece,
    freeze,
    reliability,
)


def _case(cid: str, split: str = "test", expected: str = "answer") -> SimpleNamespace:
    return SimpleNamespace(case_id=cid, query=f"q {cid}", split=SimpleNamespace(value=split),
                           expected_decision=expected, tags=["must:x"],
                           category=SimpleNamespace(value="clean"))


class TestMetrics:
    def test_perfectly_calibrated(self) -> None:
        conf = [0.25] * 4 + [0.75] * 4
        lab = [1, 0, 0, 0] + [1, 1, 1, 0]
        assert ece(conf, lab) == 0.0
        assert brier(conf, lab) == pytest.approx(0.1875)
        bins = reliability(conf, lab)
        assert bins[2]["n"] == 4 and bins[2]["accuracy"] == 0.25
        assert bins[7]["n"] == 4 and bins[7]["accuracy"] == 0.75

    def test_overconfident(self) -> None:
        assert ece([0.95] * 10, [1] * 5 + [0] * 5) == pytest.approx(0.45)

    def test_report_splits_by_risk_and_ignores_abstentions(self) -> None:
        rows = [{"confidence": 0.9, "correct": 1, "high_risk": True},
                {"confidence": 0.8, "correct": 0, "high_risk": False},
                {"confidence": 0.7, "correct": None, "high_risk": False}]
        rep = calibration_report(rows)
        assert rep["overall"]["n"] == 2
        assert rep["by_risk"]["high_risk"]["n"] == 1 and rep["by_risk"]["standard_risk"]["n"] == 1


class TestFitting:
    def test_refuses_insufficient_data(self) -> None:
        with pytest.raises(ProtocolError, match="UNCALIBRATED"):
            choose_and_fit([0.5] * 40, [1] * 20 + [0] * 20)
        with pytest.raises(ProtocolError):
            choose_and_fit([0.9] * 200, [1] * 195 + [0] * 5)  # one class too small

    def test_platt_reduces_ece_on_held_out_synthetic_scores(self) -> None:
        rng = random.Random(7)

        def sample(n: int) -> tuple[list[float], list[int]]:
            # raw scores systematically overconfident: true p = raw - 0.3
            s = [rng.uniform(0.5, 1.0) for _ in range(n)]
            return s, [1 if rng.random() < x - 0.3 else 0 for x in s]

        s_cal, y_cal = sample(600)
        s_test, y_test = sample(600)
        art, method = choose_and_fit(s_cal, y_cal)
        assert method == "platt"
        cal = [apply(art, x) for x in s_test]
        assert ece(cal, y_test) < ece(s_test, y_test) / 2


class TestFreeze:
    def test_freeze_then_detect_drift(self, tmp_path: Path) -> None:
        ds = tmp_path / "d.json"
        ds.write_text("{}")
        cases = [_case("a"), _case("b")]
        freeze(ds, cases)
        assert check_frozen(ds, cases)["n_test"] == 2
        with pytest.raises(ProtocolError, match="changed"):
            check_frozen(ds, [_case("a"), _case("b", expected="abstain")])
        with pytest.raises(ProtocolError, match="already"):
            freeze(ds, cases)

    def test_fit_requires_frozen_test(self, tmp_path: Path) -> None:
        with pytest.raises(ProtocolError, match="not frozen"):
            check_frozen(tmp_path / "x.json", [_case("a")])

    def test_dataset_hash_covers_labels_and_split(self) -> None:
        assert dataset_hash([_case("a")]) != dataset_hash([_case("a", expected="abstain")])
        assert dataset_hash([_case("a")]) != dataset_hash([_case("a", split="calibration")])


# ---------------------------------------------------------------------------
# Runtime binding: never silently stale
# ---------------------------------------------------------------------------

def _settings(tmp_path: Path, artifact: str | None = None):
    from rasvcx.config.settings import ConfidenceSettings
    from tests.test_integration_repair import _offline_settings

    s = _offline_settings(tmp_path)
    return replace(s, confidence=replace(s.confidence, calibration_artifact_path=artifact)) \
        if artifact else s


def _write_artifact(tmp_path: Path, settings, kb_version_id: str, identity_override: dict | None = None) -> str:
    from rasvcx.confidence.calibration import fit_platt
    from rasvcx.confidence.calibration_artifact import save_artifact
    from rasvcx.pipeline.factory import runtime_identity

    art = fit_platt([0.2, 0.4, 0.6, 0.8] * 10, [0, 0, 1, 1] * 10)
    ident = {**runtime_identity(settings), **(identity_override or {})}
    path = tmp_path / "cal.json"
    save_artifact(path, art, dataset_hash="d" * 64,
                  kb={"version_id": kb_version_id, "corpus_hash": "c" * 64},
                  runtime_identity=ident, n_calibration=40)
    return str(path)


class TestRuntimeBinding:
    Q = {"query": "What are the visiting hours?", "enriched": True}

    def test_default_is_uncalibrated(self, tmp_path: Path) -> None:
        from rasvcx.api.main import create_app

        with TestClient(create_app(settings=_settings(tmp_path))) as c:
            assert c.get("/ready").json()["calibration"]["status"] == "uncalibrated"
            d = c.post("/query", json=self.Q).json()
            assert d["calibration_status"] == "uncalibrated" and d["calibration_version"] is None

    def test_matching_artifact_is_applied(self, tmp_path: Path) -> None:
        from rasvcx.api.main import create_app
        from rasvcx.pipeline.factory import build_runtime

        kb = build_runtime(_settings(tmp_path)).initial_snapshot.version_id
        base = _settings(tmp_path)
        path = _write_artifact(tmp_path, base, kb)
        with TestClient(create_app(settings=_settings(tmp_path, path))) as c:
            ready = c.get("/ready").json()["calibration"]
            assert ready["status"] == "calibrated"
            d = c.post("/query", json=self.Q).json()
            assert d["calibration_status"] == "calibrated"
            assert d["calibration_version"] == ready["calibration_version"]
            assert d["calibration_dataset_hash"] == "d" * 64

    def test_other_kb_version_invalidates_per_request(self, tmp_path: Path) -> None:
        from rasvcx.api.main import create_app

        path = _write_artifact(tmp_path, _settings(tmp_path), "v_000000000000")
        with TestClient(create_app(settings=_settings(tmp_path, path))) as c:
            d = c.post("/query", json=self.Q).json()
            assert d["calibration_status"] == "invalidated"
            assert any("calibration not applied" in w for w in d["warnings"])

    def test_changed_prompt_or_config_invalidates_at_startup(self, tmp_path: Path) -> None:
        from rasvcx.api.main import create_app

        path = _write_artifact(tmp_path, _settings(tmp_path), "v_x", {"prompt_version": "sys-old"})
        with TestClient(create_app(settings=_settings(tmp_path, path))) as c:
            cal = c.get("/ready").json()["calibration"]
            assert cal["status"] == "invalidated" and "prompt_version" in cal["reason"]
            assert c.post("/query", json=self.Q).json()["calibration_status"] == "invalidated"

    def test_artifact_path_does_not_change_config_hash(self, tmp_path: Path) -> None:
        from rasvcx.pipeline.factory import config_hash

        assert config_hash(_settings(tmp_path)) == config_hash(_settings(tmp_path, "x.json"))

    def test_unreadable_artifact_stops_startup(self, tmp_path: Path) -> None:
        from rasvcx.config.settings import ConfigurationError
        from rasvcx.pipeline.factory import build_runtime

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"format": "nope"}))
        with pytest.raises(ConfigurationError, match="calibration artifact"):
            build_runtime(_settings(tmp_path, str(bad)))


class TestDemoSuiteIsNotCalibrationData:
    def test_demo_cases_are_safety_regression_only(self) -> None:
        from evaluation.dataset import load_dataset

        ds = load_dataset(Path("data/evaluation/demo_dataset.json"))
        assert {c.split.value for c in ds.cases} == {"safety_regression"}
