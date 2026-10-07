"""Confidence calibration protocol: freeze -> fit -> evaluate.

    python scripts/calibrate.py freeze   --dataset D.json
    python scripts/calibrate.py fit      --dataset D.json --expect-kb K.json --out cal.json
    python scripts/calibrate.py evaluate --dataset D.json --expect-kb K.json --artifact cal.json

Rules enforced here (evaluation/protocol.py):
  * The test split is hashed and FROZEN before anything is fitted; fit and
    evaluate refuse if it changed.
  * fit uses ONLY split == "calibration".  safety_regression, dev and test
    cases are never fitted on.
  * fit runs the pipeline WITHOUT a calibrator (raw scores) and refuses if
    the server configuration already applies one.
  * Samples are RETURNED answers labelled by gold correctness (1 correct /
    0 wrong-or-unsafe); abstentions and provider/system errors are counted,
    never labelled.  Too little data -> no artifact; confidence stays
    UNCALIBRATED.
  * evaluate runs the frozen test split once per artifact (repeat runs are
    logged in the lock file) and reports raw vs calibrated ECE, Brier,
    reliability and accuracy by confidence bucket and by risk level.
  * The artifact records calibration_version, the calibration-split hash,
    the KB identity and the runtime identity (config hash, prompt version,
    model versions); the server marks it INVALIDATED if any of them differ.

Every case is a billed provider call in research modes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def _load(args):
    from evaluation.dataset import load_dataset
    return load_dataset(Path(args.dataset))


def _split(ds, name: str) -> list:
    return [c for c in ds.cases if c.split.value == name]


def _runtime(args, require_uncalibrated: bool):
    from evaluation.protocol import ProtocolError, kb_mismatches
    from rasvcx.config.loader import load_settings
    from rasvcx.pipeline.factory import build_runtime

    expected = json.loads(Path(args.expect_kb).read_text(encoding="utf-8"))
    settings = load_settings()
    if settings.is_offline:
        raise ProtocolError("offline_test uses a stub LLM; calibration is meaningless in this mode")
    if require_uncalibrated and settings.confidence.calibration_artifact_path:
        raise ProtocolError("unset confidence.calibration_artifact_path: fitting needs raw scores")
    runtime = build_runtime(settings)
    snap = runtime.initial_snapshot
    problems = kb_mismatches(expected, snap.describe())
    if problems:
        raise ProtocolError("BENCHMARK_CONFIGURATION_ERROR: wrong KB: " + "; ".join(problems))
    return runtime, snap


def _run(runtime, snap, cases) -> list[dict]:
    from evaluation.protocol import run_b3_case

    rows = []
    for i, case in enumerate(cases, 1):
        row = run_b3_case(runtime, snap, case)
        rows.append(row)
        print(f"  [{i}/{len(cases)}] {case.case_id:<30} {row['outcome']:<16} "
              f"conf={row.get('confidence')}")
    return rows


def _meta(runtime, snap, cases, split: str) -> dict:
    import platform
    from evaluation.protocol import dataset_hash

    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "split": split, "dataset_hash": dataset_hash(cases), "n_cases": len(cases),
        "kb": snap.describe(), "run_identity": runtime.identity(),
        "cache_policy": "none: in-process runs",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }


def cmd_freeze(args) -> int:
    from evaluation.protocol import freeze
    ds = _load(args)
    rec = freeze(Path(args.dataset), _split(ds, "test"))
    print(f"test split frozen: {rec['n_test']} cases, hash {rec['test_hash'][:16]}")
    return 0


def cmd_fit(args) -> int:
    from evaluation.protocol import check_frozen, choose_and_fit, dataset_hash
    from rasvcx.confidence.calibration_artifact import save_artifact
    from rasvcx.pipeline.factory import runtime_identity

    ds = _load(args)
    check_frozen(Path(args.dataset), _split(ds, "test"))
    cal_cases = _split(ds, "calibration")
    if not cal_cases:
        print("no calibration split in this dataset; confidence stays UNCALIBRATED")
        return 3
    runtime, snap = _runtime(args, require_uncalibrated=True)
    rows = _run(runtime, snap, cal_cases)
    samples = [r for r in rows if r.get("correct") is not None]
    counts = {k: sum(1 for r in rows if r["outcome"] == k) for k in {r["outcome"] for r in rows}}
    meta = _meta(runtime, snap, cal_cases, "calibration")
    report = {"status": "INSUFFICIENT_DATA", **meta, "outcomes": counts, "rows": rows}
    try:
        art, method = choose_and_fit([r["confidence"] for r in samples], [r["correct"] for r in samples],
                                     args.method)
    except Exception as exc:  # ProtocolError: not enough data
        report["reason"] = str(exc)
        Path(args.report or (args.out + ".report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"NOT FITTED: {exc}")
        return 3
    import dataclasses
    art = dataclasses.replace(art, fitted_at=meta["timestamp_utc"])
    version = save_artifact(
        args.out, art, dataset_hash=dataset_hash(cal_cases), kb=snap.describe(),
        runtime_identity=runtime_identity(runtime.settings), n_calibration=len(samples),
        extra={"method_selected": method},
    )
    report.update(status="FITTED", calibration_version=version, method=method, n_samples=len(samples))
    Path(args.report or (args.out + ".report.json")).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"FITTED {method} on {len(samples)} answered samples -> {args.out} ({version})")
    return 0


def cmd_evaluate(args) -> int:
    from evaluation.protocol import ProtocolError, apply, calibration_report, check_frozen, lock_path
    from rasvcx.confidence.calibration_artifact import load_artifact, runtime_mismatch
    from rasvcx.pipeline.factory import runtime_identity

    ds = _load(args)
    test_cases = _split(ds, "test")
    lock = check_frozen(Path(args.dataset), test_cases)
    bound = load_artifact(args.artifact)
    if any(e.get("calibration_version") == bound.calibration_version for e in lock["evaluations"]) \
            and not args.allow_reevaluation:
        raise ProtocolError(
            f"{bound.calibration_version} was already evaluated on this held-out split; "
            "pass --allow-reevaluation to repeat it (the repeat is recorded)"
        )
    runtime, snap = _runtime(args, require_uncalibrated=True)
    why = runtime_mismatch(bound, runtime_identity(runtime.settings))
    if why or bound.kb_version_id != snap.version_id:
        raise ProtocolError(f"artifact does not match this runtime: {why or 'different KB version'}")
    rows = _run(runtime, snap, test_cases)
    for r in rows:
        if r.get("confidence") is not None:
            r["calibrated_confidence"] = round(apply(bound.artifact, r["confidence"]), 4)
    result = {
        **_meta(runtime, snap, test_cases, "test"),
        "calibration_version": bound.calibration_version,
        "calibration_dataset_hash": bound.calibration_dataset_hash,
        "raw": calibration_report(rows, "confidence"),
        "calibrated": calibration_report(rows, "calibrated_confidence"),
        "outcomes": {k: sum(1 for r in rows if r["outcome"] == k) for k in {r["outcome"] for r in rows}},
        "rows": rows,
    }
    out = Path(args.out or f"eval_output/calibration_eval_{bound.calibration_version}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lock["evaluations"].append({"calibration_version": bound.calibration_version,
                                "at": result["timestamp_utc"], "report": str(out)})
    lock_path(Path(args.dataset)).write_text(json.dumps(lock, indent=2), encoding="utf-8")
    for k in ("raw", "calibrated"):
        o = result[k]["overall"]
        print(f"{k:<11} n={o['n']} acc={o['accuracy']} ECE={o['ece']} Brier={o['brier']}")
    print(f"written: {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--dataset", required=True)
    t = sub.add_parser("fit")
    for a in ("--dataset", "--expect-kb", "--out"):
        t.add_argument(a, required=True)
    t.add_argument("--method", choices=("auto", "platt", "isotonic"), default="auto")
    t.add_argument("--report", default="")
    e = sub.add_parser("evaluate")
    for a in ("--dataset", "--expect-kb", "--artifact"):
        e.add_argument(a, required=True)
    e.add_argument("--out", default="")
    e.add_argument("--allow-reevaluation", action="store_true")
    args = p.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
    except ImportError:
        pass
    import logging
    logging.basicConfig(level="WARNING")
    from evaluation.protocol import ProtocolError
    try:
        return {"freeze": cmd_freeze, "fit": cmd_fit, "evaluate": cmd_evaluate}[args.cmd](args)
    except ProtocolError as exc:
        print(f"REFUSED: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
