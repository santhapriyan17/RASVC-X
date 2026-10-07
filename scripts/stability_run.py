"""Repeated-run stability of the RASVC-X pipeline on one dataset.

    . .\\scripts\\demo_env.ps1
    python scripts/stability_run.py --runs 5

Runs every case N times on ONE runtime (same KB, config, models), then:

  per run       correct / false-abstention / safe / unsafe rates, abstention
                rate, provider + system errors, latency p50
  across runs   mean, standard deviation, min, max of each rate
  per case      decision distribution and whether it flipped
  attribution   for every case whose decision varied: did the retrieved
                evidence (chunk ids, in order) vary, or only the generated
                answer?  Evidence identical + answer different = generation
                (LLM) nondeterminism; evidence different = retrieval side.
                A separate determinism probe re-runs retrieval + reranking
                (no LLM) per case and compares chunk ids and rerank scores.

No cache is involved (in-process).  KB identity is verified first; a wrong
KB stops the run with BENCHMARK_CONFIGURATION_ERROR.  Every case is a billed
provider call per run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {"n": len(values), "mean": round(statistics.fmean(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4), "max": round(max(values), 4)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/evaluation/demo_dataset.json")
    p.add_argument("--expect-kb", default="")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--probe-repeats", type=int, default=3)
    p.add_argument("--pace-seconds", type=float, default=0.0,
                   help="pause between cases (client-side throttle; not counted in latency)")
    p.add_argument("--out", default="")
    args = p.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
    except ImportError:
        pass
    import logging
    logging.basicConfig(level="WARNING")

    from evaluation.dataset import load_dataset
    from evaluation.protocol import dataset_hash, kb_mismatches, provider_request_report, run_b3_case
    from rasvcx.config.loader import load_settings
    from rasvcx.pipeline.factory import build_runtime
    from rasvcx.routing import route_query
    from rasvcx.schemas.common import QueryId
    from rasvcx.schemas.evidence import EvidenceBundle
    from rasvcx.schemas.query import QueryRequest

    ds = load_dataset(Path(args.dataset))
    cases = list(ds.cases)
    expect = Path(args.expect_kb) if args.expect_kb else Path(args.dataset).with_suffix(".kb.json")
    settings = load_settings()
    if settings.is_offline:
        print("BENCHMARK_CONFIGURATION_ERROR: offline_test")
        return 3
    runtime = build_runtime(settings)
    snap = runtime.initial_snapshot
    problems = kb_mismatches(json.loads(expect.read_text(encoding="utf-8")), snap.describe())
    if problems:
        print("BENCHMARK_CONFIGURATION_ERROR:", "; ".join(problems))
        return 3
    print(f"KB {snap.version_id} verified | {len(cases)} cases x {args.runs} runs")

    # -- determinism probe: retrieval + reranking only (no LLM) ------------
    probe: dict[str, dict] = {}
    for case in cases:
        sigs = set()
        for _ in range(args.probe_repeats):
            q = QueryRequest(query_id=QueryId(case.case_id), raw_text=case.query,
                             normalized_text=case.query.lower())
            b = EvidenceBundle(query_id=q.query_id, risk_profile=route_query(q.normalized_text))
            snap.retrieval_fn(q, b.risk_profile, b)
            runtime.reranking_service.rerank_bundle(b, q.normalized_text)
            sigs.add(tuple((str(it.chunk_id), None if it.rerank_score is None else round(it.rerank_score, 6))
                           for it in b.evidence_items.values()))
        probe[case.case_id] = {"distinct_retrieval_rerank_outputs": len(sigs)}
    nondet = [c for c, v in probe.items() if v["distinct_retrieval_rerank_outputs"] > 1]
    print(f"retrieval+rerank determinism: {len(cases) - len(nondet)}/{len(cases)} cases identical "
          f"across {args.probe_repeats} repeats" + (f"; varied: {nondet}" if nondet else ""))

    # -- repeated full runs ----------------------------------------------------
    runs: list[list[dict]] = []
    t_start = time.time()
    for r in range(args.runs):
        rows = []
        for case in cases:
            if args.pace_seconds:
                time.sleep(args.pace_seconds)
            rows.append(run_b3_case(runtime, snap, case))
        runs.append(rows)
        line = " ".join(f"{row['case_id'][:10]}:{row['outcome'][:4]}" for row in rows)
        print(f"run {r + 1}: {line}")

    t_end = time.time()
    EXCLUDED = ("provider_error", "system_error")

    def rates(rows: list[dict]) -> dict:
        ok = [x for x in rows if x["outcome"] not in EXCLUDED]
        ans = [x for x in ok if x["expected"] == "answer"]
        abst = [x for x in ok if x["expected"] == "abstain"]

        def frac(sub, outcome):
            return sum(1 for x in sub if x["outcome"] == outcome) / len(sub) if sub else float("nan")

        lat = sorted(x["latency_s"] for x in ok)
        return {
            "correct_rate": frac(ans, "correct"), "false_abstention_rate": frac(ans, "false_abstention"),
            "wrong_rate": frac(ans, "wrong"), "safe_rate": frac(abst, "safe"),
            "unsafe_rate": frac(abst, "unsafe"),
            "abstention_rate": sum(1 for x in ok if not x.get("answered")) / len(ok) if ok else float("nan"),
            "valid_cases": len(ok),
            "latency_p50_s": lat[len(lat) // 2] if lat else float("nan"),
        }

    all_rows = [x for rows in runs for x in rows]
    per_run = [rates(rows) for rows in runs]
    across = {k: _stats([pr[k] for pr in per_run if pr[k] == pr[k]]) for k in per_run[0]}
    all_lat = [x["latency_s"] for x in all_rows if x["outcome"] not in EXCLUDED]

    # ---- A. final-outcome stability (valid runs only) ----------------------
    per_case = {}
    for i, case in enumerate(cases):
        rows = [run[i] for run in runs]
        valid = [x for x in rows if x["outcome"] not in EXCLUDED]
        decisions = [x.get("decision") for x in valid]
        answered = [bool(x.get("answered")) for x in valid]
        safety = ["unsafe" if x["outcome"] == "unsafe" else "wrong" if x["outcome"] == "wrong" else "ok"
                  for x in valid]
        ev = {tuple(x["evidence_ids"]) for x in valid}
        ans = {x["answer_sha"] for x in valid if x.get("answered")}
        outcome_flip = len({x["outcome"] for x in valid}) > 1
        source = None
        if len(set(decisions)) > 1 or outcome_flip:
            source = "retrieval/evidence varied" if len(ev) > 1 else (
                "generation (LLM) varied with identical evidence")
        per_case[case.case_id] = {
            "expected": case.expected_decision, "attempted": len(rows), "valid": len(valid),
            "missing_due_to_provider": sum(1 for x in rows if x["outcome"] == "provider_error"),
            "missing_due_to_system": sum(1 for x in rows if x["outcome"] == "system_error"),
            "outcomes_all_attempts": [x["outcome"] for x in rows],
            "decision_stable": len(set(decisions)) <= 1, "decisions": decisions,
            "answerability_stable": len(set(answered)) <= 1,
            "safety_stable": len(set(safety)) <= 1,
            "outcome_stable": not outcome_flip,
            "distinct_evidence_sets": len(ev), "distinct_answers": len(ans), "variance_source": source,
        }

    # ---- B. pipeline / provider stability ---------------------------------
    prov_rows = [x for x in all_rows if x["outcome"] == "provider_error"]
    sys_rows = [x for x in all_rows if x["outcome"] == "system_error"]
    pstats = [x.get("provider", {}) for x in all_rows]
    provider = {
        "attempted_case_runs": len(all_rows),
        "successful_evaluation_case_count": len(all_rows) - len(prov_rows) - len(sys_rows),
        "provider_error_case_count": len(prov_rows),
        "provider_error_rate": round(len(prov_rows) / len(all_rows), 4) if all_rows else None,
        "system_error_case_count": len(sys_rows),
        "llm_calls": sum(p.get("llm_calls", 0) for p in pstats),
        "http_attempts": sum(p.get("http_attempts", 0) for p in pstats),
        "retry_count": sum(p.get("retries", 0) for p in pstats),
        "retry_wait_seconds": round(sum(p.get("retry_wait_seconds", 0.0) for p in pstats), 2),
        "case_runs_with_any_retry": sum(1 for p in pstats if p.get("retries")),
        "case_runs_needing_corrective_llm_calls": sum(1 for p in pstats if p.get("llm_calls", 0) > 1),
        "provider_failure_latency_s": _stats([x["provider_failure_latency_s"] for x in prov_rows
                                              if "provider_failure_latency_s" in x]),
        "excluded": [{"case_id": x["case_id"], "reason": x["outcome"], "detail": x.get("error", "")[:160]}
                     for x in prov_rows + sys_rows],
        "requests_seen_by_provider": provider_request_report(t_start, t_end),
    }

    print("\nA. FINAL OUTCOME STABILITY (valid runs only; provider/system errors are NOT outcomes)")
    print(f"{'metric':<24}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}")
    for k, s in across.items():
        if s.get("n"):
            print(f"{k:<24}{s['mean']:>8.3f}{s['std']:>8.3f}{s['min']:>8.3f}{s['max']:>8.3f}")
    for label, key in (("decision", "decision_stable"), ("answerability", "answerability_stable"),
                       ("safety", "safety_stable"), ("outcome", "outcome_stable")):
        n = sum(1 for v in per_case.values() if v[key])
        print(f"  {label} stable across valid runs: {n}/{len(cases)} cases")
    for c, v in per_case.items():
        if not (v["decision_stable"] and v["outcome_stable"]):
            print(f"  varied: {c:<26} {v['decisions']} -> {v['variance_source']}")
    print(f"  latency per valid case (s): {_stats(all_lat)}")

    print("\nB. PIPELINE / PROVIDER STABILITY")
    for k in ("attempted_case_runs", "successful_evaluation_case_count", "provider_error_case_count",
              "provider_error_rate", "system_error_case_count", "llm_calls", "http_attempts", "retry_count",
              "retry_wait_seconds", "case_runs_with_any_retry", "case_runs_needing_corrective_llm_calls"):
        print(f"  {k:<40} {provider[k]}")
    print(f"  provider_failure_latency_s                {provider['provider_failure_latency_s']}")
    for k, v in provider["requests_seen_by_provider"].items():
        print(f"  provider: {k:<32} {v}")
    for e in provider["excluded"]:
        print(f"  EXCLUDED {e['case_id']:<26} {e['reason']}: {e['detail']}")
    missing = {c: v["missing_due_to_provider"] for c, v in per_case.items() if v["missing_due_to_provider"]}
    if missing:
        print(f"  provider-induced missing outcomes per case: {missing}")

    import platform
    out = Path(args.out or f"eval_output/stability_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "benchmark_id": out.stem, "dataset": ds.dataset_id, "dataset_hash": dataset_hash(cases),
        "splits": sorted({c.split.value for c in cases}), "runs": args.runs,
        "kb": snap.describe(), "run_identity": runtime.identity(),
        "cache_policy": "none: in-process runs",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "determinism_probe": probe, "per_run": per_run, "across_runs": across,
        "latency_s": _stats(all_lat), "per_case": per_case, "provider_pipeline": provider,
        "pace_seconds": args.pace_seconds,
        "raw_runs": runs,
    }, indent=2, default=str), encoding="utf-8")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
