"""Latency benchmark for a RUNNING RASVC-X server.

Sends real /query requests (enriched) and reports, per pipeline stage and
end to end, the latencies the server measured for those requests.

    python benchmarks/latency_bench.py --base http://127.0.0.1:8000 --rounds 2
    python benchmarks/latency_bench.py --cache-policy allow   # also measure cache hits

Every request is classified from the server's own `request_class`:

    COLD_UNCACHED   first pipeline execution of the server process
    WARM_UNCACHED   a later pipeline execution
    CACHE_HIT       answered from the answer cache (no pipeline ran)
    PROVIDER_ERROR  the LLM provider failed
    SYSTEM_ERROR    another stage failed (or a non-200 HTTP status)

Inference latency is reported for COLD_UNCACHED and WARM_UNCACHED only,
each in its own table.  Cache hits are reported separately and never enter
the inference statistics; errors are counted, not timed.  By default the
benchmark sends bypass_cache=true so every request runs the pipeline.

Every number printed is a measurement from this run.  Tail percentiles are
printed as "n/a" when the run has too few samples for them to mean
anything (p95 needs 20 samples, p99 needs 100) -- they are never
extrapolated.  The benchmark is sequential (one request at a time), so it
measures latency, not throughput under load.

Each uncached request reaches the configured LLM provider, so in research
modes a run costs one generation per answered query.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rasvcx.api.observability import summarise  # noqa: E402

DEFAULT_QUERIES = [
    "What infections is azithromycin used to treat?",
    "What are the contraindications of sertraline?",
    "What adverse reactions are reported for olanzapine?",
    "How should lamotrigine be discontinued?",
    "What is ondansetron indicated for?",
    "What warnings apply to ketorolac tromethamine?",
    "What is the mechanism of action of tamsulosin?",
    "Which drug interactions are listed for rivastigmine?",
    "What does the sepsis bundle say about blood cultures?",
    "What are the storage conditions for famotidine?",
]

STAGE_ORDER = [
    "input_validation", "risk_routing", "hybrid_retrieval", "bm25_retrieval", "dense_retrieval",
    "rrf_fusion", "reranking", "sufficiency_gate", "targeted_retrieval", "provenance_context",
    "atomic_claim_extraction", "verified_context", "candidate_generation",
    "deterministic_validation", "contextual_validation", "selective_nli",
    "evidence_resolution", "nli_inference", "generation", "provider_backoff",
    "post_generation_verification", "confidence_estimation", "calibration", "decision",
    "corrective_passes", "serialization",
]

INFERENCE_CLASSES = ("COLD_UNCACHED", "WARM_UNCACHED")
ALL_CLASSES = INFERENCE_CLASSES + ("CACHE_HIT", "PROVIDER_ERROR", "SYSTEM_ERROR")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:10.1f}"


_HEADER = f"{'stage':<32}{'n':>5}{'mean ms':>11}{'median':>11}{'p95':>11}{'min':>11}{'max':>11}"

#: Measured inside another stage; excluded when ranking contributors.
_NESTED = {"nli_inference", "provider_backoff", "bm25_retrieval", "dense_retrieval", "rrf_fusion",
           "candidate_generation", "deterministic_validation", "contextual_validation",
           "selective_nli", "evidence_resolution", "semantic_verification", "corrective_passes"}


def _row(name: str, summary: dict) -> str:
    return (f"{name:<32}{summary['count']:>5}{_fmt(summary['mean_ms']):>11}"
            f"{_fmt(summary['p50_ms']):>11}{_fmt(summary['p95_ms']):>11}"
            f"{_fmt(summary.get('min_ms')):>11}{_fmt(summary.get('max_ms')):>11}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--rounds", type=int, default=2, help="passes over the query list")
    parser.add_argument("--queries", help="text file with one query per line")
    parser.add_argument("--cache-policy", choices=("bypass", "allow"), default="bypass",
                        help="bypass (default): every request runs the pipeline; "
                             "allow: repeated queries may be cache hits (reported separately)")
    parser.add_argument("--out", help="write the full result as JSON to this path")
    parser.add_argument("--pace-seconds", type=float, default=0.0,
                        help="pause between requests so a rate-limited provider key is not driven "
                             "into quota waits (pause is not part of any measured latency)")
    args = parser.parse_args()

    queries = DEFAULT_QUERIES
    if args.queries:
        queries = [ln.strip() for ln in Path(args.queries).read_text(encoding="utf-8").splitlines() if ln.strip()]
    bypass = args.cache_policy == "bypass"

    # per class: stage -> samples, totals, wall
    stages: dict[str, dict[str, list[float]]] = {c: {} for c in INFERENCE_CLASSES}
    totals: dict[str, list[float]] = {c: [] for c in ALL_CLASSES}
    wall: dict[str, list[float]] = {c: [] for c in ALL_CLASSES}
    decisions: dict[str, int] = {}
    mode: dict | None = None
    kb: dict | None = None

    with httpx.Client(base_url=args.base, timeout=600) as client:
        ready = client.get("/ready").json()
        print(f"server: mode={ready['execution_mode']} offline={ready['offline']} ready={ready['ready']} "
              f"kb={ready['kb_version_id']} source={ready.get('kb_source')} "
              f"docs={ready.get('kb_doc_count')} chunks={ready.get('kb_chunk_count')}")
        print(f"cache policy: {args.cache_policy}")
        started = time.perf_counter()
        for round_no in range(args.rounds):
            for query in queries:
                if args.pace_seconds:
                    time.sleep(args.pace_seconds)
                t0 = time.perf_counter()
                response = client.post(
                    "/query", json={"query": query, "enriched": True, "bypass_cache": bypass},
                )
                elapsed = (time.perf_counter() - t0) * 1000.0
                if response.status_code != 200:
                    wall["SYSTEM_ERROR"].append(elapsed)
                    print(f"  r{round_no + 1} HTTP {response.status_code:<15} {elapsed:9.0f} ms  {query[:60]}")
                    continue
                body = response.json()
                cls = body.get("request_class") or ("CACHE_HIT" if body.get("cached") else "WARM_UNCACHED")
                mode, kb = body.get("mode"), body.get("kb")
                wall[cls].append(elapsed)
                totals[cls].append(body["total_latency_ms"])
                if cls in INFERENCE_CLASSES:
                    decisions[body["decision"]] = decisions.get(body["decision"], 0) + 1
                    for stage, ms in body["stage_latencies_ms"].items():
                        stages[cls].setdefault(stage, []).append(ms)
                print(f"  r{round_no + 1} {cls:<15} {body['decision']:<20} {elapsed:9.0f} ms  {query[:60]}")
        duration = time.perf_counter() - started

    counts = {c: len(wall[c]) for c in ALL_CLASSES}
    print(f"\nrequests in {duration:.1f}s (sequential): {counts}")
    if mode:
        print(f"model: {'stub (offline)' if mode['mock_llm'] else mode['llm_model']} | "
              f"retrieval={mode['retrieval_mode']} reranker={mode['reranker']} nli={mode['nli']}")
    print(f"decisions (pipeline executions only): {decisions}")

    report: dict[str, dict] = {}
    for cls in INFERENCE_CLASSES:
        if not totals[cls]:
            continue
        print(f"\n[{cls}] inference latency (ms; p95 n/a below 20 samples)")
        print(_HEADER)
        per = stages[cls]
        ordered = [s for s in STAGE_ORDER if s in per] + sorted(set(per) - set(STAGE_ORDER))
        cls_report: dict[str, dict] = {}
        for name, values in [(s, per[s]) for s in ordered] + [
            ("TOTAL (server)", totals[cls]), ("TOTAL (client wall)", wall[cls]),
        ]:
            cls_report[name] = summarise(values)
            print(_row(name, cls_report[name]))
        # Contribution = stage time summed over all requests / total time.
        grand = sum(totals[cls]) or 1.0
        top = sorted(((sum(v), s) for s, v in per.items() if s not in _NESTED and s != "serialization"),
                     reverse=True)[:3]
        print("top contributors: " + "; ".join(
            f"{s} {100 * t / grand:.0f}% of total time" for t, s in top))
        cls_report["_top_contributors"] = [{"stage": s, "share": round(t / grand, 3)} for t, s in top]
        report[cls] = cls_report
    if wall["CACHE_HIT"]:
        print("\n[CACHE_HIT] lookup latency (NOT inference; no pipeline stage ran)")
        report["CACHE_HIT"] = {"TOTAL (server)": summarise(totals["CACHE_HIT"]),
                               "TOTAL (client wall)": summarise(wall["CACHE_HIT"])}
        for name, s in report["CACHE_HIT"].items():
            print(_row(name, s))
    for cls in ("PROVIDER_ERROR", "SYSTEM_ERROR"):
        if counts[cls]:
            print(f"\n{cls}: {counts[cls]} request(s) -- excluded from pipeline latency; "
                  f"their own latency (client wall):")
            print(_HEADER)
            report[cls] = {"TOTAL (client wall)": summarise(wall[cls])}
            print(_row("TOTAL (client wall)", report[cls]["TOTAL (client wall)"]))
    print("\nA stage has fewer samples than requests when it did not run for every request")
    print("(for example generation is skipped when the sufficiency gate abstains).")

    if not any(totals[c] for c in INFERENCE_CLASSES):
        return 1
    if args.out:
        Path(args.out).write_text(json.dumps({
            "base": args.base, "cache_policy": args.cache_policy, "request_classes": counts,
            "duration_seconds": round(duration, 3), "mode": mode, "kb": kb,
            "decisions": decisions, "latency_by_class": report,
        }, indent=2), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
