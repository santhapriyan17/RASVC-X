"""Throughput benchmark for a RUNNING RASVC-X server.

    python benchmarks/throughput_bench.py --concurrency 1,2,4 --requests 16

For each concurrency level it sends --requests /query calls from that many
client threads and reports what was MEASURED:

  QPS                 completed UNCACHED pipeline executions / wall second
  latency p50/95/99   client wall latency of those executions (tail
                      percentiles n/a below 20 / 100 samples)
  error rates         SYSTEM_ERROR + non-200, and PROVIDER_ERROR, as
                      fractions of all requests sent
  503                 requests refused by bounded admission (api.max_workers)
  queue ms            server-reported wait for a worker after admission
  CPU                 server process CPU seconds / wall second (avg cores)
  RAM                 peak server RSS sampled every 0.5 s (MB)

Only COLD_UNCACHED / WARM_UNCACHED responses count as throughput.
CACHE_HIT, PROVIDER_ERROR and SYSTEM_ERROR are counted per level and never
included in QPS or latency.  The default cache policy sends
bypass_cache=true, so repeated queries still run the pipeline.

The server's /status endpoint supplies CPU and RSS; it must be reachable
(pass --token when api.require_auth is on).  Throughput is a property of
this deployment on this hardware: in research modes it is bounded by the
LLM provider and by CPU inference (reranker, NLI).  Every uncached request
in a research mode is a billed model call.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rasvcx.api.observability import summarise  # noqa: E402

QUERIES = [
    "What class of drug is azithromycin?",
    "What is ondansetron used to prevent?",
    "What is Veltrazine indicated for?",
    "What monitoring is needed before starting Veltrazine?",
    "How should Tanzivex vials be stored?",
    "When should blood cultures be obtained in the sepsis hour-1 bundle?",
    "What is the maximum daily dose of Lumetrol in the European Union?",
    "What crystalloid fluid bolus volume is recommended for sepsis with hypotension?",
]

_INFERENCE = ("COLD_UNCACHED", "WARM_UNCACHED")


def _status(base: str, headers: dict) -> dict:
    return httpx.get(f"{base}/status", headers=headers, timeout=30).json()


def run_level(base: str, concurrency: int, total: int, bypass_cache: bool = True,
              headers: dict | None = None) -> dict:
    headers = headers or {}
    lock = threading.Lock()
    next_index = [0]
    latencies: list[float] = []
    queue_ms: list[float] = []
    statuses: dict[str, int] = {}
    decisions: dict[str, int] = {}
    classes: dict[str, int] = {}
    peak_rss = [None]
    stop = threading.Event()

    def sampler() -> None:
        while not stop.is_set():
            try:
                rss = _status(base, headers)["process"]["rss_mb"]
                if rss is not None and (peak_rss[0] is None or rss > peak_rss[0]):
                    peak_rss[0] = rss
            except Exception:  # noqa: BLE001 - sampling is best effort
                pass
            stop.wait(0.5)

    def worker() -> None:
        with httpx.Client(base_url=base, timeout=900, headers=headers) as client:
            while True:
                with lock:
                    i = next_index[0]
                    if i >= total:
                        return
                    next_index[0] += 1
                t0 = time.perf_counter()
                cls = None
                body: dict = {}
                try:
                    r = client.post("/query", json={
                        "query": QUERIES[i % len(QUERIES)], "enriched": True,
                        "bypass_cache": bypass_cache,
                    })
                    key = str(r.status_code)
                    body = r.json() if r.status_code == 200 else {}
                    cls = body.get("request_class")
                except httpx.HTTPError as exc:
                    key = type(exc).__name__
                elapsed = (time.perf_counter() - t0) * 1000.0
                with lock:
                    statuses[key] = statuses.get(key, 0) + 1
                    if cls:
                        classes[cls] = classes.get(cls, 0) + 1
                    if key == "200" and cls in _INFERENCE:
                        latencies.append(elapsed)
                        queue_ms.append(float(body.get("queue_ms") or 0.0))
                        d = body.get("decision")
                        decisions[d] = decisions.get(d, 0) + 1

    before = _status(base, headers)["process"]
    samp = threading.Thread(target=sampler, daemon=True)
    samp.start()
    started = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started
    stop.set()
    samp.join(timeout=2)
    after = _status(base, headers)["process"]

    ok = len(latencies)
    system_errors = classes.get("SYSTEM_ERROR", 0) + sum(
        v for k, v in statuses.items() if k not in ("200", "503"))
    return {
        "concurrency": concurrency, "requests": total, "completed_uncached": ok,
        "request_classes": classes, "http_statuses": statuses,
        "rejected_503": statuses.get("503", 0),
        "cache_hit_ratio": round(classes.get("CACHE_HIT", 0) / total, 3) if total else None,
        "error_rate": round(system_errors / total, 3) if total else None,
        "provider_error_rate": round(classes.get("PROVIDER_ERROR", 0) / total, 3) if total else None,
        "wall_seconds": round(wall, 2),
        "qps": round(ok / wall, 4) if wall > 0 else None,
        "latency_ms": summarise(latencies),
        "queue_ms": summarise(queue_ms),
        "server_cpu_avg_cores": round((after["cpu_seconds"] - before["cpu_seconds"]) / wall, 2) if wall > 0 else None,
        "server_peak_rss_mb": peak_rss[0],
        "decisions": decisions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", default="1,2,4", help="comma-separated levels")
    parser.add_argument("--requests", type=int, default=16, help="requests per level")
    parser.add_argument("--cache-policy", choices=("bypass", "allow"), default="bypass")
    parser.add_argument("--token", default="", help="bearer token when api.require_auth is on")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}

    ready = httpx.get(f"{args.base}/ready", timeout=10).json()
    status = _status(args.base, headers)
    print(f"server: mode={ready['execution_mode']} offline={ready['offline']} kb={ready['kb_version_id']} "
          f"({ready.get('kb_source')}) workers={status['admission']['max_workers']} "
          f"cpus={status['process']['cpu_count']} cache_policy={args.cache_policy}")
    if ready["offline"]:
        print("NOTE: offline_test -- stub LLM, no reranker/NLI/Qdrant. This is pipeline overhead only.")

    def f(v: float | None, d: int = 0) -> str:
        return "n/a" if v is None else f"{v:.{d}f}"

    results = []
    print(f"\n{'conc':>4}{'done':>6}{'503':>5}{'err%':>6}{'prov%':>6}{'QPS':>8}{'p50ms':>8}{'p95ms':>8}"
          f"{'p99ms':>8}{'queue p50':>10}{'cores':>7}{'RSS MB':>8}  classes")
    for level in levels:
        res = run_level(args.base, level, args.requests, args.cache_policy == "bypass", headers)
        results.append(res)
        lat, q = res["latency_ms"], res["queue_ms"]
        print(f"{level:>4}{res['completed_uncached']:>6}{res['rejected_503']:>5}"
              f"{f(res['error_rate'] * 100 if res['error_rate'] is not None else None):>6}"
              f"{f(res['provider_error_rate'] * 100 if res['provider_error_rate'] is not None else None):>6}"
              f"{f(res['qps'], 3):>8}{f(lat['p50_ms']):>8}{f(lat['p95_ms']):>8}{f(lat['p99_ms']):>8}"
              f"{f(q['p50_ms'], 1):>10}{f(res['server_cpu_avg_cores'], 2):>7}{f(res['server_peak_rss_mb']):>8}"
              f"  {res['request_classes']}")
    print("p95 / p99 are n/a below 20 / 100 completed requests at a level.")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "server": ready, "status_at_start": {k: status[k] for k in ("process", "admission", "mode")},
            "cache_policy": args.cache_policy, "levels": results}, indent=2), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
