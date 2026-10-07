"""In-process latency observability for the query path.

LatencyRecorder keeps a bounded window of the most recent per-stage and
end-to-end latencies measured by the orchestrator's execution trace, and
reports p50/p95/p99 over that window.

Every number it reports is a measurement of a request this process
actually served.  With too few samples a percentile is reported as null
rather than extrapolated; nothing here is a target or an estimate.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from typing import Any

# Below this many samples a tail percentile is not meaningful.
_MIN_SAMPLES_P95 = 20
_MIN_SAMPLES_P99 = 100


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an ascending list (q in (0, 100])."""
    if not sorted_values:
        raise ValueError("percentile of empty list")
    rank = max(1, math.ceil(q / 100.0 * len(sorted_values)))
    return sorted_values[rank - 1]


def summarise(values: list[float]) -> dict[str, Any]:
    """count/mean/p50/p95/p99 in ms; tail percentiles null when under-sampled."""
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None,
                "min_ms": None, "max_ms": None}
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "mean_ms": round(sum(ordered) / n, 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "p50_ms": round(percentile(ordered, 50), 3),
        "p95_ms": round(percentile(ordered, 95), 3) if n >= _MIN_SAMPLES_P95 else None,
        "p99_ms": round(percentile(ordered, 99), 3) if n >= _MIN_SAMPLES_P99 else None,
    }


def process_stats() -> dict[str, Any]:
    """CPU seconds and resident memory of THIS process (no extra dependency).

    cpu_seconds: user+system CPU time consumed so far (time.process_time);
    a load benchmark divides its delta by wall time to get average cores.
    rss_mb: resident set size; None where it cannot be read.
    """
    import os
    import sys
    import time

    rss_mb = None
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            pmc = _PMC()
            pmc.cb = ctypes.sizeof(_PMC)
            # Explicit types: the pseudo-handle is 64-bit (-1); the ctypes
            # default (32-bit int) truncates it and the call fails.
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            gpmi = ctypes.windll.psapi.GetProcessMemoryInfo
            gpmi.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
            gpmi.restype = wintypes.BOOL
            if gpmi(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                rss_mb = round(pmc.WorkingSetSize / 1048576, 1)
        else:
            with open("/proc/self/statm") as fh:
                rss_mb = round(int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576, 1)
    except Exception:  # noqa: BLE001 - observability must never fail a request
        rss_mb = None
    return {"pid": os.getpid(), "cpu_seconds": round(time.process_time(), 3), "rss_mb": rss_mb,
            "cpu_count": os.cpu_count()}


class LatencyRecorder:
    """Thread-safe sliding window of measured stage latencies."""

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._lock = threading.Lock()
        self._stages: dict[str, deque[float]] = {}
        self._requests = 0

    def record(self, stage_ms: dict[str, float], total_ms: float, serialization_ms: float) -> None:
        with self._lock:
            self._requests += 1
            samples = dict(stage_ms)
            samples["serialization"] = serialization_ms
            samples["total"] = total_ms
            for stage, ms in samples.items():
                self._stages.setdefault(stage, deque(maxlen=self._window)).append(ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            stages = {name: list(values) for name, values in self._stages.items()}
            requests = self._requests
        return {
            "requests_recorded": requests,
            "window": self._window,
            "min_samples": {"p95": _MIN_SAMPLES_P95, "p99": _MIN_SAMPLES_P99},
            "stages": {name: summarise(values) for name, values in stages.items()},
        }


__all__ = ["LatencyRecorder", "percentile", "summarise"]
