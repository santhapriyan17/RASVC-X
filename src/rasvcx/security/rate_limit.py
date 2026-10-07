"""In-process per-principal rate limiting (token bucket).

Single-process boundary: buckets live in this process's memory, which is
correct for the supported deployment (one Uvicorn worker, see
rasvcx/__main__.py).  Several processes behind a load balancer would each
enforce the limit separately; a shared store would be required there and
is NOT implemented.
"""

from __future__ import annotations

import threading
import time


class TokenBucketLimiter:
    """`rate_per_minute` requests per principal, bursts up to the same size."""

    def __init__(self, rate_per_minute: int, max_principals: int = 10_000) -> None:
        self._capacity = float(rate_per_minute)
        self._refill_per_s = rate_per_minute / 60.0
        self._max = max_principals
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, float]] = {}  # principal -> (tokens, t)

    @property
    def enabled(self) -> bool:
        return self._capacity > 0

    def acquire(self, principal: str) -> float:
        """0.0 if allowed; otherwise seconds until a request is allowed."""
        if not self.enabled:
            return 0.0
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(principal, (self._capacity, now))
            tokens = min(self._capacity, tokens + (now - last) * self._refill_per_s)
            if tokens >= 1.0:
                self._buckets[principal] = (tokens - 1.0, now)
                allowed = 0.0
            else:
                self._buckets[principal] = (tokens, now)
                allowed = (1.0 - tokens) / self._refill_per_s
            if len(self._buckets) > self._max:  # bound memory: drop the stalest
                oldest = min(self._buckets, key=lambda k: self._buckets[k][1])
                del self._buckets[oldest]
            return allowed


__all__ = ["TokenBucketLimiter"]
