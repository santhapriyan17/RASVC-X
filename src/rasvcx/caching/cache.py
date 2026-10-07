"""Bounded answer cache for the /query route.

The pipeline is deterministic for a fixed knowledge-base version
(temperature 0, fixed indexes, fixed models), so an identical question
with an identical clinical context against the same version can be served
from memory instead of re-running retrieval, reranking, validation, the
LLM call and verification.

Safety properties:
  - The knowledge-base version is part of the key.  Publishing a new
    version changes the key, so an answer is never served from evidence
    that is no longer the active corpus.
  - Entries expire after ttl_seconds.
  - Only results the caller marks cacheable are stored (the route never
    stores a response in which a component failed), so a transient
    provider outage is not replayed as a cached abstention.
  - Bounded: least-recently-used entries are evicted beyond max_entries.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable


class ResponseCache:
    """Thread-safe LRU + TTL cache. max_entries == 0 disables it."""

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self._max = max(0, int(max_entries))
        self._ttl = float(ttl_seconds)
        self._lock = threading.Lock()
        self._entries: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        return self._max > 0

    def get(self, key: Hashable) -> Any | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or now - entry[0] > self._ttl:
                if entry is not None:
                    del self._entries[key]
                self._misses += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            return entry[1]

    def put(self, key: Hashable, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = (time.monotonic(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled, "entries": len(self._entries),
                "max_entries": self._max, "ttl_seconds": self._ttl,
                "hits": self._hits, "misses": self._misses,
            }


__all__ = ["ResponseCache"]
