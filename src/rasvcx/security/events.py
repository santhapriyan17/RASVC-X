"""Structured security events (one JSON object per log line).

Logger name: ``rasvcx.security``.  Route it to its own sink to audit
authentication failures, rate limiting, rejected prompt injections, SSRF
rejections and rejected uploads.

PII-safe by construction: an event never carries query text, answer text,
document content, tokens or raw client addresses.  Principals and client
addresses are reduced to a salted SHA-256 prefix (salt = process start),
enough to correlate events within one process run and not reversible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

logger = logging.getLogger("rasvcx.security")

_SALT = os.urandom(16)
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9 ._:/\-=,()]{0,200}$")


def pseudonym(value: str | None) -> str | None:
    """Non-reversible, per-process identifier for a token or address."""
    if not value:
        return None
    return hashlib.sha256(_SALT + value.encode("utf-8")).hexdigest()[:16]


def security_event(event: str, **fields: object) -> dict:
    """Emit one structured security event and return it (for tests).

    Field values must be short identifiers / codes; anything that does not
    match a conservative character set is replaced, so free text (a query,
    a URL path with user content) cannot leak into the security log.
    """
    record: dict[str, object] = {"ts": round(time.time(), 3), "event": event}
    for key, value in fields.items():
        if value is None or isinstance(value, (bool, int, float)):
            record[key] = value
        else:
            text = str(value)
            record[key] = text if _SAFE_VALUE.match(text) else "<redacted>"
    logger.warning(json.dumps(record, sort_keys=True))
    return record


_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]{1,128}$")


def safe_request_id(value: str | None) -> str | None:
    """A caller-supplied request id is accepted only if it is a plain token:
    anything else (newlines, control characters, very long values) could
    forge or corrupt log lines and is replaced."""
    if value and _REQUEST_ID_RE.match(value):
        return value
    return None


__all__ = ["pseudonym", "safe_request_id", "security_event"]
