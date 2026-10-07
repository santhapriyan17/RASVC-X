# src/rasvcx/ingestion/source_sync.py
# ---------------------------------------------------------------------------
# External source synchronization:
#   - Async HTTP fetch with ETag/Last-Modified conditional requests
#   - SHA-256 content-hash change detection
#   - SSRF guard enforced at CONNECTION time, on every redirect hop
#   - Streaming download to temp file with a hard byte limit
#   - SyncFetchResult, compute_next_sync_at, build_ingest_job_for_source
#
# SSRF model
# ----------
# Validating a hostname and then letting the HTTP client resolve it again
# is not a defence: the second lookup can return a different (internal)
# address (DNS rebinding).  Here each hop is resolved exactly once by
# security.validate_url(), every returned address is checked, and the
# connection is then made to one of those checked IP ADDRESSES.  The
# original hostname is sent only as the Host header and the TLS SNI /
# certificate name, so certificate verification still applies to the real
# host.  Redirects are never followed automatically: each Location is
# resolved against the current URL and goes through the same validation.
# ---------------------------------------------------------------------------

from __future__ import annotations

import hashlib
import ipaddress
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# Defaults used when no IngestionSettings is supplied.
_MAX_REDIRECTS = 5
_DEFAULT_SIZE_LIMIT = 512 * 1024 * 1024
_REDIRECT_CODES = (301, 302, 303, 307, 308)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SyncFetchResult:
    source_id: str
    source_url: str
    changed: bool                    # False = content unchanged (ETag/hash match)
    temp_path: str | None            # Path to downloaded file (None if unchanged)
    content_hash: str | None         # SHA-256 of downloaded content
    etag: str | None                 # ETag from response headers
    last_modified: str | None        # Last-Modified from response headers
    content_type: str | None
    size_bytes: int
    error: str | None = None
    http_status: int | None = None


def _failed(source_id: str, source_url: str, error: str, http_status: int | None = None) -> SyncFetchResult:
    return SyncFetchResult(
        source_id=source_id, source_url=source_url, changed=False,
        temp_path=None, content_hash=None, etag=None, last_modified=None,
        content_type=None, size_bytes=0, error=error, http_status=http_status,
    )


def pinned_request_target(url: str, ip: str) -> tuple[str, str, str]:
    """Rewrite `url` to connect to `ip` while still addressing its host.

    Returns (connect_url, host_header, sni_hostname).  connect_url has the
    validated IP as its authority, so the HTTP client performs no DNS
    lookup of its own.
    """
    parts = urlsplit(url)
    hostname = parts.hostname or ""
    literal = f"[{ip}]" if ipaddress.ip_address(ip).version == 6 else ip
    default_port = 443 if parts.scheme == "https" else 80
    port = parts.port or default_port
    netloc = literal if port == default_port else f"{literal}:{port}"
    host_header = hostname if port == default_port else f"{hostname}:{port}"
    return urlunsplit((parts.scheme, netloc, parts.path or "/", parts.query, "")), host_header, hostname


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

async def fetch_source(
    source_id: str,
    source_url: str,
    known_etag: str | None,
    known_last_modified: str | None,
    known_content_hash: str | None,
    temp_dir: str,
    size_limit: int | None = None,
    timeout_seconds: float | None = None,
    settings: Any | None = None,
) -> SyncFetchResult:
    """
    Fetch an external source URL asynchronously.

    - Uses conditional GET (If-None-Match / If-Modified-Since) when available.
    - Validates every hop and connects to the validated IP (see module doc).
    - Streams the response body to a temp file; the body is never held in
      memory and the download is aborted once it exceeds the size limit.
    - Computes SHA-256 hash; returns changed=False if hash matches known.
    - On 304 Not Modified returns changed=False without downloading.
    - On any error returns SyncFetchResult with error set.

    `settings` is the IngestionSettings of the running process (limits,
    timeouts, allowed schemes, max redirects).  Explicit size_limit /
    timeout_seconds arguments take precedence over it.
    """
    import httpx
    from .security import validate_url, SecurityError

    max_redirects = int(getattr(settings, "max_redirects", _MAX_REDIRECTS))
    size_limit = int(
        size_limit if size_limit is not None
        else getattr(settings, "max_source_response_bytes", _DEFAULT_SIZE_LIMIT)
    )
    connect_timeout = float(getattr(settings, "source_connect_timeout_seconds", 10.0))
    read_timeout = float(
        timeout_seconds if timeout_seconds is not None
        else getattr(settings, "source_read_timeout_seconds", 60.0)
    )

    headers: dict[str, str] = {
        "User-Agent": "RASVCX-Sync/1.0 (medical evidence research)",
        "Accept-Encoding": "identity",  # the byte limit applies to what is stored
    }
    if known_etag:
        headers["If-None-Match"] = known_etag
    if known_last_modified:
        headers["If-Modified-Since"] = known_last_modified

    current_url = source_url
    redirects = 0
    tmp_path: str | None = None

    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,  # never route through an ambient proxy
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=0),
        ) as client:
            while True:
                # Resolve + validate this hop, then connect to a validated IP.
                try:
                    validated = validate_url(current_url, settings)
                except SecurityError as exc:
                    where = "URL" if redirects == 0 else f"redirect to {current_url}"
                    return _failed(source_id, source_url, f"SSRF guard rejected {where}: {exc}")

                connect_url, host_header, sni = pinned_request_target(
                    current_url, validated.resolved_ips[0]
                )
                request = client.build_request(
                    "GET", connect_url,
                    headers={**headers, "Host": host_header},
                    extensions={"sni_hostname": sni},
                )
                response = await client.send(request, stream=True)
                try:
                    code = response.status_code

                    if code in _REDIRECT_CODES:
                        redirects += 1
                        if redirects > max_redirects:
                            return _failed(
                                source_id, source_url,
                                f"Too many redirects (>{max_redirects})", code,
                            )
                        location = response.headers.get("location", "")
                        if not location:
                            return _failed(
                                source_id, source_url, "Redirect with no Location header", code,
                            )
                        current_url = urljoin(current_url, location)
                        # Conditional headers belong to the original resource.
                        headers.pop("If-None-Match", None)
                        headers.pop("If-Modified-Since", None)
                        continue

                    if code == 304:
                        logger.debug("source %s: 304 Not Modified", source_id)
                        return SyncFetchResult(
                            source_id=source_id, source_url=source_url, changed=False,
                            temp_path=None, content_hash=known_content_hash,
                            etag=known_etag, last_modified=known_last_modified,
                            content_type=None, size_bytes=0, http_status=304,
                        )

                    if code != 200:
                        return _failed(source_id, source_url, f"HTTP {code}", code)

                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > size_limit:
                        return _failed(
                            source_id, source_url,
                            f"Response of {declared} bytes exceeds the "
                            f"{size_limit // 1024 // 1024} MB limit", code,
                        )

                    resp_etag = response.headers.get("etag")
                    resp_last_modified = response.headers.get("last-modified")
                    content_type = response.headers.get("content-type", "")

                    Path(temp_dir).mkdir(parents=True, exist_ok=True)
                    fd, tmp_path = tempfile.mkstemp(dir=temp_dir, suffix=".download")
                    hasher = hashlib.sha256()
                    total_bytes = 0
                    with os.fdopen(fd, "wb") as fout:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            total_bytes += len(chunk)
                            if total_bytes > size_limit:
                                raise ValueError(
                                    f"Response exceeded size limit "
                                    f"({size_limit // 1024 // 1024} MB)"
                                )
                            fout.write(chunk)
                            hasher.update(chunk)
                finally:
                    await response.aclose()
                break

        new_hash = hasher.hexdigest()

        # Hash unchanged — no reprocessing needed
        if known_content_hash and new_hash == known_content_hash:
            logger.debug("source %s: content hash unchanged", source_id)
            os.unlink(tmp_path)
            return SyncFetchResult(
                source_id=source_id, source_url=source_url, changed=False,
                temp_path=None, content_hash=new_hash,
                etag=resp_etag or known_etag,
                last_modified=resp_last_modified or known_last_modified,
                content_type=content_type, size_bytes=total_bytes, http_status=code,
            )

        logger.info(
            "source %s: content changed (hash %s -> %s), %d bytes",
            source_id, (known_content_hash or "none")[:12], new_hash[:12], total_bytes,
        )
        return SyncFetchResult(
            source_id=source_id, source_url=source_url, changed=True,
            temp_path=tmp_path, content_hash=new_hash, etag=resp_etag,
            last_modified=resp_last_modified, content_type=content_type,
            size_bytes=total_bytes, http_status=code,
        )

    except Exception as exc:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        logger.warning("fetch_source %s failed: %s: %s", source_id, type(exc).__name__, exc)
        return _failed(source_id, source_url, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Scheduling helper
# ---------------------------------------------------------------------------

def compute_next_sync_at(
    sync_interval_seconds: int,
    last_sync_at: str | None = None,
) -> str:
    """
    Compute the ISO-8601 timestamp for the next scheduled sync.
    If last_sync_at is None, schedule from now.
    """
    if last_sync_at:
        try:
            base = datetime.fromisoformat(last_sync_at)
            if base.tzinfo is None:
                base = base.replace(tzinfo=timezone.utc)
        except ValueError:
            base = datetime.now(timezone.utc)
    else:
        base = datetime.now(timezone.utc)

    return (base + timedelta(seconds=sync_interval_seconds)).isoformat()


# ---------------------------------------------------------------------------
# Job builder helper
# ---------------------------------------------------------------------------

def build_ingest_job_for_source(
    source_id: str,
    source_url: str,
    fetch_result: SyncFetchResult,
) -> dict[str, Any]:
    """
    Build the kwargs dict for job_store.create_job() from a SyncFetchResult.
    Returns an empty dict if the fetch was unchanged or errored.
    """
    if not fetch_result.changed or fetch_result.temp_path is None:
        return {}

    return {
        "source_id": source_id,
        "source_type": "scheduled_url",
        "source_url": source_url,
        "filename": Path(urlsplit(source_url).path).name or "document",
        "content_hash": fetch_result.content_hash,
        "temp_path": fetch_result.temp_path,
    }

class SyncError(Exception):
    """Raised when source synchronisation fails unrecoverably."""


# Public alias -- callers and tests import sync_source
sync_source = fetch_source
