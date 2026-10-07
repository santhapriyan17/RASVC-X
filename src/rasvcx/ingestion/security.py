"""M17 ingestion security — file validation + SSRF guard.

Callers pass (path, filename) positionally; `settings` is optional. Per-format
size limits live in the module-level `_FORMAT_LIMITS` dict (monkeypatchable).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

import filetype

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when a file or URL fails a security check."""


# ---------------------------------------------------------------------------
# Format tables
# ---------------------------------------------------------------------------

_MIME_TO_FORMAT: dict[str, str] = {
    "application/pdf": "pdf",
    "text/html": "html",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
}

_ZIP_FINGERPRINTS: list[tuple[str, str]] = [
    ("word/document.xml", "docx"),
    ("xl/workbook.xml", "xlsx"),
    ("ppt/presentation.xml", "pptx"),
]

_TEXT_EXTENSIONS: dict[str, str] = {
    ".txt": "txt",
    ".md": "md",
    ".markdown": "md",
    ".csv": "csv",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
}

_MB = 1024 * 1024
_DEFAULT_LIMIT = 500 * _MB

# Per-format byte limits. Monkeypatchable by tests.
#
# The limits are honest about how each parser uses memory (parsers.py):
#   streamed / page-at-a-time  -> pdf, txt, md, csv, html, xlsx(read_only):
#       bounded by the per-chunk work, so the limit is the upload cap.
#   whole document in memory   -> docx, pptx (python-docx / python-pptx
#       build the full object tree), json, xml (kept as single documents):
#       the limit bounds peak RAM of the isolated parser process.
_FORMAT_LIMITS: dict[str, int] = {
    "pdf": _DEFAULT_LIMIT, "html": _DEFAULT_LIMIT, "txt": _DEFAULT_LIMIT,
    "md": _DEFAULT_LIMIT, "csv": _DEFAULT_LIMIT, "xlsx": 200 * _MB,
    "docx": 100 * _MB, "pptx": 50 * _MB,
    "json": 100 * _MB, "xml": 100 * _MB,
    "zip": 200 * _MB,
}

# Archive safety limits (ZIP containers: docx / xlsx / pptx / zip).
MAX_ARCHIVE_MEMBERS = 5_000
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 1024 * _MB      # total declared expansion
MAX_ARCHIVE_COMPRESSION_RATIO = 200              # per member, uncompressed/compressed
_RATIO_CHECK_MIN_BYTES = 1 * _MB                 # tiny members can be highly compressible

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    list(_MIME_TO_FORMAT.values())
    + [fmt for _, fmt in _ZIP_FINGERPRINTS]
    + list(_TEXT_EXTENSIONS.values())
)


def _default_settings() -> Any:
    return SimpleNamespace(allowed_url_schemes=frozenset({"https"}))


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FileValidationResult:
    detected_format: str
    detected_mime: str | None
    file_size_bytes: int
    file_path: Path
    content_hash: str | None = None


def validate_file(
    file_path: Path | str,
    filename_hint: str | None = None,
    settings: Any | None = None,
    *,
    content_hash: str | None = None,
) -> FileValidationResult:
    """Validate an uploaded file before parsing. Raises SecurityError on failure."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if filename_hint is None:
        filename_hint = path.name

    file_size = path.stat().st_size
    if file_size == 0:
        raise SecurityError("File is empty (0 bytes)")

    fmt, mime = _detect_format(path, filename_hint)

    limit = _FORMAT_LIMITS.get(fmt, _DEFAULT_LIMIT)
    if file_size > limit:
        raise SecurityError(
            f"File size {file_size} bytes exceeds the {fmt.upper()} "
            f"limit of {limit} bytes"
        )

    if fmt == "zip":
        # A bare archive has no single document to extract.  It is refused
        # rather than accepted and then failing (or being unpacked) later.
        raise SecurityError(
            "Plain ZIP archives are not ingested. Upload the documents "
            "individually (pdf, docx, xlsx, pptx, html, txt, md, csv, json, xml)."
        )
    if fmt in ("docx", "xlsx", "pptx"):
        _check_archive_safety(path)

    logger.debug("File validation OK: format=%s size=%d mime=%s", fmt, file_size, mime)
    return FileValidationResult(
        detected_format=fmt,
        detected_mime=mime,
        file_size_bytes=file_size,
        file_path=path,
        content_hash=content_hash,
    )


def _detect_format(path: Path, filename_hint: str) -> tuple[str, str | None]:
    try:
        with path.open("rb") as fh:
            header = fh.read(2048)
    except OSError as exc:
        raise SecurityError(f"Cannot read file for type detection: {exc}") from exc

    kind = filetype.guess(header)

    if kind is not None:
        mime = kind.mime
        if mime in _MIME_TO_FORMAT:
            return _MIME_TO_FORMAT[mime], mime
        if mime == "application/zip":
            return _detect_zip_format(path, filename_hint), mime
        raise SecurityError(
            f"Unsupported file type detected: {mime}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # No magic bytes — text format by extension
    ext = Path(filename_hint).suffix.lower()
    if ext in _TEXT_EXTENSIONS:
        _assert_text_content(header, ext)
        return _TEXT_EXTENSIONS[ext], None

    raise SecurityError(
        "Cannot determine file type from content. "
        "Supported text: .txt .md .markdown .csv .json .xml .html; "
        "binary: .pdf .docx .xlsx .pptx .zip"
    )


def _detect_zip_format(path: Path, filename_hint: str) -> str:
    """A ZIP may be an Office doc or a plain archive; all are accepted."""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        raise SecurityError("File appears to be a corrupt or invalid ZIP archive")
    except Exception as exc:
        raise SecurityError(f"Cannot inspect ZIP file contents: {exc}") from exc

    for internal_path, fmt in _ZIP_FINGERPRINTS:
        if internal_path in names:
            return fmt
    if "[Content_Types].xml" in names:
        ext = Path(filename_hint).suffix.lower()
        return {".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx"}.get(ext, "docx")
    return "zip"


def _check_archive_safety(path: Path) -> None:
    """Reject decompression bombs and path-traversal entries.

    Works from the central directory only -- nothing is extracted.  The
    declared sizes can be forged, so the parsers additionally run in an
    isolated subprocess with a hard timeout; this check stops the cheap,
    common attacks before any parser touches the file.
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile:
        raise SecurityError("File appears to be a corrupt or invalid ZIP archive")
    except Exception as exc:
        raise SecurityError(f"Cannot inspect ZIP file contents: {exc}") from exc

    if len(infos) > MAX_ARCHIVE_MEMBERS:
        raise SecurityError(
            f"Archive has {len(infos)} entries (limit {MAX_ARCHIVE_MEMBERS})"
        )
    total = 0
    for info in infos:
        name = info.filename
        normalised = name.replace("\\", "/")
        if (
            normalised.startswith("/")
            or (len(normalised) > 1 and normalised[1] == ":")
            or ".." in normalised.split("/")
        ):
            raise SecurityError(f"Archive entry has an unsafe path: {name!r}")
        total += info.file_size
        if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise SecurityError(
                "Archive expands to more than "
                f"{MAX_ARCHIVE_UNCOMPRESSED_BYTES // _MB} MB (decompression bomb guard)"
            )
        if (
            info.file_size >= _RATIO_CHECK_MIN_BYTES
            and info.file_size > MAX_ARCHIVE_COMPRESSION_RATIO * max(info.compress_size, 1)
        ):
            raise SecurityError(
                f"Archive entry {name!r} has a compression ratio above "
                f"{MAX_ARCHIVE_COMPRESSION_RATIO}:1 (decompression bomb guard)"
            )


def _assert_text_content(header: bytes, ext: str) -> None:
    if not header:
        return
    sample = header[:512]
    suspicious = sum(1 for b in sample if b == 0 or (b < 32 and b not in (9, 10, 13)))
    ratio = suspicious / len(sample)
    if ratio > 0.10:
        raise SecurityError(
            f"File with extension '{ext}' appears to contain binary content "
            f"({ratio:.0%} suspicious bytes). Only plain text is accepted."
        )


# ---------------------------------------------------------------------------
# URL / SSRF validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UrlValidationResult:
    url: str
    hostname: str
    resolved_ips: list[str]


def validate_url(url: str, settings: Any | None = None) -> UrlValidationResult:
    """Validate an external URL before any network connection (HTTPS only)."""
    settings = settings or _default_settings()
    allowed = getattr(settings, "allowed_url_schemes", frozenset({"https"}))

    if any(ch.isspace() or ord(ch) < 32 for ch in url):
        raise SecurityError("URL contains whitespace or control characters")
    try:
        parsed = urlparse(url)
        port = parsed.port
    except Exception as exc:
        raise SecurityError(f"Malformed URL: {exc}") from exc
    if parsed.username or parsed.password:
        raise SecurityError("URLs with embedded credentials are not allowed")
    if port is not None and port not in (80, 443, 8443):
        raise SecurityError(f"URL port {port} is not allowed")

    scheme = (parsed.scheme or "").lower()
    if scheme not in allowed:
        raise SecurityError(
            f"URL scheme '{scheme}' is not allowed. Only HTTPS URLs are permitted."
        )

    hostname = parsed.hostname
    if not hostname:
        raise SecurityError("URL has no hostname. Only HTTPS URLs are permitted.")

    # Reject raw IP literals outright
    try:
        raw_ip = ipaddress.ip_address(hostname)
        if _is_private_or_reserved(raw_ip):
            raise SecurityError(f"URL hostname is a private/reserved IP: {hostname}")
        raise SecurityError(
            f"URL hostname is a raw IP address ({hostname}). "
            "Use a fully qualified hostname instead."
        )
    except ValueError:
        pass  # not an IP literal — resolve via DNS

    resolved = _resolve_hostname(hostname)
    if not resolved:
        raise SecurityError(f"DNS resolution of '{hostname}' returned no addresses")

    for ip_str in resolved:
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            raise SecurityError(f"DNS of '{hostname}' returned invalid IP: {ip_str}")
        if _is_private_or_reserved(addr):
            raise SecurityError(
                f"URL '{url}' resolves to a private/reserved IP ({ip_str}). "
                "SSRF protection rejects internal targets."
            )

    return UrlValidationResult(url=url, hostname=hostname, resolved_ips=resolved)


def _resolve_hostname(hostname: str) -> list[str]:
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SecurityError(f"DNS resolution failed for '{hostname}': {exc}") from exc
    except Exception as exc:
        raise SecurityError(f"Unexpected DNS error for '{hostname}': {exc}") from exc

    ips: list[str] = []
    seen: set[str] = set()
    for _family, _type, _proto, _canon, sockaddr in results:
        ip_str = sockaddr[0]
        if ip_str not in seen:
            seen.add(ip_str)
            ips.append(ip_str)
    return ips


def _is_private_or_reserved(addr: Any) -> bool:
    # IPv4-mapped IPv6 (::ffff:10.0.0.1) must be judged as the IPv4 address.
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    # is_global is False for private, loopback, link-local (incl. the cloud
    # metadata address 169.254.169.254), CGNAT, documentation and reserved
    # ranges; the explicit checks keep the intent readable.
    return (
        addr.is_private or addr.is_loopback or addr.is_link_local
        or addr.is_multicast or addr.is_reserved or addr.is_unspecified
        or not addr.is_global
    )


__all__ = [
    "SecurityError", "FileValidationResult", "UrlValidationResult",
    "SUPPORTED_FORMATS", "validate_file", "validate_url",
]