"""Executed (not inspected) ingestion runtime behaviour.

Conditional GET (ETag / Last-Modified), parser hard-timeout kill, job
cancellation, atomic publication under failure, KB configuration warnings,
strict published-KB mode, and retention GC dry run.

No test here needs the network, a GPU, Qdrant, or an API key.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rasvcx.api.main import create_app
from tests.test_integration_repair import _offline_settings, _wait_for_job

DOC = b"Protocol RTMARKER requires a pharmacist review.\n\nProtocol RTMARKER requires a follow-up call.\n"


def _fetch(monkeypatch, tmp_path, handler, **kw):
    import httpx
    from rasvcx.ingestion import security, source_sync

    monkeypatch.setattr(security, "_resolve_hostname", lambda h: ["93.184.216.34"])
    real = httpx.AsyncClient

    def factory(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    args = dict(source_id="s", source_url="https://docs.example.org/g.txt", known_etag=None,
                known_last_modified=None, known_content_hash=None, temp_dir=str(tmp_path))
    args.update(kw)
    return asyncio.run(source_sync.fetch_source(**args))


class TestConditionalFetch:
    def test_etag_and_last_modified_are_sent_and_304_skips_download(self, monkeypatch, tmp_path) -> None:
        import httpx
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(304)

        r = _fetch(monkeypatch, tmp_path, handler, known_etag='"v7"',
                   known_last_modified="Wed, 01 Oct 2026 00:00:00 GMT", known_content_hash="abc")
        assert seen["if-none-match"] == '"v7"'
        assert seen["if-modified-since"] == "Wed, 01 Oct 2026 00:00:00 GMT"
        assert r.changed is False and r.http_status == 304 and r.temp_path is None
        assert list(tmp_path.iterdir()) == []

    def test_unchanged_content_hash_is_not_reingested(self, monkeypatch, tmp_path) -> None:
        import hashlib
        import httpx

        body = b"guideline v1"
        r = _fetch(monkeypatch, tmp_path, lambda req: httpx.Response(200, content=body, headers={"etag": "e2"}),
                   known_content_hash=hashlib.sha256(body).hexdigest())
        assert r.changed is False and r.etag == "e2" and list(tmp_path.iterdir()) == []

    def test_changed_content_is_downloaded_with_new_validators(self, monkeypatch, tmp_path) -> None:
        import httpx

        r = _fetch(monkeypatch, tmp_path, lambda req: httpx.Response(
            200, content=b"guideline v2", headers={"etag": "e3", "last-modified": "Thu, 02 Oct 2026 00:00:00 GMT"}),
            known_content_hash="old")
        assert r.changed is True and r.etag == "e3" and r.last_modified.startswith("Thu")
        assert Path(r.temp_path).read_bytes() == b"guideline v2"


class TestParserTimeout:
    def test_parser_subprocess_is_killed_at_the_timeout(self, tmp_path) -> None:
        from types import SimpleNamespace
        from rasvcx.ingestion.parsers import run_parser_subprocess

        f = tmp_path / "doc.txt"
        f.write_text("line of text\n" * 1000)
        t0 = time.perf_counter()
        res = run_parser_subprocess(str(f), "doc.txt", SimpleNamespace(
            parser_timeout_seconds=0.01, max_pdf_pages=10, max_csv_rows=10, max_xlsx_rows=10))
        assert res.error and "killed" in res.error and "timeout" in res.error
        assert time.perf_counter() - t0 < 30


class TestCancellationAndAtomicity:
    def test_cancelled_job_publishes_nothing(self, tmp_path, monkeypatch) -> None:
        from rasvcx.ingestion import parsers

        real = parsers.run_parser_subprocess

        def slow(*a, **k):
            time.sleep(1.5)
            return real(*a, **k)

        monkeypatch.setattr(parsers, "run_parser_subprocess", slow)
        with TestClient(create_app(settings=_offline_settings(tmp_path))) as c:
            before = c.get("/ingest/status").json()["serving_version_id"]
            job_id = c.post("/ingest/upload", files={"file": ("p.txt", DOC, "text/plain")}).json()["job_id"]
            assert c.delete(f"/ingest/jobs/{job_id}").status_code == 204
            time.sleep(3)
            job = c.get(f"/ingest/jobs/{job_id}").json()
            assert job["status"] == "cancelled"
            assert c.get("/ingest/status").json()["serving_version_id"] == before
            kb = tmp_path / "kb"
            assert not kb.exists() or not [p for p in kb.iterdir() if p.is_dir()]

    def test_publication_failure_keeps_old_version_and_leaves_no_orphan(self, tmp_path) -> None:
        with TestClient(create_app(settings=_offline_settings(tmp_path))) as c:
            before = c.get("/ingest/status").json()
            loader = c.app.state.rasvcx_runtime.snapshot_loader

            def boom(av):
                raise RuntimeError("simulated integrity failure")

            loader.load = boom
            job_id = c.post("/ingest/upload", files={"file": ("p.txt", DOC, "text/plain")}).json()["job_id"]
            job = _wait_for_job(c, job_id)
            assert job["status"] == "failed" and "Publication failed" in job["error_message"]
            after = c.get("/ingest/status").json()
            assert after["serving_version_id"] == before["serving_version_id"]
            assert after["pointer_version_id"] == before["pointer_version_id"]
            kb = tmp_path / "kb"
            assert not kb.exists() or not [p for p in kb.iterdir() if p.is_dir()]
            d = c.post("/query", json={"query": "protocol RTMARKER", "enriched": True}).json()
            assert not any("RTMARKER" in e["text"] for e in d["evidence"])


class TestUploadSizeLimits:
    def test_document_larger_than_json_limit_is_accepted_and_indexed(self, tmp_path) -> None:
        """Regression: the 64 KB JSON body limit used to reject every upload
        over 64 KB with 413."""
        big = (b"Ward note paragraph about routine handover procedures and checklists.\n\n" * 4000
               + b"Protocol SIZEMARKER requires a second check.\n")
        assert len(big) > 250_000
        with TestClient(create_app(settings=_offline_settings(tmp_path))) as c:
            r = c.post("/ingest/upload", files={"file": ("big.txt", big, "text/plain")})
            assert r.status_code == 202, r.text
            job = _wait_for_job(c, r.json()["job_id"], timeout=120)
            assert job["status"] == "completed", job["error_message"]
            d = c.post("/query", json={"query": "protocol SIZEMARKER", "enriched": True}).json()
            assert any("SIZEMARKER" in e["text"] for e in d["evidence"])
            # JSON routes keep the small limit.
            assert c.post("/query", json={"query": "x" * 100_000}).status_code == 413

    def test_upload_over_ingestion_limit_is_rejected(self, tmp_path) -> None:
        from dataclasses import replace

        s = _offline_settings(tmp_path)
        s = replace(s, ingestion=replace(s.ingestion, max_upload_bytes=100_000))
        with TestClient(create_app(settings=s)) as c:
            r = c.post("/ingest/upload", files={"file": ("big.txt", b"a b c\n" * 400_000, "text/plain")})
            assert r.status_code == 413


class TestKBConfiguration:
    def test_stray_pointer_and_seed_fallback_are_reported(self, tmp_path) -> None:
        from dataclasses import replace

        s = _offline_settings(tmp_path)
        versions = tmp_path / "kb"
        versions.mkdir()
        (versions / "active_version.json").write_text(json.dumps({"version_id": "v_old"}))
        s = replace(s, ingestion=replace(s.ingestion, active_version_path=str(tmp_path / "elsewhere.json")))
        with TestClient(create_app(settings=s)) as c:
            ready = c.get("/ready").json()
            assert any("stray pointer" in w for w in ready["kb_warnings"])
            assert any("no published KB version" in w for w in ready["kb_warnings"])
            comp = {x["name"]: x for x in ready["components"]}
            assert comp["kb_consistency"]["state"] == "loaded"

    def test_strict_mode_refuses_seed_fallback(self, tmp_path) -> None:
        from dataclasses import replace
        from rasvcx.config.settings import ConfigurationError

        s = _offline_settings(tmp_path)
        s = replace(s, ingestion=replace(s.ingestion, require_published_kb=True))
        with pytest.raises(ConfigurationError, match="require_published_kb"):
            with TestClient(create_app(settings=s)):
                pass

    def test_gc_dry_run_lists_without_deleting(self, tmp_path) -> None:
        from dataclasses import replace
        import os

        s = _offline_settings(tmp_path)
        s = replace(s, ingestion=replace(s.ingestion, keep_versions=1, gc_delay_seconds=60.0))
        kb = tmp_path / "kb"
        old = time.time() - 3600
        for i in range(3):
            d = kb / f"v_00000000000{i}"
            d.mkdir(parents=True)
            os.utime(d, (old + i, old + i))
        with TestClient(create_app(settings=s)) as c:
            r = c.post("/admin/kb/gc").json()
            assert r["dry_run"] is True and len(r["eligible"]) == 2
            assert all((kb / v).is_dir() for v in r["eligible"])     # nothing deleted
            r2 = c.post("/admin/kb/gc?dry_run=false").json()
            assert sorted(r2["deleted"]) == sorted(r["eligible"])
            assert [p.name for p in kb.iterdir() if p.is_dir()] == ["v_000000000002"]
