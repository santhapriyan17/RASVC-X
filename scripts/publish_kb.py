"""Publish a seed corpus as an ACTIVE knowledge-base version.

    . .\\scripts\\demo_env.ps1
    python scripts/publish_kb.py                     # publish the configured seed (corpus_demo)
    python scripts/publish_kb.py --verify-only       # re-verify the active version, change nothing

Why this exists
    scripts/build_index.py writes a SEED index (corpus_store.json, bm25,
    manifest).  It never publishes: no version directory, no active
    pointer.  A runtime with no pointer serves the seed as
    kb_source=seed_fallback.  The demo KB was only ever a seed, which is why
    every demo run reported seed_fallback.

What it does -- the EXISTING publication mechanism, nothing new
    1. loads the seed store (the corpus to publish);
    2. ingestion.corpus_builder.build_corpus_version(...) with no base
       version: writes <corpus_dir>/v_<id>/store.json + bm25.pkl +
       manifest.json and builds the version's Qdrant collection;
    3. ingestion.publisher.publish_corpus_version(...): loads + integrity-
       checks the snapshot (BM25 == store == Qdrant), writes the pointer
       atomically, registers the version with a lease tracker;
    4. independently re-loads the pointer in a fresh SnapshotLoader and
       verifies pointer, corpus hash, BM25 hash, Qdrant point count and
       version identity.

The server must be STOPPED (embedded Qdrant allows one process).
qdrant_mode=memory is refused: a published dense index must persist.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def verify_active(settings, runtime) -> dict:
    """Re-load the active pointer in a fresh loader and check everything."""
    from rasvcx.ingestion.publisher import read_active_version
    from rasvcx.retrieval.knowledge_base import SnapshotLoader, file_sha256

    pointer = read_active_version(settings.ingestion.active_version_path)
    if not pointer:
        raise SystemExit(f"FAILED: no active pointer at {settings.ingestion.active_version_path}")
    snap = SnapshotLoader(settings.retrieval, runtime.dense_backend).load(pointer)
    d = snap.describe()
    checks = {
        "pointer_names_loaded_version": pointer["version_id"] == d["version_id"],
        "version_id_matches_corpus_hash": d["version_id"] == "v_" + d["corpus_hash"][:12],
        "bm25_size_equals_store": d["bm25_size"] == d["chunk_count"],
        "index_hash_matches_file": d["index_hash"] == file_sha256(pointer["bm25_path"]),
        "qdrant_points_equal_chunks": (
            runtime.dense_backend.collection_count(d["qdrant_collection"]) == d["chunk_count"]
            if runtime.dense_backend is not None else None),
        "publication_status_active": d["publication_status"] == "ACTIVE",
        "identity_scheme_v2": d["identity_scheme"] == "kb-identity-v2",
    }
    return {"pointer": pointer, "describe": d, "checks": checks,
            "all_passed": all(v is not False for v in checks.values())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--report", default="")
    args = p.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(".env", override=False)
    except ImportError:
        pass
    import logging
    logging.basicConfig(level="INFO", format="%(levelname)s %(name)s: %(message)s")

    from rasvcx.config.loader import load_settings
    from rasvcx.ingestion.corpus_builder import build_corpus_version
    from rasvcx.ingestion.job_store import IngestJobStore
    from rasvcx.ingestion.publisher import KBVersionLeaseTracker, publish_corpus_version, read_active_version
    from rasvcx.pipeline.factory import build_runtime
    from rasvcx.retrieval.corpus import CorpusStore

    settings = load_settings()
    if settings.retrieval.mode == "hybrid" and settings.retrieval.qdrant_mode == "memory":
        print("REFUSED: qdrant_mode=memory -- a published dense index must persist "
              "(use embedded or server; demo_env.ps1 sets embedded)")
        return 3
    runtime = build_runtime(settings)
    report_path = Path(args.report or Path(settings.ingestion.corpus_versions_dir) / "publication_report.json")

    if args.verify_only:
        result = verify_active(settings, runtime)
    else:
        existing = read_active_version(settings.ingestion.active_version_path)
        if existing:
            print(f"An active version is already published: {existing.get('version_id')} -- verifying it")
            result = verify_active(settings, runtime)
        else:
            seed_store = CorpusStore.load(Path(settings.corpus.store_path))
            records = seed_store.to_records()
            job_store = IngestJobStore(db_path=settings.ingestion.db_path)
            job_id = str(uuid.uuid4())
            job_store.create_job(job_id=job_id, source_type="kb_publish",
                                 filename=str(settings.corpus.store_path))
            build = build_corpus_version(
                records, settings.ingestion.corpus_versions_dir, settings.ingestion.active_version_path,
                runtime.dense_backend, settings.retrieval.qdrant_collection, None,
            )
            tracker = KBVersionLeaseTracker()
            seed = runtime.initial_snapshot
            tracker.register(seed.version_id, snapshot=seed)
            asyncio.run(publish_corpus_version(
                build_result=build, active_version_path=settings.ingestion.active_version_path,
                app_state=SimpleNamespace(rasvcx_runtime=runtime), settings=settings,
                lease_tracker=tracker, job_store=job_store, job_id=job_id, executor=None,
            ))
            job = job_store.get_job(job_id)
            result = verify_active(settings, runtime)
            result["publication_job"] = {"job_id": job_id, "status": job.status,
                                         "corpus_version_id": job.corpus_version_id}
            result["lease_tracker_after_publish"] = tracker.list_versions()
            result["seed_version_id"] = seed.version_id

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    d = result["describe"]
    print(json.dumps({k: d[k] for k in ("version_id", "kb_source", "publication_status", "doc_count",
                                         "chunk_count", "corpus_hash", "index_hash", "qdrant_collection",
                                         "dense_index_origin", "identity_scheme")}, indent=2))
    for name, ok in result["checks"].items():
        print(f"  {'PASS' if ok else ('n/a ' if ok is None else 'FAIL')}  {name}")
    print(f"report: {report_path}")
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
