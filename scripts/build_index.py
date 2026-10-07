#!/usr/bin/env python3
"""Corpus ingestion and index building for RASVC-X.

Reads a corpus JSON file, chunks all documents, builds the BM25 index,
writes the CorpusStore, and writes a CorpusManifest.  Optionally builds
a Qdrant dense index when --qdrant is specified.

This script imports ONLY from:
  rasvcx.retrieval.corpus   (CorpusDocument, CorpusStore, build_corpus_store,
                             compute_corpus_fingerprint, CorpusManifest,
                             load_corpus_json)
  rasvcx.retrieval.chunking (ChunkConfig, ChunkStrategy)
  rasvcx.retrieval.bm25     (BM25Index)
  rasvcx.retrieval.dense    (DenseRetriever) -- only when --qdrant is used

It does NOT import from bridge.py, factory.py, or any API/config module.

Safe Qdrant rebuild strategy:
  1. Embed and upload into a temporary collection {name}_building.
  2. Validate the new collection (verify document count).
  3. Delete the old collection if it exists.
  4. Recreate live collection from same data.
  Never touching the live collection during the build phase.

Usage:
  python scripts/build_index.py \
    --corpus corpus/smoke_corpus.json \
    --store  corpus/corpus_store.json \
    --bm25   corpus/bm25_index.pkl \
    --manifest corpus/manifest.json \
    [--strategy fixed] [--max-tokens 256] [--overlap 32] \
    [--qdrant] [--qdrant-host localhost] [--qdrant-port 6333] \
    [--qdrant-collection rasvcx_chunks] \
    [--dense-model sentence-transformers/all-MiniLM-L6-v2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when run from the repo root
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rasvcx.retrieval.bm25 import BM25Index
from rasvcx.retrieval.chunking import ChunkConfig, ChunkStrategy
from rasvcx.retrieval.corpus import (
    CorpusManifest,
    CorpusStore,
    build_corpus_store,
    compute_corpus_fingerprint,
    load_corpus_json,
)


def _build_bm25(store: CorpusStore, bm25_path: Path) -> None:
    print(f"  Building BM25 index ({len(store)} chunks)...")
    pairs = store.to_documents_list()
    index = BM25Index.build(pairs)
    index.save(bm25_path)
    print(f"  BM25 index saved -> {bm25_path}")


def _build_qdrant(
    store: CorpusStore, prefix: str, host: str, port: int, dense_model: str,
    mode: str, path: str,
) -> str:
    """Pre-build the seed corpus's Qdrant collection.

    Optional: the server builds a missing collection itself at startup.
    Pre-building only saves that startup time (server / embedded modes).

    The collection is named exactly as the server expects for this corpus:
    "<prefix>_<version_id>", where version_id is derived from the corpus
    content.  build_collection() verifies the point count before returning.
    """
    try:
        from rasvcx.pipeline.factory import seed_version_id
        from rasvcx.retrieval.dense import DenseBackend
        from rasvcx.retrieval.knowledge_base import collection_name_for
    except ImportError as exc:
        print(f"ERROR: Qdrant build requires qdrant-client and sentence-transformers: {exc}")
        sys.exit(1)

    collection = collection_name_for(prefix, seed_version_id(store))
    print(f"  Building dense index '{collection}' (qdrant_mode={mode})...")
    try:
        backend = DenseBackend(dense_model, mode=mode, host=host, port=port, path=path)
        count = backend.build_collection(collection, store.to_documents_list())
    except Exception as exc:
        print(f"ERROR: dense index build failed: {exc}")
        sys.exit(1)
    print(f"  Dense index built and verified: {count} points")
    return collection


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RASVC-X corpus indexes")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSON file")
    parser.add_argument("--store", default="corpus/corpus_store.json")
    parser.add_argument("--bm25", default="corpus/bm25_index.pkl")
    parser.add_argument("--manifest", default="corpus/manifest.json")
    parser.add_argument("--strategy", default="fixed", choices=["fixed", "sentence_window"])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--qdrant", action="store_true")
    parser.add_argument("--qdrant-host", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--qdrant-collection", default="rasvcx_chunks",
                        help="collection prefix (retrieval.qdrant_collection)")
    parser.add_argument("--qdrant-mode", default="server", choices=["server", "embedded"])
    parser.add_argument("--qdrant-path", default="data/qdrant")
    parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    store_path = Path(args.store)
    bm25_path = Path(args.bm25)
    manifest_path = Path(args.manifest)

    for p in [store_path, bm25_path, manifest_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {corpus_path}...")
    documents = load_corpus_json(corpus_path)
    print(f"  Loaded {len(documents)} documents")

    synthetic_count = sum(1 for d in documents if d.is_synthetic)
    if synthetic_count < len(documents):
        print(f"  NOTE: {len(documents) - synthetic_count} document(s) marked is_synthetic=False")

    strategy = ChunkStrategy(args.strategy)
    chunk_config = ChunkConfig(strategy=strategy, max_tokens=args.max_tokens, overlap=args.overlap)

    print("Computing corpus fingerprint...")
    fingerprint = compute_corpus_fingerprint(documents, chunk_config)
    print(f"  Fingerprint: {fingerprint[:16]}...")

    print("Chunking documents and building CorpusStore...")
    store, _ = build_corpus_store(documents, chunk_config)
    print(f"  {len(store)} chunks produced")

    store.save(store_path)
    print(f"  CorpusStore saved -> {store_path}")

    _build_bm25(store, bm25_path)

    qdrant_collection = None
    if args.qdrant:
        qdrant_collection = _build_qdrant(
            store=store, prefix=args.qdrant_collection,
            host=args.qdrant_host, port=args.qdrant_port, dense_model=args.dense_model,
            mode=args.qdrant_mode, path=args.qdrant_path,
        )

    manifest = CorpusManifest(
        corpus_fingerprint=fingerprint,
        chunk_count=len(store),
        doc_count=len(documents),
        store_path=str(store_path),
        bm25_path=str(bm25_path),
        qdrant_collection=qdrant_collection,
    )
    manifest.save(manifest_path)
    print(f"  Manifest saved -> {manifest_path}")

    print("\n=== Index build complete ===")
    print(f"  Documents : {len(documents)}")
    print(f"  Chunks    : {len(store)}")
    print(f"  BM25      : {bm25_path}")
    print(f"  Store     : {store_path}")
    print(f"  Manifest  : {manifest_path}")
    if qdrant_collection:
        print(f"  Qdrant    : {qdrant_collection}")
    print(f"  Fingerprint: {fingerprint}")


if __name__ == "__main__":
    main()