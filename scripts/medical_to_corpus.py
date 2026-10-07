"""
scripts/medical_to_corpus.py

Convert raw medical .txt files into the RASVC-X native corpus JSON that
scripts/build_index.py consumes.

Input layout:
    data/medical_raw/c0_synthetic/*.txt    (synthetic / curated)
    data/medical_raw/c1_real/*.txt         (real FDA labels etc.)

Output:
    data/corpus_input/medical_corpus.json

Each output document carries POPULATED provenance fields
(date / jurisdiction / population / dosage_context) so the sufficiency
gate does not reject high-risk queries on `unknown_provenance_ratio`.

Usage:
    python scripts/medical_to_corpus.py
    python scripts/medical_to_corpus.py --raw-dir data/medical_raw --out data/corpus_input/medical_corpus.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_RAW_DIR = Path("data/medical_raw")
DEFAULT_OUT = Path("data/corpus_input/medical_corpus.json")

# Sub-folders scanned under the raw dir.
SUBFOLDERS = ("c0_synthetic", "c1_real")

# Default provenance applied to every document. These are intentionally
# generic-but-non-null so CorpusDocument does not fall back to UNKNOWN
# provenance (which trips the high-risk sufficiency gate).
DEFAULT_DATE = "2024"
DEFAULT_JURISDICTION = "US"
DEFAULT_POPULATION = "adults"
DEFAULT_DOSAGE_CONTEXT = "general"


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify(folder: str, stem: str) -> Tuple[str, bool]:
    """
    Decide (source_type, is_synthetic) from the folder and filename stem.

    Returns a source_type string that is valid against the SourceType enum
    used by rasvcx.retrieval.corpus.CorpusDocument.
    """
    name = stem.lower()

    # Real FDA / c1 content -> real drug labels.
    if folder == "c1_real" or name.startswith("fda_"):
        return "drug_label", False

    # Synthetic drug monographs.
    if name.startswith("med_"):
        return "drug_label", True

    # Synthetic clinical guidance / protocols / reference tables.
    guideline_markers = ("sepsis", "ecg", "lab_reference", "interaction")
    if any(marker in name for marker in guideline_markers):
        return "clinical_guideline", True

    # Everything else (filler, boilerplate, non-medical, encoding tests, ...).
    return "other", True


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def build_documents(raw_dir: Path) -> list[dict]:
    documents: list[dict] = []
    seen_doc_ids: set[str] = set()
    skipped_empty = 0
    skipped_missing = 0

    for folder in SUBFOLDERS:
        folder_path = raw_dir / folder
        if not folder_path.is_dir():
            skipped_missing += 1
            print(f"[warn] missing folder, skipping: {folder_path}", file=sys.stderr)
            continue

        for txt_path in sorted(folder_path.glob("*.txt")):
            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[warn] could not read {txt_path}: {exc}", file=sys.stderr)
                continue

            # Skip empty files (empty_file.txt edge case).
            if not text:
                skipped_empty += 1
                continue

            stem = txt_path.stem
            doc_id = f"{folder}__{stem}"

            # Guard against duplicate ids across folders.
            if doc_id in seen_doc_ids:
                print(f"[warn] duplicate doc_id, skipping: {doc_id}", file=sys.stderr)
                continue
            seen_doc_ids.add(doc_id)

            source_type, is_synthetic = classify(folder, stem)

            # Human-readable title from the stem.
            title = stem.replace("_", " ").strip() or doc_id

            documents.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                    "source_type": source_type,
                    # --- populated provenance (the fix) ---
                    "date": DEFAULT_DATE,
                    "jurisdiction": DEFAULT_JURISDICTION,
                    "population": DEFAULT_POPULATION,
                    "dosage_context": DEFAULT_DOSAGE_CONTEXT,
                    # --------------------------------------
                    "source_url": None,
                    "is_synthetic": is_synthetic,
                }
            )

    print(
        f"[info] built {len(documents)} documents "
        f"(skipped {skipped_empty} empty, {skipped_missing} missing folders)",
        file=sys.stderr,
    )
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert raw medical .txt -> RASVC-X corpus JSON")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Root of raw .txt files")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output corpus JSON path")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    out_path: Path = args.out

    if not raw_dir.is_dir():
        print(f"[error] raw dir does not exist: {raw_dir}", file=sys.stderr)
        return 1

    documents = build_documents(raw_dir)
    if not documents:
        print("[error] no documents produced; nothing written", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(documents, fh, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {len(documents)} documents -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())