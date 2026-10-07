"""evaluation/poisoning.py

Isolated poisoned-index builder for M15 controlled experiments.

SAFETY INVARIANTS (enforced in code):
  1. experiment_dir != active_corpus_store_path.parent (IsolationError)
  2. Active CorpusManifest fingerprint is recorded before and verified
     unchanged after build_poisoned_index().
  3. All writes go to experiment_dir only; active paths are read-only.
  4. Qdrant poisoning is not implemented; status is SKIPPED when attempted.

LIMITATIONS:
  - Adversarial documents must be clearly labelled as test artifacts.
  - Never use identifiable patient data in adversarial documents.
  - No adversarial document constitutes real clinical guidance.
  - This system is a research prototype; no clinical deployment claims.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evaluation.schema import CorpusCondition, CorpusConfig


# ---------------------------------------------------------------------------
# IsolationError
# ---------------------------------------------------------------------------


class IsolationError(Exception):
    """Raised when experiment_dir overlaps with active corpus paths."""


# ---------------------------------------------------------------------------
# AdversarialDocument
# ---------------------------------------------------------------------------


@dataclass
class AdversarialDocument:
    """One adversarial document to inject into a poisoned corpus.

    doc_id must not exist in the active corpus.
    text should be clearly synthetic; never real patient data.
    """

    doc_id: str
    title: str
    text: str
    condition: CorpusCondition
    source_type: str = "adversarial_test_artifact"
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.doc_id.strip():
            raise ValueError("AdversarialDocument.doc_id must be non-empty")
        if not self.text.strip():
            raise ValueError("AdversarialDocument.text must be non-empty")
        if self.condition == CorpusCondition.CLEAN:
            raise ValueError(
                "AdversarialDocument.condition must not be CLEAN"
            )


# ---------------------------------------------------------------------------
# PoisoningExperiment
# ---------------------------------------------------------------------------


@dataclass
class PoisoningExperiment:
    """Configuration for one controlled poisoning experiment.

    active_corpus_store_path and active_bm25_path are READ-ONLY.
    All writes go to experiment_dir.
    """

    experiment_id: str
    condition: CorpusCondition
    active_corpus_store_path: Path
    active_bm25_path: Path
    experiment_dir: Path
    adversarial_docs: list[AdversarialDocument] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.active_corpus_store_path = Path(self.active_corpus_store_path)
        self.active_bm25_path = Path(self.active_bm25_path)
        self.experiment_dir = Path(self.experiment_dir)

        if self.condition == CorpusCondition.CLEAN:
            raise ValueError(
                "PoisoningExperiment.condition must not be CLEAN"
            )

    def _check_isolation(self) -> None:
        """Raise IsolationError if experiment_dir overlaps active paths."""
        active_parent = self.active_corpus_store_path.parent.resolve()
        exp_dir = self.experiment_dir.resolve()

        if exp_dir == active_parent:
            raise IsolationError(
                f"experiment_dir {exp_dir!r} is the same as the active "
                f"corpus directory {active_parent!r}. "
                f"Experiment must use an isolated directory."
            )
        # Also check neither is a subdirectory of the other
        try:
            exp_dir.relative_to(active_parent)
            raise IsolationError(
                f"experiment_dir {exp_dir!r} is inside the active corpus "
                f"directory {active_parent!r}."
            )
        except ValueError:
            pass  # not a subdirectory — correct


def build_poisoned_index(experiment: PoisoningExperiment) -> CorpusConfig:
    """Build an isolated poisoned BM25 index and return its CorpusConfig.

    Steps:
      1. Assert isolation (experiment_dir != active paths).
      2. Record active corpus fingerprint.
      3. Load active CorpusStore read-only.
      4. Create new in-memory corpus by copying active + adding adversarial docs.
      5. Save to experiment_dir (new paths only).
      6. Build BM25Index and save to experiment_dir.
      7. Write experiment manifest.
      8. Reload active manifest; assert fingerprint unchanged.
      9. Return CorpusConfig pointing to experiment_dir.

    Raises:
      IsolationError: if experiment_dir overlaps active paths.
      FileNotFoundError: if active corpus files do not exist.
    """
    from rasvcx.retrieval.bm25 import BM25Index
    from rasvcx.retrieval.corpus import CorpusDocument, CorpusStore

    # Step 1: isolation check
    experiment._check_isolation()

    # Verify active files exist
    if not experiment.active_corpus_store_path.exists():
        raise FileNotFoundError(
            f"Active CorpusStore not found: {experiment.active_corpus_store_path}"
        )
    if not experiment.active_bm25_path.exists():
        raise FileNotFoundError(
            f"Active BM25 index not found: {experiment.active_bm25_path}"
        )

    # Step 2: record active fingerprint
    active_bytes_before = experiment.active_corpus_store_path.read_bytes()
    active_fingerprint = _sha256_of_bytes(active_bytes_before)

    # Step 3: load active CorpusStore (read-only) — get chunk_id/text/provenance pairs
    active_store = CorpusStore.load(str(experiment.active_corpus_store_path))

    # Step 4: build poisoned corpus in memory
    # Copy all active (chunk_id, text) pairs + add adversarial doc chunks
    from rasvcx.schemas.common import ChunkId
    from rasvcx.schemas.evidence import Provenance
    from rasvcx.schemas.common import SourceType, UNKNOWN

    poisoned_store = CorpusStore()
    chunk_pairs: list[tuple] = []

    # Copy active chunks
    for chunk_id in active_store.chunk_ids():
        result = active_store.lookup(chunk_id)
        if result is not None:
            text, provenance = result
            poisoned_store.add(ChunkId(chunk_id), text, provenance)
            chunk_pairs.append((ChunkId(chunk_id), text))

    # Add adversarial document chunks (one chunk per adversarial doc)
    for adv_doc in experiment.adversarial_docs:
        adv_chunk_id = ChunkId(f"{adv_doc.doc_id}__chunk_0000")
        adv_provenance = Provenance(
            source_type=SourceType.OTHER,
            date=UNKNOWN,
            jurisdiction=UNKNOWN,
            population=UNKNOWN,
            dosage_context=UNKNOWN,
        )
        poisoned_store.add(adv_chunk_id, adv_doc.text, adv_provenance)
        chunk_pairs.append((adv_chunk_id, adv_doc.text))

    # Step 5: create experiment_dir and save poisoned store
    experiment.experiment_dir.mkdir(parents=True, exist_ok=True)
    exp_store_path = experiment.experiment_dir / "corpus_store.json"
    exp_bm25_path = experiment.experiment_dir / "bm25_index.pkl"
    exp_manifest_path = experiment.experiment_dir / "manifest.json"

    poisoned_store.save(str(exp_store_path))

    # Step 6: build and save BM25 index
    exp_index = BM25Index.build(chunk_pairs)
    exp_index.save(str(exp_bm25_path))

    # Compute experiment fingerprint
    experiment_fingerprint = _sha256_of_bytes(exp_store_path.read_bytes())

    # Step 7: write experiment manifest
    manifest = {
        "experiment_id": experiment.experiment_id,
        "condition": experiment.condition.value,
        "parent_fingerprint": active_fingerprint,
        "experiment_fingerprint": experiment_fingerprint,
        "injected_doc_ids": [d.doc_id for d in experiment.adversarial_docs],
        "store_path": str(exp_store_path),
        "bm25_path": str(exp_bm25_path),
        "qdrant_collection": None,
        "note": (
            "Adversarial documents are synthetic test artifacts. "
            "This index must not be used as the trusted corpus."
        ),
    }
    with open(exp_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Step 8: verify active corpus is unchanged
    active_bytes_after = experiment.active_corpus_store_path.read_bytes()
    active_fingerprint_after = _sha256_of_bytes(active_bytes_after)
    if active_fingerprint_after != active_fingerprint:
        raise IsolationError(
            f"Active corpus fingerprint changed during experiment build! "
            f"Before: {active_fingerprint!r}, After: {active_fingerprint_after!r}. "
            f"This indicates an isolation violation."
        )

    # Step 9: return CorpusConfig for the experiment
    return CorpusConfig(
        store_path=exp_store_path,
        bm25_path=exp_bm25_path,
        fingerprint=experiment_fingerprint,
        condition=experiment.condition,
    )


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "IsolationError",
    "AdversarialDocument",
    "PoisoningExperiment",
    "build_poisoned_index",
]