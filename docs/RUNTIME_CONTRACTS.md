# RASVC-X runtime contracts

Rules the code enforces, with where they are enforced and tested.

## Knowledge-base identity (kb-identity-v2)

- **What the identity covers.** A KB version id is `v_<first 12 hex of the corpus hash>`. The corpus hash is a SHA-256 over every chunk's identity fields: `chunk_id`, `text`, `provenance`, `lifecycle`, `doc_id`, `title`, `source_url`, `filename`, `heading`, `page`. This is everything the query plane reads from the store.
- **What it excludes.** Volatile ingestion bookkeeping (`job_id`, `ingestion_timestamp`, `content_hash`, `chunk_index`, `chunker_version`, `section`, ...) is excluded. Empty string and null are the same value.
- **Metadata-only changes count.** A change to provenance, lifecycle or attribution is a new version.
  - Source: `ingestion/corpus_builder.py` (`_fingerprint`, `IDENTITY_FIELDS`).
  - Tests: `tests/test_evidence_semantics.py::TestIdentity`.
- **Load-time check.** At load, the hash is recomputed from the loaded chunks. A content-form version id that does not match is a `KBIntegrityError`.
  - Ids published before v2 (text-only hash) still load, labelled `kb-identity-v1-text-only`.
- **Exposed in:** `/ready`, `/status` and every `/query` response (`kb.identity_scheme`, `kb.corpus_hash`, `kb.index_hash`, `kb.kb_source`).
- **KB source** is `published_kb`, `seed_fallback` or `smoke_test`.
  - Seed fallback is logged at WARNING and listed in `/ready.kb_warnings`.
  - `ingestion.require_published_kb` (or `RASVCX_REQUIRE_PUBLISHED_KB=1`) turns seed fallback into a startup error.
- **Consistency check.** `/ready` component `kb_consistency` fails unless the pointer, lease tracker, snapshot, BM25 and store all name one version.

## Evidence semantics

- **Temporal state** comes only from DECLARED lifecycle metadata (`SourceLifecycle`: `status`, `effective_date`, `superseded_by`, `supersedes`, `version`):
  - The states are CURRENT, SUPERSEDED, HISTORICAL, WITHDRAWN and UNKNOWN.
  - A publication date alone never makes a source superseded.
  - `supersedes` declarations apply KB-wide: a newer document marks the older one superseded even when only the older one is retrieved.
- **Role** has one value per item, and retrieval alone never makes an item SUPPORTING. Precedence, first match wins:
  1. CONTRADICTORY
  2. SUPERSEDED
  3. SUPPORTING (M9 says it supports a supported claim)
  4. RELEVANT
  5. IRRELEVANT
  6. RETRIEVED
- **Where roles are used:**
  - Provenance warnings (outdated, population, jurisdiction, dosage, incomplete provenance) are raised only for items the answer relies on.
  - The confidence features `provenance_quality` and `source_diversity` are computed over the answer's supporting, current evidence.
  - The decision engine handles an answer resting on withdrawn or superseded evidence as follows:
    1. It is first repaired with feedback naming the stale items.
    2. If the repair budget is spent, a high-risk answer is withheld (ABSTAIN) and a low-risk one gets ANSWER_WITH_WARNING.
    3. Separately, declared-historical support caps any answer at ANSWER_WITH_WARNING.
- **Conflict resolution** (`validation/resolution.py`, rule 1b):
  - A contradiction is explained by time only when exactly one side is declared superseded, withdrawn or historical.
  - Two declared-current sources that disagree are a GENUINE_CONFLICT.
  - A contradiction with diverging dates and no declared supersession is UNRESOLVED.
  - Critical unresolved or genuine conflicts at high risk lead to ABSTAIN (the existing safety gate).

## Confidence calibration

- **Status values.**
  - `uncalibrated` means no artifact is configured.
  - `calibrated` is the only status in which confidence is an estimated probability of correctness.
  - `invalidated` means the artifact exists but its config hash, prompt version, model versions, or the request's KB version differ from what it was fitted under; the raw score is used and labelled.
- **Protocol** (`scripts/calibrate.py`, `evaluation/protocol.py`):
  1. `freeze` hashes the held-out `test` split.
  2. `fit` uses only the `calibration` split, with raw scores and gold correctness of returned answers.
     - It refuses with fewer than 100 samples or fewer than 10 per class.
     - It uses Platt scaling below 1000 samples and isotonic above.
  3. `evaluate` runs the frozen test split, reports raw vs calibrated ECE, Brier, a reliability table and per-risk metrics, and logs every evaluation in the lock file.
- **The 18 demo cases** are split `safety_regression`. They are never calibration or accuracy data.
- **No calibration artifact exists in the repository.**

## Benchmarks

Every benchmark first verifies the loaded KB against an expected-KB file (`<dataset>.kb.json`) and refuses with `BENCHMARK_CONFIGURATION_ERROR` on any mismatch. Provider errors and system errors are counted, never scored. Cache hits are never inference latency or throughput: the server tags every response with `request_class`, and the benchmarks send `bypass_cache=true` by default.

## Single-process boundary (what is NOT distributed)

- **What is in-process.** One Uvicorn worker per deployment (`python -m rasvcx` forces `workers=1`). All of the following live in this process:
  - the KB lease tracker
  - the answer cache
  - the rate limiter
  - the ingestion job queue (SQLite)
  - Qdrant `embedded` / `memory` modes
- **Scaling consequences.** Running several processes behind a load balancer would give each its own leases, cache and rate limits. That configuration is not supported and not implemented. Horizontal scaling would require:
  - a shared lease store
  - Qdrant in server mode
  - a shared job store
  - a shared rate-limit store
- **Isolation and concurrency.**
  - Per-request state lives in `_Run`; the orchestrator holds no request state (tested in `test_integration_repair.py::TestThreadSafety` and `test_security_runtime.py::TestConcurrentIsolation`).
  - Admission is bounded (`api.max_workers`); excess requests get 503 rather than queueing.
- **Auth.**
  - Bearer token (`RASVCX_AUTH_TOKEN`) for queries.
  - Optional separate `RASVCX_ADMIN_TOKEN` required for ingestion and `/admin` (a query token gets 403).
  - Per-principal token-bucket rate limit (`api.rate_limit_per_minute`).
  - Structured, PII-free events on the `rasvcx.security` logger.
  - There is no multi-tenant isolation: one knowledge base per deployment.
