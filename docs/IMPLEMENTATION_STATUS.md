# Implementation status

Status as of 2026-10-06. Every "verified" entry below was established by
running the code, not by reading it. Anything not executed is listed as
such at the end.

## Module execution matrix (research_hybrid, live server)

Verified against a running `python -m rasvcx` with real Gemini, the
cross-encoder, DeBERTa NLI and Qdrant (qdrant-client memory mode).
"Output consumed by" names the next thing that actually reads the output.

| Module | Executes in `/query` | Output consumed by | Tests |
| --- | --- | --- | --- |
| Risk routing | yes | sufficiency gate, selective router, decision engine | test_risk_router, trace test |
| BM25 | yes | RRF | test_retrieval, test_corpus |
| Qdrant dense | yes | RRF | test_retrieval (fake client), live |
| RRF fusion | yes | reranker | test_retrieval |
| Cross-encoder reranker | yes | sufficiency gate, confidence | test_reranking, live |
| Evidence sufficiency | yes | orchestrator (proceed / targeted retrieval / abstain) | test_sufficiency |
| Provenance analysis | yes | response (per-evidence verdicts), warnings | test_provenance, trace test |
| Atomic claim extraction | yes | validation | test_atomic_claims |
| Deterministic validation | yes | selective router, resolver | test_integration_repair |
| Contextual validation | yes | selective router, resolver | test_validation_contextual |
| Selective NLI (evidence pairs) | only when the router selects a pair | resolver | test_validation_selective |
| Conflict resolution | yes | prompt, verification, confidence, decision | test_resolution |
| Gemini generation | yes | verification | test_generation, live |
| Post-generation verification | yes (incl. NLI per claim) | confidence, decision | test_verification, live |
| Confidence | yes | decision | test_confidence_pipeline |
| Calibration | yes, returns `uncalibrated` | decision | test_confidence |
| Decision engine | yes | response | test_decision_engine |
| KB versioning / leases | yes | `/query`, publication, GC | test_integration_repair |
| Ingestion | yes | KB version | test_ingestion, test_integration_repair |
| Source synchronisation | code path tested with a mock transport | ingestion | test_integration_repair |
| Feedback | yes | append-only file for offline review | test_integration_repair |
| Observability | yes | `/status`, response trace | test_integration_repair |

Notes on the two partial rows:

- **Provenance analysis** runs and is reported, but no decision module
  takes its result as an input: sufficiency, confidence and resolution
  each compute their own provenance signals. It informs the user
  (warnings, per-evidence context verdicts), not the decision.
- **Selective NLI on evidence pairs** did not fire in any live query: with
  fully known provenance the contextual result is conclusive, and the
  router then does not escalate. NLI did execute live in post-generation
  verification. The pair-level path is covered by unit tests only.

## Known limitations

- **Calibration** is the identity: no calibration artifact has been fitted
  (that needs labelled evaluation data). Confidence is a raw score and the
  response says so (`calibration_status: uncalibrated`).
- **Abstention is frequent on the seed corpus.** High-risk questions need
  two independent source types; the seed corpus is ~95% FDA labels, so many
  dosing/contraindication questions abstain until a second source type is
  ingested. This is the configured policy
  (`sufficiency.min_source_diversity_high_risk`), not a fault.
- **Seed-corpus provenance is coarse** (every label is tagged
  population=adults), so an answer sentence about pediatric use can be
  marked contradicted by the population check.
- **Publishing re-embeds the whole corpus** into the new version's Qdrant
  collection (about 45 s for 1,100 chunks on CPU). Vectors of unchanged
  chunks are not reused.
- **Latency** is dominated by generation, CPU NLI and CPU reranking
  (see `benchmarks/latency_bench.py`).
- **Gemini availability** varies by model; see the note in
  `config/research_hybrid.yaml`.
- **Single process only**: leases, the job store and the embedded/memory
  Qdrant modes are per-process. Run one worker.
- **DOCX / PPTX / JSON / XML are parsed in memory** (in an isolated
  subprocess with a timeout); their size limits are lower than the
  streamed formats for that reason. Plain ZIP archives are rejected.
- **Scanned PDFs**: there is no OCR; a PDF without a text layer fails
  ingestion with "no sections" rather than being indexed empty.

## Not executed / not implemented

| Item | Status | Reason / next action |
| --- | --- | --- |
| Docker build and compose run | NOT EXECUTED | Docker is not installed on the development machine. Run `docker compose up --build`. |
| Qdrant **server** mode | NOT EXECUTED | No Qdrant server available; verified with qdrant-client memory mode, which uses the same client API. Start Qdrant and set `RASVCX_QDRANT_MODE=server`. |
| End-to-end evaluation (retrieval recall, nDCG, ECE, Brier, ...) | NOT EXECUTED | The metric functions exist and are unit-tested, but the repository contains no labelled evaluation dataset. Build one with `evaluation/dataset.py`, then run `evaluation/runner.py`. |
| Throughput / concurrency benchmark | NOT EXECUTED | Only sequential latency was measured. |
| p99 latency | NOT REPORTED | Needs at least 100 samples; 20 were collected. |
| Large-document ingestion (hundreds of MB) | NOT EXECUTED | Limits and streaming paths exist; no large file was ingested. |
| Real external URL fetch | NOT EXECUTED | The fetch path was tested with a mock transport only (no outbound request was made). |
| OCR, PII redaction, `src/rasvcx/security/*`, `utils/*`, `caching/*` | NOT IMPLEMENTED | These files are empty placeholders. |
| `deployment/docker-compose.prod.yml`, `evaluation/run_ablation.py`, `benchmarks/throughput_locust.py` and the other 0-byte files | NOT IMPLEMENTED | Empty placeholders. |
