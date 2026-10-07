# RASVC-X

Real-time medical RAG with evidence validation. A question is answered only
from retrieved documents; the evidence is validated before generation and
every generated sentence is verified after it. When the evidence is
insufficient or contradictory the system declines to answer and says why.

Research prototype. Nothing it produces is medical advice.

## Pipeline

```
chat UI ─► POST /query ─► risk routing
        ─► BM25 + Qdrant dense ─► RRF fusion ─► cross-encoder reranking
        ─► evidence sufficiency ─► provenance analysis
        ─► atomic claims ─► deterministic ─► contextual ─► selective NLI ─► conflict resolution
        ─► grounded Gemini generation ─► post-generation verification
        ─► confidence / calibration ─► decision
        ─► ANSWER | ANSWER_WITH_WARNING | REPAIR | REGENERATE | ABSTAIN
```

The knowledge plane feeds the query plane: an uploaded or synchronised
document is validated, parsed in an isolated subprocess, chunked, indexed
into a new **knowledge-base version** (corpus store + BM25 index + its own
Qdrant collection), integrity-checked, and published atomically. Each
`/query` request leases one version for its whole lifetime.

## Run

```powershell
pip install -e ".[research,gemini,dev]"
Copy-Item .env.example .env          # then set RASVCX_LLM_API_KEY
python scripts/build_index.py --corpus data/corpus_input/medical_corpus.json
cd frontend; npm install; npm run build; cd ..
python -m rasvcx                     # http://localhost:8000
```

`python -m rasvcx` starts **research_hybrid**: real Gemini, BM25, Qdrant,
RRF, cross-encoder reranker and NLI. If any of them is unavailable the
server refuses to start — nothing is replaced by a stub.

Qdrant modes (`retrieval.qdrant_mode` / `RASVCX_QDRANT_MODE`):

| mode | needs | notes |
| --- | --- | --- |
| `server` (default) | `docker run -p 6333:6333 qdrant/qdrant` | persistent |
| `embedded` | nothing | qdrant-client local mode, persisted under `data/qdrant` |
| `memory` | nothing | dense index rebuilt from the corpus at every startup |

Offline mode uses test doubles (stub LLM, no reranker, no dense retrieval,
no NLI) and must be requested explicitly; every response is then labelled
`mode.offline: true`:

```powershell
$env:RASVCX_EXECUTION_MODE = "offline_test"; python -m rasvcx
```

Docker: `docker compose up --build` (API + UI + Qdrant server).

## Test

```powershell
python -m pytest                     # backend
cd frontend; npm run build           # typecheck + production build
python benchmarks/latency_bench.py   # against a running server
```

## API

| route | purpose |
| --- | --- |
| `POST /query` | run the pipeline; `enriched: true` adds evidence, claims, conflicts, provenance, trace, latencies |
| `GET /ready` | readiness: measured state of every component |
| `GET /status` | mode, components, KB versions and leases, measured latency percentiles |
| `GET /admin/config` | sanitised configuration (read-only at runtime) |
| `POST /ingest/upload`, `POST /ingest/url` | add a document to the knowledge base |
| `GET /ingest/status`, `/ingest/jobs`, `/ingest/sources` | knowledge-base state |
| `POST /feedback` | rate a response (stores no question or answer text) |
| `GET /eval/runs` | completed evaluation runs |

See `docs/IMPLEMENTATION_STATUS.md` for what has been verified by execution
and what has not.

## Architectural invariants

- EvidenceBundle is the shared evidence data contract.
- Risk routing controls validation budget, not final safety decisions.
- Deterministic validation always runs.
- Safety-floor overrides may force deeper validation.
- NLI is selective and swappable.
- The orchestrator performs sequencing/composition only.
- Business rules remain inside owning modules.
- Corrective actions use a single bounded global attempt budget.
- REPAIR and REGENERATE are explicitly distinguished.
- No silent fallback: a failed component yields an explicit failure or ABSTAIN.
- Retrieved documents are data, never instructions.
