// frontend/src/types.ts
// ---------------------------------------------------------------------------
// TypeScript mirror of the RASVC-X HTTP API.
//
// Every interface here corresponds to a backend model; the source is named
// next to each one. Do not add a field here that the backend does not send:
// the UI must only ever display real backend state.
// ---------------------------------------------------------------------------

// ── /query ──────────────────────────────────────────────────────────────────

/** Canonical decision values (rasvcx.schemas.decision.CANONICAL_DECISIONS). */
export type Decision =
  | 'ANSWER'
  | 'ANSWER_WITH_WARNING'
  | 'REPAIR'
  | 'REGENERATE'
  | 'ABSTAIN';

/** api/models.py QUERY_CONTEXT_KEYS */
export interface QueryContext {
  population?: string;
  jurisdiction?: string;
  dosage_context?: string;
  time_sensitivity?: string;
}

/** api/models.py QueryRequest */
export interface QueryRequest {
  query: string;
  request_id?: string;
  enriched: true;
  context?: QueryContext;
}

/** api/models_enriched.py ProvenanceModel — null means UNKNOWN */
export interface Provenance {
  source_type: string;
  date: string | null;
  jurisdiction: string | null;
  population: string | null;
  dosage_context: string | null;
}

/** api/models_enriched.py ProvenanceAnalysisModel */
export interface ProvenanceAnalysis {
  source_known: boolean;
  quality_score: number;
  temporal: string;
  jurisdiction: string;
  population: string;
  dosage_context: string;
}

/** api/models_enriched.py EvidenceModel */
export interface Evidence {
  evidence_id: string;
  chunk_id: string;
  doc_id: string | null;
  title: string | null;
  source_url: string | null;
  filename: string | null;
  section: string | null;
  page: number | null;
  text: string;
  retrieval_score: number;
  rerank_score: number | null;
  cited: boolean;
  provenance: Provenance;
  analysis: ProvenanceAnalysis | null;
  /** provenance/evidence_roles.py EvidenceRole */
  role: 'RETRIEVED' | 'RELEVANT' | 'SUPPORTING' | 'CONTRADICTORY' | 'IRRELEVANT' | 'SUPERSEDED';
  role_reason: string | null;
  /** declared lifecycle only; never inferred from the date */
  temporal_status: 'CURRENT' | 'SUPERSEDED' | 'HISTORICAL' | 'WITHDRAWN' | 'UNKNOWN';
  temporal_reason: string | null;
  supports_answer: boolean;
  contradicts_answer: boolean;
  lifecycle: { status: string | null; effective_date: string | null; superseded_by: string | null;
               supersedes: string[]; version: string | null } | null;
  authority_tier: string | null;
}

/** api/models_enriched.py ClaimVerificationModel */
export interface ClaimResult {
  claim_id: string;
  text: string;
  cited_evidence_ids: string[];
  citation_status: string | null;
  is_safety_critical: boolean;
  label: string;
  confidence: number;
  stage: string;
  reason_code: string | null;
  rationale: string | null;
  supporting_item_ids: string[];
  contradicting_item_ids: string[];
}

/** api/models_enriched.py VerificationSummaryModel */
export interface VerificationSummary {
  answer_verdict: string;
  overall_confidence: number;
  claim_results: ClaimResult[];
  semantic_verification_calls: number;
  safety_critical_failure_count: number;
  budget_exhausted: boolean;
}

/** api/models_enriched.py ConflictResolutionModel */
export interface ConflictResolution {
  candidate_id: string;
  evidence_ids: string[];
  validation_stage: string | null;
  validation_label: string | null;
  relationship: string;
  confidence: number;
  contributing_claim_ids: string[];
  rationale: string | null;
}

/** api/models_enriched.py ValidationSummaryModel */
export interface ValidationSummary {
  candidates_generated: number;
  nli_calls_used: number;
  nli_failures: number;
  genuine_conflict_count: number;
  unresolved_count: number;
  resolutions: ConflictResolution[];
}

/** api/models_enriched.py ProvenanceSummaryModel */
export interface ProvenanceSummary {
  unique_source_count: number;
  unknown_provenance_ratio: number;
  mismatch_present: boolean;
  strict_context_required: boolean;
}

/** api/models_enriched.py RiskProfileEnrichedModel */
export interface RiskProfile {
  overall_risk_score: number;
  validation_depth: string;
  retrieval_retry_budget: number;
  nli_call_allowance: number;
  safety_floor_forced: boolean;
  feature_scores: Record<string, number>;
}

/** api/models_enriched.py StageTraceModel */
export interface StageTrace {
  stage: string;
  status: 'ok' | 'failed' | 'skipped';
  elapsed_ms: number;
  detail: string | null;
  attempt: number;
}

/** api/models_enriched.py RuntimeModeModel */
export interface RuntimeMode {
  execution_mode: string;
  offline: boolean;
  llm_provider: string;
  llm_model: string | null;
  mock_llm: boolean;
  retrieval_mode: string;
  reranker: boolean;
  nli: boolean;
}

export type ModuleState = 'executed' | 'failed' | 'skipped' | 'not_reached';

/** api/models_enriched.py EnrichedQueryResponse */
export interface QueryResponse {
  query_id: string;
  request_id: string | null;
  success: boolean;
  decision: Decision;
  action: string;
  confidence: number;
  rationale: string | null;
  answer: string;
  generated_text: string;
  has_answer: boolean;
  evidence: Evidence[];
  citations: string[];
  retrieved_chunk_ids: string[];
  verification: VerificationSummary | null;
  validation: ValidationSummary | null;
  provenance: ProvenanceSummary | null;
  risk_profile: RiskProfile | null;
  retrieval: Record<string, string | number | null>;
  kb_version_id: string | null;
  trace: StageTrace[];
  modules: Record<string, ModuleState>;
  stage_latencies_ms: Record<string, number>;
  total_latency_ms: number;
  nli_calls: number;
  /** uncalibrated | calibrated | invalidated | unavailable */
  calibration_status: string | null;
  calibration_version: string | null;
  calibration_dataset_hash: string | null;
  corrective_attempts: number;
  warnings: string[];
  degraded: boolean;
  pipeline_error: { stage: string; message: string; is_retryable: boolean } | null;
  mode: RuntimeMode | null;
  mock_llm: boolean;
  limitations: string;
  cached: boolean;
  /** api/models.py REQUEST_CLASSES */
  request_class:
    | 'COLD_UNCACHED' | 'WARM_UNCACHED' | 'CACHE_HIT' | 'PROVIDER_ERROR' | 'SYSTEM_ERROR' | null;
  kb: {
    version_id: string | null;
    kb_source: 'published_kb' | 'seed_fallback' | 'smoke_test' | null;
    corpus_hash?: string | null;
    doc_count?: number;
    chunk_count?: number;
  } | null;
}

/** One turn of the chat (UI-only container; holds no invented backend data). */
export interface ChatEntry {
  id: string;
  question: string;
  context?: QueryContext;
  pending: boolean;
  response?: QueryResponse;
  error?: string;
  feedback?: 'up' | 'down';
}

// ── /feedback ───────────────────────────────────────────────────────────────

/** api/routes_feedback.py FeedbackRequest */
export interface FeedbackRequest {
  query_id: string;
  rating: 'up' | 'down';
  decision: Decision;
  kb_version_id: string | null;
  reason?: string;
}

// ── /ready, /status, /admin/config ──────────────────────────────────────────

/** api/models.py ComponentState */
export type ComponentStateValue =
  | 'configured'
  | 'loaded'
  | 'reachable'
  | 'unavailable'
  | 'disabled'
  | 'stub';

/** api/models.py ComponentStatus */
export interface ComponentStatus {
  name: string;
  state: ComponentStateValue;
  detail: string | null;
}

/** api/models.py ReadinessResponse */
export interface Readiness {
  ready: boolean;
  execution_mode: string;
  offline: boolean;
  llm_provider: string;
  kb_version_id: string | null;
  kb_source?: string | null;
  kb_corpus_hash?: string | null;
  kb_doc_count?: number | null;
  kb_chunk_count?: number | null;
  calibration?: { status: string; calibration_version?: string | null; reason?: string | null } | null;
  kb_warnings?: string[];
  components: ComponentStatus[];
}

/** retrieval/knowledge_base.py KBSnapshot.describe() */
export interface KBActive {
  version_id: string;
  chunk_count: number;
  doc_count: number;
  bm25_size: number;
  qdrant_collection: string | null;
  retrieval_mode: string;
  published_at: string | null;
}

/** ingestion/publisher.py KBVersionLeaseTracker.list_versions() */
export interface KBVersion {
  version_id: string;
  active_request_count: number;
  status: 'active' | 'inactive';
  published_at: string | null;
  last_active_at: string | null;
  loaded: boolean;
}

/** api/observability.py summarise() — null percentile == too few samples */
export interface LatencySummary {
  count: number;
  mean_ms: number | null;
  p50_ms: number | null;
  p95_ms: number | null;
  p99_ms: number | null;
}

/** api/routes_health.py GET /status */
export interface SystemStatusResponse {
  ready: boolean;
  version: string;
  mode: RuntimeMode;
  components: ComponentStatus[];
  knowledge_base: { active: KBActive | null; versions: KBVersion[] };
  latency: {
    requests_recorded: number;
    window: number;
    min_samples: { p95: number; p99: number };
    stages: Record<string, LatencySummary>;
  };
}

/** api/models.py AdminConfigResponse */
export interface AdminConfig {
  execution_mode: string;
  retrieval_mode: string;
  reranker_enabled: boolean;
  nli_enabled: boolean;
  llm_provider: string;
  llm_model_name: string;
  max_corrective_attempts: number;
  offline: boolean;
  qdrant_mode: string | null;
  decision_thresholds: Record<string, number>;
  effective_routing: Record<string, unknown>;
}

// ── /ingest ─────────────────────────────────────────────────────────────────

/** ingestion/job_store.py job row */
export interface IngestJob {
  job_id: string;
  source_id: string | null;
  source_type: string;
  filename: string | null;
  source_url: string | null;
  content_hash: string | null;
  stage: string;
  status: string;
  error_message: string | null;
  corpus_version_id: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

/** ingestion/job_store.py source row */
export interface IngestSource {
  source_id: string;
  display_name: string;
  source_url: string;
  sync_interval_seconds: number;
  next_sync_at: string | null;
  sync_state: string;
  consecutive_failures: number;
  last_sync_at: string | null;
  last_error: string | null;
  is_active: boolean;
}

/** api/routes_ingest.py GET /ingest/status */
export interface IngestStatus {
  active_version: KBActive | null;
  serving_version_id: string | null;
  pointer_version_id: string | null;
  consistent: boolean;
  versions: KBVersion[];
  pending_jobs: number;
  active_sources: number;
  failed_jobs: number;
}

/** api/routes_ingest.py POST /ingest/upload */
export interface UploadResponse {
  job_id: string;
  filename?: string;
  size_bytes?: number;
  message: string;
  duplicate: boolean;
}

/** Optional provenance form fields of POST /ingest/upload */
export interface UploadMetadata {
  title?: string;
  source_type?: string;
  date?: string;
  jurisdiction?: string;
  population?: string;
  dosage_context?: string;
  /** declared lifecycle: current | superseded | historical | withdrawn */
  status?: string;
  /** comma-separated doc_ids this document replaces */
  supersedes?: string;
  effective_date?: string;
}

/** schemas/common.py SourceType */
export const SOURCE_TYPES = [
  'clinical_guideline',
  'regulatory_document',
  'peer_reviewed_literature',
  'drug_label',
  'institutional_policy',
  'other',
] as const;

// ── /eval ───────────────────────────────────────────────────────────────────

/** api/models_enriched.py EvalRunSummaryModel */
export interface EvalRun {
  run_id: string;
  baseline_id: string;
  dataset_id: string;
  dataset_version: string;
  split: string;
  corpus_fingerprint: string;
  corpus_condition: string;
  execution_mode: string;
  mock_llm: boolean;
  total_cases: number;
  offered: number;
  accepted: number;
  completed: number;
  error: number;
  skipped: number;
  rejected_overload: number;
  cancelled_deadline: number;
  not_offered_deadline: number;
  baseline_init_seconds: number;
  total_run_wall_seconds: number;
  start_utc: string;
  end_utc: string;
  results_jsonl_path: string;
  integrity_error: string | null;
  limitations: string;
}

/** api/models_enriched.py EvalRunListModel */
export interface EvalRunList {
  runs: EvalRun[];
  total: number;
}

// ── UI navigation ───────────────────────────────────────────────────────────

export type ActiveView = 'chat' | 'knowledge' | 'status' | 'evaluation';
