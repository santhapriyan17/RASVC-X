// frontend/src/api/client.ts
// ---------------------------------------------------------------------------
// RASVC-X API client — every backend endpoint the UI calls, in one place.
// ---------------------------------------------------------------------------

import type {
  AdminConfig,
  EvalRunList,
  FeedbackRequest,
  IngestJob,
  IngestSource,
  IngestStatus,
  QueryContext,
  QueryRequest,
  QueryResponse,
  Readiness,
  SystemStatusResponse,
  UploadMetadata,
  UploadResponse,
} from '../types';

const API_BASE: string = import.meta.env.VITE_API_BASE ?? '';

/** api.max_query_chars in the backend config (the server enforces it). */
export const MAX_QUERY_CHARS = 2048;

/**
 * Bearer token for deployments with api.require_auth=true. Kept in memory /
 * sessionStorage only; never baked into the bundle.
 */
function authHeader(): Record<string, string> {
  const token = sessionStorage.getItem('rasvcx_auth_token') ?? '';
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function failure(res: Response): Promise<ApiError> {
  let detail = res.statusText;
  try {
    const body = await res.json();
    const d = body?.detail ?? body?.error;
    if (typeof d === 'string') detail = d;
    else if (d !== undefined) detail = JSON.stringify(d);
  } catch {
    // non-JSON error body: keep the status text
  }
  return new ApiError(res.status, detail);
}

async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
      ...authHeader(),
      ...(init.headers as Record<string, string> | undefined),
    },
  });
  if (!res.ok) throw await failure(res);
  if (res.status === 204) return undefined as unknown as T;
  return res.json() as Promise<T>;
}

// ── Health / status ─────────────────────────────────────────────────────────

/** GET /ready returns 503 with the same body when not ready. */
export async function fetchReadiness(): Promise<Readiness> {
  const res = await fetch(`${API_BASE}/ready`);
  if (res.status !== 200 && res.status !== 503) throw await failure(res);
  return res.json() as Promise<Readiness>;
}

export function fetchStatus(): Promise<SystemStatusResponse> {
  return apiFetch<SystemStatusResponse>('/status');
}

export function fetchAdminConfig(): Promise<AdminConfig> {
  return apiFetch<AdminConfig>('/admin/config');
}

// ── Query ───────────────────────────────────────────────────────────────────

export function submitQuery(
  query: string,
  context?: QueryContext,
): Promise<QueryResponse> {
  const body: QueryRequest = { query, enriched: true };
  if (context && Object.keys(context).length > 0) body.context = context;
  return apiFetch<QueryResponse>('/query', {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

// ── Feedback ────────────────────────────────────────────────────────────────

export function submitFeedback(
  req: FeedbackRequest,
): Promise<{ feedback_id: string; recorded: boolean }> {
  return apiFetch('/feedback', { method: 'POST', body: JSON.stringify(req) });
}

// ── Evaluation ──────────────────────────────────────────────────────────────

export function fetchEvalRuns(): Promise<EvalRunList> {
  return apiFetch<EvalRunList>('/eval/runs');
}

// ── Ingestion / knowledge base ──────────────────────────────────────────────

export async function uploadFile(
  file: File,
  metadata: UploadMetadata = {},
): Promise<UploadResponse> {
  const form = new FormData();
  form.append('file', file);
  for (const [key, value] of Object.entries(metadata)) {
    if (value && value.trim()) form.append(key, value.trim());
  }
  // No Content-Type header: the browser sets the multipart boundary.
  const res = await fetch(`${API_BASE}/ingest/upload`, {
    method: 'POST',
    headers: authHeader(),
    body: form,
  });
  if (!res.ok) throw await failure(res);
  return res.json() as Promise<UploadResponse>;
}

export function ingestUrl(url: string): Promise<UploadResponse> {
  return apiFetch<UploadResponse>('/ingest/url', {
    method: 'POST',
    body: JSON.stringify({ url }),
  });
}

export function fetchJobs(): Promise<{ jobs: IngestJob[]; total: number }> {
  return apiFetch('/ingest/jobs?limit=50');
}

export function cancelJob(jobId: string): Promise<void> {
  return apiFetch<void>(`/ingest/jobs/${encodeURIComponent(jobId)}`, {
    method: 'DELETE',
  });
}

export function fetchSources(): Promise<{ sources: IngestSource[] }> {
  return apiFetch('/ingest/sources');
}

export function createSource(
  display_name: string,
  source_url: string,
  sync_interval_seconds: number,
): Promise<IngestSource> {
  return apiFetch<IngestSource>('/ingest/sources', {
    method: 'POST',
    body: JSON.stringify({ display_name, source_url, sync_interval_seconds }),
  });
}

export function syncSource(
  sourceId: string,
): Promise<{ source_id: string; message: string }> {
  return apiFetch(`/ingest/sources/${encodeURIComponent(sourceId)}/sync`, {
    method: 'POST',
  });
}

export function deleteSource(sourceId: string): Promise<void> {
  return apiFetch<void>(`/ingest/sources/${encodeURIComponent(sourceId)}`, {
    method: 'DELETE',
  });
}

export function fetchIngestStatus(): Promise<IngestStatus> {
  return apiFetch<IngestStatus>('/ingest/status');
}
