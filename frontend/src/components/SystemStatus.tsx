// System Status view: what the running backend process reports about itself.
// Read-only: the configuration is fixed at startup, so there are no toggles
// here that could disagree with what the backend is really doing.

import { useCallback, useEffect, useState } from 'react';
import { fetchAdminConfig, fetchStatus } from '../api/client';
import type { AdminConfig, SystemStatusResponse } from '../types';
import { Pill, errorMessage, ms, toneFor, when } from './ui';

const STAGE_ORDER = [
  'risk_routing', 'hybrid_retrieval', 'bm25_retrieval', 'dense_retrieval', 'rrf_fusion',
  'reranking', 'sufficiency_gate', 'targeted_retrieval', 'provenance_context',
  'atomic_claim_extraction', 'verified_context', 'candidate_generation',
  'deterministic_validation', 'contextual_validation', 'selective_nli',
  'evidence_resolution', 'generation', 'post_generation_verification',
  'confidence_estimation', 'decision', 'serialization', 'total',
];

export function SystemStatus() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [config, setConfig] = useState<AdminConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [s, c] = await Promise.all([fetchStatus(), fetchAdminConfig()]);
      setStatus(s);
      setConfig(c);
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (error) {
    return (
      <div className="panel error">
        Could not read system status: {error}{' '}
        <button onClick={() => void load()}>Retry</button>
      </div>
    );
  }
  if (!status || !config) return <div className="panel muted">Loading system status…</div>;

  const stages = status.latency.stages;
  const names = [
    ...STAGE_ORDER.filter((n) => n in stages),
    ...Object.keys(stages).filter((n) => !STAGE_ORDER.includes(n)).sort(),
  ];
  const kb = status.knowledge_base;

  return (
    <div>
      <div className="panel">
        <div className="row">
          <h2 style={{ margin: 0 }}>Runtime</h2>
          <Pill tone={status.ready ? 'ok' : 'bad'}>{status.ready ? 'ready' : 'not ready'}</Pill>
          {status.mode.offline
            ? <Pill tone="warn">OFFLINE TEST — test doubles, not real models</Pill>
            : <Pill tone="info">ONLINE</Pill>}
          <span className="spacer" />
          <button disabled={loading} onClick={() => void load()}>{loading ? 'Refreshing…' : 'Refresh'}</button>
        </div>
        <dl className="kv" style={{ marginTop: 10 }}>
          <dt>Execution mode</dt><dd className="mono">{status.mode.execution_mode}</dd>
          <dt>LLM</dt>
          <dd>{status.mode.mock_llm ? 'stub (offline test)' : `${status.mode.llm_provider} · ${status.mode.llm_model}`}</dd>
          <dt>Retrieval</dt>
          <dd>{config.retrieval_mode}{config.qdrant_mode && ` · Qdrant mode: ${config.qdrant_mode}`}</dd>
          <dt>Reranker / NLI</dt>
          <dd>{config.reranker_enabled ? 'enabled' : 'disabled'} / {config.nli_enabled ? 'enabled' : 'disabled'}</dd>
          <dt>Corrective attempts</dt><dd>max {config.max_corrective_attempts}</dd>
          <dt>API version</dt><dd>{status.version}</dd>
        </dl>
        <p className="muted small">
          Configuration is read at startup and cannot be changed from this page. Edit the config
          file and restart the server to change a mode, model or threshold.
        </p>
      </div>

      <div className="panel">
        <h2>Components</h2>
        <table>
          <thead><tr><th>Component</th><th>State</th><th>Detail</th></tr></thead>
          <tbody>
            {status.components.map((c) => (
              <tr key={c.name}>
                <td className="mono">{c.name}</td>
                <td><Pill tone={toneFor(c.state)}>{c.state}</Pill></td>
                <td className="small muted">{c.detail ?? ''}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted small">
          The LLM is reported as “configured”, not “reachable”: checking it would cost a billed
          request. Whether generation actually worked is shown per answer in the chat’s Trace tab.
        </p>
      </div>

      <div className="panel">
        <h2>Knowledge-base versions</h2>
        {kb.active ? (
          <p className="small">
            Serving <span className="mono">{kb.active.version_id}</span> · {kb.active.doc_count} documents ·{' '}
            {kb.active.chunk_count} chunks · Qdrant{' '}
            <span className="mono">{kb.active.qdrant_collection ?? 'none'}</span>
          </p>
        ) : <p className="error">No knowledge-base version is loaded.</p>}
        <table>
          <thead><tr><th>Version</th><th>Status</th><th className="num">Active requests</th><th>Published</th><th>Last used</th></tr></thead>
          <tbody>
            {kb.versions.map((v) => (
              <tr key={v.version_id}>
                <td className="mono">{v.version_id}</td>
                <td><Pill tone={v.status === 'active' ? 'ok' : 'idle'}>{v.status}</Pill></td>
                <td className="num">{v.active_request_count}</td>
                <td>{when(v.published_at)}</td>
                <td>{when(v.last_active_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <h2>Measured latency</h2>
        <p className="muted small">
          Measured over the last {status.latency.requests_recorded} request(s) served by this process
          (window {status.latency.window}). p95 needs {status.latency.min_samples.p95} samples and p99
          needs {status.latency.min_samples.p99}; “—” means not enough samples yet.
        </p>
        {names.length === 0 ? <p className="muted">No requests have been served yet.</p> : (
          <table>
            <thead><tr><th>Stage</th><th className="num">Samples</th><th className="num">Mean</th><th className="num">p50</th><th className="num">p95</th><th className="num">p99</th></tr></thead>
            <tbody>
              {names.map((n) => (
                <tr key={n}>
                  <td className="mono">{n}</td>
                  <td className="num">{stages[n].count}</td>
                  <td className="num">{ms(stages[n].mean_ms)}</td>
                  <td className="num">{ms(stages[n].p50_ms)}</td>
                  <td className="num">{ms(stages[n].p95_ms)}</td>
                  <td className="num">{ms(stages[n].p99_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <h2>Decision thresholds</h2>
        <dl className="kv">
          {Object.entries(config.decision_thresholds).map(([k, v]) => (
            <span key={k} style={{ display: 'contents' }}>
              <dt className="mono">{k}</dt><dd>{v}</dd>
            </span>
          ))}
        </dl>
      </div>
    </div>
  );
}
