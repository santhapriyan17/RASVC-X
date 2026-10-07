// Trace tab: what the pipeline actually executed for this request.

import type { QueryResponse } from '../types';
import { ModuleStatus } from './ModuleStatus';
import { Pill, ms, pct, toneFor } from './ui';

export function DecisionTrace({ response }: { response: QueryResponse }) {
  const r = response;
  const maxMs = Math.max(1, ...r.trace.map((t) => t.elapsed_ms));
  const retrieval = r.retrieval;
  return (
    <div>
      <dl className="kv">
        <dt>Decision</dt>
        <dd>{r.decision} <span className="muted">({r.rationale ?? 'no rationale'})</span></dd>
        <dt>Confidence</dt>
        <dd>
          {pct(r.confidence)}{' '}
          <span className="muted">
            · calibration: {r.calibration_status ?? 'not computed'}
            {r.calibration_status === 'uncalibrated' && ' (raw score, no fitted calibrator)'}
          </span>
        </dd>
        <dt>Risk</dt>
        <dd>
          {r.risk_profile
            ? <>score {r.risk_profile.overall_risk_score.toFixed(2)} · depth {r.risk_profile.validation_depth}
                {r.risk_profile.safety_floor_forced && ' · safety floor forced'} · NLI allowance {r.risk_profile.nli_call_allowance}</>
            : 'not computed'}
        </dd>
        <dt>Knowledge base</dt>
        <dd className="mono">{r.kb_version_id ?? 'unknown'}</dd>
        <dt>Retrieval</dt>
        <dd>
          {String(retrieval.mode ?? 'n/a')} · BM25 {String(retrieval.bm25_hits ?? '—')} hit(s)
          {retrieval.dense_hits !== undefined && ` · Qdrant ${String(retrieval.dense_hits)} hit(s)`}
          {retrieval.rrf_fused !== undefined && ` · RRF ${String(retrieval.rrf_fused)} fused`}
          {retrieval.rrf_from_both !== undefined && ` (${String(retrieval.rrf_from_both)} found by both)`}
          {retrieval.targeted_added !== undefined && ` · targeted +${String(retrieval.targeted_added)}`}
        </dd>
        <dt>Model</dt>
        <dd>
          {r.mode?.mock_llm ? 'offline stub (not a real model)' : (r.mode?.llm_model ?? 'unknown')}
          <span className="muted"> · {r.nli_calls} NLI call(s) · {r.corrective_attempts} corrective attempt(s)</span>
        </dd>
        <dt>Total latency</dt>
        <dd>{ms(r.total_latency_ms)}</dd>
      </dl>

      {r.pipeline_error && (
        <p className="error small">
          Stopped at <strong>{r.pipeline_error.stage}</strong>: {r.pipeline_error.message}
        </p>
      )}

      <h3>Modules</h3>
      <ModuleStatus modules={r.modules} />

      <h3>Executed stages</h3>
      <table>
        <thead>
          <tr><th>Stage</th><th>Status</th><th className="num">Time</th><th></th><th>Detail</th></tr>
        </thead>
        <tbody>
          {r.trace.map((t, i) => (
            <tr key={`${t.stage}-${i}`}>
              <td className="mono">
                {t.attempt > 0 && <span className="muted">retry {t.attempt} · </span>}{t.stage}
              </td>
              <td><Pill tone={toneFor(t.status)}>{t.status}</Pill></td>
              <td className="num">{t.status === 'skipped' ? '—' : ms(t.elapsed_ms)}</td>
              <td><div className="bar"><span style={{ width: `${(t.elapsed_ms / maxMs) * 100}%` }} /></div></td>
              <td className="small muted">{t.detail ?? ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
