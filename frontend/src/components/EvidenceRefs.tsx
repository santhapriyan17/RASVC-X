// Evidence tab: the evidence items the decision was based on.

import type { Evidence } from '../types';
import { Pill, toneFor, type Tone } from './ui';

// A retrieved passage is not support: only SUPPORTING items back the answer.
const ROLE_TONE: Record<Evidence['role'], Tone> = {
  SUPPORTING: 'ok',
  RELEVANT: 'info',
  RETRIEVED: 'idle',
  IRRELEVANT: 'idle',
  CONTRADICTORY: 'bad',
  SUPERSEDED: 'bad',
};

function sourceLine(e: Evidence): string {
  const parts: string[] = [];
  if (e.filename) parts.push(e.filename);
  if (e.section) parts.push(`section "${e.section}"`);
  if (e.page !== null) parts.push(`page ${e.page}`);
  return parts.join(' · ');
}

export function EvidenceRefs({ evidence, kbVersion, focusId }: {
  evidence: Evidence[];
  kbVersion: string | null;
  focusId: string | null;
}) {
  if (evidence.length === 0) {
    return <p className="muted">No evidence was retrieved for this question.</p>;
  }
  return (
    <div>
      <p className="muted small">
        {evidence.length} item(s) from knowledge-base version{' '}
        <span className="mono">{kbVersion ?? 'unknown'}</span>, ordered by reranker score.{' '}
        {evidence.filter((e) => e.supports_answer).length} support the answer; retrieved items
        that do not are listed for transparency only.
      </p>
      {evidence.map((e) => (
        <div key={e.evidence_id} id={`evidence-${e.evidence_id}`}
             className={`item ${focusId === e.evidence_id ? 'focus' : ''}`}>
          <div className="item-head">
            <span className="cite">{e.evidence_id}</span>
            <strong>{e.title ?? e.doc_id ?? e.chunk_id}</strong>
            <Pill tone={ROLE_TONE[e.role] ?? 'idle'} title={e.role_reason ?? undefined}>
              {e.role.toLowerCase()}
            </Pill>
            {e.temporal_status !== 'UNKNOWN' && (
              <Pill tone={e.temporal_status === 'CURRENT' ? 'ok' : 'bad'} title={e.temporal_reason ?? undefined}>
                {e.temporal_status.toLowerCase()}
              </Pill>
            )}
            {e.cited ? <Pill tone="ok">cited</Pill> : <Pill>not cited</Pill>}
            <Pill tone="info" title={e.authority_tier ? `authority ${e.authority_tier}` : undefined}>
              {e.provenance.source_type.replace(/_/g, ' ')}
            </Pill>
            <span className="spacer" />
            <span className="muted small" title="cross-encoder score (not a probability)">
              rerank {e.rerank_score === null ? 'n/a' : e.rerank_score.toFixed(2)}
            </span>
          </div>
          <div className="muted small">
            doc <span className="mono">{e.doc_id ?? 'unknown'}</span>
            {sourceLine(e) ? ` · ${sourceLine(e)}` : ''}
            {e.source_url ? <> · <a href={e.source_url} target="_blank" rel="noreferrer noopener">source</a></> : null}
          </div>
          {e.lifecycle && (
            <div className="muted small">
              lifecycle: {e.lifecycle.status ?? 'undeclared'}
              {e.lifecycle.superseded_by ? ` · superseded by ${e.lifecycle.superseded_by}` : ''}
              {e.lifecycle.effective_date ? ` · effective ${e.lifecycle.effective_date}` : ''}
              {e.lifecycle.version ? ` · version ${e.lifecycle.version}` : ''}
            </div>
          )}
          <div className="muted small">
            date {e.provenance.date ?? 'unknown'} · jurisdiction {e.provenance.jurisdiction ?? 'unknown'} ·
            population {e.provenance.population ?? 'unknown'} · dosage context {e.provenance.dosage_context ?? 'unknown'}
          </div>
          {e.analysis && (
            <div className="small" style={{ margin: '4px 0' }}>
              <span className="muted">vs. question context: </span>
              {(['temporal', 'jurisdiction', 'population', 'dosage_context'] as const).map((f) => (
                <Pill key={f} tone={toneFor(e.analysis![f])} title={e.analysis![f]}>
                  {f.replace('_', ' ')}: {e.analysis![f].replace('known_', '')}
                </Pill>
              ))}{' '}
              <span className="muted">source quality {e.analysis.quality_score.toFixed(2)}</span>
            </div>
          )}
          <div className="evidence-text">{e.text}</div>
        </div>
      ))}
    </div>
  );
}
