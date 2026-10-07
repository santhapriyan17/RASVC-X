// Chat: the primary RASVC-X screen.
//
// A question goes to POST /query (enriched). The response is shown as an
// answer with clickable citations plus four inspection tabs — Claims,
// Conflicts, Evidence, Trace — all rendered from the backend response.
// Answer text is never logged to the console.

import { useCallback, useEffect, useRef, useState } from 'react';
import { MAX_QUERY_CHARS, submitFeedback, submitQuery } from '../api/client';
import type { ChatEntry, QueryContext, QueryResponse } from '../types';
import { ClaimPanel } from './ClaimPanel';
import { ConflictPanel } from './ConflictPanel';
import { DecisionTrace } from './DecisionTrace';
import { EvidenceRefs } from './EvidenceRefs';
import { DecisionBadge, Pill, errorMessage, ms, pct } from './ui';

type Tab = 'answer' | 'claims' | 'conflicts' | 'evidence' | 'trace';

const EXAMPLES = [
  'What infections is azithromycin used to treat?',
  'What crystalloid fluid bolus volume is recommended for sepsis with hypotension?',
  'What are the contraindications of sertraline?',
];

const CITATION_RE = /\[([A-Za-z0-9_-]{1,64}(?:\s*,\s*[A-Za-z0-9_-]{1,64})*)\]/g;

/** Render answer text with each [E1] / [E1, E2] marker as clickable chips. */
function AnswerText({ text, known, onCite }: {
  text: string;
  known: Set<string>;
  onCite: (id: string) => void;
}) {
  const parts: (string | JSX.Element)[] = [];
  let last = 0;
  let key = 0;
  for (const match of text.matchAll(CITATION_RE)) {
    const start = match.index ?? 0;
    if (start > last) parts.push(text.slice(last, start));
    for (const id of match[1].split(',').map((s) => s.trim())) {
      parts.push(
        known.has(id)
          ? <button key={key++} className="cite" onClick={() => onCite(id)} title="Show evidence">{id}</button>
          : <span key={key++} className="cite unknown" title="This citation does not match any retrieved evidence">{id}</span>,
      );
    }
    last = start + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <div className="answer-text">{parts}</div>;
}

function Result({ entry, onFeedback }: {
  entry: ChatEntry;
  onFeedback: (entry: ChatEntry, rating: 'up' | 'down') => void;
}) {
  const [tab, setTab] = useState<Tab>('answer');
  const [focusId, setFocusId] = useState<string | null>(null);
  const r = entry.response as QueryResponse;

  const cite = useCallback((id: string) => {
    setFocusId(id);
    setTab('evidence');
    window.setTimeout(() => {
      document.getElementById(`evidence-${id}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 50);
  }, []);

  const conflicts = r.validation
    ? r.validation.genuine_conflict_count + r.validation.unresolved_count
    : 0;
  const known = new Set(r.evidence.map((e) => e.evidence_id));

  return (
    <div className="panel answer-card">
      <div className="answer-head">
        <DecisionBadge decision={r.decision} />
        {r.calibration_status === 'calibrated' ? (
          <span className="muted small" title="Calibrated estimate of the probability the answer is correct">
            est. correctness {pct(r.confidence)}
          </span>
        ) : (
          <span className="muted small"
                title={r.calibration_status === 'invalidated'
                  ? 'A calibration exists but does not apply to this KB/config/model; this is the raw heuristic score, not a probability.'
                  : 'Heuristic reliability score in [0, 1]. It is NOT calibrated and is not a probability.'}>
            raw score {r.confidence.toFixed(2)} ({(r.calibration_status ?? 'uncalibrated').toUpperCase()})
          </span>
        )}
        {r.mode?.offline && <Pill tone="warn" title="Produced by offline test doubles">OFFLINE TEST</Pill>}
        {r.request_class === 'PROVIDER_ERROR' && (
          <Pill tone="bad" title="The language-model provider failed; this is not an evidence-based abstention">
            provider error
          </Pill>
        )}
        {r.request_class === 'SYSTEM_ERROR' && (
          <Pill tone="bad" title="A pipeline stage failed; this is not an evidence-based abstention">
            system error
          </Pill>
        )}
        {r.degraded && r.request_class !== 'PROVIDER_ERROR' && r.request_class !== 'SYSTEM_ERROR' && (
          <Pill tone="bad" title="A pipeline component produced no signal for this request">degraded</Pill>
        )}
        {r.cached && <Pill tone="warn" title="Served from the answer cache for this KB version; no pipeline stage ran now">cached</Pill>}
        {r.kb?.kb_source === 'seed_fallback' && (
          <Pill tone="warn" title="No published knowledge-base version is active; the seed corpus answered">seed KB</Pill>
        )}
        <span className="spacer" />
        <span className="muted small">
          {ms(r.total_latency_ms)} · KB <span className="mono">{r.kb_version_id ?? 'unknown'}</span>
          {' '}({r.kb?.kb_source?.replace('_', ' ') ?? 'source unknown'})
        </span>
      </div>

      <div className="tabs" role="tablist">
        {([
          ['answer', 'Answer'],
          ['claims', `Claims${r.verification ? ` (${r.verification.claim_results.length})` : ''}`],
          ['conflicts', `Conflicts${conflicts ? ` (${conflicts})` : ''}`],
          ['evidence', `Evidence (${r.evidence.length})`],
          ['trace', 'Trace'],
        ] as [Tab, string][]).map(([id, label]) => (
          <button key={id} role="tab" aria-selected={tab === id}
                  className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>
            {label}
          </button>
        ))}
      </div>

      {tab === 'answer' && (
        <div>
          {r.has_answer ? (
            <AnswerText text={r.answer} known={known} onCite={cite} />
          ) : (
            <div className="withheld">
              <strong>No answer is given for this question.</strong>
              <div className="muted small">
                {r.pipeline_error
                  ? `The pipeline stopped at "${r.pipeline_error.stage}": ${r.pipeline_error.message}`
                  : (r.rationale ?? 'The evidence did not support a reliable answer.')}
              </div>
            </div>
          )}
          {r.warnings.length > 0 && (
            <ul className="warnings">{r.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
          )}
          <div className="disclaimer">{r.limitations}</div>
          <div className="row" style={{ marginTop: 8 }}>
            <span className="muted small">Was this response appropriate?</span>
            <button disabled={!!entry.feedback} className={entry.feedback === 'up' ? 'primary' : ''}
                    onClick={() => onFeedback(entry, 'up')}>Yes</button>
            <button disabled={!!entry.feedback} className={entry.feedback === 'down' ? 'primary' : ''}
                    onClick={() => onFeedback(entry, 'down')}>No</button>
            {entry.feedback && <span className="muted small">Recorded for offline review.</span>}
          </div>
        </div>
      )}
      {tab === 'claims' && <ClaimPanel verification={r.verification} onCite={cite} />}
      {tab === 'conflicts' && <ConflictPanel validation={r.validation} onCite={cite} />}
      {tab === 'evidence' && (
        <EvidenceRefs evidence={r.evidence} kbVersion={r.kb_version_id} focusId={focusId} />
      )}
      {tab === 'trace' && <DecisionTrace response={r} />}
    </div>
  );
}

export function Chat() {
  const [entries, setEntries] = useState<ChatEntry[]>([]);
  const [draft, setDraft] = useState('');
  const [context, setContext] = useState<QueryContext>({});
  const [showContext, setShowContext] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const busy = entries.some((e) => e.pending);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' });
  }, [entries.length]);

  const ask = useCallback(async (question: string) => {
    const q = question.trim();
    if (!q || q.length > MAX_QUERY_CHARS) return;
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const ctx = Object.fromEntries(
      Object.entries(context).filter(([, v]) => v && v.trim()),
    ) as QueryContext;
    setEntries((prev) => [...prev, { id, question: q, context: ctx, pending: true }]);
    setDraft('');
    try {
      const response = await submitQuery(q, ctx);
      setEntries((prev) => prev.map((e) => (e.id === id ? { ...e, pending: false, response } : e)));
    } catch (err) {
      setEntries((prev) => prev.map((e) => (
        e.id === id ? { ...e, pending: false, error: errorMessage(err) } : e
      )));
    }
  }, [context]);

  const feedback = useCallback(async (entry: ChatEntry, rating: 'up' | 'down') => {
    if (!entry.response) return;
    try {
      await submitFeedback({
        query_id: entry.response.query_id,
        rating,
        decision: entry.response.decision,
        kb_version_id: entry.response.kb_version_id,
      });
      setEntries((prev) => prev.map((e) => (e.id === entry.id ? { ...e, feedback: rating } : e)));
    } catch (err) {
      window.alert(`Feedback could not be recorded: ${errorMessage(err)}`);
    }
  }, []);

  const setCtx = (key: keyof QueryContext, value: string) =>
    setContext((prev) => ({ ...prev, [key]: value }));

  return (
    <div>
      {entries.length === 0 && (
        <div className="panel">
          <h2>Ask a medical question</h2>
          <p className="muted">
            The answer is generated only from retrieved documents, then every sentence is checked
            against the evidence it cites. When the evidence is insufficient or conflicting, the
            system declines to answer and shows why.
          </p>
        </div>
      )}

      {entries.map((entry) => (
        <div key={entry.id} className="entry">
          <div className="question">{entry.question}</div>
          {entry.context && Object.keys(entry.context).length > 0 && (
            <div className="muted small">
              context: {Object.entries(entry.context).map(([k, v]) => `${k.replace('_', ' ')} = ${v}`).join(' · ')}
            </div>
          )}
          {entry.pending && (
            <div className="panel answer-card muted">
              Retrieving, validating evidence, generating and verifying…
            </div>
          )}
          {entry.error && (
            <div className="panel answer-card error">Request failed: {entry.error}</div>
          )}
          {entry.response && <Result entry={entry} onFeedback={feedback} />}
        </div>
      ))}
      <div ref={bottom} />

      <div className="composer">
        <div className="panel">
          <textarea
            rows={3}
            value={draft}
            maxLength={MAX_QUERY_CHARS}
            placeholder="Ask a question about the documents in the knowledge base…"
            aria-label="Question"
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!busy) void ask(draft);
              }
            }}
          />
          {showContext && (
            <div className="grid" style={{ marginTop: 8 }}>
              <label>Population
                <input value={context.population ?? ''} placeholder="e.g. adults"
                       onChange={(e) => setCtx('population', e.target.value)} />
              </label>
              <label>Jurisdiction
                <input value={context.jurisdiction ?? ''} placeholder="e.g. US"
                       onChange={(e) => setCtx('jurisdiction', e.target.value)} />
              </label>
              <label>Dosage context
                <input value={context.dosage_context ?? ''} placeholder="e.g. oral"
                       onChange={(e) => setCtx('dosage_context', e.target.value)} />
              </label>
            </div>
          )}
          <div className="row" style={{ marginTop: 8 }}>
            <button className="primary" disabled={busy || !draft.trim()} onClick={() => void ask(draft)}>
              {busy ? 'Working…' : 'Ask'}
            </button>
            <button className="link" onClick={() => setShowContext((v) => !v)}>
              {showContext ? 'Hide clinical context' : 'Add clinical context'}
            </button>
            <span className="spacer" />
            <span className="muted small">{draft.length}/{MAX_QUERY_CHARS}</span>
          </div>
          {entries.length === 0 && (
            <div className="examples">
              {EXAMPLES.map((ex) => (
                <button key={ex} disabled={busy} onClick={() => void ask(ex)}>{ex}</button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
