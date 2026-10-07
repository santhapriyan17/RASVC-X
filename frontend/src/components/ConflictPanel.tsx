// Conflicts tab: how pairs of evidence items relate to each other (M8).

import type { ValidationSummary } from '../types';
import { Pill, pct, toneFor } from './ui';

export function ConflictPanel({ validation, onCite }: {
  validation: ValidationSummary | null;
  onCite: (evidenceId: string) => void;
}) {
  if (!validation) {
    return <p className="muted">Evidence validation did not run for this question.</p>;
  }
  // Show everything that is not plainly compatible first.
  const notable = validation.resolutions.filter((r) => r.relationship !== 'compatible');
  const compatible = validation.resolutions.length - notable.length;
  return (
    <div>
      <p className="small">
        <Pill tone={validation.genuine_conflict_count > 0 ? 'bad' : 'ok'}>
          {validation.genuine_conflict_count} genuine conflict(s)
        </Pill>{' '}
        <Pill tone={validation.unresolved_count > 0 ? 'warn' : 'idle'}>
          {validation.unresolved_count} unresolved
        </Pill>{' '}
        <span className="muted">
          · {validation.candidates_generated} evidence pair(s) checked · {compatible} compatible ·{' '}
          {validation.nli_calls_used} NLI call(s)
          {validation.nli_failures > 0 && ` · ${validation.nli_failures} NLI failure(s)`}
        </span>
      </p>
      {notable.length === 0 && (
        <p className="muted">No conflicting or context-divergent evidence was found.</p>
      )}
      {notable.map((r) => (
        <div key={r.candidate_id} className="item">
          <div className="item-head">
            <Pill tone={toneFor(r.relationship)}>{r.relationship}</Pill>
            {r.evidence_ids.map((id) => (
              <button key={id} className="cite" onClick={() => onCite(id)}>{id}</button>
            ))}
            <span className="muted small">
              {r.validation_stage && `decided by ${r.validation_stage.replace(/_/g, ' ')} · `}
              confidence {pct(r.confidence)}
            </span>
          </div>
          {r.rationale && <div className="small muted">{r.rationale}</div>}
        </div>
      ))}
    </div>
  );
}
