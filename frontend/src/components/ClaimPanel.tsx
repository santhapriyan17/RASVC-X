// Claims tab: every sentence of the generated answer and how it was verified.

import type { VerificationSummary } from '../types';
import { Pill, pct, toneFor } from './ui';

export function ClaimPanel({ verification, onCite }: {
  verification: VerificationSummary | null;
  onCite: (evidenceId: string) => void;
}) {
  if (!verification) {
    return (
      <p className="muted">
        No answer was generated, so there are no claims to verify.
      </p>
    );
  }
  const claims = verification.claim_results;
  return (
    <div>
      <p className="small">
        Answer verdict{' '}
        <Pill tone={toneFor(verification.answer_verdict)}>
          {verification.answer_verdict.replace(/_/g, ' ')}
        </Pill>{' '}
        <span className="muted">
          · {claims.length} claim(s) · {verification.semantic_verification_calls} NLI call(s)
          {verification.safety_critical_failure_count > 0 &&
            ` · ${verification.safety_critical_failure_count} safety-critical failure(s)`}
          {verification.budget_exhausted && ' · verification budget exhausted'}
        </span>
      </p>
      {claims.length === 0 && <p className="muted">The answer contained no verifiable claims.</p>}
      {claims.map((c) => (
        <div key={c.claim_id} className="item">
          <div className="item-head">
            <Pill tone={toneFor(c.label)}>{c.label.replace(/_/g, ' ')}</Pill>
            {c.is_safety_critical && <Pill tone="warn">safety-critical</Pill>}
            <span className="muted small">
              checked by {c.stage.replace(/_/g, ' ')} · confidence {pct(c.confidence)}
            </span>
          </div>
          <div>{c.text || <span className="muted">(claim text unavailable)</span>}</div>
          <div className="small muted" style={{ marginTop: 4 }}>
            citation{' '}
            {c.citation_status && (
              <Pill tone={toneFor(c.citation_status)}>{c.citation_status}</Pill>
            )}{' '}
            {c.cited_evidence_ids.map((id) => (
              <button key={id} className="cite" onClick={() => onCite(id)}>{id}</button>
            ))}
            {c.contradicting_item_ids.length > 0 && (
              <> · contradicted by{' '}
                {c.contradicting_item_ids.map((id) => (
                  <button key={id} className="cite unknown" onClick={() => onCite(id)}>{id}</button>
                ))}
              </>
            )}
          </div>
          {c.rationale && <div className="small muted">{c.rationale}</div>}
        </div>
      ))}
    </div>
  );
}
