// Per-response module execution states, as reported by the backend for
// THIS request (derived there from the execution trace, not from config).

import type { ModuleState } from '../types';
import { Pill, toneFor } from './ui';

const LABELS: Record<string, string> = {
  risk_routing: 'Risk routing',
  bm25: 'BM25 retrieval',
  qdrant: 'Qdrant dense retrieval',
  rrf: 'RRF fusion',
  reranker: 'Cross-encoder reranking',
  evidence_sufficiency: 'Evidence sufficiency',
  provenance: 'Provenance analysis',
  claim_extraction: 'Atomic claim extraction',
  deterministic_validation: 'Deterministic validation',
  contextual_validation: 'Contextual validation',
  nli: 'Selective NLI',
  conflict_resolution: 'Conflict resolution',
  generation: 'Grounded generation',
  post_generation_verification: 'Post-generation verification',
  confidence: 'Confidence',
  calibration: 'Calibration',
  decision_engine: 'Decision engine',
};

const STATE_TEXT: Record<ModuleState, string> = {
  executed: 'executed',
  failed: 'failed',
  skipped: 'not selected',
  not_reached: 'not reached',
};

export function ModuleStatus({ modules }: { modules: Record<string, ModuleState> }) {
  const names = Object.keys(modules);
  if (names.length === 0) return null;
  return (
    <div className="grid">
      {names.map((name) => (
        <div key={name} className="small">
          <Pill tone={modules[name] === 'not_reached' ? 'idle' : toneFor(modules[name])}>
            {STATE_TEXT[modules[name]] ?? modules[name]}
          </Pill>{' '}
          {LABELS[name] ?? name}
        </div>
      ))}
    </div>
  );
}
