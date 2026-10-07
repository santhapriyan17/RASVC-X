// Small shared presentational helpers. No data fetching here.

import type { ReactNode } from 'react';
import type { Decision } from '../types';

export type Tone = 'ok' | 'warn' | 'bad' | 'info' | 'idle';

export function Pill({ tone = 'idle', children, title }: {
  tone?: Tone;
  children: ReactNode;
  title?: string;
}) {
  return (
    <span className={`pill ${tone === 'idle' ? '' : tone}`} title={title}>
      {children}
    </span>
  );
}

const DECISION_TONE: Record<Decision, Tone> = {
  ANSWER: 'ok',
  ANSWER_WITH_WARNING: 'warn',
  REPAIR: 'info',
  REGENERATE: 'info',
  ABSTAIN: 'bad',
};

const DECISION_LABEL: Record<Decision, string> = {
  ANSWER: 'Answer',
  ANSWER_WITH_WARNING: 'Answer with warning',
  REPAIR: 'Repair',
  REGENERATE: 'Regenerate',
  ABSTAIN: 'Abstained',
};

export function DecisionBadge({ decision }: { decision: Decision }) {
  return (
    <span className={`pill decision ${DECISION_TONE[decision] ?? ''}`} title={decision}>
      {DECISION_LABEL[decision] ?? decision}
    </span>
  );
}

/** Tone for a claim SupportLabel / evidence relationship / verdict string. */
export function toneFor(value: string): Tone {
  switch (value) {
    case 'supported':
    case 'verified':
    case 'compatible':
    case 'known_match':
    case 'correct':
    case 'executed':
    case 'ok':
      return 'ok';
    case 'contradicted':
    case 'unsafe':
    case 'genuine-conflict':
    case 'known_mismatch':
    case 'incorrect':
    case 'contradictory':
    case 'failed':
    case 'unavailable':
      return 'bad';
    case 'partially_supported':
    case 'partially_verified':
    case 'unsupported':
    case 'unverified':
    case 'unresolved':
    case 'missing':
    case 'insufficient_evidence':
    case 'stub':
      return 'warn';
    case 'population-diff':
    case 'temporal-diff':
    case 'jurisdiction-diff':
    case 'dosage-diff':
    case 'configured':
      return 'info';
    default:
      return 'idle';
  }
}

export function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function ms(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${value.toFixed(1)} ms`;
}

export function when(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
