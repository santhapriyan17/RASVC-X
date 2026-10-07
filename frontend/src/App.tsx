// RASVC-X root component.
//
// Chat is the primary view. Knowledge, System Status and Evaluation are the
// administrative views behind it. The header shows the backend's real mode,
// read from GET /ready — there is no client-side online/offline switch.

import { useCallback, useEffect, useState } from 'react';
import { fetchReadiness } from './api/client';
import { Chat } from './components/Chat';
import { EvalDashboard } from './components/EvalDashboard';
import { IngestPanel } from './components/IngestPanel';
import { SystemStatus } from './components/SystemStatus';
import { Pill, errorMessage } from './components/ui';
import type { ActiveView, Readiness } from './types';

const NAV: { id: ActiveView; label: string }[] = [
  { id: 'chat', label: 'Chat' },
  { id: 'knowledge', label: 'Knowledge' },
  { id: 'status', label: 'System Status' },
  { id: 'evaluation', label: 'Evaluation' },
];

export function App() {
  const [view, setView] = useState<ActiveView>('chat');
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [backendError, setBackendError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setReadiness(await fetchReadiness());
      setBackendError(null);
    } catch (err) {
      setReadiness(null);
      setBackendError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 30000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const down = readiness?.components.filter((c) => c.state === 'unavailable') ?? [];

  return (
    <>
      <header className="app-header">
        <div>
          <div className="app-title">RASVC-X</div>
          <div className="app-sub">Evidence-validated medical question answering</div>
        </div>
        <nav className="nav" aria-label="Views">
          {NAV.map((item) => (
            <button key={item.id} className={view === item.id ? 'active' : ''}
                    aria-current={view === item.id ? 'page' : undefined}
                    onClick={() => setView(item.id)}>
              {item.label}
            </button>
          ))}
        </nav>
        <span className="spacer" />
        {readiness && (
          <>
            {readiness.offline
              ? <Pill tone="warn" title="The backend was started in offline_test mode">OFFLINE TEST</Pill>
              : <Pill tone="info" title={readiness.execution_mode}>ONLINE</Pill>}
            <Pill tone={readiness.ready ? 'ok' : 'bad'}>{readiness.ready ? 'ready' : 'not ready'}</Pill>
            <span className="muted small" title="Knowledge-base version being served">
              KB <span className="mono">{readiness.kb_version_id ?? 'none'}</span>
              {readiness.kb_source ? ` (${readiness.kb_source.replace('_', ' ')})` : ''}
            </span>
            <Pill tone={readiness.calibration?.status === 'calibrated' ? 'ok' : 'warn'}
                  title={readiness.calibration?.reason ?? 'confidence calibration status'}>
              {(readiness.calibration?.status ?? 'uncalibrated').toUpperCase()}
            </Pill>
          </>
        )}
      </header>

      {(readiness?.kb_warnings ?? []).length > 0 && (
        <div className="banner offline">
          {readiness!.kb_warnings!.map((w) => <div key={w}>{w}</div>)}
        </div>
      )}

      {backendError && (
        <div className="banner down">
          The backend is not reachable ({backendError}). Start it with <span className="mono">python -m rasvcx</span>.
        </div>
      )}
      {readiness?.offline && (
        <div className="banner offline">
          Offline test mode: answers come from a deterministic stub, not a language model, and there
          is no reranking, dense retrieval or NLI. Use this mode only to test the application itself.
        </div>
      )}
      {down.length > 0 && (
        <div className="banner down">
          Unavailable: {down.map((c) => `${c.name}${c.detail ? ` (${c.detail})` : ''}`).join('; ')}
        </div>
      )}

      <main className="main">
        {view === 'chat' && <Chat />}
        {view === 'knowledge' && <IngestPanel />}
        {view === 'status' && <SystemStatus />}
        {view === 'evaluation' && <EvalDashboard />}
      </main>
    </>
  );
}
