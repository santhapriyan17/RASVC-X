// Evaluation view: completed evaluation runs found by GET /eval/runs.
// Runs are produced offline by evaluation/runner.py; this page only lists
// what exists. It never shows a metric the backend did not report.

import { useCallback, useEffect, useState } from 'react';
import { fetchEvalRuns } from '../api/client';
import type { EvalRun } from '../types';
import { Pill, errorMessage, when } from './ui';

export function EvalDashboard() {
  const [runs, setRuns] = useState<EvalRun[] | null>(null);
  const [total, setTotal] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const list = await fetchEvalRuns();
      setRuns(list.runs);
      setTotal(list.total);
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="panel">
      <div className="row">
        <h2 style={{ margin: 0 }}>Evaluation runs</h2>
        <span className="spacer" />
        <button onClick={() => void load()}>Refresh</button>
      </div>
      {error && <p className="error">Could not load evaluation runs: {error}</p>}
      {runs && runs.length === 0 && (
        <p className="muted">
          No evaluation run has been recorded yet. Runs are created with the evaluation runner
          (see <span className="mono">evaluation/runner.py</span>) and appear here when finished.
        </p>
      )}
      {runs && runs.length > 0 && (
        <>
          <p className="muted small">{total} run(s). Results of runs marked “mock LLM” say nothing about real model accuracy.</p>
          <table>
            <thead>
              <tr>
                <th>Run</th><th>Baseline</th><th>Dataset</th><th>Mode</th>
                <th className="num">Cases</th><th className="num">Completed</th><th className="num">Errors</th>
                <th className="num">Wall time</th><th>Started</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}>
                  <td className="mono">{r.run_id}
                    {r.integrity_error && <div className="error small">{r.integrity_error}</div>}
                  </td>
                  <td>{r.baseline_id}</td>
                  <td>{r.dataset_id} <span className="muted">v{r.dataset_version} · {r.split} · {r.corpus_condition}</span></td>
                  <td>
                    <span className="mono">{r.execution_mode}</span>{' '}
                    {r.mock_llm ? <Pill tone="warn">mock LLM</Pill> : <Pill tone="info">real LLM</Pill>}
                  </td>
                  <td className="num">{r.total_cases}</td>
                  <td className="num">{r.completed}</td>
                  <td className="num">{r.error}</td>
                  <td className="num">{r.total_run_wall_seconds.toFixed(1)} s</td>
                  <td>{when(r.start_utc)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
