// Knowledge view: the versioned knowledge base behind the chat.
// Upload documents, watch ingestion jobs, manage synchronised sources.

import { useCallback, useEffect, useState } from 'react';
import {
  cancelJob,
  createSource,
  deleteSource,
  fetchIngestStatus,
  fetchJobs,
  fetchSources,
  ingestUrl,
  syncSource,
  uploadFile,
} from '../api/client';
import type { IngestJob, IngestSource, IngestStatus, UploadMetadata } from '../types';
import { SOURCE_TYPES } from '../types';
import { Pill, errorMessage, toneFor, when } from './ui';

const TERMINAL = new Set(['completed', 'failed', 'cancelled']);

function jobTone(job: IngestJob) {
  if (job.status === 'completed') return 'ok' as const;
  if (job.status === 'failed') return 'bad' as const;
  if (job.status === 'cancelled') return 'idle' as const;
  return 'info' as const;
}

export function IngestPanel() {
  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [jobs, setJobs] = useState<IngestJob[]>([]);
  const [sources, setSources] = useState<IngestSource[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [file, setFile] = useState<File | null>(null);
  const [meta, setMeta] = useState<UploadMetadata>({ source_type: 'other' });
  const [uploading, setUploading] = useState(false);
  const [url, setUrl] = useState('');
  const [srcName, setSrcName] = useState('');
  const [srcUrl, setSrcUrl] = useState('');
  const [srcHours, setSrcHours] = useState(24);

  const refresh = useCallback(async () => {
    try {
      const [s, j, src] = await Promise.all([fetchIngestStatus(), fetchJobs(), fetchSources()]);
      setStatus(s);
      setJobs(j.jobs);
      setSources(src.sources);
      setError(null);
    } catch (err) {
      setError(errorMessage(err));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll only while a job is in flight.
  const active = jobs.some((j) => !TERMINAL.has(j.status));
  useEffect(() => {
    if (!active) return undefined;
    const timer = window.setInterval(() => void refresh(), 2000);
    return () => window.clearInterval(timer);
  }, [active, refresh]);

  const run = async (action: () => Promise<string>) => {
    setNotice(null);
    try {
      setNotice(await action());
      await refresh();
    } catch (err) {
      setError(errorMessage(err));
    }
  };

  const doUpload = async () => {
    if (!file) return;
    setUploading(true);
    await run(async () => {
      const res = await uploadFile(file, meta);
      setFile(null);
      return res.duplicate ? 'This exact file is already in the knowledge base.' : `Queued: ${res.filename ?? file.name}`;
    });
    setUploading(false);
  };

  const setM = (key: keyof UploadMetadata, value: string) =>
    setMeta((prev) => ({ ...prev, [key]: value }));

  const av = status?.active_version ?? null;

  return (
    <div>
      {error && <div className="panel error">Knowledge-base API error: {error}</div>}
      {notice && <div className="panel">{notice}</div>}

      <div className="panel">
        <h2>Active knowledge base</h2>
        {av ? (
          <dl className="kv">
            <dt>Version served to chat</dt>
            <dd className="mono">
              {av.version_id}{' '}
              {status?.consistent
                ? <Pill tone="ok">consistent</Pill>
                : <Pill tone="bad" title={`pointer on disk: ${status?.pointer_version_id}`}>pointer mismatch</Pill>}
            </dd>
            <dt>Documents / chunks</dt><dd>{av.doc_count} / {av.chunk_count}</dd>
            <dt>BM25 index</dt><dd>{av.bm25_size} chunks</dd>
            <dt>Qdrant collection</dt>
            <dd className="mono">{av.qdrant_collection ?? 'none (BM25-only mode)'}</dd>
            <dt>Published</dt>
            <dd>{av.published_at ? when(av.published_at) : 'seed corpus (no ingestion published yet)'}</dd>
          </dl>
        ) : <p className="muted">No knowledge-base version is loaded.</p>}
        {status && (
          <>
            <h3>Versions</h3>
            <table>
              <thead><tr><th>Version</th><th>Status</th><th className="num">Requests in flight</th><th>Published</th></tr></thead>
              <tbody>
                {status.versions.map((v) => (
                  <tr key={v.version_id}>
                    <td className="mono">{v.version_id}</td>
                    <td><Pill tone={v.status === 'active' ? 'ok' : 'idle'}>{v.status}</Pill></td>
                    <td className="num">{v.active_request_count}</td>
                    <td>{when(v.published_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted small">
              A superseded version is deleted only when no request is using it and the retention
              policy allows it.
            </p>
          </>
        )}
      </div>

      <div className="panel">
        <h2>Add a document</h2>
        <p className="muted small">
          PDF, DOCX, XLSX, PPTX, HTML, TXT, MD, CSV, JSON or XML. Provenance fields you leave empty
          are stored as unknown — they are never guessed from the text — and unknown provenance
          makes the system more cautious on high-risk questions.
        </p>
        <div className="grid">
          <label>File
            <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <label>Title
            <input value={meta.title ?? ''} onChange={(e) => setM('title', e.target.value)} />
          </label>
          <label>Source type
            <select value={meta.source_type ?? 'other'} onChange={(e) => setM('source_type', e.target.value)}>
              {SOURCE_TYPES.map((t) => <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>)}
            </select>
          </label>
          <label>Publication date
            <input placeholder="YYYY-MM-DD" value={meta.date ?? ''} onChange={(e) => setM('date', e.target.value)} />
          </label>
          <label>Jurisdiction
            <input placeholder="e.g. US" value={meta.jurisdiction ?? ''} onChange={(e) => setM('jurisdiction', e.target.value)} />
          </label>
          <label>Population
            <input placeholder="e.g. adults" value={meta.population ?? ''} onChange={(e) => setM('population', e.target.value)} />
          </label>
          <label>Dosage context
            <input placeholder="e.g. oral" value={meta.dosage_context ?? ''} onChange={(e) => setM('dosage_context', e.target.value)} />
          </label>
          <label title="Declared by you; temporal states are never inferred from the date">Lifecycle status
            <select value={meta.status ?? ''} onChange={(e) => setM('status', e.target.value)}>
              <option value="">not declared</option>
              {['current', 'superseded', 'historical', 'withdrawn'].map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
          <label>Supersedes (doc ids)
            <input placeholder="comma-separated" value={meta.supersedes ?? ''} onChange={(e) => setM('supersedes', e.target.value)} />
          </label>
          <label>Effective date
            <input placeholder="YYYY-MM-DD" value={meta.effective_date ?? ''} onChange={(e) => setM('effective_date', e.target.value)} />
          </label>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="primary" disabled={!file || uploading} onClick={() => void doUpload()}>
            {uploading ? 'Uploading…' : 'Upload and index'}
          </button>
        </div>
        <h3>Or fetch once from a URL (HTTPS only)</h3>
        <div className="row">
          <input style={{ flex: 1, minWidth: 260 }} placeholder="https://…" value={url}
                 onChange={(e) => setUrl(e.target.value)} />
          <button disabled={!url.trim()} onClick={() => void run(async () => {
            await ingestUrl(url.trim());
            setUrl('');
            return 'URL queued for ingestion.';
          })}>Fetch</button>
        </div>
      </div>

      <div className="panel">
        <h2>Ingestion jobs</h2>
        {jobs.length === 0 ? <p className="muted">No ingestion jobs yet.</p> : (
          <table>
            <thead><tr><th>Document</th><th>Stage</th><th>Published version</th><th>Started</th><th></th></tr></thead>
            <tbody>
              {jobs.map((j) => (
                <tr key={j.job_id}>
                  <td>{j.filename ?? j.source_url ?? j.job_id}
                    {j.error_message && <div className="error small">{j.error_message}</div>}
                  </td>
                  <td><Pill tone={jobTone(j)}>{j.stage.replace(/_/g, ' ')}</Pill></td>
                  <td className="mono">{j.corpus_version_id ?? '—'}</td>
                  <td>{when(j.created_at)}</td>
                  <td>
                    {!TERMINAL.has(j.status) && j.stage !== 'publishing' && (
                      <button onClick={() => void run(async () => {
                        await cancelJob(j.job_id);
                        return 'Job cancelled.';
                      })}>Cancel</button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel">
        <h2>Synchronised sources</h2>
        <p className="muted small">
          A source is re-fetched on its interval and re-indexed only when its content changed.
        </p>
        {sources.length > 0 && (
          <table>
            <thead><tr><th>Source</th><th>State</th><th>Last sync</th><th>Next sync</th><th></th></tr></thead>
            <tbody>
              {sources.map((s) => (
                <tr key={s.source_id}>
                  <td>{s.display_name}<div className="muted small">{s.source_url}</div>
                    {s.last_error && <div className="error small">{s.last_error}</div>}
                  </td>
                  <td><Pill tone={toneFor(s.sync_state === 'error' ? 'failed' : s.sync_state === 'suspended' ? 'unavailable' : 'ok')}>{s.sync_state}</Pill></td>
                  <td>{when(s.last_sync_at)}</td>
                  <td>{when(s.next_sync_at)}</td>
                  <td className="row">
                    <button onClick={() => void run(async () => (await syncSource(s.source_id)).message)}>Sync now</button>
                    <button onClick={() => void run(async () => {
                      await deleteSource(s.source_id);
                      return 'Source removed (already-indexed documents are kept).';
                    })}>Remove</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="row" style={{ marginTop: 10 }}>
          <label>Name<input value={srcName} onChange={(e) => setSrcName(e.target.value)} /></label>
          <label style={{ flex: 1, minWidth: 240 }}>HTTPS URL
            <input style={{ width: '100%' }} value={srcUrl} onChange={(e) => setSrcUrl(e.target.value)} />
          </label>
          <label>Every (hours)
            <input type="number" min={1} style={{ width: 90 }} value={srcHours}
                   onChange={(e) => setSrcHours(Math.max(1, Number(e.target.value) || 1))} />
          </label>
          <button disabled={!srcName.trim() || !srcUrl.trim()} onClick={() => void run(async () => {
            await createSource(srcName.trim(), srcUrl.trim(), srcHours * 3600);
            setSrcName('');
            setSrcUrl('');
            return 'Source registered.';
          })}>Add source</button>
        </div>
      </div>
    </div>
  );
}
