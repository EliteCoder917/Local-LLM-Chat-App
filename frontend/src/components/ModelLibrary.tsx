import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Search, Download, Heart, ExternalLink, Trash2, X, Circle, Loader2, Check, Eye } from 'lucide-react';
import { useStore } from '../state/store';
import { BACKEND_HTTP } from '../ipc/bridge';
import type { DownloadJob } from '../state/types';

interface HFRepo {
  id: string;
  likes: number;
  downloads: number;
  lastModified: string | null;
  pipelineTag: string | null;
  tags: string[];
}

interface HFFile {
  name: string;
  sizeBytes: number;
  sizeGb: number;
  downloadUrl: string;
}

export default function ModelLibrary() {
  const {
    libraryOpen,
    setLibraryOpen,
    libraryModels,
    libraryRoot,
    downloads,
    deleteLibraryModel,
    selectLibraryModel,
    modelStatus,
  } = useStore();

  const [tab, setTab] = useState<'installed' | 'search'>('search');

  if (!libraryOpen) return null;
  const activeId = modelStatus.modelPath?.split(/[\\/]/).pop() ?? null;

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      onClick={() => setLibraryOpen(false)}
    >
      <div
        className="w-[920px] max-h-[85vh] bg-side border bd-strong rounded-xl shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="px-5 py-3 border-b border bd-soft flex items-center justify-between gap-4">
          <div className="min-w-0">
            <div className="text-sm font-semibold">Model library</div>
            <div className="text-[11px] text-[var(--fg-dim)] mono truncate" title={libraryRoot}>
              {libraryRoot || '…'}
            </div>
          </div>
          <div className="flex items-center gap-1 bg-app border bd-soft rounded-md p-0.5">
            <TabBtn label={`Discover`} active={tab === 'search'} onClick={() => setTab('search')} />
            <TabBtn label={`Installed (${libraryModels.length})`} active={tab === 'installed'} onClick={() => setTab('installed')} />
          </div>
          <button
            onClick={() => setLibraryOpen(false)}
            className="px-3 py-1.5 rounded-md text-sm bg-soft hover:bg-[var(--bd-soft)]"
          >
            Close
          </button>
        </header>

        <ActiveDownloads />

        {tab === 'search' ? (
          <DiscoverTab />
        ) : (
          <InstalledTab
            activeId={activeId}
            isLoaded={modelStatus.status === 'loaded'}
            onSelect={selectLibraryModel}
            onDelete={deleteLibraryModel}
            models={libraryModels}
          />
        )}
      </div>
    </div>
  );
}

function TabBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1 rounded text-xs ${
        active ? 'bg-soft text-[var(--fg)]' : 'text-[var(--fg-muted)] hover:text-[var(--fg)]'
      }`}
    >
      {label}
    </button>
  );
}

// ─── Active downloads (always visible at top) ──────────────────────
function ActiveDownloads() {
  const { downloads, cancelDownload } = useStore();
  const jobs = useMemo(() =>
    Object.values(downloads).filter((j) => j.status === 'queued' || j.status === 'downloading'),
  [downloads]);
  if (jobs.length === 0) return null;
  return (
    <div className="px-5 py-3 border-b border bd-soft bg-app space-y-2">
      <div className="text-[11px] uppercase tracking-wider text-[var(--fg-muted)]">Active downloads</div>
      {jobs.map((j) => <DownloadRow key={j.id} job={j} onCancel={() => cancelDownload(j.id)} />)}
    </div>
  );
}

// ─── Discover tab ─────────────────────────────────────────────────
function DiscoverTab() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<HFRepo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<number | null>(null);

  useEffect(() => {
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    if (query.trim().length < 2) {
      setResults([]);
      setError(null);
      return;
    }
    debounceRef.current = window.setTimeout(async () => {
      setLoading(true);
      setError(null);
      try {
        const r = await fetch(`${BACKEND_HTTP}/hf/search?q=${encodeURIComponent(query)}&limit=30`);
        if (!r.ok) {
          setError(`HTTP ${r.status}`);
          setResults([]);
        } else {
          const data = await r.json();
          setResults((data.results as HFRepo[]) ?? []);
        }
      } catch (e) {
        setError((e as Error).message);
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 350);
    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current);
    };
  }, [query]);

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="px-5 py-3 border-b border bd-soft">
        <div className="relative">
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search HuggingFace for GGUF models  e.g. 'llama 3 8b', 'qwen3', 'mistral'"
            className="w-full bg-app border bd-soft rounded-md px-3 py-2 pl-9 text-sm outline-none focus:bd-strong"
          />
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[var(--fg-dim)]" />
          {loading && (
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[11px] text-[var(--fg-dim)]">
              searching…
            </span>
          )}
        </div>
        {error && <div className="mt-2 text-xs text-red-400">{error}</div>}
        {query.trim().length < 2 && (
          <div className="mt-2 text-[11px] text-[var(--fg-dim)]">
            Tip: append "gguf" to refine, or search a specific quant publisher like
            "bartowski qwen3" or "mradermacher gemma".
          </div>
        )}
      </div>
      <div className="flex-1 overflow-auto scroll">
        {results.length === 0 && query.trim().length >= 2 && !loading && !error && (
          <div className="p-8 text-center text-sm text-[var(--fg-dim)]">No GGUF repos matched.</div>
        )}
        {results.map((repo) => (
          <RepoRow key={repo.id} repo={repo} />
        ))}
      </div>
    </div>
  );
}

function RepoRow({ repo }: { repo: HFRepo }) {
  const [expanded, setExpanded] = useState(false);
  const [files, setFiles] = useState<HFFile[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { libraryModels, downloads, startDownload } = useStore();

  async function toggle() {
    const next = !expanded;
    setExpanded(next);
    if (next && !files) {
      setLoading(true);
      setError(null);
      try {
        const r = await fetch(`${BACKEND_HTTP}/hf/files?repo=${encodeURIComponent(repo.id)}`);
        if (!r.ok) {
          setError(`HTTP ${r.status}`);
        } else {
          const data = await r.json();
          setFiles((data.files as HFFile[]) ?? []);
        }
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    }
  }

  return (
    <div className="border-b border bd-soft">
      <button
        onClick={toggle}
        className="w-full text-left px-5 py-3 hover:bg-side/60 flex items-center gap-3"
      >
        <span className="text-[var(--fg-dim)] w-3 text-xs">{expanded ? '▾' : '▸'}</span>
        <div className="min-w-0 flex-1">
          <div className="text-sm text-[var(--fg)] truncate">{repo.id}</div>
          <div className="text-[11px] text-[var(--fg-dim)] mono flex items-center gap-3 mt-0.5">
            <span className="inline-flex items-center gap-1"><Download className="w-3 h-3" />{formatCompact(repo.downloads)}</span>
            <span className="inline-flex items-center gap-1"><Heart className="w-3 h-3" />{formatCompact(repo.likes)}</span>
            {repo.lastModified && <span>{relativeDate(repo.lastModified)}</span>}
            {repo.tags.includes('gguf') && <span className="text-emerald-400">GGUF</span>}
            {isVisionRepo(repo) && (
              <span
                className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9.5px] bg-violet-900/40 text-violet-300"
                title="Vision-capable repo — look for mmproj-*.gguf alongside the main weights"
              >
                <Eye className="w-2.5 h-2.5" /> vision
              </span>
            )}
          </div>
        </div>
        <a
          href={`https://huggingface.co/${repo.id}`}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] hover:bg-[var(--bg-hover)] text-[var(--fg-muted)]"
          title="Open on HuggingFace"
        >
          HF <ExternalLink className="w-2.5 h-2.5" />
        </a>
      </button>
      {expanded && (
        <div className="bg-app/50 border-t border bd-soft">
          {loading && <div className="px-5 py-3 text-xs text-[var(--fg-dim)]">Loading file list…</div>}
          {error && <div className="px-5 py-3 text-xs text-red-400">Failed to load files: {error}</div>}
          {!loading && files && files.length === 0 && (
            <div className="px-5 py-3 text-xs text-[var(--fg-dim)]">No .gguf files in this repo.</div>
          )}
          {files && files.map((f) => {
            const filename = f.name.split('/').pop()?.toLowerCase() ?? '';
            const installed = libraryModels.some(
              (m) =>
                m.id.toLowerCase() === filename ||
                m.mmprojName?.toLowerCase() === filename,
            );
            const isMmproj = filename.startsWith('mmproj');
            const downloading = Object.values(downloads).find(
              (j) =>
                j.repo === repo.id &&
                j.filename === f.name &&
                (j.status === 'queued' || j.status === 'downloading'),
            );
            return (
              <div
                key={f.name}
                className="px-5 py-2 flex items-center gap-3 border-t border bd-soft/60 first:border-t-0"
              >
                <div className="min-w-0 flex-1">
                  <div className="text-xs text-[var(--fg)] truncate mono flex items-center gap-1.5">
                    <span className="truncate">{f.name.split('/').pop()}</span>
                    {isMmproj && (
                      <span
                        className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9.5px] bg-violet-900/40 text-violet-300 shrink-0"
                        title="Vision projector — pair this with the matching main weights"
                      >
                        <Eye className="w-2.5 h-2.5" /> projector
                      </span>
                    )}
                  </div>
                  <div className="text-[10.5px] text-[var(--fg-dim)]">
                    {f.sizeGb > 0 ? `${f.sizeGb.toFixed(2)} GB` : '— size unknown —'}
                    {quantLabel(f.name) && <span className="ml-2 text-[var(--fg-muted)]">{quantLabel(f.name)}</span>}
                    {isMmproj && <span className="ml-2 text-violet-400">companion file</span>}
                  </div>
                </div>
                {installed ? (
                  <span className="px-2 py-1 rounded text-[10px] bg-emerald-900/40 text-emerald-300">
                    Installed
                  </span>
                ) : downloading ? (
                  <span className="px-2 py-1 rounded text-[10px] bg-blue-900/40 text-blue-300">
                    {downloading.percent}%
                  </span>
                ) : (
                  <button
                    onClick={() => startDownload(`${repo.id}/${f.name}`)}
                    disabled={f.sizeBytes === 0}
                    className="px-2 py-1 rounded text-[11px] bg-blue-700 hover:bg-blue-600 disabled:opacity-40"
                  >
                    Download
                  </button>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ─── Installed tab ────────────────────────────────────────────────
function InstalledTab({
  activeId,
  isLoaded,
  onSelect,
  onDelete,
  models,
}: {
  activeId: string | null;
  isLoaded: boolean;
  onSelect: (id: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
  models: ReturnType<typeof useStore.getState>['libraryModels'];
}) {
  return (
    <div className="flex-1 overflow-auto scroll">
      {models.length === 0 ? (
        <div className="p-8 text-center text-sm text-[var(--fg-dim)]">
          No models installed yet. Use the Discover tab to find and download one.
        </div>
      ) : (
        models.map((m) => {
          const isActive = m.id === activeId && isLoaded;
          return (
            <div
              key={m.id}
              className="px-5 py-3 border-b border bd-soft flex items-center gap-3 hover:bg-side/60"
            >
              <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${isActive ? 'bg-emerald-500' : 'bg-neutral-700'}`} />
              <div className="min-w-0 flex-1">
                <div className="text-sm text-[var(--fg)] truncate flex items-center gap-2">
                  <span className="truncate">{m.name}</span>
                  {m.isVision && (
                    <span
                      className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] bg-violet-900/40 text-violet-300 shrink-0"
                      title={`Vision model — paired with ${m.mmprojName}`}
                    >
                      <Eye className="w-2.5 h-2.5" /> vision
                    </span>
                  )}
                </div>
                <div className="text-[11px] text-[var(--fg-dim)] mono truncate">
                  {m.sizeGb.toFixed(1)} GB
                  {m.arch && ` · ${m.arch}`}
                  {m.blockCount && ` · ${m.blockCount} layers`}
                  {m.trainedContext && ` · ${m.trainedContext.toLocaleString()} ctx`}
                  {m.mmprojName && (
                    <span className="text-violet-400"> · mmproj paired</span>
                  )}
                </div>
              </div>
              {!isActive && (
                <button
                  onClick={() => onSelect(m.id)}
                  className="px-3 py-1 rounded text-xs bg-emerald-700 hover:bg-emerald-600"
                >
                  Load
                </button>
              )}
              <button
                onClick={() => {
                  if (confirm(`Delete ${m.name}? This will remove the file from disk.`)) {
                    onDelete(m.id);
                  }
                }}
                className="px-2 py-1 rounded text-xs text-[var(--fg-muted)] hover:text-red-400 hover:bg-soft"
                title="Delete model"
              >
                ✕
              </button>
            </div>
          );
        })
      )}
    </div>
  );
}

// ─── shared ───────────────────────────────────────────────────────
function DownloadRow({ job, onCancel }: { job: DownloadJob; onCancel?: () => void }) {
  const totalMB = job.total / (1024 * 1024);
  const downloadedMB = job.downloaded / (1024 * 1024);
  const color =
    job.status === 'done' ? 'bg-emerald-600' :
    job.status === 'error' ? 'bg-red-600' :
    job.status === 'cancelled' ? 'bg-neutral-600' :
    'bg-blue-600';
  return (
    <div className="rounded border bd-soft bg-app p-2">
      <div className="flex items-center gap-2">
        <div className="min-w-0 flex-1">
          <div className="text-xs text-[var(--fg)] truncate">{job.filename}</div>
          <div className="text-[10px] text-[var(--fg-dim)] mono truncate">
            {job.repo} · {job.status}
            {job.status === 'downloading' && job.speedMBs > 0 && ` · ${job.speedMBs.toFixed(1)} MB/s`}
            {job.error && ` · ${job.error}`}
          </div>
        </div>
        <div className="text-[11px] text-[var(--fg-muted)] mono whitespace-nowrap">
          {job.total > 0
            ? `${downloadedMB.toFixed(0)} / ${totalMB.toFixed(0)} MB · ${job.percent}%`
            : ''}
        </div>
        {onCancel && (
          <button
            onClick={onCancel}
            className="px-2 py-0.5 rounded text-[10px] bg-soft hover:bg-[var(--bd-soft)]"
          >
            Cancel
          </button>
        )}
      </div>
      <div className="mt-1 h-1 rounded bg-soft overflow-hidden">
        <div className={`h-full ${color} transition-all`} style={{ width: `${job.percent}%` }} />
      </div>
    </div>
  );
}

function formatCompact(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function relativeDate(iso: string): string {
  const d = new Date(iso);
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 86400 * 30) return `${Math.floor(diff / 86400)}d ago`;
  if (diff < 86400 * 365) return `${Math.floor(diff / 86400 / 30)}mo ago`;
  return `${Math.floor(diff / 86400 / 365)}y ago`;
}

function quantLabel(name: string): string {
  const lower = name.toLowerCase();
  const match = lower.match(/q\d(?:_[a-z0-9]+)?/i);
  return match ? match[0].toUpperCase() : '';
}

// Heuristic: does this HF repo describe a vision-capable model?
// Checks pipeline tag + library tags first (most reliable), then falls back
// to keyword matching the repo id. Family-based so future versions of known
// vision families (qwen3-vl, llama-4-vision, etc.) light up automatically.
const VISION_TAGS = new Set([
  'image-text-to-text',
  'image-to-text',
  'visual-question-answering',
  'multimodal',
  'vision',
]);
const VISION_ID_RE = /(?:^|[-_/.])(?:vl|vision|llava|moondream|minicpm-?v|nanollava|obsidian|mllama)(?:[-_/.]|$)/i;

function isVisionRepo(repo: HFRepo): boolean {
  if (repo.pipelineTag && VISION_TAGS.has(repo.pipelineTag.toLowerCase())) return true;
  for (const t of repo.tags) {
    if (VISION_TAGS.has(t.toLowerCase())) return true;
  }
  return VISION_ID_RE.test(repo.id);
}
