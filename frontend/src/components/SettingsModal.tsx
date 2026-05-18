import React, { useEffect, useState } from 'react';
import { X, Cpu, Shield, Palette, Bot, ChevronRight, ChevronDown, Trash2 } from 'lucide-react';
import { useStore } from '../state/store';
import { BACKEND_HTTP } from '../ipc/bridge';
import type { SystemInfo } from '../state/types';
import PermissionToggles from './PermissionToggles';
import Toggle from './Toggle';

interface Props {
  onClose: () => void;
}

type TabKey = 'model' | 'permissions' | 'appearance' | 'agent';

const TAB_DEFS: { key: TabKey; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { key: 'model',       label: 'Model',       icon: Cpu },
  { key: 'permissions', label: 'Permissions', icon: Shield },
  { key: 'appearance',  label: 'Appearance',  icon: Palette },
  { key: 'agent',       label: 'Agent',       icon: Bot },
];

export default function SettingsModal({ onClose }: Props) {
  const { settings, setSettings } = useStore();
  const [tab, setTab] = useState<TabKey>('model');

  // Esc closes
  useEffect(() => {
    const fn = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="w-[820px] max-h-[85vh] bg-app border bd-strong rounded-2xl shadow-2xl flex overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Left rail */}
        <aside className="w-52 bg-side border-r bd-soft p-2 flex flex-col gap-0.5">
          <div className="px-3 pt-2 pb-3 text-[15px] font-medium text-[var(--fg)]">Settings</div>
          {TAB_DEFS.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setTab(key)}
              className={`text-left flex items-center gap-2.5 px-3 py-1.5 rounded-md text-[13px] transition ${
                tab === key
                  ? 'bg-[var(--bg-hover)] text-[var(--fg)]'
                  : 'text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          ))}
        </aside>

        {/* Right pane */}
        <div className="flex-1 flex flex-col min-w-0">
          <header className="h-12 px-5 flex items-center justify-between border-b bd-soft">
            <h2 className="text-[14px] font-medium text-[var(--fg)]">
              {TAB_DEFS.find((t) => t.key === tab)?.label}
            </h2>
            <button
              onClick={onClose}
              className="p-1.5 rounded-md text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]"
              title="Close (Esc)"
            >
              <X className="w-4 h-4" />
            </button>
          </header>

          <div className="flex-1 overflow-auto scroll p-5">
            {tab === 'model' && (
              <div className="space-y-5">
                <ModelStatusPanel />

                <Section title="Engine">
                  <Field label="Backend">
                    <select
                      value={settings.engine}
                      onChange={(e) => setSettings({ engine: e.target.value as 'ollama' | 'llama-cpp' })}
                      className="w-full bg-side border bd-soft rounded-lg px-3 py-2 text-sm focus:bd-strong outline-none"
                    >
                      <option value="llama-cpp">Local .gguf (llama-cpp-python)</option>
                      <option value="ollama">Ollama</option>
                    </select>
                  </Field>

                  {settings.engine === 'llama-cpp' && <LibraryShortcut />}

                  {settings.engine === 'ollama' && (
                    <>
                      <Field label="Model tag">
                        <input
                          value={settings.model}
                          onChange={(e) => setSettings({ model: e.target.value })}
                          placeholder="llama3.1:8b"
                          className="w-full bg-side border bd-soft rounded-lg px-3 py-2 mono text-sm focus:bd-strong outline-none"
                        />
                      </Field>
                      <Field label="Ollama URL">
                        <input
                          value={settings.ollamaUrl}
                          onChange={(e) => setSettings({ ollamaUrl: e.target.value })}
                          className="w-full bg-side border bd-soft rounded-lg px-3 py-2 mono text-sm focus:bd-strong outline-none"
                        />
                      </Field>
                    </>
                  )}
                </Section>

                <Section title="Sampling">
                  <Field label={`Temperature — ${settings.temperature.toFixed(2)}`}>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={settings.temperature}
                      onChange={(e) => setSettings({ temperature: parseFloat(e.target.value) })}
                      className="slim w-full"
                    />
                    <div className="text-[11px] text-[var(--fg-dim)] mt-1 flex justify-between">
                      <span>0 · deterministic</span><span>1 · default</span><span>2 · wild</span>
                    </div>
                  </Field>
                </Section>

                {settings.engine === 'llama-cpp' && (
                  <Section title="Runtime">
                    <ContextWindowField />
                    <GpuOffloadField />
                  </Section>
                )}

                <Section
                  title="Attachments"
                  hint="Images attached to messages are always shown in the chat. By default the model only sees a '[Image: filename]' placeholder so it doesn't waste tokens on base64. Turn this on only if you've loaded a vision-capable model (LLaVA, Qwen-VL, etc.)."
                >
                  <ToggleRow
                    label="Send images to model as base64"
                    desc="Off = placeholder only (fast, no context blown). On = full image data (vision models only)."
                    checked={settings.sendImagesAsBase64}
                    onChange={(v) => setSettings({ sendImagesAsBase64: v })}
                  />
                </Section>

                <Section title="System prompt" hint="Verbatim instructions prepended to every conversation. Leave empty for none.">
                  <SystemPromptEditor />
                </Section>
              </div>
            )}

            {tab === 'permissions' && (
              <div className="space-y-5">
                <Section title="Tool permissions" hint="Each tool the model can call is gated. Toggle on to grant blanket permission; leave off to be prompted each time.">
                  <PermissionToggles />
                </Section>
                <Section title="Persistent memory" hint="Long-term notes the model has saved. Survives restarts and is shared across all conversations.">
                  <MemoryPanel />
                </Section>
              </div>
            )}

            {tab === 'appearance' && (
              <div className="space-y-5">
                <Section title="Theme">
                  <Field label="Color scheme">
                    <select
                      value={settings.theme}
                      onChange={(e) => setSettings({ theme: e.target.value as 'dark' | 'light' })}
                      className="w-full bg-side border bd-soft rounded-lg px-3 py-2 text-sm focus:bd-strong outline-none"
                    >
                      <option value="dark">Dark</option>
                      <option value="light">Light</option>
                    </select>
                  </Field>
                </Section>
              </div>
            )}

            {tab === 'agent' && (
              <div className="space-y-5">
                <Section
                  title="Tool access by tab"
                  hint="Whether the model has tool access is decided by which tab you're in — not a global toggle."
                >
                  <div className="text-[13px] text-[var(--fg-muted)] space-y-2">
                    <p>
                      <span className="text-[var(--fg)]">Chat tab</span> — pure chat.
                      The model gets only your system prompt below. It cannot read your
                      files, run code, or call any tool. Fastest response, no surprises.
                    </p>
                    <p>
                      <span className="text-[var(--fg)]">Code tab</span> — tools enabled.
                      The full tool catalog is injected into the system prompt and the
                      model can chain calls. It's instructed to only invoke tools when
                      you've explicitly asked for an action (read / write / run / refactor).
                    </p>
                  </div>
                </Section>
                <Section
                  title="Agent loop"
                  hint="Only relevant in the Code tab. Sets the maximum number of tool-call cycles per response."
                >
                  <Field label={`Max iterations — ${settings.maxIterations}`}>
                    <input
                      type="range"
                      min={1}
                      max={30}
                      value={settings.maxIterations}
                      onChange={(e) => setSettings({ maxIterations: parseInt(e.target.value, 10) })}
                      className="slim w-full"
                    />
                    <div className="text-[11px] text-[var(--fg-dim)] mt-1">
                      Higher = the model can self-correct over more tool calls before
                      stopping. Lower = forces an answer sooner.
                    </div>
                  </Field>
                </Section>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// Local-state editor for the system prompt. The shared `setSettings` does
// an IPC roundtrip + Electron Store write + WebSocket settings-update +
// model-status response on EVERY keystroke. That cascade can leave keystrokes
// stranded if React batches updates against an in-flight re-render. This
// component types into local state immediately, and only commits to the
// global store on blur or after a brief idle window. The textarea always
// feels instant regardless of what's happening behind the scenes.
function SystemPromptEditor() {
  const { settings, setSettings } = useStore();
  const [local, setLocal] = useState(settings.systemPrompt);
  const commitTimer = React.useRef<number | null>(null);

  // If the global value changes from elsewhere (e.g. /system slash command),
  // mirror it back into local state — but only when the textarea isn't being
  // actively edited (we don't want to clobber what the user is typing).
  useEffect(() => {
    if (document.activeElement?.tagName !== 'TEXTAREA') {
      setLocal(settings.systemPrompt);
    }
  }, [settings.systemPrompt]);

  function scheduleCommit(value: string) {
    if (commitTimer.current) window.clearTimeout(commitTimer.current);
    commitTimer.current = window.setTimeout(() => {
      void setSettings({ systemPrompt: value });
      commitTimer.current = null;
    }, 400);
  }

  function flush() {
    if (commitTimer.current) {
      window.clearTimeout(commitTimer.current);
      commitTimer.current = null;
    }
    if (local !== settings.systemPrompt) {
      void setSettings({ systemPrompt: local });
    }
  }

  return (
    <textarea
      rows={5}
      value={local}
      onChange={(e) => { setLocal(e.target.value); scheduleCommit(e.target.value); }}
      onBlur={flush}
      className="w-full bg-side border bd-soft rounded-lg px-3 py-2 text-sm focus:bd-strong outline-none resize-y"
      placeholder="You are a helpful local AI assistant."
    />
  );
}

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border bd-soft bg-side p-4 space-y-3">
      <div>
        <div className="text-[13px] font-medium text-[var(--fg)]">{title}</div>
        {hint && <div className="text-[11.5px] text-[var(--fg-dim)] mt-0.5">{hint}</div>}
      </div>
      {children}
    </div>
  );
}

function ToggleRow({
  label, desc, checked, onChange,
}: { label: string; desc?: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-start gap-3 py-1.5">
      <div className="flex-1 min-w-0">
        <div className="text-[13px] text-[var(--fg)]">{label}</div>
        {desc && <div className="text-[11.5px] text-[var(--fg-dim)]">{desc}</div>}
      </div>
      <Toggle checked={checked} onChange={onChange} />
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-xs text-[var(--fg-muted)] mb-1">{label}</div>
      {children}
    </div>
  );
}

interface MemoryEntry { key: string; value: unknown; updatedAt: number }

function MemoryPanel() {
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [openValue, setOpenValue] = useState<string | null>(null);

  async function refresh() {
    setLoading(true);
    try {
      const r = await fetch(`${BACKEND_HTTP}/memory`);
      if (r.ok) {
        const j = await r.json();
        setEntries(j.entries ?? []);
      }
    } catch { /* backend down */ }
    setLoading(false);
  }

  useEffect(() => { refresh(); }, []);

  async function deleteOne(key: string) {
    if (!confirm(`Delete memory key "${key}"?`)) return;
    await fetch(`${BACKEND_HTTP}/memory/${encodeURIComponent(key)}`, { method: 'DELETE' });
    refresh();
  }
  async function clearAll() {
    if (!confirm(`Delete ALL ${entries.length} memory entries? This is permanent.`)) return;
    await fetch(`${BACKEND_HTTP}/memory`, { method: 'DELETE' });
    refresh();
  }

  return (
    <div className="mt-4 pt-4 border-t border bd-soft">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-semibold uppercase text-[var(--fg-muted)]">
          Persistent memory ({entries.length})
        </h3>
        <div className="flex gap-1">
          <button onClick={refresh} className="px-2 py-1 rounded text-[11px] bg-soft hover:bg-[var(--bd-soft)]">
            Refresh
          </button>
          <button
            onClick={clearAll}
            disabled={entries.length === 0}
            className="px-2 py-1 rounded text-[11px] bg-red-800/70 hover:bg-red-700 disabled:opacity-40"
          >
            Clear all
          </button>
        </div>
      </div>
      <p className="text-[11px] text-[var(--fg-dim)] mb-2">
        These are notes the model has saved via the memory tool. They persist across all
        conversations and survive restarts. Stored at
        <span className="mono"> %LOCALAPPDATA%\LocalAIIDE\memory.sqlite</span>.
      </p>
      <div className="rounded border bd-soft bg-app max-h-64 overflow-auto scroll">
        {loading && <div className="p-3 text-xs text-[var(--fg-dim)]">Loading…</div>}
        {!loading && entries.length === 0 && (
          <div className="p-3 text-xs text-[var(--fg-dim)]">
            No memory entries. (The model hasn't saved anything, or memory permission is off.)
          </div>
        )}
        {entries.map((e) => {
          const preview = typeof e.value === 'string' ? e.value : JSON.stringify(e.value);
          const expanded = openValue === e.key;
          return (
            <div key={e.key} className="border-b border bd-soft/60 last:border-b-0">
              <div className="px-3 py-2 flex items-start gap-2">
                <button
                  onClick={() => setOpenValue(expanded ? null : e.key)}
                  className="text-[var(--fg-dim)] mt-0.5 w-3 text-xs"
                >
                  {expanded ? '▾' : '▸'}
                </button>
                <div className="flex-1 min-w-0">
                  <div className="text-xs text-[var(--fg)] mono truncate">{e.key}</div>
                  {!expanded && (
                    <div className="text-[11px] text-[var(--fg-dim)] truncate mono">{preview.slice(0, 120)}</div>
                  )}
                </div>
                <button
                  onClick={() => deleteOne(e.key)}
                  className="px-2 py-0.5 rounded text-[10.5px] text-[var(--fg-muted)] hover:text-red-400 hover:bg-soft"
                  title="Delete this key"
                >
                  ✕
                </button>
              </div>
              {expanded && (
                <pre className="mx-3 mb-2 p-2 bg-black/40 rounded text-[11.5px] mono whitespace-pre-wrap overflow-auto">{preview}</pre>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function LibraryShortcut() {
  const { setLibraryOpen, libraryModels, libraryRoot, modelStatus } = useStore();
  const activeId = modelStatus.modelPath?.split(/[\\/]/).pop() ?? null;
  const active = libraryModels.find((m) => m.id === activeId);
  return (
    <Field label="Model">
      <div className="rounded-lg border bd-soft bg-app p-3 flex items-center gap-3">
        <div className="flex-1 min-w-0">
          {active ? (
            <>
              <div className="text-sm text-[var(--fg)] truncate">{active.name}</div>
              <div className="text-[11px] text-[var(--fg-dim)] mono truncate">
                {active.sizeGb.toFixed(1)} GB
                {active.arch && ` · ${active.arch}`}
              </div>
            </>
          ) : (
            <>
              <div className="text-sm text-[var(--fg-muted)]">No model selected</div>
              <div className="text-[11px] text-[var(--fg-dim)]">
                {libraryModels.length === 0
                  ? 'Open the library to download your first model.'
                  : 'Pick one from the top-bar dropdown or the library.'}
              </div>
            </>
          )}
        </div>
        <button
          onClick={() => setLibraryOpen(true)}
          className="px-3 py-1.5 rounded text-xs bg-blue-700 hover:bg-blue-600 shrink-0"
        >
          Manage library…
        </button>
      </div>
      <p className="text-[11px] text-[var(--fg-dim)] mt-1 mono truncate" title={libraryRoot}>
        {libraryRoot}
      </p>
    </Field>
  );
}

/**
 * Rough KV-cache size estimate, in GB.
 * Real formula: 2 (K+V) * n_layers * n_ctx * n_kv_heads * head_dim * dtype_bytes.
 * We don't know head dims here, so use a hand-tuned constant calibrated against
 * Llama-3 8B (~0.5 GB at 8K) and Llama-3 70B (~5 GB at 8K). Close enough to
 * keep the slider from over-promising VRAM.
 */
function estimateKvCacheGb(nCtx: number, blockCount: number): number {
  return (nCtx / 1024) * blockCount * 0.0001 * 16;
  // ~ 0.0016 GB per (1K ctx * layer). For a 32-layer model at 4K → 0.2 GB; at 32K → 1.6 GB.
}

function fetchSystemInfo(modelPath: string): Promise<SystemInfo | null> {
  const url = `${BACKEND_HTTP}/system/info?model_path=${encodeURIComponent(modelPath || '')}`;
  return fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
}

function useSystemInfo() {
  const modelPath = useStore((s) => s.settings.modelPath);
  const [info, setInfo] = useState<SystemInfo | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetchSystemInfo(modelPath).then((j) => { if (!cancelled) setInfo(j); });
    return () => { cancelled = true; };
  }, [modelPath]);
  return info;
}

function ContextWindowField() {
  const { settings, setSettings } = useStore();
  const info = useSystemInfo();
  const trained = info?.model?.trainedContext ?? 0;
  const blockCount = info?.model?.blockCount ?? 32;
  // Slider goes up to the model's trained context, with a floor of 32K so
  // models without metadata still get reasonable range, and a hard cap of
  // 262144 (256K) for sanity.
  const sliderMax = Math.min(262144, Math.max(32768, trained || 32768));
  const value = Math.min(settings.nCtx, sliderMax);
  const kvGb = estimateKvCacheGb(value, blockCount);
  const kvWarn = kvGb > 8;
  return (
    <Field
      label={
        `Context window — n_ctx (${value.toLocaleString()}${
          trained ? ` of ${trained.toLocaleString()} trained` : ''
        })`
      }
    >
      <input
        type="range"
        min={1024}
        max={sliderMax}
        step={1024}
        value={value}
        onChange={(e) => setSettings({ nCtx: parseInt(e.target.value, 10) })}
        className="slim w-full"
      />
      <div className={`mt-1 text-[11px] ${kvWarn ? 'text-amber-400' : 'text-[var(--fg-dim)]'}`}>
        Estimated KV-cache: <b>~{kvGb.toFixed(1)} GB</b>
        {' '}(grows linearly with context — eats GPU VRAM if offloaded, else RAM).
        {trained > 0 && value > trained && (
          <span className="text-red-400"> ⚠ above the model's trained context — quality may degrade.</span>
        )}
      </div>
    </Field>
  );
}

function GpuOffloadField() {
  const { settings, setSettings } = useStore();
  const info = useSystemInfo();

  const gpu = info?.gpus?.[0];
  const modelSize = info?.model?.sizeGb ?? 0;
  const gbPerLayer = info?.model?.gbPerLayer ?? 0;
  const blockCount = info?.model?.blockCount ?? 32;

  // Reserve VRAM for:
  //   • llama.cpp's compute buffer (~0.5-1 GB)
  //   • KV-cache for the configured n_ctx
  //   • the 10% slack the backend precheck enforces (vram_budget = offload * 1.10)
  // Solve: offload * 1.10 + reservedKv + 0.7 <= free_vram
  // ⇒ offload <= (free_vram - reservedKv - 0.7) / 1.10
  const kvReserve = estimateKvCacheGb(settings.nCtx || 4096, blockCount);
  const gpuFree = gpu?.free_gb ?? 0;
  // Don't artificially clamp the slider to ~0 while system info is still
  // loading — that's why the slider felt unresponsive on app launch.
  const hasInfo = info != null && (info.ramAvailableGb != null || info.gpus.length > 0);
  const maxOffload = hasInfo
    ? Math.max(0, Math.min(modelSize || 100, (gpuFree - kvReserve - 0.7) / 1.10))
    : (modelSize || 100);
  const sliderMax = Math.max(0.5, maxOffload);

  // Only clamp the displayed thumb — don't overwrite the stored setting just
  // because the user dragged n_ctx up. The stored value re-appears once
  // n_ctx is lowered back and the slider has room again.
  const value = Math.min(settings.gpuOffloadGb, sliderMax);
  const cappedByCtx = settings.gpuOffloadGb > sliderMax + 0.05;
  const resolvedLayers =
    value >= modelSize && modelSize > 0
      ? blockCount + 1
      : gbPerLayer > 0
      ? Math.max(0, Math.round(value / gbPerLayer))
      : 0;

  const ramNeed = Math.max(0, modelSize - value) * 1.1 + 1.2;
  const ramAvail = info?.ramAvailableGb ?? 0;
  const ramShort = info?.ramAvailableGb != null && ramNeed > ramAvail;

  return (
    <Field
      label={
        value <= 0
          ? 'GPU offload — CPU only'
          : value >= modelSize && modelSize > 0
          ? `GPU offload — full model (${modelSize.toFixed(1)} GB · all ${blockCount + 1} layers)`
          : `GPU offload — ${value.toFixed(1)} GB on GPU (${resolvedLayers}/${blockCount + 1} layers)`
      }
    >
      <input
        type="range"
        min={0}
        max={sliderMax}
        step={0.5}
        value={value}
        onChange={(e) => setSettings({ gpuOffloadGb: parseFloat(e.target.value) })}
        className="slim w-full"
      />

      <div className="mt-1 text-[11px] text-[var(--fg-muted)] space-y-0.5">
        {gpu ? (
          <>
            <div>
              GPU: <span className="text-[var(--fg)]">{gpu.name}</span>
              {' — '}
              <span className={value > gpu.free_gb + 0.5 ? 'text-red-400' : 'text-[var(--fg-muted)]'}>
                {value.toFixed(1)} GB used
              </span>
              {' / '}
              <span className="text-[var(--fg-muted)]">{gpu.free_gb.toFixed(1)} GB free</span>
              {' / '}
              <span className="text-[var(--fg-dim)]">{gpu.total_gb.toFixed(1)} GB total</span>
            </div>
            <div className="text-[var(--fg-dim)]">
              Reserved for KV-cache + compute buffer:{' '}
              <span className="text-[var(--fg-muted)]">~{(kvReserve + 0.7).toFixed(1)} GB</span>
              {' '}(slider max capped accordingly)
            </div>
            {cappedByCtx && (
              <div className="text-amber-400">
                Your saved offload ({settings.gpuOffloadGb.toFixed(1)} GB) is above what fits
                with n_ctx = {settings.nCtx.toLocaleString()}. Lower n_ctx to use it again, or
                drop the slider to commit a smaller value.
              </div>
            )}
          </>
        ) : (
          <div className="text-amber-400">
            No NVIDIA GPU detected (or CUDA build of llama-cpp-python not installed).
            Offload above 0 will silently stay on CPU.
          </div>
        )}

        {info?.ramAvailableGb != null && modelSize > 0 && (
          <div>
            RAM need: <span className={ramShort ? 'text-red-400' : 'text-[var(--fg-muted)]'}>{ramNeed.toFixed(1)} GB</span>
            {' / '}
            <span className="text-[var(--fg-muted)]">
              {ramAvail.toFixed(1)} GB free
            </span>
            {info.ramTotalGb && (
              <span className="text-[var(--fg-dim)]"> / {info.ramTotalGb.toFixed(1)} GB total</span>
            )}
          </div>
        )}

        {!gpu && (
          <details className="mt-1">
            <summary className="cursor-pointer text-[var(--fg-muted)] hover:text-[var(--fg)]">
              Install CUDA build of llama-cpp-python…
            </summary>
            <pre className="mt-1 px-2 py-1 bg-app border bd-soft rounded text-[10.5px] mono overflow-auto">
{`pip install --upgrade --force-reinstall --no-cache-dir \\
  --index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \\
  llama-cpp-python`}
            </pre>
          </details>
        )}
      </div>
    </Field>
  );
}

function ModelStatusPanel() {
  const { modelStatus, loadModel, unloadModel, settings } = useStore();

  const dot =
    modelStatus.status === 'loaded'
      ? 'bg-emerald-500'
      : modelStatus.status === 'loading'
      ? 'bg-amber-400 animate-pulse'
      : modelStatus.status === 'error'
      ? 'bg-red-500'
      : 'bg-neutral-600';

  const headline =
    modelStatus.status === 'loaded'
      ? `Loaded${modelStatus.loadMs != null ? ` in ${(modelStatus.loadMs / 1000).toFixed(1)}s` : ''}`
      : modelStatus.status === 'loading'
      ? 'Loading…'
      : modelStatus.status === 'error'
      ? 'Load failed'
      : 'Not loaded';

  const detail =
    modelStatus.status === 'error'
      ? modelStatus.message
      : settings.engine === 'llama-cpp'
      ? settings.modelPath || 'Pick a .gguf below before loading.'
      : `${settings.model} via ${settings.ollamaUrl}`;

  const disabled =
    modelStatus.status === 'loading' ||
    (settings.engine === 'llama-cpp' && !settings.modelPath);

  const loading = modelStatus.status === 'loading';
  const progressPct =
    loading && modelStatus.progress != null
      ? Math.round(modelStatus.progress * 100)
      : null;

  return (
    <div className="rounded-lg border bd-soft bg-app p-3">
      <div className="flex items-center gap-3">
        <span className={`w-2.5 h-2.5 rounded-full ${dot}`} />
        <div className="flex-1 min-w-0">
          <div className="text-sm text-[var(--fg)]">
            {headline}
            {progressPct != null && <span className="ml-2 text-amber-300 mono text-xs">{progressPct}%</span>}
          </div>
          <div className="text-xs text-[var(--fg-muted)] truncate mono">{detail}</div>
        </div>
        {modelStatus.status === 'loaded' ? (
          <button
            onClick={unloadModel}
            className="px-3 py-1.5 rounded-md text-xs bg-soft hover:bg-[var(--bd-soft)]"
          >
            Eject
          </button>
        ) : (
          <button
            onClick={loadModel}
            disabled={disabled}
            className="px-3 py-1.5 rounded-md text-xs bg-emerald-700 hover:bg-emerald-600 disabled:opacity-40 disabled:hover:bg-emerald-700"
          >
            {loading ? 'Loading…' : 'Load model'}
          </button>
        )}
      </div>

      {loading && (
        <div className="mt-3">
          <div className="h-1.5 rounded bg-soft overflow-hidden">
            <div
              className={`h-full bg-amber-400 transition-all ${progressPct == null ? 'animate-pulse' : ''}`}
              style={{ width: progressPct != null ? `${progressPct}%` : '20%' }}
            />
          </div>
          <div className="mt-1 text-[11px] text-[var(--fg-dim)]">{modelStatus.message || 'Loading…'}</div>
        </div>
      )}

      {modelStatus.status === 'loaded' &&
        modelStatus.loadedKey !== modelStatus.currentKey && (
          <div className="mt-2 text-[11px] text-amber-400">
            Settings changed — click Eject then Load to apply.
          </div>
        )}
      {modelStatus.status === 'error' && (
        <div className="mt-2 text-[11px] text-red-300 whitespace-pre-wrap mono">
          {modelStatus.message}
        </div>
      )}
    </div>
  );
}
