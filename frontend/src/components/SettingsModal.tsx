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
  const reloadModel = useStore((s) => s.reloadModel);
  const modelLoaded = useStore((s) => s.modelStatus.status === 'loaded');
  const [tab, setTab] = useState<TabKey>('model');

  // Snapshot the reload-only settings as the saved baseline, so we can detect
  // unsaved changes on close and offer to revert them. (Temperature etc. apply
  // live and never need this.)
  const [baseline, setBaseline] = useState(() => ({
    nCtx: settings.nCtx,
    gpuOffloadGb: settings.gpuOffloadGb,
  }));
  const [confirmClose, setConfirmClose] = useState(false);

  // Whenever the model actually (re)loads — whether via the close dialog OR the
  // inline "Save & reload" banner — the current settings BECOME the saved
  // baseline, so we don't then falsely report them as unsaved. Keyed on the
  // loaded engine identity.
  const loadedKey = useStore((s) => s.modelStatus.loadedKey);
  useEffect(() => {
    setBaseline({ nCtx: settings.nCtx, gpuOffloadGb: settings.gpuOffloadGb });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loadedKey]);

  const dirty = modelLoaded && (
    settings.nCtx !== baseline.nCtx || settings.gpuOffloadGb !== baseline.gpuOffloadGb
  );

  // Intercept close: if there are unsaved reload-only changes, ask first.
  function requestClose() {
    if (dirty) setConfirmClose(true);
    else onClose();
  }
  async function saveAndClose() {
    setConfirmClose(false);
    await reloadModel();
    onClose();
  }
  function discardAndClose() {
    // Revert the live settings (this also pushes to the backend, so the loaded
    // model is back in sync and the "Save & reload" banner clears).
    void setSettings({ nCtx: baseline.nCtx, gpuOffloadGb: baseline.gpuOffloadGb });
    setConfirmClose(false);
    onClose();
  }

  // Esc closes (guarded by the unsaved-changes check).
  useEffect(() => {
    const fn = (e: KeyboardEvent) => { if (e.key === 'Escape') requestClose(); };
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dirty, onClose]);

  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      onClick={requestClose}
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
              onClick={requestClose}
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
                      <option value="llama-cpp">Local model file (.gguf)</option>
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
                      <span>0 · precise</span><span>0.2 · default</span><span>2 · wild</span>
                    </div>
                  </Field>
                </Section>

                {settings.engine === 'llama-cpp' && (
                  <Section title="Runtime">
                    <ContextWindowField />
                    {/* Apple Silicon uses Metal with unified memory — there's no
                        separate VRAM pool to offload to, and the backend now
                        auto-sizes layers from free RAM (see llama_cpp_engine.py).
                        The slider would be misleading here, so hide it on Mac.
                        Windows/Linux still see it. */}
                    {!/Mac/i.test(
                      (typeof navigator !== 'undefined'
                        ? navigator.platform || navigator.userAgent
                        : '')
                    ) && <GpuOffloadField />}
                  </Section>
                )}

                <Section title="Images">
                  <ToggleRow
                    label="Send images to the model"
                    desc="Only useful with a vision-capable model. Otherwise the model just sees a placeholder."
                    checked={settings.sendImagesAsBase64}
                    onChange={(v) => setSettings({ sendImagesAsBase64: v })}
                  />
                </Section>

                <Section title="System prompt" hint="Instructions prepended to every conversation.">
                  <SystemPromptEditor />
                </Section>
              </div>
            )}

            {tab === 'permissions' && (
              <div className="space-y-5">
                <Section title="Tool permissions" hint="Allow the model to act on its own, or be prompted each time.">
                  <PermissionToggles />
                </Section>
                <Section title="Saved memory" hint="Notes the model has saved. Shared across all chats.">
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
                <Section title="Tools by tab">
                  <div className="text-[12.5px] text-[var(--fg-muted)] space-y-2 leading-relaxed">
                    <div>
                      <span className="text-[var(--fg)] font-medium">Chat tab</span> — plain chat. No tools, no file access.
                    </div>
                    <div>
                      <span className="text-[var(--fg)] font-medium">Code tab</span> — model can read, write, and run code (with the permissions you've granted).
                    </div>
                  </div>
                </Section>
                <Section title="Agent loop" hint="How many tool calls the model can chain per response (Code tab only).">
                  <Field label={`Max iterations — ${settings.maxIterations}`}>
                    <input
                      type="range"
                      min={1}
                      max={30}
                      value={settings.maxIterations}
                      onChange={(e) => setSettings({ maxIterations: parseInt(e.target.value, 10) })}
                      className="slim w-full"
                    />
                  </Field>
                </Section>
              </div>
            )}
          </div>
        </div>
      </div>

      {confirmClose && (
        <div
          className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60"
          onClick={(e) => { e.stopPropagation(); }}
        >
          <div className="w-[400px] max-w-[90vw] rounded-2xl border bd-strong bg-app shadow-2xl p-5">
            <div className="text-[14px] font-medium text-[var(--fg)] mb-1">Unsaved changes</div>
            <p className="text-[12.5px] text-[var(--fg-muted)] leading-relaxed mb-4">
              You changed settings that need a model reload to apply (context window
              and/or GPU offload). Save and reload now, or discard the changes?
            </p>
            <div className="flex gap-2 justify-end">
              <button
                onClick={discardAndClose}
                className="px-3 py-1.5 rounded-md text-[12.5px] border bd-soft text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bd-strong"
              >
                Discard
              </button>
              <button
                onClick={saveAndClose}
                className="px-3 py-1.5 rounded-md text-[12.5px] bg-emerald-600 hover:bg-emerald-500 text-white"
              >
                Save &amp; reload
              </button>
            </div>
          </div>
        </div>
      )}
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
    <div className="rounded-xl border bd-soft bg-side p-4 space-y-3.5">
      <div className="space-y-0.5">
        <div className="text-[13.5px] font-medium text-[var(--fg)] tracking-tight">{title}</div>
        {hint && <div className="text-[11.5px] text-[var(--fg-dim)] leading-snug">{hint}</div>}
      </div>
      {children}
    </div>
  );
}

function ToggleRow({
  label, desc, checked, onChange,
}: { label: string; desc?: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex-1 min-w-0">
        <div className="text-[13px] text-[var(--fg)]">{label}</div>
        {desc && <div className="text-[11.5px] text-[var(--fg-dim)] leading-snug mt-0.5">{desc}</div>}
      </div>
      <Toggle checked={checked} onChange={onChange} />
    </div>
  );
}

function Field({ label, children }: { label: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="space-y-1.5">
      <div className="text-[11.5px] text-[var(--fg-muted)]">{label}</div>
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
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="text-[12px] text-[var(--fg-muted)]">
          {entries.length === 0 ? 'No entries yet.' : `${entries.length} ${entries.length === 1 ? 'entry' : 'entries'}`}
        </div>
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
      <div className="rounded border bd-soft bg-app max-h-64 overflow-auto scroll">
        {loading && <div className="p-3 text-xs text-[var(--fg-dim)]">Loading…</div>}
        {!loading && entries.length === 0 && (
          <div className="p-3 text-xs text-[var(--fg-dim)]">
            Nothing saved yet. Enable the memory permission to let the model save notes.
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
  const { setLibraryOpen, libraryModels, modelStatus } = useStore();
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
          Open library
        </button>
      </div>
    </Field>
  );
}

/**
 * Accurate KV-cache size estimate in GB.
 * Real formula: 2 (K+V) × n_layers × n_ctx × n_kv_heads × head_dim × dtype_bytes.
 *
 * Reads head_count_kv and rope_dim from the GGUF metadata when available
 * (which gives a 2-3× more accurate number on GQA models like Qwen3-MoE,
 * where the old constant under-estimated significantly). Falls back to
 * the hand-tuned constant when those fields aren't present.
 *
 * `dtypeBytes` defaults to 2 (FP16). Llama.cpp uses FP16 KV by default;
 * some builds use BF16 which is also 2 bytes; INT8 KV cache (rare) would
 * be 1 byte but we don't probe for that.
 */
function estimateKvCacheGb(
  nCtx: number,
  blockCount: number,
  info?: SystemInfo | null,
  dtypeBytes: number = 2,
): number {
  const model = info?.model;
  const headDim = model?.ropeDim
    ?? (model?.embeddingLength && model?.headCount
        ? Math.floor(model.embeddingLength / model.headCount)
        : 0);
  const kvHeads = model?.headCountKv ?? model?.headCount ?? 0;
  if (headDim > 0 && kvHeads > 0 && blockCount > 0) {
    const bytes = 2 * blockCount * nCtx * kvHeads * headDim * dtypeBytes;
    return bytes / (1024 ** 3);
  }
  // Fallback hand-tuned constant for when GGUF metadata is incomplete.
  return (nCtx / 1024) * blockCount * 0.0001 * 16;
}

function fetchSystemInfo(modelPath: string): Promise<SystemInfo | null> {
  const url = `${BACKEND_HTTP}/system/info?model_path=${encodeURIComponent(modelPath || '')}`;
  return fetch(url).then((r) => (r.ok ? r.json() : null)).catch(() => null);
}

function useSystemInfo() {
  const modelPath = useStore((s) => s.settings.modelPath);
  // Also refetch when the model loads or unloads — VRAM free shifts by gigabytes
  // and the panel was showing pre-eject numbers because the deps array missed it.
  const status = useStore((s) => s.modelStatus.status);
  const [info, setInfo] = useState<SystemInfo | null>(null);
  useEffect(() => {
    let cancelled = false;
    fetchSystemInfo(modelPath).then((j) => { if (!cancelled) setInfo(j); });
    return () => { cancelled = true; };
  }, [modelPath, status]);
  return info;
}

function ContextWindowField() {
  const { settings, setSettings } = useStore();
  const info = useSystemInfo();
  const loadedNCtx = useStore((s) => s.modelStatus.nCtx);
  const trained = info?.model?.trainedContext ?? 0;
  const blockCount = info?.model?.blockCount ?? 32;
  // Slider goes up to the model's trained context, with a floor of 32K so
  // models without metadata still get reasonable range, and a hard cap of
  // 262144 (256K) for sanity.
  const sliderMax = Math.min(262144, Math.max(32768, trained || 32768));
  // nCtx <= 0 is the Auto sentinel: the backend sizes the window from memory
  // at load time. Show the resolved value (from the loaded model) when we
  // have it; fall back to a sensible slider position otherwise.
  const isAuto = settings.nCtx <= 0;
  const value = isAuto
    ? (loadedNCtx && loadedNCtx > 0 ? loadedNCtx : Math.min(8192, sliderMax))
    : Math.min(settings.nCtx, sliderMax);
  const kvGb = estimateKvCacheGb(value, blockCount, info);
  const kvWarn = kvGb > 8;
  return (
    <Field
      label={
        <div className="flex items-center justify-between gap-2">
          <span>
            {`Context window — ${value.toLocaleString()} tokens`}
            {isAuto && <span className="ml-2 text-[10px] text-emerald-400 uppercase tracking-wide">auto</span>}
            {trained ? <span className="text-[var(--fg-dim)]"> · trained {trained.toLocaleString()}</span> : ''}
          </span>
          <button
            onClick={() => setSettings({ nCtx: isAuto ? value : 0 })}
            className={`text-[10.5px] px-2 py-0.5 rounded border transition ${
              isAuto
                ? 'bg-emerald-700/30 border-emerald-700/60 text-emerald-300 hover:bg-emerald-700/50'
                : 'bd-soft text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bd-strong'
            }`}
            title={isAuto
              ? 'Auto-sizing the context to fit memory at load. Click to set it manually.'
              : 'Let the app pick the largest context the system can handle.'}
          >
            {isAuto ? 'Auto on' : 'Auto'}
          </button>
        </div>
      }
    >
      <input
        type="range"
        min={1024}
        max={sliderMax}
        step={1024}
        value={value}
        disabled={isAuto}
        onChange={(e) => setSettings({ nCtx: parseInt(e.target.value, 10) })}
        className={`slim w-full ${isAuto ? 'opacity-60 pointer-events-none' : ''}`}
      />
      <div className={`mt-1 text-[11px] ${kvWarn ? 'text-amber-400' : 'text-[var(--fg-dim)]'}`}>
        Memory cost: <b>~{kvGb.toFixed(1)} GB</b>
        {trained > 0 && value > trained && (
          <span className="text-red-400"> · above the model's trained range — quality may degrade.</span>
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

  // Slider max = the n_ctx-aware safe ceiling, snapped down to a step
  // boundary. Track ends exactly where you can safely drag to, so you can
  // use the full available VRAM (no dead zone, no cutoff).
  //
  // Note: changing n_ctx will visually shift the thumb's POSITION on the
  // track because HTML range inputs render the thumb as a percentage of
  // track length, and the track length here depends on n_ctx via safeMax.
  // The saved VALUE doesn't change — only the visual percentage does.
  const kvReserve = estimateKvCacheGb(settings.nCtx || 4096, blockCount, info);
  const gpuFree = gpu?.free_gb ?? 0;
  const STEP = 0.1;
  // `safeMax` is the maximum offload the math says will fit alongside the
  // KV cache + a 0.7 GB compute-buffer headroom + a 10% safety slack.
  const safeMax = gpu != null
    ? Math.max(0, Math.min(modelSize || 100, (gpuFree - kvReserve - 0.7) / 1.10))
    : (modelSize || 100);
  const sliderMax = Math.max(STEP, Math.floor(safeMax / STEP) * STEP);
  // Auto mode: gpuOffloadGb sentinel of -1 means "let the app pick the
  // max fittable value at load time." We display the computed safeMax
  // as the effective value and disable the slider while auto is on.
  const isAuto = settings.gpuOffloadGb < 0;
  const value = isAuto ? sliderMax : Math.min(settings.gpuOffloadGb, sliderMax);
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
        <div className="flex items-center justify-between gap-2">
          <span>
            {value <= 0
              ? 'GPU offload — CPU only'
              : value >= modelSize && modelSize > 0
              ? `GPU offload — full model on GPU (${blockCount + 1} layers)`
              : `GPU offload — ${value.toFixed(1)} GB (${resolvedLayers} of ${blockCount + 1} layers)`}
            {isAuto && <span className="ml-2 text-[10px] text-emerald-400 uppercase tracking-wide">auto</span>}
          </span>
          <button
            onClick={() => setSettings({ gpuOffloadGb: isAuto ? value : -1 })}
            className={`text-[10.5px] px-2 py-0.5 rounded border transition ${
              isAuto
                ? 'bg-emerald-700/30 border-emerald-700/60 text-emerald-300 hover:bg-emerald-700/50'
                : 'bd-soft text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bd-strong'
            }`}
            title={isAuto
              ? 'Currently auto-sizing offload to fit available VRAM. Click to fix at the current value and use the slider.'
              : 'Auto-size offload to maximum that fits alongside KV-cache and compute buffer.'}
          >
            {isAuto ? 'Auto on' : 'Auto'}
          </button>
        </div>
      }
    >
      <input
        type="range"
        min={0}
        max={sliderMax}
        step={STEP}
        value={value}
        disabled={isAuto}
        onChange={(e) => setSettings({ gpuOffloadGb: parseFloat(e.target.value) })}
        className={`slim w-full ${isAuto ? 'opacity-60 pointer-events-none' : ''}`}
      />

      <div className="mt-1.5 text-[11px] text-[var(--fg-muted)] space-y-1">
        {gpu ? (
          <>
            <div className="flex items-center justify-between">
              <span className="text-[var(--fg-dim)]">{gpu.name}</span>
              <span className="mono text-[var(--fg-muted)]">
                <span className={value > gpu.free_gb + 0.5 ? 'text-red-400' : 'text-[var(--fg)]'}>
                  {value.toFixed(1)}
                </span>
                {' / '}
                {gpu.free_gb.toFixed(1)} GB free
              </span>
            </div>
            <div className="text-[var(--fg-dim)]">
              {' '}
              Safe up to <b className="text-[var(--fg-muted)]">{safeMax.toFixed(1)} GB</b> at the current context size.
            </div>
          </>
        ) : (
          <div className="text-amber-400">
            No GPU detected — model will run on CPU.
          </div>
        )}

        {info?.ramAvailableGb != null && modelSize > 0 && (
          <div className="flex items-center justify-between text-[var(--fg-dim)]">
            <span>RAM use</span>
            <span className="mono">
              <span className={ramShort ? 'text-red-400' : 'text-[var(--fg-muted)]'}>{ramNeed.toFixed(1)}</span>
              {' / '}
              {ramAvail.toFixed(1)} GB free
            </span>
          </div>
        )}
      </div>
    </Field>
  );
}

function ModelStatusPanel() {
  const { modelStatus, loadModel, unloadModel, reloadModel, settings } = useStore();

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
          <div className="mt-2 flex items-center justify-between gap-2 rounded-md border border-amber-700/40 bg-amber-900/15 px-3 py-2">
            <span className="text-[11.5px] text-amber-300">
              Settings changed — reload the model to apply.
            </span>
            <button
              onClick={reloadModel}
              className="shrink-0 px-3 py-1 rounded-md text-[11.5px] bg-amber-600 hover:bg-amber-500 text-white"
            >
              Save &amp; reload
            </button>
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
