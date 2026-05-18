import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, CircleX, Library, RefreshCcw, Circle, Loader2, Eye } from 'lucide-react';
import { useStore } from '../state/store';
import { VISION_HANDLER_OPTIONS } from '../state/types';

export default function ModelPicker() {
  const {
    modelStatus,
    libraryModels,
    selectLibraryModel,
    unloadModel,
    setLibraryOpen,
    refreshLibrary,
    settings,
    setSettings,
  } = useStore();

  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => { if (open) refreshLibrary(); }, [open, refreshLibrary]);

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  const referencedPath = modelStatus.modelPath || settings.modelPath || '';
  const referencedId = referencedPath.split(/[\\/]/).pop() ?? '';
  const referencedInLibrary = !!referencedId && libraryModels.some((m) => m.id === referencedId);

  const loaded = modelStatus.status === 'loaded';
  const loading = modelStatus.status === 'loading';
  const error = modelStatus.status === 'error';
  const idle = modelStatus.status === 'idle';

  const dotColor =
    loaded ? 'text-emerald-400' :
    loading ? 'text-amber-400' :
    error ? 'text-red-400' :
    referencedInLibrary ? 'text-blue-400' : 'text-[var(--fg-dim)]';

  const progressPct = loading && modelStatus.progress != null
    ? Math.round(modelStatus.progress * 100) : null;

  const currentName = (() => {
    if (loaded && modelStatus.modelPath) return modelStatus.modelPath.split(/[\\/]/).pop()?.replace(/\.gguf$/i, '');
    if (referencedInLibrary) return referencedId.replace(/\.gguf$/i, '');
    return 'No model loaded';
  })();

  const statusText =
    loaded ? (modelStatus.loadMs != null ? `Ready · ${(modelStatus.loadMs / 1000).toFixed(1)}s` : 'Ready') :
    loading ? (modelStatus.message || 'Loading…') :
    error ? 'Error' :
    'Click to choose a model';

  const activeId = modelStatus.modelPath?.split(/[\\/]/).pop() ?? null;
  const showLoad = idle && referencedInLibrary;
  const showRetry = error && referencedInLibrary;

  return (
    <div ref={wrapRef} className="relative flex items-center gap-1">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex flex-col gap-0.5 px-2.5 py-1.5 rounded-lg hover:bg-[var(--bg-hover)] text-[13px] text-[var(--fg)] min-w-[260px]"
      >
        <div className="flex items-center gap-2 w-full">
          {loading ? (
            <Loader2 className={`w-3 h-3 shrink-0 animate-spin ${dotColor}`} />
          ) : (
            <Circle className={`w-2 h-2 shrink-0 fill-current ${dotColor}`} />
          )}
          <span className="truncate flex-1 text-left">{currentName}</span>
          {progressPct != null && <span className="text-amber-300 mono text-[11px]">{progressPct}%</span>}
          <ChevronDown className="w-3.5 h-3.5 text-[var(--fg-dim)]" />
        </div>
        {loading && (
          <div className="h-0.5 w-full rounded-full bg-[var(--bd-soft)] overflow-hidden">
            <div
              className="h-full bg-amber-400 transition-all"
              style={{
                width: progressPct != null ? `${progressPct}%` : '15%',
                animation: progressPct == null ? 'pulse 1.4s ease-in-out infinite' : undefined,
              }}
            />
          </div>
        )}
      </button>

      {showLoad && (
        <button
          onClick={() => selectLibraryModel(referencedId)}
          className="px-3 py-1.5 rounded-lg text-[13px] bg-emerald-700 hover:bg-emerald-600 text-white"
        >
          Load
        </button>
      )}
      {showRetry && (
        <button
          onClick={() => selectLibraryModel(referencedId)}
          className="px-3 py-1.5 rounded-lg text-[13px] bg-amber-700 hover:bg-amber-600 text-white inline-flex items-center gap-1"
        >
          <RefreshCcw className="w-3.5 h-3.5" /> Retry
        </button>
      )}
      {loaded && (
        <button
          onClick={unloadModel}
          className="px-3 py-1.5 rounded-lg text-[13px] text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--fg)] inline-flex items-center gap-1"
          title="Eject"
        >
          <CircleX className="w-3.5 h-3.5" /> Eject
        </button>
      )}

      {open && (
        <div className="absolute right-0 top-full mt-1 w-[380px] rounded-xl border bd-strong bg-[var(--bg-app)] shadow-2xl z-40 overflow-hidden">
          <div className="px-3 py-2 border-b bd-soft">
            <div className="flex items-center justify-between">
              <span className="text-[11px] uppercase tracking-wider text-[var(--fg-dim)]">Status</span>
              <span className="text-[12px] text-[var(--fg-muted)]">
                {loading && progressPct != null ? `${progressPct}%` : statusText}
              </span>
            </div>
            {loaded && (() => {
              const loadedId = modelStatus.modelPath?.split(/[\\/]/).pop() ?? '';
              const libEntry = libraryModels.find((m) => m.id === loadedId);
              const paired = !!libEntry?.isVision;
              const active = !!modelStatus.visionActive;
              if (!paired && !active) return null;
              return (
                <>
                  <div className="mt-1 flex items-center justify-between text-[11px]">
                    <span className="inline-flex items-center gap-1 text-[var(--fg-dim)]">
                      <Eye className="w-3 h-3" />
                      Vision
                    </span>
                    {active ? (
                      <span
                        className="text-emerald-400"
                        title={`Handler: ${modelStatus.visionHandler ?? 'active'}`}
                      >
                        on · {modelStatus.visionHandler?.replace('ChatHandler', '') ?? 'active'}
                      </span>
                    ) : (
                      <span
                        className="text-amber-400"
                        title="Paired mmproj didn't match any known vision family — pick the family below to force it."
                      >
                        off · auto-detect failed
                      </span>
                    )}
                  </div>
                  {/* When auto-detect failed but the user has a paired
                      mmproj, let them pick the handler family directly.
                      Saving the setting + re-selecting the model forces a
                      reload through the new vision_handler key. */}
                  {paired && !active && (
                    <div className="mt-1.5">
                      <select
                        value={settings.visionHandler ?? ''}
                        onChange={async (e) => {
                          const v = e.target.value;
                          await setSettings({ visionHandler: v });
                          if (loadedId) await selectLibraryModel(loadedId);
                        }}
                        className="w-full bg-[var(--bg-app)] border bd-soft rounded px-2 py-1 text-[11px] outline-none"
                        title="Pick the vision handler family for this mmproj. Wrong choice = noise; right choice = working vision."
                      >
                        {VISION_HANDLER_OPTIONS.map((o) => (
                          <option key={o.value} value={o.value}>{o.label}</option>
                        ))}
                      </select>
                    </div>
                  )}
                </>
              );
            })()}
            {loading && (
              <>
                <div className="mt-2 h-1.5 rounded bg-[var(--bd-soft)] overflow-hidden">
                  <div
                    className={`h-full bg-amber-400 transition-all ${progressPct == null ? 'animate-pulse' : ''}`}
                    style={{ width: progressPct != null ? `${progressPct}%` : '20%' }}
                  />
                </div>
                <div className="mt-1 text-[11px] text-[var(--fg-dim)] truncate">
                  {modelStatus.message || 'Loading…'}
                </div>
              </>
            )}
          </div>

          <div className="max-h-72 overflow-auto scroll py-1">
            {libraryModels.length === 0 ? (
              <div className="p-4 text-[12px] text-[var(--fg-dim)] text-center">
                Your library is empty.
                <br />
                Open the library to download your first model.
              </div>
            ) : (
              libraryModels.map((m) => {
                const active = m.id === activeId && loaded;
                return (
                  <button
                    key={m.id}
                    onClick={async () => { setOpen(false); await selectLibraryModel(m.id); }}
                    className={`w-full text-left px-3 py-2 hover:bg-[var(--bg-hover)] flex items-center gap-2 ${
                      active ? 'bg-[var(--bg-hover)]' : ''
                    }`}
                  >
                    <Circle className={`w-1.5 h-1.5 shrink-0 fill-current ${active ? 'text-emerald-400' : 'text-[var(--fg-dim)]'}`} />
                    <div className="min-w-0 flex-1">
                      <div className="text-[13px] text-[var(--fg)] truncate flex items-center gap-1.5">
                        <span className="truncate">{m.name}</span>
                        {m.isVision && (
                          <span
                            className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9.5px] bg-violet-900/40 text-violet-300 shrink-0"
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
                      </div>
                    </div>
                  </button>
                );
              })
            )}
          </div>

          <div className="border-t bd-soft p-1.5 flex gap-1">
            {loaded && (
              <button
                onClick={async () => { setOpen(false); await unloadModel(); }}
                className="flex-1 inline-flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg text-[12.5px] text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]"
              >
                <CircleX className="w-3.5 h-3.5" /> Eject
              </button>
            )}
            <button
              onClick={() => { setOpen(false); setLibraryOpen(true); }}
              className="flex-1 inline-flex items-center justify-center gap-1.5 px-2 py-1.5 rounded-lg text-[12.5px] bg-[var(--bg-hover)] hover:bg-[var(--bd-soft)] text-[var(--fg)]"
            >
              <Library className="w-3.5 h-3.5" /> Manage library
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
