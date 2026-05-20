import React, { useEffect } from 'react';
import { ShieldAlert, Check, X, CheckCheck } from 'lucide-react';
import { useStore } from '../state/store';
import { api } from '../ipc/bridge';

/**
 * Custom in-app permission prompt. Replaces the jarring native OS dialog —
 * the backend emits a `permission-request`, main.ts forwards it to the
 * renderer (`api.perms.onRequest`, wired in App), we stash it in the store,
 * and this modal renders the decision UI. Denying stops the agent loop with a
 * clear in-chat message (handled backend-side).
 */
export default function PermissionModal() {
  const req = useStore((s) => s.permissionRequest);
  const setReq = useStore((s) => s.setPermissionRequest);
  const setPerms = useStore((s) => s.setPerms);

  // Esc = deny (safe default).
  useEffect(() => {
    if (!req) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') respond(false, false);
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [req]);

  if (!req) return null;

  function respond(granted: boolean, remember: boolean) {
    const r = useStore.getState().permissionRequest;
    if (!r) return;
    setReq(null);
    // respond() persists the perm in the main process (when remember) and
    // returns the updated map — sync it into the store so the Settings toggle
    // immediately reflects it, and it survives app restarts.
    api.perms.respond(r.id, r.tool, granted, remember)
      .then((perms) => { if (perms) setPerms(perms); })
      .catch(() => { /* ignore */ });
  }

  const argEntries = Object.entries(req.args ?? {});

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onMouseDown={(e) => { if (e.target === e.currentTarget) respond(false, false); }}
    >
      <div className="w-[440px] max-w-[90vw] rounded-2xl border bd-strong bg-[var(--bg-app)] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 pt-5 pb-3">
          <div className="w-9 h-9 rounded-lg bg-amber-900/25 text-amber-300 flex items-center justify-center shrink-0">
            <ShieldAlert className="w-4 h-4" />
          </div>
          <div className="min-w-0">
            <div className="text-[14px] font-medium text-[var(--fg)]">Permission needed</div>
            <div className="text-[12px] text-[var(--fg-muted)]">
              The model wants to use <span className="mono text-[var(--fg)]">{req.tool}</span>
            </div>
          </div>
        </div>

        {/* Body */}
        <div className="px-5 pb-4 space-y-3">
          <p className="text-[12.5px] text-[var(--fg-muted)] leading-relaxed">{req.description}</p>
          {argEntries.length > 0 && (
            <div className="rounded-lg border bd-soft bg-[var(--bg-side)]/50 px-3 py-2 space-y-1">
              {argEntries.map(([k, v]) => (
                <div key={k} className="text-[11.5px] mono flex gap-2">
                  <span className="text-[var(--fg-dim)] shrink-0">{k}:</span>
                  <span className="text-[var(--fg)] break-all">
                    {typeof v === 'string' ? v : JSON.stringify(v)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 px-5 py-3.5 border-t bd-soft bg-[var(--bg-side)]/30">
          <button
            onClick={() => respond(false, false)}
            className="flex-1 inline-flex items-center justify-center gap-1.5 h-9 rounded-lg border bd-soft text-[12.5px] text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bd-strong transition"
          >
            <X className="w-3.5 h-3.5" /> Deny
          </button>
          <button
            onClick={() => respond(true, false)}
            className="flex-1 inline-flex items-center justify-center gap-1.5 h-9 rounded-lg border bd-soft text-[12.5px] text-[var(--fg)] hover:bg-[var(--bg-hover)] transition"
          >
            <Check className="w-3.5 h-3.5" /> Allow once
          </button>
          <button
            onClick={() => respond(true, true)}
            className="flex-1 inline-flex items-center justify-center gap-1.5 h-9 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-[12.5px] transition"
            title="Allow now and don't ask again for this capability"
          >
            <CheckCheck className="w-3.5 h-3.5" /> Allow &amp; remember
          </button>
        </div>
      </div>
    </div>
  );
}
