import React, { useEffect, useState } from 'react';
import { Minus, Square, X, Copy as CopyIcon } from 'lucide-react';
import { api } from '../ipc/bridge';

/**
 * Custom title bar. The Electron window was created with `frame: false` and
 * `titleBarStyle: 'hidden'`, so the OS chrome is gone — this component draws
 * the replacement: app name, drag region, and min/maximize/close buttons.
 *
 * The drag region is the area the user grabs to move the window. That's any
 * pixel of this bar EXCEPT the buttons (CSS `-webkit-app-region: no-drag`
 * carves them out so clicks register as button clicks instead of drag init).
 *
 * On macOS the native traffic-light buttons are rendered by the OS at the
 * top-left (because we used `titleBarStyle: 'hidden'`) — we hide our own
 * buttons there to avoid double-controls.
 */
export default function TitleBar() {
  const [maximized, setMaximized] = useState<boolean>(
    () => (typeof api?.window?.isMaximized === 'function' ? api.window.isMaximized() : false),
  );
  const isMac = typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform);

  useEffect(() => {
    if (!api?.window?.onMaximizedChange) return;
    return api.window.onMaximizedChange(setMaximized);
  }, []);

  return (
    <div
      className="h-9 flex items-center select-none border-b bd-soft bg-app text-[var(--fg-muted)]"
      style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}
    >
      {/* Leave space for macOS traffic lights, which the OS draws at top-left */}
      {isMac && <div className="w-20 shrink-0" aria-hidden />}

      <div className="px-3 text-[12px] flex items-center gap-2 min-w-0">
        <span className="text-[var(--fg)] font-medium tracking-tight">Local AI Studio</span>
      </div>

      <div className="flex-1" />

      {!isMac && (
        <div
          className="flex h-full"
          style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}
        >
          <TitleBarButton onClick={() => api.window.minimize()} title="Minimize">
            <Minus className="w-3.5 h-3.5" />
          </TitleBarButton>
          <TitleBarButton onClick={() => api.window.maximize()} title={maximized ? 'Restore' : 'Maximize'}>
            {maximized
              ? <CopyIcon className="w-3 h-3 rotate-90" />
              : <Square className="w-3 h-3" />}
          </TitleBarButton>
          <TitleBarButton onClick={() => api.window.close()} title="Close" danger>
            <X className="w-3.5 h-3.5" />
          </TitleBarButton>
        </div>
      )}
    </div>
  );
}

function TitleBarButton({
  children, onClick, title, danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      // Explicit bg-app + outline-none — so the resting state always matches
      // the bar exactly and we get no browser focus-ring artifact (which
      // would show as a stray accent border on Windows).
      className={`bg-app outline-none focus:outline-none w-12 h-full flex items-center justify-center text-[var(--fg-muted)] transition-colors ${
        danger
          ? 'hover:bg-red-600 hover:text-white'
          : 'hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]'
      }`}
    >
      {children}
    </button>
  );
}
