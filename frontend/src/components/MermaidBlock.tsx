import React, { useEffect, useRef, useState } from 'react';

// Mermaid is ~600KB; lazy-import it so users who never see a diagram don't pay.
let _mermaidPromise: Promise<typeof import('mermaid')['default']> | null = null;
function loadMermaid() {
  if (!_mermaidPromise) {
    _mermaidPromise = import('mermaid').then((mod) => {
      mod.default.initialize({
        startOnLoad: false,
        theme: 'dark',
        securityLevel: 'strict',
        fontFamily: 'JetBrains Mono, Cascadia Code, Consolas, monospace',
      });
      return mod.default;
    });
  }
  return _mermaidPromise;
}

let _idCounter = 0;

export default function MermaidBlock({ source }: { source: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [showSource, setShowSource] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadMermaid().then(async (mermaid) => {
      if (cancelled || !ref.current) return;
      const id = `mermaid-${++_idCounter}`;
      try {
        const { svg } = await mermaid.render(id, source);
        if (!cancelled && ref.current) {
          ref.current.innerHTML = svg;
          setError(null);
        }
      } catch (e) {
        setError((e as Error).message || 'Render failed');
      }
    });
    return () => { cancelled = true; };
  }, [source]);

  return (
    <div className="my-3 rounded-md border bd-soft bg-app overflow-hidden group">
      <div className="px-3 py-1 border-b border bd-soft bg-side/60 flex items-center justify-between text-[11px]">
        <span className="text-[var(--fg-muted)] mono uppercase tracking-wider">mermaid</span>
        <button
          onClick={() => setShowSource((s) => !s)}
          className="px-2 py-0.5 rounded text-[var(--fg-muted)] hover:bg-soft opacity-0 group-hover:opacity-100 transition-opacity"
        >
          {showSource ? 'Diagram' : 'Source'}
        </button>
      </div>
      {error ? (
        <div className="p-3 text-xs text-red-400 mono">
          Mermaid error: {error}
        </div>
      ) : showSource ? (
        <pre className="overflow-auto scroll text-[12.5px] mono leading-relaxed p-3">{source}</pre>
      ) : (
        <div ref={ref} className="p-3 overflow-auto scroll" />
      )}
    </div>
  );
}
