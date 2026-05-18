import React from 'react';
import { lineDiff, diffStats } from '../lib/diff';

interface Props {
  before: string;
  after: string;
  maxLines?: number;
}

export default function InlineDiff({ before, after, maxLines = 200 }: Props) {
  const ops = lineDiff(before, after);
  const { adds, dels } = diffStats(ops);
  const truncated = ops.length > maxLines;
  const display = truncated ? ops.slice(0, maxLines) : ops;

  return (
    <div className="rounded border bd-soft bg-app overflow-hidden">
      <div className="px-3 py-1 border-b border bd-soft text-[11px] text-[var(--fg-muted)] flex items-center gap-3">
        <span className="text-emerald-400 mono">+{adds}</span>
        <span className="text-red-400 mono">-{dels}</span>
        {truncated && <span className="text-amber-400">truncated to {maxLines} lines</span>}
      </div>
      <pre className="text-[12px] mono overflow-auto scroll max-h-72">
        {display.map((op, i) => {
          const bg =
            op.kind === 'add' ? 'bg-emerald-900/30' :
            op.kind === 'del' ? 'bg-red-900/30' :
            '';
          const sign =
            op.kind === 'add' ? '+' :
            op.kind === 'del' ? '-' : ' ';
          const color =
            op.kind === 'add' ? 'text-emerald-300' :
            op.kind === 'del' ? 'text-red-300' :
            'text-[var(--fg-muted)]';
          return (
            <div key={i} className={`${bg} ${color} flex`}>
              <span className="w-5 text-center select-none opacity-60">{sign}</span>
              <span className="whitespace-pre flex-1 px-1">{op.line || ' '}</span>
            </div>
          );
        })}
      </pre>
    </div>
  );
}
