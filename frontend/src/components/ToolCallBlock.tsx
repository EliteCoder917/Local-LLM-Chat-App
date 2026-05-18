import React, { useState } from 'react';
import {
  FileText, FilePen, FolderPlus, Trash2, Move, Tag, Folder, Search,
  Code2, TerminalSquare, Play, Brain, Wrench,
  ChevronDown, ChevronRight, Loader2, Check, X,
} from 'lucide-react';
import type { ToolCall } from '../state/types';

const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  read_file: FileText,
  write_file: FilePen,
  create_folder: FolderPlus,
  delete_file: Trash2,
  move_file: Move,
  rename_file: Tag,
  list_dir: Folder,
  search_text: Search,
  run_python: Code2,
  run_shell: TerminalSquare,
  run_script: Play,
  get_memory: Brain,
  set_memory: Brain,
  list_memory: Brain,
  delete_memory: Brain,
};

function summarize(tc: ToolCall): string {
  const a = tc.args ?? {};
  switch (tc.tool) {
    case 'read_file': case 'list_dir': case 'create_folder': case 'delete_file':
      return String(a.path ?? '');
    case 'write_file': {
      const len = typeof a.content === 'string' ? a.content.length : 0;
      return `${a.path ?? ''}  (${len.toLocaleString()} chars)`;
    }
    case 'move_file': return `${a.src ?? ''} → ${a.dst ?? ''}`;
    case 'rename_file': return `${a.old ?? ''} → ${a.new ?? ''}`;
    case 'search_text': return `"${a.query ?? ''}" in ${a.path ?? '.'}`;
    case 'run_python': {
      const code = typeof a.code === 'string' ? a.code : '';
      return code.split('\n')[0].slice(0, 80) + (code.length > 80 ? '…' : '');
    }
    case 'run_shell': return String(a.command ?? '').slice(0, 100);
    case 'run_script': return String(a.path ?? '');
    case 'get_memory': case 'set_memory': case 'delete_memory':
      return String(a.key ?? '');
    default: return '';
  }
}

export default function ToolCallBlock({ tc }: { tc: ToolCall }) {
  const [open, setOpen] = useState(false);
  const Icon = ICONS[tc.tool] ?? Wrench;
  const summary = summarize(tc);

  const statusPill =
    tc.status === 'running' ? (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] bg-amber-900/40 text-amber-300">
        <Loader2 className="w-2.5 h-2.5 animate-spin" /> running
      </span>
    ) : tc.status === 'done' ? (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] bg-emerald-900/40 text-emerald-300">
        <Check className="w-2.5 h-2.5" /> done
      </span>
    ) : tc.status === 'error' ? (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] bg-red-900/40 text-red-300">
        <X className="w-2.5 h-2.5" /> error
      </span>
    ) : (
      <span className="px-1.5 py-0.5 rounded text-[10px] bg-[var(--bg-hover)] text-[var(--fg-dim)]">queued</span>
    );

  return (
    <div className="my-2 rounded-lg border bd-soft bg-[var(--bg-side)]/40 overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full px-3 py-1.5 flex items-center gap-2 text-[12.5px] hover:bg-[var(--bg-hover)]"
      >
        <Icon className="w-3.5 h-3.5 shrink-0 text-[var(--fg-muted)]" />
        <span className="text-[var(--fg)] mono">{tc.tool}</span>
        {summary && <span className="text-[var(--fg-dim)] mono truncate flex-1 text-left">  {summary}</span>}
        {!summary && <span className="flex-1" />}
        {statusPill}
        {open
          ? <ChevronDown className="w-3.5 h-3.5 text-[var(--fg-dim)] ml-1" />
          : <ChevronRight className="w-3.5 h-3.5 text-[var(--fg-dim)] ml-1" />}
      </button>

      {open && (
        <div className="border-t bd-soft p-2 text-[12px] space-y-2">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-[var(--fg-dim)] mb-1">Args</div>
            <pre className="bg-black/40 rounded p-2 overflow-auto scroll mono text-[11.5px]">
              {JSON.stringify(tc.args, null, 2)}
            </pre>
          </div>
          {tc.result && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-[var(--fg-dim)] mb-1">Result</div>
              <pre className="bg-black/40 rounded p-2 overflow-auto scroll mono text-[11.5px] whitespace-pre-wrap max-h-72">
                {tc.result}
              </pre>
            </div>
          )}
          {tc.error && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-red-400 mb-1">Error</div>
              <pre className="bg-red-900/30 rounded p-2 mono text-[11.5px] whitespace-pre-wrap">
                {tc.error}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
