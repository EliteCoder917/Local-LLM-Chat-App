import React from 'react';
import {
  Eye, PenSquare, Trash2, TerminalSquare, Globe, Brain, ExternalLink,
} from 'lucide-react';
import { useStore } from '../state/store';
import Toggle from './Toggle';

interface Row {
  key: string;
  label: string;
  desc: string;
  icon: React.ComponentType<{ className?: string }>;
  danger?: boolean;
}

const ROWS: Row[] = [
  { key: 'file.read',   icon: Eye,             label: 'Read files',       desc: 'The model can read files in your workspace.' },
  { key: 'file.write',  icon: PenSquare,       label: 'Write files',      desc: 'Create or overwrite files in the workspace.', danger: true },
  { key: 'file.delete', icon: Trash2,          label: 'Delete files',     desc: 'Remove files or folders. Destructive.',       danger: true },
  { key: 'exec.code',   icon: TerminalSquare,  label: 'Run code & commands', desc: 'Execute Python, shell commands (PowerShell), and saved scripts.', danger: true },
  { key: 'system.open', icon: ExternalLink,    label: 'Open apps & files', desc: 'Launch programs, open files/folders, or open URLs (e.g. "open Steam").', danger: true },
  { key: 'network',     icon: Globe,           label: 'Network access',   desc: 'Allow tools that make outbound network requests.', danger: true },
  { key: 'memory',      icon: Brain,           label: 'Persistent memory', desc: 'Read / write notes that survive across all chats.' },
];

export default function PermissionToggles() {
  const { perms, setPerm } = useStore();

  return (
    <div className="divide-y divide-[var(--bd-soft)]">
      {ROWS.map(({ key, label, desc, icon: Icon, danger }) => (
        <div
          key={key}
          className="flex items-start gap-3 py-3 first:pt-0 last:pb-0"
        >
          <div className={`w-7 h-7 rounded-md flex items-center justify-center shrink-0 ${
            danger ? 'bg-amber-900/20 text-amber-300' : 'bg-[var(--bg-hover)] text-[var(--fg-muted)]'
          }`}>
            <Icon className="w-3.5 h-3.5" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 text-[13px] text-[var(--fg)]">
              {label}
              {danger && (
                <span className="text-[9.5px] uppercase tracking-wider text-amber-400 px-1.5 py-0.5 rounded bg-amber-900/30">
                  danger
                </span>
              )}
            </div>
            <div className="text-[11.5px] text-[var(--fg-dim)] mt-0.5">{desc}</div>
          </div>
          <Toggle checked={!!perms[key]} onChange={(v) => setPerm(key, v)} />
        </div>
      ))}
    </div>
  );
}
