import React, { useState } from 'react';
import { Copy, Check, FileEdit } from 'lucide-react';
import { useStore } from '../state/store';
import { writeFileText } from '../hooks/useFS';
import InlineDiff from './InlineDiff';

interface Props {
  language: string;
  code: string;
  /** className from react-markdown's `code` element (carries hljs-* tokens). */
  className?: string;
}

const LANG_TO_EXT: Record<string, string[]> = {
  typescript: ['ts', 'tsx'],
  javascript: ['js', 'jsx'],
  python: ['py'],
  go: ['go'],
  rust: ['rs'],
  java: ['java'],
  c: ['c', 'h'],
  cpp: ['cpp', 'hpp', 'cc'],
  json: ['json'],
  yaml: ['yml', 'yaml'],
  shell: ['sh'],
  powershell: ['ps1'],
  html: ['html'],
  css: ['css'],
  markdown: ['md'],
  sql: ['sql'],
  ini: ['ini', 'toml'],
};

function langMatchesFile(lang: string, path: string | null): boolean {
  if (!path) return false;
  const ext = path.split('.').pop()?.toLowerCase();
  if (!ext) return false;
  const exts = LANG_TO_EXT[lang.toLowerCase()];
  return !!exts && exts.includes(ext);
}

export default function CodeBlock({ language, code, className }: Props) {
  const { tab, openFile, openFileContent, setOpenFile } = useStore();
  const [copied, setCopied] = useState(false);
  const [showDiff, setShowDiff] = useState(false);
  const [applied, setApplied] = useState(false);

  const lang = language || (className?.match(/language-(\S+)/)?.[1]) || '';
  const canApply = tab === 'code' && openFile && langMatchesFile(lang, openFile);

  async function copy() {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch { /* ignore */ }
  }

  async function applyToFile() {
    if (!openFile) return;
    try {
      await writeFileText(openFile, code);
      setOpenFile(openFile, code, false);
      setShowDiff(false);
      setApplied(true);
      setTimeout(() => setApplied(false), 1500);
    } catch (e) {
      console.error(e);
    }
  }

  return (
    <div className="my-3 rounded-lg border bd-soft bg-[#0a0a0a] overflow-hidden group">
      <div className="px-3 py-1.5 border-b bd-soft bg-[var(--bg-side)]/40 flex items-center justify-between text-[11px]">
        <span className="text-[var(--fg-dim)] mono uppercase tracking-wider">
          {lang || 'plain'}
        </span>
        <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          {canApply && (
            <button
              onClick={() => setShowDiff((s) => !s)}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bg-[var(--bg-hover)]"
              title={`Apply to ${openFile?.split(/[\\/]/).pop()}`}
            >
              <FileEdit className="w-3 h-3" />
              {applied ? 'Applied' : showDiff ? 'Hide diff' : 'Apply'}
            </button>
          )}
          <button
            onClick={copy}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bg-[var(--bg-hover)]"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        </div>
      </div>

      <pre className="overflow-auto scroll text-[12.5px] mono leading-relaxed p-3">
        <code className={className}>{code}</code>
      </pre>

      {showDiff && openFile && (
        <div className="border-t bd-soft p-3 space-y-2">
          <div className="text-[11px] text-[var(--fg-muted)] flex items-center justify-between">
            <span>Diff vs <span className="mono">{openFile.split(/[\\/]/).pop()}</span></span>
            <div className="flex gap-1">
              <button
                onClick={() => setShowDiff(false)}
                className="px-2 py-0.5 rounded text-[11px] hover:bg-[var(--bg-hover)] text-[var(--fg-muted)]"
              >
                Cancel
              </button>
              <button
                onClick={applyToFile}
                className="px-2 py-0.5 rounded text-[11px] bg-emerald-700 hover:bg-emerald-600 text-white"
              >
                Confirm apply
              </button>
            </div>
          </div>
          <InlineDiff before={openFileContent} after={code} />
        </div>
      )}
    </div>
  );
}
