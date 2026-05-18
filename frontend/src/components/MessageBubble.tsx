import React, { useMemo, useState } from 'react';
import { ScrollText, FileText } from 'lucide-react';
import Markdown from './Markdown';
import ThinkingBlock from './ThinkingBlock';
import ToolCallBlock from './ToolCallBlock';
import MessageActions from './MessageActions';
import type { Attachment, Message } from '../state/types';
import { parseThinking } from '../lib/parseThinking';
import { useStore } from '../state/store';
import { editAndResend } from '../hooks/useChat';

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function SentAttachment({ a }: { a: Attachment }) {
  const [expanded, setExpanded] = useState(false);

  if (a.kind === 'image') {
    if (!a.dataUri) {
      // Edge case: attachment metadata without data — show a friendly placeholder.
      return (
        <div className="rounded-2xl border bd-strong bg-side w-32 h-32 flex flex-col items-center justify-center text-[11px] text-[var(--fg-dim)]">
          <FileText className="w-5 h-5 mb-1" />
          (image missing)
        </div>
      );
    }
    return (
      <button
        onClick={() => setExpanded((e) => !e)}
        className="block rounded-2xl overflow-hidden border bd-strong bg-side hover:bd-strong transition shadow-lg"
        title={`${a.name} · ${formatSize(a.size)} — click to ${expanded ? 'shrink' : 'enlarge'}`}
        style={
          expanded
            ? { maxWidth: 420, maxHeight: '50vh' }
            : { width: 160, height: 160 }
        }
      >
        <img
          src={a.dataUri}
          alt={a.name}
          className={expanded ? 'block w-full h-auto' : 'w-full h-full object-cover'}
          draggable={false}
        />
      </button>
    );
  }
  // text / code chip
  return (
    <div
      className="flex items-center gap-2 px-3 py-2 rounded-xl border bd-strong bg-side text-[12px] max-w-[300px]"
      title={`${a.name} · ${formatSize(a.size)}`}
    >
      <div className="w-8 h-8 rounded-lg bg-[var(--bg-hover)] flex items-center justify-center shrink-0">
        <FileText className="w-4 h-4 text-[var(--fg-muted)]" />
      </div>
      <div className="min-w-0">
        <div className="text-[var(--fg)] truncate">{a.name}</div>
        <div className="text-[var(--fg-dim)] text-[10.5px]">{formatSize(a.size)}</div>
      </div>
    </div>
  );
}

function SummaryBubble({ m }: { m: Message }) {
  const [open, setOpen] = useState(false);
  const count = m.compactedCount ?? 0;
  return (
    <div className="my-4 rounded-xl border border-dashed bd-strong bg-[var(--bg-side)]/40 overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-[12px] text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] select-none"
      >
        <ScrollText className="w-3.5 h-3.5" />
        <span className="flex-1 text-left">
          Earlier conversation summary{count > 0 && ` · ${count} turns compacted`}
        </span>
        <span className="text-[var(--fg-dim)]">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="px-3 py-2 border-t bd-soft text-[var(--fg-muted)] text-[13px] italic">
          <Markdown text={m.content} />
        </div>
      )}
    </div>
  );
}

export default function MessageBubble({ m }: { m: Message }) {
  // All hooks must run unconditionally — no early returns before this block.
  const streaming = useStore((s) => s.streaming);
  const isLastAssistant = useStore((s) => {
    const msgs = s.activeId ? s.conversations.find((c) => c.id === s.activeId)?.messages ?? [] : [];
    return msgs[msgs.length - 1]?.id === m.id && m.role === 'assistant';
  });
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(m.content);
  const segments = useMemo(() => parseThinking(m.content), [m.content]);

  const isLive = streaming && isLastAssistant;

  if (m.kind === 'summary') return <SummaryBubble m={m} />;

  function startEdit() { setDraft(m.content); setEditing(true); }
  async function saveEdit() {
    const v = draft.trim();
    setEditing(false);
    if (v && v !== m.content) await editAndResend(m.id, v);
  }

  // ─── USER: pill, right-aligned ───────────────────────────────────
  if (m.role === 'user') {
    return (
      <div className="group flex justify-end mb-6">
        <div className="relative max-w-[80%]">
          {editing ? (
            <div className="rounded-2xl bg-[var(--bg-hover)] border bd-strong p-3 space-y-2 min-w-[420px]">
              <textarea
                autoFocus
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveEdit(); }
                  else if (e.key === 'Escape') setEditing(false);
                }}
                rows={Math.min(20, Math.max(2, draft.split('\n').length))}
                className="w-full bg-[var(--bg-app)] border bd-soft rounded-lg p-2 text-[14px] outline-none focus:bd-strong text-[var(--fg)]"
              />
              <div className="flex gap-2 justify-end text-xs">
                <button onClick={() => setEditing(false)} className="px-3 py-1 rounded-lg hover:bg-[var(--bg-app)] text-[var(--fg-muted)]">Cancel</button>
                <button onClick={saveEdit} className="px-3 py-1 rounded-lg bg-[var(--fg)] text-[var(--bg-app)] hover:bg-white">Save & resend</button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-end gap-1.5">
              {m.attachments && m.attachments.length > 0 && (
                <div className="flex flex-wrap gap-2 justify-end">
                  {m.attachments.map((a) => <SentAttachment key={a.id} a={a} />)}
                </div>
              )}
              {m.content && (
                <div className="rounded-3xl bg-[var(--bg-hover)] px-4 py-2.5 text-[14.5px] whitespace-pre-wrap break-words text-[var(--fg)]">
                  {m.content}
                </div>
              )}
            </div>
          )}
          {!editing && (
            <div className="absolute -bottom-7 right-0 opacity-0 group-hover:opacity-100 transition-opacity">
              <MessageActions m={m} onEdit={startEdit} />
            </div>
          )}
        </div>
      </div>
    );
  }

  // ─── SYSTEM: small notice ────────────────────────────────────────
  if (m.role === 'system') {
    return (
      <div className="my-3 rounded-lg border bd-soft bg-[var(--bg-side)]/60 px-3 py-2 text-[13px] text-[var(--fg-muted)]">
        <Markdown text={m.content} />
      </div>
    );
  }

  // ─── ASSISTANT (and tool): no bubble, just flowing text ──────────
  return (
    <div className="group mb-8">
      {segments.map((seg, i) =>
        seg.kind === 'thinking' ? (
          <ThinkingBlock key={i} content={seg.content} streaming={seg.streaming && isLive} />
        ) : (
          <Markdown key={i} text={seg.content} />
        ),
      )}

      {isLive && m.content === '' ? (
        <span className="inline-flex items-center gap-2 text-[var(--fg-dim)] text-[12.5px] italic">
          <span className="inline-flex gap-0.5">
            <span className="w-1 h-1 rounded-full bg-[var(--fg-dim)] animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-1 h-1 rounded-full bg-[var(--fg-dim)] animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-1 h-1 rounded-full bg-[var(--fg-dim)] animate-bounce" style={{ animationDelay: '300ms' }} />
          </span>
          Processing prompt…
        </span>
      ) : isLive ? (
        <span className="inline-block w-2 h-4 bg-[var(--fg)] align-text-bottom animate-pulse ml-0.5" />
      ) : null}

      {m.toolCalls?.map((tc) => <ToolCallBlock key={tc.id} tc={tc} />)}

      <div className="mt-2 flex items-center gap-3">
        <div className="opacity-0 group-hover:opacity-100 transition-opacity">
          <MessageActions m={m} onEdit={startEdit} />
        </div>
        {m.role === 'assistant' && (m.ttftMs || m.tokensPerSec) && (
          <div className="text-[10.5px] text-[var(--fg-dim)] mono flex gap-3 opacity-0 group-hover:opacity-100 transition-opacity">
            {m.ttftMs != null && <span>TTFT {(m.ttftMs / 1000).toFixed(2)}s</span>}
            {m.tokensPerSec != null && <span>{m.tokensPerSec.toFixed(1)} tok/s</span>}
            {m.totalTokens != null && <span>{m.totalTokens.toLocaleString()} tokens</span>}
          </div>
        )}
      </div>
    </div>
  );
}
