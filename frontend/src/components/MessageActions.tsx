import React, { useState } from 'react';
import { Copy, Check, RotateCcw, Pencil, Trash2 } from 'lucide-react';
import { useStore } from '../state/store';
import { regenerateLast } from '../hooks/useChat';
import type { Message } from '../state/types';

export default function MessageActions({ m, onEdit }: { m: Message; onEdit: () => void }) {
  const { deleteMessage, streaming } = useStore();
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(m.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch { /* ignore */ }
  }

  return (
    <div className="flex items-center gap-0.5">
      <IconBtn onClick={copy} title="Copy">
        {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
      </IconBtn>
      {m.role === 'assistant' && (
        <IconBtn onClick={() => regenerateLast()} title="Regenerate" disabled={streaming}>
          <RotateCcw className="w-3.5 h-3.5" />
        </IconBtn>
      )}
      {m.role === 'user' && (
        <IconBtn onClick={onEdit} title="Edit and resend" disabled={streaming}>
          <Pencil className="w-3.5 h-3.5" />
        </IconBtn>
      )}
      <IconBtn
        onClick={() => { if (confirm('Delete this message?')) deleteMessage(m.id); }}
        title="Delete"
        disabled={streaming}
        danger
      >
        <Trash2 className="w-3.5 h-3.5" />
      </IconBtn>
    </div>
  );
}

function IconBtn({
  onClick, title, disabled, danger, children,
}: {
  onClick: () => void;
  title: string;
  disabled?: boolean;
  danger?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`p-1.5 rounded-md disabled:opacity-30 ${
        danger
          ? 'text-[var(--fg-dim)] hover:text-red-400 hover:bg-[var(--bg-hover)]'
          : 'text-[var(--fg-muted)] hover:text-[var(--fg)] hover:bg-[var(--bg-hover)]'
      }`}
    >
      {children}
    </button>
  );
}
