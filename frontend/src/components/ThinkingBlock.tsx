import React, { useEffect, useState } from 'react';
import { Brain, ChevronDown, ChevronRight } from 'lucide-react';
import Markdown from './Markdown';
import { roughTokens } from '../lib/parseThinking';

interface Props {
  content: string;
  streaming?: boolean;
}

export default function ThinkingBlock({ content, streaming }: Props) {
  const [open, setOpen] = useState(false);
  const [autoExpanded, setAutoExpanded] = useState(true);
  useEffect(() => {
    if (!streaming && autoExpanded) setAutoExpanded(false);
  }, [streaming, autoExpanded]);

  const expanded = open || (streaming && autoExpanded);
  const tokens = roughTokens(content);

  return (
    <div className="my-2 rounded-lg border bd-soft bg-[var(--bg-side)]/40 overflow-hidden">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-1.5 text-[12px] text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] select-none"
      >
        <Brain className={`w-3.5 h-3.5 ${streaming ? 'animate-pulse' : ''}`} />
        <span className="flex-1 text-left">
          {streaming ? 'Thinking…' : `Thought · ${tokens.toLocaleString()} tokens`}
        </span>
        {expanded
          ? <ChevronDown className="w-3.5 h-3.5 text-[var(--fg-dim)]" />
          : <ChevronRight className="w-3.5 h-3.5 text-[var(--fg-dim)]" />}
      </button>
      {expanded && content && (
        <div className="px-3 py-2 border-t bd-soft text-[var(--fg-muted)] text-[13px] italic opacity-80">
          <Markdown text={content} />
        </div>
      )}
    </div>
  );
}
