import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Plus, ArrowUp, Square, Mic, Sparkles, X,
  Pencil, GraduationCap, Code2, ListChecks, Lightbulb,
  FileText, Image as ImageIcon, Paperclip,
  MessageSquare, Wrench, Globe, ChevronDown,
} from 'lucide-react';
import { useStore } from '../state/store';
import { api, BACKEND_HTTP } from '../ipc/bridge';
import { sendChat, estimateAttachmentTokens, useExactCtxUsed } from '../hooks/useChat';
import type { ChatMode } from '../state/types';
import { maybeRunSlash, suggestSlash, COMMANDS } from '../lib/slashCommands';
import { roughTokens, stripThinking } from '../lib/parseThinking';
import type { Attachment } from '../state/types';
import MessageBubble from './MessageBubble';

const QUICK_ACTIONS: { label: string; icon: React.ComponentType<{ className?: string }>; prompt: string }[] = [
  { label: 'Write',      icon: Pencil,        prompt: 'Help me write ' },
  { label: 'Learn',      icon: GraduationCap, prompt: 'Explain to me how ' },
  { label: 'Code',       icon: Code2,         prompt: 'Write code for ' },
  { label: 'Plan',       icon: ListChecks,    prompt: 'Help me plan ' },
  { label: 'Brainstorm', icon: Lightbulb,     prompt: 'Brainstorm ideas for ' },
];

const TEXT_EXT = new Set([
  'txt', 'md', 'py', 'ts', 'tsx', 'js', 'jsx', 'json', 'yaml', 'yml',
  'css', 'html', 'go', 'rs', 'java', 'c', 'cpp', 'h', 'hpp', 'sh', 'ps1',
  'toml', 'ini', 'xml', 'sql', 'env',
]);
const IMG_EXT = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp']);

// Hard cap on images per outgoing message. Larger batches trigger:
//   1. /tokenize POST bodies in the tens of MB (slow to stringify + parse).
//   2. Caption-batch loops 30+ deep at ~3-5 s each ⇒ minutes of "Compacting…".
//   3. mtmd sequential CLIP eval of 100+ images per turn — VRAM and graph-
//      allocator failures kick in well before that count.
//   4. localStorage quota blown silently (5-10 MB ceiling) so the chat
//      doesn't persist on reload.
// 16 leaves comfortable headroom for all four budgets at our 384 px downsample.
const MAX_IMAGES_PER_MESSAGE = 16;

export default function ChatPane() {
  const messages = useStore((s) => (s.activeId ? s.conversations.find((c) => c.id === s.activeId)?.messages ?? [] : []));
  const streaming = useStore((s) => s.streaming);
  const empty = messages.length === 0;
  if (empty) return <WelcomeView />;
  return (
    <div className="h-full flex flex-col">
      <MessageList />
      <Composer streaming={streaming} />
    </div>
  );
}

function WelcomeView() {
  const [text, setText] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const streaming = useStore((s) => s.streaming);

  async function submit() {
    const v = text.trim();
    if (streaming) return;
    if (!v && attachments.length === 0) return;
    // Defensive cap — the attach paths already clip, but guard against any
    // pre-existing oversized batch (e.g. legacy state) before we send.
    const imgCount = attachments.filter((a) => a.kind === 'image').length;
    if (imgCount > MAX_IMAGES_PER_MESSAGE) {
      alert(`Too many images attached (${imgCount}). Limit is ${MAX_IMAGES_PER_MESSAGE} per message — remove some or split across turns.`);
      return;
    }
    setText('');
    const att = attachments;
    setAttachments([]);
    if (await maybeRunSlash(v)) return;
    const expanded = await expandFileRefs(v);
    await sendChat(expanded, att);
  }

  return (
    <div className="h-full flex flex-col items-center justify-center px-6">
      <h1 className="text-[28px] font-medium text-[var(--fg)] mb-8 flex items-center gap-3">
        <Sparkles className="w-6 h-6 text-orange-400" />
        What's on the agenda today?
      </h1>
      <div className="w-full max-w-2xl">
        <InputBox
          value={text}
          onChange={setText}
          attachments={attachments}
          onAttachmentsChange={setAttachments}
          onSubmit={submit}
          disabled={streaming}
        />
        <div className="flex flex-wrap items-center justify-center gap-2 mt-4">
          {QUICK_ACTIONS.map(({ label, icon: Icon, prompt }) => (
            <button
              key={label}
              onClick={() => setText(prompt)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-full border bd-soft hover:bd-strong hover:bg-[var(--bg-hover)] text-[12.5px] text-[var(--fg-muted)] hover:text-[var(--fg)]"
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function MessageList() {
  const messages = useStore((s) => (s.activeId ? s.conversations.find((c) => c.id === s.activeId)?.messages ?? [] : []));
  const ref = useRef<HTMLDivElement>(null);
  // Stay pinned to the bottom unless the user scrolls up. Without this latch,
  // every streamed token re-runs the messages effect and snaps the view back
  // down — making it impossible to read earlier turns while the AI is typing.
  const stuckToBottom = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (stuckToBottom.current) el.scrollTo({ top: el.scrollHeight });
  }, [messages]);
  function onScroll() {
    const el = ref.current;
    if (!el) return;
    // 32px slack so the latch re-engages when you scroll most of the way back
    // down (don't require pixel-perfect bottom).
    stuckToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
  }
  return (
    <div ref={ref} onScroll={onScroll} className="flex-1 overflow-auto scroll">
      <div className="max-w-3xl mx-auto px-6 py-6">
        {messages.map((m) => <MessageBubble key={m.id} m={m} />)}
      </div>
    </div>
  );
}

function Composer({ streaming }: { streaming: boolean }) {
  const [text, setText] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const compacting = useStore((s) => s.compacting);

  async function submit() {
    const v = text.trim();
    if (streaming) return;
    if (!v && attachments.length === 0) return;
    // Defensive cap — the attach paths already clip, but guard against any
    // pre-existing oversized batch (e.g. legacy state) before we send.
    const imgCount = attachments.filter((a) => a.kind === 'image').length;
    if (imgCount > MAX_IMAGES_PER_MESSAGE) {
      alert(`Too many images attached (${imgCount}). Limit is ${MAX_IMAGES_PER_MESSAGE} per message — remove some or split across turns.`);
      return;
    }
    setText('');
    const att = attachments;
    setAttachments([]);
    if (await maybeRunSlash(v)) return;
    const expanded = await expandFileRefs(v);
    await sendChat(expanded, att);
  }

  return (
    <div className="px-6 py-3">
      <div className="max-w-3xl mx-auto">
        {compacting && <CompactingIndicator />}
        <InputBox
          value={text}
          onChange={setText}
          attachments={attachments}
          onAttachmentsChange={setAttachments}
          onSubmit={submit}
          disabled={streaming || compacting}
          onCancel={streaming ? () => api.llm.cancel() : undefined}
        />
        <div className="mt-2 text-center text-[11px] text-[var(--fg-dim)]">
          Local AI may make mistakes. Tools run only when permitted.
        </div>
      </div>
    </div>
  );
}

function CompactingIndicator() {
  return (
    <div className="mb-2 flex items-center justify-center gap-2 text-[11.5px] text-[var(--fg-muted)]">
      <span className="inline-flex gap-1">
        <span className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse" style={{ animationDelay: '0ms' }} />
        <span className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse" style={{ animationDelay: '180ms' }} />
        <span className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse" style={{ animationDelay: '360ms' }} />
      </span>
      <span>Compacting earlier messages…</span>
    </div>
  );
}

// ─── @file expansion ──────────────────────────────────────────────
const FILE_REF_RE = /(?:^|\s)@([\w./\\-]+)/g;

export async function expandFileRefs(text: string): Promise<string> {
  const refs: string[] = [];
  text.replace(FILE_REF_RE, (_m, p1: string) => { refs.push(p1); return ''; });
  if (refs.length === 0) return text;
  const workspace = useStore.getState().workspace;
  if (!workspace) return text;
  const blocks: string[] = [];
  for (const ref of refs) {
    try {
      const r = await fetch(`${BACKEND_HTTP}/fs/read?path=${encodeURIComponent(workspace + '/' + ref)}`);
      if (!r.ok) continue;
      const content = await r.text();
      if (content.length > 100_000) {
        blocks.push(`\n\n*(file \`${ref}\` is ${content.length} chars — truncated to first 100k)*\n\`\`\`\n${content.slice(0, 100_000)}\n\`\`\``);
      } else {
        blocks.push(`\n\n**\`${ref}\`:**\n\`\`\`\n${content}\n\`\`\``);
      }
    } catch { /* skip */ }
  }
  return text + blocks.join('');
}

// ─── Input box ─────────────────────────────────────────────────────
interface AutocompleteState {
  kind: 'slash' | 'file' | null;
  items: string[];
  start: number;
  query: string;
  selected: number;
}

export function InputBox({
  value, onChange, attachments, onAttachmentsChange, onSubmit, disabled, onCancel,
}: {
  value: string;
  onChange: (v: string) => void;
  attachments: Attachment[];
  onAttachmentsChange: (next: Attachment[]) => void;
  onSubmit: () => void;
  disabled: boolean;
  onCancel?: () => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [ac, setAc] = useState<AutocompleteState>({ kind: null, items: [], start: 0, query: '', selected: 0 });
  const [dragOver, setDragOver] = useState(false);
  const [attachMenu, setAttachMenu] = useState(false);
  const workspace = useStore((s) => s.workspace);
  const [fileIndex, setFileIndex] = useState<string[]>([]);
  // Transient warning when an attach action gets clipped by MAX_IMAGES_PER_MESSAGE.
  // Cleared automatically when the next attach succeeds or after a short timeout.
  const [attachWarn, setAttachWarn] = useState<string | null>(null);
  useEffect(() => {
    if (!attachWarn) return;
    const t = setTimeout(() => setAttachWarn(null), 4000);
    return () => clearTimeout(t);
  }, [attachWarn]);

  // Slice an incoming list of additions so the total image count stays ≤ the cap.
  // Returns the trimmed list and how many were dropped so the caller can warn.
  function clipImagesToCap(incoming: Attachment[]): { kept: Attachment[]; dropped: number } {
    const currentImages = attachments.filter((a) => a.kind === 'image').length;
    const available = Math.max(0, MAX_IMAGES_PER_MESSAGE - currentImages);
    let imageBudget = available;
    const kept: Attachment[] = [];
    let dropped = 0;
    for (const a of incoming) {
      if (a.kind === 'image') {
        if (imageBudget > 0) { kept.push(a); imageBudget -= 1; }
        else dropped += 1;
      } else {
        kept.push(a);
      }
    }
    return { kept, dropped };
  }

  useEffect(() => {
    if (!workspace) { setFileIndex([]); return; }
    let cancelled = false;
    (async () => {
      try {
        const r = await fetch(`${BACKEND_HTTP}/fs/walk?path=${encodeURIComponent(workspace)}&limit=2000`);
        if (!r.ok) return;
        const data = await r.json();
        if (!cancelled) setFileIndex(data.paths ?? []);
      } catch { /* ignore */ }
    })();
    return () => { cancelled = true; };
  }, [workspace]);

  function computeAutocomplete(text: string, caret: number): AutocompleteState {
    const lineStart = text.lastIndexOf('\n', caret - 1) + 1;
    const upToCaret = text.slice(lineStart, caret);
    const slashM = upToCaret.match(/(^|\s)\/(\w*)$/);
    const atM = upToCaret.match(/(^|\s)@([\w./\\-]*)$/);
    if (slashM) {
      const query = slashM[2];
      const items = suggestSlash(query).map((c) => c.name);
      return { kind: 'slash', items, start: lineStart + slashM.index! + (slashM[1] ? 1 : 0), query, selected: 0 };
    }
    if (atM && fileIndex.length > 0) {
      const query = atM[2].toLowerCase();
      const items = fileIndex.filter((p) => p.toLowerCase().includes(query)).slice(0, 8);
      return { kind: 'file', items, start: lineStart + atM.index! + (atM[1] ? 1 : 0), query, selected: 0 };
    }
    return { kind: null, items: [], start: 0, query: '', selected: 0 };
  }

  function onTextChange(v: string) {
    onChange(v);
    const caret = textareaRef.current?.selectionStart ?? v.length;
    setAc(computeAutocomplete(v, caret));
  }

  function applySuggestion(idx: number) {
    if (ac.kind === null || !ac.items[idx]) return;
    const insert = ac.kind === 'slash' ? `/${ac.items[idx]} ` : `@${ac.items[idx]} `;
    const before = value.slice(0, ac.start);
    const afterStart = ac.start + (ac.kind === 'slash' ? ac.query.length + 1 : ac.query.length + 1);
    const after = value.slice(afterStart);
    onChange(before + insert + after);
    setAc({ kind: null, items: [], start: 0, query: '', selected: 0 });
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  // ─── Attachment helpers ───────────────────────────────────────────
  async function addFiles(files: File[]) {
    // Build the incoming list first, then clip against the image cap. We
    // decode/downsample upfront so the cap check is accurate even when the
    // user mixes images and text files in one drop/paste.
    const incoming: Attachment[] = [];
    for (const f of files) {
      const ext = f.name.split('.').pop()?.toLowerCase() ?? '';
      if (IMG_EXT.has(ext) || f.type.startsWith('image/')) {
        const dataUri = await loadAndDownsampleImage(f);
        incoming.push({
          id: crypto.randomUUID(), kind: 'image', name: f.name,
          dataUri, size: f.size, mime: f.type || `image/${ext}`,
        });
      } else if (TEXT_EXT.has(ext) || f.type.startsWith('text/')) {
        const text = await f.text();
        const content = text.length > 100_000
          ? text.slice(0, 100_000) + '\n... [truncated]'
          : text;
        incoming.push({
          id: crypto.randomUUID(), kind: 'text', name: f.name,
          content, size: f.size, mime: f.type || 'text/plain',
        });
      }
      // unsupported types silently skipped — explicit error chip is too noisy
    }
    const { kept, dropped } = clipImagesToCap(incoming);
    if (dropped > 0) {
      setAttachWarn(`Skipped ${dropped} image${dropped === 1 ? '' : 's'} — limit is ${MAX_IMAGES_PER_MESSAGE} per message. Split across turns to send more.`);
    }
    onAttachmentsChange([...attachments, ...kept]);
  }

  function removeAttachment(id: string) {
    onAttachmentsChange(attachments.filter((a) => a.id !== id));
  }

  async function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    await addFiles(Array.from(e.dataTransfer.files));
  }

  // Clipboard images (Win+Shift+S → Ctrl+V) — handled separately from
  // normal text paste; text paste still falls through.
  async function handlePaste(e: React.ClipboardEvent<HTMLTextAreaElement>) {
    const items = Array.from(e.clipboardData.items);
    const files: File[] = [];
    for (const it of items) {
      if (it.kind === 'file') {
        const f = it.getAsFile();
        if (f) files.push(f);
      }
    }
    if (files.length > 0) {
      e.preventDefault();
      await addFiles(files);
    }
  }

  // Native picker. For images we read bytes via fetch(file://...) so we get a
  // real data URI for the thumbnail, not a markdown reference.
  async function pickAttach(kind: 'any' | 'image') {
    const filters = kind === 'image'
      ? [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'] }]
      : [{ name: 'Text & code', extensions: Array.from(TEXT_EXT) },
         { name: 'All files', extensions: ['*'] }];
    const res = await api.fs.pickFile(filters, true);
    if (!res) return;
    const paths = Array.isArray(res) ? res : [res];
    // Build all candidates first, then clip against the cap so image+text
    // mixes don't accidentally evict a text file.
    const incoming: Attachment[] = [];
    for (const p of paths) {
      const ext = p.split('.').pop()?.toLowerCase() ?? '';
      const name = p.split(/[\\/]/).pop() ?? p;
      if (IMG_EXT.has(ext)) {
        try {
          // Use the Electron IPC bridge to read the file bytes via Node in
          // the main process. The renderer can't fetch('file:///...') under
          // Electron's CSP — that path silently failed and was the reason
          // "Attach image" from the OS dialog appeared to do nothing.
          const { base64, size } = await api.fs.readFile(p);
          const mime = `image/${ext === 'jpg' ? 'jpeg' : ext}`;
          const byteString = atob(base64);
          const bytes = new Uint8Array(byteString.length);
          for (let i = 0; i < byteString.length; i++) bytes[i] = byteString.charCodeAt(i);
          const blob = new Blob([bytes], { type: mime });
          const dataUri = await loadAndDownsampleImage(blob);
          incoming.push({
            id: crypto.randomUUID(), kind: 'image', name,
            dataUri, size, mime,
          });
        } catch (e) {
          console.error('failed to read picked image', p, e);
        }
        continue;
      }
      try {
        const r = await fetch(`${BACKEND_HTTP}/fs/read?path=${encodeURIComponent(p)}`);
        if (!r.ok) continue;
        const text = await r.text();
        const content = text.length > 100_000
          ? text.slice(0, 100_000) + '\n... [truncated]'
          : text;
        incoming.push({
          id: crypto.randomUUID(), kind: 'text', name,
          content, size: text.length, mime: 'text/plain',
        });
      } catch { /* skip */ }
    }
    const { kept, dropped } = clipImagesToCap(incoming);
    if (dropped > 0) {
      setAttachWarn(`Skipped ${dropped} image${dropped === 1 ? '' : 's'} — limit is ${MAX_IMAGES_PER_MESSAGE} per message. Split across turns to send more.`);
    }
    onAttachmentsChange([...attachments, ...kept]);
  }

  // ─── Token math (attachments included honestly) ───────────────────
  const settings = useStore((s) => s.settings);
  const visionActive = useStore((s) => !!s.modelStatus.visionActive);
  const modelLoaded = useStore((s) => s.modelStatus.status === 'loaded');
  // Images cost their full base64 only when they're actually sent inline —
  // i.e. the engine has vision active, or the user opted into the legacy
  // base64-in-text fallback. Otherwise we just send the short placeholder.
  const sendsImageBytes = visionActive || settings.sendImagesAsBase64;
  const tokenCount = useMemo(() => {
    let n = roughTokens(value);
    for (const a of attachments) {
      n += estimateAttachmentTokens(a, sendsImageBytes, visionActive);
    }
    return n;
  }, [value, attachments, sendsImageBytes, visionActive]);

  const activeMessages = useStore((s) =>
    s.activeId ? s.conversations.find((c) => c.id === s.activeId)?.messages ?? [] : [],
  );
  // Backend tokenises through the loaded model so the wheel matches what
  // llama.cpp would actually charge — chat-template chrome, tool catalog, and
  // a family-aware image cost. Returns null while no model is loaded; fall
  // back to the rough chars/4 estimate so the wheel still shows something.
  const exactCtxUsed = useExactCtxUsed(value, attachments);
  const fallbackCtxUsed = useMemo(() => {
    let n = roughTokens(settings.systemPrompt);
    for (const m of activeMessages) {
      // Assistant reasoning is stripped from history before it's sent to the
      // model (see _materializeForModel), so it does NOT occupy the next
      // turn's context. Count the visible text only — otherwise this fallback
      // wildly overcounts (heavy thinking blocks read as ~19% when the exact
      // path, with the model loaded, correctly shows ~1%).
      const counted = m.role === 'assistant' ? stripThinking(m.content) : m.content;
      n += roughTokens(counted);
      if (m.attachments) {
        for (const a of m.attachments) {
          n += estimateAttachmentTokens(a, sendsImageBytes, visionActive);
        }
      }
    }
    n += tokenCount;
    return n;
  }, [settings.systemPrompt, sendsImageBytes, visionActive, activeMessages, tokenCount]);
  const ctxUsed = exactCtxUsed ?? fallbackCtxUsed;
  const ctxMax = settings.nCtx || 4096;
  const ctxPct = Math.min(100, Math.round((ctxUsed / ctxMax) * 100));
  const ctxIsExact = exactCtxUsed != null;

  return (
    <div
      className={`relative rounded-3xl bg-input border transition px-4 py-3 ${
        dragOver ? 'border-blue-500/60 ring-1 ring-blue-500/30' : 'bd-soft focus-within:bd-strong'
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {attachWarn && (
        <div className="mb-2 px-3 py-1.5 rounded-xl text-[11.5px] text-amber-300 bg-amber-900/20 border border-amber-700/40">
          {attachWarn}
        </div>
      )}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-2">
          {attachments.map((a) => (
            <AttachmentChip key={a.id} a={a} onRemove={() => removeAttachment(a.id)} />
          ))}
        </div>
      )}

      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onTextChange(e.target.value)}
        onPaste={handlePaste}
        onKeyDown={(e) => {
          if (ac.kind && ac.items.length > 0) {
            if (e.key === 'ArrowDown') { e.preventDefault(); setAc({ ...ac, selected: (ac.selected + 1) % ac.items.length }); return; }
            if (e.key === 'ArrowUp')   { e.preventDefault(); setAc({ ...ac, selected: (ac.selected - 1 + ac.items.length) % ac.items.length }); return; }
            if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) { e.preventDefault(); applySuggestion(ac.selected); return; }
            if (e.key === 'Escape') { setAc({ kind: null, items: [], start: 0, query: '', selected: 0 }); return; }
          }
          if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSubmit(); }
        }}
        rows={1}
        placeholder="Ask anything"
        className="w-full bg-transparent text-[var(--fg)] placeholder:text-[var(--fg-dim)] resize-none outline-none text-[15px] leading-relaxed min-h-[24px] max-h-[50vh] scroll"
        style={{ height: 'auto' }}
        onInput={(e) => {
          const t = e.currentTarget;
          t.style.height = 'auto';
          t.style.height = t.scrollHeight + 'px';
        }}
      />

      {ac.kind && ac.items.length > 0 && (
        <div className="absolute left-3 bottom-full mb-2 w-[420px] max-h-60 overflow-auto scroll bg-[var(--bg-app)] border bd-strong rounded-xl shadow-2xl z-30">
          {ac.items.map((item, i) => {
            const cmd = ac.kind === 'slash' ? COMMANDS.find((c) => c.name === item) : null;
            return (
              <button
                key={item}
                onMouseDown={(e) => { e.preventDefault(); applySuggestion(i); }}
                className={`w-full text-left px-3 py-1.5 text-[12.5px] ${
                  i === ac.selected ? 'bg-[var(--bg-hover)] text-[var(--fg)]' : 'text-[var(--fg-muted)] hover:bg-[var(--bg-hover)]'
                }`}
              >
                <span className="mono">{ac.kind === 'slash' ? `/${item}` : `@${item}`}</span>
                {cmd && <span className="text-[var(--fg-dim)] ml-2">— {cmd.desc}</span>}
              </button>
            );
          })}
        </div>
      )}

      <div className="flex items-center justify-between mt-2">
        <div className="relative flex items-center gap-1.5">
          <button
            onClick={() => setAttachMenu((o) => !o)}
            title="Attach"
            className="w-8 h-8 rounded-full hover:bg-[var(--bg-hover)] flex items-center justify-center text-[var(--fg-muted)]"
          >
            <Plus className={`w-4 h-4 transition-transform ${attachMenu ? 'rotate-45' : ''}`} />
          </button>
          <ChatModePicker />
          <span className="text-[11px] text-[var(--fg-dim)] ml-1">
            {tokenCount.toLocaleString()} tokens
          </span>

          {attachMenu && (
            <div
              className="absolute left-0 bottom-full mb-2 w-[220px] rounded-xl border bd-strong bg-[var(--bg-app)] shadow-2xl z-30 overflow-hidden"
              onMouseLeave={() => setAttachMenu(false)}
            >
              <MenuItem
                icon={<FileText className="w-4 h-4" />}
                label="Attach text / code"
                hint="inlines as a fenced block"
                onClick={async () => { setAttachMenu(false); await pickAttach('any'); }}
              />
              <MenuItem
                icon={<ImageIcon className="w-4 h-4" />}
                label="Attach image"
                hint="needs a vision model"
                onClick={async () => { setAttachMenu(false); await pickAttach('image'); }}
              />
              <div className="border-t bd-soft px-3 py-2 text-[10.5px] text-[var(--fg-dim)]">
                <Paperclip className="w-3 h-3 inline mr-1" />
                You can also drag-drop or paste images (Win+Shift+S → Ctrl+V).
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* The wheel only means something with a model loaded (the exact
              tokenizer is the real signal); hide it entirely when idle. */}
          {modelLoaded && <ContextWheel used={ctxUsed} max={ctxMax} pct={ctxPct} exact={ctxIsExact} />}
          <button
            title="Voice (not wired)"
            className="w-8 h-8 rounded-full hover:bg-[var(--bg-hover)] flex items-center justify-center text-[var(--fg-muted)]"
            disabled
          >
            <Mic className="w-4 h-4" />
          </button>
          {onCancel ? (
            <button
              onClick={onCancel}
              className="w-8 h-8 rounded-full bg-[var(--fg)] hover:bg-white text-[var(--bg-app)] flex items-center justify-center"
              title="Stop"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
            </button>
          ) : (
            <button
              onClick={onSubmit}
              disabled={disabled || !value.trim()}
              className="w-8 h-8 rounded-full bg-[var(--fg)] hover:bg-white disabled:opacity-30 text-[var(--bg-app)] flex items-center justify-center"
              title="Send"
            >
              <ArrowUp className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// Downsample images aggressively before sending to vision models. Token
// cost per image is set by the PROJECTOR (mmproj), not by us — different
// Qwen-VL fine-tunes can produce wildly different counts for the same
// input. The only knob we have is the image's pixel dimensions:
//
//   ~1300 tokens @ 1024px on Qwen2.5-VL-base (patch=14, merge=2)
//   ~73000 tokens @ 1024px on aggressive Qwen3-VL fine-tunes (patch=14, no merge)
//
// And we also have to keep individual patch batches small enough that
// llama.cpp's mtmd compute graph doesn't fail to allocate (the
// "ggml_gallocr_needs_realloc" / "cannot reallocate multi buffer graph"
// error means a single image batch overran the pre-reserved buffer).
//
// 384 is the safe upper bound for this class of fine-tune: ~6500 tokens
// per image, fits comfortably in a 40K context with room for several
// follow-up messages, and the per-batch graph stays inside the reserved
// buffer. Slight loss of detail vs 512 but the alternative is "doesn't
// work at all."
//
// If a model with a well-behaved projector (proper patch-merge) is loaded,
// users could safely raise this — TODO: make it a setting tied to model.
const MAX_IMAGE_EDGE = 384;
const JPEG_QUALITY   = 0.82;
async function loadAndDownsampleImage(blob: Blob): Promise<string> {
  const objectUrl = URL.createObjectURL(blob);
  try {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = (e) => reject(e);
      el.src = objectUrl;
    });
    // Always re-encode through canvas. The previous "if it's already small
    // enough, pass through" shortcut meant huge PNGs / screenshots at native
    // resolution bypassed the cap when the longest edge happened to be ≤ the
    // limit — that was a sneaky path to context overflow.
    const longest = Math.max(img.naturalWidth, img.naturalHeight);
    const scale = longest > MAX_IMAGE_EDGE ? MAX_IMAGE_EDGE / longest : 1;
    const w = Math.max(1, Math.round(img.naturalWidth * scale));
    const h = Math.max(1, Math.round(img.naturalHeight * scale));
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('canvas 2d unavailable');
    ctx.drawImage(img, 0, 0, w, h);
    return canvas.toDataURL('image/jpeg', JPEG_QUALITY);
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function AttachmentChip({ a, onRemove }: { a: Attachment; onRemove: () => void }) {
  if (a.kind === 'image') {
    return (
      <div className="relative group rounded-lg overflow-hidden border bd-strong bg-side"
           style={{ width: 64, height: 64 }}
           title={`${a.name} · ${formatFileSize(a.size)}`}>
        {a.dataUri && (
          <img
            src={a.dataUri}
            alt={a.name}
            className="w-full h-full object-cover"
            draggable={false}
          />
        )}
        <button
          onClick={onRemove}
          className="absolute top-0.5 right-0.5 w-5 h-5 rounded-full bg-black/70 text-white opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"
          title="Remove"
        >
          <X className="w-3 h-3" />
        </button>
      </div>
    );
  }
  // text / code chip
  return (
    <div
      className="group flex items-center gap-2 px-2.5 py-1.5 rounded-lg border bd-strong bg-side text-[12px] max-w-[280px]"
      title={`${a.name} · ${formatFileSize(a.size)}`}
    >
      <div className="w-7 h-7 rounded bg-[var(--bg-hover)] flex items-center justify-center shrink-0">
        <FileText className="w-3.5 h-3.5 text-[var(--fg-muted)]" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-[var(--fg)] truncate">{a.name}</div>
        <div className="text-[var(--fg-dim)] text-[10.5px]">{formatFileSize(a.size)}</div>
      </div>
      <button
        onClick={onRemove}
        className="text-[var(--fg-muted)] hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
        title="Remove"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

function MenuItem({
  icon, label, hint, onClick, active,
}: { icon: React.ReactNode; label: string; hint?: string; onClick: () => void; active?: boolean }) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-2.5 px-3 py-2 text-left hover:bg-[var(--bg-hover)] text-[13px] ${
        active ? 'bg-[var(--bg-hover)]' : ''
      }`}
    >
      <span className="text-[var(--fg-muted)] shrink-0">{icon}</span>
      <div className="min-w-0 flex-1">
        <div className="text-[var(--fg)] flex items-center gap-1.5">
          {label}
          {active && (
            <span className="text-emerald-400 text-[10px]">●</span>
          )}
        </div>
        {hint && <div className="text-[10.5px] text-[var(--fg-dim)]">{hint}</div>}
      </div>
    </button>
  );
}

/**
 * Thinking-depth picker — Copilot-style dropdown next to the + button.
 * Only renders when the loaded model recognises Qwen-style `/think` /
 * `/no_think` toggles; for other models the control is hidden entirely.
 */
/** Picks how the model behaves in chat: Normal (pure chat), Cowork (agent
 *  tools — read/write/edit files like the Code tab), or Search (web, soon).
 *  Only shown in the Chat tab; the Code tab is always agentic. */
function ChatModePicker() {
  const settings = useStore((s) => s.settings);
  const setSettings = useStore((s) => s.setSettings);
  const tab = useStore((s) => s.tab);
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function onDoc(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, []);

  // The Code tab is always agentic — the mode picker only makes sense in chat.
  if (tab !== 'chat') return null;

  const mode = settings.chatMode;
  const current = CHAT_MODE_OPTIONS.find((o) => o.value === mode) ?? CHAT_MODE_OPTIONS[0];
  const CurrentIcon = current.icon;

  return (
    <div ref={wrapRef} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1 pl-2 pr-1.5 h-7 rounded-full text-[11.5px] text-[var(--fg-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]"
        title="Chat mode"
      >
        <CurrentIcon className="w-3.5 h-3.5" />
        <span>{current.label}</span>
        <ChevronDown className="w-3 h-3 text-[var(--fg-dim)]" />
      </button>
      {open && (
        <div className="absolute left-0 bottom-full mb-2 w-[280px] rounded-xl border bd-strong bg-[var(--bg-app)] shadow-2xl z-30 overflow-hidden">
          {CHAT_MODE_OPTIONS.map((opt) => {
            const Icon = opt.icon;
            const disabled = !!opt.soon;
            return (
              <MenuItem
                key={opt.value}
                icon={<Icon className="w-4 h-4" />}
                label={opt.soon ? `${opt.label} · soon` : opt.label}
                hint={opt.hint}
                active={opt.value === mode}
                onClick={() => {
                  if (disabled) return;
                  setOpen(false);
                  void setSettings({ chatMode: opt.value });
                }}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

const CHAT_MODE_OPTIONS: {
  value: ChatMode;
  label: string;
  hint: string;
  icon: React.ComponentType<{ className?: string }>;
  soon?: boolean;
}[] = [
  { value: 'normal', label: 'Chat',   hint: 'Plain conversation. No tools.',                       icon: MessageSquare },
  { value: 'cowork', label: 'Cowork', hint: 'Let the model read, write & edit files in your workspace.', icon: Wrench },
  { value: 'search', label: 'Search', hint: 'Search the web and read pages. Needs the network permission.', icon: Globe },
];

/**
 * Compact donut showing how much of the context window is in use.
 * Green < 50%, amber 50-80%, red > 80%. Hover for the exact breakdown.
 */
function ContextWheel({ used, max, pct, exact }: { used: number; max: number; pct: number; exact?: boolean }) {
  const color =
    pct >= 80 ? '#f87171'   // red-400
    : pct >= 50 ? '#fbbf24' // amber-400
    : '#34d399';            // emerald-400
  const radius = 8;
  const stroke = 2;
  const c = 2 * Math.PI * radius;
  const offset = c * (1 - Math.min(100, pct) / 100);
  // `exact` flips the tilde off when the count came from the backend tokenizer.
  const prefix = exact ? '' : '~';
  return (
    <div
      className="relative flex items-center"
      title={`Context: ${used.toLocaleString()} / ${max.toLocaleString()} tokens (${prefix}${pct}%)${exact ? ' — exact, via loaded tokenizer' : ' — estimated'}`}
    >
      <svg width="22" height="22" viewBox="0 0 22 22" className="-rotate-90">
        <circle cx="11" cy="11" r={radius} fill="none" stroke="var(--bd-strong)" strokeWidth={stroke} />
        <circle
          cx="11"
          cy="11"
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeDasharray={c}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.2s ease' }}
        />
      </svg>
      <span className="ml-1 text-[10.5px] mono text-[var(--fg-dim)] tabular-nums">{prefix}{pct}%</span>
    </div>
  );
}
