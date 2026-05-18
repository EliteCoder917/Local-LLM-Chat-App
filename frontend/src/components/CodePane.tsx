import React, { useEffect, useRef, useState } from 'react';
import Editor from '@monaco-editor/react';
import { Folder, FileText, RefreshCw, FolderOpen, ArrowUp, Square } from 'lucide-react';
import { useStore } from '../state/store';
import { api } from '../ipc/bridge';
import { listDir, readFileText, writeFileText } from '../hooks/useFS';
import { sendChat } from '../hooks/useChat';
import type { FileNode, Message } from '../state/types';
import MessageBubble from './MessageBubble';

function langFromPath(p: string): string {
  const ext = p.split('.').pop()?.toLowerCase();
  const map: Record<string, string> = {
    ts: 'typescript', tsx: 'typescript',
    js: 'javascript', jsx: 'javascript',
    py: 'python', md: 'markdown', json: 'json',
    html: 'html', css: 'css',
    yml: 'yaml', yaml: 'yaml',
    sh: 'shell', ps1: 'powershell',
    go: 'go', rs: 'rust', java: 'java', c: 'c', cpp: 'cpp', h: 'cpp',
    sql: 'sql', toml: 'ini', ini: 'ini', xml: 'xml',
  };
  return map[ext ?? ''] ?? 'plaintext';
}

export default function CodePane() {
  const { workspace, setSettings } = useStore();

  if (!workspace) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-[var(--fg-muted)] gap-3">
        <div className="text-lg">No workspace open</div>
        <button
          onClick={async () => {
            const p = await api.fs.pickFolder();
            if (p) await setSettings({ workspace: p });
          }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-[var(--bg-hover)] hover:bg-[var(--bd-soft)] text-sm text-[var(--fg)]"
        >
          <FolderOpen className="w-4 h-4" /> Open folder
        </button>
      </div>
    );
  }

  return (
    <div className="h-full grid grid-cols-[260px_1fr_360px] min-h-0">
      <FilesList />
      <EditorColumn />
      <CodeChatColumn />
    </div>
  );
}

function FilesList() {
  const workspace = useStore((s) => s.workspace);
  const setSettings = useStore((s) => s.setSettings);
  const [tree, setTree] = useState<FileNode[]>([]);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (workspace) listDir(workspace).then(setTree).catch(console.error);
  }, [workspace, tick]);

  return (
    <div className="border-r bd-soft bg-side flex flex-col min-h-0">
      <div className="px-3 py-2 text-[11px] uppercase tracking-wider text-[var(--fg-dim)] border-b bd-soft flex items-center justify-between gap-2">
        <span className="truncate flex-1">{workspace.split(/[\\/]/).pop() || workspace}</span>
        <button onClick={() => setTick((t) => t + 1)} className="p-1 rounded hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]" title="Refresh">
          <RefreshCw className="w-3 h-3" />
        </button>
        <button
          onClick={async () => {
            const p = await api.fs.pickFolder();
            if (p) await setSettings({ workspace: p });
          }}
          className="p-1 rounded hover:bg-[var(--bg-hover)] hover:text-[var(--fg)]"
          title="Change folder"
        >
          <FolderOpen className="w-3 h-3" />
        </button>
      </div>
      <div className="flex-1 overflow-auto scroll py-1">
        {tree.map((n) => (
          <FileNodeRow key={n.path} node={n} depth={0} />
        ))}
      </div>
    </div>
  );
}

function FileNodeRow({ node, depth }: { node: FileNode; depth: number }) {
  const [open, setOpen] = useState(false);
  const [children, setChildren] = useState<FileNode[] | null>(null);
  const { openFile, setOpenFile } = useStore();

  async function toggle() {
    if (node.isDir) {
      if (!children) {
        try { setChildren(await listDir(node.path)); } catch (e) { console.error(e); }
      }
      setOpen((o) => !o);
    } else {
      try {
        const content = await readFileText(node.path);
        setOpenFile(node.path, content, false);
      } catch (e) {
        console.error(e);
      }
    }
  }

  const active = openFile === node.path;
  return (
    <div>
      <div
        onClick={toggle}
        className={`flex items-center gap-1.5 px-2 py-1 text-[13px] cursor-pointer truncate rounded ${
          active ? 'bg-[var(--bg-hover)] text-[var(--fg)]' : 'hover:bg-[var(--bg-hover)] text-[var(--fg-muted)]'
        }`}
        style={{ paddingLeft: 6 + depth * 12 }}
      >
        <span className="w-3 text-[var(--fg-dim)] text-[10px]">{node.isDir ? (open ? '▾' : '▸') : ''}</span>
        {node.isDir
          ? <Folder className="w-3.5 h-3.5 shrink-0 text-[var(--fg-dim)]" />
          : <FileText className="w-3.5 h-3.5 shrink-0 text-[var(--fg-dim)]" />}
        <span className="truncate">{node.name}</span>
      </div>
      {open && children?.map((c) => <FileNodeRow key={c.path} node={c} depth={depth + 1} />)}
    </div>
  );
}

function EditorColumn() {
  const { openFile, openFileContent, fileDirty, setOpenFile, settings } = useStore();
  const theme = settings.theme === 'dark' ? 'vs-dark' : 'light';

  async function save() {
    if (!openFile) return;
    try {
      await writeFileText(openFile, openFileContent);
      setOpenFile(openFile, openFileContent, false);
    } catch (e) {
      console.error(e);
    }
  }

  if (!openFile) {
    return (
      <div className="h-full flex items-center justify-center text-[var(--fg-dim)] text-sm bg-app">
        Pick a file from the list
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-app min-h-0">
      <div className="px-3 py-1.5 text-xs flex items-center justify-between bg-[var(--bg-side)] border-b bd-soft">
        <span className="truncate mono text-[var(--fg-muted)]">
          {openFile.split(/[\\/]/).slice(-2).join('/')}
          {fileDirty && <span className="ml-1 text-amber-400">●</span>}
        </span>
        <button
          onClick={save}
          disabled={!fileDirty}
          className="px-2 py-0.5 rounded text-[11px] bg-[var(--bg-hover)] hover:bg-[var(--bd-soft)] disabled:opacity-40 text-[var(--fg)]"
        >
          Save (Ctrl+S)
        </button>
      </div>
      <div className="flex-1 min-h-0">
        <Editor
          height="100%"
          theme={theme}
          language={langFromPath(openFile)}
          value={openFileContent}
          onChange={(v) => setOpenFile(openFile, v ?? '', true)}
          onMount={(editor, monaco) => {
            editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, save);
          }}
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            fontFamily: 'JetBrains Mono, Cascadia Code, Consolas, monospace',
            smoothScrolling: true,
            scrollBeyondLastLine: false,
            automaticLayout: true,
          }}
        />
      </div>
    </div>
  );
}

function CodeChatColumn() {
  const { openFile, openFileContent, streaming } = useStore();
  const messages = useStore((s) =>
    s.activeId ? s.conversations.find((c) => c.id === s.activeId)?.messages ?? [] : [],
  );
  const [text, setText] = useState('');
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight });
  }, [messages]);

  async function send() {
    const v = text.trim();
    if (!v || streaming) return;
    setText('');
    // Inline a snippet of the open file so the model has context.
    let prefixed = v;
    if (openFile) {
      const snippet = openFileContent.length > 8000
        ? openFileContent.slice(0, 8000) + '\n... [truncated]'
        : openFileContent;
      prefixed = `Working on file: \`${openFile}\`\n\n\`\`\`\n${snippet}\n\`\`\`\n\n${v}`;
    }
    await sendChat(prefixed);
  }

  return (
    <div className="border-l bd-soft bg-app flex flex-col min-h-0">
      <div className="px-3 py-2 text-[11px] uppercase tracking-wider text-[var(--fg-dim)] border-b bd-soft">
        Ask about this file
      </div>
      <div ref={listRef} className="flex-1 overflow-auto scroll px-3 py-3">
        {messages.length === 0 ? (
          <div className="text-xs text-[var(--fg-dim)] mt-4">
            Open a file and ask: "refactor this", "explain this", "add error handling"…
          </div>
        ) : (
          messages.map((m: Message) => <MessageBubble key={m.id} m={m} />)
        )}
      </div>
      <div className="p-2">
        <div className="rounded-2xl bg-input border bd-soft px-3 py-2 focus-within:bd-strong">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
            }}
            rows={2}
            placeholder={openFile ? 'Ask about this file…' : 'Ask anything…'}
            className="w-full bg-transparent text-[var(--fg)] placeholder:text-[var(--fg-dim)] resize-none outline-none text-sm"
          />
          <div className="flex items-center justify-end mt-1">
            {streaming ? (
              <button
                onClick={() => api.llm.cancel()}
                className="w-7 h-7 rounded-full bg-[var(--fg)] hover:bg-white text-[var(--bg-app)] flex items-center justify-center"
                title="Stop"
              >
                <Square className="w-3 h-3 fill-current" />
              </button>
            ) : (
              <button
                onClick={send}
                disabled={!text.trim()}
                className="w-7 h-7 rounded-full bg-[var(--fg)] hover:bg-white disabled:opacity-30 text-[var(--bg-app)] flex items-center justify-center"
                title="Send"
              >
                <ArrowUp className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
