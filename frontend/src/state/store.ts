import { create } from 'zustand';
import type {
  Conversation,
  DownloadJob,
  LibraryModel,
  Message,
  ModelStatus,
  Settings,
  Tab,
  ToolCall,
} from './types';
import { api, BACKEND_HTTP, type PermissionRequest } from '../ipc/bridge';

const CONV_KEY = 'localAiIde.conversations';

interface State {
  conversations: Conversation[];
  activeId: string | null;
  streaming: boolean;
  // Transient: true while maybeCompact() is running. Drives the
  // "Compacting earlier messages…" indicator. Doesn't persist.
  compacting: boolean;
  modelStatus: ModelStatus;
  libraryModels: LibraryModel[];
  libraryRoot: string;
  downloads: Record<string, DownloadJob>;
  libraryOpen: boolean;
  settings: Settings;
  perms: Record<string, boolean>;
  workspace: string;
  openFile: string | null;
  openFileContent: string;
  fileDirty: boolean;
  tab: Tab;

  init: () => Promise<void>;
  setTab: (t: Tab) => void;
  setModelStatus: (s: ModelStatus) => void;
  refreshModelStatus: () => Promise<void>;
  loadModel: () => Promise<void>;
  unloadModel: () => Promise<void>;

  refreshLibrary: () => Promise<void>;
  selectLibraryModel: (id: string) => Promise<void>;
  deleteLibraryModel: (id: string) => Promise<void>;
  startDownload: (input: string) => Promise<{ ok: boolean; error?: string }>;
  cancelDownload: (jobId: string) => Promise<void>;
  updateDownload: (job: DownloadJob) => void;
  setLibraryOpen: (open: boolean) => void;

  newConversation: () => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  renameConversation: (id: string, title: string) => void;
  activeMessages: () => Message[];
  appendMessage: (m: Message) => void;
  patchMessage: (id: string, updater: (m: Message) => Message) => void;
  upsertToolCall: (msgId: string, t: Partial<ToolCall> & { id: string }) => void;
  deleteMessage: (id: string) => void;
  truncateAfter: (id: string) => void;
  editUserMessage: (id: string, newContent: string) => void;
  replaceActiveMessages: (messages: Message[]) => void;

  setStreaming: (s: boolean) => void;
  setCompacting: (v: boolean) => void;
  // The pending tool-permission request awaiting a custom-modal decision,
  // or null when none. Set by the permission:request IPC listener.
  permissionRequest: PermissionRequest | null;
  setPermissionRequest: (r: PermissionRequest | null) => void;
  setPerms: (perms: Record<string, boolean>) => void;
  setSettings: (p: Partial<Settings>) => Promise<void>;
  setPerm: (k: string, v: boolean) => Promise<void>;
  setOpenFile: (path: string | null, content: string, dirty?: boolean) => void;
}

const DEFAULT_SETTINGS: Settings = {
  theme: 'dark',
  model: 'local-gguf',
  engine: 'llama-cpp',
  modelPath: '',
  ollamaUrl: 'http://127.0.0.1:11434',
  workspace: '',
  agentMode: true,
  maxIterations: 10,
  temperature: 0.7,
  systemPrompt: 'You are a helpful local AI assistant. Be concise and accurate.',
  nCtx: 4096,
  // -1 is the "Auto" sentinel: backend picks the largest offload that fits
  // alongside KV cache + compute buffer at load time. Better default than 0
  // (CPU-only) since most users want GPU used by default.
  gpuOffloadGb: -1,
  // ON by default — the user wants the model to "see" attached images.
  // If they're on a non-vision model and it produces garbage on images,
  // they can turn this off in Settings → Model → Attachments.
  sendImagesAsBase64: true,
  visionHandler: '',
  thinkingMode: 'smart',
  chatMode: 'normal',
};

function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(CONV_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw) as Conversation[];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function saveConversations(list: Conversation[]) {
  try {
    localStorage.setItem(CONV_KEY, JSON.stringify(list));
  } catch {
    /* ignore quota */
  }
}

function persist(get: () => State) {
  saveConversations(get().conversations);
}

function updateActive(
  get: () => State,
  set: (p: Partial<State>) => void,
  mutate: (c: Conversation) => Conversation,
) {
  const id = get().activeId;
  if (!id) return;
  const list = get().conversations.map((c) =>
    c.id === id ? { ...mutate(c), updatedAt: Date.now() } : c,
  );
  set({ conversations: list });
  saveConversations(list);
}

export const useStore = create<State>((set, get) => ({
  conversations: loadConversations(),
  activeId: null,
  streaming: false,
  compacting: false,
  permissionRequest: null,
  modelStatus: {
    status: 'idle',
    message: '',
    engine: 'llama-cpp',
    model: '',
    modelPath: '',
    loadedKey: null,
    currentKey: '',
    loadMs: null,
    progress: null,
  },
  libraryModels: [],
  libraryRoot: '',
  downloads: {},
  libraryOpen: false,
  settings: DEFAULT_SETTINGS,
  perms: {},
  workspace: '',
  openFile: null,
  openFileContent: '',
  fileDirty: false,
  tab: 'chat',

  async init() {
    const allSettings = await api.settings.get();
    const stored = (allSettings.settings as Partial<Settings>) ?? {};
    const merged: Settings = { ...DEFAULT_SETTINGS, ...stored };
    // Migrate: 'deep' used to be a third thinking mode but it's functionally
    // identical to 'smart' for Qwen3 (both produce reasoning). Coerce any
    // legacy stored value to 'smart' so the picker shows a valid option.
    if ((merged.thinkingMode as string) === 'deep') merged.thinkingMode = 'smart';
    // Migrate: the old default was 0 (CPU-only). The new default is -1 (Auto).
    // Users who deliberately picked CPU-only can re-pick it on the slider; this
    // only catches users who never touched the field.
    if (merged.gpuOffloadGb === 0) merged.gpuOffloadGb = -1;
    const perms = await api.perms.get();
    const convs = get().conversations;
    set({
      settings: merged,
      perms,
      workspace: merged.workspace ?? '',
      activeId: convs[0]?.id ?? null,
    });
    await get().refreshModelStatus();
    await get().refreshLibrary();
    // Replay the persisted settings to the backend so its CONFIG matches
    // what the renderer is showing. Otherwise the first Load uses backend
    // defaults (n_ctx=4096, gpu_offload_gb=0) regardless of what's on disk.
    try { await api.settings.set({ settings: merged }); } catch { /* ignore */ }
  },

  setTab(t) {
    set({ tab: t });
  },

  newConversation() {
    const id = crypto.randomUUID();
    const c: Conversation = {
      id,
      title: 'New chat',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    const list = [c, ...get().conversations];
    set({ conversations: list, activeId: id });
    saveConversations(list);
    return id;
  },

  selectConversation(id) {
    set({ activeId: id });
  },

  deleteConversation(id) {
    const list = get().conversations.filter((c) => c.id !== id);
    const next = get().activeId === id ? list[0]?.id ?? null : get().activeId;
    set({ conversations: list, activeId: next });
    saveConversations(list);
  },

  renameConversation(id, title) {
    const list = get().conversations.map((c) =>
      c.id === id ? { ...c, title, updatedAt: Date.now() } : c,
    );
    set({ conversations: list });
    saveConversations(list);
  },

  activeMessages() {
    const id = get().activeId;
    if (!id) return [];
    return get().conversations.find((c) => c.id === id)?.messages ?? [];
  },

  appendMessage(m) {
    if (!get().activeId) {
      get().newConversation();
    }
    updateActive(get, set, (c) => {
      const next = { ...c, messages: [...c.messages, m] };
      // auto-title from first user message
      if (c.title === 'New chat' && m.role === 'user' && m.content.trim()) {
        next.title = m.content.trim().slice(0, 48);
      }
      return next;
    });
  },

  patchMessage(id, updater) {
    updateActive(get, set, (c) => ({
      ...c,
      messages: c.messages.map((m) => (m.id === id ? updater(m) : m)),
    }));
  },

  upsertToolCall(msgId, t) {
    updateActive(get, set, (c) => ({
      ...c,
      messages: c.messages.map((m) => {
        if (m.id !== msgId) return m;
        const calls = m.toolCalls ?? [];
        const i = calls.findIndex((tc) => tc.id === t.id);
        if (i === -1) {
          const next: ToolCall = {
            id: t.id,
            tool: t.tool ?? '',
            args: t.args ?? {},
            result: t.result,
            error: t.error,
            status: t.status ?? 'pending',
          };
          return { ...m, toolCalls: [...calls, next] };
        }
        const merged = { ...calls[i], ...t };
        return { ...m, toolCalls: calls.map((tc, j) => (j === i ? merged : tc)) };
      }),
    }));
  },

  setStreaming(s) {
    set({ streaming: s });
  },

  setCompacting(v) {
    set({ compacting: v });
  },

  setPermissionRequest(r) {
    set({ permissionRequest: r });
  },

  deleteMessage(id) {
    updateActive(get, set, (c) => ({
      ...c,
      messages: c.messages.filter((m) => m.id !== id),
    }));
  },

  truncateAfter(id) {
    updateActive(get, set, (c) => {
      const idx = c.messages.findIndex((m) => m.id === id);
      if (idx === -1) return c;
      return { ...c, messages: c.messages.slice(0, idx + 1) };
    });
  },

  editUserMessage(id, newContent) {
    updateActive(get, set, (c) => {
      const idx = c.messages.findIndex((m) => m.id === id);
      if (idx === -1) return c;
      const edited = { ...c.messages[idx], content: newContent };
      // Truncate everything after the edited message — the regenerate
      // happens via a separate sendChat call wrapping this.
      return { ...c, messages: [...c.messages.slice(0, idx), edited] };
    });
  },

  replaceActiveMessages(messages) {
    updateActive(get, set, (c) => ({ ...c, messages }));
  },

  setModelStatus(s) {
    set({ modelStatus: s });
  },

  async refreshModelStatus() {
    try {
      const r = await fetch(`${BACKEND_HTTP}/model/status`);
      if (r.ok) set({ modelStatus: (await r.json()) as ModelStatus });
    } catch {
      /* backend not up yet */
    }
  },

  async loadModel() {
    // Push current settings to the backend first so it loads the right .gguf.
    await api.settings.set({ settings: get().settings });
    try {
      const r = await fetch(`${BACKEND_HTTP}/model/load`, { method: 'POST' });
      if (r.ok) set({ modelStatus: (await r.json()) as ModelStatus });
    } catch (e) {
      console.error(e);
    }
  },

  async unloadModel() {
    try {
      const r = await fetch(`${BACKEND_HTTP}/model/unload`, { method: 'POST' });
      if (r.ok) set({ modelStatus: (await r.json()) as ModelStatus });
    } catch (e) {
      console.error(e);
    }
  },

  async refreshLibrary() {
    try {
      const r = await fetch(`${BACKEND_HTTP}/library`);
      if (!r.ok) return;
      const data = (await r.json()) as { root: string; models: LibraryModel[] };
      set({ libraryModels: data.models, libraryRoot: data.root });
    } catch {
      /* backend not up yet */
    }
  },

  async selectLibraryModel(id) {
    try {
      const r = await fetch(`${BACKEND_HTTP}/library/select`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      });
      if (!r.ok) {
        console.error('select failed:', await r.text());
        return;
      }
      const status = (await r.json()) as ModelStatus;
      set({ modelStatus: status });
      // Mirror the resolved path back into settings so subsequent chat calls
      // (which send the full settings object inline) don't re-blank model_path.
      if (status.modelPath && status.modelPath !== get().settings.modelPath) {
        const next = { ...get().settings, modelPath: status.modelPath };
        set({ settings: next });
        try { await api.settings.set({ settings: next }); } catch { /* ignore */ }
      }
    } catch (e) {
      console.error(e);
    }
  },

  async deleteLibraryModel(id) {
    try {
      const r = await fetch(`${BACKEND_HTTP}/library/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });
      if (r.ok) await get().refreshLibrary();
    } catch (e) {
      console.error(e);
    }
  },

  async startDownload(input) {
    try {
      const r = await fetch(`${BACKEND_HTTP}/library/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input }),
      });
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try {
          const body = await r.json();
          msg = body.detail ?? msg;
        } catch { /* ignore */ }
        return { ok: false, error: msg };
      }
      const job = (await r.json()) as DownloadJob;
      get().updateDownload(job);
      return { ok: true };
    } catch (e) {
      return { ok: false, error: (e as Error).message };
    }
  },

  async cancelDownload(jobId) {
    try {
      await fetch(`${BACKEND_HTTP}/library/downloads/${jobId}/cancel`, { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  },

  updateDownload(job) {
    set({ downloads: { ...get().downloads, [job.id]: job } });
    // When a download completes, refresh the library so the new model shows up
    if (job.status === 'done') {
      get().refreshLibrary();
    }
  },

  setLibraryOpen(open) {
    set({ libraryOpen: open });
    if (open) get().refreshLibrary();
  },

  async setSettings(p) {
    const next = { ...get().settings, ...p };
    set({ settings: next, workspace: next.workspace ?? get().workspace });
    await api.settings.set({ settings: next });
  },

  async setPerm(k, v) {
    const perms = await api.perms.set(k, v);
    set({ perms });
  },

  // Replace the whole perms map (e.g. after "Allow & remember" in the
  // permission modal persisted a key via the main process). Keeps the
  // Settings toggles in sync with what was actually saved to disk.
  setPerms(perms) {
    set({ perms });
  },

  setOpenFile(path, content, dirty = false) {
    set({ openFile: path, openFileContent: content, fileDirty: dirty });
  },
}));

// Quiet down "unused" warning for persist helper.
void persist;
