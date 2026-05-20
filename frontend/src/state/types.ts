export type Role = 'user' | 'assistant' | 'system' | 'tool';

export interface ToolCall {
  id: string;
  tool: string;
  args: Record<string, unknown>;
  result?: string;
  error?: string;
  status: 'pending' | 'running' | 'done' | 'error';
}

export interface Attachment {
  id: string;
  kind: 'image' | 'text';
  name: string;
  // image: data URI for thumbnail rendering + optional sending to vision model
  dataUri?: string;
  // text/code: raw content (truncated to ~100k chars at attach time)
  content?: string;
  size: number;             // bytes — original file size for display
  mime?: string;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  attachments?: Attachment[];   // first-class — NOT stuffed into content
  toolCalls?: ToolCall[];
  ts: number;
  // populated for assistant turns once streaming finishes
  ttftMs?: number;          // time to first token, milliseconds
  durationMs?: number;      // total stream duration
  totalTokens?: number;     // rough char/4 estimate
  tokensPerSec?: number;
  // marks a synthetic message produced by auto-compaction; renders distinctly
  kind?: 'summary';
  compactedCount?: number;  // how many original messages got rolled into this
  // Hidden suffix appended to the content when the message is sent to the
  // model — currently used for the Qwen `/think` / `/no_think` toggle so
  // the markers don't show up in the user's own message bubble.
  sendSuffix?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface FileNode {
  path: string;
  name: string;
  isDir: boolean;
  children?: FileNode[];
}

export type Engine = 'llama-cpp' | 'ollama';
export type Tab = 'chat' | 'code';

export type ModelStatusKind = 'idle' | 'loading' | 'loaded' | 'error';

export interface ModelStatus {
  status: ModelStatusKind;
  message: string;
  engine: Engine;
  model: string;
  modelPath: string;
  loadedKey: string | null;
  currentKey: string;
  loadMs: number | null;
  progress: number | null;     // 0..1 during loading; null otherwise
  // True only when the loaded engine has a vision ChatHandler wired up — the
  // library may pair an mmproj optimistically, but if the family is unknown
  // the engine refuses vision rather than producing noise from a wrong handler.
  visionActive?: boolean;
  visionHandler?: string | null;  // class name, e.g. "Qwen25VLChatHandler"
  // True only when the loaded tokenizer recognises a `<think>` token. Used
  // to gate the Smart/Quick picker so it disappears on non-thinking models.
  supportsThinking?: boolean;
  // Actual context window the engine loaded with (the resolved value when
  // n_ctx is on Auto). null when no model is loaded.
  nCtx?: number | null;
}

export interface Settings {
  theme: 'dark' | 'light';
  model: string;
  engine: Engine;
  modelPath: string;
  ollamaUrl: string;
  workspace: string;
  agentMode: boolean;
  maxIterations: number;
  temperature: number;
  systemPrompt: string;
  nCtx: number;
  gpuOffloadGb: number;   // how much of the model to put on GPU VRAM
  sendImagesAsBase64: boolean; // off by default — only on for vision models
  // Manual vision handler family override. "" = auto-detect from arch/filename.
  // Useful when an mmproj is shipped without arch metadata that names its
  // family (e.g. some custom Qwen-VL fine-tunes only set the LLM's arch).
  visionHandler: string;
  // Thinking toggle for models that support reasoning (currently the Qwen3
  // family). 'smart' = let the model think (default — Qwen3 always thinks
  // unless told not to), 'quick' = pre-fill an empty <think> block so the
  // model skips reasoning. For models without thinking support the UI hides
  // this control entirely.
  thinkingMode: 'smart' | 'quick';
  // Chat behaviour mode (replaces the old Smart/Quick thinking picker):
  //   normal — pure chat, no tools
  //   cowork — agent tools enabled in chat (read/write/edit files, like Code tab)
  //   search — web search + page fetch (not wired yet; shown as "soon")
  chatMode: ChatMode;
}

export type ThinkingMode = 'smart' | 'quick';
export type ChatMode = 'normal' | 'cowork' | 'search';

// Vision handler slugs accepted by the backend. Keep in sync with
// `LlamaCppEngine._HANDLER_BY_SLUG` in llama_cpp_engine.py.
export const VISION_HANDLER_OPTIONS: { value: string; label: string }[] = [
  { value: '',              label: 'Auto-detect' },
  { value: 'qwen25vl',      label: 'Qwen-VL (2 / 2.5 / 3 VL)' },
  { value: 'llama32vision', label: 'Llama 3.2 Vision (mllama)' },
  { value: 'minicpmv',      label: 'MiniCPM-V (2.6+)' },
  { value: 'moondream',     label: 'Moondream' },
  { value: 'llava16',       label: 'LLaVA 1.6' },
  { value: 'llava15',       label: 'LLaVA 1.5' },
  { value: 'nanollava',     label: 'NanoLLaVA' },
  { value: 'obsidian',      label: 'Obsidian' },
];


export interface LibraryModel {
  id: string;
  name: string;
  path: string;
  sizeGb: number;
  arch: string | null;
  blockCount: number | null;
  trainedContext: number | null;
  mmprojPath: string | null;
  mmprojName: string | null;
  isVision: boolean;
}

export interface DownloadJob {
  id: string;
  repo: string;
  filename: string;
  status: 'queued' | 'downloading' | 'done' | 'error' | 'cancelled';
  downloaded: number;
  total: number;
  percent: number;
  speedMBs: number;
  error: string;
}

export interface SystemInfo {
  ramAvailableGb: number | null;
  ramTotalGb: number | null;
  gpus: { name: string; free_gb: number; total_gb: number }[];
  model: {
    sizeGb?: number;
    arch?: string;
    blockCount?: number;
    trainedContext?: number;
    gbPerLayer?: number;
    embeddingLength?: number;
    headCount?: number;
    headCountKv?: number;
    ropeDim?: number;
    headAndEmbedGb?: number;
  };
}
