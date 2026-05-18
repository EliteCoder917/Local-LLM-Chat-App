import { useEffect } from 'react';
import { api, BACKEND_HTTP } from '../ipc/bridge';
import { useStore } from '../state/store';
import type { Attachment, DownloadJob, Message, ModelStatus } from '../state/types';
import { roughTokens, stripThinking } from '../lib/parseThinking';

const KEEP_RECENT = 6;         // always keep this many trailing messages verbatim
const RESERVE_FOR_REPLY = 1000; // tokens budgeted for the model's response
const COMPACT_THRESHOLD = 0.70; // trigger compaction at 70% of n_ctx

// Per-message timing tracked outside React state so streaming deltas don't
// pay a re-render cost for every token. Flushed into the message on `message-end`.
const _timing = new Map<string, { start: number; firstDelta: number | null }>();

interface LlmEvent {
  type:
    | 'message-start'
    | 'message-delta'
    | 'message-end'
    | 'tool-call'
    | 'tool-result'
    | 'agent-step'
    | 'model-status'
    | 'download-progress'
    | 'error';
  msgId: string;
  id?: string;    // download-progress carries the job id here, not msgId
  role?: 'assistant';
  delta?: string;
  toolCall?: { id: string; tool: string; args: Record<string, unknown> };
  toolResult?: { id: string; result?: string; error?: string };
  message?: string;
  step?: number;
  // model-status events carry the full ModelStatus snapshot inline
  status?: ModelStatus['status'];
  engine?: ModelStatus['engine'];
  model?: string;
  modelPath?: string;
  loadedKey?: string | null;
  currentKey?: string;
  loadMs?: number | null;
  progress?: number | null;
  visionActive?: boolean;
  visionHandler?: string | null;
  // download-progress events carry the full DownloadJob inline
  repo?: string;
  filename?: string;
  downloaded?: number;
  total?: number;
  percent?: number;
  speedMBs?: number;
}

export function useChat() {
  useEffect(() => {
    return api.llm.onEvent((raw) => {
      const e = raw as LlmEvent;
      const s = useStore.getState();
      switch (e.type) {
        case 'download-progress':
          if (e.id && e.repo && e.filename) {
            const job: DownloadJob = {
              id: e.id,
              repo: e.repo,
              filename: e.filename,
              status: (e.status ?? 'queued') as DownloadJob['status'],
              downloaded: e.downloaded ?? 0,
              total: e.total ?? 0,
              percent: e.percent ?? 0,
              speedMBs: e.speedMBs ?? 0,
              error: e.message ?? '',
            };
            s.updateDownload(job);
          }
          break;
        case 'model-status':
          if (e.status) {
            s.setModelStatus({
              status: e.status,
              message: e.message ?? '',
              engine: e.engine ?? 'llama-cpp',
              model: e.model ?? '',
              modelPath: e.modelPath ?? '',
              loadedKey: e.loadedKey ?? null,
              currentKey: e.currentKey ?? '',
              loadMs: e.loadMs ?? null,
              progress: e.progress ?? null,
              visionActive: e.visionActive ?? false,
              visionHandler: e.visionHandler ?? null,
            });
          }
          break;
        case 'message-start':
          _timing.set(e.msgId, { start: performance.now(), firstDelta: null });
          s.appendMessage({
            id: e.msgId,
            role: 'assistant',
            content: '',
            ts: Date.now(),
          });
          s.setStreaming(true);
          break;
        case 'message-delta': {
          const t = _timing.get(e.msgId);
          if (t && t.firstDelta == null && e.delta) {
            t.firstDelta = performance.now();
          }
          s.patchMessage(e.msgId, (m) => ({
            ...m,
            content: m.content + (e.delta ?? ''),
          }));
          break;
        }
        case 'message-end': {
          const t = _timing.get(e.msgId);
          if (t) {
            const now = performance.now();
            const ttftMs = t.firstDelta != null ? t.firstDelta - t.start : undefined;
            const durationMs = t.firstDelta != null ? now - t.firstDelta : undefined;
            s.patchMessage(e.msgId, (m) => {
              const totalTokens = roughTokens(m.content);
              const tokensPerSec = durationMs && durationMs > 0
                ? (totalTokens / (durationMs / 1000))
                : undefined;
              return { ...m, ttftMs, durationMs, totalTokens, tokensPerSec };
            });
            _timing.delete(e.msgId);
          }
          s.setStreaming(false);
          break;
        }
        case 'tool-call':
          if (e.toolCall) {
            s.upsertToolCall(e.msgId, {
              id: e.toolCall.id,
              tool: e.toolCall.tool,
              args: e.toolCall.args,
              status: 'running',
            });
          }
          break;
        case 'tool-result':
          if (e.toolResult) {
            s.upsertToolCall(e.msgId, {
              id: e.toolResult.id,
              result: e.toolResult.result,
              error: e.toolResult.error,
              status: e.toolResult.error ? 'error' : 'done',
            });
          }
          break;
        case 'error':
          s.appendMessage({
            id: crypto.randomUUID(),
            role: 'system',
            content: `**Error:** ${e.message ?? 'Unknown error'}`,
            ts: Date.now(),
          });
          s.setStreaming(false);
          break;
      }
    });
  }, []);
}

function _materializeForModel(m: Message) {
  // Strip <thinking> sections from assistant turns so the model's own
  // reasoning doesn't keep getting fed back into context.
  let textContent = m.role === 'assistant' ? stripThinking(m.content) : m.content;

  const attachments = m.attachments ?? [];
  if (attachments.length === 0) {
    return { role: m.role, content: textContent };
  }

  const settings = useStore.getState().settings;
  const modelStatus = useStore.getState().modelStatus;
  // Only build structured OpenAI-style content (with image_url blocks) when
  // the LOADED engine actually has a vision ChatHandler wired up. The library
  // may have paired an mmproj optimistically, but if the engine refused it
  // (unknown family, wrong projector) we must NOT send raw image_url blocks
  // — they'd get stringified by the text chat formatter and the model would
  // see a giant base64 blob and hallucinate from filename/structure.
  const isVisionModel = modelStatus.status === 'loaded' && !!modelStatus.visionActive;

  // Text attachments — always inlined as code blocks (works for any model).
  const textParts: string[] = [];
  const images: { name: string; dataUri: string }[] = [];
  for (const a of attachments) {
    if (a.kind === 'text' && a.content) {
      const truncated = a.content.length > 100_000
        ? a.content.slice(0, 100_000) + '\n... [truncated]'
        : a.content;
      textParts.push(`**\`${a.name}\`:**\n\`\`\`\n${truncated}\n\`\`\``);
    } else if (a.kind === 'image' && a.dataUri) {
      images.push({ name: a.name, dataUri: a.dataUri });
    }
  }

  if (textParts.length > 0) {
    textContent = textContent
      ? `${textContent}\n\n${textParts.join('\n\n')}`
      : textParts.join('\n\n');
  }

  if (images.length === 0) {
    return { role: m.role, content: textContent };
  }

  if (isVisionModel) {
    // Build OpenAI-style structured content. llama-cpp-python's vision chat
    // handler interprets `image_url` blocks via the CLIP projection.
    const blocks: Array<
      { type: 'text'; text: string } |
      { type: 'image_url'; image_url: { url: string } }
    > = [];
    if (textContent) blocks.push({ type: 'text', text: textContent });
    for (const img of images) {
      blocks.push({ type: 'image_url', image_url: { url: img.dataUri } });
    }
    return { role: m.role, content: blocks };
  }

  // Non-vision model: honor the legacy fallback for users who opted in via
  // the sendImagesAsBase64 toggle (might produce garbage but at least lets
  // them experiment). Default: placeholder so context isn't burned.
  for (const img of images) {
    if (settings.sendImagesAsBase64) {
      textContent += `\n\n![${img.name}](${img.dataUri})`;
    } else {
      textContent += `\n\n[Image attached: ${img.name} — load a vision model (with mmproj) for the model to actually see it]`;
    }
  }
  return { role: m.role, content: textContent };
}

/** Token cost estimate for an attachment as it will be sent to the model.
 *
 * For images going through a vision projector (CLIP / SigLIP), the model
 * doesn't actually see the raw base64 — it sees a fixed-ish number of patch
 * tokens after CLIP projection. Qwen-VL at our 1024px downsample cap emits
 * ~1000 tokens per image. Using `dataUri.length / 4` here was *wildly* off
 * (200-500 KB base64 → 50-100k estimated tokens) and made the context wheel
 * jump to 100% on a single image even at n_ctx=40000.
 *
 * For the legacy text-base64 fallback (vision off but user opted into
 * `sendImagesAsBase64`), the model really does see the raw base64 chars, so
 * the char-count estimate is correct there. */
const IMAGE_VISION_TOKEN_ESTIMATE = 1024;
export function estimateAttachmentTokens(
  a: Attachment,
  sendVisionBytes: boolean,
  visionActive: boolean = false,
): number {
  if (a.kind === 'text') return roughTokens(a.content ?? '');
  if (a.kind === 'image') {
    if (visionActive) return IMAGE_VISION_TOKEN_ESTIMATE;
    if (sendVisionBytes && a.dataUri) return Math.ceil(a.dataUri.length / 4);
    return 8;  // just the placeholder string
  }
  return 0;
}

/** Estimate the prompt size in tokens for the budget check. */
function estimatePromptTokens(): number {
  const s = useStore.getState();
  const msgs = s.activeMessages();
  let total = roughTokens(s.settings.systemPrompt);
  for (const m of msgs) {
    total += roughTokens(_materializeForModel(m).content);
  }
  return total;
}

/** If the running conversation is over budget, compact older middle messages
 *  into a single summary via the backend, then mutate the conversation in
 *  place. The system prompt and the last `KEEP_RECENT` messages are preserved. */
async function maybeCompact() {
  const s = useStore.getState();
  const nCtx = s.settings.nCtx || 4096;
  const budget = Math.max(1024, Math.floor(nCtx * COMPACT_THRESHOLD)) - RESERVE_FOR_REPLY;
  const tokens = estimatePromptTokens();
  if (tokens <= budget) return;

  const msgs = s.activeMessages();
  // Anything older than the last KEEP_RECENT messages, AND not already a summary,
  // is eligible for compaction.
  if (msgs.length <= KEEP_RECENT + 1) return;
  const cutoff = msgs.length - KEEP_RECENT;
  const toCompact = msgs.slice(0, cutoff).filter((m) => m.kind !== 'summary');
  if (toCompact.length < 2) return;

  s.setModelStatus({
    ...s.modelStatus,
    status: s.modelStatus.status === 'loaded' ? 'loaded' : s.modelStatus.status,
    message: `Compacting ${toCompact.length} earlier messages…`,
  });

  let summary = '';
  try {
    const r = await fetch(`${BACKEND_HTTP}/summarize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: toCompact.map(_materializeForModel) }),
    });
    if (!r.ok) {
      console.error('compaction failed:', await r.text());
      return;
    }
    const data = await r.json();
    summary = (data.summary ?? '').trim();
  } catch (e) {
    console.error(e);
    return;
  }
  if (!summary) return;

  // Replace the compacted slice + any prior summary with one new summary.
  const synthetic: Message = {
    id: crypto.randomUUID(),
    role: 'system',
    content: summary,
    ts: Date.now(),
    kind: 'summary',
    compactedCount: toCompact.length,
  };
  // Keep any existing summaries that precede the cutoff too — drop them, the
  // new summary subsumes them.
  const tail = msgs.slice(cutoff);
  s.replaceActiveMessages([synthetic, ...tail]);
}

async function _sendCurrent() {
  await maybeCompact();
  const s = useStore.getState();
  const messages = s.activeMessages().map(_materializeForModel);
  await api.llm.send({
    messages,
    settings: s.settings,
    workspace: s.workspace,
    // Tools are only available in the Code tab. Chat tab is pure chat
    // (no tool catalog injected, no tool calls). This overrides the
    // backend's CONFIG.agent_mode for THIS request only.
    agentMode: s.tab === 'code',
  });
}

export async function sendChat(text: string, attachments: Attachment[] = []) {
  const s = useStore.getState();
  if (s.streaming) return;
  const hasAnything = text.trim().length > 0 || attachments.length > 0;
  if (!hasAnything) return;
  if (!s.activeId) s.newConversation();
  s.appendMessage({
    id: crypto.randomUUID(),
    role: 'user',
    content: text,
    attachments: attachments.length > 0 ? attachments : undefined,
    ts: Date.now(),
  });
  await _sendCurrent();
}

/** Drop the last assistant turn and re-ask the model. */
export async function regenerateLast() {
  const s = useStore.getState();
  if (s.streaming) return;
  const msgs = s.activeMessages();
  // Walk back from the end, removing assistant + tool turns until we hit the
  // last user message; that user turn is what we'll resend from.
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].role === 'user') {
      s.truncateAfter(msgs[i].id);
      await _sendCurrent();
      return;
    }
  }
}

/** Edit a past user message; truncate everything after it and re-ask. */
export async function editAndResend(id: string, newContent: string) {
  const s = useStore.getState();
  if (s.streaming) return;
  s.editUserMessage(id, newContent);
  await _sendCurrent();
}
