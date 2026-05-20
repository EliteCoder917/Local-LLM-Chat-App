import { useEffect, useState } from 'react';
import { api, BACKEND_HTTP } from '../ipc/bridge';
import { useStore } from '../state/store';
import type { Attachment, DownloadJob, Message, ModelStatus } from '../state/types';
import { roughTokens, stripThinking } from '../lib/parseThinking';

const KEEP_RECENT = 6;          // always keep this many trailing messages verbatim
const RESERVE_FOR_REPLY = 1000; // tokens budgeted for the model's response
// Lower than the old 0.70 because the budget check now uses exact /tokenize
// counts (chat-template chrome + tool catalog + image cost included) instead
// of the chars/4 estimator that ignored attachments — false positives are
// gone, so we can trigger sooner.
const COMPACT_THRESHOLD = 0.65;
// Cap on how many independent summary bubbles can exist before we collapse
// them into one. Without a cap, repeated compactions append summaries
// indefinitely; without merging, each new compaction also pays to re-read
// the existing chain on every send.
const MAX_SUMMARY_CHAIN = 3;
// Cap caption batch size — beyond ~4 images per /caption call, vision n_ctx
// fills up fast and the model's caption quality degrades sharply.
const CAPTION_BATCH_SIZE = 4;

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
  supportsThinking?: boolean;
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
              supportsThinking: e.supportsThinking ?? false,
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

function _materializeForModel(m: Message, captionMap?: Map<string, string>) {
  // Strip <thinking> sections from assistant turns so the model's own
  // reasoning doesn't keep getting fed back into context.
  let textContent = m.role === 'assistant' ? stripThinking(m.content) : m.content;
  // Append the hidden send-suffix (e.g. " /no_think") so the model sees it
  // but the message bubble in the UI doesn't show it.
  if (m.role === 'user' && m.sendSuffix) {
    textContent = textContent + m.sendSuffix;
  }

  // Summary messages are inserted by the compactor; preserve their role
  // (system) and content so the model can read them as context. They never
  // carry attachments worth re-materializing.
  if (m.kind === 'summary') {
    return { role: m.role, content: textContent };
  }

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
  // Each image either becomes a structured `image_url` block (regular send)
  // OR a `[image: <caption>]` text line (compaction with caption map).
  const captionedImageLines: string[] = [];
  const rawImages: { name: string; dataUri: string }[] = [];
  for (const a of attachments) {
    if (a.kind === 'text' && a.content) {
      const truncated = a.content.length > 100_000
        ? a.content.slice(0, 100_000) + '\n... [truncated]'
        : a.content;
      textParts.push(`**\`${a.name}\`:**\n\`\`\`\n${truncated}\n\`\`\``);
    } else if (a.kind === 'image') {
      // Caption map takes precedence: when present, every image becomes a
      // text placeholder. This is what the compactor passes in so the
      // /summarize call never sees base64.
      const caption = captionMap?.get(a.id);
      if (caption != null) {
        captionedImageLines.push(`[image: ${caption}]`);
      } else if (a.dataUri) {
        rawImages.push({ name: a.name, dataUri: a.dataUri });
      } else {
        // dataUri was garbage-collected (e.g. after compaction). Fall back
        // to filename so the model still knows an image existed there.
        captionedImageLines.push(`[image: ${a.name} (data unavailable)]`);
      }
    }
  }

  if (textParts.length > 0) {
    textContent = textContent
      ? `${textContent}\n\n${textParts.join('\n\n')}`
      : textParts.join('\n\n');
  }
  if (captionedImageLines.length > 0) {
    textContent = textContent
      ? `${textContent}\n\n${captionedImageLines.join('\n')}`
      : captionedImageLines.join('\n');
  }

  if (rawImages.length === 0) {
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
    for (const img of rawImages) {
      blocks.push({ type: 'image_url', image_url: { url: img.dataUri } });
    }
    return { role: m.role, content: blocks };
  }

  // Non-vision model: honor the legacy fallback for users who opted in via
  // the sendImagesAsBase64 toggle (might produce garbage but at least lets
  // them experiment). Default: placeholder so context isn't burned.
  for (const img of rawImages) {
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
// Tokens we charge against the context budget per attached image, when the
// engine has vision active. The TRUE cost varies wildly by projector. With
// our 384px-max downsample plus typical "no-merge" Qwen3-VL fine-tunes the
// real cost lands around 6000-8000 tokens. Setting the wheel estimate at
// 7000 keeps it from over-promising free space.
const IMAGE_VISION_TOKEN_ESTIMATE = 7000;
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

/** Exact token count for the active conversation via the backend's loaded
 *  tokenizer (chat-template chrome + tool catalog + image cost). Returns
 *  null when no model is loaded so callers can fall back. */
async function getPromptTokens(): Promise<number | null> {
  const s = useStore.getState();
  if (s.modelStatus.status !== 'loaded') return null;
  try {
    const messages = s.activeMessages().map((m) => _materializeForModel(m));
    const r = await fetch(`${BACKEND_HTTP}/tokenize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages,
        system_prompt: s.settings.systemPrompt,
        agent_mode: s.tab === 'code',
        workspace: s.workspace,
      }),
    });
    if (!r.ok) return null;
    const d = await r.json();
    return typeof d.tokens === 'number' ? d.tokens : null;
  } catch {
    return null;
  }
}

/** Rough-token fallback used when no model is loaded. Includes attachment
 *  costs (vision images, base64) so image-heavy chats still trigger. */
function roughPromptTokens(): number {
  const s = useStore.getState();
  const msgs = s.activeMessages();
  const visionActive = !!s.modelStatus.visionActive;
  const sendsImageBytes = visionActive || s.settings.sendImagesAsBase64;
  let total = roughTokens(s.settings.systemPrompt);
  for (const m of msgs) {
    total += roughTokens(m.content);
    for (const a of (m.attachments ?? [])) {
      total += estimateAttachmentTokens(a, sendsImageBytes, visionActive);
    }
  }
  return total;
}

/** Caption every image attachment in `slice` via the vision model, batched.
 *  Returns a Map<attachmentId, caption>. When no vision model is loaded the
 *  captions are filename-based placeholders so compaction still proceeds. */
async function captionImagesIn(
  slice: Message[],
  signal?: AbortSignal,
): Promise<Map<string, string>> {
  const map = new Map<string, string>();
  // Collect every image attachment that still has a dataUri (already-GC'd
  // images are skipped — they were captioned in a prior compaction).
  type Img = { id: string; name: string; dataUri: string };
  const images: Img[] = [];
  for (const m of slice) {
    for (const a of (m.attachments ?? [])) {
      if (a.kind === 'image' && a.dataUri) {
        images.push({ id: a.id, name: a.name, dataUri: a.dataUri });
      }
    }
  }
  if (images.length === 0) return map;

  const visionActive = !!useStore.getState().modelStatus.visionActive;
  if (!visionActive) {
    // Graceful fallback — no model to caption with, but we still want the
    // summarizer to know an image WAS there. Use the filename.
    for (const img of images) map.set(img.id, img.name);
    return map;
  }

  // Batched calls: too many images in one /caption send blows past the
  // vision model's effective context for image tokens.
  for (let i = 0; i < images.length; i += CAPTION_BATCH_SIZE) {
    const batch = images.slice(i, i + CAPTION_BATCH_SIZE);
    try {
      const r = await fetch(`${BACKEND_HTTP}/caption`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          images: batch.map((b) => ({ id: b.id, dataUri: b.dataUri })),
        }),
        signal,
      });
      if (!r.ok) {
        // Fall back to filenames for this batch on error.
        for (const img of batch) map.set(img.id, img.name);
        continue;
      }
      const data = await r.json();
      const captions: Array<{ id: string; caption: string }> = data.captions ?? [];
      for (const c of captions) {
        map.set(c.id, c.caption || batch.find((b) => b.id === c.id)?.name || '(image)');
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') throw e;
      for (const img of batch) map.set(img.id, img.name);
    }
  }
  return map;
}

/** Strip base64 `dataUri` from every image attachment on messages older than
 *  the KEEP_RECENT tail. Recovers localStorage; preserves id/name/size/mime
 *  so SentAttachment can still render a "(image removed by compaction)" stub. */
function gcOldAttachments(): void {
  const s = useStore.getState();
  const msgs = s.activeMessages();
  if (msgs.length <= KEEP_RECENT) return;
  const cutoff = msgs.length - KEEP_RECENT;
  let dirty = false;
  const next: Message[] = msgs.map((m, i) => {
    if (i >= cutoff) return m;
    if (!m.attachments || m.attachments.length === 0) return m;
    let touched = false;
    const cleaned = m.attachments.map((a) => {
      if (a.kind === 'image' && a.dataUri) {
        touched = true;
        const { dataUri: _drop, content: _alsoDrop, ...rest } = a as Attachment & {
          dataUri?: string; content?: string;
        };
        return rest as Attachment;
      }
      return a;
    });
    if (!touched) return m;
    dirty = true;
    return { ...m, attachments: cleaned };
  });
  if (dirty) s.replaceActiveMessages(next);
}

/** Sentinel guarding maybeCompact() against re-entry. Same promise is
 *  awaited by concurrent _sendCurrent() calls so they don't race ahead. */
let _compactingPromise: Promise<void> | null = null;
let _compactingAbort: AbortController | null = null;

/** Caption old images, summarise the slice (extending a prior summary when
 *  present), GC base64 from non-tail messages, then commit. */
async function _runCompaction(): Promise<void> {
  const s = useStore.getState();
  const msgs = s.activeMessages();
  if (msgs.length <= KEEP_RECENT + 1) return;
  const cutoff = msgs.length - KEEP_RECENT;

  // Split the about-to-be-compacted region into: prior summaries (kept as
  // context for the new summary) + messages to actually compress.
  const head = msgs.slice(0, cutoff);
  const tail = msgs.slice(cutoff);
  const priorSummaries = head.filter((m) => m.kind === 'summary');
  const toCompact = head.filter((m) => m.kind !== 'summary');
  if (toCompact.length < 2) return;

  _compactingAbort = new AbortController();
  const signal = _compactingAbort.signal;

  try {
    // 1. Caption every image in the slice. Captions ride inside the summary
    //    text so post-compaction the model still knows what was shown.
    const captionMap = await captionImagesIn(toCompact, signal);
    if (signal.aborted) return;

    // 2. Build the payload — same materializer the model would see, but the
    //    caption map replaces every image_url with `[image: ...]`.
    const messagesPayload = toCompact.map((m) => _materializeForModel(m, captionMap));
    const captionLines: string[] = [];
    let nImg = 0;
    for (const m of toCompact) {
      for (const a of (m.attachments ?? [])) {
        if (a.kind === 'image') {
          nImg += 1;
          const cap = captionMap.get(a.id) ?? a.name;
          captionLines.push(`[${nImg}] ${cap}`);
        }
      }
    }

    // 3. POST /summarize. The prior summary (if any) is carried as context
    //    so the model writes ONE summary covering both layers.
    const priorText = priorSummaries.length > 0
      ? priorSummaries.map((m) => m.content).join('\n\n')
      : null;
    const r = await fetch(`${BACKEND_HTTP}/summarize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: messagesPayload,
        prior_summary: priorText,
        image_captions: captionLines,
        merge_only: false,
      }),
      signal,
    });
    if (!r.ok || signal.aborted) {
      if (!signal.aborted) console.error('compaction failed:', await r.text());
      return;
    }
    const data = await r.json();
    const summary = (data.summary ?? '').trim();
    if (!summary) return;

    // 4. Commit. With a prior_summary baked in, the new summary already
    //    covers everything older than the tail — so we drop the old
    //    summaries AND the compacted messages and replace with one bubble.
    const synthetic: Message = {
      id: crypto.randomUUID(),
      role: 'system',
      content: summary,
      ts: Date.now(),
      kind: 'summary',
      compactedCount: toCompact.length + priorSummaries.reduce(
        (acc, m) => acc + (m.compactedCount ?? 1), 0,
      ),
    };
    useStore.getState().replaceActiveMessages([synthetic, ...tail]);

    // 5. Free base64 from any image attachment that's no longer in the
    //    recent tail. Independent of whether we just compacted those
    //    messages — purely a storage GC pass.
    gcOldAttachments();
  } catch (e) {
    if ((e as Error).name !== 'AbortError') console.error(e);
  }
}

/** Entry point. Reads the exact context size, decides whether to compact,
 *  manages the shared promise + status indicator. Re-entrant: a concurrent
 *  caller awaits the same in-flight promise rather than firing again. */
async function maybeCompact(): Promise<void> {
  if (_compactingPromise) {
    await _compactingPromise;
    return;
  }
  const s = useStore.getState();
  const nCtx = s.settings.nCtx || 4096;
  const budget = Math.max(1024, Math.floor(nCtx * COMPACT_THRESHOLD)) - RESERVE_FOR_REPLY;
  // Exact tokens when the model is loaded; rough fallback otherwise so the
  // user with an unloaded model still gets sensible behavior (probably a
  // no-op since the send will fail with "model not loaded" anyway).
  const exact = await getPromptTokens();
  const tokens = exact ?? roughPromptTokens();
  if (tokens <= budget) {
    // Even when not over budget, opportunistically run the storage GC pass —
    // it's cheap and unloads stale base64 once messages exit the tail.
    gcOldAttachments();
    return;
  }

  // Avoid running the chain-merge AND the delta-compact in the same tick.
  // If we've got too many summaries, merge first then re-enter to compact
  // the actual messages.
  const msgs = s.activeMessages();
  const cutoff = msgs.length > KEEP_RECENT ? msgs.length - KEEP_RECENT : 0;
  const summaryCount = msgs.slice(0, cutoff).filter((m) => m.kind === 'summary').length;

  useStore.getState().setCompacting(true);
  _compactingPromise = (async () => {
    try {
      if (summaryCount >= MAX_SUMMARY_CHAIN) {
        await _mergeSummaryChain();
      }
      await _runCompaction();
    } finally {
      useStore.getState().setCompacting(false);
      _compactingPromise = null;
      _compactingAbort = null;
    }
  })();
  await _compactingPromise;
}

/** Collapse all existing summary bubbles in the active conversation into a
 *  single one. Cheap — only summarises summaries, never raw history. */
async function _mergeSummaryChain(): Promise<void> {
  const s = useStore.getState();
  const msgs = s.activeMessages();
  const summaries = msgs.filter((m) => m.kind === 'summary');
  if (summaries.length < 2) return;
  try {
    const r = await fetch(`${BACKEND_HTTP}/summarize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: summaries.map((m) => ({ role: 'system', content: m.content })),
        merge_only: true,
      }),
      signal: _compactingAbort?.signal,
    });
    if (!r.ok) return;
    const data = await r.json();
    const merged = (data.summary ?? '').trim();
    if (!merged) return;
    const synthetic: Message = {
      id: crypto.randomUUID(),
      role: 'system',
      content: merged,
      ts: Date.now(),
      kind: 'summary',
      compactedCount: summaries.reduce((acc, m) => acc + (m.compactedCount ?? 1), 0),
    };
    // Strip the old summaries; keep everything else (non-summary head +
    // tail) at original positions. Insert merged where the FIRST summary
    // used to be — preserves chronological ordering.
    const firstIdx = msgs.findIndex((m) => m.kind === 'summary');
    const nonSummaries = msgs.filter((m) => m.kind !== 'summary');
    useStore.getState().replaceActiveMessages([
      ...nonSummaries.slice(0, firstIdx),
      synthetic,
      ...nonSummaries.slice(firstIdx),
    ]);
  } catch (e) {
    if ((e as Error).name !== 'AbortError') console.error(e);
  }
}

async function _sendCurrent() {
  await maybeCompact();
  const s = useStore.getState();
  const messages = s.activeMessages().map((m) => _materializeForModel(m));
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

/** True if the currently loaded model recognises Qwen-style thinking
 *  toggles (`/think` and `/no_think`). Sourced from the backend, which
 *  probes the actual tokenizer at load time for a `<think>` token — so
 *  Qwen3 finetunes that stripped the special token (and Qwen-VL variants
 *  that don't think) correctly hide the picker, and any future thinking
 *  family lights up automatically without code changes. */
export function thinkingSupported(): boolean {
  const s = useStore.getState();
  return s.modelStatus.status === 'loaded' && !!s.modelStatus.supportsThinking;
}

/** Returns the suffix to append to a user message to bias the model's
 *  thinking depth. Currently always empty — we used to append ' /no_think'
 *  in Quick mode but combined with the backend's hard prefill of an empty
 *  `<think>\n\n</think>\n\n` block, the double-signal caused Qwen3
 *  fine-tunes to produce ultra-terse one-word replies. The prefill alone
 *  is the Qwen3-official mechanism (`enable_thinking=False` in their chat
 *  template) and is sufficient. Kept as a hook for future model families
 *  that might need an explicit marker. */
function thinkingMarker(): string {
  return '';
}

export async function sendChat(text: string, attachments: Attachment[] = []) {
  const s = useStore.getState();
  if (s.streaming) return;
  const hasAnything = text.trim().length > 0 || attachments.length > 0;
  if (!hasAnything) return;
  if (!s.activeId) s.newConversation();
  // The marker is appended to the message we SEND to the model, but kept
  // out of the message we DISPLAY in the chat history so the user doesn't
  // see "/no_think" cluttering their own message bubbles.
  const marker = thinkingMarker();
  s.appendMessage({
    id: crypto.randomUUID(),
    role: 'user',
    content: text,
    attachments: attachments.length > 0 ? attachments : undefined,
    ts: Date.now(),
    // Stash the marker as a hidden suffix that _materializeForModel will
    // append at send-time. Stored on the message so re-asks (regenerate /
    // edit) preserve the same mode that was active when first sent.
    sendSuffix: marker || undefined,
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

/** Exact token count for the next prompt, computed by the backend using the
 *  loaded model's chat template + tokenizer. Includes chat-template chrome,
 *  tool catalog (in Code tab / agent mode), and a family-aware image cost.
 *
 *  Returns `null` while the model is unloaded or the request is in flight for
 *  the first time, so callers can fall back to their rough estimate. */
export function useExactCtxUsed(liveText: string, liveAttachments: Attachment[]): number | null {
  const systemPrompt = useStore((s) => s.settings.systemPrompt);
  const tab = useStore((s) => s.tab);
  const workspace = useStore((s) => s.workspace);
  const modelLoaded = useStore((s) => s.modelStatus.status === 'loaded');
  const activeMessages = useStore((s) =>
    s.activeId ? s.conversations.find((c) => c.id === s.activeId)?.messages ?? [] : [],
  );

  // Coarse fingerprint so we don't re-fire on every keystroke during streaming.
  // Includes ids + size for attachments so swaps re-trigger but render churn
  // (caret position, etc.) doesn't.
  const attachKey = liveAttachments.map((a) => `${a.id}:${a.size}`).join('|');
  const historyKey = activeMessages
    .map((m) => `${m.id}:${m.content.length}:${(m.attachments?.length ?? 0)}`)
    .join('|');

  const [tokens, setTokens] = useState<number | null>(null);

  useEffect(() => {
    if (!modelLoaded) { setTokens(null); return; }
    let cancelled = false;
    const timer = setTimeout(async () => {
      // Materialize past messages, then synthesize a draft "user" message for
      // the live input so it counts against the budget pre-send. If both the
      // input and attachments are empty, skip the draft.
      const past = activeMessages.map((m) => _materializeForModel(m));
      const includesDraft = liveText.length > 0 || liveAttachments.length > 0;
      if (includesDraft) {
        const draft: Message = {
          id: '__draft__', role: 'user', content: liveText,
          attachments: liveAttachments, ts: Date.now(),
        };
        past.push(_materializeForModel(draft));
      }
      try {
        const r = await fetch(`${BACKEND_HTTP}/tokenize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: past,
            system_prompt: systemPrompt,
            agent_mode: tab === 'code',
            workspace,
          }),
        });
        if (!r.ok || cancelled) return;
        const data = await r.json();
        if (!cancelled) {
          setTokens(typeof data.tokens === 'number' ? data.tokens : null);
        }
      } catch {
        if (!cancelled) setTokens(null);
      }
    }, 250);
    return () => { cancelled = true; clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveText, attachKey, historyKey, systemPrompt, tab, workspace, modelLoaded]);

  return tokens;
}
