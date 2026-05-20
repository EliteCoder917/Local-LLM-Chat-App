/**
 * Streaming-safe parser for <thinking> / <think> / <reasoning> / <scratchpad>
 * sections inside assistant message text.
 *
 * Walks the string once, alternating between visible-text and thinking segments.
 * If a tag opens but never closes (because the stream hasn't reached the close
 * yet), the trailing thinking segment is marked `streaming: true` so the UI
 * can render a live indicator.
 */

export type Segment =
  | { kind: 'text'; content: string }
  | { kind: 'thinking'; content: string; streaming?: boolean };

const OPEN_RE = /<(thinking|think|reasoning|scratchpad)>/i;

export function parseThinking(text: string): Segment[] {
  const out: Segment[] = [];
  let i = 0;
  while (i < text.length) {
    const tail = text.slice(i);
    const open = tail.match(OPEN_RE);
    if (!open || open.index === undefined) {
      const rest = text.slice(i);
      if (rest) out.push({ kind: 'text', content: rest });
      break;
    }
    const tagStart = i + open.index;
    if (tagStart > i) {
      out.push({ kind: 'text', content: text.slice(i, tagStart) });
    }
    const tagName = open[1];
    const openEnd = tagStart + open[0].length;

    // Bound this block at the next OPEN tag so genuinely separate think
    // blocks still parse independently.
    const after = text.slice(openEnd);
    const nextOpen = after.match(OPEN_RE);
    const blockEnd = nextOpen && nextOpen.index !== undefined
      ? openEnd + nextOpen.index
      : text.length;
    const block = text.slice(openEnd, blockEnd);

    // Match the LAST close in the block, not the first. Quick mode's
    // budget can force an early `</think>` mid-reasoning, but this fine-tune
    // ignores it and keeps thinking until it emits its OWN `</think>`. Hiding
    // only to the first close leaks that continued reasoning as visible text;
    // matching the last close keeps the whole thought stream collapsed.
    const closeRe = new RegExp(`</${tagName}>`, 'ig');
    let lastIdx = -1;
    let lastLen = 0;
    let m: RegExpExecArray | null;
    while ((m = closeRe.exec(block)) !== null) {
      lastIdx = m.index;
      lastLen = m[0].length;
    }
    if (lastIdx === -1) {
      // No close yet — model is still emitting its thought stream.
      out.push({ kind: 'thinking', content: block, streaming: true });
      i = blockEnd;
      continue;
    }
    // Drop any interior (budget-forced) `</think>` from the displayed thought
    // so the collapsed block doesn't show a stray tag mid-text.
    const thought = block.slice(0, lastIdx).replace(new RegExp(`</${tagName}>`, 'ig'), '');
    out.push({ kind: 'thinking', content: thought });
    i = openEnd + lastIdx + lastLen;
  }
  return out;
}

/** Returns the text with all thinking sections removed. Used when sending
 *  conversation history back to the model so its own reasoning doesn't
 *  pollute the next turn's context window. */
export function stripThinking(text: string): string {
  return parseThinking(text)
    .filter((s) => s.kind === 'text')
    .map((s) => s.content)
    .join('')
    .trim();
}

/** Rough token count for a string. Used for "Thought for N tokens".
 *  4 chars/token is the standard rule-of-thumb for English+code. */
export function roughTokens(text: string): number {
  return Math.max(1, Math.round(text.length / 4));
}
