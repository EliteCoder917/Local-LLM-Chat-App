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
    const closeRe = new RegExp(`</${tagName}>`, 'i');
    const close = text.slice(openEnd).match(closeRe);
    if (!close || close.index === undefined) {
      // Unclosed — model is still emitting its thought stream.
      out.push({
        kind: 'thinking',
        content: text.slice(openEnd),
        streaming: true,
      });
      break;
    }
    const closeStart = openEnd + close.index;
    out.push({ kind: 'thinking', content: text.slice(openEnd, closeStart) });
    i = closeStart + close[0].length;
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
