/**
 * Slash command registry. Commands run client-side and are NOT sent to the
 * model — they short-circuit the chat flow.
 *
 * A handler receives the raw rest-of-input (everything after the command
 * name) and can do whatever it wants (mutate store, fire async calls, etc.).
 */
import { useStore } from '../state/store';
import { regenerateLast } from '../hooks/useChat';

export interface SlashCommand {
  name: string;
  desc: string;
  usage?: string;
  run: (args: string) => void | Promise<void>;
}

export const COMMANDS: SlashCommand[] = [
  {
    name: 'clear',
    desc: 'Clear the current conversation',
    run: () => {
      const s = useStore.getState();
      if (s.activeId) s.deleteConversation(s.activeId);
      s.newConversation();
    },
  },
  {
    name: 'regenerate',
    desc: 'Drop the last assistant turn and re-ask',
    run: () => regenerateLast(),
  },
  {
    name: 'system',
    desc: 'Set the system prompt for this conversation',
    usage: '/system <prompt>',
    run: async (args) => {
      const v = args.trim();
      if (v) await useStore.getState().setSettings({ systemPrompt: v });
    },
  },
  {
    name: 'temp',
    desc: 'Set sampling temperature (0–2)',
    usage: '/temp 0.7',
    run: async (args) => {
      const n = parseFloat(args.trim());
      if (!Number.isNaN(n)) {
        await useStore.getState().setSettings({ temperature: Math.max(0, Math.min(2, n)) });
      }
    },
  },
  {
    name: 'model',
    desc: 'Switch to a model from the library by id (filename)',
    usage: '/model llama-3.1-8b-instruct.Q4_K_M.gguf',
    run: async (args) => {
      const id = args.trim();
      if (id) await useStore.getState().selectLibraryModel(id);
    },
  },
  {
    name: 'help',
    desc: 'List available slash commands',
    run: () => {
      const s = useStore.getState();
      const content =
        '**Slash commands:**\n\n' +
        COMMANDS.map((c) => `- \`/${c.name}\` — ${c.desc}${c.usage ? `  \`${c.usage}\`` : ''}`).join('\n');
      s.appendMessage({
        id: crypto.randomUUID(),
        role: 'system',
        content,
        ts: Date.now(),
      });
    },
  },
];

/** Returns true if `text` was handled as a slash command (no chat send needed). */
export async function maybeRunSlash(text: string): Promise<boolean> {
  if (!text.startsWith('/')) return false;
  const space = text.indexOf(' ');
  const name = (space === -1 ? text.slice(1) : text.slice(1, space)).toLowerCase();
  const args = space === -1 ? '' : text.slice(space + 1);
  const cmd = COMMANDS.find((c) => c.name === name);
  if (!cmd) return false;
  await cmd.run(args);
  return true;
}

/** Returns commands matching a prefix (for the autocomplete popup). */
export function suggestSlash(prefix: string): SlashCommand[] {
  const p = prefix.toLowerCase();
  return COMMANDS.filter((c) => c.name.startsWith(p));
}
