// Renderer-side typed reference to the IPC bridge exposed by preload.

export interface RendererApi {
  // Backend URL chosen by main.ts (typically http://127.0.0.1:8765, but
  // may shift to an OS-picked port if 8765 was held by a zombie process).
  // Resolved at preload time via sync IPC so it's safe to read synchronously.
  backendUrl: string;
  settings: {
    get: () => Promise<Record<string, unknown>>;
    set: (patch: Record<string, unknown>) => Promise<Record<string, unknown>>;
  };
  perms: {
    get: () => Promise<Record<string, boolean>>;
    set: (key: string, value: boolean) => Promise<Record<string, boolean>>;
  };
  fs: {
    pickFolder: () => Promise<string | null>;
    pickFile: (
      filters?: { name: string; extensions: string[] }[],
      multi?: boolean,
    ) => Promise<string | string[] | null>;
    readFile: (filePath: string) => Promise<{ base64: string; size: number }>;
  };
  llm: {
    send: (payload: unknown) => Promise<unknown>;
    cancel: () => Promise<unknown>;
    onEvent: (cb: (e: unknown) => void) => () => void;
  };
  window: {
    minimize: () => void;
    maximize: () => void;
    close: () => void;
    isMaximized: () => boolean;
    onMaximizedChange: (cb: (maximized: boolean) => void) => () => void;
  };
}

declare global {
  interface Window {
    api: RendererApi;
  }
}

export const api: RendererApi = window.api;
// Fallback only matters in environments where the preload didn't run
// (storybook, tests). In the real app `api.backendUrl` is always set.
export const BACKEND_HTTP: string = api?.backendUrl ?? 'http://127.0.0.1:8765';
