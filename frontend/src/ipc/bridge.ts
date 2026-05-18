// Renderer-side typed reference to the IPC bridge exposed by preload.

export interface RendererApi {
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
  };
  llm: {
    send: (payload: unknown) => Promise<unknown>;
    cancel: () => Promise<unknown>;
    onEvent: (cb: (e: unknown) => void) => () => void;
  };
}

declare global {
  interface Window {
    api: RendererApi;
  }
}

export const api: RendererApi = window.api;
export const BACKEND_HTTP = 'http://127.0.0.1:8765';
