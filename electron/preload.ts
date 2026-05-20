import { contextBridge, ipcRenderer } from 'electron';

// One-shot sync IPC at preload time. The renderer needs the backend's URL
// before any fetch() runs, and we can't make BACKEND_HTTP an async getter
// without refactoring ~25 call sites. sendSync here is a tolerable cost —
// it fires exactly once during preload.
const backendUrl: string = ipcRenderer.sendSync('app:backendUrlSync');

const api = {
  backendUrl,
  settings: {
    get: () => ipcRenderer.invoke('settings:get'),
    set: (patch: Record<string, unknown>) => ipcRenderer.invoke('settings:set', patch),
  },
  perms: {
    get: () => ipcRenderer.invoke('perms:get'),
    set: (key: string, value: boolean) => ipcRenderer.invoke('perms:set', key, value),
  },
  fs: {
    pickFolder: () => ipcRenderer.invoke('fs:pickFolder'),
    pickFile: (filters?: { name: string; extensions: string[] }[], multi?: boolean) =>
      ipcRenderer.invoke('fs:pickFile', filters, multi),
    readFile: (filePath: string) => ipcRenderer.invoke('fs:readFile', filePath),
  },
  llm: {
    send: (payload: unknown) => ipcRenderer.invoke('llm:send', payload),
    cancel: () => ipcRenderer.invoke('llm:cancel'),
    onEvent: (cb: (e: unknown) => void) => {
      const fn = (_: unknown, e: unknown) => cb(e);
      ipcRenderer.on('llm:event', fn);
      return () => ipcRenderer.removeListener('llm:event', fn);
    },
  },
  // Custom title bar (we removed the native frame). The renderer's title bar
  // calls these to drive minimize / maximize / close, and subscribes to
  // window:maximized events so its icon swaps without polling.
  window: {
    minimize:    () => ipcRenderer.send('window:minimize'),
    maximize:    () => ipcRenderer.send('window:maximize'),
    close:       () => ipcRenderer.send('window:close'),
    isMaximized: (): boolean => ipcRenderer.sendSync('window:isMaximizedSync'),
    onMaximizedChange: (cb: (maximized: boolean) => void) => {
      const fn = (_: unknown, v: boolean) => cb(v);
      ipcRenderer.on('window:maximized', fn);
      return () => ipcRenderer.removeListener('window:maximized', fn);
    },
  },
};

contextBridge.exposeInMainWorld('api', api);
export type Api = typeof api;
