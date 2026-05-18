import { contextBridge, ipcRenderer } from 'electron';

const api = {
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
};

contextBridge.exposeInMainWorld('api', api);
export type Api = typeof api;
