import { BrowserWindow, dialog } from 'electron';
import { Store } from './store';

export type PermKey =
  | 'file.read'
  | 'file.write'
  | 'file.delete'
  | 'exec.code'      // unified: Python + shell + scripts (all = arbitrary code exec)
  | 'system.open'    // launch apps / open files & URLs
  | 'network'
  | 'memory';

export interface PermissionRequest {
  id: string;
  tool: string;
  description: string;
  args: Record<string, unknown>;
}

const DEFAULTS: Record<PermKey, boolean> = {
  'file.read': true,
  'file.write': false,
  'file.delete': false,
  'exec.code': false,
  'system.open': false,
  network: false,
  memory: false,
};

export class PermissionManager {
  constructor(private store: Store) {
    const cur = (store.get<Record<string, boolean>>('permissions') ?? {}) as Record<string, boolean>;
    store.set('permissions', { ...DEFAULTS, ...cur });
  }

  all(): Record<string, boolean> {
    return (this.store.get<Record<string, boolean>>('permissions') ?? {}) as Record<string, boolean>;
  }

  set(key: string, value: boolean) {
    const cur = this.all();
    cur[key] = value;
    this.store.set('permissions', cur);
  }

  has(key: PermKey): boolean {
    return !!this.all()[key];
  }

  async requestInteractive(win: BrowserWindow, req: PermissionRequest): Promise<boolean> {
    const r = await dialog.showMessageBox(win, {
      type: 'warning',
      title: 'Permission required',
      message: `Allow tool "${req.tool}" to run?`,
      detail: `${req.description}\n\nArgs:\n${JSON.stringify(req.args, null, 2)}`,
      buttons: ['Deny', 'Allow once', 'Allow & remember'],
      defaultId: 0,
      cancelId: 0,
    });
    if (r.response === 2) {
      const inferred = inferPermKey(req.tool);
      if (inferred) this.set(inferred, true);
    }
    return r.response > 0;
  }
}

export function inferPermKey(tool: string): PermKey | null {
  if (tool === 'read_file' || tool === 'list_dir' || tool === 'search_text') return 'file.read';
  if (tool === 'write_file' || tool === 'create_folder' || tool === 'move_file' || tool === 'rename_file') return 'file.write';
  if (tool === 'delete_file') return 'file.delete';
  if (tool === 'run_python' || tool === 'run_shell' || tool === 'run_script') return 'exec.code';
  if (tool === 'open_app') return 'system.open';
  if (tool === 'get_memory' || tool === 'set_memory' || tool === 'list_memory' || tool === 'delete_memory') return 'memory';
  return null;
}
