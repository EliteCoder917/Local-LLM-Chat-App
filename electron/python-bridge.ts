import { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import path from 'node:path';
import WebSocket from 'ws';

export interface BridgeOpts {
  backendDir: string;
  pythonDist: string | null;
  projectRoot: string;
  port: number;
}

export class PythonBridge extends EventEmitter {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private ws: WebSocket | null = null;
  private opts: BridgeOpts;
  private nextId = 1;
  private pending = new Map<number, (v: unknown) => void>();

  constructor(opts: BridgeOpts) {
    super();
    this.opts = opts;
  }

  async start() {
    const usingFrozen = !!this.opts.pythonDist;
    const exe = usingFrozen
      ? path.join(this.opts.pythonDist!, 'backend.exe')
      : (process.platform === 'win32' ? 'python' : 'python3');
    // The frozen entry parses --port from argv (see backend_entry.py); the
    // dev path passes it to uvicorn directly. Both honor the dynamically-
    // picked port so 8765 being held by a zombie doesn't break startup.
    const args = usingFrozen
      ? ['--port', String(this.opts.port)]
      : ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', String(this.opts.port)];
    const cwd = usingFrozen ? this.opts.pythonDist! : this.opts.projectRoot;

    this.proc = spawn(exe, args, {
      cwd,
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
      shell: false,
    });
    this.proc.stdout.on('data', (d) => process.stdout.write('[py] ' + d.toString()));
    this.proc.stderr.on('data', (d) => process.stderr.write('[py-err] ' + d.toString()));
    this.proc.on('exit', (code) => console.warn('[py] exited', code));

    await this.waitForBackend();
    await this.connectWs();
  }

  private async waitForBackend() {
    const url = `http://127.0.0.1:${this.opts.port}/health`;
    for (let i = 0; i < 120; i++) {
      try {
        const r = await fetch(url);
        if (r.ok) return;
      } catch {
        /* retry */
      }
      await new Promise((r) => setTimeout(r, 500));
    }
    throw new Error('Python backend failed to start');
  }

  private connectWs(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`ws://127.0.0.1:${this.opts.port}/ws`);
      this.ws.on('open', () => resolve());
      this.ws.on('error', (e) => {
        console.error('[ws] error', e);
        reject(e);
      });
      this.ws.on('message', (raw) => {
        let msg: { type: string; id?: number | string; result?: unknown };
        try {
          msg = JSON.parse(raw.toString());
        } catch {
          return;
        }
        if (msg.type === 'rpc-response' && typeof msg.id === 'number' && this.pending.has(msg.id)) {
          this.pending.get(msg.id)!(msg.result);
          this.pending.delete(msg.id);
        } else if (msg.type === 'permission-request') {
          this.emit('permission-request', msg);
        } else {
          this.emit('event', msg);
        }
      });
      this.ws.on('close', () => console.warn('[ws] closed'));
    });
  }

  private rpc(method: string, params: unknown): Promise<unknown> {
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.pending.set(id, resolve);
      this.ws!.send(JSON.stringify({ type: 'rpc', id, method, params }));
    });
  }

  sendChat(payload: unknown) {
    return this.rpc('chat', payload);
  }
  cancel() {
    return this.rpc('cancel', {});
  }
  notifySettings(settings: unknown) {
    this.ws?.send(JSON.stringify({ type: 'settings-update', settings }));
  }
  notifyPermissions(perms: unknown) {
    this.ws?.send(JSON.stringify({ type: 'permissions-update', perms }));
  }
  sendPermissionResponse(reqId: string, granted: boolean) {
    this.ws?.send(JSON.stringify({ type: 'permission-response', id: reqId, granted }));
  }

  async stop() {
    try { this.ws?.close(); } catch { /* ignore */ }
    try { this.proc?.kill(); } catch { /* ignore */ }
  }
}
