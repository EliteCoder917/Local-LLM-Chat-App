import { ChildProcessWithoutNullStreams, spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import WebSocket from 'ws';

export interface BridgeOpts {
  backendDir: string;
  pythonDist: string | null;
  projectRoot: string;
  port: number;
  /** Directory to write a rolling backend.log into. Used in packaged
   *  builds where stdout/stderr would otherwise disappear into the void. */
  logDir?: string;
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
    // PyInstaller's output binary has a `.exe` extension on Windows and no
    // extension on macOS/Linux. Without this branch the Mac build tries to
    // spawn `backend.exe` (which doesn't exist) and the bridge hangs forever
    // waiting on /health.
    const frozenBinary = process.platform === 'win32' ? 'backend.exe' : 'backend';
    const exe = usingFrozen
      ? path.join(this.opts.pythonDist!, frozenBinary)
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

    // In packaged builds, Electron's stdout/stderr is detached from any
    // console — meaning anything Python writes silently vanishes and we
    // have no way to diagnose load failures. Tee both streams into a log
    // file at logDir/backend.log so users (and we) can read what happened.
    let logStream: fs.WriteStream | null = null;
    if (this.opts.logDir) {
      try {
        fs.mkdirSync(this.opts.logDir, { recursive: true });
        const logPath = path.join(this.opts.logDir, 'backend.log');
        logStream = fs.createWriteStream(logPath, { flags: 'a' });
        const stamp = new Date().toISOString();
        logStream.write(`\n=== backend spawn ${stamp} ===\n`);
        logStream.write(`  exe:  ${exe}\n`);
        logStream.write(`  args: ${args.join(' ')}\n`);
        logStream.write(`  cwd:  ${cwd}\n\n`);
      } catch (e) {
        console.warn('[py] could not open backend.log:', e);
      }
    }

    this.proc.stdout.on('data', (d) => {
      const s = d.toString();
      process.stdout.write('[py] ' + s);
      logStream?.write(s);
    });
    this.proc.stderr.on('data', (d) => {
      const s = d.toString();
      process.stderr.write('[py-err] ' + s);
      logStream?.write(s);
    });
    this.proc.on('exit', (code) => {
      console.warn('[py] exited', code);
      logStream?.write(`\n=== backend exited with code ${code} ===\n`);
      logStream?.end();
    });
    this.proc.on('error', (err) => {
      console.warn('[py] spawn error:', err);
      logStream?.write(`\n=== spawn error: ${err.message} ===\n`);
    });

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
