# Local AI IDE

A Windows-native desktop AI environment: chat with a local LLM, edit files in a
VS Code-style editor (Monaco), browse a workspace, run a real PowerShell
terminal, and let an agent execute tools (file I/O, code execution) — all
behind explicit permission toggles.

```
Electron  ⇄  React renderer (Monaco, xterm.js, react-markdown, zustand)
   ▲
   │ IPC (preload bridge)
   ▼
Electron main (Node)  ⇄  Python sidecar (FastAPI + WebSockets)
                              ├── LLM engine (Ollama / llama-cpp-python)
                              ├── Agent loop (plan → act → observe)
                              ├── Tool registry (files, exec, memory)
                              ├── Permission manager (gates each tool)
                              ├── Sandbox (subprocess + timeouts)
                              └── Memory (SQLite)
```

---

## 1. Folder layout

```
local-ai-ide/
├── package.json
├── tsconfig.json
├── electron-builder.yml
├── electron/                 # Electron main process (Node + TypeScript)
│   ├── main.ts
│   ├── preload.ts
│   ├── python-bridge.ts
│   ├── permissions.ts
│   ├── pty-manager.ts
│   └── store.ts
├── frontend/                 # React renderer (Vite + TS + Tailwind)
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── components/       # ChatPane, EditorPane, FileExplorer, ...
│       ├── hooks/            # useChat, useFS
│       ├── ipc/bridge.ts
│       ├── state/            # zustand store + types
│       └── styles/index.css
├── backend/                  # Python sidecar
│   ├── main.py               # FastAPI app
│   ├── ws.py                 # WebSocket router
│   ├── config.py             # shared runtime config
│   ├── llm/                  # engine + Ollama + llama-cpp adapters
│   ├── agent/                # prompts, parser, agent loop
│   ├── tools/                # files, exec, memory + registry
│   ├── memory/               # SQLite store
│   ├── sandbox/              # subprocess runner
│   └── permissions/          # gating + interactive prompts
├── resources/                # icon.ico, default-settings.json
└── scripts/                  # build-python.ps1, dev.ps1
```

---

## 2. Prerequisites

* **Windows 10 / 11**
* **Node.js ≥ 18** with npm
* **Python ≥ 3.10**
* One of:
  * **Ollama** (recommended — install from <https://ollama.com>) and `ollama pull llama3.1:8b`
  * `pip install llama-cpp-python` plus a local `.gguf` model

---

## 3. Install dependencies (one-time)

```powershell
# from project root
npm install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

> `node-pty` ships native prebuilds for Node 18/20. If install fails on your
> machine, the terminal pane shows a friendly "unavailable" message and the
> rest of the app still works.

---

## 4. Run in dev mode

```powershell
# easiest: one command starts Vite, Electron, and the Python backend
npm run dev
```

Or the manual three-window version:

```powershell
# terminal 1 — Python backend (hot reload)
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765 --reload

# terminal 2 — Vite dev server
npm run dev:renderer

# terminal 3 — Electron (loads http://localhost:5173)
npm run dev:electron
```

---

## 5. Build a Windows .exe installer

```powershell
# bundles renderer (Vite), main process (tsc), and freezes Python with PyInstaller
npm run build

# produces release\Local AI IDE Setup <ver>.exe
npm run dist
```

The build pipeline:

1. `vite build` → `dist/`
2. `tsc -p electron/tsconfig.json` → `dist-electron/`
3. `scripts/build-python.ps1` → `python-dist/backend.exe` (+ DLLs)
4. `electron-builder --win` → NSIS installer in `release/`

In production the Electron main spawns
`<resources>/python-dist/backend.exe` instead of a system Python.

---

## 6. Security model

### Two-layer permission gating

| Layer | What it does |
|------|--------------|
| **Workspace clamp** | All file-tool paths are resolved with `os.path.commonpath` against the open workspace. Any escape attempt (`..\..`, absolute paths outside) is rejected before touching disk. |
| **Permission toggles** | Each tool maps to a permission key (`file.read`, `file.write`, `file.delete`, `exec.python`, `exec.shell`, `exec.script`, `network`). Without the matching toggle, the backend emits a `permission-request` over the WebSocket and Electron pops a native confirm dialog. Choose Deny / Allow once / Allow & remember. |

Defaults: only `file.read` is enabled. Everything that can mutate state or
run code is opt-in. Permissions live in `%APPDATA%\Local AI IDE\settings.json`.

### Sandbox

* `backend/sandbox/runner.py` runs scripts and shell commands via
  `asyncio.create_subprocess_*` with:
  * **wall-clock timeout** (default 20 s)
  * **output cap** (256 KB / stream — prevents OOM from runaway loops)
  * **scrubbed env** — keeps only `PATH`, `SystemRoot`, `TEMP`, etc., so user
    secrets in env vars don't leak to the model
  * **clamped cwd** — always the open workspace
  * Python: `python -I` to isolate user site-packages
* The terminal pane uses `node-pty` which spawns a *separate* shell the user
  drives directly — it's not exposed to the model.

### Network egress

There is no network tool by default. The `network` permission key exists for
future extensions; nothing in this codebase makes outbound calls except the
LLM client (Ollama on localhost).

### Destructive-action guardrails

* `delete_file` requires `file.delete` (off by default).
* `run_shell` requires `exec.shell` (off by default).
* The Electron permission dialog includes the literal command/args so the
  user sees exactly what's about to execute.
* `--no-verify` git flags, `rm -rf /`, etc. would still be blocked by Windows
  permissions if attempted outside the workspace, but the model has no system
  privileges anyway — it runs as the user with workspace-clamped paths.

---

## 7. Architecture

### Layers

1. **Renderer (React + TypeScript)** — pure UI. Talks only to `window.api`
   exposed by the preload bridge.
2. **Preload (`electron/preload.ts`)** — `contextBridge` whitelist of typed
   IPC calls. No `nodeIntegration`, no `remote`. The renderer cannot import
   `fs` or `child_process`.
3. **Electron main (`electron/main.ts`)** — owns the lifecycle: spawns the
   Python sidecar, opens the BrowserWindow, manages PTYs, persists settings,
   shows permission dialogs.
4. **Python backend (FastAPI)** — does the actual work: streams LLM tokens,
   runs the agent loop, dispatches tools, stores memory.

### Communication

| Hop | Transport |
|-----|-----------|
| Renderer ⇄ Main | `ipcRenderer.invoke` / `webContents.send` (preload bridge) |
| Main ⇄ Python (control) | WebSocket `ws://127.0.0.1:8765/ws` |
| Renderer ⇄ Python (file viewer only) | HTTP `http://127.0.0.1:8765/fs/...` |
| Main ⇄ Python (lifecycle) | child_process stdout/stderr |
| PTY | `node-pty` in Electron main, streamed to xterm.js |

### Tool-calling protocol

The LLM emits a fenced JSON code block tagged `tool`:

````
```tool
{ "tool": "write_file", "args": { "path": "main.py", "content": "print('hi')" } }
```
````

The backend (`backend/agent/parser.py`) regex-extracts these blocks, runs the
tool through `dispatch_tool()` (permission-checked), and feeds the result
back as a follow-up user turn tagged `tool_result`. The model decides whether
to call another tool or produce the final answer.

### Agent loop

`backend/agent/loop.py` runs up to `max_iterations` cycles:

```
for step in range(max_iters):
    stream tokens → emit message-delta
    parse for ```tool block
    if found:
        emit tool-call
        dispatch_tool (permission check)
        emit tool-result
        append result as next user message
        continue
    else:
        finish
```

In **non-agent mode** the loop stops after the first assistant turn — tools
may still be called once each, but the model can't chain.

### Memory

SQLite at `%LOCALAPPDATA%\LocalAIIDE\memory.sqlite` with two tables:

* `kv(key, value, updated_at)` — `get_memory` / `set_memory` tools (and
  user-set preferences from the renderer).
* `messages(conversation_id, role, content, ts)` — long-term conversation
  history (the renderer doesn't yet wire to it; reserved for future
  multi-conversation support).

---

## 8. Adding a new tool

1. Add a function in `backend/tools/your_tool.py` (sync or async).
2. Register it in `backend/tools/registry.py`:

   ```python
   register(Tool(
       name="my_tool",
       description="What it does",
       schema={"arg1": {"type": "string"}},
       handler=my_tool_func,
       permission="file.read",   # or None
   ))
   ```

3. If it needs a new permission key, add it in `backend/config.py`
   (`Config.permissions`) and surface it in
   `frontend/src/components/PermissionToggles.tsx`.
4. Restart the backend — the system prompt regenerates with the new tool
   listed automatically.

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Python backend failed to start` | Make sure `python` is on PATH and `pip install -r backend/requirements.txt` succeeded. |
| `Ollama HTTP 404` | `ollama pull <model>` first, then re-send. |
| Terminal pane says "unavailable" | `node-pty` failed to build. Run `npm rebuild node-pty` or `npm install --build-from-source node-pty`. |
| `permission denied for tool 'X'` | Open Settings → Permissions and toggle the relevant key. |
| File tools say "outside the workspace" | Open a folder first via the explorer (top-right **Open…**). |

---

## 10. License

MIT — do whatever you want.
