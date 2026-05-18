"""FastAPI entrypoint.

HTTP routes:
    GET  /health                 — liveness probe (Electron waits on this)
    GET  /fs/list?path=...       — list a directory (renderer file explorer)
    GET  /fs/read?path=...       — read a text file
    POST /fs/write {path,content} — write a text file
    GET  /tools                  — JSON list of registered tools

WebSocket:
    /ws                          — streaming chat + tool events (see ws.py)
"""
from __future__ import annotations

import os
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from .config import CONFIG
from .llm import loader
from .models import library, downloader, huggingface, parse_hf_input
from .tools import list_tools
from .ws import router as ws_router


app = FastAPI(title="Local AI Studio Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # local-only server; relaxed for renderer
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ws_router)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/system/info")
def system_info(model_path: str | None = None) -> dict:
    """Returns RAM + GPU snapshot, plus model metadata if path provided."""
    from .llm.gpu import detect_nvidia_gpus
    from .llm.gguf_meta import read_gguf_meta
    from .llm.llama_cpp_engine import _ram_gb
    import os as _os

    avail_gb, total_gb = _ram_gb()
    gpus = detect_nvidia_gpus()
    model_meta: dict = {}
    if model_path and _os.path.isfile(model_path):
        size_gb = _os.path.getsize(model_path) / (1024 ** 3)
        meta = read_gguf_meta(model_path)
        block_count = int(meta.get("block_count") or 32)
        model_meta = {
            "sizeGb": size_gb,
            "arch": meta.get("arch"),
            "blockCount": block_count,
            "trainedContext": meta.get("context_length"),
            "gbPerLayer": size_gb / (block_count + 1),
        }
    return {
        "ramAvailableGb": avail_gb,
        "ramTotalGb": total_gb,
        "gpus": gpus,
        "model": model_meta,
    }


# ─── Model lifecycle ─────────────────────────────────────────────────
@app.get("/model/status")
def model_status() -> dict:
    return loader.snapshot()


@app.post("/model/load")
async def model_load() -> dict:
    return await loader.load()


@app.post("/model/unload")
def model_unload() -> dict:
    return loader.unload()


# ─── Persistent memory (the model's set_memory/get_memory store) ────
@app.get("/memory")
def memory_list() -> dict:
    from .memory import memory_store
    return {"entries": memory_store.list_all()}


@app.delete("/memory")
def memory_clear() -> dict:
    from .memory import memory_store
    n = memory_store.clear()
    return {"ok": True, "removed": n}


@app.delete("/memory/{key}")
def memory_delete(key: str) -> dict:
    from .memory import memory_store
    memory_store.delete(key)
    return {"ok": True}


class SummarizePayload(BaseModel):
    messages: list[dict]    # [{role, content}, ...]


@app.post("/summarize")
async def summarize(payload: SummarizePayload) -> dict:
    """Compress a slice of conversation history into a single summary.

    Called by the frontend when the running conversation gets close to n_ctx.
    The summary replaces the original slice in the renderer's store, so future
    turns stay within budget without losing the gist of earlier discussion.
    """
    if not loader.is_loaded():
        raise HTTPException(409, "Model is not loaded. Load a model first.")
    if not payload.messages:
        return {"summary": ""}

    convo = "\n\n".join(
        f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}"
        for m in payload.messages
    )
    meta = (
        "Summarize the following conversation in a concise paragraph. "
        "Preserve: any decisions made, file paths mentioned, code changes, "
        "open questions, and user preferences. Drop chitchat. Output ONLY "
        "the summary text, no preamble.\n\n"
        f"---\n{convo}\n---"
    )

    from .llm import ChatMessage
    engine = loader.get()
    out_parts: list[str] = []
    async for delta in engine.stream([ChatMessage(role="user", content=meta)]):
        out_parts.append(delta)
    return {"summary": "".join(out_parts).strip()}


# ─── Model library ──────────────────────────────────────────────────
@app.get("/library")
def library_list() -> dict:
    return {
        "root": library.root,
        "models": [m.to_json() for m in library.list()],
    }


class SelectModelPayload(BaseModel):
    id: str


@app.post("/library/select")
async def library_select(payload: SelectModelPayload) -> dict:
    """Set the active model to one in the library and reload."""
    entry = library.get(payload.id)
    if entry is None:
        raise HTTPException(404, f"Unknown model id: {payload.id}")
    CONFIG.model_path = entry.path
    # Pair the multimodal projector if the library found one — empties the
    # field when there's no mmproj so we don't carry a stale one over.
    CONFIG.mmproj_path = entry.mmproj_path or ""
    # Eject the current engine so the loader rebuilds against the new pair
    loader.unload()
    return await loader.load()


@app.delete("/library/{model_id}")
def library_delete(model_id: str) -> dict:
    # If the deleted model is the currently loaded one, eject first
    if loader.is_loaded() and CONFIG.model_path:
        if os.path.basename(CONFIG.model_path).lower() == model_id.lower():
            loader.unload()
            CONFIG.model_path = ""
    ok = library.delete(model_id)
    if not ok:
        raise HTTPException(404, f"Unknown model id: {model_id}")
    return {"ok": True}


# ─── HuggingFace downloader ─────────────────────────────────────────
class DownloadPayload(BaseModel):
    input: str   # URL or owner/repo/file.gguf


@app.post("/library/download")
async def library_download(payload: DownloadPayload) -> dict:
    try:
        job = await downloader.start(payload.input)
    except FileExistsError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return job.snapshot()


@app.get("/library/downloads")
def library_downloads() -> list:
    return downloader.list()


@app.post("/library/downloads/{job_id}/cancel")
def library_download_cancel(job_id: str) -> dict:
    ok = downloader.cancel(job_id)
    return {"ok": ok}


@app.post("/library/parse")
def library_parse(payload: DownloadPayload) -> dict:
    """Dry-run: parse user input, return what would be downloaded."""
    try:
        repo, filename, url = parse_hf_input(payload.input)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"repo": repo, "filename": filename, "url": url}


# ─── HuggingFace search ─────────────────────────────────────────────
@app.get("/hf/search")
async def hf_search(q: str, limit: int = 25) -> dict:
    q = q.strip()
    if len(q) < 2:
        return {"results": []}
    try:
        repos = await huggingface.search_repos(q, limit=min(50, max(5, limit)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"HuggingFace search failed: {e}")
    return {"results": [r.to_json() for r in repos]}


@app.get("/hf/files")
async def hf_files(repo: str) -> dict:
    try:
        files = await huggingface.list_gguf_files(repo)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"HuggingFace file listing failed: {e}")
    return {"files": [f.to_json() for f in files]}


@app.get("/tools")
def tools() -> list:
    out = []
    for t in list_tools().values():
        out.append({
            "name": t.name,
            "description": t.description,
            "schema": t.schema,
            "permission": t.permission,
        })
    return out


# ─── FS endpoints used by the renderer's file explorer ───────────────
@app.get("/fs/walk")
def fs_walk(path: str, limit: int = 2000) -> dict:
    """Recursively list files under `path`, returning relative POSIX paths.
    Used by the renderer's @file autocomplete. Skips common noise dirs.

    Returns an empty list instead of 404 when the path is missing, so a
    stale workspace setting doesn't spam the dev console with errors —
    autocomplete just shows no matches until the user picks a valid folder.
    """
    if not os.path.isdir(path):
        return {"paths": [], "truncated": False}
    SKIP = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist",
            "build", ".next", "release", ".build", ".vite", "python-dist"}
    out: list[str] = []
    base = os.path.abspath(path)
    try:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(".")]
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, base).replace("\\", "/")
                out.append(rel)
                if len(out) >= limit:
                    return {"paths": out, "truncated": True}
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return {"paths": out, "truncated": False}


@app.get("/fs/list")
def fs_list(path: str) -> List[dict]:
    if not os.path.isdir(path):
        raise HTTPException(404, f"Not a directory: {path}")
    entries = []
    try:
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            entries.append({
                "name": name,
                "path": full,
                "isDir": os.path.isdir(full),
            })
    except PermissionError as e:
        raise HTTPException(403, str(e))
    return entries


@app.get("/fs/read", response_class=PlainTextResponse)
def fs_read(path: str) -> str:
    if not os.path.isfile(path):
        raise HTTPException(404, f"Not a file: {path}")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        raise HTTPException(500, str(e))


class WritePayload(BaseModel):
    path: str
    content: str


@app.post("/fs/write")
def fs_write(payload: WritePayload) -> JSONResponse:
    # Direct write from the editor — permission lives in the UI itself.
    try:
        os.makedirs(os.path.dirname(payload.path) or ".", exist_ok=True)
        with open(payload.path, "w", encoding="utf-8", newline="\n") as f:
            f.write(payload.content)
    except OSError as e:
        raise HTTPException(500, str(e))
    return JSONResponse({"ok": True, "bytes": len(payload.content)})


@app.get("/")
def root() -> dict:
    return {
        "service": "Local AI Studio Backend",
        "version": app.version,
        "engine": CONFIG.engine,
        "model": CONFIG.model,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8765, reload=False)
