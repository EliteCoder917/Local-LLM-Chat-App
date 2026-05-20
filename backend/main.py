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
        # Refined per-layer cost estimate. Output (lm_head) and embedding
        # tensors are typically ~1.5-3 GB on a 30B+ MoE model; subtracting
        # one layer's worth as "head + embed overhead" leaves a more honest
        # gb_per_layer. Caller will use this to convert the gpu-offload GB
        # slider into a layer count.
        head_and_embed_gb = max(0.0, size_gb * 0.04)  # rough — 4% of model
        per_layer_gb = max(0.01, (size_gb - head_and_embed_gb) / max(1, block_count))
        model_meta = {
            "sizeGb": size_gb,
            "arch": meta.get("arch"),
            "blockCount": block_count,
            "trainedContext": meta.get("context_length"),
            "embeddingLength": meta.get("embedding_length"),
            "headCount": meta.get("head_count"),
            "headCountKv": meta.get("head_count_kv"),
            "ropeDim": meta.get("rope_dimension_count"),
            "headAndEmbedGb": head_and_embed_gb,
            "gbPerLayer": per_layer_gb,
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


class TokenizePayload(BaseModel):
    messages: list[dict]            # [{role, content}, ...] — content may be str or list
    system_prompt: str | None = None
    agent_mode: bool = False
    workspace: str = ""


@app.post("/tokenize")
def tokenize(payload: TokenizePayload) -> dict:
    """Exact token count for the prompt llama.cpp would build for this conversation.

    Builds the same system prompt the runner would (with tool catalog when
    agent_mode), renders the model's chat template, tokenizes the result, and
    adds a family-aware per-image cost for any image blocks. Returns
    `{tokens: null}` when no model is loaded so the frontend can fall back to
    its rough chars/4 estimate.
    """
    if not loader.is_loaded():
        return {"tokens": None, "reason": "no_model_loaded"}
    engine = loader.get()
    from .llm.llama_cpp_engine import LlamaCppEngine
    if not isinstance(engine, LlamaCppEngine):
        return {"tokens": None, "reason": "not_llama_cpp"}

    from .agent.prompts import build_system_prompt
    from .llm import ChatMessage
    base_prompt = (
        payload.system_prompt if payload.system_prompt is not None else CONFIG.system_prompt
    )
    system = build_system_prompt(
        base_prompt, list_tools(),
        payload.workspace or CONFIG.workspace,
        payload.agent_mode,
    )

    chat_messages: list[ChatMessage] = []
    if system:
        chat_messages.append(ChatMessage(role="system", content=system))
    for m in payload.messages:
        chat_messages.append(ChatMessage(
            role=m.get("role") or "user",
            content=m.get("content") if m.get("content") is not None else "",
        ))

    return engine.count_prompt_tokens(chat_messages)


class CaptionPayload(BaseModel):
    # [{"id": "abc", "dataUri": "data:image/png;base64,..."}, ...]
    images: list[dict]


@app.post("/caption")
async def caption(payload: CaptionPayload) -> dict:
    """Caption a batch of images via the loaded vision model.

    Used by the auto-compaction pipeline to convert image attachments on
    older messages into one-line text captions BEFORE feeding the slice to
    /summarize. Without this, base64 dataUris get stringified into the
    summarizer's prompt as raw garbage. Returns `{captions: [{id, caption}]}`.
    """
    if not loader.is_loaded():
        raise HTTPException(409, "Model is not loaded.")
    engine = loader.get()
    from .llm.llama_cpp_engine import LlamaCppEngine
    if not isinstance(engine, LlamaCppEngine) or not engine.vision_active:
        # No vision handler — caller falls back to filename placeholders.
        raise HTTPException(409, "No vision model loaded.")
    if not payload.images:
        return {"captions": []}

    # Build a single multimodal turn with all images numbered. One forward
    # pass beats N round-trips when the slice has several images, though we
    # cap the batch at the caller side (~4 images) so n_ctx stays sane.
    content_blocks: list[dict] = [{
        "type": "text",
        "text": (
            f"Caption each of the {len(payload.images)} image(s) below in one "
            "short sentence describing what's depicted. Format your response "
            "as a numbered list, one caption per line, no preamble:\n"
            "1. <caption for image 1>\n"
            "2. <caption for image 2>\n"
            "...\n"
            "Each caption should be a single sentence focusing on the main "
            "content (no chatter, no hedging)."
        ),
    }]
    for img in payload.images:
        content_blocks.append({
            "type": "image_url",
            "image_url": {"url": img.get("dataUri", "")},
        })

    from .llm import ChatMessage
    out_parts: list[str] = []
    async for delta in engine.stream([ChatMessage(role="user", content=content_blocks)]):
        out_parts.append(delta)
    raw = "".join(out_parts).strip()

    # Parse "1. text\n2. text" — tolerate missing numbers / extra prose by
    # falling back to splitting on newlines. Order matches input order.
    captions: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading "1." / "1)" / "- " / "* " markers.
        for prefix in (".", ")"):
            if len(line) > 2 and line[0].isdigit() and line[1] == prefix:
                line = line[2:].strip()
                break
        if line.startswith(("-", "*")):
            line = line[1:].strip()
        if line:
            captions.append(line)

    # Pad/truncate to match input length.
    while len(captions) < len(payload.images):
        captions.append("(image)")
    captions = captions[: len(payload.images)]

    return {
        "captions": [
            {"id": img.get("id"), "caption": cap}
            for img, cap in zip(payload.images, captions)
        ],
    }


class SummarizePayload(BaseModel):
    # The slice of messages to compress. Content may be string OR list of
    # {type, text} blocks (image blocks must be substituted up-front by the
    # caller via /caption — this endpoint never sees raw base64).
    messages: list[dict]
    # When set, the model is told to extend an existing summary instead of
    # writing one from scratch; lets the chain stay short and avoids paying
    # to re-summarize older content on every compaction.
    prior_summary: str | None = None
    # Pre-computed [image-N: caption] strings to inject as a bullet list so
    # the model knows what images were attached to the slice.
    image_captions: list[str] = []
    # When True, collapse `messages` (which is a list of prior summaries
    # concatenated as plain text) into a single merged summary. Used to cap
    # the summary chain length.
    merge_only: bool = False


@app.post("/summarize")
async def summarize(payload: SummarizePayload) -> dict:
    """Compress a slice of conversation history into a single summary.

    Three modes:
        merge_only=True             — collapse multiple prior summaries into one
        prior_summary provided      — extend the existing summary with new content
        otherwise                   — fresh summary of the slice (legacy behavior)

    The summary replaces the compacted slice in the renderer's store.
    """
    if not loader.is_loaded():
        raise HTTPException(409, "Model is not loaded. Load a model first.")
    if not payload.messages and not payload.prior_summary:
        return {"summary": ""}

    convo = "\n\n".join(
        f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}"
        for m in payload.messages
    )

    captions_block = ""
    if payload.image_captions:
        captions_block = (
            "Images referenced in this slice:\n"
            + "\n".join(f"- {c}" for c in payload.image_captions)
            + "\n\n"
        )

    if payload.merge_only:
        meta = (
            "Merge the following summary fragments into one coherent paragraph, "
            "deduplicating overlapping facts. Preserve all file paths, decisions, "
            "code changes, and open questions. Output ONLY the merged summary, "
            "no preamble.\n\n"
            f"---\n{convo}\n---"
        )
    elif payload.prior_summary:
        meta = (
            "You are extending an ongoing conversation summary. Below is the "
            "existing summary followed by NEW messages that need to be folded "
            "into it. Output a SINGLE updated summary paragraph that integrates "
            "both — keep the existing facts and add what the new messages "
            "contributed. Preserve file paths, decisions, code changes, open "
            "questions, user preferences. Drop chitchat. Output ONLY the "
            "summary text, no preamble.\n\n"
            f"{captions_block}"
            f"EXISTING SUMMARY:\n{payload.prior_summary}\n\n"
            f"NEW MESSAGES:\n---\n{convo}\n---"
        )
    else:
        meta = (
            "Summarize the following conversation in a concise paragraph. "
            "Preserve: any decisions made, file paths mentioned, code changes, "
            "open questions, and user preferences. Drop chitchat. Output ONLY "
            "the summary text, no preamble.\n\n"
            f"{captions_block}"
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
