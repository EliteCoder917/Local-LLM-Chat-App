"""WebSocket router.

Protocol (JSON messages over a single socket):

    Client → server:
        { "type": "rpc", "id": <int>, "method": "chat" | "cancel" | "model-load" | "model-unload", "params": ... }
        { "type": "settings-update", "settings": {...} }
        { "type": "permissions-update", "perms": {...} }
        { "type": "permission-response", "id": <str>, "granted": bool }

    Server → client:
        { "type": "rpc-response", "id": <int>, "result": ... }
        { "type": "message-start" | "message-delta" | "message-end" | ...,
          ... event payload ... }
        { "type": "model-status", "status": "idle"|"loading"|"loaded"|"error", ... }
        { "type": "permission-request", "id": <str>, "tool": ..., "args": ... }
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .agent import AgentRunner
from .config import CONFIG
from .llm import loader
from .models import downloader
from .permissions import permission_manager


router = APIRouter()


@router.websocket("/ws")
async def ws_endpoint(socket: WebSocket) -> None:
    await socket.accept()

    send_lock = asyncio.Lock()
    loop = asyncio.get_event_loop()

    async def send(obj: Dict[str, Any]) -> None:
        async with send_lock:
            try:
                await socket.send_text(json.dumps(obj))
            except Exception:  # noqa: BLE001 — client may have disconnected
                pass

    permission_manager.attach_sender(send)

    # Broadcast model status changes (loader runs in worker threads → schedule
    # the async send back on this socket's event loop).
    def on_status(snap: Dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(
            send({"type": "model-status", **snap}), loop,
        )

    loader.add_listener(on_status)

    def on_download(snap: Dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(
            send({"type": "download-progress", **snap}), loop,
        )

    downloader.add_listener(on_download)

    # Send initial snapshot so the renderer's status indicator hydrates.
    await send({"type": "model-status", **loader.snapshot()})

    runner = AgentRunner(emit=send)
    current_task: asyncio.Task | None = None

    async def handle_chat(params: Dict[str, Any]) -> str:
        nonlocal current_task
        if "settings" in params and isinstance(params["settings"], dict):
            CONFIG.update(params["settings"])
        if "workspace" in params and isinstance(params["workspace"], str):
            CONFIG.workspace = params["workspace"]

        history = params.get("messages", []) or []
        # Per-request agent_mode override (from the frontend tab). Lets Chat
        # tab requests skip the tool catalog while Code tab requests get it,
        # without bouncing the persisted CONFIG.agent_mode setting.
        agent_override = params.get("agentMode")
        if not isinstance(agent_override, bool):
            agent_override = None

        if current_task and not current_task.done():
            runner.cancel()
            try:
                await current_task
            except Exception:  # noqa: BLE001
                pass

        current_task = asyncio.create_task(
            runner.run(history, agent_mode=agent_override),
        )
        return "ok"

    try:
        while True:
            raw = await socket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = msg.get("type")

            if t == "rpc":
                method = msg.get("method")
                params = msg.get("params") or {}
                rid = msg.get("id")
                if method == "chat":
                    result: Any = await handle_chat(params)
                elif method == "cancel":
                    runner.cancel()
                    result = "cancelling"
                elif method == "model-load":
                    # Apply any settings sent inline before loading.
                    if isinstance(params.get("settings"), dict):
                        CONFIG.update(params["settings"])
                    result = await loader.load()
                elif method == "model-unload":
                    result = loader.unload()
                elif method == "model-status":
                    result = loader.snapshot()
                else:
                    result = f"unknown method: {method}"
                await send({"type": "rpc-response", "id": rid, "result": result})

            elif t == "settings-update":
                settings = msg.get("settings") or {}
                inner = settings.get("settings") if isinstance(settings, dict) else None
                payload = inner if isinstance(inner, dict) else settings
                CONFIG.update(payload)
                perms = settings.get("permissions") if isinstance(settings, dict) else None
                if isinstance(perms, dict):
                    CONFIG.update_permissions(perms)
                # Notify renderer that the loaded engine might now be stale
                # (different .gguf chosen, etc.) so the indicator updates.
                await send({"type": "model-status", **loader.snapshot()})

            elif t == "permissions-update":
                perms = msg.get("perms") or {}
                if isinstance(perms, dict):
                    CONFIG.update_permissions(perms)

            elif t == "permission-response":
                permission_manager.resolve(msg.get("id"), bool(msg.get("granted")))

    except WebSocketDisconnect:
        runner.cancel()
        if current_task:
            try:
                await asyncio.wait_for(current_task, timeout=1)
            except Exception:  # noqa: BLE001
                pass
    finally:
        loader.remove_listener(on_status)
        downloader.remove_listener(on_download)
