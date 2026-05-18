"""Permission gating for tool calls.

The renderer maintains the source of truth for permissions and pushes the
current state into the backend via `settings-update` / `permissions-update`
messages. When a tool runs without the relevant blanket permission, the
backend emits a `permission-request` event on the WebSocket. The Electron main
process pops a native dialog and replies with `permission-response`.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable, Dict, Optional

from ..config import CONFIG


_PERM_KEY_FOR_TOOL: Dict[str, str] = {
    "read_file": "file.read",
    "list_dir": "file.read",
    "search_text": "file.read",
    "write_file": "file.write",
    "create_folder": "file.write",
    "move_file": "file.write",
    "rename_file": "file.write",
    "delete_file": "file.delete",
    "run_python": "exec.python",
    "run_shell": "exec.shell",
    "run_script": "exec.script",
    "get_memory": "memory",
    "set_memory": "memory",
    "list_memory": "memory",
    "delete_memory": "memory",
}


class PermissionManager:
    """Coordinates interactive permission requests over the WebSocket."""

    def __init__(self) -> None:
        # request_id -> future resolved when the renderer replies
        self._pending: Dict[str, asyncio.Future[bool]] = {}
        # injected by ws.py once the socket is connected
        self._send_request: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

    def attach_sender(self, send: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        self._send_request = send

    def perm_key_for(self, tool: str) -> Optional[str]:
        return _PERM_KEY_FOR_TOOL.get(tool)

    def is_granted(self, tool: str) -> bool:
        key = self.perm_key_for(tool)
        if key is None:
            return True  # tool not gated
        return bool(CONFIG.permissions.get(key, False))

    async def check(self, tool: str, description: str, args: Dict[str, Any]) -> bool:
        """Return True if the tool may proceed. Prompts the user if needed."""
        if self.is_granted(tool):
            return True
        if self._send_request is None:
            return False
        req_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending[req_id] = fut
        await self._send_request({
            "type": "permission-request",
            "id": req_id,
            "tool": tool,
            "description": description,
            "args": args,
        })
        try:
            return await asyncio.wait_for(fut, timeout=120)
        except asyncio.TimeoutError:
            return False
        finally:
            self._pending.pop(req_id, None)

    def resolve(self, req_id: str, granted: bool) -> None:
        fut = self._pending.get(req_id)
        if fut and not fut.done():
            fut.set_result(granted)


permission_manager = PermissionManager()
