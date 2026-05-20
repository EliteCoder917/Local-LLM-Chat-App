"""Code-execution tools (gated by permissions, run via sandbox)."""
from __future__ import annotations

import os

from ..config import CONFIG
from ..sandbox.runner import launch_detached, run_python_code, run_subprocess


async def run_python(code: str, timeout: float = 20.0) -> str:
    res = await run_python_code(code, cwd=CONFIG.workspace, timeout=timeout)
    return res.to_text()


async def open_app(target: str, args: list | None = None) -> str:
    """Open an app, file, folder, or URL and leave it running (fire-and-forget).

    `target` may be an app name (e.g. "steam", "chrome"), an absolute path, a
    workspace-relative file/folder path, or a URL/protocol (https://…,
    steam://…). Returns once the launch is dispatched — it does NOT wait for
    the opened thing to close."""
    if not target or not str(target).strip():
        return "Error: no target to open."
    t = str(target).strip()
    # Resolve a workspace-relative path only when it actually exists there;
    # otherwise leave it untouched so app names and URLs pass through.
    if "://" not in t and not os.path.isabs(t) and CONFIG.workspace:
        cand = os.path.join(CONFIG.workspace, t)
        if os.path.exists(cand):
            t = cand
    res = await launch_detached(t, args)
    if res.exit_code == 0 and not res.timed_out:
        return f"Launched: {target}"
    # Surface the launcher's error (e.g. unknown app name) to the model.
    return f"Failed to open '{target}'.\n{res.to_text()}"


async def run_shell(command: str, timeout: float = 20.0) -> str:
    res = await run_subprocess(command, cwd=CONFIG.workspace, timeout=timeout)
    return res.to_text()


async def run_script(path: str, timeout: float = 30.0) -> str:
    """Run a script file. Dispatches by extension."""
    import os
    from .files import _resolve  # noqa: WPS437 — internal helper reuse
    abs_path = _resolve(path)
    ext = os.path.splitext(abs_path)[1].lower()
    if ext == ".py":
        with open(abs_path, "r", encoding="utf-8") as f:
            code = f.read()
        return await run_python(code, timeout=timeout)
    if ext in (".ps1", ".bat", ".cmd"):
        return await run_shell(f'& "{abs_path}"', timeout=timeout)
    if ext == ".sh":
        return await run_shell(f'bash "{abs_path}"', timeout=timeout)
    return f"Unsupported script type: {ext}"
