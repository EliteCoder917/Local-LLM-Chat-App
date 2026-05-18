"""Code-execution tools (gated by permissions, run via sandbox)."""
from __future__ import annotations

from ..config import CONFIG
from ..sandbox.runner import run_python_code, run_subprocess


async def run_python(code: str, timeout: float = 20.0) -> str:
    res = await run_python_code(code, cwd=CONFIG.workspace, timeout=timeout)
    return res.to_text()


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
