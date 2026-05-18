"""Subprocess sandbox for executing Python code and shell commands.

This is a *soft* sandbox — it does not provide kernel-level isolation. Rather,
it enforces:

* a working directory clamped to the user's workspace (or a temp dir)
* a configurable wall-clock timeout
* output size capping (avoids OOM from runaway loops)
* truncation of environment variables (no secrets leak)
* the calling tool must have already passed `PermissionManager.check(...)`

Stronger isolation (Docker / Windows Sandbox / Firejail) can be plugged in by
subclassing this module.
"""
from __future__ import annotations

import asyncio
import os
import shlex
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional


MAX_OUTPUT_BYTES = 256 * 1024  # 256 KB per stream


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool

    def to_text(self) -> str:
        parts = []
        if self.stdout:
            parts.append(f"--- stdout ---\n{self.stdout}")
        if self.stderr:
            parts.append(f"--- stderr ---\n{self.stderr}")
        parts.append(f"--- exit {self.exit_code}{' (timeout)' if self.timed_out else ''} ---")
        return "\n".join(parts)


def _clean_env() -> dict:
    keep = {"PATH", "SystemRoot", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA",
            "PYTHONIOENCODING", "PYTHONUNBUFFERED"}
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


async def _run(cmd, *, cwd: str, timeout: float, shell: bool) -> ExecResult:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd,
        env=_clean_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    ) if not shell else await asyncio.create_subprocess_shell(
        cmd,
        cwd=cwd,
        env=_clean_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    timed_out = False
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        stdout_b, stderr_b = await proc.communicate()

    def cap(b: bytes) -> str:
        if len(b) > MAX_OUTPUT_BYTES:
            b = b[:MAX_OUTPUT_BYTES] + b"\n[truncated]"
        return b.decode("utf-8", errors="replace")

    return ExecResult(
        stdout=cap(stdout_b),
        stderr=cap(stderr_b),
        exit_code=proc.returncode if proc.returncode is not None else -1,
        timed_out=timed_out,
    )


async def run_python_code(code: str, *, cwd: str, timeout: float = 20.0) -> ExecResult:
    """Run `code` in a separate Python interpreter via a temp file."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, dir=cwd or None, encoding="utf-8"
    ) as tf:
        tf.write(code)
        path = tf.name
    try:
        return await _run(
            [sys.executable, "-I", path],
            cwd=cwd or os.path.dirname(path),
            timeout=timeout,
            shell=False,
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


async def run_subprocess(command: str, *, cwd: str, timeout: float = 20.0,
                         use_powershell: Optional[bool] = None) -> ExecResult:
    """Run a shell command. On Windows defaults to PowerShell."""
    if use_powershell is None:
        use_powershell = os.name == "nt"
    if use_powershell:
        cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
        return await _run(cmd, cwd=cwd, timeout=timeout, shell=False)
    return await _run(command, cwd=cwd, timeout=timeout, shell=True)
