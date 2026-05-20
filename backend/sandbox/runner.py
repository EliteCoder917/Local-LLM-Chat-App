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
    """Environment for spawned tool processes.

    We inherit the FULL parent environment. The previous tight allow-list
    (PATH/SystemRoot/TEMP/…) stripped vars that PowerShell's .NET host needs
    to even start — it died with "Loading managed Windows PowerShell failed,
    error 8009001d", which broke run_shell / run_script / open_app entirely.
    This is a local single-user app and exec tools are permission-gated, so
    inheriting the environment (as any terminal would) is the right call. We
    only force UTF-8 / unbuffered IO so we capture output cleanly.
    """
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


async def _run(cmd, *, cwd: str, timeout: float, shell: bool) -> ExecResult:
    # An empty/missing cwd is a hard failure on Windows: create_subprocess_*
    # raises WinError 123 ("filename, directory name, or volume label syntax
    # is incorrect"). That's exactly what run_shell hit when no workspace was
    # open. Fall back to the user's home dir so commands still run.
    if not cwd or not os.path.isdir(cwd):
        cwd = os.path.expanduser("~")
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


async def launch_detached(target: str, args: Optional[list] = None) -> ExecResult:
    """Open an app, file, folder, or URL via the OS — fire-and-forget.

    Unlike run_subprocess, this is meant for things that KEEP RUNNING (GUI
    apps). We invoke the OS's universal opener, which spawns the target as an
    independent process and returns immediately, so:
      * we never block waiting for a GUI app to close, and
      * the wait/kill-on-timeout in `_run` only ever touches the launcher
        (which exits in milliseconds), never the launched app.

    Windows : Start-Process  — resolves app names (App Paths), .exe paths, file
              associations, folders, and URLs/protocols (e.g. steam://).
    macOS   : open / open -a
    Linux   : xdg-open
    """
    args = [str(a) for a in (args or [])]
    if os.name == "nt":
        t = target.replace("'", "''")
        arglist = ""
        if args:
            quoted = ", ".join("'" + a.replace("'", "''") + "'" for a in args)
            arglist = f" -ArgumentList {quoted}"
        # `-ErrorAction Stop` + try/catch makes a failed launch return a
        # NON-zero exit (Start-Process otherwise often succeeds-silently, so
        # open_app would falsely report "Launched"). If a bare name like
        # 'steam' can't be resolved, we retry once via Start-Process again with
        # the cmd-style `start` shell verb, which consults App Paths/protocols
        # more liberally.
        command = (
            f"try {{ Start-Process -FilePath '{t}'{arglist} -ErrorAction Stop }} "
            f"catch {{ "
            f"  try {{ & cmd /c start \"\" '{t}' }} "
            f"  catch {{ Write-Error $_; exit 1 }} "
            f"}}"
        )
        cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]
    elif sys.platform == "darwin":
        if "://" in target or os.path.exists(target):
            cmd = ["open", target, *args]
        else:
            cmd = ["open", "-a", target, *(["--args", *args] if args else [])]
    else:  # linux / other
        cmd = ["xdg-open", target]

    # cwd is irrelevant for launching; use the home dir so we don't pin the
    # opener to a (possibly missing) workspace path.
    home = os.path.expanduser("~")
    return await _run(cmd, cwd=home, timeout=15.0, shell=False)
