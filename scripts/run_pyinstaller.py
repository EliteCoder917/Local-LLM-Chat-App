"""PyInstaller launcher that's immune to a hung Windows WMI service.

PyInstaller's `compat.py` calls `platform.win32_ver()` at import time. On
Python 3.12 that issues a WMI query (`Win32_OperatingSystem`). When the
machine's `winmgmt` service is wedged, that query blocks forever — so
`import PyInstaller` hangs before it prints anything or creates a workdir,
which looks exactly like a stuck build.

We pre-empt it: force `platform._wmi_query` to raise so `win32_ver()` falls
back to its registry-based path (fast, no WMI). Then hand off to PyInstaller's
normal entry point with the original argv untouched.
"""
import platform


def _no_wmi(*_args, **_kwargs):
    raise OSError("WMI disabled by run_pyinstaller.py to avoid hung-winmgmt stalls")


# Patch both the private query helper (3.12+) — guard with getattr so this
# stays a no-op on Python versions that don't have it.
if hasattr(platform, "_wmi_query"):
    platform._wmi_query = _no_wmi  # type: ignore[attr-defined]

from PyInstaller.__main__ import run  # noqa: E402 — must follow the patch

if __name__ == "__main__":
    run()
