"""Auto-imported by Python's `site` machinery at interpreter startup when this
directory is on PYTHONPATH. Disables platform.win32_ver()'s WMI query so it
falls back to the (fast) registry path.

The build sets PYTHONPATH to include this dir before invoking PyInstaller, so
the patch reaches not just the parent process but every isolated subprocess
PyInstaller spawns during analysis — each of which re-imports PyInstaller's
compat module, which calls win32_ver() and would otherwise hang on a wedged
winmgmt service.
"""
try:
    import platform

    if hasattr(platform, "_wmi_query"):
        def _no_wmi(*_a, **_k):
            raise OSError("WMI disabled by build sitecustomize (avoids hung winmgmt)")

        platform._wmi_query = _no_wmi  # type: ignore[attr-defined]
except Exception:
    pass
