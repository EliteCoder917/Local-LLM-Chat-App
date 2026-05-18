"""GPU detection via `nvidia-smi`.

We shell out instead of importing pynvml because (a) no extra dep and (b)
nvidia-smi is installed alongside any working NVIDIA driver on Windows.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Dict, List


def detect_nvidia_gpus() -> List[Dict[str, float | str]]:
    """Returns one entry per visible NVIDIA GPU, or an empty list if none.

    Each entry: {'name': str, 'free_gb': float, 'total_gb': float}.
    """
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        r = subprocess.run(
            [smi, "--query-gpu=name,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if r.returncode != 0:
        return []
    out: List[Dict[str, float | str]] = []
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            out.append({
                "name": parts[0],
                "free_gb": float(parts[1]) / 1024.0,
                "total_gb": float(parts[2]) / 1024.0,
            })
        except ValueError:
            continue
    return out
