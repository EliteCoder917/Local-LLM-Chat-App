"""File-system tools.

All paths are resolved against `CONFIG.workspace` and rejected if they escape
it (no `..` traversal).
"""
from __future__ import annotations

import os
import shutil
from typing import Dict

from ..config import CONFIG


def _resolve(path: str) -> str:
    """Resolve `path` inside the workspace. Raise on escape."""
    if not CONFIG.workspace:
        raise ValueError("No workspace open. Open a folder before using file tools.")
    base = os.path.abspath(CONFIG.workspace)
    target = os.path.abspath(
        path if os.path.isabs(path) else os.path.join(base, path)
    )
    try:
        common = os.path.commonpath([base, target])
    except ValueError:
        raise ValueError(f"Path '{path}' is outside the workspace") from None
    if common != base:
        raise ValueError(f"Path '{path}' is outside the workspace")
    return target


def read_file(path: str) -> str:
    abs_path = _resolve(path)
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    abs_path = _resolve(path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    with open(abs_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return f"Wrote {len(content)} chars to {path}"


def create_folder(path: str) -> str:
    abs_path = _resolve(path)
    os.makedirs(abs_path, exist_ok=True)
    return f"Created folder {path}"


def delete_file(path: str) -> str:
    abs_path = _resolve(path)
    if os.path.isdir(abs_path):
        shutil.rmtree(abs_path)
        return f"Deleted folder {path}"
    os.remove(abs_path)
    return f"Deleted file {path}"


def move_file(src: str, dst: str) -> str:
    src_abs = _resolve(src)
    dst_abs = _resolve(dst)
    os.makedirs(os.path.dirname(dst_abs) or ".", exist_ok=True)
    shutil.move(src_abs, dst_abs)
    return f"Moved {src} → {dst}"


def rename_file(old: str, new: str) -> str:
    return move_file(old, new)


def list_dir(path: str = ".") -> list:
    abs_path = _resolve(path)
    entries = []
    for name in sorted(os.listdir(abs_path)):
        full = os.path.join(abs_path, name)
        entries.append({
            "name": name,
            "path": full,
            "isDir": os.path.isdir(full),
        })
    return entries


def search_text(query: str, path: str = ".", max_results: int = 100) -> list:
    """Naive text search for grep-style results."""
    abs_path = _resolve(path)
    out = []
    for root, _, files in os.walk(abs_path):
        for fn in files:
            full = os.path.join(root, fn)
            try:
                with open(full, "r", encoding="utf-8", errors="ignore") as f:
                    for ln, line in enumerate(f, 1):
                        if query in line:
                            out.append({"path": full, "line": ln,
                                        "preview": line.rstrip()[:200]})
                            if len(out) >= max_results:
                                return out
            except OSError:
                continue
    return out
