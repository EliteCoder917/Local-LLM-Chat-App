"""Local model library.

A "model" is a `.gguf` file in `<data_dir>/models/`. The library scans that
folder on demand and exposes typed entries (size, architecture, layer count)
to the renderer.

Identity is the **filename** — stable and human-readable. We never use full
absolute paths as model IDs because they leak the user's home folder into
the UI and break across reinstalls.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional

from ..config import data_dir
from ..llm.gguf_meta import read_gguf_meta


def _is_mmproj(name: str) -> bool:
    """A multimodal projector / CLIP companion. Always starts with 'mmproj'
    in every quant publisher's convention I've seen (bartowski, ggml-org,
    Mozilla LLamafile, etc.)."""
    return name.lower().startswith("mmproj")


def _strip_quant_suffix(stem: str) -> str:
    """Strip the trailing quant tag (e.g. -Q4_K_M, -f16, -q8_0) so a main
    model and its mmproj can be matched by shared stem."""
    import re
    return re.sub(
        r"[-_.](?:q\d_[\w_]+|q\d|f16|f32|bf16|fp16|fp32|iq\d_[\w_]+)\b\.?",
        "",
        stem,
        flags=re.IGNORECASE,
    ).rstrip("._- ")


def _find_mmproj_for(
    main_name: str, mmprojs: list[tuple[str, str]],
) -> Optional[tuple[str, str]]:
    """Find the best mmproj for a given main model filename.

    Strategy:
      1. Strip .gguf and quant suffix from both, compare stems case-insensitively
      2. If exactly one mmproj is in the library, just pair it (assume the user
         downloaded matching pair)
    """
    if not mmprojs:
        return None
    main_stem = _strip_quant_suffix(main_name[:-5] if main_name.lower().endswith(".gguf") else main_name).lower()
    for name, path in mmprojs:
        # Strip "mmproj-" prefix and the .gguf extension before comparing.
        proj_stem = name[:-5] if name.lower().endswith(".gguf") else name
        proj_stem = proj_stem[7:] if proj_stem.lower().startswith("mmproj-") else proj_stem
        proj_stem = _strip_quant_suffix(proj_stem).lower()
        if proj_stem and (proj_stem in main_stem or main_stem in proj_stem):
            return name, path
    # Fallback: if there's only one mmproj total, assume it's for this model.
    if len(mmprojs) == 1:
        return mmprojs[0]
    return None


@dataclass
class ModelEntry:
    id: str             # filename (unique within the library)
    name: str           # display name (currently same as id minus .gguf)
    path: str           # absolute path on disk
    size_gb: float
    arch: Optional[str]
    block_count: Optional[int]
    trained_context: Optional[int]
    # Multimodal companion — paired mmproj-*.gguf if present in the library.
    mmproj_path: Optional[str] = None
    mmproj_name: Optional[str] = None

    @property
    def is_vision(self) -> bool:
        return self.mmproj_path is not None

    def to_json(self) -> dict:
        d = asdict(self)
        d["sizeGb"] = d.pop("size_gb")
        d["blockCount"] = d.pop("block_count")
        d["trainedContext"] = d.pop("trained_context")
        d["mmprojPath"] = d.pop("mmproj_path")
        d["mmprojName"] = d.pop("mmproj_name")
        d["isVision"] = self.is_vision
        return d


class ModelLibrary:
    """Filesystem-backed model library. Thread-safe."""

    def __init__(self, root: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._root = root or os.path.join(data_dir(), "models")
        os.makedirs(self._root, exist_ok=True)

    @property
    def root(self) -> str:
        return self._root

    def list(self) -> List[ModelEntry]:
        with self._lock:
            all_files: list[tuple[str, str]] = []
            for name in sorted(os.listdir(self._root)):
                if not name.lower().endswith(".gguf"):
                    continue
                full = os.path.join(self._root, name)
                if not os.path.isfile(full):
                    continue
                all_files.append((name, full))

            # Separate multimodal projector files from main models. A vision
            # GGUF pair has two files: the LLM weights + an `mmproj-*.gguf`
            # CLIP projector. We hide the projector as a standalone model and
            # attach it to the matching main weights.
            mmprojs = [(n, p) for n, p in all_files if _is_mmproj(n)]
            mains = [(n, p) for n, p in all_files if not _is_mmproj(n)]

            entries: List[ModelEntry] = []
            for name, full in mains:
                entry = self._inspect(name, full)
                pair = _find_mmproj_for(name, mmprojs)
                if pair:
                    entry.mmproj_name, entry.mmproj_path = pair
                entries.append(entry)
            return entries

    def get(self, model_id: str) -> Optional[ModelEntry]:
        with self._lock:
            full = self._resolve(model_id)
            if not full:
                return None
            entry = self._inspect(model_id, full)
            # Pair with any matching mmproj-*.gguf in the same folder. Without
            # this the vision handler never gets constructed on /library/select,
            # the engine boots text-only, and the model hallucinates from the
            # filename instead of actually seeing the image.
            mmprojs: list[tuple[str, str]] = []
            for name in sorted(os.listdir(self._root)):
                if not name.lower().endswith(".gguf") or not _is_mmproj(name):
                    continue
                p = os.path.join(self._root, name)
                if os.path.isfile(p):
                    mmprojs.append((name, p))
            pair = _find_mmproj_for(model_id, mmprojs)
            if pair:
                entry.mmproj_name, entry.mmproj_path = pair
            return entry

    def path_for(self, model_id: str) -> Optional[str]:
        return self._resolve(model_id)

    def delete(self, model_id: str) -> bool:
        with self._lock:
            full = self._resolve(model_id)
            if not full:
                return False
            os.remove(full)
            return True

    # ─── internal ────────────────────────────────────────────────────
    def _resolve(self, model_id: str) -> Optional[str]:
        # Strip any path components — id must be a bare filename.
        safe = os.path.basename(model_id)
        if not safe.lower().endswith(".gguf"):
            safe = safe + ".gguf"
        full = os.path.join(self._root, safe)
        return full if os.path.isfile(full) else None

    def _inspect(self, model_id: str, full: str) -> ModelEntry:
        size_gb = os.path.getsize(full) / (1024 ** 3)
        meta = read_gguf_meta(full)
        return ModelEntry(
            id=model_id,
            name=model_id[:-5] if model_id.lower().endswith(".gguf") else model_id,
            path=full,
            size_gb=size_gb,
            arch=meta.get("arch"),
            block_count=meta.get("block_count"),
            trained_context=meta.get("context_length"),
        )


library = ModelLibrary()
