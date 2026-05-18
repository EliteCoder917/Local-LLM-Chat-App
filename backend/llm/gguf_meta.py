"""Minimal GGUF v3 metadata reader.

We only need a handful of fields (`general.architecture`, `<arch>.block_count`,
`<arch>.context_length`) to convert between "GB to offload" and
"n_gpu_layers". Reading the header is dirt-cheap — a few hundred bytes — so
we don't take a hard dependency on the `gguf` package.

If the file is malformed or uses a future GGUF version we don't recognize,
returns an empty dict; callers fall back to heuristics.
"""
from __future__ import annotations

import struct
from typing import Any, Dict


_MAGIC = b"GGUF"

# Value type tags per the GGUF v3 spec.
_T_UINT8 = 0
_T_INT8 = 1
_T_UINT16 = 2
_T_INT16 = 3
_T_UINT32 = 4
_T_INT32 = 5
_T_FLOAT32 = 6
_T_BOOL = 7
_T_STRING = 8
_T_ARRAY = 9
_T_UINT64 = 10
_T_INT64 = 11
_T_FLOAT64 = 12


def _read_string(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    if n > 1 << 16:           # sanity: any key/string longer than 64 KB is garbage
        raise ValueError(f"GGUF string too long: {n}")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int) -> Any:
    if vtype == _T_UINT8:   return f.read(1)[0]
    if vtype == _T_INT8:    return struct.unpack("<b", f.read(1))[0]
    if vtype == _T_UINT16:  return struct.unpack("<H", f.read(2))[0]
    if vtype == _T_INT16:   return struct.unpack("<h", f.read(2))[0]
    if vtype == _T_UINT32:  return struct.unpack("<I", f.read(4))[0]
    if vtype == _T_INT32:   return struct.unpack("<i", f.read(4))[0]
    if vtype == _T_FLOAT32: return struct.unpack("<f", f.read(4))[0]
    if vtype == _T_BOOL:    return bool(f.read(1)[0])
    if vtype == _T_STRING:  return _read_string(f)
    if vtype == _T_UINT64:  return struct.unpack("<Q", f.read(8))[0]
    if vtype == _T_INT64:   return struct.unpack("<q", f.read(8))[0]
    if vtype == _T_FLOAT64: return struct.unpack("<d", f.read(8))[0]
    if vtype == _T_ARRAY:
        (item_type,) = struct.unpack("<I", f.read(4))
        (length,) = struct.unpack("<Q", f.read(8))
        if length > 1 << 20:
            raise ValueError(f"GGUF array too long: {length}")
        return [_read_value(f, item_type) for _ in range(length)]
    raise ValueError(f"Unknown GGUF value type: {vtype}")


def read_gguf_meta(path: str) -> Dict[str, Any]:
    """Return a dict with keys: arch, block_count, context_length, embedding_length.
    Missing keys mean we couldn't parse them — caller should provide defaults.
    """
    out: Dict[str, Any] = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != _MAGIC:
                return out
            (version,) = struct.unpack("<I", f.read(4))
            if version not in (1, 2, 3):
                return out
            (_tensor_count,) = struct.unpack("<Q", f.read(8))
            (kv_count,) = struct.unpack("<Q", f.read(8))

            # Cap the read; metadata above a few hundred entries is suspicious.
            for _ in range(min(kv_count, 1024)):
                key = _read_string(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                value = _read_value(f, vtype)
                if key == "general.architecture":
                    out["arch"] = value
                elif key.endswith(".block_count"):
                    out["block_count"] = int(value)
                elif key.endswith(".context_length"):
                    out["context_length"] = int(value)
                elif key.endswith(".embedding_length"):
                    out["embedding_length"] = int(value)
                if "arch" in out and "block_count" in out and "context_length" in out:
                    break
    except Exception:  # noqa: BLE001 — best-effort parse
        pass
    return out
