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
    """Return a dict of model metadata needed for accurate VRAM accounting.

    Keys (any may be missing — caller should fall back to heuristics):
        arch                      e.g. "qwen3vl", "qwen35moe", "llama"
        block_count               number of transformer layers
        context_length            trained context window
        embedding_length          hidden dim (n_embd)
        head_count                number of attention heads
        head_count_kv             number of KV heads (= head_count for MHA,
                                  < head_count for GQA — Qwen3 MoE uses GQA)
        rope_dimension_count      head_dim for RoPE (often n_embd / head_count)
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
                elif key.endswith(".attention.head_count"):
                    out["head_count"] = int(value)
                elif key.endswith(".attention.head_count_kv"):
                    out["head_count_kv"] = int(value)
                elif key.endswith(".rope.dimension_count"):
                    out["rope_dimension_count"] = int(value)
    except Exception:  # noqa: BLE001 — best-effort parse
        pass
    return out


def estimate_kv_cache_gb(
    n_ctx: int,
    block_count: int,
    head_count_kv: int = 0,
    rope_dim: int = 0,
    embedding_length: int = 0,
    head_count: int = 0,
    dtype_bytes: int = 2,
) -> float:
    """Accurate KV-cache size formula:
        2 (K + V) × n_layers × n_ctx × n_kv_heads × head_dim × dtype_bytes

    Falls back to a hand-tuned constant if architecture details are missing.
    """
    if head_count_kv and rope_dim and block_count:
        bytes_total = 2 * block_count * n_ctx * head_count_kv * rope_dim * dtype_bytes
        return bytes_total / (1024 ** 3)
    # Fallback: derive head_dim from n_embd / n_heads if those are known.
    if embedding_length and head_count and block_count:
        head_dim = embedding_length // max(1, head_count)
        # For GQA models without explicit head_count_kv we assume MHA (kv_heads = heads).
        kv = head_count_kv or head_count
        bytes_total = 2 * block_count * n_ctx * kv * head_dim * dtype_bytes
        return bytes_total / (1024 ** 3)
    # Hand-tuned last resort — calibrated against Llama-3 8B at 8K (~0.5 GB).
    return (n_ctx / 1024) * max(1, block_count) * 0.0001 * 16
