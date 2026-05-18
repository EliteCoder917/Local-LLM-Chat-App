"""LlamaServerEngine — speaks to a `llama-server.exe` subprocess.

`llama-server` is the official HTTP server from the upstream llama.cpp project
(https://github.com/ggerganov/llama.cpp/releases). It ships prebuilt CUDA
binaries with up-to-date GPU + architecture support (Blackwell, Qwen3-MoE,
etc.) — what we'd get from `llama-cpp-python` if its wheels actually compiled
on bleeding-edge hardware.

Lifecycle:
* `__init__` validates settings, computes GPU offload, spawns the binary,
  polls `/health` until it answers
* `stream()` calls `/v1/chat/completions` (OpenAI-compatible) with SSE
* `close()` terminates the subprocess

The loader owns the engine instance and calls `close()` on unload, so VRAM /
RAM gets freed cleanly when the user swaps models.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from typing import AsyncIterator, List, Optional

import httpx

from ..config import CONFIG
from .engine import ChatMessage, LLMEngine
from .gguf_meta import read_gguf_meta
from .gpu import detect_nvidia_gpus


def _pick_free_port(prefer: int = 18080) -> int:
    """Return a free TCP port — prefers `prefer`, falls back to anything."""
    for p in (prefer, prefer + 1, prefer + 2, prefer + 3, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", p))
                return s.getsockname()[1]
        except OSError:
            continue
    raise RuntimeError("Could not find a free port for llama-server")


def _resolve_n_gpu_layers(size_gb: float, total_layers: int) -> int:
    """Translate the user's GB-of-VRAM slider into a layer count."""
    gb_per_layer = size_gb / (total_layers + 1)
    requested = max(0.0, float(CONFIG.gpu_offload_gb))
    if requested >= size_gb:
        return 999  # "all of them"; llama-server caps internally
    if requested <= 0 or gb_per_layer <= 0:
        return 0
    return max(1, min(total_layers, int(round(requested / gb_per_layer))))


class LlamaServerEngine(LLMEngine):
    def __init__(self) -> None:
        server = CONFIG.llama_server_path
        if not server:
            raise RuntimeError(
                "No llama-server.exe configured. Open Settings → Model → "
                "Browse for llama-server.exe (download from "
                "https://github.com/ggerganov/llama.cpp/releases — pick the "
                "Windows CUDA build matching your driver, e.g. cu12.8)."
            )
        if not os.path.isfile(server):
            raise RuntimeError(f"llama-server.exe not found at: {server}")

        model = CONFIG.model_path
        if not model or not os.path.isfile(model):
            raise RuntimeError(
                "No .gguf model selected. Open Settings → Model → Browse."
            )

        # Compute GPU layer count from the GB-of-VRAM slider.
        size_gb = os.path.getsize(model) / (1024 ** 3)
        meta = read_gguf_meta(model)
        total_layers = int(meta.get("block_count") or 32)
        n_gpu_layers = _resolve_n_gpu_layers(size_gb, total_layers)

        port = _pick_free_port()
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self._client = httpx.AsyncClient(timeout=None)

        args = [
            server,
            "--model", model,
            "--host", "127.0.0.1",
            "--port", str(port),
            "--ctx-size", str(max(512, int(CONFIG.n_ctx) or 4096)),
            "--n-gpu-layers", str(n_gpu_layers),
            "--threads", str(max(1, (os.cpu_count() or 4) // 2)),
            "--log-disable",     # we capture stdout/stderr ourselves
        ]

        # Spawn detached enough that Ctrl+C in our shell doesn't take it
        # down, but kill-able when we want.
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW
        try:
            self.proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=flags,
                text=True,
                bufsize=1,
            )
        except OSError as e:
            raise RuntimeError(f"Could not spawn llama-server: {e}") from e

        # Wait for /health to answer. llama-server prints "model loaded" once
        # the .gguf is in memory; before then it returns 503 from /health.
        ready, last_log = self._wait_for_health(timeout=180)
        if not ready:
            # Drain whatever stdout we have so the error is useful.
            tail = self._drain_proc_output(max_lines=40)
            self.close()
            raise RuntimeError(
                "llama-server did not become ready in 180s.\n\n"
                f"Command: {' '.join(args)}\n\n"
                f"--- llama-server log tail ---\n{tail}\n"
                "-----------------------------\n\n"
                "Common causes:\n"
                "  • GPU offload too high — lower the slider.\n"
                "  • Wrong CUDA build of llama-server (driver/runtime mismatch).\n"
                "  • Model architecture not supported by this llama-server "
                "version — download a fresher release."
            )

        # Remember the most recent stderr/stdout line for diagnostics; the
        # rest goes to the void to avoid filling memory.
        self._reader = asyncio.create_task(self._drain_async())

    # ─── lifecycle ─────────────────────────────────────────────────
    def _wait_for_health(self, timeout: float) -> tuple[bool, str]:
        deadline = time.time() + timeout
        last_status = "—"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False, f"process exited with code {self.proc.returncode}"
            try:
                r = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200:
                    return True, ""
                last_status = f"HTTP {r.status_code}"
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        return False, last_status

    def _drain_proc_output(self, max_lines: int) -> str:
        if not self.proc.stdout:
            return ""
        # Non-blocking-ish drain: we read whatever's currently buffered.
        try:
            self.proc.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        out: List[str] = []
        try:
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    break
                out.append(line.rstrip())
                if len(out) >= max_lines:
                    break
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(out[-max_lines:])

    async def _drain_async(self) -> None:
        """Continuously discard stdout in the background so the OS pipe
        buffer never fills (which would deadlock llama-server)."""
        if not self.proc.stdout:
            return
        loop = asyncio.get_event_loop()
        while True:
            line = await loop.run_in_executor(None, self.proc.stdout.readline)
            if not line:
                return
            # Forward to our own stderr so it shows up in the dev console.
            sys.stderr.write(f"[llama-server] {line}")

    def close(self) -> None:
        try:
            if getattr(self, "_reader", None):
                self._reader.cancel()
        except Exception:  # noqa: BLE001
            pass
        proc = getattr(self, "proc", None)
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        client = getattr(self, "_client", None)
        if client is not None:
            try:
                asyncio.create_task(client.aclose())
            except Exception:  # noqa: BLE001
                pass

    def __del__(self):  # noqa: D401
        # Belt-and-braces in case the loader forgets to call close().
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ─── streaming ─────────────────────────────────────────────────
    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": "local",     # llama-server ignores this; some clients require it
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "temperature": CONFIG.temperature,
            "cache_prompt": True,
        }
        async with self._client.stream("POST", url, json=payload) as r:
            if r.status_code != 200:
                body = await r.aread()
                raise RuntimeError(
                    f"llama-server HTTP {r.status_code}: "
                    f"{body.decode(errors='replace')[:500]}"
                )
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (
                    obj.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if delta:
                    yield delta


# Re-export some helpers so the loader can use them without circular imports.
__all__ = ["LlamaServerEngine", "detect_nvidia_gpus"]
