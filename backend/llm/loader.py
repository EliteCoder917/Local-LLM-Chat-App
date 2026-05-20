"""ModelLoader — LM-Studio-style explicit model lifecycle.

States:
    idle      no model in memory
    loading   construction in progress (Llama() is mmap-ing weights)
    loaded    ready to stream
    error     last load failed; `message` carries the reason

The loader holds exactly one engine instance. Re-loading the same key is a
no-op; loading a different key (engine swap or new .gguf path) unloads the
previous engine first so its weights can be GC'd.

Status changes are pushed to subscribers via callbacks — `ws.py` registers
one so the renderer sees state transitions live.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from ..config import CONFIG
from .engine import LLMEngine


Snapshot = Dict[str, Any]
Listener = Callable[[Snapshot], "Awaitable[None] | None"]


def _current_key() -> str:
    return (
        f"{CONFIG.engine}|{CONFIG.model_path}|{CONFIG.mmproj_path}|"
        f"vh={CONFIG.vision_handler}|{CONFIG.model}|"
        f"{CONFIG.ollama_url}|ctx={CONFIG.n_ctx}|gpu={CONFIG.gpu_offload_gb}"
    )


def _construct(on_progress=None) -> LLMEngine:
    if CONFIG.engine == "ollama":
        from .ollama_engine import OllamaEngine
        return OllamaEngine()
    if CONFIG.engine == "llama-cpp":
        from .llama_cpp_engine import LlamaCppEngine
        return LlamaCppEngine(on_progress=on_progress)
    raise ValueError(f"Unknown engine: {CONFIG.engine}")


class ModelLoader:
    def __init__(self) -> None:
        self.status: str = "idle"      # idle | loading | loaded | error
        self.message: str = ""
        self.engine: Optional[LLMEngine] = None
        self.engine_key: Optional[str] = None
        self.load_started_at: float = 0.0
        self.load_finished_at: float = 0.0
        self.progress: float = 0.0
        self._last_progress_notify: float = 0.0
        self._lock = threading.RLock()
        self._loading_task: Optional[asyncio.Task] = None
        self._listeners: set[Listener] = set()

    # ─── subscriptions ────────────────────────────────────────────────
    def add_listener(self, cb: Listener) -> None:
        self._listeners.add(cb)

    def remove_listener(self, cb: Listener) -> None:
        self._listeners.discard(cb)

    def _notify(self) -> None:
        snap = self.snapshot()
        for cb in list(self._listeners):
            try:
                r = cb(snap)
                if asyncio.iscoroutine(r):
                    asyncio.create_task(r)
            except Exception:  # noqa: BLE001
                pass

    # ─── status ───────────────────────────────────────────────────────
    def snapshot(self) -> Snapshot:
        load_ms = (
            int((self.load_finished_at - self.load_started_at) * 1000)
            if self.load_finished_at and self.status == "loaded"
            else None
        )
        vision_active = bool(getattr(self.engine, "vision_active", False))
        vision_handler = getattr(self.engine, "vision_handler_name", None)
        # True only when the loaded tokenizer recognises a `<think>` token —
        # i.e. the model can actually reason. Used by the UI to decide whether
        # to show the Smart/Quick picker (hiding it for non-thinking models
        # avoids the user injecting `/no_think` markers the model echoes back).
        supports_thinking = bool(getattr(self.engine, "supports_thinking", False))
        return {
            "status": self.status,
            "message": self.message,
            "engine": CONFIG.engine,
            "model": CONFIG.model,
            "modelPath": CONFIG.model_path,
            "loadedKey": self.engine_key,
            "currentKey": _current_key(),
            "loadMs": load_ms,
            "progress": self.progress if self.status == "loading" else None,
            # True only when an mmproj was paired AND a vision ChatHandler was
            # successfully constructed for this model's family. The library may
            # display a "vision" badge optimistically based on a paired mmproj,
            # but this flag reflects whether the LOADED engine will actually
            # see images.
            "visionActive": vision_active,
            "visionHandler": vision_handler,
            "supportsThinking": supports_thinking,
            # Actual context window the engine loaded with — lets the UI show
            # the real number when n_ctx is on Auto (0).
            "nCtx": getattr(self.engine, "n_ctx_resolved", None),
        }

    def is_loaded(self) -> bool:
        # "Is there a usable engine in memory?" — NOT "do current settings
        # match it?". A pending change to a reload-only setting (n_ctx / GPU
        # offload) makes _current_key() diverge from engine_key, but the loaded
        # engine still works fine with the settings it was BUILT with. Gating
        # inference on the key would make the model look "not loaded" the moment
        # the user nudges a slider, which is the crash they were seeing. The
        # "settings changed, reload to apply" signal lives in the snapshot
        # (loadedKey != currentKey) and drives the Save & reload button.
        return self.engine is not None

    def is_stale(self) -> bool:
        """Engine is loaded but built with settings that no longer match
        CONFIG — i.e. a reload is needed to apply the change."""
        return self.engine is not None and self.engine_key != _current_key()

    # ─── actions ──────────────────────────────────────────────────────
    async def load(self) -> Snapshot:
        with self._lock:
            if self.status == "loading":
                return self.snapshot()
            if self.is_loaded():
                return self.snapshot()
            self._reset_engine()
            self.status = "loading"
            self.message = "Reading model metadata…"
            self.load_started_at = time.time()
            self.load_finished_at = 0.0
            self.progress = 0.0
            self._last_progress_notify = 0.0
        self._notify()

        # Progress callback wired into llama-cpp; runs on a worker thread.
        # Once progress reaches 1.0, the engine's __init__ continues with a
        # warm-up inference — flip the message to "Warming kernels…" so the
        # user knows what the brief tail of the loading bar represents.
        def on_progress(p: float) -> None:
            now = time.time()
            with self._lock:
                self.progress = max(0.0, min(1.0, p))
                if (now - self._last_progress_notify) < 0.1 and p < 1.0:
                    return
                self._last_progress_notify = now
                pct = int(self.progress * 100)
                if self.progress >= 0.999:
                    self.message = "Warming kernels…"
                else:
                    self.message = f"Loading weights… {pct}%"
            self._notify()

        # Elapsed-time heartbeat — keeps the UI from looking frozen while
        # llama.cpp is mmap-ing weights or allocating KV cache.
        load_task = asyncio.create_task(
            asyncio.to_thread(_construct, on_progress)
        )
        heartbeat = asyncio.create_task(self._heartbeat(load_task))
        try:
            engine = await load_task
        except BaseException as e:  # noqa: BLE001 — covers native crashes
            heartbeat.cancel()
            with self._lock:
                self.status = "error"
                self.message = f"{type(e).__name__}: {e}"
                self.engine = None
                self.engine_key = None
            self._notify()
            return self.snapshot()
        finally:
            heartbeat.cancel()

        with self._lock:
            self.engine = engine
            self.engine_key = _current_key()
            self.status = "loaded"
            self.message = "Ready"
            self.load_finished_at = time.time()
        self._notify()
        return self.snapshot()

    def unload(self) -> Snapshot:
        with self._lock:
            self._reset_engine()
            self.status = "idle"
            self.message = ""
        self._notify()
        return self.snapshot()

    def _reset_engine(self) -> None:
        # Drop reference so the underlying Llama (heavy native object) is freed.
        # CUDA buffers don't get released until the Llama object actually goes
        # away — so we call its `close()` (if the engine exposes one) and force
        # a GC pass before returning. Without this, VRAM stays pinned until the
        # next load, which the GPU panel reflects as "0 GB free".
        engine = self.engine
        if engine is not None:
            close = getattr(engine, "close", None)
            if callable(close):
                try: close()
                except Exception: pass  # noqa: BLE001
        self.engine = None
        self.engine_key = None
        import gc
        gc.collect()

    def get(self) -> LLMEngine:
        if not self.is_loaded():
            raise RuntimeError(
                "Model is not loaded. Open Settings → Model → Load model."
            )
        assert self.engine is not None
        return self.engine

    async def _heartbeat(self, watched: asyncio.Task) -> None:
        """Push elapsed time to the UI every few seconds during a load."""
        try:
            while not watched.done():
                await asyncio.sleep(2.0)
                if self.status != "loading":
                    return
                elapsed = time.time() - self.load_started_at
                with self._lock:
                    self.message = (
                        f"Still loading… {elapsed:.0f}s elapsed. "
                        f"Large MoE models can take 60-180s on first cold load."
                    )
                self._notify()
        except asyncio.CancelledError:
            return


loader = ModelLoader()
