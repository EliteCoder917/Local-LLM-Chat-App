"""llama-cpp-python engine. Loads a local .gguf model.

Two things make load failures legible:

* We capture OS-level stderr (FD 2) while `Llama()` runs. llama.cpp's own
  progress / error log goes there, not to Python's sys.stderr, so without
  this redirect a native crash gives us no clue what step failed. With it,
  the error surfaces the last few lines llama.cpp printed before the
  access violation — that's the actual diagnostic information.

* We force a sane `n_ctx` rather than asking for the model's training
  context. Qwen3 / Llama-3 advertise 32K-128K, and the KV-cache for that
  on a large MoE doesn't fit on most machines — when llama.cpp's
  allocator returns NULL it gets dereferenced and the process crashes.
"""
from __future__ import annotations

import asyncio
import os
import sys
import threading
from contextlib import contextmanager
from tempfile import TemporaryFile
from typing import AsyncIterator, Callable, Dict, List, Optional


class _AbortGeneration(Exception):
    """Raised inside llama.cpp's logits_processor when the user clicks Stop.
    The producer catches this and exits cleanly so the next prompt can run."""
    pass

from ..config import CONFIG
from .engine import ChatMessage, LLMEngine
from .gguf_meta import read_gguf_meta
from .gpu import detect_nvidia_gpus


# Sane default; users can raise it later via Settings if their machine can
# afford the KV cache.
DEFAULT_N_CTX = 4096


# ─── Process-scoped log filter ──────────────────────────────────────
# llama.cpp's `llama_log_set` registers a raw C function pointer. If we
# wrap a Python callback as an instance attribute and the engine is GC'd
# (which happens on every Eject), llama.cpp ends up calling a dangling
# pointer the next time it logs — manifests as STATUS_ILLEGAL_INSTRUCTION
# on the second model load. Install the callback exactly once per process
# and keep a permanent module-level reference so the ctypes thunk stays
# alive for the whole app lifetime.
_LOG_FILTER_INSTALLED = False
_LOG_FILTER_REF = None  # module-global reference holder; do NOT remove

_NOISE = (
    "ggml_cuda_graph_check_compability",
    "ggml_cuda_graph_update",
    # Compute-buffer auto-grow log. Vision models trip this on every image
    # because patch counts vary per image → graph topology changes → buffer
    # is re-reserved. The "failed to allocate" phrasing is misleading; the
    # scheduler always succeeds on the retry with the bigger buffer.
    "ggml_backend_sched_alloc_splits: failed to allocate graph, reserving",
    # M-RoPE position quirk that fires when vision tokens get inserted into
    # the KV cache — also benign, the embeddings still land correctly thanks
    # to our wrapper bypass.
    "find_slot: non-consecutive token position",
)


def _install_log_filter_once() -> None:
    global _LOG_FILTER_INSTALLED, _LOG_FILTER_REF
    if _LOG_FILTER_INSTALLED:
        return
    try:
        from llama_cpp import llama_log_set, llama_log_callback  # type: ignore
    except Exception:  # noqa: BLE001
        return

    @llama_log_callback
    def _filter(_level, msg, _user_data):  # noqa: ANN001
        try:
            text = msg.decode("utf-8", errors="replace") if msg else ""
            if any(n in text for n in _NOISE):
                return
            sys.stderr.write(text)
        except Exception:  # noqa: BLE001
            pass

    try:
        llama_log_set(_filter, None)
        _LOG_FILTER_REF = _filter
        _LOG_FILTER_INSTALLED = True
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def _capture_native_stderr(box: Dict[str, str]):
    """Capture writes to FD 2 (what llama.cpp uses) during the with-block.

    Result lands in `box['text']` AFTER the block exits.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    tmp = TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            tmp.seek(0)
            box["text"] = tmp.read().decode("utf-8", errors="replace")
    finally:
        os.close(saved)
        tmp.close()


def _ram_gb() -> tuple[Optional[float], Optional[float]]:
    """Returns (available_gb, total_gb) — both None if we can't query."""
    try:
        import ctypes
        class MEMSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        s = MEMSTATUS()
        s.dwLength = ctypes.sizeof(MEMSTATUS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
        return s.ullAvailPhys / (1024 ** 3), s.ullTotalPhys / (1024 ** 3)
    except Exception:  # noqa: BLE001
        return None, None


ProgressCb = Callable[[float], None]


# Vision diagnostics go to BOTH stderr (for live tail) AND a persistent file
# (so we don't lose them to terminal scrollback). The file path is reported
# at module load so the user can find it.
def _vision_log_path() -> str:
    from ..config import data_dir
    return os.path.join(data_dir(), "vision-debug.log")


def _vlog(msg: str) -> None:
    """Write a [vision] diagnostic to both stderr and the on-disk log file."""
    line = msg if msg.endswith("\n") else msg + "\n"
    try:
        sys.stderr.write(line)
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(_vision_log_path(), "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:  # noqa: BLE001
        pass


# Drop a marker on module load so we know which build is running and where
# the log file lives.
try:
    _p = _vision_log_path()
    with open(_p, "a", encoding="utf-8") as _f:
        import time as _t
        _f.write(f"\n=== module load at {_t.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    _vlog(f"[vision] diagnostics will be written to: {_p}\n")
except Exception:  # noqa: BLE001
    pass


class _VisionHandlerWrapper:
    """Delegates to a llama-cpp-python multimodal ChatHandler, but bypasses
    its destructive `Llama.create_completion` step.

    After the wrapped handler runs `mtmd_helper_eval_chunk_single`, the KV
    cache contains image embeddings at M-RoPE positions. The handler then
    calls `llama.create_completion(prompt=input_ids[:n_tokens].tolist())`,
    which inside `Llama.generate` calls `kv_cache_seq_rm` to chop the cache
    at the prefix-match breakpoint. For M-RoPE caches that call returns
    False, triggering a full `self.reset()` that WIPES the image embeddings.
    The model then sees raw `<|image_pad|>` token IDs and outputs `!`.

    Our fix: redirect `llama.create_completion` to a generator that does
    none of that — it samples directly from the existing KV state with no
    re-eval. The image embeddings stay intact and the model can actually
    attend to them.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        # Forward attribute access (clip_model_path, mtmd_ctx, get_image_urls,
        # etc.) to the wrapped handler so llama-cpp-python's internals see a
        # transparent stand-in.
        return getattr(self._inner, name)

    def __call__(self, *, llama, **kwargs):
        _vlog("[vision] wrapper.__call__ entered\n")
        orig_create_completion = llama.create_completion

        def _create_completion_direct(prompt, **cc_kwargs):  # noqa: ARG001
            _vlog(f"[vision] intercepted create_completion "
                f"(n_tokens={llama.n_tokens}, prompt_len={len(prompt) if prompt is not None else 'None'})\n"
            )
            """Replacement that skips prefix-match + re-eval.

            By the time the handler calls this, mtmd has populated the KV
            cache with the entire prompt INCLUDING image embeddings — the
            `prompt` token list is therefore unused (it's the same data,
            re-tokenized; re-evaluating it would just overwrite the image
            embeddings). We sample from the current state directly. The
            handler wraps our output with `_convert_completion_to_chat`,
            so it must match `create_completion`'s shape (iterator of
            chunks for streaming, or a single completion otherwise).
            """
            return _sample_from_current_state(
                llama=llama,
                temperature=cc_kwargs.get("temperature", 0.7),
                top_p=cc_kwargs.get("top_p", 0.95),
                top_k=cc_kwargs.get("top_k", 40),
                min_p=cc_kwargs.get("min_p", 0.05),
                max_tokens=cc_kwargs.get("max_tokens"),
                stop=cc_kwargs.get("stop"),
                stream=cc_kwargs.get("stream", False),
                model_name=cc_kwargs.get("model") or llama.model_path,
                repeat_penalty=cc_kwargs.get("repeat_penalty", 1.0),
            )

        llama.create_completion = _create_completion_direct
        try:
            return self._inner(llama=llama, **kwargs)
        finally:
            llama.create_completion = orig_create_completion


def _sample_from_current_state(
    *,
    llama,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    max_tokens: Optional[int],
    stop,
    stream: bool,
    model_name: str,
    repeat_penalty: float,
):
    """Sample tokens from the model's current KV state (post-mtmd).

    Replicates just enough of `Llama.create_completion`'s output structure
    for `_convert_completion_to_chat` to consume it. No prefix-match, no
    re-eval, no reset — the KV cache is sacrosanct here, because it's the
    only place the image embeddings live.
    """
    import time as _time
    import uuid as _uuid
    completion_id = f"cmpl-{_uuid.uuid4()}"
    created = int(_time.time())
    # Default cap was 512 — way too low for reasoning models. Qwen3-VL's
    # `<think>` mode routinely emits 1000-3000 tokens of reasoning before the
    # final answer. Hitting 512 mid-think makes the bubble look like "thought
    # but said nothing." Cap instead at "fill the remaining context window",
    # which lets the model run to its natural EOS in almost every case.
    headroom = max(64, llama.n_ctx() - int(llama.n_tokens) - 32)
    max_n = max_tokens if (max_tokens is not None and max_tokens > 0) else headroom
    stop_strs = stop if isinstance(stop, list) else ([stop] if isinstance(stop, str) else [])
    eos_ids = {llama.token_eos()}
    try:
        eos_ids.add(llama._model.token_eot())
    except Exception:  # noqa: BLE001
        pass

    # Probe the current logits to see whether mtmd produced reasonable
    # output. If the post-mtmd state samples token 0 (which often detokenizes
    # to `!` in Qwen tokenizers) deterministically, mtmd itself is broken
    # and no amount of sampler tuning will help.
    try:
        import numpy as _np
        import llama_cpp as _lcpp
        ctx = llama._ctx.ctx
        n_vocab = llama._n_vocab
        logits_ptr = _lcpp.llama_get_logits(ctx)
        logits = _np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
        top5 = _np.argsort(-logits)[:5]
        _vlog(f"[vision] post-mtmd logits probe: top5 token IDs = {top5.tolist()}, "
            f"top5 values = {[float(logits[i]) for i in top5]}, "
            f"mean={float(logits.mean()):.3f}, std={float(logits.std()):.3f}, "
            f"n_tokens={llama.n_tokens}\n"
        )
        # Detokenize the top candidates so we can SEE what the model wants
        # to emit next. If everything is `!`/`!!`/etc., mtmd is broken.
        for tid in top5.tolist():
            piece = llama.detokenize([int(tid)]).decode("utf-8", errors="replace")
            _vlog(f"[vision]   token {tid} -> {piece!r}\n")
    except Exception as e:  # noqa: BLE001
        _vlog(f"[vision] logits probe failed: {e}\n")

    # Init a sampler matching the caller's params.
    llama._sampler = llama._init_sampler(
        top_k=top_k, top_p=top_p, min_p=min_p, typical_p=1.0,
        temp=temperature, repeat_penalty=repeat_penalty,
    )

    def _produce():
        generated_text = b""
        generated_count = 0
        finish_reason = "length"
        # Buffer of bytes whose UTF-8 sequence is incomplete. Emojis (and any
        # non-ASCII char) often span multiple BPE tokens. Decoding each token's
        # bytes in isolation breaks mid-codepoint and yields � replacement
        # characters. We instead accumulate bytes and only emit fully decoded
        # text — any trailing partial sequence stays in the buffer until the
        # next token completes it.
        pending = b""
        # Sample one token at a time from the last position currently in the
        # KV cache. `idx=None` tells llama.sample to use the most recent
        # logits, which were produced by the last mtmd/eval call.
        #
        # Retry guard for the "first token is EOS" failure mode: when the
        # post-mtmd logits happen to favor <|im_end|>, random sampling at
        # temperature picks it as the very first token, the loop breaks
        # immediately, and the user gets an empty bubble. We give the model
        # one second chance: if generated_count is still 0 when EOS is
        # sampled, we re-init the sampler with that token banned via a
        # logit bias and try again. Once any real content is produced we
        # let EOS work normally.
        eos_banned = False
        abort_event = getattr(llama, "_gen_abort", None)

        # Quick mode: pre-fill an EMPTY think block into the KV cache so the
        # model sees its reasoning section as already closed. This is the
        # technique Qwen3's official chat template uses for
        # `enable_thinking=False`: the assistant prefix becomes
        # `<|im_start|>assistant\n<think>\n\n</think>\n\n` instead of just
        # `<|im_start|>assistant\n`. The model then generates from after the
        # close tag — directly to the answer, no reasoning. This is FAR more
        # reliable than logit-banning `<think>`, which just leaves the model
        # stuck emitting whitespace because it was trained to think first.
        if CONFIG.thinking_mode == "quick":
            try:
                prefill = llama.tokenize(
                    b"<think>\n\n</think>\n\n", add_bos=False, special=True,
                )
                if prefill:
                    llama.eval(prefill)
                    _vlog(
                        f"[thinking] Quick (vision path): pre-filled empty "
                        f"think block ({len(prefill)} tokens) so the model "
                        "skips reasoning.\n"
                    )
            except Exception as e:  # noqa: BLE001
                _vlog(f"[thinking] Quick pre-fill failed (non-fatal): {e}\n")

        while generated_count < max_n:
            # User clicked Stop — break the sampling loop so the next prompt
            # isn't queued behind a still-running vision generation.
            if abort_event is not None and abort_event.is_set():
                finish_reason = "stop"
                break
            tok = llama.sample(
                top_k=top_k, top_p=top_p, min_p=min_p, typical_p=1.0,
                temp=temperature, repeat_penalty=repeat_penalty,
                idx=None,
            )
            if tok in eos_ids:
                if generated_count == 0 and not eos_banned:
                    # First sampled token was EOS — likely a sampling lottery
                    # loss, not the model's actual intent. Re-pick the next
                    # token from the SAME post-mtmd logits with EOS excluded
                    # (simple argmax-without-EOS — robust across builds and
                    # doesn't depend on the lower-level sampler-chain API).
                    # Once content is flowing, EOS works normally for
                    # turn-end.
                    _vlog(
                        f"[vision] first-token EOS detected (tok={tok}); "
                        "re-picking from logits with EOS excluded.\n"
                    )
                    eos_banned = True
                    try:
                        import numpy as _np
                        import llama_cpp as _lcpp
                        logits_ptr = _lcpp.llama_get_logits(llama._ctx.ctx)
                        logits = _np.ctypeslib.as_array(
                            logits_ptr, shape=(llama._n_vocab,)
                        ).copy()
                        for e in eos_ids:
                            logits[int(e)] = -1e30
                        tok = int(_np.argmax(logits))
                    except Exception as e:  # noqa: BLE001
                        _vlog(f"[vision] EOS-rescue failed: {e}\n")
                        finish_reason = "stop"
                        break
                else:
                    finish_reason = "stop"
                    break
            # Feed the sampled token back into the model so the next sample
            # sees it. eval() advances n_tokens and writes input_ids.
            llama.eval([tok])
            piece = llama.detokenize([tok])
            generated_text += piece
            generated_count += 1
            # Stop-string match (rough — only checks the recent suffix).
            if stop_strs:
                tail = generated_text.decode("utf-8", errors="replace")
                if any(s and s in tail for s in stop_strs):
                    finish_reason = "stop"
                    break
            # Decode safely: keep any trailing partial UTF-8 sequence buffered
            # until the bytes that complete it arrive.
            pending += piece
            try:
                emit_text = pending.decode("utf-8")
                pending = b""
            except UnicodeDecodeError as ude:
                # `ude.start` is where the invalid/incomplete sequence begins.
                # Anything before that is valid; keep the rest for next time.
                emit_text = pending[: ude.start].decode("utf-8", errors="replace")
                pending = pending[ude.start :]
            if stream and emit_text:
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "text": emit_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }],
                }
        # Flush any remaining pending bytes — at end of stream they're never
        # going to complete, so emit with replace as a last resort.
        if stream and pending:
            tail = pending.decode("utf-8", errors="replace")
            if tail:
                yield {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "text": tail,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None,
                    }],
                }
        if stream:
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }],
            }
        else:
            yield {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "choices": [{
                    "text": generated_text.decode("utf-8", errors="replace"),
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": int(llama.n_tokens) - int(generated_count),
                    "completion_tokens": int(generated_count),
                    "total_tokens": int(llama.n_tokens),
                },
            }

    gen = _produce()
    if stream:
        return gen
    # Non-streaming: pull the single result.
    return next(gen)


class LlamaCppEngine(LLMEngine):
    def __init__(self, on_progress: Optional[ProgressCb] = None) -> None:
        # User-triggered abort flag. When the Stop button is pressed, abort()
        # is called; the logits_processor we register with llama.cpp checks
        # this every sample step and raises to terminate generation mid-flight
        # so the next prompt isn't blocked behind the prior one.
        self._abort = threading.Event()
        try:
            import llama_cpp  # type: ignore
            from llama_cpp import Llama  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run "
                "`pip install llama-cpp-python` or switch to the Ollama engine."
            ) from e

        path = CONFIG.model_path
        if not path:
            raise RuntimeError(
                "No .gguf model selected. Open Settings → Model → Browse."
            )
        if not os.path.isfile(path):
            raise RuntimeError(
                f"GGUF file not found: {path}\n"
                "Pick a valid .gguf file in Settings → Model → Browse."
            )
        size_mb = os.path.getsize(path) / (1024 * 1024)
        size_gb = size_mb / 1024
        if size_mb < 1:
            raise RuntimeError(
                f"GGUF file looks too small ({size_mb:.2f} MB) — "
                "likely a partial download. Re-download the model."
            )

        # ---- Resolve GPU offload (slider is now in GB; convert to layers) ----
        meta = read_gguf_meta(path)
        # +1 because the output / final norm layer is offloaded as one extra unit.
        total_layers = int(meta.get("block_count") or 32) + 1
        gb_per_layer = size_gb / total_layers

        offload_gb_req = max(0.0, float(CONFIG.gpu_offload_gb))
        if offload_gb_req >= size_gb:
            n_gpu_layers = -1                          # all layers
            offload_gb = size_gb
        elif offload_gb_req <= 0:
            n_gpu_layers = 0
            offload_gb = 0.0
        else:
            n_gpu_layers = max(1, int(round(offload_gb_req / gb_per_layer)))
            n_gpu_layers = min(n_gpu_layers, total_layers - 1)
            offload_gb = n_gpu_layers * gb_per_layer
        cpu_gb = max(0.0, size_gb - offload_gb)

        # ---- Memory pre-check (now accounts for GPU offload) ----
        avail_gb, total_gb = _ram_gb()
        gpus = detect_nvidia_gpus()
        # Pick the GPU with most free VRAM (or None).
        gpu = max(gpus, key=lambda g: g["free_gb"]) if gpus else None
        gpu_free = float(gpu["free_gb"]) if gpu else 0.0

        ram_budget = cpu_gb * 1.10 + 1.2          # weights on CPU + compute buf + KV cache + py overhead
        vram_budget = offload_gb * 1.10 + 0.5     # offloaded layers + GPU KV slice
        if avail_gb is not None and avail_gb < ram_budget:
            raise RuntimeError(
                "Not enough free system RAM for this configuration.\n\n"
                f"Model:           {size_gb:.1f} GB\n"
                f"On CPU (RAM):    {cpu_gb:.1f} GB\n"
                f"On GPU (VRAM):   {offload_gb:.1f} GB"
                + (f"   (target {offload_gb_req:.1f} GB)" if offload_gb_req else "")
                + "\n"
                f"RAM need:        {ram_budget:.1f} GB\n"
                f"RAM available:   {avail_gb:.1f} GB"
                + (f" / {total_gb:.1f} GB total" if total_gb else "") + "\n\n"
                "Slide GPU offload higher in Settings → Model "
                f"(your GPU has {gpu_free:.1f} GB VRAM free), pick a "
                "smaller quant, or close other apps."
            )
        if gpu and vram_budget > gpu_free + 0.5:
            raise RuntimeError(
                "Not enough free VRAM for the requested GPU offload.\n\n"
                f"Requested on GPU: {offload_gb:.1f} GB "
                f"(≈ {n_gpu_layers}/{total_layers} layers)\n"
                f"VRAM free:        {gpu_free:.1f} GB on {gpu['name']}\n\n"
                "Lower the GPU-offload slider in Settings → Model."
            )

        chat_format: Optional[str] = self._infer_chat_format(path)
        lcp_version = getattr(llama_cpp, "__version__", "unknown")

        # Vision support: if a paired mmproj-*.gguf was found in the library,
        # construct the matching multimodal ChatHandler. llama-cpp-python's
        # handlers wire CLIP image features into the LLM and accept structured
        # `image_url` content blocks (OpenAI format) instead of base64 in text.
        chat_handler = None
        self.vision_active = False
        self.vision_handler_name: Optional[str] = None
        if CONFIG.mmproj_path and os.path.isfile(CONFIG.mmproj_path):
            main_arch = (read_gguf_meta(path).get("arch") or "").lower()
            mmproj_arch = (read_gguf_meta(CONFIG.mmproj_path).get("arch") or "").lower()
            chat_handler = self._make_vision_chat_handler(
                CONFIG.mmproj_path,
                model_name=os.path.basename(path),
                main_arch=main_arch,
                mmproj_arch=mmproj_arch,
                handler_override=(CONFIG.vision_handler or "").lower(),
            )
            if chat_handler is not None:
                chat_format = None   # the handler owns formatting now
                self.vision_active = True
                # Unwrap the wrapper so the UI reports the actual handler class
                # (e.g. "Qwen25VLChatHandler") instead of "_VisionHandlerWrapper".
                inner = getattr(chat_handler, "_inner", chat_handler)
                self.vision_handler_name = type(inner).__name__
                _vlog(f"[vision] passing handler to Llama(): "
                    f"type={type(chat_handler).__name__}, "
                    f"id={id(chat_handler)}\n"
                )
        n_ctx = max(512, int(CONFIG.n_ctx) if CONFIG.n_ctx else DEFAULT_N_CTX)
        n_threads = max(1, (os.cpu_count() or 4) // 2)

        # Wrap the user callback to accept any signature variant that newer
        # llama-cpp-python wheels might use ((float) -> bool vs (float, Any) -> bool).
        def _progress_cb(progress, *_args, **_kwargs):  # noqa: ANN001
            try:
                if on_progress is not None:
                    on_progress(float(progress))
            except Exception:  # noqa: BLE001 — callback errors must never propagate into C
                pass
            return True

        captured: Dict[str, str] = {"text": ""}
        try:
            with _capture_native_stderr(captured):
                # `offload_kqv` puts the KV cache on GPU. It's a big win when
                # most/all layers are on GPU (KV reads stay local), but a net
                # negative when most layers are on CPU (CPU layers then pay a
                # PCIe round-trip to fetch KV every token). Auto-pick based on
                # the offload ratio: ON if ≥50% of layers will run on GPU.
                gpu_ratio = (
                    1.0 if n_gpu_layers < 0 or n_gpu_layers >= total_layers
                    else n_gpu_layers / max(1, total_layers)
                )
                offload_kqv = gpu_ratio >= 0.5

                # NOTE: flash_attn left OFF by default. It's a known crash
                # vector with MoE models on certain llama.cpp builds (causes
                # STATUS_ILLEGAL_INSTRUCTION at sample time). Re-enable per
                # model only after verifying it works.
                #
                # flash_attn for vision: collapses the attention compute graph
                # into a single fused kernel, dramatically smaller. Without it,
                # mtmd's M-RoPE-positioned image batches keep triggering
                # backend_ids_changed → "failed to allocate graph, reserving"
                # thrash. Off for text-only because flash_attn has historically
                # been a crash vector on some MoE builds (STATUS_ILLEGAL_-
                # INSTRUCTION at sample time); vision needs it badly enough to
                # accept that risk.
                flash_attn = chat_handler is not None
                self._llm = Llama(
                    model_path=path,
                    n_ctx=n_ctx,
                    n_threads=n_threads,
                    n_batch=512,
                    n_gpu_layers=n_gpu_layers,
                    use_mmap=True,
                    use_mlock=False,
                    flash_attn=flash_attn,
                    offload_kqv=offload_kqv,
                    chat_format=chat_format,
                    chat_handler=chat_handler,
                    verbose=True,
                    progress_callback=_progress_cb if on_progress else None,
                )
            # Install (once-per-process) the log filter that drops the
            # CUDA-graph-compat spam MoE models emit per-token.
            self._install_log_filter()
            # Verify the chat handler we passed actually made it into the
            # Llama instance unchanged.
            try:
                stored = getattr(self._llm, "chat_handler", None)
                _vlog(f"[vision] post-construct check: "
                    f"llama.chat_handler type={type(stored).__name__ if stored else 'None'}, "
                    f"id={id(stored) if stored else 0}, "
                    f"llama.chat_format={self._llm.chat_format!r}\n"
                )
            except Exception as e:  # noqa: BLE001
                _vlog(f"[vision] post-construct check failed: {e}\n")
            # Stash a reference to our abort flag on the Llama instance so the
            # vision sampler (_sample_from_current_state, which only gets the
            # Llama, not the engine) can check it on every token.
            try:
                self._llm._gen_abort = self._abort  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        except BaseException as e:  # noqa: BLE001 — covers native crashes
            tail = _tail(captured.get("text", ""), lines=40)
            ram_line = (
                f"RAM avail: {avail_gb:.1f} GB"
                + (f" / {total_gb:.1f} GB total\n" if total_gb else "\n")
                if avail_gb is not None else ""
            )
            log_block = (
                f"--- llama.cpp log (last lines before crash) ---\n"
                f"{tail or '(no output captured — crash happened before any log line)'}\n"
                f"-----------------------------------------------\n\n"
            )
            raise RuntimeError(
                "Failed to load model: " + str(e) + "\n\n"
                f"File:      {path}\n"
                f"Size:      {size_gb:.1f} GB\n"
                f"n_ctx:     {n_ctx}\n"
                + ram_line +
                f"llama-cpp-python: {lcp_version}\n\n"
                + log_block +
                "Most likely causes:\n"
                "  • Allocation failed (not enough RAM for weights + KV "
                "cache + compute buffer). Try a smaller quant, smaller "
                "n_ctx, or close other apps.\n"
                "  • Model architecture not built into your installed "
                f"llama-cpp-python {lcp_version}. Upgrade:\n"
                "       pip install --upgrade --force-reinstall "
                "llama-cpp-python\n"
                "  • Truncated / corrupt file."
            ) from e
        finally:
            # Also forward the captured log to our own stderr so it lives in
            # the dev console / logs.
            text = captured.get("text", "")
            if text:
                sys.stderr.write(text)

        # Warm-up — JIT-compile CUDA kernels now (while the user is still
        # watching the loading bar) so their first real chat is fast.
        # Outside the FD capture so any crash here surfaces normally.
        # Non-fatal: a warm-up failure doesn't break the engine.
        if on_progress is not None:
            on_progress(1.0)
        try:
            self._llm.create_completion(
                prompt=" ", max_tokens=1, temperature=0.0, stream=False,
            )
        except BaseException as e:  # noqa: BLE001
            sys.stderr.write(
                f"[warm-up failed, non-fatal] {type(e).__name__}: {e}\n"
            )

        self._path = path

    def _install_log_filter(self) -> None:
        """Install the per-process log filter once. Safe to call repeatedly."""
        _install_log_filter_once()

    def _discover_think_token_ids(self) -> List[int]:
        """Return the token IDs that open a Qwen-style `<think>` block.

        Different model tokenizers encode this differently — some have a
        single special token (`<think>`), others split it into BPE pieces.
        We try the most common encodings and return whatever the model
        actually has, deduplicated. If nothing's found the Quick-mode
        suppressor becomes a no-op (which is fine — we still fall back
        to the soft `/no_think` marker).
        """
        candidates = ["<think>", "<|think_start|>", "<|think|>"]
        ids: list[int] = []
        seen: set[int] = set()
        for s in candidates:
            try:
                toks = self._llm.tokenize(s.encode("utf-8"), add_bos=False, special=True)
                # We want the FIRST distinctive token of the sequence —
                # banning that prevents the block from opening.
                if toks and toks[0] not in seen:
                    seen.add(toks[0])
                    ids.append(toks[0])
            except Exception:  # noqa: BLE001
                continue
        return ids

    def abort(self) -> None:
        """Signal the generation loop to stop ASAP.

        Sets a thread-safe flag the logits_processor (and our custom vision
        sampler) check on every sample step. The current `create_chat_completion`
        call raises out of the C-level loop, the producer thread exits, llama.cpp
        becomes available for the next request. Without this, clicking Stop
        only stops the renderer from RECEIVING tokens — llama.cpp keeps
        running and the next prompt hangs because llama.cpp isn't reentrant.
        """
        self._abort.set()

    # Map of override slug → ChatHandler class name. Used both by manual
    # override (Settings → Model → Vision handler) and by the auto-detect
    # fallback below.
    _HANDLER_BY_SLUG: dict[str, str] = {
        "qwen25vl":       "Qwen25VLChatHandler",
        "qwenvl":         "Qwen25VLChatHandler",
        "llama32vision":  "Llama32VisionChatHandler",
        "mllama":         "Llama32VisionChatHandler",
        "minicpmv":       "MiniCPMv26ChatHandler",
        "minicpmv26":     "MiniCPMv26ChatHandler",
        "moondream":      "MoondreamChatHandler",
        "llava16":        "Llava16ChatHandler",
        "llava15":        "Llava15ChatHandler",
        "nanollava":      "NanoLlavaChatHandler",
        "obsidian":       "ObsidianChatHandler",
    }

    @staticmethod
    def _make_vision_chat_handler(
        mmproj_path: str,
        model_name: str,
        main_arch: str = "",
        mmproj_arch: str = "",
        handler_override: str = "",
    ):
        """Construct the right multimodal ChatHandler for the given model.

        Detection order (most → least reliable):
          1. Manual override from Settings → Model → Vision handler (slug
             like "qwen25vl"). Lets the user fix mis-detected pairs.
          2. mmproj's own GGUF arch — most reliable signal when the projector
             reports its target family (e.g. "qwen2vl", "mllama").
          3. Main model's GGUF arch.
          4. Filename keyword match (main weights + mmproj filename).

        Returns None if no family matches — we won't guess, because the wrong
        handler feeds garbage features into the LLM.
        """
        try:
            import llama_cpp.llama_chat_format as fmt  # type: ignore
        except Exception:  # noqa: BLE001
            return None

        mmproj_name = os.path.basename(mmproj_path).lower().replace(".gguf", "")

        # 1. Manual override wins.
        if handler_override and handler_override != "auto":
            cls_name = LlamaCppEngine._HANDLER_BY_SLUG.get(handler_override)
            if cls_name and hasattr(fmt, cls_name):
                try:
                    cls = getattr(fmt, cls_name)
                    inner = cls(clip_model_path=mmproj_path, verbose=True)
                    _vlog(f"[vision] using {cls_name} "
                        f"(manual override = {handler_override}) "
                        f"with {os.path.basename(mmproj_path)}\n"
                    )
                    wrapped = _VisionHandlerWrapper(inner)
                    _vlog(
                        f"[vision] wrapper installed (override path): "
                        f"type={type(wrapped).__name__}, "
                        f"callable={callable(wrapped)}\n"
                    )
                    return wrapped
                except Exception as e:  # noqa: BLE001
                    _vlog(f"[vision] manual override {handler_override} failed to init: {e}\n"
                    )
                    return None
            _vlog(f"[vision] override '{handler_override}' is not a known handler slug; "
                "falling back to auto-detect.\n"
            )

        # 2 + 3 + 4. Family signature → handler class name. Matched as substrings
        # against the mmproj arch, main arch, mmproj filename, and main filename.
        FAMILIES: list[tuple[tuple[str, ...], str]] = [
            # ((substring-1, substring-2, ...), handler)
            # The tuple is treated as a conjunction — ALL substrings must be
            # present in the haystack.
            (("qwen", "vl"),       "Qwen25VLChatHandler"),   # qwen2-vl, qwen2.5-vl, qwen3-vl, ...
            (("mllama",),          "Llama32VisionChatHandler"),   # arch name used by Llama-3.2-Vision
            (("llama", "vision"),  "Llama32VisionChatHandler"),
            (("minicpm", "v"),     "MiniCPMv26ChatHandler"),  # minicpm-v, minicpmv26, etc.
            (("moondream",),       "MoondreamChatHandler"),
            (("llava", "1.6"),     "Llava16ChatHandler"),
            (("llava", "16"),      "Llava16ChatHandler"),     # llava1_6, llava16
            (("nanollava",),       "NanoLlavaChatHandler"),
            (("obsidian",),        "ObsidianChatHandler"),
            (("llava",),           "Llava15ChatHandler"),     # generic llava → 1.5
        ]

        # Most-trusted signals first. The mmproj's own arch is the most
        # reliable because it's set by whoever built the projector.
        haystacks = [
            ("mmproj-arch", mmproj_arch.lower()),
            ("main-arch",   main_arch.lower()),
            ("mmproj-name", mmproj_name),
            ("main-name",   model_name.lower().replace(".gguf", "")),
        ]

        chosen_cls = None
        matched_via = ""
        for substrings, cls_name in FAMILIES:
            if not hasattr(fmt, cls_name):
                continue
            for source, hay in haystacks:
                if hay and all(s in hay for s in substrings):
                    chosen_cls = getattr(fmt, cls_name)
                    matched_via = source
                    break
            if chosen_cls is not None:
                break

        # No silent fallback. If nothing matches, the projector is probably
        # incompatible — loading the wrong handler (e.g. LLaVA-1.5 CLIP-L
        # into a Qwen-VL backbone) feeds noise into the LLM. Refuse instead.
        if chosen_cls is None:
            _vlog("[vision] paired mmproj does not match any known vision family.\n"
                f"          main arch     = {main_arch or '(missing)'}\n"
                f"          mmproj arch   = {mmproj_arch or '(missing)'}\n"
                f"          main file     = {model_name}\n"
                f"          mmproj file   = {os.path.basename(mmproj_path)}\n"
                "          If you know this pair works, set "
                "Settings → Model → Vision handler to the right family "
                "(e.g. 'qwen25vl', 'llama32vision', 'minicpmv', 'llava16').\n"
            )
            return None
        try:
            inner = chosen_cls(clip_model_path=mmproj_path, verbose=True)
            _vlog(f"[vision] using {chosen_cls.__name__} "
                f"(matched via {matched_via}, "
                f"main_arch={main_arch or '?'}, mmproj_arch={mmproj_arch or '?'}) "
                f"with {os.path.basename(mmproj_path)}\n"
            )
            # Wrap the handler. After mtmd_helper_eval_chunk_single injects
            # image embeddings into the KV cache, the handler calls
            # Llama.create_completion(prompt=input_ids[:n_tokens].tolist()).
            # Inside Llama.generate, a prefix-match loop walks _input_ids vs
            # the prompt to find how much KV state to keep. mtmd doesn't
            # populate _input_ids for image positions, so the loop breaks
            # partway through. Then kv_cache_seq_rm is called to chop the
            # cache there — for vision caches with M-RoPE positions, that
            # returns False ("partial kv removal not supported"), falling
            # back to a full reset that destroys the image embeddings. The
            # model then sees raw <|image_pad|> IDs and outputs `!`.
            #
            # The fix is to intercept the handler's call to create_completion
            # and mirror the prompt list back into _input_ids first, so the
            # prefix-match loop runs cleanly to the end and only removes the
            # tail token (which usually IS supported and works).
            wrapped = _VisionHandlerWrapper(inner)
            _vlog(f"[vision] wrapper installed: type={type(wrapped).__name__}, "
                f"callable={callable(wrapped)}, "
                f"has_dunder_call={hasattr(type(wrapped), '__call__')}\n"
            )
            return wrapped
        except Exception as e:  # noqa: BLE001
            _vlog(f"[vision] handler init failed: {e}\n")
            return None

    @staticmethod
    def _infer_chat_format(path: str) -> str:
        name = os.path.basename(path).lower()
        if "qwen3" in name:
            return "chatml"
        if "qwen2.5" in name or "qwen2" in name:
            return "chatml"
        if "qwen" in name:
            return "chatml"
        if "llama-3" in name or "llama3" in name:
            return "llama-3"
        if "llama-2" in name or "llama2" in name:
            return "llama-2"
        if "mistral" in name:
            return "mistral-instruct"
        if "phi-3" in name or "phi3" in name:
            return "phi-3"
        if "gemma" in name:
            return "gemma"
        return "chatml"

    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | BaseException | None] = asyncio.Queue()

        # Reset the abort flag for this new request. The flag is set by the
        # Stop button (via engine.abort()) and checked on every sample step
        # by the logits_processor below — and by _sample_from_current_state
        # for vision turns.
        self._abort.clear()

        # Logits processors chain:
        #   1. Abort processor — terminates llama.cpp mid-decode when Stop
        #      is clicked. Without it, the C-level loop runs until EOS and
        #      blocks the next prompt.
        #   2. No-think processor (Quick mode only) — suppresses the
        #      <think> special token for the first few sample steps so
        #      the model can't start a reasoning block. Soft markers like
        #      `/no_think` aren't reliable on aggressive fine-tunes; this
        #      forces the issue at the logit level.
        try:
            from llama_cpp import LogitsProcessorList  # type: ignore
            def _abort_processor(_input_ids, logits):  # noqa: ANN001
                if self._abort.is_set():
                    raise _AbortGeneration()
                return logits

            processors = [_abort_processor]

            if CONFIG.thinking_mode == "quick":
                think_ids = self._discover_think_token_ids()
                if think_ids:
                    sys.stderr.write(
                        f"[thinking] Quick mode: suppressing tokens {think_ids} "
                        "for the opening sample steps\n"
                    )
                    state = {"steps_remaining": 6}
                    def _no_think_processor(_input_ids, logits):  # noqa: ANN001
                        if state["steps_remaining"] > 0:
                            for tid in think_ids:
                                try:
                                    logits[tid] = -1e30
                                except Exception:  # noqa: BLE001
                                    pass
                            state["steps_remaining"] -= 1
                        return logits
                    processors.append(_no_think_processor)

            abort_processors = LogitsProcessorList(processors)
        except Exception:  # noqa: BLE001 — older builds may differ
            abort_processors = None

        # If this request includes images, force a clean KV cache. The Qwen-VL
        # handler (and most multimodal handlers) re-tokenize and re-insert
        # image embeddings on every call, but llama.cpp's slot allocator
        # tracks prior turn positions — when image embeddings land where text
        # tokens used to be, find_slot fires "non-consecutive token position"
        # and the image injection bails. The cost (re-eating the system prompt
        # each turn) is unavoidable for correctness with vision turns.
        has_image = any(
            isinstance(m.content, list)
            and any(isinstance(b, dict) and b.get("type") == "image_url" for b in m.content)
            for m in messages
        )
        if has_image and self.vision_active:
            try:
                self._llm.reset()
            except Exception:  # noqa: BLE001
                pass

        def producer() -> None:
            try:
                kwargs: Dict[str, object] = {
                    "messages": [{"role": m.role, "content": m.content} for m in messages],
                    "stream": True,
                    "temperature": CONFIG.temperature,
                }
                if abort_processors is not None:
                    kwargs["logits_processor"] = abort_processors
                for chunk in self._llm.create_chat_completion(**kwargs):
                    delta = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if delta:
                        loop.call_soon_threadsafe(queue.put_nowait, delta)
            except _AbortGeneration:
                # Normal exit — user clicked Stop. No error to surface.
                pass
            except BaseException as e:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        asyncio.create_task(asyncio.to_thread(producer))

        while True:
            item = await queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise RuntimeError(f"Inference failed: {item}") from item
            yield item


def _tail(text: str, lines: int) -> str:
    parts = text.strip().splitlines()
    if len(parts) <= lines:
        return text.strip()
    return "\n".join(parts[-lines:])
