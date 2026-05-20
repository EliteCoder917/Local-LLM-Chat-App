"""Runtime configuration shared across backend modules.

Settings flow from the Electron renderer over the WebSocket (`settings-update`
events) into this module. Tools, the agent loop, and LLM engines all read from
the singleton `CONFIG`.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Config:
    workspace: str = ""
    engine: str = "llama-cpp"       # "llama-cpp" | "ollama"
    model: str = "local-gguf"
    mmproj_path: str = ""           # paired multimodal projector for vision models
    vision_handler: str = ""        # manual handler family override; "" = auto-detect
    thinking_mode: str = "smart"    # "smart" | "quick" | "deep" — Qwen-style reasoning toggle
    ollama_url: str = "http://127.0.0.1:11434"
    model_path: str = ""
    temperature: float = 0.7
    n_ctx: int = 4096           # context window for llama-cpp engine
    gpu_offload_gb: float = -1.0  # -1 = Auto; engine picks max fittable at load
    system_prompt: str = (
        "You are a helpful local AI assistant. Be concise and accurate."
    )
    agent_mode: bool = True
    max_iterations: int = 10

    permissions: Dict[str, bool] = field(default_factory=lambda: {
        "file.read": True,
        "file.write": False,
        "file.delete": False,
        "exec.python": False,
        "exec.shell": False,
        "exec.script": False,
        "network": False,
        "memory": False,
    })

    # Fields that must NEVER be silently blanked by a partial update —
    # if the incoming value is an empty string, keep the existing one.
    _PRESERVE_IF_EMPTY = frozenset({"model_path", "mmproj_path"})

    def update(self, patch: Dict[str, Any]) -> None:
        for k, v in patch.items():
            key = {
                "ollamaUrl": "ollama_url",
                "modelPath": "model_path",
                "mmprojPath": "mmproj_path",
                "systemPrompt": "system_prompt",
                "agentMode": "agent_mode",
                "maxIterations": "max_iterations",
                "nCtx": "n_ctx",
                "gpuOffloadGb": "gpu_offload_gb",
                "visionHandler": "vision_handler",
                "thinkingMode": "thinking_mode",
            }.get(k, k)
            if not hasattr(self, key):
                continue
            if key in self._PRESERVE_IF_EMPTY and v == "" and getattr(self, key):
                # Don't let a stale frontend settings object clobber a path
                # that was set out-of-band (e.g. by /library/select).
                continue
            setattr(self, key, v)

    def update_permissions(self, perms: Dict[str, bool]) -> None:
        self.permissions.update(perms)


CONFIG = Config()
LOCK = threading.RLock()


def data_dir() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    p = os.path.join(base, "LocalAIIDE")
    os.makedirs(p, exist_ok=True)
    return p
