"""LLM engine abstraction.

Lifecycle is owned by `loader.ModelLoader` — call `loader.load()` to construct
an engine, `loader.get()` to retrieve it, `loader.unload()` to drop it. The
cache/swap logic that used to live here is now in loader.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Union


# Content may be a plain string (text-only message) OR an OpenAI-style list
# of typed blocks `[{type: "text", text: ...}, {type: "image_url", image_url: ...}]`
# when a vision handler is active.
@dataclass
class ChatMessage:
    role: str
    content: Union[str, List[dict]]


class LLMEngine(ABC):
    @abstractmethod
    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        """Yield content deltas."""
        ...
