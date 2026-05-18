"""Streaming Ollama client. Talks to a local Ollama server on 11434."""
from __future__ import annotations

import json
from typing import AsyncIterator, List

import httpx

from ..config import CONFIG
from .engine import ChatMessage, LLMEngine


class OllamaEngine(LLMEngine):
    async def stream(self, messages: List[ChatMessage]) -> AsyncIterator[str]:
        url = CONFIG.ollama_url.rstrip("/") + "/api/chat"
        payload = {
            "model": CONFIG.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "options": {"temperature": CONFIG.temperature},
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=payload) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    raise RuntimeError(
                        f"Ollama HTTP {r.status_code}: {body.decode(errors='replace')}"
                    )
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = obj.get("message") or {}
                    chunk = msg.get("content") or ""
                    if chunk:
                        yield chunk
                    if obj.get("done"):
                        return
