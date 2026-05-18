"""Memory tools — expose the persistent KV store to the model."""
from __future__ import annotations

from typing import Any

from ..memory import memory_store


def get_memory(key: str) -> Any:
    val = memory_store.get(key, default=None)
    return val if val is not None else f"(no value for '{key}')"


def set_memory(key: str, value: Any) -> str:
    memory_store.set(key, value)
    return f"Stored '{key}'"


def list_memory() -> list:
    return memory_store.keys()


def delete_memory(key: str) -> str:
    memory_store.delete(key)
    return f"Deleted '{key}'"
