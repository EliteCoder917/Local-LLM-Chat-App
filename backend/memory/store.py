"""Persistent memory backed by SQLite.

Stores:
- key/value blobs (preferences, long-term notes, project metadata)
- conversation history (one row per message)

The DB file lives in `%LOCALAPPDATA%\\LocalAIIDE\\memory.sqlite`.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, List, Optional

from ..config import data_dir


class MemoryStore:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.path.join(data_dir(), "memory.sqlite")
        self._lock = threading.RLock()
        self._init()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def _init(self) -> None:
        with self._lock, self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    ts REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conv
                    ON messages(conversation_id, ts);
                """
            )

    # ---- key/value ----
    def set(self, key: str, value: Any) -> None:
        payload = json.dumps(value)
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO kv(key, value, updated_at) VALUES(?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (key, payload, time.time()),
            )

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
            if not row:
                return default
            try:
                return json.loads(row["value"])
            except json.JSONDecodeError:
                return default

    def delete(self, key: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM kv WHERE key=?", (key,))

    def keys(self) -> List[str]:
        with self._lock, self._conn() as c:
            return [r["key"] for r in c.execute("SELECT key FROM kv ORDER BY key")]

    def list_all(self) -> List[dict]:
        """For the UI viewer: every entry with its decoded value + updated_at."""
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT key, value, updated_at FROM kv ORDER BY updated_at DESC"
            ).fetchall()
            out: List[dict] = []
            for r in rows:
                try:
                    v = json.loads(r["value"])
                except json.JSONDecodeError:
                    v = r["value"]
                out.append({"key": r["key"], "value": v, "updatedAt": r["updated_at"]})
            return out

    def clear(self) -> int:
        with self._lock, self._conn() as c:
            cur = c.execute("DELETE FROM kv")
            return cur.rowcount

    # ---- conversation history ----
    def append_message(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO messages(conversation_id, role, content, ts) "
                "VALUES(?, ?, ?, ?)",
                (conversation_id, role, content, time.time()),
            )

    def history(self, conversation_id: str, limit: int = 200) -> List[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT role, content, ts FROM messages "
                "WHERE conversation_id=? ORDER BY ts ASC LIMIT ?",
                (conversation_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_conversation(self, conversation_id: str) -> None:
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))


memory_store = MemoryStore()
