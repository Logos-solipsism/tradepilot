"""进程内会话存储（默认后端，仅供开发/单实例）。"""

from __future__ import annotations

import asyncio
import time

from app.session.base import Message, session_key, trim_window


class MemorySessionStore:
    """dict + asyncio.Lock。

    进程重启即失忆，多 worker 之间互相看不见——生产请切 SESSION_BACKEND=redis。
    """

    backend = "memory"

    def __init__(self, window: int = 20, ttl_days: int = 7) -> None:
        self._window = window
        self._ttl_seconds = ttl_days * 86400
        self._store: dict[str, list[Message]] = {}
        self._expires_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def load(self, user_id: int, session_id: str) -> list[Message]:
        key = session_key(user_id, session_id)
        async with self._lock:
            self._drop_if_expired(key)
            stored = self._store.get(key)
            if not stored:
                return []
            return [dict(item) for item in stored]

    async def append(self, user_id: int, session_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        key = session_key(user_id, session_id)
        async with self._lock:
            self._drop_if_expired(key)
            merged = list(self._store.get(key, [])) + [dict(item) for item in messages]
            self._store[key] = trim_window(merged, self._window)
            self._expires_at[key] = time.time() + self._ttl_seconds

    async def clear(self, user_id: int) -> None:
        prefix = f"tradepilot:session:{user_id}:"
        async with self._lock:
            for key in [k for k in self._store if k.startswith(prefix)]:
                self._store.pop(key, None)
                self._expires_at.pop(key, None)

    def _drop_if_expired(self, key: str) -> None:
        deadline = self._expires_at.get(key)
        if deadline is not None and deadline <= time.time():
            self._store.pop(key, None)
            self._expires_at.pop(key, None)
