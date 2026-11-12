"""Redis 会话存储：整段 JSON 序列化一个 list，带 TTL。

一个坏 key（被人手改过 / 旧格式）只降级成"空历史 + WARNING"，不能让整个服务挂掉。
"""

from __future__ import annotations

import json
import logging

from app.session.base import Message, session_key, trim_window

log = logging.getLogger(__name__)


class RedisSessionStore:
    backend = "redis"

    def __init__(self, client, window: int = 20, ttl_days: int = 7) -> None:
        self._client = client
        self._window = window
        self._ttl_seconds = ttl_days * 86400

    async def ping(self) -> bool:
        return bool(await self._client.ping())

    async def load(self, user_id: int, session_id: str) -> list[Message]:
        key = session_key(user_id, session_id)
        raw = await self._client.get(key)
        if not raw:
            return []
        try:
            decoded = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            log.warning("会话反序列化失败，按空历史降级: key=%s error=%s", key, exc)
            return []
        if not isinstance(decoded, list) or not all(isinstance(i, dict) for i in decoded):
            log.warning("会话内容结构不符，按空历史降级: key=%s type=%s", key, type(decoded).__name__)
            return []
        return [dict(item) for item in decoded]

    async def append(self, user_id: int, session_id: str, messages: list[Message]) -> None:
        if not messages:
            return
        key = session_key(user_id, session_id)
        history = await self.load(user_id, session_id)
        merged = history + [dict(item) for item in messages]
        payload = json.dumps(trim_window(merged, self._window), ensure_ascii=False)
        await self._client.set(key, payload, ex=self._ttl_seconds)

    async def clear(self, user_id: int) -> None:
        # 契约里 clear 只按用户清；session_id 未知，用 SCAN 前缀删除
        prefix = f"tradepilot:session:{user_id}:"
        keys = [key async for key in self._client.scan_iter(match=f"{prefix}*")]
        if keys:
            await self._client.delete(*keys)
