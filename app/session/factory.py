"""会话后端工厂：SESSION_BACKEND=memory|redis。"""

from __future__ import annotations

import logging

from app.config import Settings
from app.session.base import SessionStore
from app.session.memory import MemorySessionStore

log = logging.getLogger(__name__)

MEMORY_WARNING = (
    "会话存在进程内存里，进程重启即失忆；多 worker 之间互相看不见。"
    "生产请切 SESSION_BACKEND=redis"
)


def build_memory_store(settings: Settings) -> MemorySessionStore:
    log.warning(MEMORY_WARNING)
    return MemorySessionStore(window=settings.session_window, ttl_days=settings.session_ttl_days)


async def create_session_store(settings: Settings) -> SessionStore:
    """按配置建后端。选 redis 但连不上时明确告警并降级为内存，绝不静默假装成功。"""
    if settings.session_backend != "redis":
        return build_memory_store(settings)

    from redis.asyncio import Redis

    from app.session.redis_store import RedisSessionStore

    client = Redis.from_url(settings.redis_url, decode_responses=False)
    store = RedisSessionStore(
        client, window=settings.session_window, ttl_days=settings.session_ttl_days
    )
    try:
        await store.ping()
    except Exception as exc:  # noqa: BLE001 - 任何连接失败都只能降级
        log.warning("Redis 不可用，会话无法持久化，已降级为进程内存: %s", exc)
        try:
            await client.aclose()
        except Exception:  # noqa: BLE001,S110 - 关不上就算了
            pass
        return build_memory_store(settings)
    log.info("会话后端: redis (%s)", settings.redis_url)
    return store
