"""内部票据（Java -> Python 的唯一信任凭据）。

格式与 Java 侧 InternalTicketSigner 逐字一致：
    {userId}.{ts}.{nonce}.{hexHmacSha256}
签名原文：userId + "\n" + ts + "\n" + nonce（三段用换行连接，末尾没有换行）。
算法：HMAC-SHA256(INTERNAL_SECRET, 原文)，输出小写 hex，64 字符。

安全约定：
- INTERNAL_SECRET 为空时 fail-closed：签不出、验不过，所有请求一律 401。
- 签名比较必须常数时间（hmac.compare_digest），逐字符提前返回会泄露签名前缀。
- nonce 防重放：本模块的字典是**单进程内存实现**——多 worker 部署时各进程各自独立，
  不构成跨进程防重放；真要做分布式去重需要 Redis 之类的共享存储（后续步骤再说）。
"""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from threading import Lock

from app.config import get_settings

MAX_CLOCK_SKEW_SECONDS = 300
NONCE_RETENTION_SECONDS = 300
_HEX_DIGEST_LENGTH = 64


class TicketError(Exception):
    """票据缺失/格式错/过期/签名不符/nonce 重放。调用方一律回 401。"""


class NonceCache:
    """单进程 nonce 去重：保留 NONCE_RETENTION_SECONDS，过期即回收。

    多 worker / 多实例部署时每个进程各有一份，跨进程防重放是不成立的。
    """

    def __init__(self, retention_seconds: int = NONCE_RETENTION_SECONDS) -> None:
        self._retention = retention_seconds
        self._seen: dict[str, float] = {}
        self._lock = Lock()

    def consume(self, nonce: str, now: float, ttl_seconds: int = NONCE_RETENTION_SECONDS) -> None:
        """登记一次 nonce；已存在则抛 TicketError（重放）。"""
        with self._lock:
            self._prune(now)
            if nonce in self._seen:
                raise TicketError("nonce 重放")
            self._seen[nonce] = now + max(ttl_seconds, self._retention)

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()

    def __len__(self) -> int:
        return len(self._seen)

    def _prune(self, now: float) -> None:
        expired = [key for key, deadline in self._seen.items() if deadline <= now]
        for key in expired:
            del self._seen[key]


_default_nonces = NonceCache()


def reset_nonce_cache() -> None:
    """只给测试用：清空进程级 nonce 记录。"""
    _default_nonces.clear()


def _resolve_secret(secret: str | None) -> str:
    resolved = (secret if secret is not None else get_settings().internal_secret).strip()
    if not resolved:
        # fail-closed：宁可全量失败，也不静默降级成"不签名的裸请求"
        raise TicketError("INTERNAL_SECRET 未配置")
    return resolved


def _payload(user_id: str, ts: int, nonce: str) -> str:
    return f"{user_id}\n{ts}\n{nonce}"


def _hmac_hex(secret: str, payload: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def sign(
    user_id: int,
    *,
    secret: str | None = None,
    ts: int | None = None,
    nonce: str | None = None,
) -> str:
    """签发一张票据。secret 为 None 时取环境变量 INTERNAL_SECRET。"""
    resolved = _resolve_secret(secret)
    epoch_second = int(time.time()) if ts is None else int(ts)
    token = nonce if nonce is not None else uuid.uuid4().hex
    digest = _hmac_hex(resolved, _payload(str(int(user_id)), epoch_second, token))
    return f"{int(user_id)}.{epoch_second}.{token}.{digest}"


def verify(
    ticket: str | None,
    *,
    secret: str | None = None,
    now: int | None = None,
    nonces: NonceCache | None = None,
) -> int:
    """校验票据，返回 userId。任何不合格的情况都抛 TicketError。"""
    resolved = _resolve_secret(secret)
    if not ticket:
        raise TicketError("票据缺失")
    parts = ticket.split(".")
    if len(parts) != 4:
        raise TicketError("票据格式错误")
    user_id_text, ts_text, nonce, signature = parts
    if not user_id_text.isdigit():
        raise TicketError("票据 userId 非法")
    try:
        epoch_second = int(ts_text)
    except ValueError as exc:
        raise TicketError("票据时间戳非法") from exc
    current = int(time.time()) if now is None else int(now)
    if abs(current - epoch_second) > MAX_CLOCK_SKEW_SECONDS:
        raise TicketError("票据已过期或时间戳越界")
    if len(signature) != _HEX_DIGEST_LENGTH:
        raise TicketError("票据签名长度非法")
    expected = _hmac_hex(resolved, _payload(user_id_text, epoch_second, nonce))
    # 常数时间比较：不要换成 ==
    if not hmac.compare_digest(expected, signature):
        raise TicketError("票据签名不匹配")
    cache = _default_nonces if nonces is None else nonces
    cache.consume(nonce, float(current), MAX_CLOCK_SKEW_SECONDS)
    return int(user_id_text)
