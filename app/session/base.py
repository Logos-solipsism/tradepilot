"""会话存储抽象：三个异步方法，memory / redis 两个后端都要满足。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

Message = dict[str, str]


def session_key(user_id: int, session_id: str) -> str:
    return f"tradepilot:session:{user_id}:{session_id}"


def trim_window(messages: list[Message], window: int) -> list[Message]:
    """只保留最近 window 条。

    理由是主动控制 token 成本：历史无限增长会让每一轮都变贵、变慢，
    并且早期内容对当前问题几乎没有边际价值。
    """
    if window <= 0:
        return []
    if len(messages) <= window:
        return [dict(item) for item in messages]
    return [dict(item) for item in messages[-window:]]


@runtime_checkable
class SessionStore(Protocol):
    """会话存储：load / append / clear 全部是异步的。"""

    backend: str

    async def load(self, user_id: int, session_id: str) -> list[Message]:
        ...

    async def append(self, user_id: int, session_id: str, messages: list[Message]) -> None:
        ...

    async def clear(self, user_id: int) -> None:
        ...
