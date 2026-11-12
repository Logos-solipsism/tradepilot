"""内存会话存储：写入读回、窗口裁剪、按用户清理。"""

from __future__ import annotations

import asyncio

from app.session.memory import MemorySessionStore


def test_append_then_load_roundtrip():
    store = MemorySessionStore(window=20)

    async def run():
        await store.append(1, "s1", [{"role": "user", "content": "hi"}])
        await store.append(1, "s1", [{"role": "assistant", "content": "yo"}])
        return await store.load(1, "s1")

    assert asyncio.run(run()) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_window_keeps_only_last_20_messages():
    store = MemorySessionStore(window=20)
    batch = [{"role": "user", "content": f"m{i}"} for i in range(25)]

    async def run():
        await store.append(2, "s2", batch)
        return await store.load(2, "s2")

    loaded = asyncio.run(run())
    assert len(loaded) == 20
    assert loaded[0]["content"] == "m5"
    assert loaded[-1]["content"] == "m24"


def test_clear_empties_user_sessions():
    store = MemorySessionStore()

    async def run():
        await store.append(3, "a", [{"role": "user", "content": "x"}])
        await store.append(3, "b", [{"role": "user", "content": "y"}])
        await store.clear(3)
        return (await store.load(3, "a"), await store.load(3, "b"))

    assert asyncio.run(run()) == ([], [])


def test_sessions_are_isolated_by_user():
    store = MemorySessionStore()

    async def run():
        await store.append(4, "same", [{"role": "user", "content": "u4"}])
        await store.append(5, "same", [{"role": "user", "content": "u5"}])
        return (await store.load(4, "same"), await store.load(5, "same"))

    a, b = asyncio.run(run())
    assert a == [{"role": "user", "content": "u4"}]
    assert b == [{"role": "user", "content": "u5"}]
