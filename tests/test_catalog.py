"""目录客户端与缓存：全离线，httpx.MockTransport 注入，禁止真打网络。"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import Settings
from app.errors import UpstreamError
from app.retrieval import catalog as catalog_mod
from app.retrieval.catalog import CatalogClient, CatalogIndex, ProductDoc


def _settings(**overrides) -> Settings:
    base = dict(
        internal_secret="test-secret",
        agent_service_base_url="http://catalog.test",
        agent_service_token="agent-token",
        catalog_cache_ttl_seconds=300,
    )
    base.update(overrides)
    return Settings(**base)


def _product(pid: int, title: str = "跑步鞋") -> dict:
    return {
        "id": pid,
        "title": title,
        "description": None,
        "originPrice": 19900,
        "coverColor": None,
        "status": 1,
        "createTime": "",
        "skuCount": 2,
        "minPrice": 12900,
    }


def _envelope(items: list[dict]) -> dict:
    return {"code": 0, "message": "success", "data": items}


class _Pager:
    """按 offset 返回分页数据，并记录每次请求。"""

    def __init__(self, total: int, page_payloads=None) -> None:
        self.total = total
        self.calls: list[dict] = []
        self._payloads = page_payloads

    def __call__(self, request: httpx.Request) -> httpx.Response:
        limit = int(request.url.params["limit"])
        offset = int(request.url.params["offset"])
        self.calls.append({"limit": limit, "offset": offset, "url": str(request.url)})
        if self._payloads is not None:
            return self._payloads(len(self.calls))
        remaining = max(0, self.total - offset)
        count = min(limit, remaining)
        return httpx.Response(200, json=_envelope([_product(offset + i + 1) for i in range(count)]))


def test_single_page_parses_product_doc():
    pager = _Pager(total=3)
    client = CatalogClient(_settings(), transport=httpx.MockTransport(pager))
    docs = asyncio.run(client.fetch_all())
    assert [d.id for d in docs] == [1, 2, 3]
    first = docs[0]
    assert isinstance(first, ProductDoc)
    assert first.title == "跑步鞋"
    assert first.description == ""  # null 归一成空串
    assert first.origin_price == 19900
    assert first.min_price == 12900
    assert first.cover_color is None
    assert first.sku_count == 2
    assert len(pager.calls) == 1


def test_two_pages_are_followed_until_short_page():
    pager = _Pager(total=150)
    client = CatalogClient(_settings(), transport=httpx.MockTransport(pager))
    docs = asyncio.run(client.fetch_all())
    assert len(docs) == 150
    assert [c["offset"] for c in pager.calls] == [0, 100]
    assert all(c["limit"] == 100 for c in pager.calls)


def test_exact_multiple_stops_on_empty_final_page():
    # 恰好 100 条 = 整页，必须再翻一页拿到空数组才知道结束
    pager = _Pager(total=100)
    client = CatalogClient(_settings(), transport=httpx.MockTransport(pager))
    docs = asyncio.run(client.fetch_all())
    assert len(docs) == 100
    assert [c["offset"] for c in pager.calls] == [0, 100]


def test_page_size_is_clamped_into_1_to_100():
    pager = _Pager(total=5)
    client = CatalogClient(_settings(), transport=httpx.MockTransport(pager))
    asyncio.run(client.fetch_all(page_size=500))
    assert pager.calls[0]["limit"] == 100
    pager2 = _Pager(total=5)
    client2 = CatalogClient(_settings(), transport=httpx.MockTransport(pager2))
    asyncio.run(client2.fetch_all(page_size=0))
    assert pager2.calls[0]["limit"] == 1


def test_business_code_non_zero_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 4001, "message": "boom", "data": None})

    client = CatalogClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamError) as exc:
        asyncio.run(client.fetch_all())
    assert "4001" in str(exc.value)


def test_http_500_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = CatalogClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamError) as exc:
        asyncio.run(client.fetch_all())
    assert "500" in str(exc.value)


def test_data_not_array_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 0, "data": {"id": 1}})

    client = CatalogClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamError):
        asyncio.run(client.fetch_all())


def test_empty_token_fails_closed_without_request():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_envelope([]))

    client = CatalogClient(_settings(agent_service_token=""), transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamError) as exc:
        asyncio.run(client.fetch_all())
    assert "AGENT_SERVICE_TOKEN" in str(exc.value)
    assert calls == []  # 一个请求都没发


def test_token_is_sent_as_header():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_envelope([_product(1)]))

    client = CatalogClient(_settings(), transport=httpx.MockTransport(handler))
    asyncio.run(client.fetch_all())
    assert seen[0].headers["X-Agent-Token"] == "agent-token"
    assert str(seen[0].url).startswith("http://catalog.test/api/agent/products")


def test_cache_hit_fetches_only_once():
    pager = _Pager(total=2)
    index = CatalogIndex(
        CatalogClient(_settings(), transport=httpx.MockTransport(pager)), ttl_seconds=300
    )

    async def scenario():
        first = await index.docs()
        second = await index.docs()
        return first, second

    first, second = asyncio.run(scenario())
    assert first is second  # 同一份缓存对象
    assert len(pager.calls) == 1
    assert index.indexed_count == 2
    assert index.last_refreshed_at is not None


def test_ttl_expiry_refetches(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(catalog_mod, "_now", lambda: clock["now"])
    pager = _Pager(total=2)
    index = CatalogIndex(
        CatalogClient(_settings(), transport=httpx.MockTransport(pager)), ttl_seconds=60
    )

    async def scenario():
        first = await index.docs()
        clock["now"] += 59.0
        still_cached = await index.docs()
        clock["now"] += 2.0  # 61s > 60s TTL
        refreshed = await index.docs()
        return first, still_cached, refreshed

    first, still_cached, refreshed = asyncio.run(scenario())
    assert len(pager.calls) == 2
    assert first is still_cached
    assert refreshed is not first


def test_failed_fetch_does_not_poison_cache():
    state = {"fail": True}
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if state["fail"]:
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json=_envelope([_product(1)]))

    index = CatalogIndex(
        CatalogClient(_settings(), transport=httpx.MockTransport(handler)), ttl_seconds=300
    )

    async def scenario():
        with pytest.raises(UpstreamError):
            await index.docs()
        assert index.indexed_count == 0  # 失败没写进缓存
        state["fail"] = False
        return await index.docs()

    docs = asyncio.run(scenario())
    assert [d.id for d in docs] == [1]  # 再次调用是重试，不是返回空
    assert calls["n"] == 2


def test_empty_upstream_result_is_cached_as_empty():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_envelope([]))

    index = CatalogIndex(
        CatalogClient(_settings(), transport=httpx.MockTransport(handler)), ttl_seconds=300
    )

    async def scenario():
        await index.docs()
        await index.docs()

    asyncio.run(scenario())
    assert calls["n"] == 1
    assert index.indexed_count == 0


def test_concurrent_calls_fetch_only_once():
    pager = _Pager(total=2)
    index = CatalogIndex(
        CatalogClient(_settings(), transport=httpx.MockTransport(pager)), ttl_seconds=300
    )

    async def scenario():
        return await asyncio.gather(*(index.docs() for _ in range(5)))

    results = asyncio.run(scenario())
    assert len(pager.calls) == 1
    assert all(r is results[0] for r in results)


def test_aclose_returns_none():
    client = CatalogClient(_settings())
    assert asyncio.run(client.aclose()) is None
