"""fetch_detail：全离线，httpx.MockTransport 注入，禁止真打网络。"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.config import Settings
from app.errors import UpstreamError
from app.retrieval.catalog import CatalogClient, ProductDetail, SkuDoc


def _settings(**overrides) -> Settings:
    base = dict(
        internal_secret="test-secret",
        agent_service_base_url="http://catalog.test",
        agent_service_token="agent-token",
    )
    base.update(overrides)
    return Settings(**base)


def _detail_payload(product_id: int = 7) -> dict:
    return {
        "id": product_id,
        "title": "轻量跑步鞋",
        "description": "适合日常慢跑",
        "originPrice": 39900,
        "coverColor": "#333333",
        "status": 1,
        "createTime": "",
        "skus": [
            {"skuId": 1, "skuCode": "RUN-40", "specDesc": "40 码 黑色", "price": 12900, "stock": 5},
            {"skuId": 2, "skuCode": "RUN-41", "specDesc": "41 码 黑色", "price": 13900, "stock": 0},
        ],
    }


def _client(handler, settings: Settings | None = None) -> CatalogClient:
    return CatalogClient(
        settings or _settings(), transport=httpx.MockTransport(handler)
    )


def test_fetch_detail_parses_skus():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"code": 0, "message": "success", "data": _detail_payload()})

    detail = asyncio.run(_client(handler).fetch_detail(7))

    assert detail == ProductDetail(
        id=7,
        title="轻量跑步鞋",
        description="适合日常慢跑",
        origin_price=39900,
        cover_color="#333333",
        skus=(
            SkuDoc(sku_id=1, sku_code="RUN-40", spec_desc="40 码 黑色", price=12900, stock=5),
            SkuDoc(sku_id=2, sku_code="RUN-41", spec_desc="41 码 黑色", price=13900, stock=0),
        ),
    )
    assert str(seen[0].url) == "http://catalog.test/api/agent/products/7"
    assert seen[0].headers["X-Agent-Token"] == "agent-token"


def test_fetch_detail_returns_none_on_business_code_400():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 400, "message": "商品不存在", "data": None})

    # 商品不存在/已下架是正常业务结果，不是故障 —— 不许改成抛异常
    assert asyncio.run(_client(handler).fetch_detail(999)) is None


def test_fetch_detail_non_2xx_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"code": 0, "data": {}})

    with pytest.raises(UpstreamError) as err:
        asyncio.run(_client(handler).fetch_detail(7))
    assert "商品详情" in err.value.message and "500" in err.value.message


def test_fetch_detail_without_token_fails_closed():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"code": 0, "data": _detail_payload()})

    client = _client(handler, _settings(agent_service_token=""))
    with pytest.raises(UpstreamError) as err:
        asyncio.run(client.fetch_detail(7))
    assert "AGENT_SERVICE_TOKEN 未配置" in err.value.message
    assert calls == []


def test_fetch_detail_non_object_data_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 0, "data": [_detail_payload()]})

    with pytest.raises(UpstreamError) as err:
        asyncio.run(_client(handler).fetch_detail(7))
    assert "data 不是对象" in err.value.message


def test_fetch_detail_non_json_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>oops</html>")

    with pytest.raises(UpstreamError) as err:
        asyncio.run(_client(handler).fetch_detail(7))
    assert "不是合法 JSON" in err.value.message


def test_fetch_detail_other_business_code_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": 500, "message": "boom"})

    with pytest.raises(UpstreamError) as err:
        asyncio.run(_client(handler).fetch_detail(7))
    assert "code=500" in err.value.message
