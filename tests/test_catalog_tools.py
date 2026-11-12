"""商品工具层：全离线，索引与上游都用桩，禁止真打网络。"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from app.errors import UpstreamError
from app.retrieval.catalog import ProductDetail, ProductDoc, SkuDoc
from app.tools import catalog_tools
from app.tools.catalog_tools import (
    MAX_TOOL_RESULT_CHARS,
    TOOL_SPECS,
    TRUNCATED_SUFFIX,
    CatalogToolbox,
)


def _doc(doc_id: int, title: str, min_price=12900, sku=2) -> ProductDoc:
    return ProductDoc(
        id=doc_id,
        title=title,
        description="适合日常慢跑",
        origin_price=19900,
        cover_color="#333",
        sku_count=sku,
        min_price=min_price,
    )


class _StubIndex:
    def __init__(self, docs=None, fail: bool = False) -> None:
        self._docs = list(docs or [])
        self.fail = fail

    async def docs(self):
        if self.fail:
            raise UpstreamError("商品目录上游返回 HTTP 500")
        return list(self._docs)


class _StubClient:
    def __init__(self, detail=None, fail: bool = False) -> None:
        self.detail = detail
        self.fail = fail
        self.ids: list[int] = []

    async def fetch_detail(self, product_id: int):
        self.ids.append(product_id)
        if self.fail:
            raise UpstreamError("商品详情上游返回 HTTP 500")
        return self.detail


def _detail(pid: int = 7) -> ProductDetail:
    return ProductDetail(
        id=pid,
        title="轻量跑步鞋",
        description="适合日常慢跑",
        origin_price=39900,
        cover_color="#333333",
        skus=(
            SkuDoc(sku_id=1, sku_code="RUN-40", spec_desc="40 码 黑色", price=12900, stock=5),
            SkuDoc(sku_id=2, sku_code="RUN-41", spec_desc="41 码 黑色", price=13900, stock=0),
        ),
    )


def _box(index=None, client=None) -> CatalogToolbox:
    return CatalogToolbox(index or _StubIndex([_doc(1, "轻量跑步鞋")]), client or _StubClient())


def _invoke(box, name: str, arguments) -> tuple[str, bool]:
    return asyncio.run(box.invoke(name, arguments))


def test_tool_specs_shape():
    assert isinstance(TOOL_SPECS, list) and len(TOOL_SPECS) == 2
    by_name = {spec["function"]["name"]: spec for spec in TOOL_SPECS}
    assert set(by_name) == {"search_products", "get_product_detail"}
    for spec in TOOL_SPECS:
        assert spec["type"] == "function"
        params = spec["function"]["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False
        assert isinstance(spec["function"]["description"], str)
        assert spec["function"]["description"]
    assert by_name["search_products"]["function"]["parameters"]["required"] == ["query"]
    assert by_name["get_product_detail"]["function"]["parameters"]["required"] == ["product_id"]


def test_unknown_tool_is_reported_not_raised():
    text, ok = _invoke(_box(), "delete_everything", "{}")
    assert ok is False
    assert json.loads(text)["error"] == "未知工具: delete_everything"


def test_invalid_json_arguments_are_reported():
    text, ok = _invoke(_box(), "search_products", "{not json")
    assert ok is False
    assert json.loads(text)["error"] == "工具参数不是合法 JSON"


def test_non_object_arguments_are_reported():
    text, ok = _invoke(_box(), "search_products", '["跑步鞋"]')
    assert ok is False
    assert json.loads(text)["error"] == "工具参数不是合法 JSON"


def test_empty_query_is_rejected():
    text, ok = _invoke(_box(), "search_products", '{"query":"   "}')
    assert ok is False
    assert "query" in json.loads(text)["error"]


def test_top_k_out_of_range_is_rejected():
    for bad in (0, 21, "8", True, 3.5):
        text, ok = _invoke(_box(), "search_products", json.dumps({"query": "跑步鞋", "top_k": bad}))
        assert ok is False, bad
        assert "top_k" in json.loads(text)["error"]


def test_search_success_returns_compact_json_with_expected_fields():
    box = _box(_StubIndex([_doc(1, "轻量跑步鞋"), _doc(2, "越野跑步鞋")]))
    text, ok = _invoke(box, "search_products", '{"query":"跑步鞋"}')
    assert ok is True
    payload = json.loads(text)
    assert payload["count"] == len(payload["items"]) >= 1
    assert set(payload["items"][0]) == {"id", "title", "min_price", "sku_count"}
    # 紧凑 JSON：不能有缩进或多余空格，工具结果按 token 计费
    assert "\n" not in text and ": " not in text


def test_search_top_k_defaults_to_eight_and_caps_at_twenty():
    docs = [_doc(i + 1, f"跑步鞋 {i}") for i in range(30)]
    box = _box(_StubIndex(docs))
    default_text, _ = _invoke(box, "search_products", '{"query":"跑步鞋"}')
    assert json.loads(default_text)["count"] <= 8
    capped_text, _ = _invoke(box, "search_products", '{"query":"跑步鞋","top_k":20}')
    assert json.loads(capped_text)["count"] <= 20


def test_detail_success_returns_skus_and_min_price():
    client = _StubClient(_detail(7))
    text, ok = _invoke(_box(client=client), "get_product_detail", '{"product_id":7}')
    assert ok is True
    payload = json.loads(text)
    assert set(payload) == {"id", "title", "description", "min_price", "skus"}
    assert payload["min_price"] == 12900
    assert set(payload["skus"][0]) == {"sku_code", "spec_desc", "price", "stock"}
    assert client.ids == [7]


def test_detail_missing_product_returns_error_without_failure():
    text, ok = _invoke(_box(client=_StubClient(None)), "get_product_detail", '{"product_id":999}')
    assert ok is False
    assert json.loads(text)["error"] == "没有找到这个商品"


def test_detail_invalid_product_id_is_rejected():
    for bad in (0, -1, "7", True, None, 1.5):
        text, ok = _invoke(_box(), "get_product_detail", json.dumps({"product_id": bad}))
        assert ok is False, bad
        assert "product_id" in json.loads(text)["error"]


def test_upstream_error_is_folded_into_chinese_error_and_logged(caplog):
    box = _box(_StubIndex([], fail=True))
    with caplog.at_level(logging.WARNING, logger="app.tools.catalog_tools"):
        text, ok = _invoke(box, "search_products", '{"query":"跑步鞋"}')
    assert ok is False
    payload = json.loads(text)
    # 返回给模型的只有可归类摘要，上游原因绝不能出现在文本里
    assert payload["error"] == "商品服务暂时不可用，请稍后再试"
    assert "HTTP 500" not in text
    # 但原因必须写进日志，否则线上无法定位
    assert "search_products" in caplog.text
    assert "商品目录上游返回 HTTP 500" in caplog.text


def test_detail_upstream_error_is_folded_and_logged(caplog):
    box = _box(client=_StubClient(fail=True))
    with caplog.at_level(logging.WARNING, logger="app.tools.catalog_tools"):
        text, ok = _invoke(box, "get_product_detail", '{"product_id":7}')
    assert ok is False
    assert json.loads(text)["error"] == "商品服务暂时不可用，请稍后再试"
    assert "get_product_detail" in caplog.text


def test_long_result_is_truncated():
    long_desc = "很长的描述" * 600
    detail = ProductDetail(id=7, title="长描述商品", description=long_desc, skus=())
    box = _box(client=_StubClient(detail))
    text, ok = _invoke(box, "get_product_detail", '{"product_id":7}')
    assert ok is True
    assert len(text) == MAX_TOOL_RESULT_CHARS + len(TRUNCATED_SUFFIX)
    assert text.endswith(TRUNCATED_SUFFIX)


def test_short_result_is_not_truncated():
    text, ok = _invoke(_box(), "search_products", '{"query":"跑步鞋"}')
    assert ok is True
    assert not text.endswith(TRUNCATED_SUFFIX)
    assert len(text) < MAX_TOOL_RESULT_CHARS


def test_tools_are_read_only_no_write_verbs():
    # 只读是硬约束：源码里不许出现写操作的 HTTP 动词
    source = open(catalog_tools.__file__, encoding="utf-8").read()
    for verb in ("POST", "PUT", "DELETE", "PATCH"):
        assert verb not in source
