"""POST /search 的 HTTP 契约（TestClient 同步驱动，全离线）。"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import UpstreamError
from app.llm.client import FAKE_ANSWER, LlmResult, LlmUsage
from app.main import create_app
from app.retrieval.bm25 import BM25Index
from app.retrieval.catalog import ProductDoc
from app.retrieval.expand import expand_query
from app.security import ticket as ticket_mod
from app.security.ticket import sign
from app.session.memory import MemorySessionStore

SECRET = "test-secret"


class _StubLlm:
    is_fake = True

    async def complete(self, messages):
        return LlmResult(content=FAKE_ANSWER, usage=LlmUsage())

    async def aclose(self) -> None:
        return None


class _StubCatalog:
    """可切换"有货 / 空目录 / 上游失败"的目录桩。"""

    def __init__(self, docs=None, fail: bool = False) -> None:
        self._docs = list(docs or [])
        self.fail = fail
        self.calls = 0

    async def fetch_all(self, page_size: int = 100):
        self.calls += 1
        if self.fail:
            raise UpstreamError("商品目录上游返回 HTTP 500")
        return list(self._docs)

    async def aclose(self) -> None:
        return None


def _doc(doc_id: int, title: str, description: str = "", min_price=None, sku=1) -> ProductDoc:
    return ProductDoc(
        id=doc_id,
        title=title,
        description=description,
        min_price=min_price,
        sku_count=sku,
    )


@pytest.fixture(autouse=True)
def _clean_nonces():
    ticket_mod.reset_nonce_cache()
    yield
    ticket_mod.reset_nonce_cache()


def make_client(catalog=None) -> TestClient:
    app = create_app(
        Settings(internal_secret=SECRET, llm_api_key=""),
        llm_client=_StubLlm(),
        session_store=MemorySessionStore(),
        catalog_client=catalog or _StubCatalog(),
    )
    return TestClient(app)


def ticket(user_id: int = 1) -> str:
    return sign(user_id, secret=SECRET, ts=int(time.time()))


def _post(client: TestClient, payload: dict):
    return client.post("/search", json=payload, headers={"X-Internal-Ticket": ticket()})


def test_search_happy_path_field_names_are_exact():
    catalog = _StubCatalog(
        [_doc(3, "跑步鞋", "轻量缓震", min_price=12900, sku=2), _doc(9, "运动水壶", min_price=5900)]
    )
    with make_client(catalog) as c:
        r = _post(c, {"query": "跑步鞋", "top_k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 0 and body["message"] == "ok"
    data = body["data"]
    # query 原样回显
    assert data["query"] == "跑步鞋"
    assert data["indexed"] == 2
    assert len(data["hits"]) == 1
    hit = data["hits"][0]
    # 逐字：只有这五个字段，绝不多塞 description
    assert set(hit) == {"id", "title", "min_price", "sku_count", "score"}
    assert hit["id"] == 3
    assert hit["title"] == "跑步鞋"
    assert hit["min_price"] == 12900
    assert hit["sku_count"] == 2
    # 分数量级是 RRF 融合分，不参与任何阈值判定；0.0 表示"只为凑满 top_k 的补位项"，
    # 是合法结果而不是异常——不许为了让这条断言好过就把 0.0 的条目过滤掉。
    assert isinstance(hit["score"], float) and hit["score"] >= 0


def test_search_query_is_echoed_without_trimming():
    catalog = _StubCatalog([_doc(1, " 跑步鞋 ")])
    with make_client(catalog) as c:
        r = _post(c, {"query": "  跑步鞋  "})
    assert r.status_code == 200
    assert r.json()["data"]["query"] == "  跑步鞋  "


def test_search_without_ticket_is_real_401():
    with make_client() as c:
        r = c.post("/search", json={"query": "跑步鞋"})
    assert r.status_code == 401
    assert r.json() == {"code": 401, "message": "internal ticket 无效或已过期", "data": None}


def test_search_extra_field_is_rejected():
    with make_client() as c:
        r = _post(c, {"query": "跑步鞋", "user_id": 999})
    assert r.status_code == 422


def test_search_query_over_200_chars_is_rejected():
    with make_client() as c:
        r = _post(c, {"query": "x" * 201})
    assert r.status_code == 422


def test_search_empty_query_is_rejected():
    with make_client() as c:
        r = _post(c, {"query": ""})
    assert r.status_code == 422


def test_search_top_k_out_of_range_is_rejected():
    with make_client() as c:
        assert _post(c, {"query": "跑步鞋", "top_k": 0}).status_code == 422
        assert _post(c, {"query": "跑步鞋", "top_k": 51}).status_code == 422


def test_search_top_k_limits_hits():
    catalog = _StubCatalog([_doc(i, f"跑步鞋 型号{i}") for i in range(1, 6)])
    with make_client(catalog) as c:
        r = _post(c, {"query": "跑步鞋", "top_k": 2})
    assert r.status_code == 200
    assert len(r.json()["data"]["hits"]) == 2


def test_search_query_with_no_tokens_returns_200_with_no_hits():
    # 只有标点/emoji 的 query 过得了 min_length=1，但分词后为空：不能 500
    catalog = _StubCatalog([_doc(1, "轻量跑步鞋")])
    with make_client(catalog) as c:
        r = _post(c, {"query": "!!!"})
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["query"] == "!!!"
    assert body["data"]["hits"] == []
    assert body["data"]["indexed"] == 1


def test_search_empty_catalog_returns_200_with_no_hits():
    with make_client(_StubCatalog([])) as c:
        r = _post(c, {"query": "跑步鞋"})
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["hits"] == []
    assert body["data"]["indexed"] == 0


def test_search_upstream_failure_is_real_502():
    with make_client(_StubCatalog([], fail=True)) as c:
        r = _post(c, {"query": "跑步鞋"})
    assert r.status_code == 502
    body = r.json()
    assert body["code"] == 502 and body["data"] is None


def test_health_reports_search_ready_and_empty():
    with make_client(_StubCatalog([_doc(1, "跑步鞋")])) as c:
        assert c.get("/health").json()["search"] == "empty"  # 索引还没建
        _post(c, {"query": "跑步鞋"})
        assert c.get("/health").json()["search"] == "ready"
    with make_client(_StubCatalog([])) as c:
        assert c.get("/health").json()["search"] == "empty"


def test_search_without_agent_token_is_502_not_empty_results():
    # 不注入 catalog_client：走真实 CatalogClient，token 未配置必须 fail-closed 成 502
    app = create_app(
        Settings(internal_secret=SECRET, llm_api_key="", agent_service_token=""),
        llm_client=_StubLlm(),
        session_store=MemorySessionStore(),
    )
    with TestClient(app) as c:
        assert c.get("/health").json()["search"] == "empty"
        r = _post(c, {"query": "跑步鞋"})
    assert r.status_code == 502
    assert r.json()["data"] is None


def test_search_reuses_bm25_index_across_requests():
    catalog = _StubCatalog([_doc(1, "跑步鞋")])
    with make_client(catalog) as c:
        _post(c, {"query": "跑步鞋"})
        _post(c, {"query": "跑步"})
    # 目录只拉了一次，索引也只建一次（缓存按 list 身份复用）
    assert catalog.calls == 1

# ---------------------------------------------------------------------------
# 扩展词在 HTTP 层的差分验证
#
# 目录是**测试自建的固定语料**（15 条），不是线上目录、也不复用 app/eval 的语料：
# 评估语料是"尺子"，改它会牵动基线数字，测试不该跟尺子绑在一起。
# 这份目录的结构是刻意的——充电类商品用「补电/充电」这类词，笔记本类商品
# 一个「补」「电」都不含，只有「手提电脑 → 笔记本」这条同义词规则能把它们救回来。
# ---------------------------------------------------------------------------
_DIFF_QUERY = "手提电脑 补电"

_DIFF_DOCS = [
    _doc(1, "氮化镓 65W GaN 笔记本适配器 便携旅行", min_price=12900),
    _doc(2, "联想 ThinkPad X1 商务笔记本 14 英寸", min_price=899900),
    _doc(3, "苹果 MacBook Air M3 轻薄笔记本", min_price=849900),
    _doc(4, "65W 手机快充充电器 单口", min_price=6900),
    _doc(5, "20000mAh 充电宝 大容量", min_price=9900),
    _doc(6, "无线充电板 15W 快充", min_price=7900),
    _doc(7, "车载充电器 双口 24W", min_price=3900),
    _doc(8, "充电器收纳包 防摔", min_price=1900),
    _doc(9, "Type-C 数据线 2 米 快充", min_price=2900),
    _doc(10, "氮化镓充电头 100W 多口", min_price=15900),
    _doc(11, "电源适配器 45W 笔记本用", min_price=10900),
    _doc(12, "多口充电器 65W 桌面款", min_price=11900),
    _doc(13, "充电器 硅胶保护壳", min_price=900),
    _doc(14, "慢充补电头 5W", min_price=1900),
    _doc(15, "显示器 27 英寸 2K 办公", min_price=129900),
]


def test_search_http_recovers_hits_that_pure_bm25_cannot_reach():
    """同一 query：/search（融合）的结果里必须有纯 BM25 拿不到、靠扩展词救回的条目。

    这条是"扩展逻辑在生产路径上真的生效"的唯一硬证据——只断言 HTTP 返回 200
    证明不了接的是哪条路。
    """
    catalog = _StubCatalog(_DIFF_DOCS)
    with make_client(catalog) as c:
        r = _post(c, {"query": _DIFF_QUERY, "top_k": 10})
    assert r.status_code == 200
    http_ids = [hit["id"] for hit in r.json()["data"]["hits"]]

    # 对照：同一份目录、同一条 query，走纯 BM25（这正是改动前 /search 的行为）
    index = BM25Index(_DIFF_DOCS)
    bm25_ids = [doc.id for doc, _ in index.search(_DIFF_QUERY, top_k=10)]

    assert http_ids != bm25_ids, "HTTP 结果与纯 BM25 完全一致：扩展路没有生效"
    rescued = [doc_id for doc_id in http_ids if doc_id not in bm25_ids]
    assert rescued, f"融合结果里没有多出任何条目；BM25={bm25_ids} HTTP={http_ids}"

    by_id = {doc.id: doc for doc in _DIFF_DOCS}
    titles = [by_id[doc_id].title for doc_id in rescued]
    # 多出来的条目必须是因为「手提电脑 → 笔记本」这条规则才进榜的。
    # 扩展词是过 tokenize 的（"笔记本" 会拆成 笔/记/本/笔记/记本），所以这里
    # 断言规则名而不是断言某个整词 token 存在。
    expansion = expand_query(_DIFF_QUERY)
    assert "synonym:手提电脑" in expansion.applied, f"规则未命中：{expansion.applied}"
    assert any("笔记本" in title for title in titles), f"多出来的条目不是扩展命中：{titles}"


def test_search_top_k_is_prefix_of_larger_top_k():
    """名次深度是模块常量、不随 top_k 变：top-5 必须是 top-10 的前缀。"""
    with make_client(_StubCatalog(_DIFF_DOCS)) as c:
        small = _post(c, {"query": _DIFF_QUERY, "top_k": 5}).json()["data"]["hits"]
        large = _post(c, {"query": _DIFF_QUERY, "top_k": 10}).json()["data"]["hits"]
    small_ids = [hit["id"] for hit in small]
    large_ids = [hit["id"] for hit in large]
    assert len(small_ids) == 5
    assert small_ids == large_ids[:5], f"top-5={small_ids} 不是 top-10={large_ids} 的前缀"


def test_search_empty_catalog_with_expanding_query_returns_200():
    """空目录要在**扩展发生之前或之后都安全**：query 会命中同义词规则时也不能 500。"""
    with make_client(_StubCatalog([])) as c:
        r = _post(c, {"query": _DIFF_QUERY, "top_k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["query"] == _DIFF_QUERY
    assert body["data"]["hits"] == []
    assert body["data"]["indexed"] == 0
