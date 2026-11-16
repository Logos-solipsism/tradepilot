"""POST /search 的**三路**契约：向量路真的接上了 HTTP 路径，且失效时降级两路。

全离线：embedding 全走假 embedder 或 httpx.MockTransport，不联网、不需要 key。

语料用**自建的 7 条玩具商品**，不复用 app/eval 的评估语料——评估集是"尺子"，
测试跟尺子绑在一起的话，日后调评估集会牵动这些用例的红绿。

"向量路生效"怎么证明：query「隔音耳麦」与 1 号「蓝牙降噪**耳**机」只有一个字重叠，
两路 BM25 只拿得到 1 号这一条；而 mock embedder 是一个**离线神谕**——它按表声明
"这条 query 与 1、7、2、3 号相近"。于是三路会多出两路拿不到的候选，
这就是"HTTP 上真的接了三路"的硬证据：只看 200 和"结果非空"证明不了接的是哪条路。
"""

from __future__ import annotations

import json
import logging
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import UpstreamError
from app.llm.client import FAKE_ANSWER, LlmResult, LlmUsage
from app.main import create_app
from app.retrieval.bm25 import BM25Index
from app.retrieval.catalog import CatalogIndex, ProductDoc
from app.retrieval.embedding import ApiEmbedder, HashingEmbedder
from app.retrieval.vector import doc_text
from app.retrieval.vector_cache import (
    STATE_BUILDING,
    STATE_DEGRADED,
    STATE_IDLE,
    STATE_READY,
    CachedEmbedder,
    VectorCache,
    build_vector_cache,
)
from app.security import ticket as ticket_mod
from app.security.ticket import sign
from app.session.memory import MemorySessionStore

SECRET = "test-secret"

_QUERY = "隔音耳麦"

_DOCS = [
    ProductDoc(id=1, title="蓝牙降噪耳机 头戴式", description="主动降噪 长续航"),
    ProductDoc(id=2, title="不锈钢保温杯 480ml", description="长效保温"),
    ProductDoc(id=3, title="机械键盘 红轴 有线", description="全键无冲"),
    ProductDoc(id=4, title="跑步鞋 轻量缓震", description="透气网面"),
    ProductDoc(id=5, title="Type-C 数据线 2 米", description="快充 编织线"),
    ProductDoc(id=6, title="无线充电板 15W", description="立式快充"),
    ProductDoc(id=7, title="显示器 27 英寸 2K", description="办公 护眼"),
]

_QUERY_VECTOR = [1.0, 0.0, 0.0, 0.0]

# mock 神谕：文本 → 4 维向量。键是 vector.doc_text(doc)，即 "{title} {title} {description}"
_VECTOR_TABLE: dict[str, list[float]] = {
    _QUERY: _QUERY_VECTOR,
    "蓝牙降噪耳机 头戴式 蓝牙降噪耳机 头戴式 主动降噪 长续航": [0.9, 0.1, 0.0, 0.0],
    "不锈钢保温杯 480ml 不锈钢保温杯 480ml 长效保温": [0.1, 1.0, 0.0, 0.0],
    "机械键盘 红轴 有线 机械键盘 红轴 有线 全键无冲": [0.1, 0.0, 1.0, 0.0],
    "跑步鞋 轻量缓震 跑步鞋 轻量缓震 透气网面": [0.0, 0.1, 0.0, 1.0],
    "Type-C 数据线 2 米 Type-C 数据线 2 米 快充 编织线": [0.0, 0.0, 0.9, 0.1],
    "无线充电板 15W 无线充电板 15W 立式快充": [0.0, 0.0, 0.8, 0.2],
    "显示器 27 英寸 2K 显示器 27 英寸 2K 办公 护眼": [0.2, 0.2, 0.2, 0.2],
}


class _StubEmbedder:
    """按表返回 4 维向量；**记录每次被要求 embed 了哪些文本**。

    记录是这些用例的核心证据：文档向量"只 embed 一次"、query 缓存"第二次不调"，
    都只能靠调用记录证明，不能靠"响应还是 200"。
    """

    name = "stub:test"
    is_fake = False
    dimension = 4

    def __init__(self, table: dict[str, list[float]] | None = None) -> None:
        self.table = dict(_VECTOR_TABLE)
        if table:
            self.table.update(table)
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self.table.get(text, [0.0, 1.0, 0.0, 0.0]) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def query_calls(self) -> list[str]:
        return [text for text in self.calls if text == _QUERY]

    @property
    def doc_calls(self) -> list[str]:
        return [text for text in self.calls if text != _QUERY]


class _FailingEmbedder(_StubEmbedder):
    """可控失败：文档 embed 或 query embed 分别挂掉。"""

    def __init__(self, *, fail_docs: bool = False, fail_query: bool = False) -> None:
        super().__init__()
        self.fail_docs = fail_docs
        self.fail_query = fail_query

    def embed(self, texts: list[str]) -> list[list[float]]:
        is_query_only = _QUERY in texts and all(text == _QUERY for text in texts)
        if is_query_only and self.fail_query:
            self.calls.extend(texts)
            raise UpstreamError("embedding 上游超时(1.0s): ReadTimeout")
        if self.fail_docs and not is_query_only:
            self.calls.extend(texts)
            raise UpstreamError("embedding 上游返回 HTTP 500")
        return super().embed(texts)


class _StubLlm:
    is_fake = True

    async def complete(self, messages):
        return LlmResult(content=FAKE_ANSWER, usage=LlmUsage())

    async def aclose(self) -> None:
        return None


class _Catalog:
    """目录桩：**身份稳定**的 fetch_all（同一份 list 每次原样交出），与线上一致。"""

    def __init__(self, docs=None, *, copy_each_call: bool = False, fail: bool = False) -> None:
        # **原样持有传进来的 list**，不复制：目录刷新的判据就是 list 身份，
        # 桩在这里复制一份会让"同一份目录"变成"每次都是新目录"，缓存永远不命中。
        # 这与线上 CatalogIndex 的行为一致：同一份快照反复交出同一个 list 对象。
        self._docs = _DOCS if docs is None else docs
        self.copy_each_call = copy_each_call
        self.fail = fail
        self.calls = 0
        # 最后一次真的交出去的 list 对象：CatalogIndex 会缓存它，
        # 而向量缓存的换代判据正是这个身份，用例需要拿到同一个对象来核对
        self.last_returned: list | None = None

    async def fetch_all(self, page_size: int = 100):
        self.calls += 1
        if self.fail:
            raise UpstreamError("商品目录上游返回 HTTP 500")
        self.last_returned = list(self._docs) if self.copy_each_call else self._docs
        return self.last_returned

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _clean_nonces():
    ticket_mod.reset_nonce_cache()
    yield
    ticket_mod.reset_nonce_cache()


def _client(catalog, vector_cache, *, settings: Settings | None = None, ttl: int | None = None):
    resolved = settings or Settings(internal_secret=SECRET, llm_api_key="")
    app = create_app(
        resolved,
        llm_client=_StubLlm(),
        session_store=MemorySessionStore(),
        catalog_client=catalog,
    )
    # 显式注入，不让后台线程的时序参与判定
    app.state.vector_cache = vector_cache
    if ttl is not None:
        # ttl=0：每次 docs() 都重新拉，模拟"目录刷新"
        app.state.catalog_index = CatalogIndex(catalog, ttl_seconds=ttl)
    return TestClient(app)


def _post(client: TestClient, payload: dict):
    return client.post(
        "/search",
        json=payload,
        headers={"X-Internal-Ticket": sign(1, secret=SECRET, ts=int(time.time()))},
    )


def _two_way_ids(query: str = _QUERY, docs: list[ProductDoc] | None = None) -> list[int]:
    """对照：同一份目录、同一条 query 走两路（这正是改动前 /search 的行为）。"""
    return [doc.id for doc, _ in BM25Index(docs or _DOCS).search_fused(query, top_k=10)]


# --- 三路真的生效 -------------------------------------------------------


def test_three_way_route_surfaces_candidates_two_way_cannot_reach():
    """同一份目录、同一条 query：三路必须多出两路拿不到的候选，且名次可解释。"""
    embedder = _StubEmbedder()
    vector_cache = VectorCache(embedder)
    vector_cache.build_now(_DOCS)  # 同步建好，路由确定是三路
    assert vector_cache.state(_DOCS) == STATE_READY

    with _client(_Catalog(_DOCS), vector_cache) as c:
        three = _post(c, {"query": _QUERY, "top_k": 10}).json()["data"]["hits"]

    three_ids = [hit["id"] for hit in three]
    two_ids = _two_way_ids()

    assert two_ids == [1], f"两路本来就不止一条，这条用例的对照前提不成立：{two_ids}"
    assert three_ids != two_ids, f"三路与两路完全相同，向量路没有接上：{three_ids}"
    # 向量路给 1/7/2/3 号各投一票：这些是两路**根本拿不到**的候选
    assert set(three_ids) > set(two_ids)
    assert set(three_ids) == {1, 7, 2, 3}, three_ids
    # 1 号在向量路排第一（神谕声明它与 query 最相近），因此拿满三票排在榜首
    assert three_ids[0] == 1
    # 三条路都只交本快照内的商品：不许出现目录外的 id
    assert set(three_ids) <= {doc.id for doc in _DOCS}


def test_scores_stay_rrf_scale_when_three_way_is_active():
    """三路的 score 仍是 RRF 分量级（~0.03），**不是**余弦分。"""
    vector_cache = VectorCache(_StubEmbedder())
    vector_cache.build_now(_DOCS)
    with _client(_Catalog(_DOCS), vector_cache) as c:
        hits = _post(c, {"query": _QUERY, "top_k": 10}).json()["data"]["hits"]

    scores = [hit["score"] for hit in hits]
    # 三路同权 RRF：任何一条最多拿 3 张 1/(60+rank) 的票，上界是 3/61
    assert all(0.0 <= score <= 3.0 / 61 + 1e-9 for score in scores), scores
    assert max(scores) == pytest.approx(3.0 / 61, rel=1e-9)
    # 混进余弦分的话榜首会是接近 1.0 的数
    assert max(scores) < 0.1, scores


def test_two_way_scores_are_unchanged_when_vector_route_absent():
    """向量路缺席时逐字用两路：结果与 BM25Index.search_fused 完全一致。"""
    vector_cache = VectorCache(_StubEmbedder(), auto_build=False)
    with _client(_Catalog(_DOCS), vector_cache) as c:
        hits = _post(c, {"query": _QUERY, "top_k": 10}).json()["data"]["hits"]

    expected = BM25Index(_DOCS).search_fused(_QUERY, top_k=10)
    assert [hit["id"] for hit in hits] == [doc.id for doc, _ in expected]
    assert [hit["score"] for hit in hits] == [score for _, score in expected]
    assert max(hit["score"] for hit in hits) < 0.1


# --- 两层缓存 -----------------------------------------------------------


def test_doc_vectors_are_embedded_once_across_requests():
    """连续多次请求：文档向量只 embed 一遍，不随请求数翻倍。"""
    embedder = _StubEmbedder()
    vector_cache = VectorCache(embedder)
    catalog = _Catalog(_DOCS)
    with _client(catalog, vector_cache) as c:
        for _ in range(3):
            assert _post(c, {"query": _QUERY, "top_k": 5}).status_code == 200
    vector_cache.wait_settled()

    # 7 条商品各 embed 一次；请求 3 次不该变成 21 次
    assert len(embedder.doc_calls) == len(_DOCS), embedder.doc_calls
    assert catalog.calls == 1


def test_query_vector_cache_hit_avoids_second_embed():
    """同一条 query 搜两次：第二次完全命中缓存，不调上游。"""
    embedder = _StubEmbedder()
    vector_cache = VectorCache(embedder)
    vector_cache.build_now(_DOCS)
    with _client(_Catalog(_DOCS), vector_cache) as c:
        first = _post(c, {"query": _QUERY, "top_k": 5}).json()["data"]["hits"]
        assert embedder.query_calls == [_QUERY]
        second = _post(c, {"query": _QUERY, "top_k": 5}).json()["data"]["hits"]

    assert embedder.query_calls == [_QUERY], f"第二次请求又 embed 了一次: {embedder.calls}"
    # 命中缓存不能改变结果：两次响应逐字相同
    assert first == second


def test_query_cache_embeds_each_distinct_text_once():
    """重复的文本在同一批里也只问一次。"""
    inner = _StubEmbedder()
    cached = CachedEmbedder(inner, max_entries=8, ttl_seconds=60)
    assert cached.embed([_QUERY]) == [_QUERY_VECTOR]
    assert cached.embed([_QUERY, _QUERY]) == [_QUERY_VECTOR, _QUERY_VECTOR]
    assert inner.calls == [_QUERY]


def test_query_cache_is_bounded_lru():
    """容量满时弹**最久未用**的那一条，其余继续命中。"""
    inner = _StubEmbedder()
    cached = CachedEmbedder(inner, max_entries=2, ttl_seconds=60)
    cached.embed(["a", "b", "c"])          # 容量 2：a 被挤出，缓存里是 b、c
    assert inner.calls == ["a", "b", "c"]

    cached.embed(["c"])                     # 命中；c 变成最近使用，b 成了最久未用
    cached.embed(["a"])                     # 未命中：a 当初已淘汰，现在重新问上游
    assert inner.calls == ["a", "b", "c", "a"]

    # 这一下被弹出的是 b（最近的插入已经把 a 变成"较新"），不是 c
    cached.embed(["c"])                     # 命中
    assert inner.calls == ["a", "b", "c", "a"]
    cached.embed(["b"])                     # 未命中
    assert inner.calls == ["a", "b", "c", "a", "b"]


def test_query_cache_expires_after_ttl():
    inner = _StubEmbedder()
    cached = CachedEmbedder(inner, max_entries=8, ttl_seconds=0.01)
    cached.embed([_QUERY])
    time.sleep(0.05)
    cached.embed([_QUERY])
    # 过期必须重新 embed，不能把一个陈旧值一直发出去
    assert inner.calls == [_QUERY, _QUERY]


def test_cached_embedder_passes_through_contract_fields():
    """包装器必须透传 name / dimension / is_fake：否则降级状态在线上就看不见了。"""
    cached = CachedEmbedder(HashingEmbedder(32))
    assert cached.is_fake is True
    assert cached.dimension == 32
    assert cached.name == "hash-32"


def test_vector_index_is_rebuilt_when_catalog_object_changes():
    """目录换代（新的 docs list 对象）必须重建向量索引。"""
    embedder = _StubEmbedder()
    vector_cache = VectorCache(embedder)
    first = list(_DOCS)
    vector_cache.build_now(first)
    assert vector_cache.peek(first) is not None

    second = list(_DOCS[:6])  # 身份不同 = 目录刷新
    assert vector_cache.peek(second) is None
    rebuilt = vector_cache.build_now(second)
    assert rebuilt is not None and rebuilt.n_docs == 6
    # 旧快照的索引不再被认作"当前"
    assert vector_cache.peek(first) is None


def test_search_router_rebuilds_vector_index_after_catalog_refresh():
    """端到端：目录刷新（换了 list 对象）后，向量索引跟着重建，请求全程 200。"""
    embedder = _StubEmbedder()
    vector_cache = VectorCache(embedder)
    catalog = _Catalog(_DOCS, copy_each_call=True)
    # ttl=0：每次 docs() 都重新拉，目录桩每次都交出**新的 list 对象**，模拟刷新
    with _client(catalog, vector_cache, ttl=0) as c:
        assert _post(c, {"query": _QUERY, "top_k": 5}).status_code == 200
        vector_cache.wait_settled()
        first_docs = catalog.last_returned
        first_index = vector_cache.peek(first_docs)
        assert first_index is not None and first_index.n_docs == len(_DOCS)

        catalog._docs = _DOCS[:5]
        assert _post(c, {"query": _QUERY, "top_k": 5}).status_code == 200
        vector_cache.wait_settled()

        second_docs = catalog.last_returned
        second_index = vector_cache.peek(second_docs)

    # 换了一份快照：索引必须是新的、条数跟着变，旧快照的索引不再被认作"当前"
    assert second_docs is not first_docs
    assert second_index is not first_index
    assert second_index.n_docs == 5
    assert vector_cache.peek(first_docs) is None
    # 文档向量进的是同一份 LRU：两代共有的前 5 条不会再 embed 一遍
    # （7 条第一次 + 本代新增 0 条；换代不等于"把没变的商品重算一遍"）
    assert len(embedder.doc_calls) == len(_DOCS), embedder.doc_calls


# --- 降级策略 -----------------------------------------------------------


def test_document_embed_failure_degrades_to_two_way_and_warns(caplog):
    """文档向量建不起来：请求仍是 200 + 两路结果，日志必须有 WARNING。"""
    vector_cache = VectorCache(_FailingEmbedder(fail_docs=True))
    with caplog.at_level(logging.WARNING, logger="app.retrieval.vector_cache"):
        assert vector_cache.build_now(_DOCS) is None
    assert "降级为两路" in caplog.text
    assert vector_cache.state(_DOCS) == STATE_DEGRADED
    assert vector_cache.last_error and "HTTP 500" in vector_cache.last_error

    with _client(_Catalog(_DOCS), vector_cache) as c:
        r = _post(c, {"query": _QUERY, "top_k": 10})

    assert r.status_code == 200, "向量建不起来绝不许把 /search 变成 503/500"
    hits = r.json()["data"]["hits"]
    assert hits, "降级不等于空列表"
    assert [hit["id"] for hit in hits] == _two_way_ids()
    assert r.json()["data"]["indexed"] == len(_DOCS)


def test_document_embed_failure_is_not_retried_every_request():
    """一代目录只建一次：失败后不再每个请求都重试 2-3 秒的 embed。"""
    embedder = _FailingEmbedder(fail_docs=True)
    vector_cache = VectorCache(embedder)
    with _client(_Catalog(_DOCS), vector_cache) as c:
        for _ in range(3):
            assert _post(c, {"query": _QUERY, "top_k": 5}).status_code == 200
        vector_cache.wait_settled()

    # 第一次尝试的 7 条被记录，之后两次请求不再重试
    assert len(embedder.calls) == len(_DOCS), embedder.calls
    assert vector_cache.state(_DOCS) == STATE_DEGRADED


def test_query_embed_failure_degrades_that_request_only(caplog):
    """文档索引建好了、单次 query 的 embed 挂了：该请求降级两路，后端仍健康。"""
    embedder = _FailingEmbedder(fail_query=True)
    vector_cache = VectorCache(embedder)
    assert vector_cache.build_now(_DOCS) is not None  # 文档向量正常
    with caplog.at_level(logging.WARNING, logger="app.api.routers.search"):
        with _client(_Catalog(_DOCS), vector_cache) as c:
            r = _post(c, {"query": _QUERY, "top_k": 10})

    assert r.status_code == 200
    assert [hit["id"] for hit in r.json()["data"]["hits"]] == _two_way_ids()
    assert "降级为两路 BM25" in caplog.text
    # 失败的是这一次 query，不是整条向量路
    assert vector_cache.state(_DOCS) == STATE_READY


def test_blank_query_still_two_way_without_vector_call():
    """空/纯标点 query 走不到向量路，但同样不许崩（VectorIndex 返回空名次）。"""
    vector_cache = VectorCache(_StubEmbedder())
    vector_cache.build_now(_DOCS)
    with _client(_Catalog(_DOCS), vector_cache) as c:
        r = _post(c, {"query": "!!!"})
    assert r.status_code == 200
    assert r.json()["data"]["hits"] == []
    assert r.json()["data"]["indexed"] == len(_DOCS)


def test_empty_catalog_never_touches_embedder():
    """上游真的没商品时不建向量索引：别为 0 条文档白花 5 次 API 调用。"""
    embedder = _StubEmbedder()
    vector_cache = VectorCache(embedder)
    with _client(_Catalog([]), vector_cache) as c:
        r = _post(c, {"query": _QUERY})
    vector_cache.wait_settled()
    assert r.status_code == 200 and r.json()["data"]["indexed"] == 0
    assert embedder.calls == []


# --- 首次请求不许等 2-3 秒 ----------------------------------------------


def test_first_request_does_not_block_on_vector_build():
    """第一次请求先返回两路，向量索引在后台建——响应不许被文档向量卡住。"""

    class _SlowEmbedder(_StubEmbedder):
        def embed(self, texts):
            time.sleep(0.2)  # 模拟真实上游的 2-3 秒，缩短以便用例可跑
            return super().embed(texts)

    vector_cache = VectorCache(_SlowEmbedder())
    with _client(_Catalog(_DOCS), vector_cache) as c:
        started = time.monotonic()
        r = _post(c, {"query": _QUERY, "top_k": 10})
        elapsed = time.monotonic() - started
        assert r.status_code == 200
        assert elapsed < 0.2, f"首次请求等了 {elapsed:.3f}s：文档向量在请求路径里同步建了"
        assert vector_cache.state(_DOCS) in (STATE_BUILDING, STATE_READY)
        assert [hit["id"] for hit in r.json()["data"]["hits"]] == _two_way_ids()

        vector_cache.wait_settled()
        assert vector_cache.state(_DOCS) == STATE_READY
        # 建好之后同一份目录的请求才走三路
        after = _post(c, {"query": _QUERY, "top_k": 10}).json()["data"]["hits"]
    assert [hit["id"] for hit in after] != _two_way_ids()


# --- 启动日志：哈希降级必须喊出来 ---------------------------------------


def test_build_vector_cache_warns_when_embedding_is_fake(caplog):
    with caplog.at_level(logging.WARNING, logger="app.retrieval.vector_cache"):
        cache = build_vector_cache(Settings(embedding_api_key=""))
    assert cache.is_fake is True
    assert "哈希降级" in caplog.text
    assert "没有语义能力" in caplog.text


def test_build_vector_cache_uses_real_embedder_with_key(caplog):
    with caplog.at_level(logging.WARNING, logger="app.retrieval.vector_cache"):
        cache = build_vector_cache(Settings(embedding_api_key="sk-test"))
    assert cache.is_fake is False
    assert "哈希降级" not in caplog.text


def test_create_app_warns_at_startup_when_embedding_is_fake(caplog):
    """装配期就要喊：不能等第一个请求才让运维知道线上是降级向量。"""
    with caplog.at_level(logging.WARNING, logger="app.retrieval.vector_cache"):
        app = create_app(
            Settings(internal_secret=SECRET, llm_api_key="", embedding_api_key=""),
            llm_client=_StubLlm(),
            session_store=MemorySessionStore(),
            catalog_client=_Catalog(_DOCS),
        )
    assert app.state.vector_cache.is_fake is True
    assert "哈希降级" in caplog.text


def test_state_transition_warning_is_logged_once(caplog):
    """状态切换只打一次：既能让运维数出"退化过几次"，又不把日志刷爆。"""
    vector_cache = VectorCache(_FailingEmbedder(fail_docs=True))
    vector_cache.build_now(_DOCS)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.retrieval.vector_cache"):
        assert vector_cache.log_transition(STATE_DEGRADED) is True
        assert vector_cache.log_transition(STATE_DEGRADED) is False
    assert caplog.text.count("状态切换") == 1
    assert "降级为两路" in caplog.text
    assert vector_cache.state(_DOCS) == STATE_DEGRADED


def test_state_reports_idle_before_any_request():
    vector_cache = VectorCache(_StubEmbedder())
    assert vector_cache.state(_DOCS) == STATE_IDLE
    assert vector_cache.is_fake is False
    assert vector_cache.embedder_name == "stub:test"


# --- 真实 ApiEmbedder + MockTransport 的端到端 --------------------------


def _api_settings(**overrides) -> Settings:
    base = dict(
        internal_secret=SECRET,
        llm_api_key="",
        embedding_api_key="sk-test",
        embedding_base_url="https://emb.test/v1",
        embedding_model="test-embed",
        embedding_dimension=4,
    )
    base.update(overrides)
    return Settings(**base)


def _mock_transport(seen: list[list[str]], *, fail: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["input"]
        seen.append(list(texts))
        if fail:
            return httpx.Response(500, json={"message": "boom"})
        data = [
            {"index": position, "embedding": _VECTOR_TABLE.get(text, [0.0, 1.0, 0.0, 0.0])}
            for position, text in enumerate(texts)
        ]
        return httpx.Response(200, json={"data": data})

    return httpx.MockTransport(handler)


def test_three_way_with_real_api_embedder_over_mock_transport():
    """走真实 ApiEmbedder（只把 transport 换成 MockTransport）：三路全链路 + 两层缓存。"""
    seen: list[list[str]] = []
    settings = _api_settings()
    vector_cache = VectorCache(ApiEmbedder(settings, transport=_mock_transport(seen)))
    vector_cache.build_now(_DOCS)
    assert vector_cache.is_fake is False

    with _client(_Catalog(_DOCS), vector_cache, settings=settings) as c:
        first = _post(c, {"query": _QUERY, "top_k": 10})
        calls_after_first = len(seen)
        second = _post(c, {"query": _QUERY, "top_k": 10})

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["data"]["hits"] == second.json()["data"]["hits"]
    # 建索引 1 次（7 条一批）+ 第一次 query 1 次；第二次 query 命中缓存
    assert calls_after_first == 2, seen
    assert len(seen) == 2, seen
    assert seen[0] == [doc_text(doc) for doc in _DOCS]
    assert seen[1] == [_QUERY]
    # 三路确实生效：多出两路拿不到的候选
    assert [hit["id"] for hit in first.json()["data"]["hits"]] != _two_way_ids()


def test_api_embedder_failure_degrades_to_two_way(caplog):
    """真实 ApiEmbedder + 上游 500：文档向量建不起来 → 两路 + WARNING，不是 503。"""
    seen: list[list[str]] = []
    settings = _api_settings()
    vector_cache = VectorCache(ApiEmbedder(settings, transport=_mock_transport(seen, fail=True)))
    with caplog.at_level(logging.WARNING, logger="app.retrieval.vector_cache"):
        with _client(_Catalog(_DOCS), vector_cache, settings=settings) as c:
            r = _post(c, {"query": _QUERY, "top_k": 10})
            vector_cache.wait_settled()

    assert r.status_code == 200
    assert r.json()["data"]["hits"]
    assert "降级为两路" in caplog.text
    assert seen, "根本没向 embedding 上游发过请求：所谓降级是假的"


def test_catalog_upstream_failure_still_502_not_degraded():
    """目录拉取失败仍是 502：那条线跟向量降级是两回事，不许被这次改动混淆。"""
    vector_cache = VectorCache(_StubEmbedder())
    with _client(_Catalog(_DOCS, fail=True), vector_cache) as c:
        r = _post(c, {"query": _QUERY})
    assert r.status_code == 502
    assert r.json()["data"] is None
