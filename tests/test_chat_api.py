"""/chat 与 /health 的 HTTP 契约（TestClient 同步驱动，全离线）。"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import UpstreamError
from app.llm.client import FAKE_ANSWER, LlmClient
from app.main import create_app
from app.retrieval.catalog import ProductDoc
from app.retrieval.embedding import HashingEmbedder
from app.retrieval.vector_cache import VectorCache
from app.security import ticket as ticket_mod
from app.security.ticket import sign
from app.session.memory import MemorySessionStore

SECRET = "test-secret"


class _FakeEmbedder(HashingEmbedder):
    """可控的哈希 embedder：只为了让 /health 能区分 real / fake 两种向量路。"""

    def __init__(self, is_fake: bool = True) -> None:
        super().__init__(64)
        self.is_fake = is_fake
        self.name = "fake-64" if is_fake else "fake-real-64"


class _StubLlm:
    """记录每次收到的 messages，可切换成抛上游错误。"""

    def __init__(self, fake: bool = True) -> None:
        self.is_fake = fake
        self.calls: list[list[dict]] = []
        self.tools_seen: list[list[dict] | None] = []
        self.fail = False

    async def complete(self, messages, *, tools=None):
        self.calls.append(messages)
        self.tools_seen.append(tools)
        if self.fail:
            raise UpstreamError("上游模型返回 HTTP 500")
        from app.llm.client import LlmResult, LlmUsage

        return LlmResult(content=FAKE_ANSWER, usage=LlmUsage(input=1, output=2, total=3))


class _TwoStepLlm:
    """第一轮要求调一次工具，第二轮给最终答案。"""

    is_fake = False

    def __init__(self, answer: str = "129 元的这双，40 码还有货。") -> None:
        self.answer = answer
        self.calls: list[list[dict]] = []

    async def complete(self, messages, *, tools=None):
        from app.llm.client import LlmResult, LlmUsage, ToolCall

        self.calls.append(messages)
        if len(self.calls) == 1:
            return LlmResult(
                content="",
                usage=LlmUsage(input=10, output=2, total=12),
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="search_products",
                        arguments='{"query":"跑步鞋"}',
                    )
                ],
            )
        return LlmResult(content=self.answer, usage=LlmUsage(input=20, output=5, total=25))

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _clean_nonces():
    ticket_mod.reset_nonce_cache()
    yield
    ticket_mod.reset_nonce_cache()


def make_client(secret: str = SECRET, llm=None, store=None) -> TestClient:
    app = create_app(
        Settings(internal_secret=secret, llm_api_key=""),
        llm_client=llm or _StubLlm(),
        session_store=store or MemorySessionStore(),
    )
    return TestClient(app)


def ticket(user_id: int = 1) -> str:
    return sign(user_id, secret=SECRET, ts=int(time.time()))


def test_health_has_no_ticket_requirement():
    with make_client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "ok": True,
        "llm": "fake",
        "version": "0.1.0",
        "sessions": "memory",
        "search": "empty",
        # 探活不建索引、不打上游：进程刚起来时向量路还没跑过，如实报 off/idle
        "vector": "off",
        "vector_state": "idle",
    }


def test_health_reports_real_when_key_configured():
    with make_client() as c:
        c.app.state.llm_client.is_fake = False
        r = c.get("/health")
    assert r.json()["llm"] == "real"


def _catalog_with_docs(c, docs):
    """把目录直接塞进 CatalogIndex 的缓存（模拟"已经拉过一次"）。

    不调 docs()：那个方法会走 CatalogClient，而它没配 token 时是 fail-closed。
    这里要验的是 /health 的只读展示，不该顺带验证上游拉取。
    """
    catalog = c.app.state.catalog_index
    catalog._docs = list(docs)
    catalog._last_refreshed_at = time.time()
    return catalog


def test_health_reports_vector_off_before_any_index():
    """探活不许触发建索引：没建过就是 off/idle，绝不能在 /health 里打上游。"""
    with make_client() as c:
        cache = VectorCache(_FakeEmbedder(), auto_build=True)
        c.app.state.vector_cache = cache
        j = c.get("/health").json()
        assert j["vector"] == "off"
        assert j["vector_state"] == "idle"
        # 连读两次都不许把后台建索引叫起来
        c.get("/health")
        assert cache.peek(c.app.state.catalog_index.cached_docs) is None


def test_health_reports_vector_fake_and_ready_after_build():
    """哈希降级在线上必须能从 /health 看出来——这是日志之外唯一的自动发现路径。"""
    docs = [ProductDoc(id=i, title="商品%d 测试" % i) for i in range(1, 4)]
    with make_client() as c:
        catalog = _catalog_with_docs(c, docs)
        cache = VectorCache(_FakeEmbedder(), auto_build=False)
        c.app.state.vector_cache = cache
        cache.build_now(catalog.cached_docs)
        j = c.get("/health").json()
        assert j["vector"] == "fake"
        assert j["vector_state"] == "ready"
        assert j["search"] == "ready"


def test_health_reports_vector_real_when_embedder_is_real():
    docs = [ProductDoc(id=i, title="商品%d 测试" % i) for i in range(1, 4)]
    with make_client() as c:
        catalog = _catalog_with_docs(c, docs)
        cache = VectorCache(_FakeEmbedder(is_fake=False), auto_build=False)
        c.app.state.vector_cache = cache
        cache.build_now(catalog.cached_docs)
        assert c.get("/health").json()["vector"] == "real"


def test_health_reports_degraded_when_build_fails():
    """/health 要能回答"这台机器现在是不是一直在降级"。"""

    class _Boom(_FakeEmbedder):
        def embed(self, texts):
            raise UpstreamError("embedding 上游返回 HTTP 503")

    docs = [ProductDoc(id=i, title="商品%d 测试" % i) for i in range(1, 4)]
    with make_client() as c:
        catalog = _catalog_with_docs(c, docs)
        cache = VectorCache(_Boom(), auto_build=False)
        c.app.state.vector_cache = cache
        assert cache.build_now(catalog.cached_docs) is None
        j = c.get("/health").json()
        assert j["vector"] == "fake"
        assert j["vector_state"] == "degraded"


def test_chat_without_ticket_is_real_401():
    with make_client() as c:
        r = c.post("/chat", json={"message": "hi"})
    assert r.status_code == 401
    assert r.json() == {"code": 401, "message": "internal ticket 无效或已过期", "data": None}


def test_chat_with_bad_ticket_is_real_401():
    with make_client() as c:
        r = c.post(
            "/chat",
            json={"message": "hi"},
            headers={"X-Internal-Ticket": "1.2.3.deadbeef"},
        )
    assert r.status_code == 401
    assert r.json()["code"] == 401


def test_chat_with_expired_ticket_is_real_401():
    stale = sign(1, secret=SECRET, ts=int(time.time()) - 400)
    with make_client() as c:
        r = c.post("/chat", json={"message": "hi"}, headers={"X-Internal-Ticket": stale})
    assert r.status_code == 401


def test_chat_happy_path_returns_envelope():
    with make_client() as c:
        r = c.post(
            "/chat",
            json={"message": "帮我找一双跑步鞋", "session_id": None, "trace_id": None},
            headers={"X-Internal-Ticket": ticket()},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 0 and body["message"] == "ok"
    data = body["data"]
    assert data["answer"] == FAKE_ANSWER
    assert len(data["session_id"]) == 36
    assert data["tool_calls"] == []
    assert set(data["usage"]) == {"input", "output", "total"}
    assert all(isinstance(v, int) for v in data["usage"].values())


def test_chat_body_with_user_id_is_rejected():
    # 防越权红线：用户身份只能来自票据，请求体带 user_id 必须被拒
    with make_client() as c:
        r = c.post(
            "/chat",
            json={"message": "hi", "user_id": 999},
            headers={"X-Internal-Ticket": ticket()},
        )
    assert r.status_code in (400, 422)
    assert r.status_code != 200


def test_chat_empty_and_oversized_message_rejected():
    # nonce 一次性：每个请求各签一张票，否则第二次会被当成重放报 401，测不到校验分支
    with make_client() as c:
        for body in ({"message": ""}, {"message": "x" * 2001}):
            r = c.post("/chat", json=body, headers={"X-Internal-Ticket": ticket()})
            assert r.status_code == 422, body


def test_chat_generates_session_id_when_null():
    with make_client() as c:
        r = c.post("/chat", json={"message": "a"}, headers={"X-Internal-Ticket": ticket()})
    assert r.json()["data"]["session_id"]


def test_chat_reuses_history_when_session_id_given():
    llm = _StubLlm()
    with make_client(llm=llm) as c:
        first = c.post("/chat", json={"message": "第一问"}, headers={"X-Internal-Ticket": ticket()})
        sid = first.json()["data"]["session_id"]
        second = c.post(
            "/chat",
            json={"message": "第二问", "session_id": sid},
            headers={"X-Internal-Ticket": ticket()},
        )
    assert second.status_code == 200
    assert second.json()["data"]["session_id"] == sid
    sent = llm.calls[1]
    # 契约是"历史窗口里只有 user/assistant"（工具往返不进历史），
    # 不是"整个 payload 没有 system"——system 提示词由 Agent 每轮加在最前面。
    assert [m["role"] for m in sent][0] == "system"
    dialog = [m for m in sent if m["role"] != "system"]
    assert [m["role"] for m in dialog] == ["user", "assistant", "user"]
    assert dialog[0]["content"] == "第一问"
    assert dialog[1]["content"] == FAKE_ANSWER
    assert dialog[2]["content"] == "第二问"


def test_chat_upstream_failure_is_real_502():
    llm = _StubLlm()
    llm.fail = True
    with make_client(llm=llm) as c:
        r = c.post("/chat", json={"message": "hi"}, headers={"X-Internal-Ticket": ticket()})
    assert r.status_code == 502
    body = r.json()
    assert body["code"] == 502 and body["data"] is None
    # 只允许透出可归类的摘要（超时 / HTTP 状态码），不许把上游报文带出来
    assert body["message"].startswith("上游模型")
    assert "boom" not in body["message"]

class _StubToolbox:
    """桩工具集合：默认返回一条命中，可切换成"失败结果"。"""

    def __init__(self, result=None) -> None:
        self.result = result or ('{"count":1,"items":[]}', True)
        self.calls: list[tuple[str, str]] = []

    async def invoke(self, name: str, arguments: str):
        self.calls.append((name, arguments))
        return self.result


def _chat_with_toolbox(llm, toolbox):
    """把桩 toolbox 注入 FastAPI 依赖，避免 HTTP 测试碰真实目录/上游。"""
    from app.api.deps import get_toolbox

    app = create_app(
        Settings(internal_secret=SECRET, llm_api_key=""),
        llm_client=llm,
        session_store=MemorySessionStore(),
    )
    app.dependency_overrides[get_toolbox] = lambda: toolbox
    return TestClient(app)


def test_chat_returns_tool_calls_and_second_round_answer():
    llm = _TwoStepLlm(answer="轻量跑步鞋 129 元，40 码还有 5 件。")
    toolbox = _StubToolbox()
    with _chat_with_toolbox(llm, toolbox) as c:
        r = c.post("/chat", json={"message": "有跑步鞋吗"}, headers={"X-Internal-Ticket": ticket()})

    assert r.status_code == 200
    data = r.json()["data"]
    # answer 必须来自第二轮，不是第一轮那个空的 content
    assert data["answer"] == "轻量跑步鞋 129 元，40 码还有 5 件。"
    assert len(data["tool_calls"]) == 1
    trace = data["tool_calls"][0]
    assert trace["name"] == "search_products"
    assert trace["arguments"] == '{"query":"跑步鞋"}'
    assert isinstance(trace["ok"], bool) and trace["ok"] is True
    assert toolbox.calls == [("search_products", '{"query":"跑步鞋"}')]
    assert sum(data["usage"].values()) > 0


def test_chat_survives_tool_failure_with_200_and_non_empty_answer():
    llm = _TwoStepLlm(answer="商品服务暂时不可用，请稍后再试。")
    toolbox = _StubToolbox(('{"error":"商品服务暂时不可用，请稍后再试"}', False))
    with _chat_with_toolbox(llm, toolbox) as c:
        r = c.post("/chat", json={"message": "有跑步鞋吗"}, headers={"X-Internal-Ticket": ticket()})

    # 工具失败是业务事件，不是 502：整条对话要能继续跑完
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["tool_calls"][0]["ok"] is False
    assert data["answer"]

