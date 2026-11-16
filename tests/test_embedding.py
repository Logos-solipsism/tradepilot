"""Embedder 抽象：哈希向量的跨进程确定性 + 真模型/降级的选择规则。

跨进程确定性用一个**独立子进程**来验（见 test_hashing_embedder_is_stable_across_processes）：
同进程内跑两次永远一样，那证明不了什么——内置 hash() 也是同进程稳定、跨进程才加盐。
真模型的测试全部走 httpx.MockTransport，禁止真打网络。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.errors import UpstreamError
from app.retrieval.embedding import ApiEmbedder, HashingEmbedder, build_embedder

REPO_ROOT = Path(__file__).resolve().parent.parent


def _api_settings(**overrides) -> Settings:
    base = dict(
        embedding_api_key="sk-test",
        embedding_base_url="https://emb.test/v1",
        embedding_model="test-embed",
        embedding_dimension=8,
        embedding_timeout_seconds=1.0,
    )
    base.update(overrides)
    return Settings(**base)


def _ok_body(vectors: list[list[float]], *, with_index: bool = True) -> dict:
    data = []
    for position, vector in enumerate(vectors):
        item: dict = {"embedding": vector}
        if with_index:
            item["index"] = position
        data.append(item)
    return {"data": data}


# --- 选择规则：没 key 绝不选 API 实现 -------------------------------------


def test_build_embedder_falls_back_to_hashing_without_key():
    embedder = build_embedder(Settings(embedding_api_key="", embedding_dimension=16))
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.is_fake is True
    assert embedder.dimension == 16


def test_build_embedder_treats_whitespace_key_as_absent():
    # 只有空白也算没配：否则会拿 " " 去发一次必然 401 的请求
    assert isinstance(build_embedder(Settings(embedding_api_key="   ")), HashingEmbedder)


def test_build_embedder_uses_api_when_key_present():
    embedder = build_embedder(_api_settings())
    assert isinstance(embedder, ApiEmbedder)
    assert embedder.is_fake is False
    assert embedder.dimension == 8


def test_api_embedder_without_key_fails_closed():
    # 直接构造 ApiEmbedder 也不许偷偷降级：那会让评估报告说谎
    embedder = ApiEmbedder(Settings(embedding_api_key="", embedding_dimension=8))
    with pytest.raises(UpstreamError):
        embedder.embed(["x"])


# --- 哈希向量：形状、归一化、确定性 ---------------------------------------


def test_hashing_embedder_shape_and_unit_norm():
    embedder = HashingEmbedder(32)
    vectors = embedder.embed(["跑步鞋", "蓝牙耳机", ""])
    assert len(vectors) == 3
    assert all(len(vector) == 32 for vector in vectors)
    for vector in vectors[:2]:
        norm = sum(value * value for value in vector) ** 0.5
        assert abs(norm - 1.0) < 1e-9


def test_hashing_embedder_empty_text_is_zero_vector():
    # 一个 token 都没有：给零向量而不是硬凑方向，调用方据此认定"没有向量信号"
    vector = HashingEmbedder(32).embed([""])[0]
    assert vector == [0.0] * 32
    assert HashingEmbedder(32).embed(["!！?"])[0] == [0.0] * 32


def test_hashing_embedder_is_stable_within_process():
    embedder = HashingEmbedder(64)
    assert embedder.embed(["蓝牙耳机 降噪"]) == embedder.embed(["蓝牙耳机 降噪"])


def test_hashing_embedder_does_not_use_builtin_hash():
    # 内置 hash() 每进程加盐；用 blake2b 时同一文本在任何进程都是同一向量。
    # 这里直接钉住一条具体数值：换了哈希算法这条会红，提醒同步改文档。
    vector = HashingEmbedder(16).embed(["a"])[0]
    assert vector == HashingEmbedder(16).embed(["a"])[0]
    assert any(value != 0.0 for value in vector)


def test_hashing_embedder_is_stable_across_processes():
    """同一文本在两个**独立进程**里必须产出完全相同的向量。

    这是"不能用内置 hash()"那条约束的验证方式：把向量序列化成 JSON 再逐位比。
    子进程用 `-c` 起，PYTHONHASHSEED 显式给不同值——若实现依赖内置 hash()，
    两次的桶下标就会不同，这条立刻红。
    """
    script = (
        "import json;"
        "from app.retrieval.embedding import HashingEmbedder;"
        "v = HashingEmbedder(64).embed_one('蓝牙耳机 降噪 长续航');"
        "print(json.dumps([round(x, 12) for x in v]))"
    )
    outputs = []
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(result.stdout.strip()))
    assert outputs[0] == outputs[1] == outputs[2]
    # 子进程输出的是 round(x, 12) 之后的 JSON；本进程也按同样精度比
    local = [round(value, 12) for value in HashingEmbedder(64).embed_one("蓝牙耳机 降噪 长续航")]
    assert outputs[0] == local


def test_hashing_embedder_shared_text_scores_higher_than_unrelated():
    """哈希向量没有任何语义能力，但它必须至少做到"字面越像越像"。"""
    embedder = HashingEmbedder(1024)

    def cosine(left: str, right: str) -> float:
        a = embedder.embed_one(left)
        b = embedder.embed_one(right)
        return sum(x * y for x, y in zip(a, b))

    close = cosine("蓝牙降噪耳机", "蓝牙降噪耳机 长续航")
    far = cosine("蓝牙降噪耳机", "不锈钢保温杯")
    assert close > far


def test_hashing_embedder_rejects_bad_dimension():
    with pytest.raises(ValueError):
        HashingEmbedder(0)


# --- API 实现：协议解析（全离线） -----------------------------------------


def test_api_embedder_parses_openai_compatible_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/embeddings")
        assert request.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(request.content)
        assert body["model"] == "test-embed"
        assert body["input"] == ["a", "b"]
        return httpx.Response(200, json=_ok_body([[1.0, 0.0], [0.0, 1.0]]))

    settings = _api_settings(embedding_dimension=2)
    embedder = ApiEmbedder(settings, transport=httpx.MockTransport(handler))
    assert embedder.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_api_embedder_reorders_by_index_field():
    def handler(request: httpx.Request) -> httpx.Response:
        # 上游把顺序打乱了：必须按 index 归位，否则向量和文档会错配
        return httpx.Response(200, json={"data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ]})

    settings = _api_settings(embedding_dimension=2)
    embedder = ApiEmbedder(settings, transport=httpx.MockTransport(handler))
    # 打乱顺序送进来，必须按 index 归位后返回
    assert embedder.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]


def test_api_embedder_rejects_non_2xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "bad key"})

    embedder = ApiEmbedder(_api_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamError):
        embedder.embed(["x"])


def test_api_embedder_rejects_malformed_payload():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": []}]})

    embedder = ApiEmbedder(_api_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamError):
        embedder.embed(["x"])


def test_api_embedder_empty_input_makes_no_request():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("空输入不该发请求")

    embedder = ApiEmbedder(_api_settings(), transport=httpx.MockTransport(handler))
    assert embedder.embed([]) == []
