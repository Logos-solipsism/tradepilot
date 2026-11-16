"""Embedder 抽象：有 key 调真实 embedding，没有 key 用确定性哈希向量降级。

两个实现共用同一个 **维度**（默认 1024，与 BAAI/bge-m3 一致），
因为下游 faiss 索引的维度是建索引时钉死的：换实现就换维度的话，
"这台机器上有没有配 key"会决定索引能不能加载，那是不可接受的环境耦合。

**降级不是假装**：`HashingEmbedder.is_fake` 恒为 True，评估报告与日志据此标注
"向量路走的是哈希降级"。哈希向量没有任何语义能力，它只保证
"字符重叠多的文本余弦更高"——把它当成真 embedding 会得出错误结论。

**确定性为什么必须靠 hashlib 而不是内置 hash()**：CPython 的 str hash 每进程加盐
（PYTHONHASHSEED），同一个词在两个进程里落进不同桶，向量对不上、测试随机红。
blake2b 是密码学哈希，同输入同输出、跨进程跨机器恒定。

哈希向量怎么做的（signed hashing trick，Weinberger et al. 2009）：
把 token 映射到某一个桶，并额外用一个独立比特决定 +1/-1 符号。
带符号的好处是**碰撞会互相抵消而不是累加**——不加符号时，两条只共享常见字的
无关文本也会因为桶重叠而拿到虚高的余弦，向量路就变成噪声放大器。
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod

import httpx

from app.config import Settings
from app.errors import UpstreamError
from app.retrieval.tokenize import tokenize

log = logging.getLogger(__name__)

# blake2b 的 person 参数最多 16 字节；钉死一个域分隔串，
# 免得将来别处也用 blake2b 做哈希时两套取值意外撞在一起
_HASH_PERSON = b"tradepilot-emb"
# 一次 POST 里塞多少条文本。bge-m3 单请求能收更多，但 150 条商品一次全塞进去
# 只是把超时风险从一个请求摊到一次失败上；分批失败还能定位到是哪一批
_API_BATCH_SIZE = 32


class Embedder(ABC):
    """把文本变成等长向量的最小契约。实现必须**同步**：检索路径是同步的。"""

    name: str
    dimension: int
    # True = 没有真模型在跑（哈希降级 / 未配 key），调用方不得据此宣称"用了语义模型"
    is_fake: bool

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """按输入顺序返回向量；长度必须与 texts 一致（不足即视为实现违约）。"""

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def _token_bucket(token: str, dimension: int) -> tuple[int, float]:
    """把一个 token 确定性地映射到 (桶下标, ±1)。

    一次摘要里同时取桶和符号：两次单独哈希既慢又没必要，
    而"从同一份摘要的不同字节取两样东西"在统计上同样独立。
    """
    digest = hashlib.blake2b(
        token.encode("utf-8"), digest_size=16, person=_HASH_PERSON
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % dimension
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


class HashingEmbedder(Embedder):
    """确定性的词袋哈希向量；纯标准库，不发任何请求。

    权重取 `1 + ln(tf)` 并再乘一个"词越长越重"的系数：单字（"鞋"）几乎出现在
    每个同类商品里，二元组（"跑步"）才有区分度，而英文型号（"pegasus"）几乎是唯一标识。
    没有语料就估不出 IDF，只能用长度当区分度的代理。
    """

    is_fake = True

    def __init__(self, dimension: int = 1024) -> None:
        if dimension < 1:
            raise ValueError("哈希向量维度必须 >= 1")
        self.dimension = int(dimension)
        self.name = f"hash-{self.dimension}"

    def _vector(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        # tokenize 已经做了 NFKC + lower：全角/大小写差异不会改变向量
        for token in tokenize(text):
            counts[token] = counts.get(token, 0) + 1
        vector = [0.0] * self.dimension
        # 按 token 排序累加：浮点加法不满足结合律，顺序一变末位就可能不同，
        # 跨进程一致性要求累加顺序也固定下来（tokenize 的输出顺序本身与文本相关）
        for token in sorted(counts):
            weight = (1.0 + math.log(counts[token])) * (1.0 + 0.5 * (len(token) - 1))
            bucket, sign = _token_bucket(token, self.dimension)
            vector[bucket] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # 一个 token 都没有（空串 / 纯标点 / 纯 emoji）：给零向量而不是硬凑一个方向。
            # 零向量在任何检索里得分都是 0，调用方据此认定"这条 query 没有向量信号"
            return vector
        return [value / norm for value in vector]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text or "") for text in texts]


class ApiEmbedder(Embedder):
    """OpenAI 兼容的 /embeddings 客户端（SiliconFlow 的 bge-m3 走这套）。

    刻意**不用** async：检索路径（BM25Index 的 search 系列）是同步的，
    为一次 embedding 把整条链路染成 async 得不偿失。每次调用自建 httpx.Client，
    与 LlmClient/CatalogClient "一个实例一份配置、无长连接"的既有风格一致。
    """

    is_fake = False

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self.dimension = int(settings.embedding_dimension)
        self.name = f"api:{settings.embedding_model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _API_BATCH_SIZE):
            vectors.extend(self._embed_batch(texts[start : start + _API_BATCH_SIZE]))
        if len(vectors) != len(texts):
            raise UpstreamError(
                f"embedding 返回条数与请求不符: 请求 {len(texts)} 条，拿到 {len(vectors)} 条"
            )
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        key = (self._settings.embedding_api_key or "").strip()
        if not key:
            # 与 CatalogClient 同一套 fail-closed：没 key 就不发请求，
            # 也绝不偷偷换成哈希向量冒充成功——那会让评估报告说谎
            raise UpstreamError("EMBEDDING_API_KEY 未配置")
        body = {"model": self._settings.embedding_model, "input": list(texts)}
        headers = {"Authorization": f"Bearer {key}"}
        try:
            with httpx.Client(
                timeout=self._settings.embedding_timeout_seconds, transport=self._transport
            ) as client:
                response = client.post(self._settings.embeddings_url, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                f"embedding 上游超时({self._settings.embedding_timeout_seconds}s): "
                f"{type(exc).__name__}"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"embedding 上游连接失败: {type(exc).__name__}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise UpstreamError(f"embedding 上游返回 HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError("embedding 上游返回的不是合法 JSON") from exc
        return self._parse(payload)

    def _parse(self, payload) -> list[list[float]]:
        if not isinstance(payload, dict):
            raise UpstreamError("embedding 上游返回结构不符: 顶层不是对象")
        data = payload.get("data")
        if not isinstance(data, list):
            raise UpstreamError("embedding 上游返回结构不符: data 不是数组")
        vectors: list[list[float]] = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise UpstreamError("embedding 上游返回结构不符: data 元素不是对象")
            raw = item.get("embedding")
            if not isinstance(raw, list) or not raw:
                raise UpstreamError("embedding 上游返回结构不符: embedding 不是非空数组")
            # 上游返回的位置可能乱序（OpenAI 协议有 index 字段），按 index 归位；
            # 缺 index 时退回响应顺序
            vectors.append([float(value) for value in raw])
        if data and isinstance(data[0], dict) and isinstance(data[0].get("index"), int):
            order = [int(item.get("index", pos)) for pos, item in enumerate(data)]
            if sorted(order) != list(range(len(vectors))):
                raise UpstreamError("embedding 上游返回的 index 不是 0..n-1 的排列")
            ranked = [None] * len(vectors)
            for position, index in enumerate(order):
                ranked[index] = vectors[position]
            return [vector for vector in ranked if vector is not None]
        return vectors


def build_embedder(settings: Settings) -> Embedder:
    """有 key 走真实 embedding，没 key 走哈希降级。

    降级只记一条 warning，不抛错：没有 key 是本地开发的常态，
    但真实调用的那次必须能被日志抓到。
    """
    if settings.embedding_is_fake:
        log.info("embedding 走哈希降级（未配置 EMBEDDING_API_KEY）dim=%d", settings.embedding_dimension)
        return HashingEmbedder(settings.embedding_dimension)
    return ApiEmbedder(settings)
