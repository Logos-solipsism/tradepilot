"""商品目录客户端 + 进程内索引缓存。

目录数据来自 Java 侧 trademall 的 /api/agent/products（只读）。
本模块只做两件事：把上游分页翻完（CatalogClient），
和在进程内按 TTL 缓存结果（CatalogIndex，不是数据库）。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from app.config import Settings
from app.errors import UpstreamError

log = logging.getLogger(__name__)

# 目录接口是内部调用，延迟要求比模型调用低；单独给一个常量，不扩 Settings 字段
CATALOG_TIMEOUT_SECONDS = 10.0
# 上游硬上限：limit 传更大也只返回 100 条
MAX_PAGE_SIZE = 100


@dataclass(frozen=True)
class ProductDoc:
    """目录快照的一条商品。字段名逐字对齐上游 /api/agent/products。"""

    id: int
    title: str
    # 上游 description 可能是 null；在这里归一成空串，后面所有字符串处理都不用判空
    description: str = ""
    origin_price: int | None = None
    cover_color: str | None = None
    sku_count: int = 0
    min_price: int | None = None


@dataclass(frozen=True)
class SkuDoc:
    """详情接口里的一条 SKU。字段名逐字对齐上游 /api/agent/products/{id}。"""

    sku_id: int
    sku_code: str
    spec_desc: str
    price: int | None
    stock: int


@dataclass(frozen=True)
class ProductDetail:
    """单个商品的详情快照（含全部在售 SKU）。

    上游 description / coverColor / originPrice 都可能为 null，这里的归一规则与
    ProductDoc 一致：字符串 null → 空串，数字 null → None，不做任何猜测性补值。
    """

    id: int
    title: str
    description: str = ""
    origin_price: int | None = None
    cover_color: str | None = None
    skus: tuple[SkuDoc, ...] = ()


def _as_int(value, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _as_opt_int(value) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _as_text(value) -> str:
    return value if isinstance(value, str) else ""


def _clamp_page_size(page_size: int) -> int:
    try:
        size = int(page_size)
    except (TypeError, ValueError):
        return MAX_PAGE_SIZE
    return max(1, min(MAX_PAGE_SIZE, size))


class CatalogClient:
    """一个实例 = 一份配置。每次请求自建 AsyncClient，与 LlmClient 保持一致。"""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    @property
    def products_url(self) -> str:
        return self._settings.catalog_products_url

    async def fetch_all(self, page_size: int = MAX_PAGE_SIZE) -> list[ProductDoc]:
        """翻完所有页，返回全量商品。任何异常路径都抛 UpstreamError，绝不返回半包。"""
        size = _clamp_page_size(page_size)
        token = (self._settings.agent_service_token or "").strip()
        if not token:
            # fail-closed：没配 token 就不发请求。空目录会被 /search 当成"真的没商品"，
            # 那是对调用方说假话，所以这里必须显式失败。
            raise UpstreamError("AGENT_SERVICE_TOKEN 未配置")

        docs: list[ProductDoc] = []
        offset = 0
        while True:
            page = await self._fetch_page(size, offset, token)
            docs.extend(page)
            # 结束标志是"本页返回条数 < limit"，不是"本页为空"
            if len(page) < size:
                break
            offset += size
        return docs

    async def _fetch_page(self, limit: int, offset: int, token: str) -> list[ProductDoc]:
        params = {"limit": limit, "offset": offset}
        headers = {"X-Agent-Token": token}
        try:
            async with httpx.AsyncClient(
                timeout=CATALOG_TIMEOUT_SECONDS, transport=self._transport
            ) as client:
                response = await client.get(self.products_url, params=params, headers=headers)
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                f"商品目录上游超时({CATALOG_TIMEOUT_SECONDS}s): {type(exc).__name__}"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"商品目录上游连接失败: {type(exc).__name__}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise UpstreamError(f"商品目录上游返回 HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise UpstreamError("商品目录上游返回的不是合法 JSON") from exc
        return self._parse_page(body)

    @staticmethod
    def _parse_page(body) -> list[ProductDoc]:
        if not isinstance(body, dict):
            raise UpstreamError("商品目录上游返回结构不符: 顶层不是对象")
        code = body.get("code")
        if code != 0:
            raise UpstreamError(f"商品目录上游返回业务错误 code={code!r}")
        data = body.get("data")
        if not isinstance(data, list):
            raise UpstreamError("商品目录上游返回结构不符: data 不是数组")
        docs: list[ProductDoc] = []
        for item in data:
            if not isinstance(item, dict):
                raise UpstreamError("商品目录上游返回结构不符: data 元素不是对象")
            docs.append(
                ProductDoc(
                    id=_as_int(item.get("id")),
                    title=_as_text(item.get("title")),
                    # null → 空串：下游字符串处理一律不用判 None
                    description=_as_text(item.get("description")),
                    origin_price=_as_opt_int(item.get("originPrice")),
                    cover_color=item.get("coverColor")
                    if isinstance(item.get("coverColor"), str)
                    else None,
                    sku_count=_as_int(item.get("skuCount")),
                    min_price=_as_opt_int(item.get("minPrice")),
                )
            )
        return docs

    async def fetch_detail(self, product_id: int) -> ProductDetail | None:
        """查单个商品详情。

        返回 None 只表示**上游业务上说"没有这个商品"**（code=400：商品不存在或已下架），
        那是正常业务结果而不是故障，调用方据此回答"没找到"即可；其余任何异常路径都
        抛 UpstreamError，绝不把故障伪装成"没有这个商品"——那是对调用方说假话。
        """
        token = (self._settings.agent_service_token or "").strip()
        if not token:
            # 与 fetch_all 同一 fail-closed：没配 token 就不发请求
            raise UpstreamError("AGENT_SERVICE_TOKEN 未配置")

        url = self._settings.agent_service_base_url.rstrip("/") + f"/api/agent/products/{int(product_id)}"
        headers = {"X-Agent-Token": token}
        try:
            async with httpx.AsyncClient(
                timeout=CATALOG_TIMEOUT_SECONDS, transport=self._transport
            ) as client:
                response = await client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                f"商品详情上游超时({CATALOG_TIMEOUT_SECONDS}s): {type(exc).__name__}"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"商品详情上游连接失败: {type(exc).__name__}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            raise UpstreamError(f"商品详情上游返回 HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise UpstreamError("商品详情上游返回的不是合法 JSON") from exc
        return self._parse_detail(body)

    @staticmethod
    def _parse_detail(body) -> ProductDetail | None:
        if not isinstance(body, dict):
            raise UpstreamError("商品详情上游返回结构不符: 顶层不是对象")
        code = body.get("code")
        if code == 400:
            # 业务上的"没有这个商品"：HTTP 仍是 200，code 才是判定依据
            return None
        if code != 0:
            raise UpstreamError(f"商品详情上游返回业务错误 code={code!r}")
        data = body.get("data")
        if not isinstance(data, dict):
            raise UpstreamError("商品详情上游返回结构不符: data 不是对象")
        skus: list[SkuDoc] = []
        raw_skus = data.get("skus")
        for item in raw_skus if isinstance(raw_skus, list) else []:
            if not isinstance(item, dict):
                continue
            skus.append(
                SkuDoc(
                    sku_id=_as_int(item.get("skuId")),
                    sku_code=_as_text(item.get("skuCode")),
                    spec_desc=_as_text(item.get("specDesc")),
                    price=_as_opt_int(item.get("price")),
                    stock=_as_int(item.get("stock")),
                )
            )
        return ProductDetail(
            id=_as_int(data.get("id")),
            title=_as_text(data.get("title")),
            description=_as_text(data.get("description")),
            origin_price=_as_opt_int(data.get("originPrice")),
            cover_color=data.get("coverColor")
            if isinstance(data.get("coverColor"), str)
            else None,
            skus=tuple(skus),
        )

    async def aclose(self) -> None:
        """每次调用自建连接，无常驻资源需要释放（与 LlmClient 一致）。"""
        return None


def _now() -> float:
    """单调时钟；单独成函数便于测试替换，TTL 不受系统时间回拨影响。"""
    return time.monotonic()


class CatalogIndex:
    """进程内全量目录缓存：TTL 内直接复用，过期后整份重建。"""

    def __init__(self, client: CatalogClient, *, ttl_seconds: int) -> None:
        self._client = client
        self._ttl_seconds = max(0, int(ttl_seconds))
        self._lock = asyncio.Lock()
        self._docs: list[ProductDoc] | None = None
        self._last_refreshed_at: float | None = None

    @property
    def indexed_count(self) -> int:
        return len(self._docs) if self._docs is not None else 0

    @property
    def last_refreshed_at(self) -> float | None:
        return self._last_refreshed_at

    def _is_fresh(self, now: float) -> bool:
        if self._docs is None or self._last_refreshed_at is None:
            return False
        return (now - self._last_refreshed_at) < self._ttl_seconds

    async def docs(self) -> list[ProductDoc]:
        """首次拉全量并缓存；超 TTL 后重建。并发安全：同一时刻只有一个真去拉。

        关键：上游失败直接抛 UpstreamError，**绝不**把失败结果写成空列表缓存——
        空列表只有在上游真的返回 0 条时才是合法缓存值。
        """
        now = _now()
        cached = self._docs
        if cached is not None and self._is_fresh(now):
            return cached

        async with self._lock:
            # 等锁期间别人可能已经拉完了，这里再判一次新鲜度（double-checked）
            now = _now()
            cached = self._docs
            if cached is not None and self._is_fresh(now):
                return cached

            fetched = await self._client.fetch_all()
            self._docs = fetched
            self._last_refreshed_at = _now()
            log.info("catalog refreshed count=%d", len(fetched))
            return fetched
