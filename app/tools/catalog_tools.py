"""商品目录工具：给模型用的两个只读工具（搜索 / 详情）。

设计要点：
- 工具**只读**：只有 GET，不写库、不碰 /api/agent 之外的任何地址。
- 参数校验与异常收敛都在本层完成：invoke() 永远返回 (文本, 是否成功)，不抛异常。
  模型拿到一段错误文本会自己换个说法再试；抛异常则整个 /chat 变 502，
  一次参数写错就打断整轮对话，代价不对等。
- 错误文本只给**可归类的摘要**，上游原因写进日志：与 Java 侧"只报可归类摘要"一致，
  避免把内部地址 / 上游报文带进模型上下文再漏给用户。
- 结果文本按 MAX_TOOL_RESULT_CHARS 截断：工具结果会原样回灌进模型上下文并按 token 计费，
  一条详情带上百个 SKU 就能把历史窗口挤掉，所以要给单条结果一个硬上限。
"""

from __future__ import annotations

import json
import logging

from app.errors import UpstreamError
from app.retrieval.bm25 import BM25Index
from app.retrieval.catalog import CatalogClient, CatalogIndex, ProductDetail

log = logging.getLogger(__name__)

# 单个工具结果回灌给模型的字符上限（理由见模块 docstring）
MAX_TOOL_RESULT_CHARS = 2000
TRUNCATED_SUFFIX = "...(已截断)"

DEFAULT_TOP_K = 8
MAX_TOP_K = 20
MAX_QUERY_CHARS = 200

TOOL_SEARCH = "search_products"
TOOL_DETAIL = "get_product_detail"

_ERR_BAD_JSON = '{"error":"工具参数不是合法 JSON"}'

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_SEARCH,
            "description": (
                "按自然语言关键词搜索在售商品，只读。"
                "用户问有没有某类商品、要推荐、要比价时用它。"
                "返回命中商品的 id、标题、最低价（单位：分）与 SKU 数量；"
                "count 为 0 表示确实没有匹配商品，不要自行编造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_QUERY_CHARS,
                        "description": "搜索关键词，通常是用户原话里的商品需求，不要带寒暄",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOP_K,
                        "default": DEFAULT_TOP_K,
                        "description": f"返回条数，默认 {DEFAULT_TOP_K}，最多 {MAX_TOP_K}",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": TOOL_DETAIL,
            "description": (
                "按商品 id 查详情与全部在售 SKU，只读。"
                "需要具体规格、价格或库存时用它，product_id 必须来自 search_products 的结果，"
                "不要自己编 id。返回价格单位是分。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "商品 id，来自 search_products 返回的 items[].id",
                    }
                },
                "required": ["product_id"],
                "additionalProperties": False,
            },
        },
    },
]


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False, separators=(",", ":"))


def _dump(payload: dict) -> str:
    """紧凑 JSON（无空格）：工具结果按 token 计费，缩进纯属浪费。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _truncate(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CHARS] + TRUNCATED_SUFFIX


def _as_int(value) -> int | None:
    """只认真正的整数：bool 是 int 的子类，必须挡掉（True 会变成 1）。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class CatalogToolbox:
    """把 CatalogIndex / CatalogClient 包成模型可调用的工具集合。

    BM25 索引按目录快照的对象身份缓存（与 /search 路由同一策略）：
    CatalogIndex 每次刷新都换新 list，身份变化即意味着要重建。
    """

    def __init__(self, index: CatalogIndex, client: CatalogClient) -> None:
        self._index = index
        self._client = client
        self._bm25_source: list | None = None
        self._bm25: BM25Index | None = None

    @property
    def index(self) -> CatalogIndex:
        return self._index

    def _search_index(self, docs: list) -> BM25Index:
        if self._bm25 is None or self._bm25_source is not docs:
            self._bm25 = BM25Index(docs)
            self._bm25_source = docs
        return self._bm25

    async def invoke(self, name: str, arguments: str) -> tuple[str, bool]:
        """执行一次工具调用，返回 (给模型看的结果文本, 是否成功)。

        任何失败都收敛成 (错误 JSON, False)，绝不向上抛：一次参数写错不该让整个 /chat 502。
        """
        try:
            raw = json.loads(arguments)
        except (TypeError, ValueError):
            return _ERR_BAD_JSON, False
        if not isinstance(raw, dict):
            return _ERR_BAD_JSON, False

        if name == TOOL_SEARCH:
            return await self._search(raw)
        if name == TOOL_DETAIL:
            return await self._detail(raw)
        return _error(f"未知工具: {name}"), False

    async def _search(self, raw: dict) -> tuple[str, bool]:
        query = raw.get("query")
        if not isinstance(query, str) or not query.strip():
            return _error("query 不能为空"), False
        if len(query) > MAX_QUERY_CHARS:
            return _error(f"query 长度不能超过 {MAX_QUERY_CHARS}"), False
        top_k = raw.get("top_k", DEFAULT_TOP_K)
        if top_k is None:
            top_k = DEFAULT_TOP_K
        parsed_k = _as_int(top_k)
        if parsed_k is None or parsed_k < 1 or parsed_k > MAX_TOP_K:
            return _error(f"top_k 必须是 1..{MAX_TOP_K} 的整数"), False

        try:
            docs = await self._index.docs()
        except UpstreamError as exc:
            return self._upstream_failure(TOOL_SEARCH, exc)

        items: list[dict] = []
        if docs:
            hits = self._search_index(docs).search_fused(query, top_k=parsed_k)
            items = [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "min_price": doc.min_price,
                    "sku_count": doc.sku_count,
                }
                for doc, _score in hits
            ]
        return _truncate(_dump({"count": len(items), "items": items})), True

    async def _detail(self, raw: dict) -> tuple[str, bool]:
        product_id = _as_int(raw.get("product_id"))
        if product_id is None or product_id < 1:
            return _error("product_id 必须是正整数"), False

        try:
            detail = await self._client.fetch_detail(product_id)
        except UpstreamError as exc:
            return self._upstream_failure(TOOL_DETAIL, exc)

        if detail is None:
            # 上游明确说"没有这个商品"：这是业务结果，不是故障
            return _error("没有找到这个商品"), False
        return _truncate(_dump(_detail_payload(detail))), True

    @staticmethod
    def _upstream_failure(tool: str, exc: UpstreamError) -> tuple[str, bool]:
        # 原因只进日志：模型与用户只需要知道"暂时不可用"，细节属于内部信息
        log.warning("工具上游失败 tool=%s reason=%s", tool, exc.message)
        return _error("商品服务暂时不可用，请稍后再试"), False


def _detail_payload(detail: ProductDetail) -> dict:
    """详情 → 给模型的最小字段集合。

    min_price 由 SKU 价格取最小值得出（上游详情接口只给 SKU 价格、不给商品级最低价）；
    一个可定价 SKU 都没有时为 None，不拿 originPrice 顶替 —— 那是原价，不是现价。
    """
    prices = [sku.price for sku in detail.skus if sku.price is not None]
    return {
        "id": detail.id,
        "title": detail.title,
        "description": detail.description,
        "min_price": min(prices) if prices else None,
        "skus": [
            {
                "sku_code": sku.sku_code,
                "spec_desc": sku.spec_desc,
                "price": sku.price,
                "stock": sku.stock,
            }
            for sku in detail.skus
        ],
    }
