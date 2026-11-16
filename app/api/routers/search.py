"""POST /search：AI 搜索的检索入口（纯词法+向量，不调生成模型）。

检索走**三路 RRF 融合**（原文 token 的 BM25 + 泛化词的同源 BM25 + 向量余弦），
因此响应里的 score 是**融合分**（量级 ~0.03），不是 BM25 分、不是余弦分、
也不是相关度百分比。

**向量路随时可能缺席，缺席时退化为两路而不是失败**：文档向量要 embed 全量商品、
query 向量要调一次上游，两者都可能挂。两路结果本身是正确且可用的，向量只是排序增强，
所以这里统一降级两路 + WARNING，绝不把"向量没建起来"变成 503、也绝不变空结果。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.api.deps import (
    TicketContext,
    get_catalog_index,
    get_vector_cache,
    require_ticket,
)
from app.api.schemas import SearchData, SearchEnvelope, SearchHit, SearchRequestBody
from app.errors import UpstreamError
from app.retrieval.bm25 import BM25Index
from app.retrieval.catalog import CatalogIndex, ProductDoc
from app.retrieval.vector_cache import VectorCache

log = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


class _Bm25Cache:
    """目录不变就不重建索引：BM25 建索引是 O(文档数)，不能每个请求重做。

    以 CatalogIndex 交出的 list 对象身份判断是否换代——CatalogIndex 每次
    刷新都会赋一个新 list，身份变化即意味着要重建。
    """

    def __init__(self) -> None:
        self._source: list | None = None
        self._index: BM25Index | None = None

    def get(self, docs: list) -> BM25Index:
        if self._index is None or self._source is not docs:
            self._index = BM25Index(docs)
            self._source = docs
        return self._index


def get_bm25_cache(request: Request) -> _Bm25Cache:
    cache = getattr(request.app.state, "bm25_cache", None)
    if cache is None:
        cache = _Bm25Cache()
        request.app.state.bm25_cache = cache
    return cache


def _search_three_way(
    index: BM25Index, vectors, body: SearchRequestBody, ctx: TicketContext
) -> list[tuple[ProductDoc, float]] | None:
    """走三路；query 向量这一步失败时返回 None 让调用方退回两路。

    **只接 UpstreamError**（上游超时/非 2xx/结构不符——这才是"embedding 调用失败"）。
    其他的 ValueError 不接：那是维度不符或文档集守卫，属于"代码/状态出错了"，
    降级会把一个结构性缺陷变成一条永远存在的 WARNING，正好和守卫要"顶到脸上"的意图相反。
    """
    try:
        return index.search_fused_three(body.query, vectors, top_k=body.top_k)
    except UpstreamError as exc:
        log.warning(
            "query 向量失败，本次请求降级为两路 BM25 user_id=%s trace_id=%s is_fake=%s: %s: %s",
            ctx.user_id,
            body.trace_id or ctx.trace_id,
            vectors.is_fake,
            type(exc).__name__,
            exc,
        )
        return None


@router.post("/search", response_model=SearchEnvelope)
async def search(
    body: SearchRequestBody,
    ctx: TicketContext = Depends(require_ticket),
    catalog: CatalogIndex = Depends(get_catalog_index),
    cache: _Bm25Cache = Depends(get_bm25_cache),
    vector_cache: VectorCache = Depends(get_vector_cache),
) -> SearchEnvelope:
    # UpstreamError 由 main.py 的异常处理器转成真实 HTTP 502：
    # 目录拉取失败绝不能降级成空结果——那等于对调用方说假话
    docs = await catalog.docs()
    hits: list[SearchHit] = []
    route = "two"
    if docs:
        index = cache.get(docs)
        # 向量索引**永远不在这里同步建**：全量 embed 要 2-3 秒，同步建等于让第一个
        # 用户白等。没建好就先起后台线程，本次请求照常用两路返回稳定结果。
        vectors = vector_cache.peek(docs)
        if vectors is None:
            vector_cache.request_build(docs)
        fused = None
        if vectors is not None:
            fused = _search_three_way(index, vectors, body, ctx)
            if fused is not None:
                route = "three"
        if fused is None:
            # 向量路缺席（没建好 / 建失败 / 本次 query 失败）→ 两路。
            # 这不是降级的近似：search_fused 就是接入三路之前线上一直在跑的那条路
            fused = index.search_fused(body.query, top_k=body.top_k)
        hits = [
            SearchHit(
                id=doc.id,
                title=doc.title,
                min_price=doc.min_price,
                sku_count=doc.sku_count,
                score=score,
            )
            for doc, score in fused
        ]
    # 向量路的状态**只在切换点打日志**（不随请求刷屏）。降级那一次一定是 WARNING，
    # 因此"这一代目录退化了"不会只活在某个被淹没的 debug 行里。
    if docs:
        vector_cache.log_transition(vector_cache.state(docs))
    indexed = len(docs)
    log.info(
        "search done user_id=%s trace_id=%s indexed=%d hits=%d route=%s vectors=%s is_fake=%s",
        ctx.user_id,
        body.trace_id or ctx.trace_id,
        indexed,
        len(hits),
        route,
        vector_cache.state(docs) if docs else "idle",
        vector_cache.is_fake,
    )
    # query 原样回显：不 trim、不改写
    return SearchEnvelope(
        code=0,
        message="ok",
        data=SearchData(query=body.query, hits=hits, indexed=indexed),
    )
