"""POST /search：AI 搜索的检索入口（纯词法，不调模型）。

检索走**两路 RRF 融合**（原文 token 的 BM25 + 泛化词的同源 BM25），
因此响应里的 score 是**融合分**（量级 ~0.03），不是 BM25 分、也不是相关度百分比。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Request

from app.api.deps import TicketContext, get_catalog_index, require_ticket
from app.api.schemas import SearchData, SearchEnvelope, SearchHit, SearchRequestBody
from app.retrieval.bm25 import BM25Index
from app.retrieval.catalog import CatalogIndex

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


@router.post("/search", response_model=SearchEnvelope)
async def search(
    body: SearchRequestBody,
    ctx: TicketContext = Depends(require_ticket),
    catalog: CatalogIndex = Depends(get_catalog_index),
    cache: _Bm25Cache = Depends(get_bm25_cache),
) -> SearchEnvelope:
    # UpstreamError 由 main.py 的异常处理器转成真实 HTTP 502：
    # 目录拉取失败绝不能降级成空结果——那等于对调用方说假话
    docs = await catalog.docs()
    hits: list[SearchHit] = []
    if docs:
        index = cache.get(docs)
        hits = [
            SearchHit(
                id=doc.id,
                title=doc.title,
                min_price=doc.min_price,
                sku_count=doc.sku_count,
                score=score,
            )
            for doc, score in index.search_fused(body.query, top_k=body.top_k)
        ]
    indexed = len(docs)
    log.info(
        "search done user_id=%s trace_id=%s indexed=%d hits=%d",
        ctx.user_id,
        body.trace_id or ctx.trace_id,
        indexed,
        len(hits),
    )
    # query 原样回显：不 trim、不改写
    return SearchEnvelope(
        code=0,
        message="ok",
        data=SearchData(query=body.query, hits=hits, indexed=indexed),
    )
