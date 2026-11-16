"""GET /health：Java 侧探活用，不需要票据。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import (
    get_catalog_index,
    get_llm_client,
    get_session_store,
    get_settings_dep,
    get_vector_cache,
)
from app.api.schemas import HealthData
from app.config import Settings
from app.llm.client import LlmClient
from app.retrieval.catalog import CatalogIndex
from app.retrieval.vector_cache import STATE_IDLE, VectorCache
from app.session.base import SessionStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthData)
async def health(
    settings: Settings = Depends(get_settings_dep),
    llm: LlmClient = Depends(get_llm_client),
    store: SessionStore = Depends(get_session_store),
    catalog: CatalogIndex = Depends(get_catalog_index),
    vectors: VectorCache = Depends(get_vector_cache),
) -> HealthData:
    # 只读计数，不触发拉取：探活接口不许打上游。
    # cached_docs 拿的是"已经在内存里的那份快照"，没有就是 None——这里绝不调 docs()。
    docs = catalog.cached_docs
    # 没建过索引时没有"本代目录"可比，状态报 idle；建过才问 VectorCache 要真实状态。
    vector_state = vectors.state(docs) if docs else STATE_IDLE
    # 展示"真 embedding 还是哈希降级"只在**建过索引**时才有意义：
    # 没建过就说 off，而不是提前把 is_fake 报出去——那会让"配置了 key 但压根没跑"
    # 看起来和"配置了 key 且跑着真模型"一模一样。
    if vector_state == STATE_IDLE:
        vector = "off"
    else:
        vector = "fake" if vectors.is_fake else "real"
    return HealthData(
        ok=True,
        # 如实反映当前是真是假，绝不永远返回 real
        llm="fake" if llm.is_fake else "real",
        version=settings.version,
        sessions="redis" if store.backend == "redis" else "memory",
        search="ready" if catalog.indexed_count > 0 else "empty",
        vector=vector,
        vector_state=vector_state,
    )
