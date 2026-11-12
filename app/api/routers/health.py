"""GET /health：Java 侧探活用，不需要票据。"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_catalog_index, get_llm_client, get_session_store, get_settings_dep
from app.api.schemas import HealthData
from app.config import Settings
from app.llm.client import LlmClient
from app.retrieval.catalog import CatalogIndex
from app.session.base import SessionStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthData)
async def health(
    settings: Settings = Depends(get_settings_dep),
    llm: LlmClient = Depends(get_llm_client),
    store: SessionStore = Depends(get_session_store),
    catalog: CatalogIndex = Depends(get_catalog_index),
) -> HealthData:
    return HealthData(
        ok=True,
        # 如实反映当前是真是假，绝不永远返回 real
        llm="fake" if llm.is_fake else "real",
        version=settings.version,
        sessions="redis" if store.backend == "redis" else "memory",
        # 只读计数，不触发拉取：探活接口不许打上游
        search="ready" if catalog.indexed_count > 0 else "empty",
    )
