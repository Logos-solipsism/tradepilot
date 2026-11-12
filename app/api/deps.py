"""FastAPI 依赖：票据校验（唯一可信身份来源）、trace_id、共享组件。"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request

from app.config import Settings
from app.errors import TicketInvalidError
from app.llm.client import LlmClient
from app.retrieval.catalog import CatalogClient, CatalogIndex
from app.security.ticket import TicketError, verify
from app.session.base import SessionStore
from app.tools.catalog_tools import CatalogToolbox


@dataclass(frozen=True)
class TicketContext:
    user_id: int
    trace_id: str | None


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store


def get_llm_client(request: Request) -> LlmClient:
    return request.app.state.llm_client


def get_catalog_index(request: Request) -> CatalogIndex:
    """通常由 lifespan 建好；若没跑（极端场景）就地补一个，别让 /health 500。"""
    index = getattr(request.app.state, "catalog_index", None)
    if index is None:
        settings = request.app.state.settings
        client = getattr(request.app.state, "catalog_client", None)
        if client is None:
            client = CatalogClient(settings)
            request.app.state.catalog_client = client
        index = CatalogIndex(client, ttl_seconds=settings.catalog_cache_ttl_seconds)
        request.app.state.catalog_index = index
    return index


async def require_ticket(
    request: Request,
    settings: Settings = Depends(get_settings_dep),
) -> TicketContext:
    """校验 X-Internal-Ticket，返回调用方 userId。票据是 userId 的唯一来源。"""
    ticket = request.headers.get("X-Internal-Ticket")
    try:
        user_id = verify(ticket, secret=settings.internal_secret)
    except TicketError as exc:
        # 不把具体失败原因回给调用方：401 的语义只有"你没被信任"
        raise TicketInvalidError() from exc
    return TicketContext(user_id=user_id, trace_id=request.headers.get("X-Trace-Id"))

def get_toolbox(request: Request) -> CatalogToolbox:
    """给导购 Agent 用的工具集合。

    按索引对象身份复用：toolbox 内部缓存了 BM25 索引，每次请求重造就等于每次重建
    （与 /search 的 _Bm25Cache 同一策略）。索引被替换时缓存随之失效。
    """
    index = get_catalog_index(request)
    cached = getattr(request.app.state, "toolbox", None)
    if cached is not None and cached.index is index:
        return cached

    client = getattr(request.app.state, "catalog_client", None)
    if client is None:
        client = CatalogClient(request.app.state.settings)
        request.app.state.catalog_client = client
    toolbox = CatalogToolbox(index, client)
    request.app.state.toolbox = toolbox
    return toolbox

