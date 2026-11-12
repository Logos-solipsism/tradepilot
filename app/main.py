"""应用装配：create_app() 建实例，模块级 app 供 uvicorn 直接用。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routers import chat, health, search
from app.config import Settings, get_settings
from app.errors import AppError
from app.llm.client import LlmClient
from app.retrieval.catalog import CatalogClient, CatalogIndex
from app.session.base import SessionStore
from app.session.factory import create_session_store

log = logging.getLogger(__name__)


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
        )


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: LlmClient | None = None,
    session_store: SessionStore | None = None,
    catalog_client: CatalogClient | None = None,
) -> FastAPI:
    _configure_logging()
    resolved = settings if settings is not None else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.session_store is None:
            app.state.session_store = await create_session_store(app.state.settings)
        if app.state.llm_client is None:
            app.state.llm_client = LlmClient(app.state.settings)
        if app.state.catalog_client is None:
            app.state.catalog_client = CatalogClient(app.state.settings)
        if app.state.catalog_index is None:
            app.state.catalog_index = CatalogIndex(
                app.state.catalog_client,
                ttl_seconds=app.state.settings.catalog_cache_ttl_seconds,
            )
        yield
        for client in (app.state.llm_client, app.state.catalog_client):
            if client is not None and hasattr(client, "aclose"):
                await client.aclose()

    app = FastAPI(title="tradepilot agent", version=resolved.version, lifespan=lifespan)
    app.state.settings = resolved
    app.state.llm_client = llm_client
    app.state.session_store = session_store
    app.state.catalog_client = catalog_client
    # 索引在 lifespan 里按注入的 client 建；测试可直接覆盖 app.state.catalog_index
    app.state.catalog_index = None

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        # 真实 HTTP 状态码：Java 侧靠状态码判断，绝不能包成 200 + code
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.code, "message": exc.message, "data": None},
        )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(search.router)
    return app


app = create_app()
