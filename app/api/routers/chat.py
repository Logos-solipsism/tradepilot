"""POST /chat：Java 后端转发进来的唯一入口（浏览器不直接调它）。"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends

from app.agent.loop import run_tool_loop
from app.api.deps import (
    TicketContext,
    get_llm_client,
    get_session_store,
    get_toolbox,
    require_ticket,
)
from app.api.schemas import ApiEnvelope, ChatData, ChatRequestBody, Usage
from app.llm.client import LlmClient
from app.session.base import SessionStore
from app.tools.catalog_tools import CatalogToolbox

log = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ApiEnvelope)
async def chat(
    body: ChatRequestBody,
    ctx: TicketContext = Depends(require_ticket),
    store: SessionStore = Depends(get_session_store),
    llm: LlmClient = Depends(get_llm_client),
    toolbox: CatalogToolbox = Depends(get_toolbox),
) -> ApiEnvelope:
    session_id = body.session_id or str(uuid.uuid4())
    trace_id = body.trace_id or ctx.trace_id

    history = await store.load(ctx.user_id, session_id)

    # UpstreamError 由 main.py 的异常处理器转成真实 HTTP 502
    outcome = await run_tool_loop(llm, toolbox, history, body.message)

    # 只落用户与最终回答：工具往返不写进会话，否则历史窗口会被工具 JSON 塞满，
    # 后面每一轮都要为这些中间结果重复付 token。
    await store.append(
        ctx.user_id,
        session_id,
        [
            {"role": "user", "content": body.message},
            {"role": "assistant", "content": outcome.answer},
        ],
    )
    log.info(
        "chat done user_id=%s session_id=%s trace_id=%s history=%d tools=%d rounds=%d",
        ctx.user_id,
        session_id,
        trace_id,
        len(history),
        len(outcome.tool_calls),
        len(outcome.tool_calls) + 1,
    )
    return ApiEnvelope(
        code=0,
        message="ok",
        data=ChatData(
            answer=outcome.answer,
            session_id=session_id,
            tool_calls=[
                {"name": t.name, "arguments": t.arguments, "ok": t.ok}
                for t in outcome.tool_calls
            ],
            usage=Usage(
                input=outcome.usage.input,
                output=outcome.usage.output,
                total=outcome.usage.total,
            ),
        ),
    )
