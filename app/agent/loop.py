"""导购 Agent 的工具循环：模型 ⇄ 工具，直到给出最终回答或达到轮次上限。

为什么要有轮次上限：模型可能反复调同一个工具却不收敛（尤其工具一直返回空结果时），
没有上限就是一次请求无限次打上游、按 token 无限计费。MAX_TOOL_ROUNDS = 3 意味着
一次 /chat 最多 1 + 3 = 4 次模型调用，成本上界是可算的。

这里不碰会话存储：历史怎么落盘是 router 的事；本模块只负责"这一轮对话怎么跑完"。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.agent.prompt import SYSTEM_PROMPT
from app.llm.client import LlmClient, LlmUsage
from app.tools.catalog_tools import TOOL_SPECS, CatalogToolbox

MAX_TOOL_ROUNDS = 3

MAX_ROUNDS_ANSWER = "这个问题我暂时没能查清楚，换个说法再问我一次？"


@dataclass(frozen=True)
class ToolTrace:
    """一次工具调用的审计记录：调了什么、参数原文、是否成功。"""

    name: str
    arguments: str
    ok: bool


@dataclass(frozen=True)
class ChatOutcome:
    answer: str
    tool_calls: list[ToolTrace] = field(default_factory=list)
    usage: LlmUsage = LlmUsage()


def _plus(total: LlmUsage, extra: LlmUsage) -> LlmUsage:
    return LlmUsage(
        input=total.input + extra.input,
        output=total.output + extra.output,
        total=total.total + extra.total,
    )


def _assistant_message(content: str, tool_calls: list) -> dict:
    """把带工具调用的 assistant 轮按 OpenAI 格式回灌。

    content 为空要写成 None 而不是 ""：部分上游会拒绝空串 assistant 消息。
    """
    return {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in tool_calls
        ],
    }


async def run_tool_loop(
    llm: LlmClient,
    toolbox: CatalogToolbox,
    history: list[dict],
    user_message: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    max_rounds: int = MAX_TOOL_ROUNDS,
) -> ChatOutcome:
    """跑一轮完整对话：system + 历史 + 本轮问题，必要时穿插工具调用。"""
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_message},
    ]

    if getattr(llm, "is_fake", False):
        # fake 模式不该假装调工具：不发 tools，也就不会有 tool_calls
        result = await llm.complete(messages)
        return ChatOutcome(answer=result.content, tool_calls=[], usage=result.usage)

    traces: list[ToolTrace] = []
    usage = LlmUsage()

    for _ in range(max_rounds):
        result = await llm.complete(messages, tools=TOOL_SPECS)
        usage = _plus(usage, result.usage)
        if not result.tool_calls:
            return ChatOutcome(answer=result.content, tool_calls=traces, usage=usage)

        messages.append(_assistant_message(result.content, result.tool_calls))
        for call in result.tool_calls:
            text, ok = await toolbox.invoke(call.name, call.arguments)
            traces.append(ToolTrace(name=call.name, arguments=call.arguments, ok=ok))
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": text}
            )

    # 用满轮次还没收敛：再问一次，只要文字。这次即使又给 tool_calls 也忽略，
    # 否则"上限"形同虚设 —— 必须有一次是无论如何都要收口的。
    final = await llm.complete(messages, tools=TOOL_SPECS)
    usage = _plus(usage, final.usage)
    return ChatOutcome(
        answer=final.content or MAX_ROUNDS_ANSWER,
        tool_calls=traces,
        usage=usage,
    )
