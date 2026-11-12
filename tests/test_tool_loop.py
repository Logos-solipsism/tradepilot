"""Agent 工具循环：全离线，LLM 与工具都用桩，禁止真打网络。"""

from __future__ import annotations

import asyncio
import json

from app.agent.loop import (
    MAX_ROUNDS_ANSWER,
    MAX_TOOL_ROUNDS,
    ChatOutcome,
    ToolTrace,
    run_tool_loop,
)
from app.agent.prompt import SYSTEM_PROMPT
from app.llm.client import FAKE_ANSWER, LlmResult, LlmUsage, ToolCall
from app.tools.catalog_tools import TOOL_SPECS


def _call(call_id="call-1", name="search_products", arguments='{"query":"跑步鞋"}') -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


class _StubLlm:
    """按脚本依次返回；脚本项可以是 LlmResult，也可以是异常实例。"""

    is_fake = False

    def __init__(self, script) -> None:
        self._script = list(script)
        self.calls: list[list[dict]] = []
        self.tools_seen: list[object] = []

    async def complete(self, messages, *, tools=None):
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        item = self._script[min(len(self.calls) - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(messages)
        return item


class _FakeLlm:
    is_fake = True

    def __init__(self) -> None:
        self.tools_seen: list[object] = []

    async def complete(self, messages, *, tools=None):
        self.tools_seen.append(tools)
        return LlmResult(content=FAKE_ANSWER, usage=LlmUsage())


class _StubToolbox:
    def __init__(self, result=('{"count":0,"items":[]}', True)) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def invoke(self, name: str, arguments: str):
        self.calls.append((name, arguments))
        return self.result


def _run(llm, toolbox=None, *, history=None, message="想买跑步鞋", **kwargs) -> ChatOutcome:
    return asyncio.run(
        run_tool_loop(llm, toolbox or _StubToolbox(), history or [], message, **kwargs)
    )


def test_single_tool_round_then_answer():
    llm = _StubLlm(
        [
            LlmResult(content="", usage=LlmUsage(1, 2, 3), tool_calls=[_call()]),
            LlmResult(content="129 元的轻量跑步鞋还在售。", usage=LlmUsage(4, 5, 9)),
        ]
    )
    box = _StubToolbox(('{"count":1,"items":[]}', True))

    outcome = _run(llm, box)

    assert outcome.answer == "129 元的轻量跑步鞋还在售。"
    assert outcome.tool_calls == [ToolTrace(name="search_products", arguments='{"query":"跑步鞋"}', ok=True)]
    assert box.calls == [("search_products", '{"query":"跑步鞋"}')]
    assert len(llm.calls) == 2


def test_two_tool_rounds_before_answer():
    llm = _StubLlm(
        [
            LlmResult(content="", tool_calls=[_call()]),
            LlmResult(content="", tool_calls=[_call("call-2", "get_product_detail", '{"product_id":7}')]),
            LlmResult(content="40 码有 5 件，129 元。"),
        ]
    )
    box = _StubToolbox()

    outcome = _run(llm, box)

    assert outcome.answer == "40 码有 5 件，129 元。"
    assert [t.name for t in outcome.tool_calls] == ["search_products", "get_product_detail"]
    assert len(box.calls) == 2
    assert len(llm.calls) == 3


def test_no_tool_calls_answers_immediately():
    llm = _StubLlm([LlmResult(content="你好，想找点什么？")])
    box = _StubToolbox()

    outcome = _run(llm, box)

    assert outcome.answer == "你好，想找点什么？"
    assert outcome.tool_calls == []
    assert box.calls == []
    assert len(llm.calls) == 1


def test_fake_mode_sends_no_tools():
    llm = _FakeLlm()

    outcome = _run(llm, _StubToolbox())

    # fake 模式不该假装调工具：既不发 tools，也不可能有 tool_calls
    assert llm.tools_seen == [None]
    assert outcome.answer == FAKE_ANSWER
    assert outcome.tool_calls == []


def test_real_mode_sends_tool_specs():
    llm = _StubLlm([LlmResult(content="好的")])

    _run(llm)

    assert llm.tools_seen == [TOOL_SPECS]


def test_max_rounds_ignores_further_tool_calls_and_answers():
    loop_calls = [LlmResult(content="", tool_calls=[_call(f"call-{i}")]) for i in range(MAX_TOOL_ROUNDS)]
    # 用满轮次后那次收口请求即使又给 tool_calls，也必须忽略
    llm = _StubLlm([*loop_calls, LlmResult(content="兜底查到的结论。", tool_calls=[_call("call-x")])])
    box = _StubToolbox()

    outcome = _run(llm, box)

    assert outcome.answer == "兜底查到的结论。"
    assert len(outcome.tool_calls) == MAX_TOOL_ROUNDS
    assert len(box.calls) == MAX_TOOL_ROUNDS
    # 1 + MAX_TOOL_ROUNDS 次模型调用是成本上界
    assert len(llm.calls) == MAX_TOOL_ROUNDS + 1


def test_max_rounds_empty_answer_falls_back_to_fixed_text():
    loop_calls = [LlmResult(content="", tool_calls=[_call(f"call-{i}")]) for i in range(MAX_TOOL_ROUNDS)]
    llm = _StubLlm([*loop_calls, LlmResult(content="")])

    outcome = _run(llm)

    assert outcome.answer == MAX_ROUNDS_ANSWER


def test_failed_tool_keeps_loop_going():
    llm = _StubLlm(
        [
            LlmResult(content="", tool_calls=[_call()]),
            LlmResult(content="系统繁忙，请稍后再试。"),
        ]
    )
    box = _StubToolbox(('{"error":"商品服务暂时不可用，请稍后再试"}', False))

    outcome = _run(llm, box)

    assert outcome.answer == "系统繁忙，请稍后再试。"
    assert outcome.tool_calls == [ToolTrace(name="search_products", arguments='{"query":"跑步鞋"}', ok=False)]
    assert len(llm.calls) == 2


def test_usage_is_summed_across_rounds():
    llm = _StubLlm(
        [
            LlmResult(content="", usage=LlmUsage(10, 20, 30), tool_calls=[_call()]),
            LlmResult(content="", usage=LlmUsage(1, 2, 3), tool_calls=[_call("call-2")]),
            LlmResult(content="好了", usage=LlmUsage(100, 200, 300)),
        ]
    )

    outcome = _run(llm)

    assert (outcome.usage.input, outcome.usage.output, outcome.usage.total) == (111, 222, 333)


def test_system_prompt_is_first_and_single():
    llm = _StubLlm(
        [
            LlmResult(content="", tool_calls=[_call()]),
            LlmResult(content="好了"),
        ]
    )

    _run(llm, history=[{"role": "user", "content": "在吗"}, {"role": "assistant", "content": "在"}])

    first = llm.calls[0]
    assert first[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert [m["role"] for m in first].count("system") == 1
    assert [m["role"] for m in first] == ["system", "user", "assistant", "user"]
    assert first[-1]["content"] == "想买跑步鞋"


def test_tool_messages_are_appended_before_next_completion():
    llm = _StubLlm(
        [
            LlmResult(content="", tool_calls=[_call()]),
            LlmResult(content="好了"),
        ]
    )
    box = _StubToolbox(('{"count":1,"items":[]}', True))

    _run(llm, box)

    second = llm.calls[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assistant = second[2]
    # content 为空必须写 None：部分上游拒绝空串 assistant 消息
    assert assistant["content"] is None
    assert assistant["tool_calls"] == [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "search_products", "arguments": '{"query":"跑步鞋"}'},
        }
    ]
    tool_msg = second[3]
    assert tool_msg["tool_call_id"] == "call-1"
    assert json.loads(tool_msg["content"])["count"] == 1


def test_multiple_tool_calls_in_one_round_all_run():
    llm = _StubLlm(
        [
            LlmResult(
                content="",
                tool_calls=[_call("a"), _call("b", "get_product_detail", '{"product_id":7}')],
            ),
            LlmResult(content="都查完了"),
        ]
    )
    box = _StubToolbox()

    outcome = _run(llm, box)

    assert len(box.calls) == 2
    assert [t.ok for t in outcome.tool_calls] == [True, True]
    roles = [m["role"] for m in llm.calls[1]]
    assert roles == ["system", "user", "assistant", "tool", "tool"]
