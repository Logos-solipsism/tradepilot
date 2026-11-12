"""LlmClient 的工具调用支持：全离线，httpx.MockTransport 注入，禁止真打网络。"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.errors import UpstreamError
from app.llm.client import LlmClient, ToolCall


def _settings(**overrides) -> Settings:
    base = dict(
        internal_secret="test-secret",
        llm_base_url="https://llm.test/v1",
        llm_api_key="sk-test",
        llm_model="test-model",
        llm_timeout_seconds=1.0,
    )
    base.update(overrides)
    return Settings(**base)


def _tool_body(arguments: str = '{"query":"跑步鞋"}') -> dict:
    """只调工具、不答话的上游响应：content 为 null。"""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search_products", "arguments": arguments},
                        }
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def _client(handler) -> LlmClient:
    return LlmClient(_settings(), transport=httpx.MockTransport(handler))


def test_tools_are_sent_with_tool_choice_when_provided():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_tool_body())

    tools = [{"type": "function", "function": {"name": "search_products"}}]
    asyncio.run(_client(handler).complete([{"role": "user", "content": "hi"}], tools=tools))

    assert seen[0]["tools"] == tools
    assert seen[0]["tool_choice"] == "auto"


def test_payload_has_no_tool_keys_when_tools_omitted():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    asyncio.run(_client(handler).complete([{"role": "user", "content": "hi"}]))

    # 加工具前后 payload 必须逐字一致：没要工具就不许凭空多出字段
    assert set(seen[0]) == {"model", "messages", "temperature"}


def test_empty_tools_list_behaves_like_none():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    asyncio.run(_client(handler).complete([{"role": "user", "content": "hi"}], tools=[]))

    assert "tools" not in seen[0] and "tool_choice" not in seen[0]


def test_null_content_with_tool_calls_parses_as_empty_string():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_body())

    result = asyncio.run(_client(handler).complete([{"role": "user", "content": "hi"}]))

    assert result.content == ""
    assert result.tool_calls == [
        ToolCall(id="call-1", name="search_products", arguments='{"query":"跑步鞋"}')
    ]


def test_null_content_without_tool_calls_still_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": None}}]})

    with pytest.raises(UpstreamError) as err:
        asyncio.run(_client(handler).complete([{"role": "user", "content": "hi"}]))
    assert "content 不是字符串" in err.value.message


def test_numeric_content_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": 123}}]})

    with pytest.raises(UpstreamError) as err:
        asyncio.run(_client(handler).complete([{"role": "user", "content": "hi"}]))
    assert "content 不是字符串" in err.value.message


def test_non_string_arguments_are_dumped_to_json_string():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_tool_body(arguments={"query": "跑步鞋"}))

    result = asyncio.run(_client(handler).complete([{"role": "user", "content": "hi"}]))

    assert isinstance(result.tool_calls[0].arguments, str)
    assert json.loads(result.tool_calls[0].arguments) == {"query": "跑步鞋"}


def test_no_tool_calls_yields_empty_list():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    result = asyncio.run(_client(handler).complete([{"role": "user", "content": "hi"}]))

    assert result.tool_calls == []
    assert result.content == "hello"
