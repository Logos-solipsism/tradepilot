"""LLM 客户端：全离线，httpx.MockTransport 注入，禁止真打网络。"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.errors import UpstreamError
from app.llm.client import FAKE_ANSWER, LlmClient


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


def _ok_body(content="hello", prompt=12, completion=34, total=46) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
    }


def test_fake_mode_when_api_key_empty_makes_no_request():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_ok_body())

    client = LlmClient(_settings(llm_api_key=""), transport=httpx.MockTransport(handler))
    result = asyncio.run(client.complete([{"role": "user", "content": "hi"}]))

    assert result.content == FAKE_ANSWER
    assert (result.usage.input, result.usage.output, result.usage.total) == (0, 0, 0)
    assert calls == []  # 一次网络请求都没发
    assert client.is_fake is True


def test_real_mode_parses_content_and_usage():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_ok_body("-".join("x" * 5)))

    client = LlmClient(_settings(), transport=httpx.MockTransport(handler))
    result = asyncio.run(client.complete([{"role": "user", "content": "hi"}]))

    assert result.content == "x-x-x-x-x"
    assert (result.usage.input, result.usage.output, result.usage.total) == (12, 34, 46)
    req = seen[0]
    assert str(req.url) == "https://llm.test/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.3
    assert body["messages"][0]["content"] == "hi"


def test_http_500_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = LlmClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamError) as exc:
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert "500" in str(exc.value)


def test_timeout_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = LlmClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


def test_bad_structure_raises_upstream_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = LlmClient(_settings(), transport=httpx.MockTransport(handler))
    with pytest.raises(UpstreamError):
        asyncio.run(client.complete([{"role": "user", "content": "hi"}]))


def test_missing_usage_falls_back_to_zero():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = LlmClient(_settings(), transport=httpx.MockTransport(handler))
    result = asyncio.run(client.complete([{"role": "user", "content": "hi"}]))
    assert result.content == "ok"
    assert (result.usage.input, result.usage.output, result.usage.total) == (0, 0, 0)
