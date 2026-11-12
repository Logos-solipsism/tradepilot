"""大模型客户端：直接打 OpenAI 兼容的 /chat/completions（不引 openai SDK）。

LLM_API_KEY 为空时进入 **fake 模式**：不发任何网络请求，返回固定占位文案，
usage 全 0，/health 报 llm="fake"。这与后端既有决策"宁可降级也不让应用起不来"一致。

注意：fake 文案只用于本地开发占位，**不允许**在生产把它悄悄换成正常回复。

工具调用：tool_calls 里的 arguments 原样透传（模型给的原始 JSON 字符串），
本模块不解析、不校验，那是工具层（app/tools/）的事。
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field

import httpx

from app.config import Settings
from app.errors import UpstreamError

log = logging.getLogger(__name__)

FAKE_ANSWER = "[dev] 模型未配置，这是本地占位回复。"


@dataclass(frozen=True)
class LlmUsage:
    input: int = 0
    output: int = 0
    total: int = 0


@dataclass(frozen=True)
class ToolCall:
    """模型发起的一次工具调用。

    arguments 是模型给的**原始 JSON 字符串**，这里原样透传：解析失败也要如实记下来，
    因为它既是工具层的输入，也是回给调用方做审计的证据。
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class LlmResult:
    content: str
    usage: LlmUsage = LlmUsage()
    tool_calls: list[ToolCall] = field(default_factory=list)


def _as_int(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _parse_tool_calls(raw) -> list[ToolCall]:
    """把上游的 tool_calls 归一成 ToolCall 列表。

    上游字段可能是 None、可能是别的类型、也可能混进残项：这里的策略是"取不出来的就不算
    一次工具调用"，宁可让模型少调一次工具，也不让半条记录漏到工具层去炸掉整个请求。
    """
    if not isinstance(raw, list):
        return []
    calls: list[ToolCall] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            # 少数上游直接回对象；统一成字符串，工具层只认一种输入形状
            arguments = json.dumps(arguments, ensure_ascii=False)
        call_id = item.get("id")
        calls.append(
            ToolCall(
                id=call_id if isinstance(call_id, str) else "",
                name=name,
                arguments=arguments,
            )
        )
    return calls


class LlmClient:
    """一个实例 = 一份配置。每次调用自建 AsyncClient，调用结束即关闭。"""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport

    @property
    def is_fake(self) -> bool:
        return self._settings.llm_is_fake

    async def complete(
        self, messages: list[dict], *, tools: list[dict] | None = None
    ) -> LlmResult:
        if self.is_fake:
            return LlmResult(content=FAKE_ANSWER, usage=LlmUsage())

        payload = {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": 0.3,
        }
        # tools 为空时 payload 与加工具前逐字相同：不给上游凭空多出两个字段
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._settings.llm_timeout_seconds, transport=self._transport
            ) as client:
                response = await client.post(
                    self._settings.chat_completions_url, json=payload, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                f"上游模型超时({self._settings.llm_timeout_seconds}s): {type(exc).__name__}"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"上游模型连接失败: {type(exc).__name__}") from exc

        if response.status_code < 200 or response.status_code >= 300:
            # 只报状态码，不回传上游报文（可能含 key 或用户数据）
            raise UpstreamError(f"上游模型返回 HTTP {response.status_code}")

        try:
            body = response.json()
        except ValueError as exc:
            raise UpstreamError("上游模型返回的不是合法 JSON") from exc
        return self._parse(body)

    @staticmethod
    def _parse(body) -> LlmResult:
        if not isinstance(body, dict):
            raise UpstreamError("上游模型返回结构不符: 顶层不是对象")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise UpstreamError("上游模型返回结构不符: choices 缺失或为空")
        first = choices[0]
        if not isinstance(first, dict):
            raise UpstreamError("上游模型返回结构不符: choices[0] 不是对象")
        message = first.get("message")
        if not isinstance(message, dict):
            raise UpstreamError("上游模型返回结构不符: message 缺失")
        tool_calls = _parse_tool_calls(message.get("tool_calls"))
        content = message.get("content")
        if content is None:
            # OpenAI 兼容协议里"只调工具不答话"会返回 content=null：
            # 只有确实带了工具调用才允许，否则仍按结构不符处理
            if not tool_calls:
                raise UpstreamError("上游模型返回结构不符: content 不是字符串")
            content = ""
        elif not isinstance(content, str):
            raise UpstreamError("上游模型返回结构不符: content 不是字符串")
        usage_raw = body.get("usage")
        usage = usage_raw if isinstance(usage_raw, dict) else {}
        return LlmResult(
            content=content,
            usage=LlmUsage(
                input=_as_int(usage.get("prompt_tokens")),
                output=_as_int(usage.get("completion_tokens")),
                total=_as_int(usage.get("total_tokens")),
            ),
            tool_calls=tool_calls,
        )

    async def aclose(self) -> None:
        """当前实现每次调用自建连接，没有常驻资源需要释放。"""
        return None
