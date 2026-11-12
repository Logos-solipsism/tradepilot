"""HTTP 契约：请求体严格禁止多余字段，响应统一信封。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequestBody(BaseModel):
    """POST /chat 的请求体。

    extra="forbid" 是防越权的结构性保证：**请求体里没有、也不允许有 user_id**。
    用户身份只能来自 X-Internal-Ticket，带 user_id 的请求直接被 422 拒绝。
    """

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    trace_id: str | None = None


class Usage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class ChatData(BaseModel):
    answer: str
    session_id: str
    tool_calls: list[Any] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class ApiEnvelope(BaseModel):
    code: int = 0
    message: str = "ok"
    data: ChatData | None = None


class SearchRequestBody(BaseModel):
    """POST /search 的请求体。

    extra="forbid" 与 /chat 一致：请求体里没有、也不允许有 user_id。
    query 不在这里 trim：回显必须是调用方原样发来的字符串。
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)
    top_k: int = Field(default=10, ge=1, le=50)
    trace_id: str | None = None


class SearchHit(BaseModel):
    """命中项：只有这五个字段，描述不进检索响应（Java 侧按此逐字解析）。"""

    id: int
    title: str
    min_price: int | None = None
    sku_count: int = 0
    # 融合分（RRF，量级 ~0.03），只用于排序展示：**不是相关度百分比、不可跨版本比较**。
    # 0.0 表示该条只是为凑满 top_k 补位的尾部结果，不是异常。
    score: float


class SearchData(BaseModel):
    query: str
    hits: list[SearchHit] = Field(default_factory=list)
    indexed: int = 0


class SearchEnvelope(BaseModel):
    code: int = 0
    message: str = "ok"
    data: SearchData | None = None


class HealthData(BaseModel):
    ok: Literal[True] = True
    llm: Literal["real", "fake"]
    version: str
    sessions: Literal["memory", "redis"]
    # ready = 索引里已有商品；empty = 还没建起来或上游真的没有商品
    search: Literal["ready", "empty"] = "empty"
