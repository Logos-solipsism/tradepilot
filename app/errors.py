"""业务异常与上游异常：由 main.py 的异常处理器统一转成信封响应。"""

from __future__ import annotations


class AppError(Exception):
    """业务异常基类：携带真实 HTTP 状态码，绝不包成 200 + code。"""

    status_code = 500
    code = 500
    message = "服务内部错误"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message


class TicketInvalidError(AppError):
    """票据缺失/格式错/过期/签名不符/nonce 重放——一律 401。"""

    status_code = 401
    code = 401
    message = "internal ticket 无效或已过期"


class UpstreamError(AppError):
    """上游大模型不可用（超时 / 非 2xx / 结构不符）——一律 502。

    消息里只写可归类的摘要（超时？HTTP 500？结构不符？），
    绝不回传上游原始报文：里面可能带 key 或用户数据。
    """

    status_code = 502
    code = 502
    message = "上游模型暂时不可用"
