"""进程配置：全部来自环境变量，进程启动后不再变化。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

VERSION = "0.1.0"

_DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_DEFAULT_AGENT_BASE_URL = "http://127.0.0.1:18090"


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else raw.strip()


def _env_number(name: str, default, cast):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 不是合法数值: {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """运行期配置快照。frozen：任何组件都别偷偷改配置。"""

    internal_secret: str = ""
    llm_base_url: str = _DEFAULT_BASE_URL
    llm_api_key: str = ""
    llm_model: str = _DEFAULT_MODEL
    llm_timeout_seconds: float = 30.0
    session_backend: str = "memory"
    session_window: int = 20
    session_ttl_days: int = 7
    redis_url: str = "redis://127.0.0.1:6379/0"
    # 商品目录上游（Java 侧 trademall 的 agent 服务）；token 为空时本地服务仍能起，
    # 但 CatalogClient 会 fail-closed 直接报上游错误，绝不假装"没有商品"
    agent_service_base_url: str = _DEFAULT_AGENT_BASE_URL
    agent_service_token: str = ""
    catalog_cache_ttl_seconds: int = 300
    version: str = VERSION

    def __post_init__(self) -> None:
        if self.session_backend not in ("memory", "redis"):
            raise ValueError(
                f"SESSION_BACKEND 只支持 memory|redis，收到 {self.session_backend!r}"
            )
        if self.session_window < 1:
            raise ValueError("SESSION_WINDOW 必须 >= 1")
        if self.session_ttl_days < 1:
            raise ValueError("SESSION_TTL_DAYS 必须 >= 1")

    @property
    def llm_is_fake(self) -> bool:
        """没有 API key 就只能是占位模式——绝不假装调过模型。"""
        return not self.llm_api_key.strip()

    @property
    def chat_completions_url(self) -> str:
        return self.llm_base_url.rstrip("/") + "/chat/completions"

    @property
    def catalog_products_url(self) -> str:
        return self.agent_service_base_url.rstrip("/") + "/api/agent/products"


def load_settings() -> Settings:
    return Settings(
        internal_secret=_env_str("INTERNAL_SECRET", ""),
        llm_base_url=_env_str("LLM_BASE_URL", _DEFAULT_BASE_URL),
        llm_api_key=_env_str("LLM_API_KEY", ""),
        llm_model=_env_str("LLM_MODEL", _DEFAULT_MODEL),
        llm_timeout_seconds=_env_number("LLM_TIMEOUT_SECONDS", 30.0, float),
        session_backend=_env_str("SESSION_BACKEND", "memory").lower(),
        session_window=_env_number("SESSION_WINDOW", 20, int),
        session_ttl_days=_env_number("SESSION_TTL_DAYS", 7, int),
        redis_url=_env_str("REDIS_URL", "redis://127.0.0.1:6379/0"),
        agent_service_base_url=_env_str("AGENT_SERVICE_BASE_URL", _DEFAULT_AGENT_BASE_URL),
        # 空 token 不在 __post_init__ 里抛错：本地没配 token 服务也要能起，
        # fail-closed 的判断放在 CatalogClient 里（真正要发请求的那一刻）
        agent_service_token=_env_str("AGENT_SERVICE_TOKEN", ""),
        catalog_cache_ttl_seconds=_env_number("CATALOG_CACHE_TTL_SECONDS", 300, int),
        version=VERSION,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """带缓存：环境变量只在首次读取时生效，避免同一个进程里配置漂移。"""
    return load_settings()
