"""最小 .env 加载器：纯标准库，不引 python-dotenv。

为什么自己写：.env 只有 `KEY=VALUE` 一种语法，而本项目对依赖的态度是
"每个包都要能被解释"。python-dotenv 能做变量展开、多文件合并、跨行续写等
这里永远用不到的事，代价是多一个运行时依赖。

**优先级（契约的一部分）：真实环境变量 > .env 文件 > Settings 默认值。**
只有 os.environ 里**还没有**这个键时才写进去。理由：CI / 容器 / IDE 里设置的
变量是"这台机器当下的运行时事实"，仓库里的 .env 只是开发者的本机便利默认值；
若 .env 能盖掉真实变量，线上一次 SESSION_BACKEND=redis 会被一份遗留的 .env
悄悄顶回 memory，而且一句报错都不会有。

语法只认这些（.env 是人手写的，遇到不认识的行要**跳过而不是抛错**——
一个手滑的注释不该让服务起不来）：
- 空行、`#` 开头的行：忽略；
- 允许 `export ` 前缀，方便直接把 .env source 进 shell；
- 只按**第一个** `=` 切；键为空或键里带空白的行忽略；没有 `=` 的行忽略；
- 值两端成对的引号去掉；双引号内识别 \\n \\r \\t \\" \\\\ 五个转义，单引号内不转义；
- 引号**外**且前面是空白的 `#` 算行内注释；紧贴着的 `#` 不算，
  否则 `REDIS_URL=redis://host/db#frag` 会被截断成半个 URL；
- 变量**不做展开**：`A=${B}` 就是字面量。展开会让解析结果依赖键的出现顺序，
  排查"为什么这个值是空的"要多跳一层。
"""

from __future__ import annotations

import os
from pathlib import Path

# 默认读仓库根目录的 .env（本文件在 app/ 下，往上一级就是仓库根），不看进程 cwd：
# uvicorn / pytest 从哪个目录启动都应该读到同一份文件。
DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

_DOUBLE_QUOTE_ESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}


def _strip_inline_comment(raw: str) -> str:
    """去掉引号外的行内注释：`abc # 说明` → `abc`。

    只把**前面是空白**的 `#` 当注释起点：`redis://host/db#frag` 里的 `#`
    属于值本身，截断它等于悄悄改掉用户写的配置。
    """
    quote: str | None = None
    for index, ch in enumerate(raw):
        if quote is None:
            if ch in ("'", '"'):
                quote = ch
            elif ch == "#" and index > 0 and raw[index - 1].isspace():
                return raw[:index]
        elif ch == quote:
            quote = None
    return raw


def _unescape_double(text: str) -> str:
    """双引号内的转义；不认识的转义原样保留（含反斜杠），不擅自吞字符。"""
    out: list[str] = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch == "\\" and index + 1 < len(text):
            replacement = _DOUBLE_QUOTE_ESCAPES.get(text[index + 1])
            if replacement is not None:
                out.append(replacement)
                index += 2
                continue
        out.append(ch)
        index += 1
    return "".join(out)


def _unquote(raw: str) -> str:
    """成对引号才去掉；不配对的引号原样保留（写的人显然不是想引用）。"""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        inner = raw[1:-1]
        return _unescape_double(inner) if raw[0] == '"' else inner
    return raw


def parse_env_text(text: str) -> dict[str, str]:
    """解析 .env 文本；纯函数，**不碰 os.environ**（优先级判断留给 apply_env_file）。

    重复的键后者覆盖前者，与 `source .env` 的行为一致。
    """
    values: dict[str, str] = {}
    for line in text.lstrip("\ufeff").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped[:7].lower() == "export ":
            stripped = stripped[7:].lstrip()
        if "=" not in stripped:
            continue
        key, _, rest = stripped.partition("=")
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            continue
        values[key] = _unquote(_strip_inline_comment(rest.strip()).strip())
    return values


def parse_env_file(path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """读文件并解析；文件不存在 / 读不了 / 不是 UTF-8 → 空 dict。

    .env 由人手工创建且被 .gitignore 忽略，**不存在是常态**：
    "没有配置文件"与"配置文件是空的"在这里是同一种结果，绝不因此让服务起不来。
    编码不对（比如在 GBK 控制台里存成了 GBK）同样按"读不到"处理：
    与其抛一个 traceback 拦住整个启动，不如明确地退回默认配置。
    """
    target = Path(path) if path is not None else DEFAULT_ENV_PATH
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    return parse_env_text(text)


def apply_env_file(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> tuple[str, ...]:
    """把 .env 里**尚未被真实环境变量设置**的键写进 environ，返回真正写入的键名。

    优先级：真实环境变量 > .env > Settings 默认值。已经存在的键**一个都不覆盖**，
    哪怕它的值是空串——"我把 LLM_API_KEY 显式设成空"是一个明确的决定（强制 fake 模式），
    不该被一份 .env 悄悄改成有 key。

    environ 默认就是 os.environ；参数只为测试能换成普通 dict，不必污染进程环境。
    """
    target = os.environ if environ is None else environ
    applied: list[str] = []
    for key, value in parse_env_file(path).items():
        if key in target:
            continue
        target[key] = value
        applied.append(key)
    return tuple(applied)
