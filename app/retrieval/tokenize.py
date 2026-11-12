"""中文/英文分词：字符 n-gram，不引 jieba（零依赖、可预测）。

规则：
1. unicodedata.normalize("NFKC") 归一（全角→半角）后整体 lower()；
2. ASCII 字母数字连续 run 算一个 token（整体小写）；
3. CJK 连续 run 单独切出：长度 1 输出该单字；长度 >= 2 先输出 n 个单字，
   再输出 n-1 个相邻二元组（Lucene CJKFilter 的 outputUnigrams=true 语义）；
4. 其余字符（标点、空白、emoji）丢弃。

为什么单字和二元组都要（实测结论，不是推测）：只输出二元组时 2 字 query 会零召回——
文档"轻量跑步鞋 男款"切出 [轻量, 量跑, 跑步, 步鞋, 鞋男, 男款]，里面**没有** "跑鞋"
这个二元组，query "跑鞋" 的 df=0 直接出局；单字 query "鞋" 同理，在纯二元组索引里
没有任何对应 term。补上单字后这两类 query 都能命中，代价是多出 1.0 上下的噪声分，
与真实命中（3.0+）差一个身位，排序不受影响，所以这里不引入单字降权系数
（那是没有评估集之前的过早优化，留给后续用评估集说话）。
"""

from __future__ import annotations

import unicodedata

# CJK 统一表意文字（含扩展 A）与兼容表意文字；不含假名/韩文音节
_CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    for low, high in _CJK_RANGES:
        if low <= code <= high:
            return True
    return False


def _is_ascii_alnum(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


def tokenize(text: str) -> list[str]:
    """把一段文本切成 token 列表；空串/None 一律返回空列表。"""
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).lower()

    tokens: list[str] = []
    ascii_run: list[str] = []
    cjk_run: list[str] = []

    def flush_ascii() -> None:
        if ascii_run:
            tokens.append("".join(ascii_run))
            ascii_run.clear()

    def flush_cjk() -> None:
        if not cjk_run:
            return
        if len(cjk_run) == 1:
            tokens.append(cjk_run[0])
        else:
            # 先出全部单字，再出相邻二元组：顺序固定，测试与调试都可预期
            tokens.extend(cjk_run)
            for i in range(len(cjk_run) - 1):
                tokens.append(cjk_run[i] + cjk_run[i + 1])
        cjk_run.clear()

    for ch in normalized:
        if _is_ascii_alnum(ch):
            flush_cjk()
            ascii_run.append(ch)
        elif _is_cjk(ch):
            flush_ascii()
            cjk_run.append(ch)
        else:
            flush_ascii()
            flush_cjk()

    flush_ascii()
    flush_cjk()
    return tokens
