"""查询泛化：把用户口语 query 扩展到库里的商品用词。纯函数，无 IO。

一次 expand 做三件事，顺序固定（换顺序会改变结果，测试按这个钉）：

1. **错别字纠正**（TYPOS）：对整条 query 做子串替换。**长键优先**——
   先 "运动些" -> "运动鞋"，再考虑更短的键；否则短键先把长键打碎，
   长键就永远命中不了。
2. **词表扫描**（SYNONYMS + HYPERNYMS）：中文没有空格，键是**子串**匹配；
   同样按键长度降序扫。一个键命中后，它覆盖的字符区间被标记为已消费，
   **完全落在已消费区间内**的短键不再触发——这就是"长键优先"的落地方式，
   否则 "耳机塞" 会同时触发 "耳机塞" 和 "耳机" 两条规则，扩展词直接翻倍。
3. **扩展词再分词**：命中的扩展词各自过 tokenize（单字 + 相邻二元组），
   去重后并入 original_tokens。

为什么扩展词也要 tokenize：库里索引的 token 是单字 + 二元组，从没有三元组，
所以把 "跑步鞋" 整串塞进查询是匹配不上任何东西的，必须拆成
["跑","步","鞋","跑步","步鞋"] 才有用。

退化行为（**契约的一部分**）：空串、纯标点、一个字都没命中时，
expanded_tokens **必须等于** original_tokens，绝不返回空——返回空会让
调用方以为"这条 query 什么都不搜"，把原本能过的 exact 查询打成零召回。
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval import synonyms
from app.retrieval.tokenize import tokenize


@dataclass(frozen=True)
class QueryExpansion:
    """一次扩展的结果；frozen 便于当缓存键，也防止调用方就地改坏。"""

    original_tokens: tuple[str, ...]
    expanded_tokens: tuple[str, ...]
    # 命中了哪些规则，如 ('hypernym:运动鞋', 'typo:蓝芽->蓝牙')；只用于日志与调试
    applied: tuple[str, ...]


def _dedup(tokens: list[str]) -> list[str]:
    """按首次出现顺序去重：保持 original 在前，扩展词在后，顺序可预期。"""
    return list(dict.fromkeys(tokens))


def _apply_typos(query: str) -> tuple[str, list[str]]:
    """子串替换纠正错别字，长键优先；返回 (纠正后文本, 命中的规则名)。"""
    corrected = query
    applied: list[str] = []
    # sorted 是稳定排序：等长键按字典序，输出不依赖 dict 插入顺序
    for wrong in sorted(synonyms.TYPOS, key=lambda k: (-len(k), k)):
        if wrong and wrong in corrected:
            corrected = corrected.replace(wrong, synonyms.TYPOS[wrong])
            applied.append(f"typo:{wrong}->{synonyms.TYPOS[wrong]}")
    return corrected, applied


def _match_keys(text: str, table: dict[str, tuple[str, ...]], kind: str):
    """按键长度降序做子串扫描，返回 (扩展词列表, 规则名列表)。

    命中一个键就把它的区间标记为已消费；完全被包含在已消费区间里的短键
    直接跳过（长键优先，见模块 docstring 第 2 条）。
    """
    if not text:
        return [], []
    consumed: list[tuple[int, int]] = []
    expansions: list[str] = []
    applied: list[str] = []

    def is_consumed(start: int, end: int) -> bool:
        return any(cs <= start and end <= ce for cs, ce in consumed)

    for key in sorted(table, key=lambda k: (-len(k), k)):
        if not key:
            continue
        start = text.find(key)
        while start != -1:
            end = start + len(key)
            if not is_consumed(start, end):
                consumed.append((start, end))
                expansions.extend(table[key])
                applied.append(f"{kind}:{key}")
            start = text.find(key, start + 1)
    return expansions, applied


def expand_query(query: str) -> QueryExpansion:
    """把 query 扩展成 (原文 token ∪ 泛化词 token)；不抛异常、不返回空 token。"""
    original = tuple(tokenize(query))
    if not query:
        return QueryExpansion(original_tokens=original, expanded_tokens=original, applied=())

    corrected, typo_applied = _apply_typos(query)
    # 纠正后的文本本身也算新词：错字不改，纠了也白纠（"冲电" -> "充电" 必须真的进 token）
    extra: list[str] = []
    if corrected != query:
        extra.extend(tokenize(corrected))

    synonym_words, synonym_applied = _match_keys(corrected, synonyms.SYNONYMS, "synonym")
    hypernym_words, hypernym_applied = _match_keys(corrected, synonyms.HYPERNYMS, "hypernym")

    for word in synonym_words + hypernym_words:
        extra.extend(tokenize(word))

    expanded = tuple(_dedup(list(original) + extra))
    applied = tuple(typo_applied + synonym_applied + hypernym_applied)
    return QueryExpansion(original_tokens=original, expanded_tokens=expanded, applied=applied)
