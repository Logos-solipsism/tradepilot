"""IR 指标：Recall@k / NDCG@k / MRR@k，纯标准库。

三条约定（全模块统一，测试按这个钉）：
1. relevance 只记录 rel >= 1 的商品；rel = 0 表示"不写"，与"写了 0"等价。
2. 三个函数对退化输入一律返回 0.0，**绝不抛异常**：
   ranked 为空、k <= 0、k 大于列表长度、relevance 为空字典。
3. 所有 k 都是"看前 k 个"的截断：ranked[k:] 里的命中不计分。

另外提供 `char_coverage` / `max_char_coverage` 与两个阈值常量：它们**不参与指标计算**，
只用来给 query 贴"字面难度"标签（run_eval 的分档诊断与 labels 的护栏共用同一套口径）。

关于 NDCG 的理想 DCG：这里用**全部相关商品**（不截断到 k）按 rel 降序计算。
因此当一个 query 的相关商品多于 k 条时，即使 top-k 完全命中且顺序完美，
NDCG@k 也会小于 1.0——这是有意的：它同时惩罚"漏掉排在后面的相关商品"。
k 之下的标准 IDCG 会把这个信息吞掉，而本评估集里恰恰有多条 query 的相关商品
超过 10 条（如 "跑步鞋"），我们想看见那部分损失。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


def _relevant(relevance: Mapping[int, int]) -> dict[int, int]:
    """只保留 rel >= 1 的条目；rel 为负数或 0 视作不相关。"""
    return {doc_id: rel for doc_id, rel in relevance.items() if rel > 0}


def _top_k(ranked_ids: Sequence[int], k: int) -> list[int]:
    if k <= 0:
        return []
    return list(ranked_ids[:k])


# 字面覆盖率的两个分界：>= HIGH 认为"靠词面匹配就能过"，< MID 才是真语义改写。
# 两个数字与 labels 的护栏共用；改这里等于改护栏，测试会一起告诉你。
LEXICAL_HIGH = 0.7
LEXICAL_MID = 0.4


def char_coverage(query: str, title: str) -> float:
    """query 的字符集被 title 字符集覆盖的比例（0.0~1.0）。

    归一：去空白 + lower + **去重字符集**。与字序无关，同一个字重复出现不加分——
    它度量的是"这些字在标题里出现过没有"，是"靠词面匹配能不能过"的廉价代理。

    query 归一后为空（空串或纯空白）时返回 0.0：没有字符就没有覆盖率，
    给 1.0 会让空 query 冒充"词面完全重合"。
    """
    query_chars = set("".join(query.split()).lower())
    if not query_chars:
        return 0.0
    title_chars = set("".join(title.split()).lower())
    return len(query_chars & title_chars) / len(query_chars)


def max_char_coverage(query: str, titles: Iterable[str]) -> float:
    """该 query 在若干候选标题里的**最高**覆盖率（无候选时 0.0）。

    取 max 而不是均值，是因为要回答的问题是"存不存在一条期望商品，
    光靠字面就能对上"——只要有一条能，这条 query 就不是纯净的语义改写。
    """
    return max((char_coverage(query, title) for title in titles), default=0.0)


def recall_at_k(
    ranked_ids: Iterable[int],
    relevance: Mapping[int, int],
    k: int,
) -> float:
    """相关商品被召回的比例：分母是全部 rel >= 1 的商品数。

    分母为 0（relevance 为空或全 0）时返回 0.0——没有可召回的东西，
    给 1.0 会把"没有标签"伪装成"召回归满"，污染分组均值。
    """
    relevant = _relevant(relevance)
    if not relevant:
        return 0.0
    top = _top_k(list(ranked_ids), k)
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    ranked_ids: Iterable[int],
    relevance: Mapping[int, int],
    k: int,
) -> float:
    """gain = 2^rel - 1，discount = 1/log2(i + 2)，i 从 0 起（第 1 位不打折）。"""
    relevant = _relevant(relevance)
    if not relevant:
        return 0.0
    top = _top_k(list(ranked_ids), k)
    if not top:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(top):
        rel = relevant.get(doc_id, 0)
        if rel <= 0:
            continue
        dcg += (2**rel - 1) / math.log2(i + 2)

    # 理想 DCG：全部相关商品按 rel 降序（不截断到 k，见模块 docstring）
    ideal_gains = sorted(((2**rel - 1) for rel in relevant.values()), reverse=True)
    idcg = 0.0
    for i, gain in enumerate(ideal_gains):
        idcg += gain / math.log2(i + 2)

    if idcg <= 0:
        return 0.0
    return dcg / idcg


def mrr_at_k(
    ranked_ids: Iterable[int],
    relevance: Mapping[int, int],
    k: int,
) -> float:
    """第一个 rel >= 1 的位次取倒数；前 k 个里一个都没有则 0.0。"""
    relevant = _relevant(relevance)
    if not relevant:
        return 0.0
    top = _top_k(list(ranked_ids), k)
    for i, doc_id in enumerate(top):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0
