"""评估指标：用手算小例子钉死 recall / ndcg / mrr，并覆盖退化输入。"""

from __future__ import annotations

import math

from app.eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k

_REL = {1: 2, 2: 1}


def test_recall_counts_hits_over_all_relevant():
    # 相关商品共 2 条，top-3 里命中 2 条 -> 1.0
    assert recall_at_k([1, 2, 3], _REL, 3) == 1.0
    # 只命中第一条 -> 1/2
    assert recall_at_k([1, 9, 8], _REL, 3) == 0.5


def test_recall_denominator_is_all_relevant_not_k():
    # 相关商品 2 条，k=1 最多命中 1 条 -> 0.5，而不是按 k 归一
    assert recall_at_k([1, 2], _REL, 1) == 0.5


def test_ndcg_hand_computed():
    # gain(rel=2)=3, gain(rel=1)=1；理想顺序 [1, 2]
    # DCG = 3/log2(2) + 1/log2(3) = 3 + 1/log2(3)
    # IDCG = 3 + 1/log2(3) -> 完美排序就是 1.0
    assert ndcg_at_k([1, 2, 3], _REL, 3) == 1.0

    # 顺序反了：DCG = 1/log2(2) + 3/log2(3)
    expected = (1.0 + 3.0 / math.log2(3)) / (3.0 + 1.0 / math.log2(3))
    assert math.isclose(ndcg_at_k([2, 1], _REL, 3), expected, rel_tol=1e-12)


def test_ndcg_penalises_missing_relevant_beyond_k():
    # 相关商品 3 条，IDCG 用全部 3 条（不截断到 k），
    # 所以即使 top-2 完美也只能拿到 < 1.0
    rel = {1: 2, 2: 1, 3: 1}
    idcg = 3.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    expected = (3.0 + 1.0 / math.log2(3)) / idcg
    assert math.isclose(ndcg_at_k([1, 2], rel, 2), expected, rel_tol=1e-12)
    assert ndcg_at_k([1, 2], rel, 2) < 1.0


def test_mrr_first_relevant_position():
    assert mrr_at_k([1, 2, 3], _REL, 3) == 1.0
    assert mrr_at_k([9, 1, 2], _REL, 3) == 0.5
    assert mrr_at_k([9, 8, 1], _REL, 3) == 1.0 / 3


def test_mrr_ignores_hits_beyond_k():
    assert mrr_at_k([9, 8, 1], _REL, 2) == 0.0


def test_empty_ranked_returns_zero():
    assert recall_at_k([], _REL, 10) == 0.0
    assert ndcg_at_k([], _REL, 10) == 0.0
    assert mrr_at_k([], _REL, 10) == 0.0


def test_empty_relevance_returns_zero():
    assert recall_at_k([1, 2, 3], {}, 3) == 0.0
    assert ndcg_at_k([1, 2, 3], {}, 3) == 0.0
    assert mrr_at_k([1, 2, 3], {}, 3) == 0.0


def test_k_beyond_length_does_not_raise():
    assert recall_at_k([1, 2], _REL, 100) == 1.0
    assert ndcg_at_k([1, 2], _REL, 100) == 1.0
    assert mrr_at_k([1, 2], _REL, 100) == 1.0


def test_k_non_positive_returns_zero():
    assert recall_at_k([1, 2], _REL, 0) == 0.0
    assert ndcg_at_k([1, 2], _REL, 0) == 0.0
    assert mrr_at_k([1, 2], _REL, 0) == 0.0


def test_zero_relevance_entries_are_ignored():
    # rel=0 与"不写"等价：既不算相关，也不进分母
    assert recall_at_k([1, 2, 3], {1: 2, 2: 1, 3: 0}, 3) == 1.0
    assert ndcg_at_k([1, 2, 3], {1: 2, 2: 1, 3: 0}, 3) == 1.0
