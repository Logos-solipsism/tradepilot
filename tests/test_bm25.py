"""BM25：纯标准库，排序稳定，IDF 不翻负。"""

from __future__ import annotations

import math

from app.retrieval.bm25 import BM25Index
from app.retrieval.catalog import ProductDoc


def _doc(doc_id: int, title: str, description: str = "") -> ProductDoc:
    return ProductDoc(id=doc_id, title=title, description=description)


def test_hit_ordering_by_score():
    docs = [
        _doc(1, "篮球", "耐磨室外球场"),
        _doc(2, "跑步鞋", "轻量缓震"),
        _doc(3, "运动水壶", "500ml"),
    ]
    index = BM25Index(docs)
    hits = index.search("跑步鞋")
    assert [doc.id for doc, _ in hits] == [2]
    assert hits[0][1] > 0


def test_unrelated_query_returns_empty():
    index = BM25Index([_doc(1, "跑步鞋"), _doc(2, "篮球鞋")])
    assert index.search("冰箱") == []


def test_empty_query_returns_empty():
    index = BM25Index([_doc(1, "跑步鞋")])
    assert index.search("") == []
    assert index.search("，。！") == []


def test_ties_are_broken_by_id_ascending():
    # 两条完全同分的文档：必须先 id 小的
    docs = [_doc(7, "跑步鞋"), _doc(3, "跑步鞋"), _doc(5, "跑步鞋")]
    index = BM25Index(docs)
    hits = index.search("跑步鞋")
    assert [doc.id for doc, _ in hits] == [3, 5, 7]
    scores = [score for _, score in hits]
    assert scores[0] == scores[1] == scores[2]


def test_idf_stays_positive_when_every_doc_contains_term():
    docs = [_doc(1, "鞋"), _doc(2, "鞋"), _doc(3, "鞋")]
    index = BM25Index(docs)
    assert index.n_docs == 3
    assert index.term_df("鞋") == 3
    n, df = 3, 3
    expected = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
    assert index.idf("鞋") == expected
    assert index.idf("鞋") > 0
    assert len(index.search("鞋")) == 3


def test_title_weighting_beats_description_only_hit():
    # 两条文档都含 "跑步鞋"，只有一处是标题命中：
    # 标题加权（title token 重复一遍）必须让标题命中排前面
    docs = [
        _doc(1, "跑步鞋", ""),
        _doc(2, "篮球", "跑步鞋"),
    ]
    index = BM25Index(docs)
    assert index.n_docs == 2
    # 单字+二元组：doc1 = 5*2 + 0 = 10，doc2 = 3*2 + 5 = 11 → avgdl 10.5
    assert index.avgdl == 10.5
    hits = index.search("跑步鞋")
    assert [doc.id for doc, _ in hits] == [1, 2]
    assert hits[0][1] > hits[1][1]


def test_two_char_query_reaches_doc_containing_it_as_substring():
    # P5-1 返工靶子：文档切出的二元组里没有 "跑鞋"（只有 跑步/步鞋），
    # 只靠二元组会零召回；靠单字 + 二元组必须首位命中
    docs = [
        _doc(1, "轻量跑步鞋 男款"),
        _doc(2, "篮球鞋 高帮"),
        _doc(3, "运动水壶 500ml"),
    ]
    index = BM25Index(docs)
    hits = index.search("跑鞋")
    assert hits, "2 字 query 不许零召回"
    assert hits[0][0].id == 1
    assert hits[0][0].title == "轻量跑步鞋 男款"


def test_single_char_query_recalls_all_matching_docs():
    # 单字 query 在纯二元组索引里没有任何对应 term，必须靠单字 token 命中
    docs = [
        _doc(1, "轻量跑步鞋 男款"),
        _doc(2, "篮球鞋 高帮"),
        _doc(3, "运动水壶 500ml"),
    ]
    index = BM25Index(docs)
    ids = [doc.id for doc, _ in index.search("鞋")]
    # 只断言召回，不断言顺序：长标题被 BM25 长度归一压分是预期行为
    assert sorted(ids) == [1, 2]


def test_exact_term_query_still_ranks_first():
    # 修召回不能修坏排序：完整词查询首位仍是精确命中的那条
    docs = [
        _doc(1, "轻量跑步鞋 男款"),
        _doc(2, "篮球鞋 高帮"),
        _doc(3, "运动水壶 500ml"),
    ]
    index = BM25Index(docs)
    hits = index.search("跑步鞋")
    assert hits[0][0].id == 1
    assert hits[0][0].title == "轻量跑步鞋 男款"


def test_top_k_limits_result_size():
    docs = [_doc(i, f"跑步鞋 型号{i}") for i in range(1, 6)]
    index = BM25Index(docs)
    assert len(index.search("跑步鞋", top_k=2)) == 2
    assert len(index.search("跑步鞋", top_k=0)) == 0


def test_docs_without_tokens_are_skipped():
    index = BM25Index([_doc(1, "", "")])
    assert index.n_docs == 1
    assert index.avgdl == 0.0
    assert index.search("跑步鞋") == []
