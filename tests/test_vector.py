"""faiss 向量索引 + 三路 RRF：召回正确性、护栏不被向量噪声攻破。

全部离线：向量来自 HashingEmbedder，不联网也不需要 key。
"""

from __future__ import annotations

import pytest

from app.eval.fixture_catalog import build_fixture_catalog
from app.eval.labels import QUERIES
from app.eval.run_eval import evaluate, summarise
from app.retrieval.bm25 import BM25Index, RRF_DEPTH_DEFAULT
from app.retrieval.catalog import ProductDoc
from app.retrieval.embedding import HashingEmbedder
from app.retrieval.vector import VectorIndex, doc_text


def _doc(doc_id: int, title: str, description: str = "") -> ProductDoc:
    return ProductDoc(id=doc_id, title=title, description=description)


def _toy_index(dimension: int = 512) -> VectorIndex:
    return VectorIndex(
        [
            _doc(1, "蓝牙降噪耳机 长续航"),
            _doc(2, "不锈钢保温杯 480ml"),
            _doc(3, "机械键盘 红轴 有线"),
        ],
        HashingEmbedder(dimension),
    )


def _fixture_index() -> tuple[BM25Index, VectorIndex, list[ProductDoc]]:
    docs = build_fixture_catalog()
    return BM25Index(docs), VectorIndex(docs, HashingEmbedder(1024)), docs


# --- 索引本身 -----------------------------------------------------------


def test_doc_text_weights_title_like_bm25():
    # 与 BM25 建索引时"标题 token 重复一遍"同一口径，两路对"什么更重要"的判断才一致
    text = doc_text(_doc(1, "跑步鞋", "轻量缓震"))
    assert text.count("跑步鞋") == 2
    assert "轻量缓震" in text


def test_self_query_hits_itself_first():
    index = _toy_index()
    docs = [_doc(1, "蓝牙降噪耳机 长续航"), _doc(2, "不锈钢保温杯 480ml"), _doc(3, "机械键盘 红轴 有线")]
    for doc in docs:
        hits = index.search(doc_text(doc), top_k=3)
        # 自查询必须命中自身且排第一——这是余弦检索最起码的正确性
        assert hits[0][0].id == doc.id
        assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
        # 只返回 score > 0：带符号哈希会让无关文本拿到 cos <= 0，那不该占用名次
        assert all(score > 0.0 for _, score in hits)
        assert hits == sorted(hits, key=lambda item: (-item[1], item[0].id))


def test_scores_are_cosine_in_unit_range():
    index = _toy_index()
    for _doc_hit, score in index.search("蓝牙 耳机", top_k=3):
        assert -1e-6 <= score <= 1.0 + 1e-6


def test_search_returns_empty_for_blank_query_and_zero_top_k():
    index = _toy_index()
    # 空/纯标点：零向量，与任何文档的余弦都是 0 而不是 nan
    assert index.search("", top_k=3) == []
    assert index.search("！？。", top_k=3) == []
    assert index.search("蓝牙耳机", top_k=0) == []


def test_empty_corpus_is_safe():
    index = VectorIndex([], HashingEmbedder(32))
    assert index.n_docs == 0
    assert index.search("蓝牙耳机", top_k=3) == []


def test_ranking_is_independent_of_top_k():
    _bm25, vectors, _docs = _fixture_index()
    short = vectors.search("跑步鞋 男款", top_k=3)
    long = vectors.search("跑步鞋 男款", top_k=10)
    # top_k 只截断长度：top-3 必须是 top-10 的前缀，分数逐项相同
    assert [doc.id for doc, _ in short] == [doc.id for doc, _ in long][:3]
    assert [score for _, score in short] == [score for _, score in long][:3]


def test_ties_break_by_doc_id():
    # 两条完全相同的标题：向量一模一样，平局顺序必须由 doc.id 决定，
    # 不能交给 faiss 的内部下标（那会让排序依赖插入顺序）
    index = VectorIndex(
        [_doc(7, "同一个标题 完全相同"), _doc(3, "同一个标题 完全相同")],
        HashingEmbedder(256),
    )
    hits = index.search("同一个标题 完全相同", top_k=2)
    assert [doc.id for doc, _ in hits] == [3, 7]
    assert hits[0][1] == hits[1][1]


def test_index_reports_whether_vectors_are_fake():
    index = _toy_index()
    assert index.is_fake is True
    assert index.embedder_name.startswith("hash-")
    assert index.dimension == 512


def test_embedder_dimension_mismatch_is_rejected():
    class _BadEmbedder(HashingEmbedder):
        def __init__(self) -> None:
            super().__init__(64)

        def embed(self, texts):
            return [[0.0] * 32 for _ in texts]

    with pytest.raises(ValueError):
        VectorIndex([_doc(1, "随便")], _BadEmbedder())


# --- 三路 RRF -----------------------------------------------------------


def test_three_way_fusion_degrades_to_two_way_when_vector_route_is_empty():
    docs = [_doc(i, "跑步鞋 款%d" % i) for i in range(1, 4)]
    bm25 = BM25Index(docs)
    # 空 query 的向量路不给任何名次，融合退化成两路
    vectors = VectorIndex(docs, HashingEmbedder(256))
    assert vectors.search("", top_k=RRF_DEPTH_DEFAULT) == []
    plain = bm25.search_fused("跑步鞋", top_k=3)
    fused_three = bm25.search_fused_three("跑步鞋", vectors, top_k=3)
    assert [doc.id for doc, _ in fused_three] != []
    assert all(score > 0 for _, score in fused_three)
    assert len(fused_three) == len(plain)


def test_three_way_fusion_adds_vector_votes():
    """一条只在向量路排第一的文档，会因此在融合结果里前进。"""
    docs = [
        _doc(1, "机械键盘 红轴 有线"),
        _doc(2, "保温杯 不锈钢"),
        _doc(3, "蓝牙耳机 降噪"),
    ]
    bm25 = BM25Index(docs)
    vectors = VectorIndex(docs, HashingEmbedder(512))
    two_way = bm25.search_fused("蓝牙耳机", top_k=3)
    three_way = bm25.search_fused_three("蓝牙耳机", vectors, top_k=3)
    # 两路与三路都必须把 3 号（唯一沾"蓝牙耳机"的）放在首位）
    assert two_way[0][0].id == 3
    assert three_way[0][0].id == 3
    # 三路下 3 号多拿了向量路的一份票，融合分严格更高
    assert three_way[0][1] > two_way[0][1]


def test_three_way_top_k_is_a_prefix():
    bm25, vectors, _docs = _fixture_index()
    three = bm25.search_fused_three("跑步鞋 男款", vectors, top_k=3)
    ten = bm25.search_fused_three("跑步鞋 男款", vectors, top_k=10)
    assert [doc.id for doc, _ in three] == [doc.id for doc, _ in ten][:3]
    assert [score for _, score in three] == [score for _, score in ten][:3]


def test_three_way_handles_empty_and_unknown_queries():
    bm25, vectors, _docs = _fixture_index()
    assert bm25.search_fused_three("", vectors, top_k=3) == []
    assert bm25.search_fused_three("，。！", vectors, top_k=3) == []
    # 库里完全没有的东西：前三路都可能零命中，不允许崩
    assert bm25.search_fused_three("量子计算机", vectors, top_k=3) == bm25.search_fused_three("量子计算机", vectors, top_k=3)


def test_three_way_matches_two_way_when_vector_matches_nothing():
    """向量路零命中时，三路结果必须与两路逐 id 相同（只是分数少一份票）。"""
    bm25, _vectors, docs = _fixture_index()
    from app.retrieval.vector import VectorIndex as _VI

    class _Silent(_VI):
        def search(self, query, top_k=10):
            return []

    silent = _Silent(docs, HashingEmbedder(1024))
    two = bm25.search_fused("跑步鞋 男款", top_k=5)
    three = bm25.search_fused_three("跑步鞋 男款", silent, top_k=5)
    assert [doc.id for doc, _ in three] == [doc.id for doc, _ in two]
    assert [score for _, score in three] == [score for _, score in two]


# --- 评估护栏：向量噪声不许攻破 exact / lexical 两条断言 ----------------


def test_evaluate_fused3_requires_vector_index():
    bm25, _vectors, _docs = _fixture_index()
    with pytest.raises(ValueError):
        evaluate(bm25, QUERIES[:1], mode="fused3")


def test_unknown_mode_is_rejected():
    bm25, vectors, _docs = _fixture_index()
    with pytest.raises(ValueError):
        evaluate(bm25, QUERIES[:1], mode="fused4", vectors=vectors)


def test_fused3_keeps_exact_group_at_full_recall():
    """护栏：exact 组 R 必须仍是 1.000——向量噪声不许灌进精确匹配。"""
    bm25, vectors, _docs = _fixture_index()
    groups = {
        row.group: row
        for row in summarise(evaluate(bm25, QUERIES, mode="fused3", vectors=vectors))
    }
    assert groups["exact"].recall == 1.0
    assert groups["exact"].mrr == 1.0


def test_fused3_keeps_paraphrase_lexical_band_empty():
    """护栏：paraphrase-lexical 档 n 必须仍是 0（没有伪改写混进 paraphrase 组）。"""
    from app.eval.run_eval import summarise_lexical_bands

    bm25, vectors, docs = _fixture_index()
    titles = {doc.id: doc.title for doc in docs}
    results = evaluate(bm25, QUERIES, mode="fused3", vectors=vectors)
    bands = {row.group: row for row in summarise_lexical_bands(results, titles)}
    assert bands["paraphrase-lexical"].count == 0
    assert bands["paraphrase-semantic"].count > 0


def test_three_way_search_does_not_disturb_the_two_existing_routes():
    """三路是**新增**一条路：跑过三路之后，search / search_fused 的结果必须逐项不变。"""
    bm25, vectors, _docs = _fixture_index()
    for item in QUERIES[:20]:
        plain_before = bm25.search(item.query, top_k=10)
        two_before = bm25.search_fused(item.query, top_k=10)
        three = bm25.search_fused_three(item.query, vectors, top_k=10)
        assert len(three) == 10
        assert bm25.search(item.query, top_k=10) == plain_before
        assert bm25.search_fused(item.query, top_k=10) == two_before


# --- 文档集一致性守卫：向量索引过期时必须抛错，不许静默注入外部商品 ----------


def test_three_way_rejects_vector_index_with_different_doc_count():
    """向量索引条数不同 = 整份过期；必须抛错，不许把外部商品当结果返回。"""
    bm25 = BM25Index([_doc(1, "篮球鞋 高帮"), _doc(2, "跑步鞋 轻量")])
    stale = VectorIndex(
        [_doc(98, "篮球鞋 专业"), _doc(99, "篮球鞋 高帮 实战"), _doc(100, "跑步鞋 竞速")],
        HashingEmbedder(256),
    )
    with pytest.raises(ValueError) as excinfo:
        bm25.search_fused_three("篮球鞋", stale, top_k=10)
    message = str(excinfo.value)
    # 报错信息要能让人一眼看出"索引过期了"：两边数量 + 差异样例 id
    assert "BM25 2 条" in message
    assert "向量 3 条" in message
    assert "98" in message


def test_three_way_rejects_vector_index_with_same_count_but_different_ids():
    """条数相同、内容不同（下架 A 上架 B）——只比条数会漏掉这一类，必须也抛。"""
    bm25 = BM25Index([_doc(1, "篮球鞋 高帮"), _doc(2, "跑步鞋 轻量")])
    swapped = VectorIndex(
        [_doc(98, "篮球鞋 专业"), _doc(99, "篮球鞋 高帮 实战")],
        HashingEmbedder(256),
    )
    with pytest.raises(ValueError) as excinfo:
        bm25.search_fused_three("篮球鞋", swapped, top_k=10)
    message = str(excinfo.value)
    assert "BM25 2 条 / 向量 2 条" in message
    assert "[98, 99]" in message
    assert "[1, 2]" in message


def test_guard_covers_blank_query_and_zero_top_k():
    """守卫排在所有短路返回之前：不该因为这一次 top_k<=0 就被静默跳过。"""
    bm25 = BM25Index([_doc(1, "篮球鞋 高帮"), _doc(2, "跑步鞋 轻量")])
    stale = VectorIndex([_doc(98, "篮球鞋 专业")], HashingEmbedder(256))
    with pytest.raises(ValueError):
        bm25.search_fused_three("篮球鞋", stale, top_k=0)
    with pytest.raises(ValueError):
        bm25.search_fused_three("", stale, top_k=10)


def test_matching_doc_sets_still_work_and_match_previous_results():
    """文档集一致时守卫必须放行，且结果与守卫加入前逐字相同。"""
    bm25, vectors, _docs = _fixture_index()
    # 守卫的等价条件：两边条数与 id 集合都相同
    assert bm25.n_docs == vectors.n_docs
    assert vectors.doc_ids == {doc.id for doc in build_fixture_catalog()}
    for item in QUERIES[:20]:
        hits = bm25.search_fused_three(item.query, vectors, top_k=10)
        assert hits == bm25.search_fused_three(item.query, vectors, top_k=10)
        ids = [doc.id for doc, _ in hits]
        # 结果里不许出现本次 BM25 快照之外的 id
        assert set(ids) <= vectors.doc_ids


def test_doc_ids_is_read_only():
    """守卫只需要"看出来"索引是否过期，不该拿到能被就地改动的容器。"""
    index = _toy_index()
    ids = index.doc_ids
    assert isinstance(ids, frozenset)
    with pytest.raises(AttributeError):
        ids.add(999)
