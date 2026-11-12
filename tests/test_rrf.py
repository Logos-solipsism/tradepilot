"""RRF 融合：手算分、退化等价、名次深度截断。"""

from __future__ import annotations

from app.retrieval.bm25 import RRF_DEPTH_DEFAULT, RRF_K_DEFAULT, BM25Index
from app.retrieval.catalog import ProductDoc
from app.retrieval.expand import expand_query


def _doc(doc_id: int, title: str, description: str = "") -> ProductDoc:
    return ProductDoc(id=doc_id, title=title, description=description)


def _two_route_fixture() -> BM25Index:
    """「球鞋」在两条路上排出来的名次不一样，融合结果可以手算。

    第一路（原文 token 球/鞋/球鞋）：命中 1(篮球鞋)、2(跑步鞋)；3(运动水壶) 零分。
    第二路（扩展后多出 运/动/运动/动鞋/篮/篮球）：1、3、2 依次排列——
    "运动水壶" 靠 "运动" 二元组挤进第二路，这是融合唯一能救回它的途径。
    """
    return BM25Index(
        [
            _doc(1, "篮球鞋 高帮 实战"),
            _doc(2, "跑步鞋 轻量 缓震"),
            _doc(3, "运动水壶 500ml"),
        ]
    )


def _pre_fix_fused_ids(
    index: BM25Index, query: str, *, rrf_k: int = RRF_K_DEFAULT
) -> list[int]:
    """修复前 search_fused 的排序（只有两路前 depth 名参与），用来钉住「补位不碰头部」。"""
    expansion = expand_query(query)
    depth = min(RRF_DEPTH_DEFAULT, index.n_docs)
    first = index._score_tokens(list(expansion.original_tokens), depth)
    second = index._score_tokens(list(expansion.expanded_tokens), depth)
    scores: dict[int, float] = {}
    for ranking in (first, second):
        for rank, (doc, _bm25) in enumerate(ranking, start=1):
            scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def _ids(hits) -> list[int]:
    return [doc.id for doc, _ in hits]


def _long_fixture(n: int = 30) -> BM25Index:
    """30 条全部命中第一路，但两路前 10 名的并集只有 10 条 → 必须补位才够 top_k。"""
    return BM25Index([_doc(i, "运动鞋 款%d" % i) for i in range(1, n + 1)])



def test_expansion_preconditions_of_fixture():
    expansion = expand_query("球鞋")
    assert expansion.applied == ("synonym:球鞋",)
    assert "运动" not in expansion.original_tokens
    assert "运动" in expansion.expanded_tokens


def test_hand_computed_fusion_scores_and_order():
    index = _two_route_fixture()
    hits = index.search_fused("球鞋", top_k=3, rrf_k=60)
    by_id = {doc.id: score for doc, score in hits}
    # 第一路 [1, 2]；第二路 [1, 3, 2]（rank 从 1 起）
    assert by_id[1] == 2 / 61
    assert by_id[2] == 1 / 62 + 1 / 63
    assert by_id[3] == 1 / 62
    assert [doc.id for doc, _ in hits] == [1, 2, 3]


def test_second_route_only_doc_is_rescued_by_fusion():
    index = _two_route_fixture()
    assert [doc.id for doc, _ in index.search("球鞋", top_k=3)] == [1, 2]
    assert 3 in [doc.id for doc, _ in index.search_fused("球鞋", top_k=3)]


def test_ranking_is_independent_of_top_k():
    # top_k 只截断长度，不参与排序：top-2 必须是 top-3 的前缀，且分数不变
    index = _two_route_fixture()
    short = index.search_fused("球鞋", top_k=2, rrf_k=60)
    long = index.search_fused("球鞋", top_k=3, rrf_k=60)
    assert [doc.id for doc, _ in short] == [doc.id for doc, _ in long][:2]
    assert [score for _, score in short] == [score for _, score in long][:2]
    assert short[0][1] == 2 / 61
    assert short[1][1] == 1 / 62 + 1 / 63


def test_route_depth_is_a_fixed_constant():
    assert RRF_DEPTH_DEFAULT == 10
    # 索引里的文档少于常量时按文档数取（不会越界）；超过时只有前 10 名参与 RRF 打分
    index = _two_route_fixture()
    # 3 条文档：候选池本身就够，top_k 只是截断，两次结果逐项相同
    assert index.search_fused("球鞋", top_k=50) == index.search_fused("球鞋", top_k=3)

    many = BM25Index([_doc(i, "运动鞋 款%d" % i) for i in range(1, 21)])
    expansion = expand_query("运动鞋")
    assert len(many._score_tokens(list(expansion.original_tokens), min(RRF_DEPTH_DEFAULT, 20))) == 10
    hits = many.search_fused("运动鞋", top_k=50)
    # 20 条全部命中第一路：RRF 只能给前 10 名打分，后 10 名由补位补满（分数 0.0）
    assert len(hits) == 20
    assert all(score > 0 for _, score in hits[:RRF_DEPTH_DEFAULT])
    assert all(score == 0.0 for _, score in hits[RRF_DEPTH_DEFAULT:])


def test_identical_routes_degrade_to_plain_search_order():
    # 「跑步鞋 男款」一条规则都不命中：expanded == original，两路完全同序
    docs = [
        _doc(1, "轻量跑步鞋 男款"),
        _doc(2, "跑步鞋 女款"),
        _doc(3, "篮球鞋 高帮"),
    ]
    index = BM25Index(docs)
    expansion = expand_query("跑步鞋 男款")
    assert expansion.applied == ()
    assert expansion.expanded_tokens == expansion.original_tokens

    plain = index.search("跑步鞋 男款", top_k=3)
    fused = index.search_fused("跑步鞋 男款", top_k=3, rrf_k=60)
    assert [doc.id for doc, _ in fused] == [doc.id for doc, _ in plain]
    # 两路同分：融合分恰好是单路分的两倍（名次相同）
    for rank, (doc, score) in enumerate(fused, start=1):
        assert doc.id == plain[rank - 1][0].id
        assert score == 2 / (60 + rank)


def test_larger_rrf_k_flattens_head_gap():
    index = _two_route_fixture()
    tight = index.search_fused("球鞋", top_k=3, rrf_k=1)
    flat = index.search_fused("球鞋", top_k=3, rrf_k=60)
    tight_gap = tight[0][1] - tight[1][1]
    flat_gap = flat[0][1] - flat[1][1]
    assert tight_gap > flat_gap
    # 具体分数（rrf_k=1，两路 [1,2] / [1,3,2]）
    by_id = {doc.id: score for doc, score in tight}
    assert by_id[1] == 2 / 2
    assert by_id[2] == 1 / 3 + 1 / 4
    assert by_id[3] == 1 / 3
    assert flat[0][1] == 2 / 61


def test_default_rrf_k_is_the_documented_constant():
    index = _two_route_fixture()
    assert index.search_fused("球鞋", top_k=3) == index.search_fused(
        "球鞋", top_k=3, rrf_k=RRF_K_DEFAULT
    )


def test_empty_and_unmatched_queries_return_empty():
    index = _two_route_fixture()
    assert index.search_fused("", top_k=3) == []
    assert index.search_fused("，。！", top_k=3) == []
    assert index.search_fused("冰箱", top_k=3) == []


def test_top_k_zero_and_empty_index_are_safe():
    index = _two_route_fixture()
    assert index.search_fused("球鞋", top_k=0) == []
    assert BM25Index([]).search_fused("球鞋", top_k=3) == []


def test_ties_in_fused_score_break_by_doc_id():
    # doc 2 与 doc 3 在 rrf_k 相同下同分（都只在一路里排第 2）→ id 小的在前
    index = _two_route_fixture()
    hits = index.search_fused("球鞋", top_k=3, rrf_k=60)
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[1][0].id == 2 and hits[2][0].id == 3



def test_backfill_fills_short_fused_pool_to_top_k():
    index = _long_fixture(30)
    assert len(_pre_fix_fused_ids(index, "运动鞋")) == RRF_DEPTH_DEFAULT
    assert len(index.search_fused("运动鞋", top_k=20)) == 20
    assert len(index.search_fused("运动鞋", top_k=10)) == 10
    assert len(index.search_fused("运动鞋", top_k=50)) == 30
    # 候选耗尽后不再无限补：超过库容时只返回库里有的那么多
    assert len(index.search_fused("运动鞋", top_k=999)) == 30


def test_backfill_only_appends_and_never_reorders_the_head():
    index = _long_fixture(30)
    head = _pre_fix_fused_ids(index, "运动鞋")
    # 两路候选够 top_k：结果与修复前逐 id 相同
    assert _ids(index.search_fused("运动鞋", top_k=5)) == head[:5]
    assert _ids(index.search_fused("运动鞋", top_k=10)) == head
    # 候选不够 top_k：头部仍是修复前那一段，补位项只接在后面
    assert _ids(index.search_fused("运动鞋", top_k=30))[: len(head)] == head


def test_backfilled_scores_are_zero_and_sort_after_rrf_scores():
    index = _long_fixture(30)
    hits = index.search_fused("运动鞋", top_k=30)
    rrf_pool = len(_pre_fix_fused_ids(index, "运动鞋"))
    assert all(score > 0 for _, score in hits[:rrf_pool])
    assert all(score == 0.0 for _, score in hits[rrf_pool:])
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_backfill_follows_first_route_bm25_order():
    index = _long_fixture(30)
    head = _pre_fix_fused_ids(index, "运动鞋")
    head_ids = set(head)
    first_route = [doc.id for doc, _ in index.search("运动鞋", top_k=30)]
    tail = _ids(index.search_fused("运动鞋", top_k=30))[len(head) :]
    assert tail == [doc_id for doc_id in first_route if doc_id not in head_ids]


def test_top_k_lengths_are_nested_prefixes():
    index = _long_fixture(30)
    five = _ids(index.search_fused("运动鞋", top_k=5))
    ten = _ids(index.search_fused("运动鞋", top_k=10))
    fifty = _ids(index.search_fused("运动鞋", top_k=50))
    assert five == ten[:5]
    assert ten == fifty[:10]
    # 分数也逐项一致：top_k 只截断长度
    assert [s for _, s in index.search_fused("运动鞋", top_k=5)] == [
        s for _, s in index.search_fused("运动鞋", top_k=50)
    ][:5]


def test_identical_routes_match_plain_search_even_when_backfilling():
    # 「跑步鞋」一条规则都不命中：expanded == original；30 条全命中，必须补位到 30
    docs = [_doc(i, "跑步鞋 款%d" % i) for i in range(1, 31)]
    index = BM25Index(docs)
    expansion = expand_query("跑步鞋")
    assert expansion.applied == ()
    assert expansion.expanded_tokens == expansion.original_tokens

    plain = _ids(index.search("跑步鞋", top_k=30))
    assert len(plain) == 30
    assert _ids(index.search_fused("跑步鞋", top_k=30)) == plain
