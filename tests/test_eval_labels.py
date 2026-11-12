"""评估集自检：100 条 query、三组配比、标签指向真实商品、无重复。

外加三道**机械护栏**（见下），钉住评估集的结构性质而非某一次的具体分数：

- 禁止自证：paraphrase query 归一后不得是它任一期望标题的子串；
- 禁止高字面重合：paraphrase 的字符覆盖率必须 < LEXICAL_HIGH；
- exact 反向护栏：exact 的字符覆盖率必须 >= LEXICAL_HIGH。

为什么需要护栏：评估集自己也会腐化。手写标签最容易发生的退化不是"检索变差"，
而是"为了让分数好看去改标签"——把难 query 挪出 paraphrase、把期望 id 改小、
把改写换回原词、或者把阈值往下调一格。上面三条把这些退化变成一条红色的 pytest，
而不是评审时一句"感觉不太对"。所以这里刻意断言**结构**（子串关系、覆盖率区间），
不涉及任何检索结果：护栏不该随检索实现变化而变化。
"""

from __future__ import annotations

from collections import Counter

from app.eval.fixture_catalog import build_fixture_catalog
from app.eval.labels import GROUPS, QUERIES
from app.eval.metrics import LEXICAL_HIGH, char_coverage, max_char_coverage


def _titles() -> dict[int, str]:
    return {doc.id: doc.title for doc in build_fixture_catalog()}


def _norm(text: str) -> str:
    """与 char_coverage 同口径的去空白 + lower。"""
    return "".join(text.split()).lower()


def test_query_count():
    assert len(QUERIES) == 100


def test_group_distribution():
    counts = Counter(q.group for q in QUERIES)
    # 三组约 40 / 30 / 30，允许小幅偏移，但别让某一组消失
    assert 35 <= counts["exact"] <= 45
    assert 25 <= counts["partial"] <= 35
    assert 25 <= counts["paraphrase"] <= 35
    assert set(counts) == set(GROUPS)


def test_every_query_has_at_least_one_relevant():
    for q in QUERIES:
        assert any(rel >= 1 for rel in q.relevance.values()), q.query


def test_relevance_levels_are_valid():
    for q in QUERIES:
        for doc_id, rel in q.relevance.items():
            assert isinstance(doc_id, int), q.query
            assert rel in (1, 2), (q.query, doc_id, rel)


def test_relevance_ids_exist_in_fixture():
    catalog_ids = {doc.id for doc in build_fixture_catalog()}
    for q in QUERIES:
        missing = set(q.relevance) - catalog_ids
        assert not missing, (q.query, sorted(missing))


def test_queries_are_unique():
    queries = [q.query for q in QUERIES]
    assert len(queries) == len(set(queries))


# ---------------------------------------------------------------- 护栏


def test_paraphrase_queries_are_not_self_evident():
    """paraphrase 不许自证：归一后的 query 不能是任一期望标题的子串。

    反例（这条护栏就是为它写的）：query "降噪耳机" 标注期望 70 号，
    而 70 号标题逐字含 "降噪耳机"。那不是改写，是词面直配——它拿 R=1.000
    却记进 paraphrase 组，等于用 exact 的题算 paraphrase 的分。
    """
    titles = _titles()
    offenders = []
    for q in QUERIES:
        if q.group != "paraphrase":
            continue
        needle = _norm(q.query)
        for doc_id in q.relevance:
            if needle and needle in _norm(titles[doc_id]):
                offenders.append((q.query, doc_id, titles[doc_id]))
    assert not offenders, f"paraphrase query 是期望标题的子串（自证式标签）：{offenders}"


def test_paraphrase_queries_have_low_lexical_coverage():
    """paraphrase 覆盖率必须 < LEXICAL_HIGH：靠词面匹配能过的不算语义改写。

    口径取 max_char_coverage（所有期望标题里的最高值）：只要有一条期望商品
    光靠字面就能对上，这条 query 就测不到同义泛化，留在组里只会稀释
    P5-4 同义泛化带来的增益。不满足正确做法是改写 query，不是下调阈值。
    """
    titles = _titles()
    offenders = [
        (q.query, round(max_char_coverage(q.query, [titles[i] for i in q.relevance]), 3))
        for q in QUERIES
        if q.group == "paraphrase"
        and max_char_coverage(q.query, [titles[i] for i in q.relevance]) >= LEXICAL_HIGH
    ]
    assert not offenders, f"paraphrase query 字面重合过高（>= {LEXICAL_HIGH}）：{offenders}"


def test_exact_queries_have_high_lexical_coverage():
    """exact 反向护栏：覆盖率必须 >= LEXICAL_HIGH，否则 exact 混进了改写。

    两个组是互相校验的：若 exact 里出现低重合 query，它掉分到底是检索坏了
    还是标签写错了将无法区分；反过来若 paraphrase 里出现高重合 query，
    "语义能力"就变成了"词面能力"的美化说法。两个方向都要钉住。
    """
    titles = _titles()
    offenders = [
        (q.query, round(max_char_coverage(q.query, [titles[i] for i in q.relevance]), 3))
        for q in QUERIES
        if q.group == "exact"
        and max_char_coverage(q.query, [titles[i] for i in q.relevance]) < LEXICAL_HIGH
    ]
    assert not offenders, f"exact query 字面重合过低（< {LEXICAL_HIGH}，疑似改写混入）：{offenders}"


def test_char_coverage_ignores_whitespace_and_order():
    """护栏依赖的度量本身要有确定行为：空白不影响、字序不影响、重复不加分。"""
    assert char_coverage("跑步 鞋", "跑步鞋") == 1.0
    assert char_coverage("鞋跑步", "跑步鞋") == 1.0
    assert char_coverage("QQ", "q") == 1.0  # lower + 去重：两个 Q 只算一个字符
    assert char_coverage("   ", "跑步鞋") == 0.0
    assert char_coverage("", "跑步鞋") == 0.0
