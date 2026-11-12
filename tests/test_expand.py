"""查询泛化：错字纠正、上位扩展、长键优先、无命中等价。"""

from __future__ import annotations

import dataclasses

import pytest

from app.retrieval import synonyms
from app.retrieval.expand import QueryExpansion, expand_query
from app.retrieval.tokenize import tokenize


def _extra(expansion: QueryExpansion) -> tuple[str, ...]:
    """扩展出来的增量 token（保持顺序），用来断言"多加了什么"。"""
    return tuple(t for t in expansion.expanded_tokens if t not in expansion.original_tokens)


def test_typo_is_corrected_before_tokenizing():
    expansion = expand_query("蓝芽耳机")
    assert "typo:蓝芽->蓝牙" in expansion.applied
    # 纠错后的词必须真的进了 token：光改字符串不改 token 等于没纠
    for token in tokenize("蓝牙"):
        assert token in expansion.expanded_tokens


def test_typo_table_keys_are_shorter_than_two_chars_only_when_harmless():
    # 反作弊约束的落地检查：错字表只收「无歧义的电商高频错写」，逐条都得能被纠对
    assert synonyms.TYPOS["运动些"] == "运动鞋"
    assert expand_query("运动些").applied[0] == "typo:运动些->运动鞋"
    assert "鞋" in expand_query("运动些").expanded_tokens


def test_hypernym_expands_to_subordinate_terms():
    expansion = expand_query("运动鞋")
    assert expansion.applied == ("hypernym:运动鞋",)
    extra = _extra(expansion)
    # 下位词各自 tokenize 后并入，所以至少要看到下位词的单字与二元组
    for token in ("跑步", "板鞋", "帆布", "布鞋"):
        assert token in extra


def test_hypernym_is_one_directional():
    # 下位词本身不再向外扩展（双向会连锁炸词）
    for sub in synonyms.HYPERNYMS["运动鞋"]:
        if sub in synonyms.HYPERNYMS:
            pytest.fail(f"下位词 {sub} 不该同时是 HYPERNYMS 的键")


def test_long_key_wins_over_its_prefix():
    # "手提" 是 "手提电脑" 的前缀：两个键都在表里，长键必须先命中并吃掉这段区间，
    # 否则同一条查询会同时记下 synonym:手提电脑 与 synonym:手提（扩展词翻倍）
    long_hit = expand_query("手提电脑")
    assert long_hit.applied == ("synonym:手提电脑",)
    # 长键命中后 "手提" 的区间已被消费，规则列表里不许再出现短键
    assert "synonym:手提" not in long_hit.applied
    assert expand_query("手提").applied == ("synonym:手提",)


def test_long_key_does_not_swallow_short_key_elsewhere():
    # "手提" 出现在 "手提电脑" 之外的另一段文字里：这段区间没被长键消费，短键必须照常触发
    expansion = expand_query("手提电脑 手提")
    assert "synonym:手提电脑" in expansion.applied
    assert "synonym:手提" in expansion.applied


def test_earphone_plug_long_key_does_not_double_count():
    # "耳机塞" 与 "耳塞" 有重叠字符但键不同：只有更长的那个该被同一次扫描吃掉
    expansion = expand_query("耳机塞")
    assert expansion.applied.count("synonym:耳机塞") == 1
    assert "synonym:耳塞" not in expansion.applied


def test_no_hit_leaves_tokens_unchanged():
    expansion = expand_query("跑步鞋 男款")
    assert expansion.applied == ()
    assert expansion.expanded_tokens == expansion.original_tokens
    assert expansion.original_tokens == tuple(tokenize("跑步鞋 男款"))


def test_empty_and_punctuation_do_not_raise_and_never_empty_out():
    for query in ("", "，。！", "   "):
        expansion = expand_query(query)
        assert expansion.original_tokens == tuple(tokenize(query))
        assert expansion.expanded_tokens == expansion.original_tokens
        assert expansion.applied == ()
    assert expand_query("").expanded_tokens == ()


def test_expanded_is_superset_of_original_and_keeps_order():
    for query in ("运动鞋", "蓝芽耳机", "手提电脑", "球鞋", "跑步鞋 男款"):
        expansion = expand_query(query)
        n = len(expansion.original_tokens)
        assert expansion.expanded_tokens[:n] == expansion.original_tokens
        assert len(set(expansion.expanded_tokens)) == len(expansion.expanded_tokens)


def test_expansion_is_frozen_and_tuple_typed():
    expansion = expand_query("运动鞋")
    assert isinstance(expansion.original_tokens, tuple)
    assert isinstance(expansion.expanded_tokens, tuple)
    assert isinstance(expansion.applied, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        expansion.applied = ()  # type: ignore[misc]


def test_applied_entries_are_prefixed_by_rule_kind():
    kinds = {entry.split(":", 1)[0] for entry in expand_query("蓝芽运动鞋").applied}
    assert kinds <= {"typo", "synonym", "hypernym"}
    assert {"typo", "hypernym"} <= kinds
