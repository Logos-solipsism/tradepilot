"""分词：字符 n-gram（零依赖，不引 jieba）；CJK 多字 run 同时出单字与相邻二元组。"""

from __future__ import annotations

from app.retrieval.tokenize import tokenize


def test_pure_chinese_produces_adjacent_bigrams():
    assert tokenize("跑步鞋") == ["跑", "步", "鞋", "跑步", "步鞋"]


def test_multi_char_run_emits_both_unigrams_and_adjacent_bigrams():
    # 语义：长度 >= 2 的 CJK run 既出单字也出相邻二元组（Lucene outputUnigrams=true）
    tokens = tokenize("跑步鞋")
    assert {"跑", "步", "鞋"} <= set(tokens)
    assert {"跑步", "步鞋"} <= set(tokens)
    # 只出"相邻"二元组：非相邻组合绝不能出现
    assert "跑鞋" not in tokens
    assert all(len(t) <= 2 for t in tokens)


def test_single_cjk_char_is_kept():
    assert tokenize("鞋") == ["鞋"]
    # 空白把 CJK run 切开，两侧都是单字
    assert tokenize("鞋 袜") == ["鞋", "袜"]


def test_mixed_chinese_and_ascii():
    assert tokenize("Nike 跑步鞋 42码") == ["nike", "跑", "步", "鞋", "跑步", "步鞋", "42", "码"]


def test_punctuation_and_emoji_are_dropped():
    assert tokenize("跑步鞋，好！🔥") == ["跑", "步", "鞋", "跑步", "步鞋", "好"]
    assert tokenize("!!!??") == []


def test_case_and_fullwidth_are_normalized():
    assert tokenize("NIKE") == ["nike"]
    # 全角字母数字经 NFKC 归一成半角
    assert tokenize("ＮＩＫＥ") == ["nike"]
    assert tokenize("４２") == ["42"]


def test_empty_input_returns_empty_list():
    assert tokenize("") == []
