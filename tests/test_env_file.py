"""纯标准库 .env 解析：注释/空行/引号/无等号行都必须容错。

这些测试全部只调纯函数，不碰真实 `.env`——那个文件由用户手工创建且被 .gitignore
忽略，测试去读写它既会污染环境，又会让"这台机器上碰巧有 .env"改变测试结果。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.env_file import apply_env_file, parse_env_file, parse_env_text


@pytest.fixture()
def workdir():
    """临时目录，用例结束即删。

    不用 pytest 的 tmp_path：它的根目录来自 tempfile.gettempdir()，而在 Windows 上
    只要那里拿不到 TEMP 就会落到进程 cwd（本仓库根目录），留下 pytest-of-<user>/
    一堆垃圾文件。这里的用例几乎全是"写一个 .env 文件"，不该往仓库里落任何东西。
    """
    with tempfile.TemporaryDirectory() as path:
        yield Path(path)


@pytest.fixture()
def no_dotenv(monkeypatch, workdir: Path):
    """"这台机器上没有 .env"——凡是断言默认值/纯环境变量的用例都必须用它。

    为什么必须把 .env 路径指走去：`load_settings()` 会读仓库根目录那份**用户手工
    维护的** .env。开发机上它一旦填了 EMBEDDING_API_KEY，"不设环境变量 → is_fake
    为 True"就不再成立 —— 测试结果取决于开发者本机恰好有没有那个文件，这正是
    "本机碰巧有 .env 就改测试结果"的坑。这里把路径换成临时目录里一个不存在的文件，
    让用例在任何机器上都读不到 .env。
    """
    import app.env_file as env_file

    monkeypatch.setattr(env_file, "DEFAULT_ENV_PATH", workdir / "absent.env")
    return workdir / "absent.env"


def test_parse_basic_key_value_pairs():
    parsed = parse_env_text("A=1\nB=two\n")
    assert parsed == {"A": "1", "B": "two"}


def test_parse_skips_comments_and_blank_lines():
    text = "# 顶部注释\n\nA=1\n   # 缩进的注释\n\nB=2\n"
    assert parse_env_text(text) == {"A": "1", "B": "2"}


def test_parse_skips_lines_without_equals():
    # 手滑写漏等号 / 写了半行：跳过而不是抛错，一个笔误不该让服务起不来
    text = "A=1\nJUST_A_WORD\n=无键\nB=2\n"
    assert parse_env_text(text) == {"A": "1", "B": "2"}


def test_parse_allows_export_prefix():
    assert parse_env_text("export A=1\n") == {"A": "1"}


def test_parse_splits_on_first_equals_only():
    parsed = parse_env_text("REDIS_URL=redis://h/0?x=1\n")
    assert parsed["REDIS_URL"] == "redis://h/0?x=1"


def test_parse_strips_matching_quotes_only():
    # 成对引号去掉；不配对的引号原样保留（写的人显然不是想引用）
    parsed = parse_env_text('A="带 空格"\nB=\'single\'\nC="没闭合\n')
    assert parsed["A"] == "带 空格"
    assert parsed["B"] == "single"
    assert parsed["C"] == '"没闭合'


def test_parse_handles_escapes_inside_double_quotes():
    parsed = parse_env_text('A="行1\\n行2"\nB="引号 \\" 里面"\n')
    assert parsed["A"] == "行1\n行2"
    assert parsed["B"] == '引号 " 里面'


def test_parse_keeps_non_escape_backslash_verbatim():
    # 认不出的转义原样保留：Windows 路径不能被吞掉反斜杠
    assert parse_env_text('P="C:\\path\\dir"\n')["P"] == "C:\\path\\dir"


def test_parse_does_not_expand_variables():
    # 不做变量展开：结果不依赖键的出现顺序，排查空值不用多跳一层
    assert parse_env_text("A=${B}\nB=1\n") == {"A": "${B}", "B": "1"}


def test_parse_inline_comment_needs_leading_whitespace():
    # 前面有空白的 # 算注释；紧贴值的 # 属于值本身（redis 的 db#frag 不能被截断）
    parsed = parse_env_text("A=1 # 说明\nB=redis://h/db#frag\n")
    assert parsed["A"] == "1"
    assert parsed["B"] == "redis://h/db#frag"


def test_parse_hash_inside_quotes_is_kept():
    assert parse_env_text('A="x # 不是注释"\n')["A"] == "x # 不是注释"


def test_parse_later_duplicate_wins():
    assert parse_env_text("A=1\nA=2\n") == {"A": "2"}


def test_parse_strips_bom():
    assert parse_env_text("\ufeffA=1\n") == {"A": "1"}


def test_parse_empty_text_is_empty_mapping():
    assert parse_env_text("") == {}


def test_parse_file_missing_returns_empty(workdir: Path):
    # .env 不存在是常态：服务与测试都必须照常工作
    assert parse_env_file(workdir / "nope.env") == {}
    assert parse_env_file(workdir / "no-such-dir" / "sample.env") == {}


def test_parse_file_reads_utf8(workdir: Path):
    target = workdir / "sample.env"
    target.write_text("A=中文值\n", encoding="utf-8")
    assert parse_env_file(target) == {"A": "中文值"}


def test_apply_env_file_does_not_override_real_environ(workdir: Path):
    target = workdir / "sample.env"
    target.write_text("SESSION_BACKEND=redis\nLLM_MODEL=from-dotenv\n", encoding="utf-8")
    environ = {"SESSION_BACKEND": "memory"}
    applied = apply_env_file(target, environ=environ)
    # 真实环境变量优先；只有它没设置的键才被 .env 填上
    assert environ["SESSION_BACKEND"] == "memory"
    assert environ["LLM_MODEL"] == "from-dotenv"
    assert applied == ("LLM_MODEL",)


def test_apply_env_file_treats_present_empty_string_as_a_decision(workdir: Path):
    # 显式设成空串是一个明确的决定（强制 fake 模式），不该被 .env 悄悄改成有 key
    target = workdir / "sample.env"
    target.write_text("LLM_API_KEY=sk-from-file\n", encoding="utf-8")
    environ = {"LLM_API_KEY": ""}
    applied = apply_env_file(target, environ=environ)
    assert environ["LLM_API_KEY"] == ""
    assert applied == ()


def test_apply_env_file_without_file_is_a_noop(workdir: Path):
    environ: dict[str, str] = {}
    assert apply_env_file(workdir / "missing.env", environ=environ) == ()
    assert environ == {}


def test_settings_loads_without_env_file(monkeypatch, no_dotenv):
    """load_settings 在 .env 不存在时照常工作。"""
    from app.config import load_settings

    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    settings = load_settings()
    assert settings.embedding_dimension >= 1
    assert isinstance(settings.embedding_is_fake, bool)


def test_settings_reads_embedding_keys_from_environ(monkeypatch, no_dotenv):
    from app.config import load_settings

    monkeypatch.setenv("EMBEDDING_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "256")
    settings = load_settings()
    assert settings.embedding_api_key == "sk-test"
    assert settings.embedding_model == "custom-model"
    assert settings.embedding_dimension == 256
    assert settings.embedding_is_fake is False


def test_settings_default_is_fake_without_key(monkeypatch, no_dotenv):
    from app.config import load_settings

    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    settings = load_settings()
    # 没有 key 必须是 fake：绝不假装调过模型
    assert settings.embedding_is_fake is True
    assert settings.embeddings_url.endswith("/embeddings")


def test_load_settings_applies_env_file_below_real_environ(monkeypatch, workdir: Path):
    """端到端：load_settings 会真的读 .env，但真实环境变量压在上面。"""
    import app.env_file as env_file
    from app.config import load_settings

    target = workdir / "sample.env"
    target.write_text("LLM_MODEL=from-dotenv\nSESSION_WINDOW=99\n", encoding="utf-8")
    # 只换默认路径：这份文件不叫 .env，不碰用户手工维护的那一份
    monkeypatch.setattr(env_file, "DEFAULT_ENV_PATH", target)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("SESSION_WINDOW", "7")

    settings = load_settings()
    # .env 补上了环境变量没设的键
    assert settings.llm_model == "from-dotenv"
    # 环境变量已设的键不被 .env 覆盖
    assert settings.session_window == 7
