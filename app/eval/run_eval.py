"""跑一次离线评估：固定语料 -> BM25 -> 100 条 query -> 分组指标。

用法（在 tradepilot 根目录）：
    python -m app.eval.run_eval                    # BM25 单路，全量 100 条
    python -m app.eval.run_eval --mode fused       # BM25 + 泛化词 RRF 融合
    python -m app.eval.run_eval --mode fused --split holdout

`--mode` 决定用哪条检索路径，`--split` 决定在哪个子集上算指标
（dev/holdout 由 `dev_holdout_split` 按 sha256 首字符奇偶确定性切分，
不用随机数，所以任何人任何时候跑都是同一份切分）。

输出四段，便于不同用途：
1. 逐 query 的失败明细（未全召回的 query：期望 id vs 实际 top-3）；
2. 分组 + 总体的 Recall@10 / NDCG@10 / MRR@10 表格；
2.5 字面难度分档（**诊断用，不改任何指标定义**）：把 paraphrase 组按
    max_char_coverage 再切成 lexical / mid / semantic 三档。理由：paraphrase
    均值会被“靠词面就能过”的伪改写稀释，看不出同义泛化到底有没有帮助；
    semantic 档（cov < 0.4）才是真正的语义改写能力。
3. 最后一行压成单行的 REGRESSION 摘要——改检索后重跑，把这一行贴进
   README 或 PR 就是可对比的回归数字。摘要末尾额外带 para-semantic 三项，
   让这一行本身就能区分“整体均值涨了”与“语义改写真的变好了”。

失败的定义是 recall@k < 1.0，即"至少漏掉一个相关商品"。这不是说排序一定错，
但一条 query 没把全部相关商品挤进 top-k，就是后续要解释的差额。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from app.eval.fixture_catalog import build_fixture_catalog
from app.eval.labels import GROUPS, QUERIES, EvalQuery
from app.eval.metrics import (
    LEXICAL_HIGH,
    LEXICAL_MID,
    max_char_coverage,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from app.retrieval.bm25 import BM25Index, RRF_K_DEFAULT

TOP_K = 10
DETAIL_HEAD = 3

# 两路检索的统一签名：给 query 返回 [(doc, score)]，score 的量纲由实现自己决定
Searcher = Callable[[str, int], list]
MODES: tuple[str, ...] = ("bm25", "fused")
SPLITS: tuple[str, ...] = ("all", "dev", "holdout")

# 字面难度分档：(档名, 下界含, 上界不含)；上界 None = 无上界
LEXICAL_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("paraphrase-lexical", LEXICAL_HIGH, None),
    ("paraphrase-mid", LEXICAL_MID, LEXICAL_HIGH),
    ("paraphrase-semantic", 0.0, LEXICAL_MID),
)


@dataclass(frozen=True)
class QueryResult:
    query: EvalQuery
    ranked_ids: list[int]
    recall: float
    ndcg: float
    mrr: float


@dataclass(frozen=True)
class GroupResult:
    group: str
    count: int
    recall: float
    ndcg: float
    mrr: float


def dev_holdout_split(
    queries: Sequence[EvalQuery],
) -> tuple[list[EvalQuery], list[EvalQuery]]:
    """按 sha256(query) 首字符奇偶把 queries 确定性切成 (dev, holdout)。

    为什么不用 random：评估必须可复现——同一份 labels 任何时候、任何机器
    跑出来都必须是同一份切分，否则"holdout 提升了多少"这个数字无法复核。
    十六进制首字符的取值是 0-9a-f：偶数集合 {0,2,4,6,8,a,c,e} **是 8 个**，
    奇数集合 {1,3,5,7,9,b,d,f} 也是 8 个，因此对均匀哈希而言切分确实是 50/50
    （若误用 range(10) 取奇偶就会变成 6:10）。判定用 int(ch, 16) 取模，
    不是按字符本身是数字还是字母。
    """
    dev: list[EvalQuery] = []
    holdout: list[EvalQuery] = []
    for item in queries:
        digest = hashlib.sha256(item.query.encode("utf-8")).hexdigest()
        if int(digest[0], 16) % 2 == 0:
            dev.append(item)
        else:
            holdout.append(item)
    return dev, holdout


def select_split(queries: Sequence[EvalQuery], split: str) -> list[EvalQuery]:
    """按 split 名取子集；非法名字抛 ValueError（main 会转成 argparse 的报错）。"""
    if split not in SPLITS:
        raise ValueError(f"未知 split: {split!r}（可选 {', '.join(SPLITS)}）")
    if split == "all":
        return list(queries)
    dev, holdout = dev_holdout_split(queries)
    return dev if split == "dev" else holdout


def evaluate(
    index: BM25Index,
    queries: list[EvalQuery],
    top_k: int = TOP_K,
    *,
    mode: str = "bm25",
    rrf_k: int = RRF_K_DEFAULT,
) -> list[QueryResult]:
    """对每条 query 取 top_k 并算三个指标；纯函数，不打印。

    `mode` 只影响取候选的那一步：指标计算、分组、分档全部共用同一套代码，
    这样 bm25 与 fused 两个数字的差异只可能来自检索本身。
    """
    if mode not in MODES:
        raise ValueError(f"未知 mode: {mode!r}（可选 {', '.join(MODES)}）")
    results: list[QueryResult] = []
    for item in queries:
        if mode == "fused":
            hits = index.search_fused(item.query, top_k=top_k, rrf_k=rrf_k)
        else:
            hits = index.search(item.query, top_k=top_k)
        ranked_ids = [doc.id for doc, _ in hits]
        results.append(
            QueryResult(
                query=item,
                ranked_ids=ranked_ids,
                recall=recall_at_k(ranked_ids, item.relevance, top_k),
                ndcg=ndcg_at_k(ranked_ids, item.relevance, top_k),
                mrr=mrr_at_k(ranked_ids, item.relevance, top_k),
            )
        )
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarise(results: list[QueryResult], groups: tuple[str, ...] = GROUPS) -> list[GroupResult]:
    """按 group 汇总，末尾追加 overall。组内为空时给 0.0 而不是崩掉。"""
    out: list[GroupResult] = []
    buckets = [(g, [r for r in results if r.query.group == g]) for g in groups]
    buckets.append(("overall", list(results)))
    for name, bucket in buckets:
        out.append(
            GroupResult(
                group=name,
                count=len(bucket),
                recall=_mean([r.recall for r in bucket]),
                ndcg=_mean([r.ndcg for r in bucket]),
                mrr=_mean([r.mrr for r in bucket]),
            )
        )
    return out


def lexical_band(coverage: float) -> str:
    """按覆盖率给档名；三档边界与 LEXICAL_BANDS 一致（左闭右开）。"""
    for name, low, high in LEXICAL_BANDS:
        if coverage >= low and (high is None or coverage < high):
            return name
    return LEXICAL_BANDS[-1][0]


def summarise_lexical_bands(
    results: list[QueryResult],
    titles: Mapping[int, str],
) -> list[GroupResult]:
    """把 paraphrase 组按字面覆盖率切三档，各算 R/N/M。

    覆盖率取 `max_char_coverage`（该 query 在所有期望标题里的最高值）：
    只要有一条期望商品能靠字面命中，这条 query 就测不到语义泛化。
    只对 group == "paraphrase" 生效，其余组不受影响——**指标定义没有任何改动**，
    这里只是把同一个 paraphrase 均值换个切法看一眼。
    """
    out: list[GroupResult] = []
    para = [r for r in results if r.query.group == "paraphrase"]
    for name, _low, _high in LEXICAL_BANDS:
        bucket = [
            r
            for r in para
            if lexical_band(max_char_coverage(r.query.query, [titles[i] for i in r.query.relevance])) == name
        ]
        out.append(
            GroupResult(
                group=name,
                count=len(bucket),
                recall=_mean([r.recall for r in bucket]),
                ndcg=_mean([r.ndcg for r in bucket]),
                mrr=_mean([r.mrr for r in bucket]),
            )
        )
    return out


def _fmt_expected(relevance: dict[int, int], group: str, limit: int = 8) -> str:
    # 组内相关商品可能非常多（"运动鞋" 有 24 条），只列 rel=2 与最靠前的几条
    ids = sorted(relevance.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = [f"{doc_id}({rel})" for doc_id, rel in ids[:limit]]
    if len(ids) > limit:
        shown.append(f"...共{len(ids)}条")
    return " ".join(shown)


def format_report(
    results: list[QueryResult],
    groups: list[GroupResult],
    *,
    top_k: int = TOP_K,
    n_docs: int = 0,
    lexical_bands: list[GroupResult] | None = None,
    mode: str = "bm25",
    split: str = "all",
) -> list[str]:
    lines: list[str] = []
    mode_label = "BM25" if mode == "bm25" else "BM25+扩展词 RRF"
    lines.append("=" * 78)
    lines.append(
        f"Tradepilot 离线检索评估 | {mode_label} | top-{top_k} | "
        f"{n_docs} 条固定商品 / {len(results)} 条 query | split={split}"
    )
    lines.append("=" * 78)

    failures = [r for r in results if r.recall < 1.0]
    lines.append("")
    lines.append(f"[1] 失败明细（recall@{top_k} < 1.0，共 {len(failures)} 条）")
    lines.append("-" * 78)
    if not failures:
        lines.append("（无）")
    for item in failures:
        head = " ".join(str(doc_id) for doc_id in item.ranked_ids[:DETAIL_HEAD]) or "（空）"
        lines.append(f"  q: {item.query.query}")
        lines.append(f"     group={item.query.group}  R={item.recall:.3f} N={item.ndcg:.3f} M={item.mrr:.3f}")
        lines.append(f"     期望 id: {_fmt_expected(item.query.relevance, item.query.group)}")
        lines.append(f"     实际 top-{DETAIL_HEAD}: {head}")

    lines.append("")
    lines.append(f"[2] 分组指标")
    lines.append("-" * 78)
    header = f"{'group':<12}{'n':>5}{'Recall@' + str(top_k):>12}{'NDCG@' + str(top_k):>12}{'MRR@' + str(top_k):>12}"
    lines.append(header)
    for row in groups:
        lines.append(
            f"{row.group:<12}{row.count:>5}{row.recall:>12.3f}{row.ndcg:>12.3f}{row.mrr:>12.3f}"
        )

    if lexical_bands:
        lines.append("")
        lines.append("[2.5] paraphrase 按字面难度分档（诊断用，指标定义未改）")
        lines.append("-" * 78)
        lines.append("  cov = query 字符集被期望标题覆盖的最高比例（metrics.max_char_coverage）")
        lines.append(f"  {'band':<22}{'cov 区间':<16}{'n':>4}{'Recall@' + str(top_k):>12}{'NDCG@' + str(top_k):>12}{'MRR@' + str(top_k):>12}")
        ranges = {"paraphrase-lexical": f">= {LEXICAL_HIGH:.1f}", "paraphrase-mid": f"{LEXICAL_MID:.1f} ~ {LEXICAL_HIGH:.1f}", "paraphrase-semantic": f"< {LEXICAL_MID:.1f}"}
        for row in lexical_bands:
            lines.append(
                f"  {row.group:<22}{ranges.get(row.group, '?'):<16}{row.count:>4}"
                f"{row.recall:>12.3f}{row.ndcg:>12.3f}{row.mrr:>12.3f}"
            )
        lines.append(
            "  （lexical 档靠词面就能过，n=0 说明护栏生效：没有伪改写留在 paraphrase 里；"
        )
        lines.append("    semantic 档才是真语义改写，看它而不是看被稀释过的均值）")

    lines.append("")
    lines.append("[3] 回归摘要（改检索后重跑，贴这一行做对比）")
    lines.append("-" * 78)
    parts = [
        f"{row.group} R={row.recall:.3f} N={row.ndcg:.3f} M={row.mrr:.3f} n={row.count}"
        for row in groups
    ]
    semantic = next((r for r in (lexical_bands or []) if r.group == "paraphrase-semantic"), None)
    if semantic is not None:
        parts.append(
            f"para-semantic R={semantic.recall:.3f} N={semantic.ndcg:.3f} M={semantic.mrr:.3f} n={semantic.count}"
        )
    # 行首打上 mode/split 标记：两份报告并排贴进 README 时不用靠上下文猜是哪一跑
    lines.append(f"{mode} | {split} | " + " | ".join(parts))
    return lines


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """只解析命令行；抽成函数便于测试直接喂 argv 列表。"""
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.run_eval",
        description="Tradepilot 离线检索评估：BM25 单路 vs 泛化词 RRF 融合。",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="bm25",
        help="检索路径：bm25 = 单路（默认，与改动前等价）；fused = BM25 + 扩展词 RRF",
    )
    parser.add_argument(
        "--split",
        choices=SPLITS,
        default="all",
        help="评估子集：all = 全部（默认）；dev/holdout = sha256 首字符奇偶的确定性 50/50 切分",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    docs = build_fixture_catalog()
    index = BM25Index(docs)
    titles = {doc.id: doc.title for doc in docs}
    queries = select_split(QUERIES, args.split)
    results = evaluate(index, queries, top_k=TOP_K, mode=args.mode)
    groups = summarise(results)
    bands = summarise_lexical_bands(results, titles)
    report = format_report(
        results,
        groups,
        top_k=TOP_K,
        n_docs=index.n_docs,
        lexical_bands=bands,
        mode=args.mode,
        split=args.split,
    )
    # 控制台可能是 GBK；统一按 UTF-8 写，避免中文乱码
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
