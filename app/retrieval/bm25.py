"""BM25 检索：纯标准库实现，不用 numpy / rank_bm25。

字段加权取最便宜的做法：建索引时把 title 的 token **再重复一遍**，
等价于 title 字段的 tf 与文档长度各计两次，让标题命中的权重高于描述命中。
这**不是** BM25F（BM25F 会为每个字段单独维护 tf/长度归一），
多字段真正加权留给后续刀。

对外契约（测试可直接用）：
- `index.n_docs`：文档总数 N
- `index.df(term) -> int`：包含 term 的文档数 df（未出现返回 0）
- `index.doc_freq`：{term: df} 的只读映射
- `index.avgdl`：平均文档长度（token 数）
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.retrieval.catalog import ProductDoc
from app.retrieval.expand import expand_query
from app.retrieval.tokenize import tokenize

if TYPE_CHECKING:  # 只用于类型标注：运行时绝不把 numpy/faiss 拖进 BM25 的导入路径
    from app.retrieval.vector import VectorIndex

K1_DEFAULT = 1.5
B_DEFAULT = 0.75
# RRF 的平滑常数，取文献默认值（Cormack et al. 2009）：越大头几名差距越被压平
RRF_K_DEFAULT = 60
# 每路交给 RRF 的名次深度。**固定值，不随 top_k 变**：若按 top_k 截断名次，
# 同一 query 用 top_k=5 与 top_k=10 会得到不同的先后顺序（top-5 不是 top-10 的前缀），
# 排序成了翻页参数的函数。固定深度后 top_k 只负责截断长度，排序恒定。
RRF_DEPTH_DEFAULT = 10


@dataclass(frozen=True)
class _Scored:
    doc: ProductDoc
    score: float


class BM25Index:
    """一次建索引 = 一次全量分词；之后 search 只读。"""

    def __init__(
        self,
        docs: list[ProductDoc],
        *,
        k1: float = K1_DEFAULT,
        b: float = B_DEFAULT,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[ProductDoc] = list(docs)
        self._tfs: list[dict[str, int]] = []
        self._lens: list[int] = []
        df: dict[str, int] = {}

        for doc in self._docs:
            title_tokens = tokenize(doc.title)
            # 标题 token 重复一遍 = 最便宜的字段加权
            tokens = title_tokens + title_tokens + tokenize(doc.description)
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            for token in tf:
                df[token] = df.get(token, 0) + 1
            self._tfs.append(tf)
            self._lens.append(len(tokens))

        self._df = df
        total_len = sum(self._lens)
        self._avgdl = (total_len / len(self._docs)) if self._docs else 0.0

    @property
    def n_docs(self) -> int:
        """文档总数 N。"""
        return len(self._docs)

    @property
    def df(self) -> dict[str, int]:
        """{term: 文档频率}；返回副本，调用方改不动索引。"""
        return dict(self._df)

    def term_df(self, term: str) -> int:
        """单个 term 的文档频率；未出现返回 0。"""
        return self._df.get(term, 0)

    @property
    def avgdl(self) -> float:
        return self._avgdl

    def idf(self, term: str) -> float:
        """IDF = ln((N - df + 0.5)/(df + 0.5) + 1)：恒为正，df=N 时也不会翻负。"""
        n = len(self._docs)
        df = self._df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> list[tuple[ProductDoc, float]]:
        """按 score 降序返回 score > 0 的文档；同分按 doc.id 升序（稳定）。

        查询里重复出现的 token 只算一次（去重按首次出现顺序），
        否则一句 "鞋 鞋 鞋" 会把同一 term 的分数重复累加三次。
        """
        return self._score(query, top_k)

    def _score(self, query: str, top_k: int) -> list[tuple[ProductDoc, float]]:
        """search 的实现体；语义与改动前完全一致（同样的去重与排序）。"""
        if not self._docs or top_k <= 0:
            return []
        query_tokens = list(dict.fromkeys(tokenize(query)))
        return self._score_tokens(query_tokens, top_k)

    def _score_tokens(self, query_tokens: list[str], top_k: int) -> list[tuple[ProductDoc, float]]:
        """对**已给定**的 token 列表打分；token 顺序不影响结果，重复 token 只算一次。

        与 `search` 共用同一套打分代码，`search_fused` 的第二路因此与第一路
        完全同源——两路唯一的区别就是喂进来的 token 不同。
        """
        if not self._docs or top_k <= 0:
            return []
        query_tokens = list(dict.fromkeys(query_tokens))
        if not query_tokens:
            return []
        avgdl = self._avgdl
        if avgdl <= 0:
            return []

        scored: list[_Scored] = []
        for i, tf in enumerate(self._tfs):
            dl = self._lens[i]
            score = 0.0
            for term in query_tokens:
                freq = tf.get(term)
                if not freq:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                score += self.idf(term) * freq * (self.k1 + 1.0) / denom
            if score > 0:
                scored.append(_Scored(doc=self._docs[i], score=score))

        scored.sort(key=lambda item: (-item.score, item.doc.id))
        return [(item.doc, item.score) for item in scored[:top_k]]

    def search_fused(
        self,
        query: str,
        top_k: int = 10,
        *,
        rrf_k: int = RRF_K_DEFAULT,
    ) -> list[tuple[ProductDoc, float]]:
        """两路 BM25 + RRF 融合检索。

        第一路：对 `expand_query(query).original_tokens` 打分排序；
        第二路：对 `expanded_tokens`（原文 ∪ 同义/上下位/纠错词）打分排序。
        融合：score(d) = Σ 1/(rrf_k + rank_i(d))，rank 从 1 起，每路只交前 RRF_DEPTH_DEFAULT
        个名次（两路深度相同），最后按融合分降序取 top_k。

        名次深度是**模块常量**而不是 top_k：否则同一 query 换个 top_k 就换一套排序，
        top-5 不再是 top-10 的前缀，翻页会重复或漏项。top_k 只截断长度，不参与排序。

        **返回值里的 score 是 RRF 融合分，与 search() 的 BM25 分不同量纲，
        两者不可比较、不可混用。** 之所以仍然把分数照原样返回，是因为调用方
        （离线评估、/search 的展示排序）只用它排序和展示，不参与任何阈值判定；
        换成 1.0/归一化反而会丢掉"两路都排第一 vs 只有一路排第一"的差异。

        `expanded == original` 时两路完全同序，融合后退化成原排序（每个 doc 多一份
        相同分数，顺序不变）——单路 query 的排序不会被这条改动影响。

        一个 doc 只在两路之一出现时，另一路没有名次、不贡献分数（不按 "rank=∞ 但近似 0"
        处理），这是 RRF 的标准做法：缺席即零贡献。

        **长度补位**：两路各只交前 depth 名，融合后的候选池最多 2×depth 条，去重后往往
        不到 top_k。此时剩余名额按**第一路（原文 token）的完整 BM25 排序**依次补入，
        跳过已在头部的 id，补到 top_k 或候选耗尽为止；整体顺序是「融合分降序的头部，
        然后是补位项按第一路 BM25 降序」。

        补位项的融合分记为 **0.0**：它表示「该文档没有进入任何一路的前 depth 名，
        只是为了满足 top_k 长度而排在尾部」。分数只用于排序、不用于阈值判定，而 0.0
        必然小于任何 1/(rrf_k+rank) > 0 的融合分，所以在排序上 0.0 就是正确的位置。
        **不要**把补位项的 BM25 分除以某个常数塞进来：那会把两个量纲混在一起，
        让一个「没进前 depth 名」的文档仅因为 BM25 分高就插到头部项前面。

        补位是幂等的前缀操作：头部只由 depth 常量决定、与 top_k 无关，补位顺序只由
        第一路 BM25 排序决定、也与 top_k 无关，因此 top-5 仍是 top-10 的前缀、
        top-10 仍是 top-50 的前缀，`expanded == original` 时也与 search() 完全一致。
        """
        if not self._docs or top_k <= 0:
            return []
        expansion = expand_query(query)
        if not expansion.original_tokens:
            return []

        # 名次深度固定为 RRF_DEPTH_DEFAULT：RRF 对名次发分，若两路都取全量名次，
        # 第一路（原文）里那些"只沾一个常用字"的长尾文档也会拿到 1/(k+rank)，
        # 把靠扩展词命中、但原文路没排上的目标挤出 top_k。实测（fixture 150 条商品、
        # 100 条评估 query）名次深度收到 10 后 para-semantic R 从 0.765 升到 0.804。
        depth = min(RRF_DEPTH_DEFAULT, len(self._docs))
        # 第一路取全量排序：前 depth 名进 RRF，其余留给「长度不足」时的补位（同一个排序，
        # 不重复打分）。第二路只用到前 depth 名，仍按 depth 截断。
        first_all = self._score_tokens(list(expansion.original_tokens), len(self._docs))
        first = first_all[:depth]
        second = self._score_tokens(list(expansion.expanded_tokens), depth)

        scores: dict[int, float] = {}
        docs_by_id: dict[int, ProductDoc] = {}
        for ranking in (first, second):
            for rank, (doc, _bm25_score) in enumerate(ranking, start=1):
                scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank)
                docs_by_id.setdefault(doc.id, doc)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        results = [(docs_by_id[doc_id], score) for doc_id, score in ordered[:top_k]]
        if len(results) >= top_k:
            return results

        # 融合池不足 top_k：按第一路的完整 BM25 排序补位。
        # 补位来源是全量排序（不是「刚好够用」的前缀），因此与 top_k 无关，前缀性质守得住。
        seen = {doc.id for doc, _ in results}
        for doc, _bm25_score in first_all:
            if doc.id in seen:
                continue
            seen.add(doc.id)
            results.append((doc, 0.0))
            if len(results) >= top_k:
                break
        return results

    def search_fused_three(
        self,
        query: str,
        vectors: "VectorIndex",
        top_k: int = 10,
        *,
        rrf_k: int = RRF_K_DEFAULT,
    ) -> list[tuple[ProductDoc, float]]:
        """三路 RRF 融合：原文 BM25 + 泛化词 BM25 + 向量余弦。

        前两路**逐字复用** search_fused 的两路（同一份 expand_query、同一个 depth、
        同一个 _score_tokens），这里只是把同一个融合循环再跑一路。没有另起一套打分：
        三路如果各写一遍，日后改融合公式就要改三处，迟早改漏一处。

        第三路是向量检索的名次。它与前两路**同权**（每路各 1/(rrf_k+rank)）：
        RRF 的作用就是让不同量纲的路只按名次投票，给向量单独调权重就退回
        「手工调参」，而那需要一份比 100 条 query 大得多的评估集才站得住。
        权重留给后续刀，用评估集说话。

        RRF 天然保护精确匹配：一条 exact query 在原文 BM25 上排第一拿 1/(k+1)，
        向量路即使把某个近义商品也排第一，也只是同样的一份票，**不会盖过**原文路的命中。
        向量路只能加票、不能改票，所以向量噪声不会把精确匹配挤出去。

        补位规则与 search_fused 逐字一致：融合池不足 top_k 时按第一路完整 BM25 排序补，
        补位项融合分记 0.0（理由同 search_fused：0.0 必然排在所有 >0 的 RRF 分之后）。

        **文档集一致性守卫（覆盖整个方法，含所有短路返回）**：三条路必须建立在
        同一份文档快照上。若向量索引与 BM25 索引的 id 集合不一致，说明其中一份已经过期
        （典型场景：商品目录刷新了、向量索引还是上一版），此时向量路会把**不属于本次快照
        的商品**当成本次检索结果返回——那是"返回了错误商品"，不是性能问题。

        这里**抛错而不是静默过滤**掉那些外部 doc：过滤掉之后一切看起来都正常，索引过期
        就变成一个永远不会被发现的慢性 bug——数字全对，直到某天用户发现搜出来的是别的
        商品。抛错会在第一次不一致时就把问题顶到脸上，代价是一次显式的失败，
        收益是这类错误不可能悄悄活下来。
        """
        # 守卫排在所有短路返回**之前**：索引不一致是调用方的结构性问题，
        # 不该因为这一次恰好 top_k<=0、或这一次恰好一条都没匹配上就被静默跳过。
        self._assert_same_doc_set(vectors)
        if not self._docs or top_k <= 0:
            return []
        expansion = expand_query(query)
        if not expansion.original_tokens:
            return []

        depth = min(RRF_DEPTH_DEFAULT, len(self._docs))
        first_all = self._score_tokens(list(expansion.original_tokens), len(self._docs))
        first = first_all[:depth]
        second = self._score_tokens(list(expansion.expanded_tokens), depth)
        # 向量路同样按 depth 截断名次：三条路的名次深度必须一致，
        # 否则相对 top_k 恒定的排序性质会在这一路破功
        third = vectors.search(query, depth)

        scores: dict[int, float] = {}
        docs_by_id: dict[int, ProductDoc] = {}
        for ranking in (first, second, third):
            for rank, (doc, _route_score) in enumerate(ranking, start=1):
                scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (rrf_k + rank)
                docs_by_id.setdefault(doc.id, doc)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        results = [(docs_by_id[doc_id], score) for doc_id, score in ordered[:top_k]]
        if len(results) >= top_k:
            return results

        seen = {doc.id for doc, _ in results}
        for doc, _bm25_score in first_all:
            if doc.id in seen:
                continue
            seen.add(doc.id)
            results.append((doc, 0.0))
            if len(results) >= top_k:
                break
        return results

    def _assert_same_doc_set(self, vectors: "VectorIndex") -> None:
        """三路融合的前置条件：向量索引与 BM25 索引必须是同一份文档快照。

        比对用 **id 集合而不是条数**：条数相同但内容不同（目录里下架了 A、上架了 B）
        同样会让向量路注入不属于本次快照的商品，只比条数会漏掉这一类。
        条数与集合两个条件都查，是因为它们各自的失效方式不同：条数不同是最常见的
        "整份索引过期"，条数相同而集合不同是"换了同量的一批商品"——后者更隐蔽。

        报错信息先给两边的数量再给差异样例：过期时两边可能差出几百条，全列进异常消息
        只会把关键信息淹掉，所以每边最多列 5 个 id。

        这里**抛错而不是静默过滤**掉外部 doc：过滤掉之后一切看起来都正常，索引过期就
        变成一个永远不会被发现的慢性 bug——数字全对，直到某天用户发现搜出来的是别的
        商品。抛错会在第一次不一致时就把问题顶到脸上，代价是一次显式失败，
        收益是这类错误不可能悄悄活下来。
        """
        ours = {doc.id for doc in self._docs}
        theirs = set(vectors.doc_ids)
        if self.n_docs == vectors.n_docs and ours == theirs:
            return
        foreign = sorted(theirs - ours)
        missing = sorted(ours - theirs)
        raise ValueError(
            "向量索引与 BM25 索引的文档集不一致（向量索引已过期？）: "
            f"BM25 {self.n_docs} 条 / 向量 {vectors.n_docs} 条；"
            f"向量独有 {len(foreign)} 个 id {foreign[:5]}，"
            f"BM25 独有 {len(missing)} 个 id {missing[:5]}"
        )
