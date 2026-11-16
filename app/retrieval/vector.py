"""向量索引：faiss IndexFlatIP 上的余弦检索。

为什么是 faiss 而不是 chromadb：chromadb 在本机（Windows + 它的 Rust binding）
**任何路径都起不来**（`unable to open database file`），而这里的规模是"几百到几万条
商品、单进程内存索引"，需要的只是"归一化向量 + 内积 top-k"。faiss 的 IndexFlatIP
正好就是这件事：暴力全扫，没有近似、没有量化误差，150 条和 10 万条都精确。
真正需要 ANN 时再换 IndexIVFFlat，接口不用动。

余弦 = L2 归一化后的内积：`IndexFlatIP` 本身只算内积，**归一化是这里的责任**。
两个实现（真 embedding / 哈希降级）都在这一步换算了，所以下面的阈值
（只保留 score > 0）在任何实现下含义一致：cos ≤ 0 就是"没有共同方向"。

与 BM25 对齐的两条约定，避免三条路各自一套语义：
1. 只返回 score > 0 的文档（BM25 的 search 同样只返回 score > 0）；
2. 同分按 doc.id 升序（BM25 的排序键就是 `(-score, doc.id)`）。
   平局顺序如果交给 faiss 的内部下标，排序就依赖插入顺序而不是数据本身。

要不要用 numpy 建矩阵：要。faiss 只吃 `float32` 的二维连续数组，
逐条 add 一次一条会让 10 万条建索引慢一个数量级；一次性 frombuffer 更干净。
"""

from __future__ import annotations

import numpy as np

from app.retrieval.catalog import ProductDoc
from app.retrieval.embedding import Embedder

# faiss 的 IndexFlatIP 只接受 float32；这里不暴露精度参数，
# 换成别的精度只会让"同一份数据在不同机器上算出的分数差一点点"变成一个新 bug
_DTYPE = np.float32


def doc_text(doc: ProductDoc) -> str:
    """商品进向量索引时用的文本。

    title 出现两次，与 BM25 建索引时"标题 token 重复一遍"的字段加权保持同一口径：
    两条路对"什么更重要"的判断一致，融合时它们的名次才有可比性。
    描述为空时不会留下多余分隔符造成空 token（分词会丢掉空白）。
    """
    return f"{doc.title} {doc.title} {doc.description}".strip()


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """按行做 L2 归一化；零向量保持零向量（不硬凑方向）。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 分母为 0 时用 1.0：结果仍是零向量，但不会产生 nan/inf 警告
    return matrix / np.where(norms == 0.0, 1.0, norms)


class VectorIndex:
    """一个实例 = 一份商品快照 + 一个 embedder；建好之后只读。"""

    def __init__(self, docs: list[ProductDoc], embedder: Embedder) -> None:
        self._docs: list[ProductDoc] = list(docs)
        self._embedder = embedder
        self.dimension = int(embedder.dimension)
        self._ids: list[int] = []
        self._index = None
        if not self._docs:
            return

        raw = embedder.embed([doc_text(doc) for doc in self._docs])
        if len(raw) != len(self._docs):
            raise ValueError(
                f"embedder 返回条数与文档数不符: 文档 {len(self._docs)} 条，向量 {len(raw)} 条"
            )
        matrix = self._to_matrix(raw)

        import faiss  # 局部导入：只有真正建向量索引时才需要这个重依赖

        index = faiss.IndexFlatIP(self.dimension)
        index.add(matrix)
        self._index = index
        self._ids = [doc.id for doc in self._docs]

    def _to_matrix(self, vectors: list[list[float]]) -> np.ndarray:
        """list[list[float]] → 归一化后的 float32 连续矩阵；维度不符立刻报错。"""
        for row in vectors:
            if len(row) != self.dimension:
                raise ValueError(
                    f"向量维度不符: 期望 {self.dimension}，收到 {len(row)}"
                )
        matrix = np.asarray(vectors, dtype=_DTYPE)
        if matrix.ndim != 2:
            raise ValueError("向量必须能组成二维矩阵")
        # np.ascontiguousarray：faiss 要求 C 连续内存，asarray 不保证（切片出来的数组就不是）
        return np.ascontiguousarray(_normalise_rows(matrix))

    @property
    def n_docs(self) -> int:
        return len(self._docs)

    @property
    def doc_ids(self) -> frozenset[int]:
        """本索引里有哪些 doc.id（只读）。

        给 BM25Index.search_fused_three 的「同一份文档集」守卫用：那条守卫只需要
        **看出来**向量索引是不是过期了，不该为此去碰检索逻辑，也不该拿到一个
        能被就地增删的可变容器——所以返回 frozenset 而不是 list 或 set。
        """
        return frozenset(self._ids)

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    @property
    def is_fake(self) -> bool:
        """True = 向量来自哈希降级，报告里必须如实标注。"""
        return bool(self._embedder.is_fake)

    def search(self, query: str, top_k: int = 10) -> list[tuple[ProductDoc, float]]:
        """按余弦降序返回 score > 0 的文档；同分按 doc.id 升序。"""
        if self._index is None or top_k <= 0:
            return []
        query_vector = self._embedder.embed([query])[0]
        if len(query_vector) != self.dimension:
            raise ValueError(
                f"query 向量维度不符: 期望 {self.dimension}，收到 {len(query_vector)}"
            )
        matrix = self._to_matrix([query_vector])
        if not matrix.any():
            # 零向量（空 query / 纯标点）：与任何文档的余弦都是 0，
            # 与其让 faiss 返回一堆 0 分候选占掉名次，不如明确地"这条路没有贡献"
            return []

        # k 取全量：IndexFlatIP 本来就是全表扫描，要前 k 个和要全部同样是一次扫描，
        # 但只有拿全量才能在 Python 侧按 (-score, doc.id) 重新定序，
        # 保证平局顺序由数据决定、不由 faiss 的内部下标决定
        scores, indices = self._index.search(matrix, len(self._docs))
        hits: list[tuple[ProductDoc, float]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            value = float(score)
            if value <= 0.0:
                continue
            hits.append((self._docs[int(index)], value))
        hits.sort(key=lambda item: (-item[1], item[0].id))
        return hits[:top_k]
