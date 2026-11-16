"""第三路的进程内缓存：文档向量（整份目录快照建一次）+ query 向量（有界 LRU + TTL）。

**两层缓存都用「list 对象身份」判换代，与 `_Bm25Cache` 同一套判据。**
为什么不换成"内容哈希 / id 集合"之类更聪明的判据：`search_fused_three` 有一条
文档集一致性守卫，BM25Index 与 VectorIndex 必须由**同一个 docs list** 构建，否则直接
抛 ValueError。判据一旦分叉，"BM25 已经换代、向量还停在上一版"就变成一个可以长期存在
的窗口——那正是守卫要防的事。同一份判据让两者的换代时刻必然对齐。

**query 向量为什么靠包装器而不是改 `vector.py`：** `VectorIndex.search` 每次都调
`self._embedder.embed([query])`，索引本身没有地方挂缓存。改 vector.py 会把"缓存"这件
与检索算法无关的事塞进检索算法里；而 Embedder 是公开 ABC
（`embed / embed_one / dimension / is_fake / name`），包一层就够——vector.py 一个字符不动。
`Embedder.embed` 是同步的，所以这里用 `threading.Lock` 而不是 asyncio 锁。

**文档向量为什么在后台线程里建：** 全量 150 条要 2.6 秒、5 次 API 调用。放在请求路径里
等于让第一个用户白等 3 秒；放在 lifespan 里又会让启动强依赖上游目录（目录拉不到就起不来）。
所以做成"第一次 /search 顺手起一个后台线程，本次请求先用两路返回"。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

from app.config import Settings
from app.retrieval.catalog import ProductDoc
from app.retrieval.embedding import Embedder, build_embedder
from app.retrieval.vector import VectorIndex

log = logging.getLogger(__name__)

# query 向量的缓存容量与存活时间。256 条足够盖住"一轮会话里反复出现的同几个问法"
# （前端联想、用户手滑重发、Java 侧重试都会重复同一条 query），又不会把进程内存拖大；
# TTL 取 5 分钟：配置/模型换了以后老向量会自然过期，不用手工清缓存。
QUERY_CACHE_MAX_ENTRIES = 256
QUERY_CACHE_TTL_SECONDS = 300.0

# 向量路的三种可观测状态（日志与测试都用这三个字面量，别各写一套）
STATE_READY = "ready"        # 本代目录的向量索引已就绪，走三路
STATE_BUILDING = "building"  # 后台正在建，本次及后续若干请求仍走两路
STATE_DEGRADED = "degraded"  # 本代目录建索引失败，已降级两路（等目录换代才重试）
STATE_IDLE = "idle"          # 还没请求过 / 空目录


class CachedEmbedder(Embedder):
    """带 LRU + TTL 的 Embedder 包装器：契约与内层实现逐字一致。

    缓存的是"文本 → 向量"这个纯函数结果，所以命中与未命中的下游行为完全相同：
    它不改变任何语义，只减少对上游的重复调用。
    """

    def __init__(
        self,
        inner: Embedder,
        *,
        max_entries: int = QUERY_CACHE_MAX_ENTRIES,
        ttl_seconds: float = QUERY_CACHE_TTL_SECONDS,
    ) -> None:
        self._inner = inner
        self._max_entries = max(1, int(max_entries))
        self._ttl_seconds = float(ttl_seconds)
        # OrderedDict 就是 LRU：命中时 move_to_end，超容量时弹最旧的一条
        self._entries: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        # 锁只护字典的增删与联动，**不包住内层 embed 调用**：否则一次 2.6 秒的文档
        # embed 会把期间的 query embed 全部堵在锁上，缓存就从"省时间"变成"加延迟"
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dimension(self) -> int:
        return int(self._inner.dimension)

    @property
    def is_fake(self) -> bool:
        # 降级状态必须**透传**：包一层之后调用方仍要能看出底层是不是哈希降级
        return bool(self._inner.is_fake)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        now = time.monotonic()
        results: list[list[float] | None] = [None] * len(texts)
        # 同一批里重复的文本只上去问一次：先记下位置，回来一次性铺开
        missing_positions: dict[str, list[int]] = {}
        with self._lock:
            for position, text in enumerate(texts):
                key = text or ""
                entry = self._entries.get(key)
                if entry is not None and (now - entry[0]) < self._ttl_seconds:
                    self._entries.move_to_end(key)
                    results[position] = entry[1]
                    continue
                if entry is not None:
                    # 过期的条目就地删掉：留着只会让下次命中一个过期值
                    del self._entries[key]
                missing_positions.setdefault(key, []).append(position)

        missing = list(missing_positions)
        if missing:
            fresh = self._inner.embed(missing)
            if len(fresh) != len(missing):
                raise ValueError(
                    f"embedder 返回条数与请求不符: 请求 {len(missing)} 条，拿到 {len(fresh)} 条"
                )
            with self._lock:
                for key, vector in zip(missing, fresh):
                    self._remember(key, list(vector), now)
                    for position in missing_positions[key]:
                        results[position] = vector

        # 到这里每个位置都必须有值；缺了说明上面的分支写漏了，宁可报错也不返回空向量
        if any(vector is None for vector in results):
            raise RuntimeError("query 向量缓存返回了未填充的位置")
        return [vector for vector in results if vector is not None]

    def _remember(self, key: str, vector: list[float], now: float) -> None:
        self._entries[key] = (now, vector)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


class VectorCache:
    """一份商品快照 → 一个 VectorIndex；换代判据是 docs list 的对象身份。

    `peek` 永远不阻塞：没建好就是 None，调用方据此走两路。
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        query_cache_entries: int = QUERY_CACHE_MAX_ENTRIES,
        query_cache_ttl: float = QUERY_CACHE_TTL_SECONDS,
        auto_build: bool = True,
    ) -> None:
        # 建库时的文档 embed 与每次 search 的 query embed 都走这一层：
        # 文档向量同理进同一份 LRU（不这么做就只有两个选择——碰 VectorIndex 的私有字段，
        # 或者在缓存外面另建一条绕过它的路径，两者都比"共用一份 LRU"更难解释）。
        # 代价是换代时那 150 条会挤掉一部分 query 缓存，而换代是低频事件（目录刷新才发生）。
        self._embedder = CachedEmbedder(
            embedder, max_entries=query_cache_entries, ttl_seconds=query_cache_ttl
        )
        self._auto_build = bool(auto_build)
        self._lock = threading.Lock()
        self._source: list[ProductDoc] | None = None
        self._index: VectorIndex | None = None
        # 这一代目录建索引失败过的标记：失败后不再每个请求都重试一次 2-3 秒的 embed，
        # 只有目录换代（换了新 list）才会重新尝试
        self._failed_source: list[ProductDoc] | None = None
        self._building_source: list[ProductDoc] | None = None
        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        # 上一次打过日志的状态：状态切换才打一行，不随请求刷屏（否则 WARNING 会被淹掉）
        self._last_logged_state: str | None = None

    # --- 只读观测点 ---------------------------------------------------

    @property
    def is_fake(self) -> bool:
        """True = 向量来自哈希降级，无语义能力。"""
        return bool(self._embedder.is_fake)

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    @property
    def last_error(self) -> str | None:
        """最近一次建索引失败的原因；None 表示这一代建成功了（或还没建过）。"""
        return self._last_error

    def state(self, docs: list[ProductDoc]) -> str:
        """本代目录当前处在哪个状态；用于日志与运维观测。"""
        with self._lock:
            if self._index is not None and self._source is docs:
                return STATE_READY
            if self._building_source is docs:
                return STATE_BUILDING
            if self._failed_source is docs:
                return STATE_DEGRADED
            return STATE_IDLE

    # --- 请求路径用的两个入口 -----------------------------------------

    def peek(self, docs: list[ProductDoc]) -> VectorIndex | None:
        """当前快照对应的向量索引；没有、或不是同一份快照就返回 None（绝不阻塞）。"""
        with self._lock:
            if self._index is not None and self._source is docs:
                return self._index
            return None

    def request_build(self, docs: list[ProductDoc]) -> bool:
        """后台起一次构建；返回是否真的起了线程（False = 已就绪 / 正在建 / 本代失败过）。"""
        with self._lock:
            if self._index is not None and self._source is docs:
                return False
            if self._failed_source is docs or self._building_source is docs:
                return False
            if not self._auto_build:
                return False
            self._building_source = docs
        thread = threading.Thread(
            target=self.build_now,
            args=(docs,),
            name="vector-index-build",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        return True

    def build_now(self, docs: list[ProductDoc]) -> VectorIndex | None:
        """同步建索引（测试与预热用）；失败返回 None 并记 WARNING，绝不把错抛给调用方。"""
        with self._lock:
            if self._index is not None and self._source is docs:
                return self._index
        try:
            index = VectorIndex(docs, self._embedder)
        except Exception as exc:  # noqa: BLE001 —— 见下面的降级理由
            # 这里**刻意**接住所有异常（不只 UpstreamError）：失败原因可能是上游 5xx/超时，
            # 也可能是 embedder 违约（返回条数/维度不符）——两者只有一个共同后果：
            # "这一代目录的向量路不可用"。把它抛给请求路径会让 /search 变成 500，
            # 而两路结果本来是正确且可用的；exc_info=True 保证原因不会被吞掉。
            with self._lock:
                self._failed_source = docs
                self._building_source = None
                self._last_error = f"{type(exc).__name__}: {exc}"
            log.warning(
                "向量第三路建索引失败，本代目录降级为两路 BM25（n_docs=%d is_fake=%s）: "
                "%s: %s",
                len(docs),
                self.is_fake,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None

        with self._lock:
            self._index = index
            self._source = docs
            self._building_source = None
            self._failed_source = None
            self._last_error = None
        log.info(
            "向量索引已建立 n_docs=%d embedder=%s is_fake=%s",
            index.n_docs,
            index.embedder_name,
            index.is_fake,
        )
        return index

    def log_transition(self, state: str) -> bool:
        """向量路状态**变化时**打一行日志；返回是否变化过。

        为什么除了"降级的那一次 WARNING"还要这一行：运维要回答的不是"某次请求失败了"，
        而是"这台机器现在是不是一直在降级"。每个请求都打一行 WARNING 会把日志淹掉
        （反而更难发现），只在**切换点**打一行，就从日志里能数出"今天退化了几次"。
        """
        with self._lock:
            if state == self._last_logged_state:
                return False
            previous = self._last_logged_state or STATE_IDLE
            self._last_logged_state = state
            last_error = self._last_error
        if state == STATE_DEGRADED:
            log.warning(
                "向量第三路状态切换 %s → %s（本代目录已降级为两路 BM25）is_fake=%s 原因=%s",
                previous,
                state,
                self.is_fake,
                last_error,
            )
        else:
            log.info("向量第三路状态切换 %s → %s is_fake=%s", previous, state, self.is_fake)
        return True

    def wait_settled(self, timeout: float = 30.0) -> None:
        """等后台构建线程结束（测试用来钉死时序；生产路径不需要调它）。"""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)


def build_vector_cache(settings: Settings, *, auto_build: bool = True) -> VectorCache:
    """按配置造一份向量缓存；**哈希降级必须在启动日志里喊出来**。

    没有 EMBEDDING_API_KEY 时向量路仍会跑（确定性哈希向量，只有字面重叠能力），
    于是线上它会静默地给排序提供一份没有语义能力的票。"看起来一切正常"的退化必须
    至少有一行 WARNING，否则运维永远不知道线上跑的是降级实现。
    """
    embedder = build_embedder(settings)
    if embedder.is_fake:
        log.warning(
            "向量第三路走**哈希降级**（未配置 EMBEDDING_API_KEY，is_fake=True）："
            "这一路只有字面重叠能力、没有语义能力，排序质量不等同于真实 embedding"
        )
    return VectorCache(embedder, auto_build=auto_build)
