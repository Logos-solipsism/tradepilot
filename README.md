# tradepilot —— 电商 AI 导购与检索服务

![CI](https://github.com/Logos-solipsism/tradepilot/actions/workflows/ci.yml/badge.svg) ![Python](https://img.shields.io/badge/Python-3.12-3776AB) ![FastAPI](https://img.shields.io/badge/FastAPI-0.162-009688) ![License](https://img.shields.io/badge/license-MIT-green)

链路：浏览器 → Java 后端 (Spring Boot, :18090) → **本服务 (FastAPI, :8010)** → 大模型。
Java 不直接调大模型，所有 LLM 调用都发生在这里。`/chat` 与 `/search` 的调用方永远是
Java，不是浏览器；两个接口都必须带 Java 签发的内部票据。
Java 侧仓库：[Logos-solipsism/trademall](https://github.com/Logos-solipsism/trademall)（本服务只被它调用，不直接面向浏览器）。

## 已实现
- 内部票据校验（Java → Python 的唯一信任凭据，fail-closed）
- `POST /chat`：带工具循环的导购对话（搜商品 / 查详情），会话落 memory 或 Redis
- `POST /search`：两路 BM25 + RRF 融合的商品检索（含同义词 / 上下位 / 错别字泛化）
- `GET /health`：探活，如实报告 llm 真假、会话后端、索引是否就绪
- 离线 IR 评估集（150 条商品 / 100 条 query）与 pytest 离线单测

不做（有意）：多智能体 / A2A / 向量库 / MCP / SSE 流式 / 关系库持久化。
理由写在对应模块的 docstring 里，不是漏做。

## 本地启动
```
D:\Anaconda3\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-dev.txt
$env:INTERNAL_SECRET="test-secret"
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010
```

## HTTP 契约

### POST /chat
头：`X-Internal-Ticket`（必需）、`X-Trace-Id`（可选）
体（额外字段一律 422）：`{"message": "...", "session_id": null, "trace_id": null}`
**请求体里没有 user_id**，用户身份只能来自票据。

成功 200：
```json
{"code":0,"message":"ok","data":{"answer":"...","session_id":"<uuid4>","tool_calls":[{"name":"search_products","arguments":"{\"query\":\"跑步鞋\"}","ok":true}],"usage":{"input":12,"output":34,"total":46}}}
```

`tool_calls` 是本次回答实际发生的工具调用审计（`ok=false` 表示工具失败但对话继续）。

票据问题 → 真实 **401** `{"code":401,"message":"internal ticket 无效或已过期","data":null}`
上游模型问题 → 真实 **502** `{"code":502,"message":"上游模型暂时不可用","data":null}`

### POST /search
头：`X-Internal-Ticket`（必需）
体（额外字段一律 422）：`{"query": "...", "top_k": 10, "trace_id": null}`
`top_k` 取值 1–50，默认 10；`query` 原样回显，不 trim、不改写。

`hits[].score` 是**两路 RRF 融合分**（量级 ~0.03），只用于排序与展示：
它不是相关度百分比，也**不可跨版本比较**。`score == 0.0` 表示该条只是为凑满
`top_k` 补位的尾部结果，不是异常。详见下一节。

### GET /health（免票据）
`{"ok":true,"llm":"real|fake","version":"0.1.0","sessions":"memory|redis"}`

## 导购 Agent 的工具循环

`/chat` 走 `app/agent/loop.py`：`system` 提示词 + 历史 + 本轮问题 → 模型。
模型要求调工具时，执行后把结果以 `role=tool` 回灌，最多 **3 轮**（一次 `/chat` 上界
1+3=4 次模型调用，成本可算）。用满轮次仍未收敛时，再发一次只收文字的收口请求，
即使又给 `tool_calls` 也忽略——否则「上限」形同虚设。

工具只有两个，都只读（`app/tools/catalog_tools.py`）：

| 工具 | 参数 | 打到的上游 |
| --- | --- | --- |
| `search_products` | `query`、`top_k`（默认 8，上限 20） | 本服务的 `search_fused` |
| `get_product_detail` | `product_id` | Java `GET /api/agent/products/{id}` |

工具失败（上游超时 / 非 2xx / 结构不符）一律返回一条中文 error JSON、`ok=false`，
对话**继续**：模型看到的是「商品服务暂时不可用」，不是一个 traceback。
原因只进日志，不进返回给模型的文本。单条工具结果超 2000 字会被截断。

会话只落 `user` + `assistant` 两条，工具往返不进历史（否则每轮上下文线性膨胀）。
fake 模式（无 `LLM_API_KEY`）单次调用且不传 tools。

## 检索（三路 RRF：原文 BM25 + 泛化词 BM25 + 向量召回）

**`POST /search` 现在走三路** `BM25Index.search_fused_three`（此前只有两路
`search_fused`；三路此前只接在离线评估 `--mode fused3` 上）。

- **第一路**：原文 token 的 BM25；
- **第二路**：泛化词（同义词/上下位/错别字纠正，见 `app/retrieval/synonyms.py`）的
  同源 BM25；
- **第三路**：向量余弦召回（见下一节）；
- 三路 RRF（k=60）**同权**融合，`hits[].score` 仍是融合分量级（~0.03），
  **不是**余弦分、不是相关度百分比；
- **名次深度固定 10**：每路只交前 10 名，所以 `top_k` 只截断长度、不参与排序，
  top-5 恒等于 top-10 的前 5 个；
- 融合池不足 `top_k` 时，按第一路完整 BM25 排序补位，补位项融合分记 0.0。

### 第三路：向量召回

`BM25Index.search_fused_three` 在同一套 RRF 循环里再加一路**向量余弦**名次：

- 索引是 faiss `IndexFlatIP`（`app/retrieval/vector.py`）：向量先 L2 归一化，
  内积即余弦。**不用 chromadb**——它在本机（Windows + 它的 Rust binding）任何路径
  都起不来（`unable to open database file`），而这里只需要"归一化 + 内积 top-k"；
- Embedder 抽象在 `app/retrieval/embedding.py`：有 `EMBEDDING_API_KEY` 走
  OpenAI 兼容的 `/embeddings`（默认 `BAAI/bge-m3`，1024 维），没有就走
  **确定性哈希向量**降级——纯标准库 blake2b 的有符号 hashing trick，
  不用内置 `hash()`（每进程加盐，跨进程对不上）。降级不是假装：`is_fake=True`，
  报告头会写「哈希降级，无语义能力」；
- 三路**同权**融合（每路各 1/(k+rank)），名次深度同样是 10，补位规则与两路逐字一致。
  RRF 只按名次投票，所以向量路只能给候选**加票**、不能给已有命中**减票**——
  这正是 exact 组 R 得以守住 1.000 的原因；
- 额外入口只新增签名：`search` / `search_fused` 一个字符都没改。

#### 两层缓存（`app/retrieval/vector_cache.py`）

线上 `/search` 每来一次就重建索引是不可接受的，所以两层缓存都用**同一套判据**：
`CatalogIndex.docs()` 交出的 **list 对象身份**。这与 `_Bm25Cache` 的换代判据逐字一致，
而一致是硬要求——`search_fused_three` 有文档集一致性守卫，BM25Index 与 VectorIndex
必须由**同一个 docs list** 构建，判据分叉就会抛 `ValueError`。

| 层 | 存在哪里 | 失效判据 | 代价 |
| --- | --- | --- | --- |
| 文档向量（faiss 索引） | `VectorCache._index`（进程内） | docs list 身份变化 / TTL 到期的目录刷新 | 全量 embed 一次（150 条 ≈ 2.6 秒、5 次 API 调用） |
| query 向量 | `CachedEmbedder` 的有界 LRU（256 条，TTL 300 秒） | 容量淘汰（LRU）/ TTL 过期 | 一次性 miss 才调上游 |

query 向量缓存**没有改 `vector.py`**：`VectorIndex.search` 每次都调
`self._embedder.embed([query])`，所以这里给 `VectorIndex` 传一个实现了同样契约
（`embed / embed_one / dimension / is_fake / name`）的包装器 `CachedEmbedder`。
文档向量也走同一层（换代时那 150 条会挤掉一部分 query 缓存，换代是低频事件）。

#### 首次请求不等 3 秒

文档向量**不在请求路径里同步建**，也不在 lifespan 里抢跑（那会让启动依赖上游目录）。
做法是懒加载 + 后台线程：第一次 `/search` 发现没有本代索引 → 起一个 daemon 线程去建 →
**本次请求照常返回两路结果**；建好之后同一份目录的后续请求自动走三路。
lifespan 只做"造缓存对象 + 打降级 WARNING"，不碰网络。

#### embedding 失败时的降级策略（明确）

| 失败点 | 行为 | 后果 |
| --- | --- | --- |
| 文档向量建不起来（上游 5xx/超时/条数不符） | 该代目录**降级为两路**，WARNING 一行（带 traceback），状态记 `degraded`，**本代不再重试**；目录换代才重试 | `/search` 仍是 200 + 两路的正确结果 |
| 单次 query 的 embedding 失败（`UpstreamError`） | **该请求**降级两路，WARNING 一行；文档索引保持 `ready` | 该请求仍是 200 + 两路的正确结果 |
| 目录拉取失败（`CatalogIndex.docs()` 抛 `UpstreamError`） | **不降级**，仍是真实 HTTP 502 | 与向量无关：那是对调用方说假话 |

**降级不是空结果**：`indexed` 仍是真实条数，`hits` 仍是两路的真实命中，
绝不把 embed 失败伪装成"没有商品"。

为什么选"降级两路"而不是"宁可 503 也要严格"：
- 两路结果**本身就是正确且可用**的（它是接入三路之前线上一直在跑的那条路），
  向量路只改善排序名次，不改变"哪些商品是对的"；
- 向量路的价值集中在**排序质量**（实测 MRR@10 0.939 → 0.952），不是召回数量
  （Recall@10 0.946 → 0.945，持平）。为一份"排序增强"把一次正常查询变成 503，
  等于用一个可用但排名略差的结果换一个不可用的结果；
- 反过来，静默降级也不行：所以每一类降级都有 WARNING，且**状态切换只在切换点
  打一行**（`log_transition`），运维据此能数出"今天退化了几次"，而不是被刷屏淹没。

代价（如实）：降级期间检索**排序质量回到两路水平**，而且**调用方完全看不出来**——
响应体没有任何字段标注本次是两路还是三路。这是一次有意的取舍：Java 侧按五个字段逐字
解析 `hits`，加一个 `route` 字段会动契约；代价是"这一轮退化了"只能从服务端日志发现。

离线评估口径：150 条固定商品（`app/eval/fixture_catalog.py`）/ 100 条 query
（exact 40 / partial 30 / paraphrase 30），指标 Recall@10 / NDCG@10 / MRR@10。

实测（同一份语料，fused vs bm25）：para-semantic（13 条真语义改写）
**R 0.727 → 0.804**；exact 组 R 保持 **1.000**（N 0.997），partial 组 R 0.959 不变。

复现：
```
.venv\Scripts\python.exe -m app.eval.run_eval --mode bm25
.venv\Scripts\python.exe -m app.eval.run_eval --mode fused
.venv\Scripts\python.exe -m app.eval.run_eval --mode fused3
```

三路（`fused3`）在同一份语料、**真实 `BAAI/bge-m3` embedding** 下的实测：

| 指标 | 两路 `fused` | 三路 `fused3` | 差值 | 跨次稳定性 |
| --- | --- | --- | --- | --- |
| overall Recall@10 | 0.946 | 0.945 | **-0.001** | 三次跑完全一致 |
| overall NDCG@10 | 0.931 | 0.936 ~ 0.937 | +0.005 ~ +0.006 | 波动 ±0.001 |
| overall MRR@10 | 0.939 | **0.952 ~ 0.953** | **+0.013 ~ +0.014** | 波动 ±0.001 |
| paraphrase MRR@10 | 0.814 | **0.873 ~ 0.878** | **+0.059 ~ +0.064** | 波动 ±0.005 |
| para-semantic（13 条真语义改写）Recall@10 | 0.804 | 0.791 | **-0.013** | 三次跑完全一致 |
| exact 组 Recall@10 | 1.000 | 1.000 | 0 | 三次跑完全一致 |

> **为什么 MRR / NDCG 给区间而 Recall 给单值**：上游 `/v1/embeddings` 每次返回的向量
> **不是逐位相同**的——同一段文本连调三次，最大绝对误差约 `1.2e-3 ~ 2.3e-3`，
> 余弦相似度 0.99996。这点抖动足以让个别名次互换，于是 NDCG/MRR 会动 ±0.001 ~ 0.005；
> 而 Recall 只看"有没有进前 10"，名次互换不影响它，所以三次完全一致。
> 这也是本仓库所有向量数字都标注区间的理由：给一个精确到小数点后三位的单值，
> 等于把上游的浮点噪声当成结论。

**结论：向量路改善的是排序名次，不是召回数量。** MRR@10 涨了 0.013（语义改写组涨 0.059），
说明正确的商品被排得更靠前；而 Recall@10 基本持平、para-semantic 甚至略降 0.013，
说明词面两路本来就已经把相关商品召回了，「还能再发现多少」这条路已经走到头——
10 个位置装不下 24 条期望命中时，任何一路都救不了。

**哈希降级的对照（无 key 时）**：para-semantic R 0.772 / N 0.706 / M 0.723，低于两路。
哈希向量没有语义能力，那一行数字只证明三路管线通了，不代表任何语义收益——两行必须分开看，
不能混为一谈。

**限制（如实）**：词表只有 22 个键（同义 12 / 上位 4 / 错别字 6，共展开 23 个词），
只在 150 条自建语料上评估过，**不是线上流量**；向量路的数字来自真实 `BAAI/bge-m3`
（`/v1/embeddings`，1024 维），但只有这一份语料、这一种 embedding 模型，没有做过
多模型或大规模对照。

## 内部票据格式（与 Java `InternalTicketSigner` 逐字一致）
`{userId}.{ts}.{nonce}.{hexHmacSha256}`
签名原文 = `userId + "\n" + ts + "\n" + nonce`（末尾无换行），HMAC-SHA256，小写 hex。
ts 为 Unix 秒，容差 ±300s；`INTERNAL_SECRET` 为空时 fail-closed（全部 401）；
nonce 去重是**单进程内存**实现，多 worker 下不构成跨进程防重放。

## 配置（环境变量）
| 变量 | 默认 |
| --- | --- |
| INTERNAL_SECRET | 空（fail-closed） |
| LLM_BASE_URL | https://api.siliconflow.cn/v1 |
| LLM_API_KEY | 空 → fake 模式，仅本地开发占位 |
| LLM_MODEL | Qwen/Qwen2.5-7B-Instruct |
| LLM_TIMEOUT_SECONDS | 30 |
| SESSION_BACKEND | memory |
| SESSION_WINDOW | 20 |
| SESSION_TTL_DAYS | 7 |
| REDIS_URL | redis://127.0.0.1:6379/0 |
| AGENT_SERVICE_BASE_URL | http://127.0.0.1:18090 |
| AGENT_SERVICE_TOKEN | 空 → /search 直接 502（fail-closed，不返回空列表） |
| CATALOG_CACHE_TTL_SECONDS | 300 |

## 测试
```
.venv\Scripts\python.exe -m pytest -q
```

**176 passed**，全部离线：LLM 用 `httpx.MockTransport` 注入，商品目录用假 client 注入，
不发真实网络请求，不需要 MySQL / Redis / 大模型 key。

## 已知边界（如实）

- **生成模型**（`/chat` 的 `LLM_API_KEY`）仍只在 fake 模式下跑过；带工具的**真实**
  上游调用（返回 `tool_calls` 的那种）没有在真 key 下验证过。注意这与
  **embedding 模型**是两件事：后者的 key 已配置，`fused3` 的数字是真实调 API 得到的。
- 本机 Docker 未运行、无 MySQL，商品目录索引只有 mock 级证据；真实
  `trademall /api/agent/products` 分页拉取未做过端到端。
- 检索质量只在自建 150 条语料上测过，不是线上流量。
- 向量召回已在真实 `BAAI/bge-m3` 下评估过：**MRR@10 0.939 → 0.952~0.953、paraphrase
  MRR 0.814 → 0.873~0.878**，但 **Recall@10 持平（0.946 → 0.945）、para-semantic R
  略降（0.804 → 0.791）**。所以准确的说法是"改善排序、没改善召回"，任何"提升了
  召回率"的表述都与实测不符。
- **上游 embedding 不是确定性接口**：同一文本连调三次向量最大绝对误差 `~1.2e-3`，
  余弦 0.99996。因此向量路相关的 NDCG/MRR 每次跑会浮动 ±0.001~0.005，**不要用单次
  结果做小数点后三位的断言**；Recall 不受影响（名次互换不改变是否进前 10）。
- 哈希降级模式（无 `EMBEDDING_API_KEY`）也保留可跑：此时 `fused3` 的 para-semantic
  R 是 0.772，**低于**两路。这一行只证明管线通了，不能当作语义收益，两行不能混看。
- faiss 索引是进程内、**不持久化**的：进程重启后第一次 `/search` 会重新 embed
  一遍全量商品。150 条无感，量大时需要落盘 + 增量更新，那还没做。
- **降级对调用方不可见**：`/search` 的响应体没有 "本次走的是两路还是三路" 的字段
  （`hits` 只有五个字段，Java 侧逐字解析）。发现降级只有两条路：服务端日志
  （WARNING + 状态切换行）、以及自己比对两路/三路的排名差异。
- **哈希降级在生产里只会"看起来一切正常"**：没配 `EMBEDDING_API_KEY` 时向量路照跑，
  但它只有字面重叠能力。启动日志会有一行醒目 WARNING（`is_fake=True`），除此之外
  **没有别的线上自检手段**——`/health` 没有暴露 `is_fake`。
- **query 向量缓存是进程内的**：多 worker 部署时每个 worker 各有一份 256 条的 LRU，
  命中率按 worker 数打折；跨进程共享（Redis）那还没做。
- **文档向量和 query 向量共用同一份 256 条 LRU**：目录有 N 条商品时，稳态下留给 query
  的槽位只有 `256 - N`。N=150 时约 106 个 query 槽位够用；若线上目录涨到几百上千条，
  文档向量会开始把 query 缓存挤出去（正确性不受影响，只是命中率下降）——
  届时应把两层缓存拆成独立容量，那需要在 `VectorIndex` 上开一个旁路，还没做。
- **文档向量命中率依赖目录 TTL**：`CATALOG_CACHE_TTL_SECONDS`（默认 300 秒）到期后
  `CatalogIndex` 会交出一份**新的 list**，向量索引随之整份重建——TTL 设得太短会让
  "每 5 分钟重新 embed 一次全量商品"变成常态。
- `.env` 加载是**进程启动时一次性**的（`get_settings()` 带 lru_cache），
  改完 `.env` 必须重启进程才生效；优先级为 真实环境变量 > `.env` > 默认值。
