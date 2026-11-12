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

## 检索（两路 BM25 + RRF 融合）

`/search` 走 `BM25Index.search_fused`：

- **两路召回**：原文 token 的 BM25 + 泛化词（同义词/上下位/错别字纠正，见
  `app/retrieval/synonyms.py`）的同源 BM25，RRF（k=60）融合；
- **名次深度固定 10**：每路只交前 10 名，所以 `top_k` 只截断长度、不参与排序，
  top-5 恒等于 top-10 的前 5 个；
- 融合池不足 `top_k` 时，按第一路完整 BM25 排序补位，补位项融合分记 0.0。

离线评估口径：150 条固定商品（`app/eval/fixture_catalog.py`）/ 100 条 query
（exact 40 / partial 30 / paraphrase 30），指标 Recall@10 / NDCG@10 / MRR@10。

实测（同一份语料，fused vs bm25）：para-semantic（13 条真语义改写）
**R 0.727 → 0.804**；exact 组 R 保持 **1.000**（N 0.997），partial 组 R 0.959 不变。

复现：
```
.venv\Scripts\python.exe -m app.eval.run_eval --mode fused
.venv\Scripts\python.exe -m app.eval.run_eval --mode bm25
```

**限制（如实）**：词表只有 22 个键（同义 12 / 上位 4 / 错别字 6，共展开 23 个词），
只在 150 条自建语料上评估过，**不是线上流量**；没有向量召回，同义改写全靠词表，
覆盖不到词表之外的任何说法。

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

- 本机无 `SILICONFLOW_API_KEY`，真实模型只在 fake 模式下跑过；带工具的**真实**上游
  调用（Qwen 返回 `tool_calls` 的那种）没有在真 key 下验证过。
- 本机 Docker 未运行、无 MySQL，商品目录索引只有 mock 级证据；真实
  `trademall /api/agent/products` 分页拉取未做过端到端。
- 检索质量只在自建 150 条语料上测过，不是线上流量；没有向量召回。
