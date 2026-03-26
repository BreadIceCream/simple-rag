# Simple RAG（检索增强生成）系统

[English Version](./README.md) | [前端项目](https://github.com/BreadIceCream/simple-rag-frontend)

Simple RAG 是一个基于 **FastAPI**、**LangChain** 和 **LangGraph** 的 **Agentic RAG 后端系统**。项目覆盖了多格式文档接入、结构化切块、混合检索、重排、反思式回答生成、SSE 流式输出，以及一套基于 **真实 RAG 执行** 的 **RAGAS 评测链路**。

项目的设计原则很明确：在线问答和离线评测都应尽量复用真实检索与真实 Graph 主链路，而不是依赖 prompt-only 拼装或单纯的合成自评。因此，系统重点放在真实检索质量、可控的 Graph 行为和基于实际运行结果的评测闭环上。

更详细的设计说明请查看 `docs/` 目录。

## 核心特征

- **Agentic LangGraph 工作流**：将检索、直接回答、问题重写、幻觉检查、有效性检查、历史摘要组织成持久化状态机。
- **混合检索**：结合向量检索与 BM25 类稀疏检索，并通过 RRF 融合提升召回稳定性。
- **父子块分层检索**：父块保留语义完整性，子块提高召回粒度，兼顾回答质量与检索效率。
- **结构感知文档接入**：Markdown、HTML、代码、PDF、Office 文档、网页 URL 采用不同加载和切块策略，而不是统一粗暴处理。
- **可选重排层**：支持 Qwen Reranker 对最终父块候选进行二次排序。
- **会话可恢复**：基于 PostgreSQL checkpointer 保存 LangGraph 状态，可在异常中断后恢复对话。
- **SSE 实时流式输出**：支持 token 流、Graph 进度、最终答案与参考文档同步推送。
- **真实 RAG 评测闭环**：支持数据集构建、真实执行、RAGAS 评分与检索指标汇总。

## 核心组成

### 在线服务主链路

1. `app/main.py` 负责初始化配置、数据库、Embedding、向量库、文档存储、加载器、切块器、检索器、重排器、Elasticsearch 和 LangGraph。
2. `app/core/document_loader.py` 负责将本地文件和 URL 统一加载为 `Document`。
3. `app/core/chunking.py` 负责结构化父块切分与子块切分。
4. `app/core/retriever.py` 负责向量检索、ES 稀疏检索、父块回溯和混合融合。
5. `app/core/reranker.py` 负责候选文档重排。
6. `app/core/graph.py` 定义完整的回答生成、检索调用与自检回路。
7. `app/routers/conversation.py` 在 Graph 之上暴露 SSE 对话接口。

### 离线评测主链路

1. `app/evals/build_replay_dataset.py`、`build_synthetic_dataset.py`、`import_seed_dataset.py` 负责从不同来源构建数据集。
2. `app/evals/live_rag_runner.py` 负责对数据集逐条执行真实 RAG 系统。
3. `app/evals/ragas_scorer.py` 负责执行 RAGAS 与检索指标评分。
4. `app/evals/ragas_runner.py` 提供一键串联入口。
5. 所有评测产物统一落在 `store/evals/datasets/` 和 `store/evals/experiments/`。

## 技术亮点

- **Graph 式纠偏闭环**：系统不是“检索一次然后回答一次”，而是可以在检索不相关、回答不可靠、回答无用时自动重写问题和重新生成。
- **Parent Document Retriever 设计**：先检索细粒度子块，再回溯父块作为回答上下文，降低碎片化上下文带来的回答断裂。
- **结构保真的切块策略**：Markdown 标题、HTML 标题和代码语言边界会被尽量保留，再做长度级递归切块。
- **范围受控检索**：检索可以绑定到指定文件集合，这一能力同时服务于检索调试接口和对话 Graph。
- **可持久化对话状态**：Graph 状态通过 PostgreSQL 持久化，便于恢复、调试和追踪。
- **评测链路彻底解耦**：数据集构建与真实执行、评分分离，`replay`、`synthetic`、`seed` 可以共用同一条测评主链路。
- **synthetic 生成稳态增强**：对重文件引入动态 batch 控制、低并发 RAGAS 执行、重试退避和自适应批次拆分，降低生成阶段连接异常概率。

## 系统结构

### 在线 RAG 主链路

1. 从 `config.yml` 和环境变量加载配置。
2. 初始化数据库、Embedding、向量库和父文档存储。
3. 初始化加载器、切块器、检索器、重排器和 LangGraph。
4. 对外提供文档接入、检索测试和对话接口。

### 离线评测主链路

1. 构建或导入评测数据集。
2. 按需导出审核表并进行人工审核。
3. 对数据集样本执行真实 RAG 系统。
4. 使用 RAGAS 与检索指标对真实结果评分。
5. 查看 `summary.json`、`report.md` 等评测产物。

## 目录结构

```text
RAG/
├── app/
│   ├── main.py
│   ├── config/
│   ├── core/
│   ├── crud/
│   ├── evals/
│   ├── exception/
│   ├── models/
│   └── routers/
├── docs/
│   ├── 项目说明文档.md
│   ├── RAGAS集成方案.md
│   └── Evals数据集审核说明.md
├── store/
│   ├── chroma_langchain_db/
│   ├── parent_docs/
│   └── evals/
│       ├── datasets/
│       └── experiments/
├── test_docs/
├── v1/
├── config.yml
├── docker-compose.yml
├── Dockerfile
├── README.md
└── README-zh.md
```

### `app/core` 目录说明

- `document_loader.py`：多格式文档与 URL 加载，统一 metadata。
- `chunking.py`：父块和子块切分策略注册与分发。
- `embeddings.py`：Embedding 后端初始化与切换。
- `vector_store.py`：Chroma 向量库与本地父文档存储管理。
- `retriever.py`：ES 稀疏检索、父文档检索、混合融合与检索范围控制。
- `reranker.py`：候选文档重排。
- `graph.py`：在线问答的 LangGraph 状态机。

### `app/evals` 目录说明

- `build_replay_dataset.py`
- `build_synthetic_dataset.py`
- `import_seed_dataset.py`
- `dataset_builder.py`
- `live_rag_runner.py`
- `ragas_runner.py`
- `ragas_scorer.py`
- `retrieval_scorer.py`
- `metrics_registry.py`
- `reporter.py`
- `runtime.py`
- `schema.py`

每个文件的详细职责与命令说明见：[docs/RAGAS集成方案.md](./docs/RAGAS集成方案.md)

## 技术栈

- **语言**：Python 3.12+
- **后端**：FastAPI
- **RAG 框架**：LangChain、LangGraph
- **数据库**：PostgreSQL、SQLAlchemy (async)、psycopg
- **稀疏检索**：Elasticsearch
- **向量库**：ChromaDB
- **模型能力**：HuggingFace 或 OpenAI 兼容后端
- **重排**：Qwen Reranker
- **评测**：RAGAS

## 安装与运行

### 环境要求

- Python 3.12+
- PostgreSQL
- Elasticsearch
- PyTorch
- `requirements.txt` 中的依赖

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
uvicorn app.main:app --reload
```

### 主要 API 路径

- `/api/documents`：文档导入与管理
- `/api/retrieval`：检索测试与参考范围绑定
- `/api/conversation`：SSE 对话接口

## 配置说明

当前项目通过 `config.yml` 控制主要运行参数。

### `config.yml` 关键配置

```yaml
env_override: false

database:
  url: postgresql+asyncpg://postgres:pg123456@localhost:5432/simple_rag

elasticsearch:
  url: https://localhost:9200
  username: elastic

chat_model:
  default: gpt-4o-mini
  light: gpt-4o-mini

embedding:
  model: Qwen/Qwen3-Embedding-0.6B
  openai:
    enabled: false
  huggingface_remote_inference:
    enabled: false

chunking:
  parent:
    chunk_size: 1000
    chunk_overlap: 120
  child:
    chunk_size: 256
    chunk_overlap: 50

vector_store:
  collection_name: default

retriever:
  final_k: 8
  reranker:
    enabled: true

chat:
  max_rewrite_time: 2
  max_generate_time: 3
  conversation_summarize_threshold: 10

text_file_length_threshold: 1500

debug:
  enabled: true
  docling_front: true
  trafilatura_front: true
  graph_visualization: false
```

### 配置重点

- `database`：数据库与 checkpoint 持久化基础。
- `elasticsearch`：稀疏检索后端配置。
- `chat_model`：默认回答模型与轻量控制模型。
- `embedding`：Embedding 模型来源。
- `chunking`：父块与子块大小及重叠配置。
- `retriever`：最终返回数量与是否启用重排。
- `chat`：问题重写、回答重试与历史摘要阈值。
- `debug`：文档加载和 graph 调试开关。

## RAGAS 评测

当前项目已经集成了基于 **真实执行** 的 RAGAS 评测链路：

1. 先构建或导入评测数据集。
2. 再调用真实 RAG 系统执行。
3. 最后对真实运行结果做 RAGAS 与检索指标评分。

### 支持的数据集类型

- `replay`：从历史会话构建
- `synthetic`：从知识库父块生成
- `seed`：从人工或外部样本导入

### 最简单的 smoke test

```bash
python -m app.evals.build_synthetic_dataset --name synthetic_smoke --version v1 --category exploration --size 20 --doc-limit 10 --use-light-model
python -m app.evals.ragas_runner --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1 --limit 10 --review-status pending,approved
```

### 可选的人审流程

```bash
python -m app.evals.dataset_builder export-review --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1
python -m app.evals.dataset_builder apply-review --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1 --review-file store/evals/datasets/exploration/synthetic_smoke/v1/review_sheet.csv
```

### 两步式执行流程

```bash
python -m app.evals.live_rag_runner --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1 --review-status pending,approved
python -m app.evals.ragas_scorer --run-dir <run_dir>
```

### 一键串联入口

```bash
python -m app.evals.ragas_runner --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1 --review-status pending,approved
```

### 评测产物位置

- 数据集目录：`store/evals/datasets/...`
- 评测运行目录：`store/evals/experiments/...`
- 常见产物：`manifest.json`、`samples.jsonl`、`review_sheet.csv`、`records.jsonl`、`summary.json`、`report.md`

完整方案、字段设计和脚本说明见：[docs/RAGAS集成方案.md](./docs/RAGAS集成方案.md)

## 常见问题

1. **数据库连接失败**：检查 `config.yml` 中的 PostgreSQL 连接字符串。
2. **Elasticsearch 不可用**：检查 ES 地址、认证信息和本地证书配置。
3. **Embedding 或模型加载失败**：检查本地模型依赖和环境变量配置。
4. **评测时出现连接错误**：可降低 synthetic 生成并发，或直接使用 `build_synthetic_dataset` 内置的低并发配置。
5. **Graph 内循环过多**：调整 `chat.max_rewrite_time` 与 `chat.max_generate_time`。

## 贡献

欢迎提交 Issue 和 Pull Request。
