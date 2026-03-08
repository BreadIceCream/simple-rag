# Simple RAG (检索增强生成) 系统

[English Version](./README.md)      [前端项目](https://github.com/BreadIceCream/simple-rag-frontend)

基于 **LangChain**、**LangGraph** 和 **FastAPI** 开发的高级 **Agentic RAG** 系统后端服务，支持多模态文档检索。它具有智能反射状态机、混合检索策略（BM25 + 向量语义搜索 + RRF）、分层切分（Small-to-Big）和重排序（Re-ranking）功能，通过 SSE 流式传输提供高精度且无幻觉的对话体验。

更多详细信息，请查看 `docs` 目录。

## 功能特性

- 🧠 **Agentic 反射图工作流**：基于 LangGraph 构建，具有用于自我反思、幻觉检查、有用性评估和问题重写（思考-重写循环）的认知状态机。
- 🔄 **混合检索与 RRF 融合**：结合了 BM25 稀疏关键词检索和基于嵌入的稠密语义搜索，由互惠排名融合 (RRF) 统一。
- 🍱 **智能分层切分**：利用父文档检索（Parent Document Retriever）模式，配备语言/格式感知的切分器（Markdown/HTML 标题、代码切分器）。
- 🎯 **先进的重排序优化**：无缝集成 Qwen Reranker 和其他轻量级压缩器，进行深层语义抽象重排序。
- 🌐 **多模态文档摄取**：为本地文件（PDF、HTML、Markdown、代码）和远程 Web URL 提供级联解析器。
- ⚡ **高性能后端**：基于 FastAPI 的异步架构，由异步 SQLAlchemy 和 psycopg 连接池支持。
- 📡 **实时 SSE 流式传输**：提供逐令牌的流式响应，具有智能错误恢复和 LangGraph Checkpoint 状态持久化。

## 系统架构

### 核心组件

1. **FastAPI & 路由层**
   - `/api/documents`: 文档摄取和管理。
   - `/api/retrieval`: 检索测试和参考文献绑定。
   - `/api/conversation`: 具有 SSE 流式传输和历史记录管理的聊天接口。
2. **文档处理与切分**
   - 用于加载器（PDF、HTML、文本、WebBase）的责任链模式。
   - 用于动态切分策略（Markdown/HTML/代码/通用）的注册表模式。
3. **检索与向量库引擎**
   - `EnhancedParentDocumentRetriever` + `HybridPDRetriever` 提供稳健的搜索。
   - ChromaDB（向量库）和 LocalFileStore（键值库）用于级联索引。
4. **反射式 LangGraph Agent**
   - 通过 PostgresSaver 持久化状态（Checkpoints）。
   - 包含以下节点：`retrieve`（检索）、`grade_documents`（评估文档）、`generate_answer`（生成答案）、`check_hallucination`（幻觉检查）、`check_usefulness`（有用性检查）、`rewrite_question`（问题重写）和 `summarize_conversation`（总结对话）。

## 技术栈

- **核心框架**: Python 3.12+ | FastAPI | LangChain | LangGraph
- **数据库与 ORM**: PostgreSQL | SQLAlchemy (async) | psycopg (pool) | ChromaDB
- **AI/NLP**: HuggingFace(Qwen) / OpenAI Embeddings | Qwen Reranker | NLTK + jieba

## 安装与配置

### 环境要求

- Python 3.12+。
- PostgreSQL 数据库实例。
- PyTorch (可选 CUDA 支持)。
- 以下软件包依赖。

### 依赖安装

见 `requirements.txt`
```bash
pip install -r requirements.txt
```

### 环境变量配置

复制并配置 `.env-backup` 文件为 `.env`，或调整 `config.yml` 内的属性：

```yaml
# config.yml 亮点：
database:
  url: postgresql+asyncpg://user:password@localhost:5432/simple_rag

chat_model:
  default: gpt-5-mini           # 响应生成的主要模型
  light: gpt-4o-mini            # 用于评估/反射的轻量级 LLM

embedding:
  model: Qwen/Qwen3-Embedding-0.6B

retriever:
  final_k: 8
  reranker:
    enabled: true
```

## 使用说明

### 启动系统

```bash
uvicorn app.main:app --reload
```

### 执行流程

1. **初始化阶段（应用启动）**
   - 从 `config.yml` 和 `.env` 加载环境变量。
   - 初始化数据库连接池和向量库实例。
   - 初始化文档加载器链、文本切分器注册表和检索器。
   - 设置嵌入和重排序模型。
   - 编译 LangGraph StateGraph 引擎。
2. **交互阶段**
   - 通过 POST `/api/documents/local` 或 `/api/documents/url` **摄取文档**。
   - 使用 POST `/api/retrieval/references` **设置参考文献**。
   - 通过 POST `/api/conversation/chat` **通过 SSE 聊天**，接收令牌流和对话状态。

### 文档加载

支持多种文档加载方式：
- 调用 API 端点 `/api/documents/local` 上传并摄取本地文件类型。
- 系统会动态选择加载器（如 PDFLoader、HTMLLoader）并通过注册的文本切分器（如 MarkdownTextSplitter、CodeTextSplitter）对内容进行切分，最后自动将嵌入存储在 ChromaDB 中，将父级块存储在 LocalFileStore 中。

### 模式说明
系统通过 LLM 工具调用（tool-calling）智能地决定是利用检索工具还是直接返回响应。

## 技术细节

### 文本预处理与切分

- 上下文感知切分：MarkdownHeaderTextSplitter、HTMLHeaderTextSplitter、针对语言的 RecursiveCharacterTextSplitter，以获得最佳的嵌入表示。
- 中英文双语文本预处理，使用 `jieba` 和 `nltk` 停用词语义过滤进行 BM25 处理。

### 检索策略

1. **父文档检索（Parent Document Retriever）**：分离大文档（父级）和小语义块（子级）。
2. **混合搜索**：合并稀疏关键词（BM25）和稠密向量检索。
3. **RRF 融合**：对合并后的列表重新排名，减轻领域偏见。
4. **交叉编码器重排序（Cross-Encoder Re-ranking）**：利用 Qwen Reranker 进行最终精度调整。

### LangGraph Agent 状态机

Agent 决定是完成直接查询还是检索文档。如果检索到的文档不相关，LLM 会自动重写查询。生成的答案会经过幻觉和有用性的双重检查；失败会触发内部循环（生成 -> 检查 -> 重写），直到达到最大重试阈值。

## 文件结构

```
RAG/
├── app/                          
│   ├── main.py                   # FastAPI 应用入口
│   ├── config/                   # 全局和数据库配置
│   ├── core/                     # 核心引擎（加载器、切分、检索、图）
│   ├── crud/                     # PostgreSQL 操作
│   ├── routers/                  # API 端点（文档、检索、聊天）
│   ├── models/                   # 模式、VO、图状态
│   └── exception/                # 异常处理
│  
├── docs/ 						  # 参考文档
│
├── .env-backup                   # 环境变量备份
├── config.yml                    # 项目配置文件
├── Dockerfile                    # Docker 部署
├── README.md                     # 英文文档
└── README-zh.md                  # 中文文档
```

## 扩展开发

### 添加新工具与功能
1. 将新功能集成为独立的 FastAPI 路由。
2. 对于图扩展，在 `app/core/graph.py` 中添加新节点或修改条件边，并在 LangGraph 状态机内部实现相应的业务逻辑。

### 自定义加载器或切分器
继承 `DocumentLoader` 以支持新的 MIME 类型，并在 `DocumentLoaderChain` 中注册它们。
通过 `SplitterRegistry` 注册自定义文本切分逻辑。

## 故障排除

### 常见问题

1. **数据库连接失败**：确保 PostgreSQL 正在运行且凭据与 `config.yml` 匹配。
2. **CUDA 不可用**：应用会自动回退到 CPU 以使用原生 HuggingFace 嵌入。
3. **图循环超过最大限制**：在配置中降低 `max_rewrite_time` 和 `max_generate_time`。

## 更新日志

### v1.0

- 实现了基本 RAG 系统。
- 混合检索功能。
- 工具调用集成。
- 异步处理优化。

### v2.0 (当前版本)
- 全面重构为 FastAPI 后端服务。
- 实现了 LangGraph 自我反思 Agent 架构。
- 用混合 PDRetriever (BM25 + 向量 + RRF + PDRetrieve + Rerank) 取代了基本的 RAG 策略。
- 将状态迁移到异步持久化 (SQLAlchemy + PostgresSaver)。
- 用 SSE 流式 REST API 端点取代了 CLI 聊天。

## 贡献

欢迎贡献！请提交 Issue 和 Pull Request 以改进项目。
