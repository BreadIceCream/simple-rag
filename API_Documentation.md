# Personal Knowledge Base - API Documentation

> API Version: v1 | Base URL: `http://localhost:8000/api/v1`

---

## 全局约定

### 请求格式

- Content-Type: `application/json`
- 所有请求体均为 JSON 格式

### 响应格式

所有非流式响应遵循统一结构:

```json
{
  "code": 200,
  "message": "success",
  "data": { ... }
}
```

### 错误响应

```json
{
  "code": 400,
  "message": "error description",
  "data": null
}
```

常见错误码:

| HTTP 状态码 | 含义 |
|------------|------|
| `400` | 请求参数错误(如文件路径无效、格式不支持) |
| `404` | 资源不存在(文档/会话/集合未找到) |
| `409` | 资源冲突(如文档已存在) |
| `422` | 请求体校验失败 |
| `500` | 服务器内部错误 |

### 分页

列表类接口支持分页参数:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `page` | int | 1 | 页码(从 1 开始) |
| `page_size` | int | 20 | 每页数量(最大 100) |

分页响应:

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [...],
    "total": 42,
    "page": 1,
    "page_size": 20
  }
}
```

---

## 一、文档管理 `/documents`

### 1.1 上传并处理文档

处理流程: 校验文件路径 → DocumentLoader 加载 → SplitterChain 分块 → Embedding 向量化 → 存入 ChromaDB → 记录元数据。

```
POST /api/v1/documents
```

**请求体:**

```json
{
  "file_path": "/path/to/document.pdf",
  "collection_name": "default"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file_path` | string | 是 | 本地文件绝对路径，后端直接读取 |
| `collection_name` | string | 否 | 目标集合名称，默认 `"default"` |

**成功响应:** `201 Created`

```json
{
  "code": 201,
  "message": "Document processed successfully",
  "data": {
    "doc_id": "doc_a1b2c3d4",
    "file_name": "document.pdf",
    "file_path": "/path/to/document.pdf",
    "file_type": ".pdf",
    "collection_name": "default",
    "chunk_count": 42,
    "status": "completed",
    "created_at": "2026-02-28T22:30:00+08:00"
  }
}
```

**错误场景:**

| 错误码 | 场景 |
|--------|------|
| `400` | 文件路径不存在或文件格式不支持 |
| `409` | 该文件已在指定集合中存在(需先删除或使用更新接口) |

---

### 1.2 获取文档列表

```
GET /api/v1/documents
```

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `collection_name` | string | 否 | 按集合名称过滤 |
| `status` | string | 否 | 按状态过滤: `completed`, `processing`, `failed` |
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页数量 |

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [
      {
        "doc_id": "doc_a1b2c3d4",
        "file_name": "document.pdf",
        "file_path": "/path/to/document.pdf",
        "file_type": ".pdf",
        "collection_name": "default",
        "chunk_count": 42,
        "status": "completed",
        "created_at": "2026-02-28T22:30:00+08:00",
        "updated_at": "2026-02-28T22:30:05+08:00"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20
  }
}
```

---

### 1.3 获取文档详情

```
GET /api/v1/documents/{doc_id}
```

**路径参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `doc_id` | string | 文档唯一标识 |

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "doc_id": "doc_a1b2c3d4",
    "file_name": "document.pdf",
    "file_path": "/path/to/document.pdf",
    "file_type": ".pdf",
    "collection_name": "default",
    "chunk_count": 42,
    "chunk_ids": ["chunk_001", "chunk_002", "..."],
    "status": "completed",
    "metadata": {
      "file_size_bytes": 102400,
      "embedding_model": "BAAI/bge-m3",
      "splitter_used": "RecursiveCharacterTextSplitter",
      "chunk_size": 1000,
      "chunk_overlap": 200
    },
    "created_at": "2026-02-28T22:30:00+08:00",
    "updated_at": "2026-02-28T22:30:05+08:00"
  }
}
```

---

### 1.4 更新文档

删除旧版本的所有 chunks，重新执行加载 → 分块 → 嵌入流程。适用于源文件内容发生变更的场景。

```
PUT /api/v1/documents/{doc_id}
```

**路径参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `doc_id` | string | 文档唯一标识 |

**请求体(可选):**

```json
{
  "file_path": "/path/to/updated_document.pdf"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file_path` | string | 否 | 新的文件路径。不提供则使用原路径重新处理 |

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "Document updated successfully",
  "data": {
    "doc_id": "doc_a1b2c3d4",
    "file_name": "updated_document.pdf",
    "file_path": "/path/to/updated_document.pdf",
    "file_type": ".pdf",
    "collection_name": "default",
    "chunk_count": 38,
    "old_chunk_count": 42,
    "status": "completed",
    "updated_at": "2026-02-28T23:00:00+08:00"
  }
}
```

---

### 1.5 删除文档

删除文档及其在向量数据库中的所有 chunks。

```
DELETE /api/v1/documents/{doc_id}
```

**路径参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `doc_id` | string | 文档唯一标识 |

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "Document deleted successfully",
  "data": {
    "doc_id": "doc_a1b2c3d4",
    "chunks_deleted": 42
  }
}
```

---

## 二、会话管理 `/conversations`

### 2.1 创建会话

创建一个新的对话会话，指定知识来源(文档范围)和使用的 LLM 模型。

```
POST /api/v1/conversations
```

**请求体:**

```json
{
  "title": "关于 RAG 架构的讨论",
  "document_ids": ["doc_a1b2c3d4", "doc_e5f6g7h8"],
  "model_name": "qwen-plus",
  "max_history_turns": 20
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `title` | string | 否 | 会话标题，不提供则自动生成 |
| `document_ids` | list[string] | 是 | 选中的文档 ID 列表，检索将限定在这些文档内 |
| `model_name` | string | 否 | LLM 模型名称，默认使用 `.env` 中配置的模型 |
| `max_history_turns` | int | 否 | 最大记忆轮数，默认 20 |

**成功响应:** `201 Created`

```json
{
  "code": 201,
  "message": "Conversation created",
  "data": {
    "conversation_id": "conv_x1y2z3",
    "title": "关于 RAG 架构的讨论",
    "document_ids": ["doc_a1b2c3d4", "doc_e5f6g7h8"],
    "model_name": "qwen-plus",
    "max_history_turns": 20,
    "message_count": 0,
    "created_at": "2026-02-28T22:35:00+08:00"
  }
}
```

---

### 2.2 获取会话列表

```
GET /api/v1/conversations
```

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `page` | int | 否 | 页码 |
| `page_size` | int | 否 | 每页数量 |

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [
      {
        "conversation_id": "conv_x1y2z3",
        "title": "关于 RAG 架构的讨论",
        "document_ids": ["doc_a1b2c3d4", "doc_e5f6g7h8"],
        "model_name": "qwen-plus",
        "message_count": 5,
        "created_at": "2026-02-28T22:35:00+08:00",
        "last_active_at": "2026-02-28T23:10:00+08:00"
      }
    ],
    "total": 1,
    "page": 1,
    "page_size": 20
  }
}
```

---

### 2.3 获取会话详情(含消息历史)

```
GET /api/v1/conversations/{conversation_id}
```

**路径参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `conversation_id` | string | 会话唯一标识 |

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `include_messages` | bool | 否 | 是否包含消息历史，默认 `true` |
| `message_limit` | int | 否 | 返回的最大消息数，默认全部 |

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "conversation_id": "conv_x1y2z3",
    "title": "关于 RAG 架构的讨论",
    "document_ids": ["doc_a1b2c3d4", "doc_e5f6g7h8"],
    "model_name": "qwen-plus",
    "max_history_turns": 20,
    "message_count": 4,
    "created_at": "2026-02-28T22:35:00+08:00",
    "last_active_at": "2026-02-28T23:10:00+08:00",
    "messages": [
      {
        "message_id": "msg_001",
        "role": "user",
        "content": "RAG 系统的核心组件有哪些？",
        "created_at": "2026-02-28T22:36:00+08:00"
      },
      {
        "message_id": "msg_002",
        "role": "assistant",
        "content": "根据您的知识库文档，RAG 系统的核心组件包括...",
        "sources": [
          {
            "doc_id": "doc_a1b2c3d4",
            "file_name": "rag_architecture.pdf",
            "chunk_id": "chunk_015",
            "relevance_score": 0.92
          }
        ],
        "created_at": "2026-02-28T22:36:05+08:00"
      }
    ]
  }
}
```

---

### 2.4 更新会话配置

支持动态修改会话的文档范围和模型选择，无需新建会话。

```
PATCH /api/v1/conversations/{conversation_id}
```

**路径参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `conversation_id` | string | 会话唯一标识 |

**请求体(按需提供):**

```json
{
  "title": "更新后的标题",
  "document_ids": ["doc_a1b2c3d4", "doc_new_id"],
  "model_name": "gpt-4o",
  "max_history_turns": 30
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `title` | string | 否 | 更新会话标题 |
| `document_ids` | list[string] | 否 | 更新文档范围(全量替换) |
| `model_name` | string | 否 | 切换 LLM 模型 |
| `max_history_turns` | int | 否 | 更新最大记忆轮数 |

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "Conversation updated",
  "data": {
    "conversation_id": "conv_x1y2z3",
    "title": "更新后的标题",
    "document_ids": ["doc_a1b2c3d4", "doc_new_id"],
    "model_name": "gpt-4o",
    "max_history_turns": 30,
    "updated_at": "2026-02-28T23:15:00+08:00"
  }
}
```

---

### 2.5 删除会话

删除会话及其所有消息记录。

```
DELETE /api/v1/conversations/{conversation_id}
```

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "Conversation deleted",
  "data": {
    "conversation_id": "conv_x1y2z3",
    "messages_deleted": 4
  }
}
```

---

## 三、对话(Chat) `/chat`

### 3.1 发送消息(SSE 流式)

向指定会话发送消息，获取基于 RAG 的回答。响应使用 SSE 流式传输。

```
POST /api/v1/chat
```

**请求体:**

```json
{
  "conversation_id": "conv_x1y2z3",
  "message": "RAG 系统中检索模块的最佳实践是什么？"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `conversation_id` | string | 是 | 会话 ID(决定了文档范围、模型、历史记忆) |
| `message` | string | 是 | 用户消息内容 |

**SSE 响应流:**

Content-Type: `text/event-stream`

```
event: message_start
data: {"message_id": "msg_003", "conversation_id": "conv_x1y2z3"}

event: retrieval
data: {"sources": [{"doc_id": "doc_a1b2c3d4", "file_name": "rag_best_practices.pdf", "chunk_id": "chunk_023", "relevance_score": 0.95}]}

event: content_delta
data: {"delta": "根据"}

event: content_delta
data: {"delta": "您的知识库，"}

event: content_delta
data: {"delta": "RAG 系统检索模块的最佳实践包括..."}

event: message_end
data: {"message_id": "msg_003", "usage": {"prompt_tokens": 1200, "completion_tokens": 350, "total_tokens": 1550}}

event: done
data: [DONE]
```

**SSE 事件类型说明:**

| 事件 | 说明 |
|------|------|
| `message_start` | 消息开始，包含消息 ID |
| `retrieval` | 检索完成，返回引用的文档来源 |
| `content_delta` | 流式内容增量 |
| `message_end` | 消息结束，包含 token 使用统计 |
| `error` | 错误事件 |
| `done` | 流结束标志 |

**错误场景:**

| 错误码 | 场景 |
|--------|------|
| `404` | 会话不存在 |
| `400` | 消息内容为空 |

---

## 四、集合管理 `/collections`

### 4.1 获取集合列表

```
GET /api/v1/collections
```

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "items": [
      {
        "name": "default",
        "document_count": 5,
        "chunk_count": 210,
        "metadata": {
          "create_time": "2026-02-28 22:30 Friday",
          "embedding_model": "BAAI/bge-m3",
          "hnsw:space": "cosine"
        }
      }
    ],
    "total": 1
  }
}
```

---

### 4.2 创建集合

```
POST /api/v1/collections
```

**请求体:**

```json
{
  "name": "research_papers",
  "metadata": {
    "description": "学术论文集合"
  }
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | 是 | 集合名称(唯一) |
| `metadata` | object | 否 | 自定义元数据 |

**成功响应:** `201 Created`

```json
{
  "code": 201,
  "message": "Collection created",
  "data": {
    "name": "research_papers",
    "metadata": {
      "create_time": "2026-02-28 23:00 Friday",
      "embedding_model": "BAAI/bge-m3",
      "hnsw:space": "cosine",
      "description": "学术论文集合"
    }
  }
}
```

---

### 4.3 删除集合

删除集合及其中的所有文档和 chunks。

```
DELETE /api/v1/collections/{name}
```

**路径参数:**

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | string | 集合名称 |

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "Collection deleted",
  "data": {
    "name": "research_papers",
    "documents_deleted": 5,
    "chunks_deleted": 210
  }
}
```

---

## 五、系统信息 `/system`

### 5.1 健康检查

```
GET /api/v1/system/health
```

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "status": "healthy",
    "version": "2.0.0",
    "components": {
      "chromadb": "connected",
      "sqlite": "connected",
      "embedding_model": "loaded",
      "llm": "available"
    },
    "uptime_seconds": 3600
  }
}
```

---

### 5.2 获取支持的文件格式

```
GET /api/v1/system/supported-formats
```

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "document_formats": [".pdf", ".md", ".txt", ".docx", ".html", ".htm", ".csv"],
    "code_formats": [".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".cs", ".scala", ".swift", ".kt", ".php", ".lua", ".hs", ".c", ".cpp", ".h", ".latex"]
  }
}
```

---

### 5.3 获取可用模型列表

返回 `.env` 中配置的可用 LLM 模型列表，供前端模型选择器使用。

```
GET /api/v1/system/models
```

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "current_model": "qwen-plus",
    "available_models": [
      {
        "name": "qwen-plus",
        "provider": "dashscope"
      },
      {
        "name": "gpt-4o",
        "provider": "openai"
      }
    ],
    "embedding_model": "BAAI/bge-m3"
  }
}
```

---

## 六、评估 `/evaluate`

### 6.1 运行评估

使用 RAGAS 框架对 RAG Pipeline 进行评估。

```
POST /api/v1/evaluate
```

**请求体:**

```json
{
  "dataset_name": "qa_pairs_v1",
  "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
  "collection_name": "default"
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `dataset_name` | string | 是 | 评估数据集名称(位于 `evaluation/datasets/` 下) |
| `metrics` | list[string] | 否 | 评估指标，默认全部 4 个核心指标 |
| `collection_name` | string | 否 | 指定集合，默认 `"default"` |

**成功响应:** `202 Accepted`

```json
{
  "code": 202,
  "message": "Evaluation started",
  "data": {
    "eval_id": "eval_m1n2o3",
    "status": "running",
    "dataset_name": "qa_pairs_v1",
    "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
    "started_at": "2026-02-28T23:30:00+08:00"
  }
}
```

---

### 6.2 获取评估结果

```
GET /api/v1/evaluate/results
```

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `eval_id` | string | 否 | 指定评估 ID，不提供则返回最近一次 |

**成功响应:** `200 OK`

```json
{
  "code": 200,
  "message": "success",
  "data": {
    "eval_id": "eval_m1n2o3",
    "status": "completed",
    "dataset_name": "qa_pairs_v1",
    "results": {
      "faithfulness": 0.85,
      "answer_relevancy": 0.78,
      "context_precision": 0.82,
      "context_recall": 0.75
    },
    "sample_count": 50,
    "started_at": "2026-02-28T23:30:00+08:00",
    "completed_at": "2026-02-28T23:35:00+08:00"
  }
}
```

---

## 附录: 数据模型

### EmbeddedDocument (文档)

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 文档唯一标识(UUID) |
| `filename` | string | 文件名，唯一 |
| `file_path` | string | 文件绝对路径，唯一 |
| `filetype` | string | 文件后缀(如 `.pdf`) |
| `chunk_count` | int | 分块数量 |
| `chunk_ids` | list[string] | 所有 chunk 的 ID 列表 |
| `status` | string | 状态: `processing`, `completed`, `failed` |
| `metadata` | string | 处理元数据(embedding 模型、分块参数等) |
| `created_at` | datetime | 创建时间 |
| `updated_at` | datetime | 最后更新时间 |

### Conversation (会话)

| 字段 | 类型 | 说明 |
|------|------|------|
| `conversation_id` | string | 会话唯一标识(UUID) |
| `title` | string | 会话标题 |
| `document_ids` | list[string] | 选中的文档 ID 列表(知识来源) |
| `model_name` | string | 使用的 LLM 模型名称 |
| `max_history_turns` | int | 最大记忆轮数 |
| `message_count` | int | 消息总数 |
| `created_at` | datetime | 创建时间 |
| `last_active_at` | datetime | 最后活跃时间 |

### Message (消息)

| 字段 | 类型 | 说明 |
|------|------|------|
| `message_id` | string | 消息唯一标识(UUID) |
| `conversation_id` | string | 所属会话 ID |
| `role` | string | 角色: `user` 或 `assistant` |
| `content` | string | 消息内容 |
| `sources` | list[object] | 引用来源(仅 assistant 消息) |
| `created_at` | datetime | 创建时间 |
