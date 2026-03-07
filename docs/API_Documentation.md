# RAG 项目 API 接口文档

本文档基于 `app/routers` 和 `app/models` 下的代码生成，详细描述了所有 API 接口的请求和返回数据结构。

---

## 模型定义 (Models)

### 基础返回结构

所有非流式 (Non-SSE) 的响应都统一被封装在 `Result` 模型中：
```json
{
  "code": 200,          // 整数，状态码（如 200 表示成功）
  "message": "提示信息", // 字符串，提示或错误信息
  "data": {}            // 对应各接口返回的实际数据主体，结构见具体接口
}
```

### 文档相关对象

#### EmbeddedDocumentVO

用于表示数据库中的文档信息（不含切片块等详细内容）。
- `id` (str): 文档唯一标识(UUID)
- `path` (str): 文件路径/URL（唯一）
- `is_url` (bool): 是否为 URL
- `file_directory` (str|null): 源文件所在目录
- `file_name` (str|null): 源文件名称
- `file_extension` (str): 文件后缀名（如 `.pdf`）
- `mime_type` (str|null): MIME 类型
- `last_modified` (datetime|null): 文件最后修改时间

#### EmbeddedDocument

用于表示存入数据库表 `embedded_document` 的完整文档信息记录。
- `id` (str): 文档唯一标识(UUID)
- `path` (str): 文件路径/url, 唯一
- `is_url` (bool): 是否为URL
- `file_directory` (str|null): 源文件所在目录, 仅本地文件使用
- `file_name` (str|null): 源文件名称, 仅本地文件使用
- `file_extension` (str): 后缀名(如 .pdf, .html)
- `mime_type` (str|null): 文件mime类型
- `last_modified` (str|null): 文件最后修改时间, 仅本地文件使用
- `parent_doc_ids` (list[str]): 父文档的 ID 列表
- `children_count` (int): 子文档数量
- `load_metadata` (dict): 加载文档时的元信息，如使用的嵌入模型、加载器、分块器等
- `created_at` (datetime): 创建时间

### 聊天对话相关视图对象

#### ConversationVO
表示一个对话列表项
- `id` (str): 对话 ID (即 `conversationId`)
- `title` (str): 自动生成的对话标题
- `created_at` (datetime|null): 创建时间
- `updated_at` (datetime|null): 最后更新时间

#### ChatReferenceParentDocVO
聊天参考资料中的具体父文档层级片段详细信息
- `id` (str): 父文档ID
- `parent_index` (int): 父文档在原文件中的分块顺序索引
- `content` (str): 父文档文本内容

#### ChatReferenceVO
聊天参考资料的视图对象（表示某一个参考的文件或 URL）
- `id` (str): 源文件文档UUID
- `path` (str): 源路径/URL
- `file_name` (str|null): 文件名
- `parent_docs` (list[ChatReferenceParentDocVO]): 本次聊天检索引用的父文档块列表

#### ChatMessageVO
一条对话消息的视图记录
- `id` (int): 消息自增 ID
- `role` (str): 消息角色，通常为 `user` 或 `ai`
- `content` (str): 消息文本内容
- `references` (list[ChatReferenceVO]): 若为 AI 消息，则含其生成的参考文档列表
- `created_at` (datetime|null): 创建时间

#### ChatHistoryResponseVO
获取历史消息时的外层包装
- `conversation_id` (str): 对话 ID
- `messages` (list[ChatMessageVO]): 有序消息列表

### SSE 流式输出返回模型

流式对话的响应事件模型均会被序列化为字符串通过 `text/event-stream` 进行推送：
- **SseTokenVO**: 发送 Token 流 (`status: "progress"`, `event: "Token streaming."`, `token: "..."`)
- **SseAnswerVO**: 最终结果 (`status: "finished"`, `event: "Answer generated."`, `answer: "..."`, `references: list`, `conversation_id: "..."`)
- **SseDoneVO**: 流结束标志 (`status: "finished"`, `event: "Done"`)
- **SseErrorVO**: 抛出异常标志 (`status: "error"`, `event: "Error"`, `message: "..."`, `code: int`, `conversation_id: "..."`)


---

## API 接口详情

### 1. 对话模块 (`/api/conversation`)

#### 1.1 发起对话 (流式获取输出)
- **POST** `/api/conversation/chat`
- **请求体 (Body)**
  ```json
  {
      "message": "我要咨询什么内容...",
      "conversationId": "xxxx-xxxx-xxxx-xxxx" // 选填：已有对话的 UUID，留空则自动创建新对话
  }
  ```
- **响应**: `text/event-stream`
  流式多次返回 SSE 数据。正常序列结束时返回 `SseDoneVO`。如遇异常发送 `SseErrorVO` 并关闭连接。支持断点恢复续答。

#### 1.2 获取所有对话列表 
- **GET** `/api/conversation/list`
- **响应体数据域 (`data`)**: `list[ConversationVO]`

#### 1.3 获取某个对话的历史记录
- **GET** `/api/conversation/history`
- **请求参数 (Query)**:
  - `conversation_id` (UUID): 必填，对话的唯一 ID
- **响应体数据域 (`data`)**: `ChatHistoryResponseVO`

#### 1.4 修改对话标题
- **PUT** `/api/conversation/{conversation_id}/title`
- **路径参数**: `conversation_id` (UUID) 
- **请求体 (Body)**:
  ```json
  {
      "title": "新名字"
  }
  ```
- **响应体数据域 (`data`)**:
  ```json
  {
      "id": "xxxx",
      "title": "新名字"
  }
  ```

#### 1.5 删除指定对话 
- **DELETE** `/api/conversation/{conversation_id}`
- **路径参数**: `conversation_id` (UUID)
- **响应体数据域 (`data`)**: `null`

---

### 2. 知识库文档模块 (`/api/documents`)

#### 2.1 列出所有已上传的文档
- **GET** `/api/documents/list`
- **响应体数据域 (`data`)**: `list[EmbeddedDocumentVO]`

#### 2.2 获取文件父切块列表 (分页)
- **GET** `/api/documents/parents`
- **请求参数 (Query)**:
  - `docId` (UUID): 必填，数据库中文档的 uuid
  - `offset` (int): 取片起始偏移量，默认 `0`
  - `limit` (int): 每页限制行数，最大 `20`，默认 `10`
- **响应体数据域 (`data`)**:
  ```json
  {
      "total": 100, // 此文件的分块总数
      "chunks": []  // 具体 Document 分块列表数据 (受Langchain原生Document限制) 
  }
  ```

#### 2.3 上传本地文档
- **POST** `/api/documents/local`
- **请求参数 (Query)**:
  - `file_path` (str): 必填，本地绝对/相对文件路径
- **响应体数据域 (`data`)**: `null` （如发生错误直接抛出普通异常中断）

#### 2.4 上传网络 URL
- **POST** `/api/documents/url`
- **请求参数 (Query)**:
  - `url` (str): 必填，希望读取并嵌入的网页 URL
- **响应体数据域 (`data`)**: `null`

#### 2.5 获取文档详情信息 
- **GET** `/api/documents/{doc_id}`
- **路径参数**: `doc_id` (UUID)
- **响应体数据域 (`data`)**: `EmbeddedDocument`

#### 2.6 删除文档 
- **DELETE** `/api/documents/{doc_id}`
- **路径参数**: `doc_id` (UUID)
- **响应体数据域 (`data`)**: `null`

---

### 3. 检索配置模块 (`/api/retrieval`)

当我们在多个长文知识库中，希望限定仅根据特定几篇文件进行检索回复时，可使用该组接口来预设“参考文档”白名单，该限制将应用于后续的聊天（以及直接检索查询）中。

#### 3.1 设置参考文档限定列表
- **POST** `/api/retrieval/references`
- **请求体 (Body)**:
  ```json
  {
      "docIds": ["uuid-1", "uuid-2"] // 希望限定的已入库文档 UUID 列表
  }
  ```
- **响应体数据域 (`data`)**: `null`

#### 3.2 获取当前已设置的限定参考文档列表
- **GET** `/api/retrieval/references`
- **响应体数据域 (`data`)**: `list[EmbeddedDocumentVO]`

#### 3.3 清空限定列表
- **DELETE** `/api/retrieval/references`
- **引申描述**: 清空当前设置的参考文档列表，即不参考任何文档。
- **响应体数据域 (`data`)**: `null`

#### 3.4 根据文字描述直接检索文档块（测试用，无需实现）
- **POST** `/api/retrieval/query`
- **请求体 (Body)**:
  
  ```json
  {
      "message": "用户要查询的相关词汇或句子"
  }
  ```
- **响应体数据域 (`data`)**: `list` 返回经过检索和 Rerank 后命中的最终父级相关文档块片段 (受Langchain原生Document限制)
