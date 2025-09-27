# RAG (Retrieval-Augmented Generation) System

基于 LangGraph 开发的高级 RAG 系统，支持双语（中英文）文档检索，采用混合检索策略（BM25 + 语义搜索）和重排序功能，提供两种运行模式：检索增强模式和直接对话模式。

## 项目特性

- 🔄 **混合检索**: 结合 BM25 稀疏检索和语义搜索的混合检索策略
- 🌐 **双语支持**: 完整的中英文混合文本预处理和检索能力
- 🎯 **重排序优化**: 支持 Qwen 重排序模型和 FlashRank 重排序器
- 💾 **持久化存储**: 基于 ChromaDB 的向量数据库，支持多集合管理
- 🛠️ **工具集成**: 基于 LangGraph 的智能工具调用系统
- ⚡ **异步处理**: 文档加载和嵌入过程采用异步处理，提升性能
- 🎛️ **可配置**: 丰富的环境变量配置选项

## 系统架构

### 核心组件

1. **文档处理模块**
   - PDF 文档加载
   - 文本分块（支持重叠）
   - 双语文本预处理

2. **检索模块**
   - BM25 稀疏检索器
   - 向量语义检索器
   - 混合检索器（Ensemble Retriever）
   - 重排序压缩器

3. **语言模型**
   - 支持多种 LLM 提供商
   - 可配置温度参数
   - 工具调用能力

4. **状态管理**
   - 基于 LangGraph 的状态图
   - 支持文档去重
   - 模式切换（检索/直接）


## 安装和配置

### 环境要求

- Python 3.8+
- PyTorch (可选 CUDA 支持)
- 以下依赖包

### 依赖安装

```bash
pip install langchain langgraph chromadb
pip install langchain-openai langchain-huggingface
pip install langchain-community flashrank transformers
pip install nltk jieba torch python-dotenv
# optional
pip install qwen-reranker
```

### 环境变量配置

复制并配置 `.env` 文件，更多详细配置信息请查看 `.env-backup` 文件：

```env
# LangSmith 追踪
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key

# 模型配置
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
OPENAI_BASE_URL=your_openai_base_url
OPENAI_API_KEY=your_openai_api_key

# 嵌入模型配置
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
OPENAI_EMBEDDING=false
OPENAI_EMBEDDING_BASE_URL=your_openai_embedding_base_url
OPENAI_EMBEDDING_API_KEY=your_openai_embedding_api_key

# 重排序配置
RERANKER_ENABLED=false
QWEN_RERANKER=false
```

## 使用说明

### 启动系统

```bash
python rag.py
```

### 运行流程

1. **初始化阶段**
   - 加载环境变量
   - 初始化语言模型和嵌入模型
   - 创建或选择 ChromaDB 集合
   - 加载 PDF 文档

2. **交互阶段**
   - 选择运行模式：
     - `/retrieve` - 检索增强模式（默认）
     - `/direct` - 直接对话模式
   - 输入问题进行交互
   - 输入 `exit` 退出系统

### 文档加载

支持批量加载 PDF 文档：
- 输入 PDF 文件路径进行加载
- 输入 `done` 完成加载
- 系统会异步处理文档嵌入

### 模式说明

- **检索增强模式** (`/retrieve`): 系统会先检索相关文档，然后基于检索结果生成答案
- **直接对话模式** (`/direct`): 系统直接回答问题，可用于数学计算等任务

## 技术细节

### 文本预处理

- 使用 jieba 进行中文分词
- NLTK 进行英文词形还原
- 支持中英文停用词过滤
- 标点符号处理

### 检索策略

1. **BM25 检索**: 基于关键词的稀疏检索
2. **语义检索**: 基于向量相似度的检索
3. **混合检索**: 使用 RRF (Reciprocal Rank Fusion) 融合两种检索结果
4. **重排序**: 可选的文档重排序优化

### 重排序器选择

- **Qwen Native Reranker**: 基于 Qwen3-Reranker-0.6B 模型
- **FlashRank**: 基于 ms-marco-MiniLM-L-12-v2 模型
- **Simple Compressor**: 仅返回前 N 个文档（无重排序）

### 性能优化

- 异步文档处理
- CUDA 加速（如可用）
- 并行任务执行
- 文档去重机制

## 文件结构

```
RAG/
├── rag.py                 # 主程序文件
├── qwen_reranker.py       # Qwen 重排序器实现
├── simple_compressor.py   # 简单压缩器实现
├── .env                   # 环境变量配置
├── chroma_langchain_db/   # ChromaDB 数据库目录
└── README.md             # 项目说明文档
```

## 扩展开发

### 添加新工具

1. 使用 `@tool` 装饰器创建新工具
2. 实现工具函数
3. 系统会自动检测并注册工具

### 自定义预处理

修改 `bilingual_preprocess_func` 函数来自定义文本预处理逻辑。

### 集成新的嵌入模型

在 `init_embedding_model` 函数中添加新的嵌入模型支持。

## 故障排除

### 常见问题

1. **CUDA 不可用**: 系统会自动回退到 CPU 模式
2. **文档加载失败**: 检查文件路径和权限
3. **嵌入模型兼容性**: 确保使用的嵌入模型与集合兼容
4. **内存不足**: 考虑减少文档批次大小或使用更小的模型

### 调试选项

- 启用 LangSmith 追踪进行调试
- 查看控制台输出的详细日志
- 检查 ChromaDB 集合元数据

## 更新日志

### v1.0 (当前版本)
- 基础 RAG 系统实现
- 混合检索功能
- 双语支持
- 工具调用集成
- 异步处理优化

## 贡献

欢迎提交 Issue 和 Pull Request 来改进项目。
