# Simple RAG (Retrieval-Augmented Generation) System

[中文版本](./README-zh.md)

An advanced **Agentic RAG** system backend service developed based on **LangChain**, **LangGraph**, and **FastAPI**, supporting multimodal document retrieval. It features an intelligent reflective state machine, a hybrid retrieval strategy (BM25 + Vector Semantic Search + RRF), hierarchical chunking (Small-to-Big), and Re-ranking, offering highly accurate and hallucination-free conversations via SSE streaming.

For more detailed information, please view the `docs` directory.

## Features

  - 🧠 **Agentic Reflexive Graph Workflow**: Built on LangGraph, featuring a cognitive state machine for self-reflection, hallucination checking, usefulness evaluation, and question rewriting (Thinking-Rewriting loop).
  - 🔄 **Hybrid Retrieval & RRF Fusion**: Combines BM25 sparse keyword retrieval with embedding-based dense semantic search, united by Reciprocal Rank Fusion (RRF).
  - 🍱 **Intelligent Hierarchical Chunking**: Utilizes the Parent Document Retriever pattern with language/format-aware splitters (Markdown/HTML headers, Code splitters).
  - 🎯 **Advanced Re-ranking Optimization**: Seamless integration with Qwen Native Reranker and other lightweight compressors for deep semantic abstract re-ranking.
  - 🌐 **Multi-modal Document Ingestion**: Cascading parsers for local files (PDF, HTML, Markdown, Code) and remote Web URLs.
  - ⚡ **High-Performance Backend**: FastAPI powered asynchronous architecture, backed by async SQLAlchemy and psycopg connection pooling.
  - 📡 **Real-time SSE Streaming**: Delivers token-by-token streaming responses with intelligent error recovery and LangGraph Checkpoint state persistence.

## System Architecture

### Core Components

1.  **FastAPI & Routing Layer**
      - `/api/documents`: Document ingestion and management.
      - `/api/retrieval`: Retrieval testing and references binding.
      - `/api/conversation`: Chat interface with SSE streaming and history management.
2.  **Document Processing & Chunking**
      - Chain of Responsibility pattern for loaders (PDF, HTML, Text, WebBase).
      - Registry pattern for dynamic splitting strategies (Markdown/HTML/Code/Universal).
3.  **Retrieval & Vector Store Engine**
      - `EnhancedParentDocumentRetriever` + `HybridPDRetriever` for robust search.
      - ChromaDB (Vector Store) and LocalFileStore (KV Store) for cascading indices.
4.  **Reflexive LangGraph Agent**
      - State persistence via PostgresSaver (Checkpoints).
      - Nodes for: `retrieve`, `grade_documents`, `generate_answer`, `check_hallucination`, `check_usefulness`, `rewrite_question`, and `summarize_conversation`.

## Technical Stack
- **Core Frameworks**: Python 3.12+ | FastAPI | LangChain | LangGraph
- **Database & ORM**: PostgreSQL | SQLAlchemy (async) | psycopg (pool) | ChromaDB
- **AI/NLP**: HuggingFace(Qwen) / OpenAI Embeddings | Qwen Reranker | NLTK + jieba

## Installation and Configuration

### Environment Requirements

  - Python 3.12+.
  - PostgreSQL Database instance.
  - PyTorch (with optional CUDA support).
  - The following package dependencies.

### Dependency Installation

See `requirements.txt`
```bash
pip install -r requirements.txt
```

### Environment Variable Configuration

Copy and configure the `.env-backup` file to `.env` or adjust properties inside `config.yml`:

```yaml
# config.yml highlights:
database:
  url: postgresql+asyncpg://user:password@localhost:5432/simple_rag

chat_model:
  default: gpt-5-mini           # Primary model for response generation
  light: gpt-4o-mini            # Lightweight LLM for grading/reflection

embedding:
  model: Qwen/Qwen3-Embedding-0.6B

retriever:
  final_k: 8
  reranker:
    enabled: true
```

## Usage Instructions

### Starting the System

```bash
uvicorn app.main:app --reload
```

### Execution Flow

1.  **Initialization Phase (Application Startup)**
      - Load environment variables from `config.yml` and `.env`.
      - Initialize database connection pools and Vector Store instances.
      - Initialize document loader chain, text splitter registry and retrievers.
      - Set up embedding & reranking models.
      - Compile LangGraph StateGraph engine.
2.  **Interaction Phase**
      - **Ingest documents** via POST `/api/documents/local` or `/api/documents/url`.
      - **Set references** using POST `/api/retrieval/references`.
      - **Chat via SSE** via POST `/api/conversation/chat`, receiving token stream and conversation states.

### Document Loading

Supports versatile document loading:
  - Call the API endpoint `/api/documents/local` to upload and ingest local file types.
  - The system dynamically selects loaders (like PDFLoader, HTMLLoader) and chunks content via registered text splitters (like MarkdownTextSplitter, CodeTextSplitter), finally storing embeddings in ChromaDB and parent chunks in LocalFileStore automatically.

### Mode Descriptions
The system intelligently decides between utilizing retrieval tools or returning direct responses dynamically via LLM tool-calling.

## Technical Details

### Text Preprocessing & Chunking

  - Context-aware chunking: MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter, Language-specific RecursiveCharacterTextSplitter for optimal embedding representation.
  - Chinese/English bilingual text preprocessing for BM25 with `jieba` and `nltk` stop-word semantic filtering.

### Retrieval Strategy

1.  **Parent Document Retriever**: Segregates large documents (parents) and small semantic chunks (children).
2.  **Hybrid Search**: Merges sparse keyword (BM25) and dense vector retrievals.
3.  **RRF Fusion**: Re-ranks the combined lists mitigating domain bias.
4.  **Cross-Encoder Re-ranking**: Final precision adjustment utilizing Qwen Reranker.

### LangGraph Agent State Machine

The Agent decides between fulfilling direct queries or retrieving documents. If retrieved documents are irrelevant, the LLM rewrites the query automatically. Generated answers get double-checked for hallucination and usefulness; failures trigger internal loops (generate -> check -> rewrite) until maximum retry thresholds are reached.

## File Structure

```
RAG/
├── app/                          
│   ├── main.py                   # FastAPI Application Entry
│   ├── config/                   # Global & DB Configuration
│   ├── core/                     # Core Engine (Loader, Chunking, Retrieval, Graph)
│   ├── crud/                     # PostgreSQL Operations
│   ├── routers/                  # API Endpoints (Docs, Retrieval, Chat)
│   ├── models/                   # Schemas, VOs, Graph States
│   └── exception/                # Exception Handling
│  
├── docs/ 						  # reference document
│
├── .env-backup                   # Env variable backup
├── config.yml                    # Project Configuration File
├── Dockerfile                    # Docker Deployment
├── README.md                     # English Documentation
└── README-zh.md                  # Chinese Documentation
```

## Extension Development

### Adding New Tools & Capabilities
1. Integrate new features as distinct FastAPI Routers.
2. For Graph extensions, add new nodes or modify conditional edges within `app/core/graph.py` and implement corresponding business logic inside LangGraph states.

### Customizing Loaders or Splitters
Inherit `DocumentLoader` to support new MIME types and register them in `DocumentLoaderChain`.
Register custom text split logic via `SplitterRegistry`.

## Troubleshooting

### Common Issues

1. **Database Connection Failure**: Ensure PostgreSQL is running and credentials match `config.yml`.
2. **CUDA Unavailable**: Application automatically falls back to CPU for native HuggingFace embeddings.
3. **Graph Looping Exceeded Max Limits**: Lower the `max_rewrite_time` and `max_generate_time` in configurations.

## Changelog

### v1.0

- Implementation of the basic RAG system.
- Hybrid retrieval functionality.
- Tool-calling integration.
- Asynchronous processing optimization.

### v2.0 (Current Version)
- Fully refactored into a FastAPI backend service.
- Implemented LangGraph self-reflection Agent architecture.
- Replaced basic RAG strategy with Hybrid PDRetriever (BM25 + Vector + RRF + PDRetrieve + Rerank).
- Migrated state to asynchronous persistence (SQLAlchemy + PostgresSaver).
- Replaced CLI chat with SSE streaming REST API endpoints.

## Contributing

Contributions are welcome! Please submit Issues and Pull Requests to improve the project.
