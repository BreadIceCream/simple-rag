# RAG (Retrieval-Augmented Generation) System

An advanced RAG system developed based on LangGraph, supporting bilingual (Chinese and English) document retrieval. It employs a hybrid retrieval strategy (BM25 + semantic search) and a re-ranking function, offering two operational modes: retrieval-augmented mode and direct conversation mode.

[Chinese version](https://github.com/BreadIceCream/simple-rag/blob/master/README-zh.md)

## Features

  - 🔄 **Hybrid Retrieval**: A hybrid retrieval strategy combining BM25 sparse retrieval and semantic search.
  - 🌐 **Bilingual Support**: Comprehensive support for mixed Chinese and English text preprocessing and retrieval.
  - 🎯 **Re-ranking Optimization**: Support for the Qwen re-ranking model and the FlashRank re-ranker.
  - 💾 **Persistent Storage**: Vector database based on ChromaDB, with support for multi-collection management.
  - 🛠️ **Tool Integration**: An intelligent tool-calling system based on LangGraph.
  - ⚡ **Asynchronous Processing**: Asynchronous handling of document loading and embedding to improve performance.
  - 🎛️ **Configurable**: A rich set of environment variable configuration options.

## System Architecture

### Core Components

1.  **Document Processing Module**

      - PDF document loading.
      - Text chunking (with overlap support).
      - Bilingual text preprocessing.

2.  **Retrieval Module**

      - BM25 sparse retriever.
      - Vector semantic retriever.
      - Ensemble Retriever (for hybrid search).
      - Re-ranking compressor.

3.  **Language Model**

      - Support for multiple LLM providers.
      - Configurable temperature parameter.
      - Tool-calling capabilities.

4.  **State Management**

      - State graph based on LangGraph.
      - Support for document deduplication.
      - Mode switching (retrieval/direct).

## Installation and Configuration

### Environment Requirements

  - Python 3.8+.
  - PyTorch (with optional CUDA support).
  - The following package dependencies.

### Dependency Installation

```bash
pip install langchain langgraph chromadb 
pip install langchain-openai langchain-huggingface langchain-community 
pip install flashrank nltk jieba  python-dotenv pydantic ipython
pip install torch # optional if not using GPU
pip install transformers # optional if not using Qwen reranker
```

### Environment Variable Configuration

Copy and configure the `.env-backup` file (IGNORE `.env`). For more detailed configuration information, please refer to the `.env-backup` file:

```env
# LangSmith Tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key

# Model Configuration
MODEL_PROVIDER=openai
MODEL_NAME=gpt-4o-mini
OPENAI_BASE_URL=your_openai_base_url
OPENAI_API_KEY=your_openai_api_key

# Embedding Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
OPENAI_EMBEDDING=false
OPENAI_EMBEDDING_BASE_URL=your_openai_embedding_base_url
OPENAI_EMBEDDING_API_KEY=your_openai_embedding_api_key

# Re-ranking Configuration
RERANKER_ENABLED=false
QWEN_RERANKER=false
```

## Usage Instructions

### Starting the System

```bash
python rag.py
```

### Execution Flow

1.  **Initialization Phase**

      - Load environment variables.
      - Initialize the language model and embedding model.
      - Create or select a ChromaDB collection.
      - Load PDF documents.

2.  **Interaction Phase**

      - Select an operational mode:
          - `/retrieve` - Retrieval-augmented mode (default).
          - `/direct` - Direct conversation mode.
      - Enter a question to interact.
      - Type `exit` to quit the system.

### Document Loading

Supports batch loading of PDF documents:

  - Enter the path to a PDF file to load it.
  - Enter `done` to finish loading.
  - The system will process document embeddings asynchronously.

### Mode Descriptions

  - **Retrieval-Augmented Mode** (`/retrieve`): The system first retrieves relevant documents and then generates an answer based on the retrieval results.
  - **Direct Conversation Mode** (`/direct`): The system answers the question directly, which can be used for tasks like mathematical calculations.

## Technical Details

### Text Preprocessing

  - Uses jieba for Chinese word segmentation.
  - Uses NLTK for English lemmatization.
  - Supports filtering of both Chinese and English stop words.
  - Punctuation handling.

### Retrieval Strategy

1.  **BM25 Retrieval**: Keyword-based sparse retrieval.
2.  **Semantic Retrieval**: Vector similarity-based retrieval.
3.  **Hybrid Retrieval**: Fuses the results of the two retrieval methods using RRF (Reciprocal Rank Fusion).
4.  **Re-ranking**: Optional document re-ranking for optimization.

### Re-ranker Selection

  - **Qwen Native Reranker**: Based on the Qwen3-Reranker-0.6B model.
  - **FlashRank**: Based on the ms-marco-MiniLM-L-12-v2 model.
  - **Simple Compressor**: Returns only the top N documents (no re-ranking).

### Performance Optimization

  - Asynchronous document processing.
  - CUDA acceleration (if available).
  - Parallel task execution.
  - Document deduplication mechanism.

## File Structure

```
RAG/
├── rag.py                 # Main program file
├── qwen_reranker.py       # Qwen re-ranker implementation
├── simple_compressor.py   # Simple compressor implementation
├── .env                   # Environment variable configuration
├── chroma_langchain_db/   # ChromaDB database directory
└── README.md              # Project documentation
```

## Extension Development

### Adding New Tools

1.  Create a new tool using the `@tool` decorator.
2.  Implement the tool function.
3.  The system will automatically detect and register the tool.

### Customizing Preprocessing

Modify the `bilingual_preprocess_func` function to customize the text preprocessing logic.

### Integrating New Embedding Models

Add support for new embedding models in the `init_embedding_model` function.

## Troubleshooting

### Common Issues

1.  **CUDA Unavailable**: The system will automatically fall back to CPU mode.
2.  **Document Loading Failure**: Check file paths and permissions.
3.  **Embedding Model Compatibility**: Ensure the embedding model used is compatible with the collection.
4.  **Insufficient Memory**: Consider reducing the document batch size or using a smaller model.

### Debugging Options

  - Enable LangSmith tracing for debugging.
  - Check the detailed logs in the console output.
  - Inspect the ChromaDB collection metadata.

## Changelog

### v1.0 (Current Version)

  - Implementation of the basic RAG system.
  - Hybrid retrieval functionality.
  - Bilingual support.
  - Tool-calling integration.
  - Asynchronous processing optimization.

## Contributing

Contributions are welcome\! Please submit Issues and Pull Requests to improve the project.