# Simple RAG (Retrieval-Augmented Generation) System

[中文版本](./README-zh.md) | [Front End Project](https://github.com/BreadIceCream/simple-rag-frontend)

Simple RAG is an **Agentic RAG backend** built with **FastAPI**, **LangChain**, and **LangGraph**. It combines multi-format document ingestion, hierarchical chunking, hybrid retrieval, reranking, reflective answer generation, SSE streaming, and a **real-execution evaluation pipeline** based on **RAGAS**.

The system is designed around a simple principle: both online answering and offline evaluation should run through the real retrieval and graph workflow as much as possible. The project therefore focuses on production-style retrieval quality, controllable graph behavior, and evaluation grounded in actual system execution rather than synthetic self-scoring only.

For detailed technical documents, see the `docs/` directory.

## Core Features

- **Agentic LangGraph Workflow**: Retrieval, direct response, question rewriting, hallucination checking, usefulness checking, and conversation summarization are organized as a persistent state machine.
- **Hybrid Retrieval**: Semantic vector retrieval and BM25-style sparse retrieval are fused with reciprocal rank fusion.
- **Hierarchical Parent-Child Chunking**: Parent chunks preserve semantic completeness while child chunks improve recall granularity.
- **Structure-Aware Ingestion**: Markdown, HTML, code, Office documents, PDFs, and web pages are loaded with different strategies instead of a single generic loader.
- **Optional Reranking**: Qwen-based rerankers can be enabled to improve final parent-document ordering.
- **Conversation Persistence and Resume**: LangGraph checkpoints are stored in PostgreSQL so interrupted chats can be resumed.
- **SSE Streaming**: Token events, graph progress, final answers, and references are streamed to the client in real time.
- **Real RAG Evaluation with RAGAS**: Datasets, live execution, retrieval metrics, and RAGAS scoring are integrated into one offline evaluation workflow.

## Core Components

### Online Serving Path

1. `app/main.py` initializes config, database, embeddings, vector store, docstore, loaders, splitters, retrievers, rerankers, Elasticsearch, and LangGraph.
2. `app/core/document_loader.py` loads local files and URLs into unified `Document` objects.
3. `app/core/chunking.py` applies structure-aware parent splitting and smaller child splitting.
4. `app/core/retriever.py` builds the hybrid retrieval pipeline across Chroma, Elasticsearch, and parent-doc backtracking.
5. `app/core/reranker.py` optionally reranks fused parent-document candidates.
6. `app/core/graph.py` defines the answer-generation workflow and recovery loop.
7. `app/routers/conversation.py` exposes SSE chat APIs on top of the graph.

### Offline Evaluation Path

1. `app/evals/build_replay_dataset.py`, `build_synthetic_dataset.py`, and `import_seed_dataset.py` prepare datasets from different sources.
2. `app/evals/live_rag_runner.py` executes the real RAG system against dataset samples.
3. `app/evals/ragas_scorer.py` scores the run with RAGAS and retrieval metrics.
4. `app/evals/ragas_runner.py` provides a one-command wrapper for the full flow.
5. Evaluation artifacts are stored under `store/evals/datasets/` and `store/evals/experiments/`.

## Technical Highlights

- **Graph-based recovery loop**: The workflow does not just retrieve once and answer. It can rewrite the question, regenerate, and self-check support and usefulness before ending.
- **Parent-document retrieval design**: The system retrieves fine-grained child chunks, then reconstructs answer context from parent chunks for better coherence.
- **Structure-preserving splitting**: Markdown headers, HTML headers, and code language boundaries are preserved as much as possible before recursive splitting.
- **Scoped retrieval**: Retrieval can be restricted to a selected set of files, which is used both by the online retriever endpoints and by the conversation graph.
- **Persistent graph state**: PostgreSQL-backed checkpointers make conversation state resumable and inspectable.
- **Evaluation decoupling**: Dataset construction is separated from live execution and scoring, allowing replay, synthetic, and imported datasets to share the same evaluation runner.
- **Evaluation robustness for synthetic generation**: Synthetic dataset generation includes low-concurrency RAGAS execution, dynamic batch control, retry/backoff, and adaptive batch splitting for heavy files.

## Architecture

### Online RAG Path

1. Load configuration from `config.yml` and environment variables.
2. Initialize database, embedding model, vector store, and parent document store.
3. Initialize loaders, splitters, retrievers, rerankers, and LangGraph.
4. Expose APIs for document ingestion, retrieval testing, and conversation.

### Offline Evaluation Path

1. Build or import an evaluation dataset.
2. Optionally export and apply a review sheet.
3. Execute the real RAG pipeline for every sample.
4. Score the run with RAGAS and retrieval metrics.
5. Inspect run artifacts such as `summary.json` and `report.md`.

## Project Structure

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

### `app/core` Overview

- `document_loader.py`: multi-format and URL ingestion with unified metadata.
- `chunking.py`: structure-aware parent splitting and child splitting registry.
- `embeddings.py`: embedding backend initialization and switching.
- `vector_store.py`: Chroma vector store and local parent-doc store management.
- `retriever.py`: Elasticsearch retrieval, parent-doc retrieval, hybrid fusion, and retrieval scoping.
- `reranker.py`: optional reranking layer.
- `graph.py`: LangGraph state machine for online answering.

### `app/evals` Overview

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

See [docs/RAGAS集成方案.md](./docs/RAGAS集成方案.md) for the file-by-file explanation and command reference.

## Technical Stack

- **Language**: Python 3.12+
- **Backend**: FastAPI
- **RAG Frameworks**: LangChain, LangGraph
- **Database**: PostgreSQL, SQLAlchemy (async), psycopg
- **Sparse Retrieval**: Elasticsearch
- **Vector Store**: ChromaDB
- **Embedding / LLM**: HuggingFace or OpenAI-compatible backends
- **Reranking**: Qwen Reranker
- **Evaluation**: RAGAS

## Installation

### Requirements

- Python 3.12+
- PostgreSQL
- Elasticsearch
- PyTorch
- Dependencies from `requirements.txt`

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Configure `.env` as needed and update `config.yml`.

### `config.yml` Highlights

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

### Important Config Areas

- `database`: async database and checkpoint persistence.
- `elasticsearch`: sparse retrieval backend.
- `chat_model`: default answer model and lightweight control model.
- `embedding`: local or remote embedding backend.
- `chunking`: parent / child chunk sizes and overlap.
- `retriever`: final top-k and reranker switch.
- `chat`: rewrite / generation retry limits and summarize threshold.
- `debug`: ingestion and graph debugging switches.

## Usage

### Start the API Service

```bash
uvicorn app.main:app --reload
```

### Main API Paths

- `/api/documents`: document ingestion and management
- `/api/retrieval`: retrieval testing and reference scoping
- `/api/conversation`: SSE chat interface

## RAGAS Evaluation

The evaluation pipeline is designed around **real execution**:

1. Build or import a dataset.
2. Optionally review the dataset.
3. Run the real RAG system.
4. Score the results with RAGAS and retrieval metrics.

### Supported Dataset Types

- `replay`: built from historical conversations
- `synthetic`: generated from parent document chunks
- `seed`: imported from curated `.json` / `.jsonl` files

### Quick Smoke Test

```bash
python -m app.evals.build_synthetic_dataset --name synthetic_smoke --version v1 --category exploration --size 20 --doc-limit 10 --use-light-model
python -m app.evals.ragas_runner --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1 --limit 10 --review-status pending,approved
```

### Optional Review Flow

```bash
python -m app.evals.dataset_builder export-review --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1
python -m app.evals.dataset_builder apply-review --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1 --review-file store/evals/datasets/exploration/synthetic_smoke/v1/review_sheet.csv
```

### Two-Step Execution Flow

```bash
python -m app.evals.live_rag_runner --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1 --review-status pending,approved
python -m app.evals.ragas_scorer --run-dir <run_dir>
```

### One-Step Wrapper

```bash
python -m app.evals.ragas_runner --dataset-dir store/evals/datasets/exploration/synthetic_smoke/v1 --review-status pending,approved
```

### Evaluation Artifacts

- Datasets: `store/evals/datasets/...`
- Runs: `store/evals/experiments/...`
- Common outputs: `manifest.json`, `samples.jsonl`, `review_sheet.csv`, `records.jsonl`, `summary.json`, `report.md`

For the full evaluation design, dataset schema, and command reference, see [docs/RAGAS集成方案.md](./docs/RAGAS集成方案.md).

## Troubleshooting

1. **Database connection issues**: verify the PostgreSQL URL in `config.yml`.
2. **Elasticsearch issues**: check the ES URL, credentials, and local certificate setup.
3. **Embedding or model loading failures**: check local model dependencies and environment variables.
4. **Evaluation connection errors**: lower synthetic generation concurrency or use the low-concurrency defaults built into `build_synthetic_dataset`.
5. **Graph retry loops are too aggressive**: tune `chat.max_rewrite_time` and `chat.max_generate_time`.

## Contributing

Issues and pull requests are welcome.
