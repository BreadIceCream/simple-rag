from __future__ import annotations

import asyncio
import sys
from pathlib import Path


_INITIALIZED_PROFILE: str | None = None
_GRAPH_INITIALIZED = False
_ASYNCIO_CONFIGURED = False


def _configure_asyncio_policy() -> None:
    global _ASYNCIO_CONFIGURED
    if _ASYNCIO_CONFIGURED:
        return
    # Windows + httpx/anyio/OpenAI async clients are more stable with Selector policy.
    if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    _ASYNCIO_CONFIGURED = True


def _ensure_base_runtime() -> None:
    from app.config.db_config import DatabaseManager
    from app.config.global_config import global_config

    _configure_asyncio_policy()
    global_config.load()
    DatabaseManager.init()


def _init_embedding_and_storage() -> None:
    from app.config.global_config import global_config
    from app.core.embeddings import EmbeddingModelFactory
    from app.core.vector_store import VectorStoreFactory

    EmbeddingModelFactory.init_embedding_model()
    VectorStoreFactory.init_vector_store(
        embeddings=EmbeddingModelFactory.get_instance(),
        collection_name=global_config.get("vector_store", {}).get("collection_name"),
    )
    VectorStoreFactory.init_docstore()


def _init_parent_retriever() -> None:
    from app.core.retriever import EnhancedParentDocumentRetrieverFactory
    from app.core.vector_store import VectorStoreFactory

    EnhancedParentDocumentRetrieverFactory.init(
        VectorStoreFactory.get_vector_store(),
        VectorStoreFactory.get_docstore(),
    )


def _init_full_graph_runtime() -> None:
    from app.core.chunking import SplitterRegistry
    from app.core.document_loader import DocumentLoaderChain
    from app.core.graph import Graph
    from app.core.reranker import RerankerFactory
    from app.core.retriever import ElasticSearchFactory, HybridPDRetrieverFactory

    global _GRAPH_INITIALIZED

    DocumentLoaderChain.init()
    SplitterRegistry.init()
    _init_parent_retriever()
    RerankerFactory.init()
    ElasticSearchFactory.init()
    HybridPDRetrieverFactory.init()
    Graph.build()
    _GRAPH_INITIALIZED = True


def init_eval_runtime(profile: str = "full") -> None:
    global _INITIALIZED_PROFILE
    if _INITIALIZED_PROFILE == profile:
        return
    if _INITIALIZED_PROFILE is not None and _INITIALIZED_PROFILE != profile:
        raise RuntimeError(
            f"Eval runtime already initialized with profile '{_INITIALIZED_PROFILE}', cannot reinitialize to '{profile}' in the same process."
        )

    _ensure_base_runtime()

    if profile == "dataset_seed":
        pass
    elif profile == "dataset_synthetic":
        _init_embedding_and_storage()
    elif profile == "dataset_replay":
        _init_embedding_and_storage()
        _init_parent_retriever()
    elif profile == "full":
        _init_embedding_and_storage()
        _init_full_graph_runtime()
    else:
        raise ValueError(f"Unsupported eval runtime profile: {profile}")

    _INITIALIZED_PROFILE = profile


async def close_eval_runtime_async() -> None:
    global _INITIALIZED_PROFILE, _GRAPH_INITIALIZED
    if _INITIALIZED_PROFILE is None:
        return

    if _GRAPH_INITIALIZED:
        from app.core.graph import Graph

        Graph.close()
        _GRAPH_INITIALIZED = False

    from app.config.db_config import DatabaseManager

    await DatabaseManager.close()
    _INITIALIZED_PROFILE = None


def close_eval_runtime() -> None:
    asyncio.run(close_eval_runtime_async())


def list_all_documents():
    from sqlalchemy import select

    from app.config.db_config import DatabaseManager
    from app.models.schemas import EmbeddedDocument

    with DatabaseManager.get_sync_db() as db:
        stmt = select(EmbeddedDocument).order_by(EmbeddedDocument.created_at.desc())
        return db.execute(stmt).scalars().all()


def get_documents_by_ids(doc_ids: list[str]):
    if not doc_ids:
        return []

    from sqlalchemy import select

    from app.config.db_config import DatabaseManager
    from app.models.schemas import EmbeddedDocument

    with DatabaseManager.get_sync_db() as db:
        stmt = select(EmbeddedDocument).where(EmbeddedDocument.id.in_(doc_ids))
        docs = db.execute(stmt).scalars().all()
    doc_map = {doc.id: doc for doc in docs}
    return [doc_map[doc_id] for doc_id in doc_ids if doc_id in doc_map]


def get_parent_chunks_by_ids(parent_doc_ids: list[str]):
    if not parent_doc_ids:
        return []

    from app.core.vector_store import VectorStoreFactory

    docstore = VectorStoreFactory.init_docstore()
    parent_docs = docstore.mget(parent_doc_ids)
    resolved = []
    for parent_doc_id, parent_doc in zip(parent_doc_ids, parent_docs):
        if parent_doc is None:
            continue
        resolved.append((parent_doc_id, parent_doc))
    return resolved


def get_all_document_ids() -> list[str]:
    return [doc.id for doc in list_all_documents()]


def default_dataset_storage_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "store" / "evals" / "datasets"
