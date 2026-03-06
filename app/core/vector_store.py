from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langchain_classic.storage import LocalFileStore, create_kv_docstore

# RAG 项目根目录：vector_store.py -> core -> app -> RAG/
_RAG_ROOT = Path(__file__).resolve().parent.parent.parent
_STORE_DIR = _RAG_ROOT / "store"


class VectorStoreFactory:
    """
    向量数据库 & 文档存储工厂(单例)。
    持久化路径统一基于文件位置计算，确保与程序运行路径无关。
    """
    _vector_store: VectorStore | None = None
    _docstore: BaseStore | None = None

    @classmethod
    def _get_vector_persist_directory(cls) -> str:
        """Chroma 持久化目录：RAG/store/chroma_langchain_db"""
        return str(_STORE_DIR / "chroma_langchain_db")

    @classmethod
    def _get_docstore_directory(cls) -> str:
        """LocalFileStore 持久化目录：RAG/store/parent_docs"""
        return str(_STORE_DIR / "parent_docs")

    @classmethod
    def init_vector_store(cls, embeddings: Embeddings, collection_name: str = "rag_default") -> VectorStore:
        """
        初始化 Chroma 向量数据库并缓存为单例。
        :param embeddings: 嵌入模型
        :param collection_name: ChromaDB 集合名称
        """
        if cls._vector_store is not None:
            print("INIT VECTOR STORE: Instance already exists, returning cached instance.")
            return cls._vector_store

        persist_dir = cls._get_vector_persist_directory()
        print(f"INIT VECTOR STORE: Initializing (collection={collection_name}, persist={persist_dir}). "
              f"NOTE: The same collection can only use embedding models with the same embedding dimension.")

        cls._vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

        print("INIT VECTOR STORE: Initialized successfully.")
        return cls._vector_store

    @classmethod
    def init_docstore(cls) -> BaseStore:
        """
        初始化 LocalFileStore + create_kv_docstore，用于存储父文档。
        """
        if cls._docstore is not None:
            print("INIT DOCSTORE: Instance already exists, returning cached instance.")
            return cls._docstore

        docstore_dir = cls._get_docstore_directory()
        print(f"INIT DOCSTORE: Initializing (path={docstore_dir})...")

        fs = LocalFileStore(docstore_dir)
        cls._docstore = create_kv_docstore(fs)

        print("INIT DOCSTORE: Initialized successfully.")
        return cls._docstore

    @classmethod
    def get_vector_store(cls) -> VectorStore:
        """获取 VectorStore 单例。未初始化时抛出异常。"""
        if cls._vector_store is None:
            raise RuntimeError("VectorStore has not been initialized. Call init_vector_store() first.")
        return cls._vector_store

    @classmethod
    def get_docstore(cls) -> BaseStore:
        """获取 DocStore 单例。未初始化时抛出异常。"""
        if cls._docstore is None:
            raise RuntimeError("DocStore has not been initialized. Call init_docstore() first.")
        return cls._docstore

    # 向后兼容旧的调用方
    @classmethod
    def get_instance(cls) -> VectorStore:
        """兼容旧代码，等同于 get_vector_store()"""
        return cls.get_vector_store()