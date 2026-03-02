from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class VectorStoreFactory:
    """
    向量数据库工厂(单例)：通过 init_vector_store 创建并缓存唯一的 VectorStore 实例。
    需要 EmbeddingModelFactory 已完成初始化。
    """
    _instance: VectorStore | None = None

    @classmethod
    def init_vector_store(cls,embeddings: Embeddings, collection_name: str = "rag_default") -> VectorStore:
        """
        初始化向量数据库并缓存为单例。
        自动从 EmbeddingModelFactory 获取 embedding 实例。
        若实例已存在则直接返回，不会重复创建。
        :param embeddings: 嵌入模型
        :param collection_name: ChromaDB 集合名称
        """
        if cls._instance is not None:
            print("INIT VECTOR STORE: Instance already exists, returning cached instance.")
            return cls._instance
        print(f"INIT VECTOR STORE: Initializing vector store (collection={collection_name})...")

        cls._instance = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory="./chroma_langchain_db",
        )

        print("INIT VECTOR STORE: Initialized successfully.")
        return cls._instance

    @classmethod
    def get_instance(cls) -> VectorStore:
        """获取 VectorStore 单例。未初始化时抛出异常。"""
        if cls._instance is None:
            raise RuntimeError("VectorStore has not been initialized. Call init_vector_store() first.")
        return cls._instance