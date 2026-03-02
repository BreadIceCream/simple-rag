# import datetime
# import os
#
# import chromadb
# from chromadb import ClientAPI
# from chromadb.api.models.Collection import Collection
# from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# from langchain_core.embeddings import Embeddings
# from langchain_core.retrievers import BaseRetriever
# from langchain_chroma import Chroma
# from langchain_community.document_compressors import FlashrankRerank
# from flashrank import Ranker
#
# from app.core.reranker import QwenNativeReranker, SimpleCompressor
#
#
# # 初始化向量数据库
# async def init_vector_store(embeddings: Embeddings, client: ClientAPI, collection: Collection) -> Chroma:
#     print("INIT VECTOR STORE: Initializing vector store...")
#     return Chroma(
#         client=client,
#         collection_name=collection.name,
#         embedding_function=embeddings
#     )
#
#
# def vector_store_use_or_create_collection(client: ClientAPI) -> Collection:
#     collections = client.list_collections()
#     exist_collections_name = []
#     collection_name = "rag_default"
#     if collections:
#         print("INIT VECTOR STORE: History collections\n-------------------------------------------------")
#         for c in collections:
#             exist_collections_name.append(c.name)
#             print(f"|   Collection: {c.name}, metadata:{c.metadata}   |")
#         print("-------------------------------------------------")
#     while True:
#         collection_name = input(
#             f"INIT VECTOR STORE: Please input a collection name to {'use or create a new one:' if exist_collections_name else 'create:'}")
#         if collection_name in exist_collections_name:
#             print(f"INIT VECTOR STORE: Using history collection {collection_name}")
#         else:
#             print(f"INIT VECTOR STORE: Creating collection {collection_name}")
#         collection = client.get_or_create_collection(name=collection_name,
#                                                      metadata={
#                                                          "create_time": datetime.datetime.now().strftime(
#                                                              "%Y-%m-%d %H:%M %A"),
#                                                          "embedding_model": os.environ["EMBEDDING_MODEL"],
#                                                          "hnsw:space": "cosine"
#                                                      })
#         if collection.metadata["embedding_model"] != os.environ["EMBEDDING_MODEL"]:
#             print(
#                 f"INIT VECTOR STORE: WARNING! Current embedding model {os.environ['EMBEDDING_MODEL']} is not compatible with the collection, please delete the collection or create a new one.")
#         else:
#             return collection
#
#
# # 创建hybrid_retriever，使用RRF进行融合
# def init_hybrid_retriever(sparse_retriever: BaseRetriever, semantic_retriever: BaseRetriever,
#                           weights: list[float] = None) -> BaseRetriever:
#     """weights: A list of weights corresponding to the retrievers. Defaults to equal weighting for all retrievers."""
#     if weights is None:
#         weights = [0.5, 0.5]
#     print("INIT HYBRID RETRIEVER: Initializing hybrid retriever...")
#     return EnsembleRetriever(retrievers=[sparse_retriever, semantic_retriever], weights=weights)
#
#
# # 初始化CompressionRetriever，内置rerank
# def init_compression_retriever(base_retriever: BaseRetriever, top_n: int = 7) -> BaseRetriever:
#     """
#     top_n: Number of documents to return by CompressionRetriever. Default 7.
#     Notice that only when RERANKER_ENABLED in .env file is true, a real compression retriever which wraps a base retriever and a reranker (a base compressor) will be returned.
#     Otherwise, the method will use a simple compressor which returns top_n documents from the base retriever.
#     """
#     print("INIT COMPRESSION RETRIEVER: Initializing compression retriever...")
#     compressor = None
#     if os.environ["RERANKER_ENABLED"] == "true":
#         if os.environ["QWEN_RERANKER"] == "true":
#             compressor = QwenNativeReranker(top_n=top_n)
#         else:
#             ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
#             compressor = FlashrankRerank(client=ranker, top_n=top_n)
#     else:
#         print("INIT COMPRESSION RETRIEVER: Reranker is disabled...")
#         compressor = SimpleCompressor(top_n=top_n)
#     return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)
