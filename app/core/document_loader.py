import asyncio
import os
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever

from app.utils.preprocessing import nltk_resource_download, bilingual_preprocess_func


# 加载文档并嵌入向量数据库
def load_doc_to_vector_store(text_splitter: RecursiveCharacterTextSplitter, vector_store: Chroma, file_path: str,
                             task_id: int | None = None) -> \
        dict[str, Exception | None | int] | dict[str, None | list[str] | int]:
    print(f"LOADING DOCUMENTS TASK: Loading task <{task_id}>, document: {file_path}...")
    try:
        start_time = time.time()
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        # Essentially, aadd_documents calls the run_in_executor method of asyncio, which is equivalent to .to_thread(), executing in a separate thread.
        document_ids = vector_store.add_documents(documents=all_splits)
        end_time = time.time()
        print(
            f"LOADING DOCUMENTS TASK: Task <{task_id}> Done! Added {len(all_splits)} documents to the vector store this time, cost: {end_time - start_time:.2f}s")
        return {"task_id": task_id, "error": None, "document_ids": document_ids}
    except Exception as e:
        return {"task_id": task_id, "error": e, "document_ids": None}


# 加载文档、设置并返回sparse_retriever和semantic_retriever
async def load_docs_and_get_retriever(text_splitter: RecursiveCharacterTextSplitter, vector_store: Chroma,
                                      sparse_k: int = 30, semantic_k: int = 30) -> dict[
    str, BaseRetriever | list[str]]:
    """
    sparse_k: Amount of documents to return by sparse retriever, which retrieves using keywords (Default: 30, to improve the precision rate)
    semantic_k: Amount of documents to return by semantic retriever, which is provided by vector store and retrieves using similarity (Default: 30, to improve the precision rate)
    """
    print(
        f"LOADING DOCUMENTS: Before adding documents, The number of documents in vector store is {vector_store._collection.count()}")
    # download NLTK resources for bilingual preprocessing
    nltk_resource_download_task = asyncio.to_thread(nltk_resource_download)
    all_document_ids = []
    load_task_list = []
    load_task_id = 1
    load_task_id_to_file_path = {}
    while True:
        file_path = input("LOADING DOCUMENTS: Enter the file path, only supported .pdf (if ok, input 'done')：")
        if file_path == "done":
            break
        elif os.path.exists(file_path):
            # run task in another process, map the task id to file path, and add task to task list
            load_task_id_to_file_path[load_task_id] = file_path
            task = asyncio.to_thread(load_doc_to_vector_store, text_splitter, vector_store, file_path, load_task_id)
            load_task_id += 1
            load_task_list.append(task)
        else:
            print("LOADING DOCUMENTS: File not exist!")
    # wait for all tasks to complete, get the results and handle the exceptions
    results = await asyncio.gather(*load_task_list, nltk_resource_download_task)
    for result in results:
        if result is None:
            continue
        elif result["error"]:
            print(
                f"LOADING DOCUMENTS: Task <{result['task_id']}> failed \n file path: {load_task_id_to_file_path[result['task_id']]} \n error message : {result['error']}")
        else:
            all_document_ids.extend(result["document_ids"])
    docs_info = vector_store.get()
    bm25_retriever = BM25Retriever.from_texts(
        texts=docs_info["documents"], metadatas=docs_info["metadatas"],
        ids=docs_info["ids"], k=sparse_k, preprocess_func=bilingual_preprocess_func)
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": semantic_k})
    print(f"LOADING DOCUMENTS: Added {len(all_document_ids)} documents to the vector store in total.\n"
          f"LOADING DOCUMENTS: The number of documents of current collection is NOW {len(docs_info['ids'])}.")
    return {"sparse_retriever": bm25_retriever, "semantic_retriever": semantic_retriever,
            "all_document_ids": all_document_ids}
