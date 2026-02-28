import asyncio
import os
from typing import Any

import chromadb
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

from app.config import load_env
from app.core.embeddings import init_embedding_model
from app.core.chunking import init_text_splitter
from app.core.retriever import init_vector_store, vector_store_use_or_create_collection, init_hybrid_retriever, init_compression_retriever
from app.core.document_loader import load_docs_and_get_retriever
from app.core.graph import init_tool_infos


# 创建LLM
async def init_llm():
    print("INIT LLM: Initializing LLM...")
    llm = ChatOpenAI(model=os.environ["MODEL_NAME"], temperature=0.2)
    if os.environ["MODEL_PROVIDER"] != "openai":
        llm.openai_api_base = os.environ["OPENAI_API_BASE"]
    return llm.configurable_fields(temperature=ConfigurableField(
        id="temperature",
        name="Runtime Temperature",
        description="The runtime temperature provided by user"
    ))


async def init_rag_application() -> dict[str, Any]:
    init_result = {}
    load_env()

    llm_init_task = asyncio.create_task(init_llm())
    text_splitter_init_task = asyncio.create_task(init_text_splitter())
    tool_init_task = asyncio.create_task(init_tool_infos())

    init_result["embeddings"] = await asyncio.to_thread(init_embedding_model)

    init_result["client"] = chromadb.PersistentClient("./chroma_langchain_db")
    init_result["collection"] = vector_store_use_or_create_collection(init_result["client"])
    vector_store_init_task = asyncio.create_task(
        init_vector_store(init_result["embeddings"], init_result["client"], init_result["collection"]))
    results = await asyncio.gather(text_splitter_init_task, vector_store_init_task)
    init_result["text_splitter"] = results[0]
    init_result["vector_store"] = results[1]

    retrievers_and_docs_ids = await asyncio.create_task(
        load_docs_and_get_retriever(init_result["text_splitter"], init_result["vector_store"]))
    init_result["docs_ids"] = retrievers_and_docs_ids["all_document_ids"]
    hybrid_retriever = init_hybrid_retriever(sparse_retriever=retrievers_and_docs_ids["sparse_retriever"],
                                             semantic_retriever=retrievers_and_docs_ids["semantic_retriever"])
    compression_retriever = init_compression_retriever(base_retriever=hybrid_retriever)
    init_result["hybrid_retriever"] = hybrid_retriever
    init_result["compression_retriever"] = compression_retriever

    results = await asyncio.gather(llm_init_task, tool_init_task)
    init_result["llm"] = results[0]
    return init_result
