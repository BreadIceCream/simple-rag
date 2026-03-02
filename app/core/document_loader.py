import os
import time
import uuid

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_unstructured import UnstructuredLoader
from unstructured.partition.api import partition_via_api

from app.config.global_config import global_config
from app.core.chunking import SplitterChain, EXT_TO_LANGUAGE
from app.models.common import LoadDocToVectorStoreResult

# 所有文件类型通用的核心元数据字段
_COMMON_METADATA_KEYS = {
    "file_id", "id", "element_id", "parent_id", "index", "splitters", "document_loader",
    "filename", "file_directory", "filetype", "code_file", "start_index",
    "last_modified", "languages", "category", "category_depth",
}

# 按文件类型额外保留的元数据字段
_EXTRA_METADATA_BY_EXT: dict[str, set[str]] = {
    ".pdf": {"page_number", "links", "link_start_indexes", "image_mime_type"},
    ".docx": {"page_number", "header_footer_type"},
    ".html": {"image_url", "image_mime_type", "link_urls", "link_texts", "link_start_indexes"},
    ".htm": {"image_url", "image_mime_type", "link_urls", "link_texts", "link_start_indexes"},
    # .md, .txt, .csv 仅使用通用字段，无额外字段
}

# 所有 Loader 产出的 Document 都会额外添加的自定义字段
_CUSTOM_METADATA_KEYS = {"file_extension"}


def _get_metadata_keys_for_ext(file_ext: str) -> set[str]:
    """根据文件后缀返回该类型应保留的全部元数据 key 集合"""
    return _COMMON_METADATA_KEYS | _EXTRA_METADATA_BY_EXT.get(file_ext, set()) | _CUSTOM_METADATA_KEYS


def _clean_metadata(doc: Document, keys_to_keep: set[str]) -> None:
    """原地清理 Document 的 metadata，仅保留指定的 key"""
    keys_to_remove = [k for k in doc.metadata if k not in keys_to_keep]
    for k in keys_to_remove:
        del doc.metadata[k]


# ======================== 文件校验 ========================

# 支持的文件后缀
SUPPORTED_CODES_EXTENSIONS = set(EXT_TO_LANGUAGE.keys())
SUPPORTED_OTHER_EXTENSIONS = {".pdf", ".md", ".txt", ".docx", ".html", ".htm", ".csv"}
SUPPORTED_EXTENSIONS = SUPPORTED_CODES_EXTENSIONS | SUPPORTED_OTHER_EXTENSIONS


def _get_file_extension(file_path: str) -> str:
    """获取文件后缀（含点号），如 '.pdf', '.md'"""
    return os.path.splitext(file_path)[1].lower()


def _validate_local_file(file_path: str) -> str:
    """
    校验文件是否存在且后缀受支持，返回文件后缀。
    :raises FileNotFoundError: 文件不存在
    :raises ValueError: 文件类型不受支持
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist or is not a file.")
    file_ext = _get_file_extension(file_path)
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{file_ext}' for file '{file_path}'.")
    return file_ext


# ======================== 文档加载 ========================

async def load_doc_to_vector_store(splitter_chain: SplitterChain, vector_store: VectorStore,
                             file_path: str, file_id: str, is_url: bool = False) -> \
        LoadDocToVectorStoreResult:
    """
    加载文档，并通过 SplitterChain 按文件类型选择分块策略，然后嵌入向量数据库。
    - 代码文件：直接读取文本内容传入 SplitterChain（由 CodeSplitter.create_documents 处理）
    - 其他文件：使用 UnstructuredLoader 加载为 Document 列表，按文件类型保留对应元数据
    :param splitter_chain: 分块器链，根据文件类型自动选择合适的分块策略
    :param vector_store: 向量数据库
    :param file_path: 文档路径，支持 pdf, md, txt, docx, html, csv, 代码文件等
    :param file_id: 文档唯一标识（数据库id）
    :param is_url: 是否为url
    :return: 包含 file_id, error, document_ids 的字典
    """
    print(f"LOADING DOCUMENTS TASK: Loading document <{file_id}>: {file_path}...")
    try:
        start_time = time.time()

        # 1. 文件校验（存在性 + 后缀支持性）
        if file_id is None:
            raise ValueError("file_id cannot be None")
        if is_url:
            file_ext = ".html"
            filename = file_path
        else:
            file_ext = _validate_local_file(file_path)
            filename = os.path.basename(file_path)
        document_loader = "DIRECT READ"
        # 2. 加载文档
        if file_ext in SUPPORTED_CODES_EXTENSIONS:
            # 代码文件：直接读取原始文本，传给 SplitterChain（CodeSplitter 使用 create_documents）
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 直接将原始文本列表传入 splitter_chain
            split_result = splitter_chain.do_split(file_type=file_ext, docs=[content], file_is_raw=True)
            splitters = split_result.splitters
            chunks = split_result.chunks
            # 为分块后的每个 Document 设置相同元数据，其中 file_id 是该文件的唯一标识
            chunks_metadata = {
                "document_loader": document_loader, # 没有使用document_loader加载，直接读取文本，设置为None
                "file_id": file_id,
                "splitters": splitters,
                "filename": filename,
                "file_directory": os.path.dirname(os.path.abspath(file_path)),
                "file_extension": file_ext,
                "code_file": True,
                "last_modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(file_path))),
            }
            for idx, chunk in enumerate(chunks):
                chunk.metadata.update(chunks_metadata)
                # 为每个chunk分配唯一的id，并添加index
                chunk.metadata["id"] = str(uuid.uuid4())
                chunk.metadata["index"] = idx
            loaded_count = 1  # 代码文件视为 1 个原始文档
        else:
            # 非代码文件：使用 UnstructuredLoader 加载，会先返回一个加载后的Document列表
            document_loader = "Unstructured"
            if is_url:
                loader = UnstructuredLoader(
                    web_url=file_path
                )
            else:
                loader = UnstructuredLoader(
                    file_path=file_path
                )
            loaded_docs: list[Document] = loader.load()

            # 清理第一次切分后的元数据：按文件类型保留所需元数据
            keys_to_keep = _get_metadata_keys_for_ext(file_ext)
            for doc in loaded_docs:
                doc.metadata["document_loader"] = document_loader
                doc.metadata["file_extension"] = file_ext
                doc.metadata["code_file"] = False
                doc.metadata["file_id"] = file_id
                _clean_metadata(doc, keys_to_keep)

            # 使用 SplitterChain 分块
            split_result = splitter_chain.do_split(file_type=file_ext, docs=loaded_docs, file_is_raw=False)
            splitters = split_result.splitters
            chunks = split_result.chunks
            for idx, chunk in enumerate(chunks):
                # 为每个chunk分配唯一的id，并添加index
                chunk.metadata["id"] = str(uuid.uuid4())
                chunk.metadata["index"] = idx
                chunk.metadata["splitters"] = splitters
            loaded_count = len(loaded_docs)

        # 3. 嵌入向量数据库，会使用元数据中的id作为向量数据库的id
        document_ids = vector_store.add_documents(documents=chunks)

        end_time = time.time()
        print(
            f"LOADING DOCUMENTS TASK: document <{file_id}> Done! "
            f"Loaded {loaded_count} elements, split into {len(chunks)} chunks, "
            f"cost: {end_time - start_time:.2f}s")
        return LoadDocToVectorStoreResult(file_id=file_id, error=None,
                                          splitters=splitters, document_loader=document_loader,
                                          document_ids=document_ids)
    except Exception as e:
        return LoadDocToVectorStoreResult(file_id=file_id, error=e, splitters=None, document_loader=None, document_ids=None)


# ======================== 检索器初始化 ========================
# todo 改为API导向
# async def load_docs_and_get_retriever(splitter_chain: SplitterChain, vector_store: VectorStore,
#                                       sparse_k: int = 30, semantic_k: int = 30) -> dict[
#     str, BaseRetriever | list[str]]:
#     """
#     交互式加载文档，使用 SplitterChain 分块，设置并返回 sparse_retriever 和 semantic_retriever。
#     文件校验逻辑已内置于 load_doc_to_vector_store 中，此处仅做基本的存在性预检。
#     :param splitter_chain: 分块器链
#     :param vector_store: 向量数据库
#     :param sparse_k: BM25 稀疏检索器返回的文档数量（默认 30）
#     :param semantic_k: 语义检索器返回的文档数量（默认 30）
#     :return: 包含 sparse_retriever, semantic_retriever, all_document_ids 的字典
#     """
#     print(
#         f"LOADING DOCUMENTS: Before adding documents, "
#         f"The number of documents in vector store is {vector_store._collection.count()}")
#     supported_ext_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
#
#     # download NLTK resources for bilingual preprocessing
#     nltk_resource_download_task = asyncio.to_thread(nltk_resource_download)
#     all_document_ids = []
#     load_task_list = []
#     load_task_id = 1
#     load_task_id_to_file_path = {}
#     while True:
#         file_path = input(
#             f"LOADING DOCUMENTS: Enter the file path (supported: {supported_ext_str})\n"
#             f"  Input 'done' to finish: ")
#         if file_path == "done":
#             break
#         elif not os.path.exists(file_path):
#             print("LOADING DOCUMENTS: File not exist!")
#             continue
#         ext = _get_file_extension(file_path)
#         if ext not in SUPPORTED_EXTENSIONS:
#             print(f"LOADING DOCUMENTS: Unsupported file type '{ext}'")
#             continue
#         load_task_id_to_file_path[load_task_id] = file_path
#         task = asyncio.to_thread(load_doc_to_vector_store, splitter_chain, vector_store, file_path, load_task_id)
#         load_task_id += 1
#         load_task_list.append(task)
#
#     # wait for all tasks to complete, get the results and handle the exceptions
#     results = await asyncio.gather(*load_task_list, nltk_resource_download_task)
#     for result in results:
#         if result is None:
#             continue
#         elif result["error"]:
#             print(
#                 f"LOADING DOCUMENTS: Task <{result['file_id']}> failed \n"
#                 f"  file path: {load_task_id_to_file_path[result['file_id']]} \n"
#                 f"  error message: {result['error']}")
#         else:
#             all_document_ids.extend(result["document_ids"])
#     docs_info = vector_store.get()
#     bm25_retriever = BM25Retriever.from_texts(
#         texts=docs_info["documents"], metadatas=docs_info["metadatas"],
#         ids=docs_info["ids"], k=sparse_k, preprocess_func=bilingual_preprocess_func)
#     semantic_retriever = vector_store.as_retriever(search_kwargs={"k": semantic_k})
#     print(f"LOADING DOCUMENTS: Added {len(all_document_ids)} documents to the vector store in total.\n"
#           f"LOADING DOCUMENTS: The number of documents of current collection is NOW {len(docs_info['ids'])}.")
#     return {"sparse_retriever": bm25_retriever, "semantic_retriever": semantic_retriever,
#             "all_document_ids": all_document_ids}
