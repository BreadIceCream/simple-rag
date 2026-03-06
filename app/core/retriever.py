"""
retriever模块。

核心组件：
- EnhancedParentDocumentRetriever: 自定义 ParentDocumentRetriever，增强元数据和返回值
- EnhancedParentDocumentRetrieverFactory: 根据文件类型配置好 parent_splitter 的 ParentDocumentRetriever 工厂
"""
import copy
import math
import string
import uuid
from pathlib import Path
from typing import Any, Optional

import jieba
import nltk
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

from app.config.global_config import global_config
from app.core.chunking import SplitterRegistry
from app.exception.exception import CustomException
from app.models.common import EnhancedPDRetrieverAddDocumentsResult

_RAG_ROOT = Path(__file__).resolve().parent.parent.parent


# ======================== EnhancedParentDocumentRetriever ========================

class EnhancedParentDocumentRetriever(ParentDocumentRetriever):
    """
    增强版 ParentDocumentRetriever，在原版基础上：
    1. add_documents / aadd_documents 返回父文档 ID 列表
    2. 为每个父文档块添加 children_ids 和 children_count 元数据
    3. 为每个父文档块添加 parent_index，为每个子文档块添加 child_index 元数据
    4. 添加获取父/子文档块的方法
    5. 添加级联删除父子文档的方法
    6. 重写_get_relevant_documents和_aget_relevant_documents方法，添加search_kwargs参数。不返回父文档块，直接返回子文档块
    """

    def get_parent_docs(self, parent_doc_ids: list[str]) -> list[tuple[str, Document]]:
        """
        根据父文档 ID 列表获取父文档列表。无需配置parent_splitter
        :param parent_doc_ids:
        :return: 合法的父文档列表(id和Document对象的元组有序列表),会过滤掉不存在的父文档ID
        """
        docs = self.docstore.mget(parent_doc_ids)
        valid_result = []
        invalid_ids = []
        for _id, doc in zip(parent_doc_ids, docs):
            if doc is not None:
                valid_result.append((_id, doc))
            else:
                invalid_ids.append(_id)
        if invalid_ids:
            print(f"Warning: The following parent_doc_ids were not found in docstore and will be skipped:\n{invalid_ids}")
        return valid_result

    def get_child_docs(self, parent_doc_id: str) -> list[Document]:
        """
        根据父文档 ID 获取对应的子文档列表。无需配置parent_splitter
        :param parent_doc_id:
        :return: 子文档列表
        """
        # 验证父文档 ID 是否存在, 并获取父文档Document对象
        result = self.get_parent_docs([parent_doc_id])
        _, parent_doc = result[0]
        # 获取父文档的 children_ids 元数据
        children_ids = parent_doc.metadata.get("children_ids", [])
        if not children_ids:
            raise ValueError(f"Parent document <{parent_doc_id}> has no children_ids metadata or is empty.")
        # 根据 children_ids 获取子文档列表
        children_docs = self.vectorstore.get_by_ids(children_ids)
        if not children_docs:
            raise ValueError(f"No child documents found for parent_doc <{parent_doc_id}>")
        return children_docs

    def get_parent_docs_from_children(self, children_docs: list[Document]):
        """
        根据子文档列表获取对应的父文档列表，使用父类_get_relevant_documents方法中的逻辑。
        无需配置parent_splitter
        :param children_docs:
        :return: 父文档有序列表
        """
        # We do this to maintain the order of the IDs that are returned
        ids = []
        for d in children_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        return [d for d in docs if d is not None]

    async def aget_parent_docs_from_children(self, children_docs: list[Document]):
        """
        根据子文档列表获取对应的父文档列表，使用父类_aget_relevant_documents方法中的逻辑。
        :param children_docs:
        :return: 父文档有序列表
        """
        # We do this to maintain the order of the IDs that are returned
        ids = []
        for d in children_docs:
            if self.id_key in d.metadata and d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = await self.docstore.amget(ids)
        return [d for d in docs if d is not None]

    def delete_docs_cascade(self, parent_doc_ids: list[str]):
        """
        根据父文档 ID 列表级联删除父文档和对应的子文档。无需配置parent_splitter
        :param parent_doc_ids:
        :return:
        """
        # 获取合法的父文档列表
        valid_parents = self.get_parent_docs(parent_doc_ids)
        if not valid_parents:
            print("DELETE CHUNKS: No valid parent_doc_ids found for deletion. No documents will be deleted.")
            return

        # 收集所有要删除的父文档 ID 和子文档 ID
        all_parent_ids_to_delete = []
        all_child_ids_to_delete = []
        for parent_id, parent_doc in valid_parents:
            all_parent_ids_to_delete.append(parent_id)
            children_ids = parent_doc.metadata.get("children_ids", [])
            all_child_ids_to_delete.extend(children_ids)

        # 删除子文档
        if all_child_ids_to_delete:
            del_child_result = self.vectorstore.delete(all_child_ids_to_delete)
            print(f"DELETE CHUNKS: Deleted <{len(all_child_ids_to_delete)}> child documents. Result <{del_child_result}>.")
        else:
            print("DELETE CHUNKS: No child documents found to delete.")

        # 删除父文档
        self.docstore.mdelete(all_parent_ids_to_delete)
        print(f"DELETE CHUNKS: Deleted <{len(all_parent_ids_to_delete)}> parent documents.")


    def _split_docs_for_adding(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        *,
        add_to_docstore: bool = True,
    ) -> tuple[list[Document], list[tuple[str, Document]]]:
        """
        重写父类方法，增加：
        - 为子文档生成并设置 id 属性，添加 id（后续EnsembleRetriever RRF使用）, child_index 元数据
        - 父文档设置 id 属性，添加 id, children_ids, children_count, parent_index 元数据
        :param documents:
        :param ids:
        :param add_to_docstore:
        :return: children_docs(全部子文档列表), full_docs(父文档id和 Document 对象的元组列表)
        """
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        if ids is None:
            parent_doc_ids = [str(uuid.uuid4()) for _ in documents]
            if not add_to_docstore:
                msg = "If IDs are not passed in, `add_to_docstore` MUST be True"
                raise ValueError(msg)
        else:
            if len(documents) != len(ids):
                msg = (
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
                raise ValueError(msg)
            parent_doc_ids = ids

        children_docs = []
        full_docs = [] # 父文档列表，id和 Document 对象的元组列表
        for parent_index, parent_doc in enumerate(documents):
            parent_id = parent_doc_ids[parent_index]
            sub_docs = self.child_splitter.split_documents([parent_doc])

            # 为子文档筛选元数据字段（与原版一致）
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }

            # 为每个子文档生成 id、设置 parent id、child_index
            children_ids = []
            for child_index, _doc in enumerate(sub_docs):
                child_id = str(uuid.uuid4())
                _doc.id = child_id
                _doc.metadata["id"] = child_id
                _doc.metadata[self.id_key] = parent_id
                _doc.metadata["child_index"] = child_index
                children_ids.append(child_id)

            # 为父文档设置id属性，并添加增强元数据
            parent_doc.id = parent_id
            parent_doc.metadata["id"] = parent_id
            parent_doc.metadata["parent_index"] = parent_index
            parent_doc.metadata["children_ids"] = children_ids
            parent_doc.metadata["children_count"] = len(children_ids)

            children_docs.extend(sub_docs)
            full_docs.append((parent_id, parent_doc))

        return children_docs, full_docs

    def add_documents(
        self,
        documents: list[Document],
        ids: Optional[list[str]] = None,
        add_to_docstore: bool = True,
        **kwargs: Any,
    ) -> tuple[list[str], int]:
        """
        添加文档到 docstore 和 vectorstore。
        Splitter切分时会添加text_splitters元数据。
        :return: (父文档 ID 有序列表, 添加的子文档总数)
        """
        children_docs, full_docs = self._split_docs_for_adding(
            documents,
            ids,
            add_to_docstore=add_to_docstore,
        )
        self.vectorstore.add_documents(children_docs, **kwargs)
        if add_to_docstore:
            self.docstore.mset(full_docs)
        return [_id for _id, _ in full_docs], len(children_docs)

    async def aadd_documents(
        self,
        documents: list[Document],
        ids: Optional[list[str]] = None,
        add_to_docstore: bool = True,
        **kwargs: Any,
    ) -> tuple[list[str], int]:
        """
        异步添加文档到 docstore 和 vectorstore
        Splitter切分时会添加text_splitters元数据。
        :return: (父文档 ID 有序列表, 添加的子文档总数)
        """
        children_docs, full_docs = self._split_docs_for_adding(
            documents,
            ids,
            add_to_docstore=add_to_docstore,
        )
        await self.vectorstore.aadd_documents(children_docs, **kwargs)
        if add_to_docstore:
            await self.docstore.amset(full_docs)
        return [_id for _id, _ in full_docs], len(children_docs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        search_kwargs: Optional[dict] = None,
    ) -> list[Document]:
        """
        重写父类方法，直接返回子文档列表
        :param query:
        :param run_manager:
        :param search_kwargs: 可选，搜索参数字典，会合并全局搜索参数，调用时传入的 search_kwargs 优先级高于全局搜索参数
                              参考链接 https://docs.trychroma.com/docs/querying-collections/metadata-filtering
        :return: 子文档有序列表
        """
        current_search_kwargs = copy.deepcopy(self.search_kwargs) if hasattr(self, "search_kwargs") else {}
        if search_kwargs:
            current_search_kwargs.update(search_kwargs)
        print(f"SEARCHING VECTORSTORE: search_type={self.search_type}, search_kwargs={current_search_kwargs}")

        # 执行和父类方法相同的向量检索逻辑，但直接返回子文档列表
        if self.search_type == SearchType.mmr:
            sub_docs = self.vectorstore.max_marginal_relevance_search(
                query,
                **current_search_kwargs,
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            sub_docs_and_similarities = (
                self.vectorstore.similarity_search_with_relevance_scores(
                    query,
                    **current_search_kwargs,
                )
            )
            sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]
        else:
            sub_docs = self.vectorstore.similarity_search(query, **current_search_kwargs)
        return sub_docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        search_kwargs: Optional[dict] = None,
    ) -> list[Document]:
        """
        重写父类方法，直接返回子文档列表
        :param query:
        :param run_manager:
        :param search_kwargs: 可选，搜索参数字典，会合并全局搜索参数，调用时传入的 search_kwargs 优先级高于全局搜索参数
        :return: 子文档有序列表
        """
        current_search_kwargs = copy.deepcopy(self.search_kwargs) if hasattr(self, "search_kwargs") else {}
        if search_kwargs:
            current_search_kwargs.update(search_kwargs)
        print(f"SEARCHING VECTORSTORE ASYNC: search_type={self.search_type}, search_kwargs={current_search_kwargs}")

        # 执行和父类方法相同的向量检索逻辑，但直接返回子文档列表
        if self.search_type == SearchType.mmr:
            sub_docs = await self.vectorstore.amax_marginal_relevance_search(
                query,
                **current_search_kwargs,
            )
        elif self.search_type == SearchType.similarity_score_threshold:
            sub_docs_and_similarities = (
                await self.vectorstore.asimilarity_search_with_relevance_scores(
                    query,
                    **current_search_kwargs,
                )
            )
            sub_docs = [sub_doc for sub_doc, _ in sub_docs_and_similarities]
        else:
            sub_docs = await self.vectorstore.asimilarity_search(
                query,
                **current_search_kwargs,
            )
        return sub_docs



# ======================== EnhancedParentDocumentRetrieverFactory ========================

class EnhancedParentDocumentRetrieverFactory:
    """
    EnhancedParentDocumentRetriever 工厂(单例)。
    特别注意, 项目中全部使用同一个 EnhancedParentDocumentRetriever 实例，
    因此在添加文档前必须先调用 configure_pd_for_file_type()，否则可能使用上次的 parent_splitter 配置，导致分块错误。
    建议使用工厂的 add_documents / aadd_documents 方法添加文档。
    职责：
    1. 初始化 EnhancedParentDocumentRetriever：child_splitter 固定为 RecursiveCharacterTextSplitter，所有文件类型共用，从 SplitterRegistry 获取。
    2. 添加文档：根据文件类型，从 SplitterRegistry 获取对应的 parent_splitter，配置好 retriever.parent_splitter 并执行添加。
    """
    _pd_retriever: EnhancedParentDocumentRetriever | None = None

    @classmethod
    def init(cls, vectorstore, docstore) -> EnhancedParentDocumentRetriever:
        """
        初始化 EnhancedParentDocumentRetriever
        需先初始化SplitterRegistry，要从SplitterRegistry中获取ChildSplitter
        """
        if cls._pd_retriever is not None:
            print("INIT PARENT DOCUMENT RETRIEVER: Instance already exists, returning cached instance.")
            return cls._pd_retriever

        # child_splitter 固定：用于将父文档切成小块做向量检索
        child_splitter = SplitterRegistry.get_child_splitter()

        # 获取全局搜索参数，默认使用 MMR 检索
        cls._pd_retriever = EnhancedParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            search_type=SearchType.mmr, # 默认使用 MMR 检索
            id_key="parent_id" # 父文档ID在子文档metadata中的键名，默认为parent_id
            # parent_splitter 不在此设置，由 configure_pd_for_file_type() 动态设置
        )

        print("INIT PARENT DOCUMENT RETRIEVER: Initialized successfully.")
        return cls._pd_retriever

    @classmethod
    def _configure_pd_for_file_type(cls, file_type: str, use_parent: bool = True) -> \
            tuple[str, EnhancedParentDocumentRetriever]:
        """
        根据文件类型配置 EnhancedParentDocumentRetriever 的 parent_splitter 并返回。
        :param file_type: 文件后缀
        :param use_parent: 是否使用父分块器，False 则 parent_splitter=None（原始文档整体作为父文档）, splitter_name = None
        :return: (parent_splitter_name, EnhancedParentDocumentRetriever)
        """
        retriever = cls.get_instance()
        if use_parent:
            splitter_registry = SplitterRegistry.get_instance()
            splitter_name, parent_splitter = splitter_registry.get_splitter(file_type)
            retriever.parent_splitter = parent_splitter
        else:
            splitter_name = None
            retriever.parent_splitter = None
        return splitter_name, retriever

    @classmethod
    def add_documents(cls, file_type: str, documents: list[Document], ids: Optional[list[str]] = None,
                      use_parent: bool = True, add_to_docstore: bool = True, **kwargs: Any,) -> \
            EnhancedPDRetrieverAddDocumentsResult:
        """
        封装 EnhancedParentDocumentRetriever 的 add_documents 方法，
        先调用 configure_pd_for_file_type 配置，再使用单例 retriever 添加文档。
        :param file_type: 文件后缀
        :param documents:
        :param ids: 可选，为None会自动生成父文档ID
        :param use_parent: 是否使用父分块器，False 则 parent_splitter=None（原始文档整体作为父文档）, splitter_name = None
        :param add_to_docstore:
        :param kwargs:
        :return: EnhancedPDRetrieverAddDocumentsResult
        """
        parent_splitter_name, pd_retriever = cls._configure_pd_for_file_type(file_type, use_parent)
        parent_doc_ids, children_count = pd_retriever.add_documents(documents, ids, add_to_docstore, **kwargs)
        return EnhancedPDRetrieverAddDocumentsResult(
            parent_doc_ids=parent_doc_ids,
            children_count=children_count,
            parent_splitter_name=parent_splitter_name
        )

    @classmethod
    async def aadd_documents(cls, file_type: str, documents: list[Document], ids: Optional[list[str]] = None,
                             use_parent: bool = True, add_to_docstore: bool = True,**kwargs: Any,) -> \
            EnhancedPDRetrieverAddDocumentsResult:
        """
        封装 EnhancedParentDocumentRetriever 的 aadd_documents 方法，
        :param file_type:
        :param documents:
        :param ids:
        :param use_parent:
        :param add_to_docstore:
        :param kwargs:
        :return:
        """
        parent_splitter_name, pd_retriever = cls._configure_pd_for_file_type(file_type, use_parent)
        parent_doc_ids, children_count = await pd_retriever.aadd_documents(documents, ids, add_to_docstore, **kwargs)
        return EnhancedPDRetrieverAddDocumentsResult(
            parent_doc_ids=parent_doc_ids,
            children_count=children_count,
            parent_splitter_name=parent_splitter_name
        )

    @classmethod
    def get_instance(cls) -> EnhancedParentDocumentRetriever:
        """获取 EnhancedParentDocumentRetriever 单例。"""
        if cls._pd_retriever is None:
            raise RuntimeError("Retriever has not been initialized. Call EnhancedParentDocumentRetrieverFactory.init() first.")
        return cls._pd_retriever


class RetrievalParams:
    """
    检索使用的参数
    """
    def __init__(self, bm25_k: int, vector_k: int, vector_fetch_k: int, rrf_k: int, final_k: int):
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.vector_fetch_k = vector_fetch_k
        self.rrf_k = rrf_k
        self.final_k = final_k




class _NLPPreprocessor:
    """
    NLP 预处理工具类（非 Pydantic 模型）。
    独立于 HybridPDRetriever（Pydantic 子类），避免 _ 前缀属性被 Pydantic 包装为 ModelPrivateAttr。
    """
    _stop_words: set[str] = set()
    _punctuation: set[str] = set()
    _lemmatizer: WordNetLemmatizer | None = None
    _initialized: bool = False

    @classmethod
    def init(cls):
        """初始化 NLTK/jieba 预处理资源（只需调用一次）"""
        if cls._initialized:
            print("NLP PREPROCESSOR: Already initialized.")
            return
        print("NLP PREPROCESSOR: Downloading NLTK resources...")
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")
        cls._stop_words = set(stopwords.words('english') + stopwords.words('chinese'))
        cls._punctuation = set(string.punctuation) | set("，。！？【】（）《》""''：；、—…—")
        cls._lemmatizer = WordNetLemmatizer()
        cls._initialized = True
        print("NLP PREPROCESSOR: Initialized successfully.")

    @classmethod
    def _is_english_word(cls, s: str) -> bool:
        return all(ord(c) < 128 for c in s)

    @classmethod
    def bilingual_preprocess_func(cls, text: str) -> list[str]:
        """中英文混合预处理函数，供 BM25Retriever 使用"""
        text = text.lower()
        tokens = jieba.lcut(text, cut_all=False)
        processed_tokens = []
        for token in tokens:
            if token in cls._stop_words or token in cls._punctuation:
                continue
            if cls._is_english_word(token):
                token = cls._lemmatizer.lemmatize(token)
            if len(token) > 1:
                processed_tokens.append(token)
        return processed_tokens


class HybridPDRetriever(EnsembleRetriever):
    """
    混合ParentDocument检索器: 使用BM25Retriever做关键词检索，使用EnhancedParentDocumentRetriever做向量检索，并进行RRF融合。
    NOTE：需先通过 HybridPDRetrieverFactory.init() 创建实例
    """

    # ==== 实例级别方法 ====
    def get_file_ids(self) -> set[str]:
        """获取当前使用的文件ID集合"""
        return self._file_ids

    def _get_bm25_retriever(self) -> BM25Retriever:
        if self._bm25_retriever is None:
            raise RuntimeError("BM25Retriever has not been initialized. "
                               "Call reset_file_ids() first.")
        return self._bm25_retriever

    def _get_pd_retriever(self) -> EnhancedParentDocumentRetriever:
        if self._pd_retriever is None:
            raise RuntimeError("EnhancedParentDocumentRetriever has not been initialized. "
                               "Call reset_file_ids() first.")
        return self._pd_retriever

    def _get_retrieval_params(self) -> RetrievalParams:
        if self._retrieval_params is None:
            raise RuntimeError("Retrieval parameters have not been initialized. "
                               "Call reset_file_ids() first.")
        return self._retrieval_params


    def _prepare_retrieval(self, input: str) -> RetrievalParams:
        """
        invoke / ainvoke 的公共前置逻辑：参数校验、配置两个 retriever 的检索参数、
        设置 self.retrievers，返回 RetrievalParams。
        """
        if not input:
            print("HYBRID PD RETRIEVER INVOKE: Empty query provided.")
            raise CustomException(code=400, message="查询语句不能为空，请输入查询语句")
        file_ids = self.get_file_ids()
        if not file_ids:
            print("HYBRID PD RETRIEVER INVOKE: No file_ids configured.")
            raise CustomException(code=400, message="未选择任何文档，请先选择文档进行检索")

        bm25_retriever = self._get_bm25_retriever()
        pd_retriever = self._get_pd_retriever()
        params = self._get_retrieval_params()

        # 配置 BM25
        bm25_retriever.k = params.bm25_k

        # 配置语义向量检索
        search_kwargs = {
            "filter": {
                "file_id": {"$in": list(file_ids)}
            },
            "k": params.vector_k,
        }
        if pd_retriever.search_type == SearchType.mmr:
            search_kwargs["fetch_k"] = params.vector_fetch_k
        pd_retriever.search_kwargs = search_kwargs

        # 设置 EnsembleRetriever 的 retrievers
        self.retrievers = [bm25_retriever, pd_retriever]

        print(f"HYBRID PD RETRIEVER INVOKE: query='{input[:50]}...', "
              f"bm25_k={params.bm25_k}, vector_k={params.vector_k}, vector_fetch_k={params.vector_fetch_k}, rrf_k={params.rrf_k}, "
              f"rrf_c={self.c}, final_k={params.final_k}, file_ids_count={len(file_ids)}")

        return params


    def reset_file_ids(self, file_ids: set[str], child_docs: list[Document] | None = None) -> int:
        """
        重置已选择的文件ID集合，并创建对应的BM25Retriever实例（使用child文档，BM25Plus变体），
        重新设置search_kwargs以使用新的child文档集合进行检索。
        如果file_ids为空，则重置两个Retriever为None
        :param file_ids: 数据库中的文档ID列表
        :param child_docs: 可选，file_ids对应的child文档列表.
        :return: 0表示成功
        """
        self._file_ids = file_ids
        if not file_ids:
            self._bm25_retriever = None
            self._pd_retriever = None
            self._retrieval_params = None
            print("HYBRID PD RETRIEVER: No file_ids provided, "
                  "BM25Retriever and EnhancedParentDocumentRetriever have been reset to None.")
            if child_docs:
                print("HYBRID PD RETRIEVER: Warning: child_docs provided but file_ids is empty, "
                      "child_docs will be ignored.")
            return 0
        if not child_docs:
            raise ValueError("HYBRID PD RETRIEVER: child_docs must be provided when file_ids is not empty.")

        # 创建BM25Retriever实例，使用BM25Plus
        self._bm25_retriever = BM25Retriever.from_documents(
            child_docs,
            bm25_variant="plus",
            preprocess_func=_NLPPreprocessor.bilingual_preprocess_func,
        )

        # 获取EnhancedParentDocumentRetriever（仅用于检索）
        self._pd_retriever = EnhancedParentDocumentRetrieverFactory.get_instance()

        # 设置检索参数
        k = math.ceil(max(40, len(child_docs) // 8))
        self._retrieval_params = RetrievalParams(
            bm25_k=k,
            vector_k=math.ceil(k * 0.8),
            vector_fetch_k=math.ceil(k * 0.8 * 3),
            rrf_k=math.ceil(10 + k * 0.2),
            final_k=self._retriever_config.get("final_k", 8)
        )

        return 0


    def merge_continuous_parent_docs(self, parent_docs: list[Document]) -> list[Document]:
        """
        合并连续的父文档块，如果它们的 parent_index 是连续的且属于同一个文件（file_id相同），则合并它们的文本内容。
        合并到连续块中第一个父文档的文本中，保持顺序。
        :param parent_docs:
        :return:
        """
        # todo

    def invoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        同步混合检索。
        RRF 融合后返回 rrf_k 个子 Document，获取对应父文档列表，执行可选的 rerank 后返回 final_k 个父Document。
        """
        params = self._prepare_retrieval(input)
        fused_results = super().invoke(input, config, **kwargs)
        # 获取 rrf_k 个子 Document 对应的父文档列表，保持顺序
        children_docs = fused_results[:params.rrf_k]
        parent_docs = self._get_pd_retriever().get_parent_docs_from_children(children_docs)
        # 可选的 rerank 逻辑
        if not self._retriever_config.get("reranker", {}).get("enabled", False):
            print("HYBRID PD RETRIEVER INVOKE: Reranker is disabled, skipping reranking step.")
            return parent_docs[:params.final_k]
        print("HYBRID PD RETRIEVER INVOKE: Reranker is enabled, performing reranking step...")
        # todo 执行重排序
        return parent_docs[:params.final_k]

    async def ainvoke(
        self,
        input: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """
        异步混合检索：RRF 融合后返回 rrf_k 个子 Document。
        """
        params = self._prepare_retrieval(input)
        fused_results = await super().ainvoke(input, config, **kwargs)
        # 获取 rrf_k 个子 Document 对应的父文档列表，保持顺序
        children_docs = fused_results[:params.rrf_k]
        parent_docs = await self._get_pd_retriever().aget_parent_docs_from_children(children_docs)
        # 可选的 rerank 逻辑
        if not self._retriever_config.get("reranker", {}).get("enabled", False):
            print("HYBRID PD RETRIEVER AINVOKE: Reranker is disabled, skipping reranking step.")
            return parent_docs[:params.final_k]
        print("HYBRID PD RETRIEVER AINVOKE: Reranker is enabled, performing reranking step...")
        # todo 执行重排序
        return parent_docs[:params.final_k]



# ======================== HybridPDRetrieverFactory ========================

class HybridPDRetrieverFactory:
    """
    HybridPDRetriever 工厂（单例）。
    职责：
    1. 初始化 NLP 资源和 HybridPDRetriever 单例实例
    2. 对外提供 get_instance() 获取单例
    """
    _instance: HybridPDRetriever | None = None

    @classmethod
    def init(cls) -> HybridPDRetriever:
        """
        初始化 HybridPDRetriever 单例。
        会先初始化 NLP 资源，再创建实例。
        """
        if cls._instance is not None:
            print("INIT HYBRID PD RETRIEVER: Instance already exists, returning cached instance.")
            return cls._instance

        # 初始化类级别 NLP 资源
        _NLPPreprocessor.init()

        # 创建实例，初始时 retrievers 为空列表
        cls._instance = HybridPDRetriever(
            retrievers=[],  # invoke 时动态设置
            weights=[0.5, 0.5],  # BM25 和向量检索的权重
            id_key="id",  # 用于 RRF 去重的文档唯一标识 metadata key
        )
        # 初始化实例级别属性（Pydantic model 不通过 __init__ 设置非 field 属性）
        cls._instance._file_ids = set()
        cls._instance._bm25_retriever = None
        cls._instance._pd_retriever = None
        cls._instance._retrieval_params = None
        cls._instance._retriever_config = global_config.get("retriever", {})  # 可选的 rerank 开关，默认为 False

        print("INIT HYBRID PD RETRIEVER: Initialized successfully.")
        return cls._instance

    @classmethod
    def get_instance(cls) -> HybridPDRetriever:
        """获取 HybridPDRetriever 单例。未初始化时抛出异常。"""
        if cls._instance is None:
            raise RuntimeError("HybridPDRetriever has not been initialized. "
                               "Call HybridPDRetrieverFactory.init() first.")
        return cls._instance

