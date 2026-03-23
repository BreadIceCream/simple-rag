"""
retriever模块。

核心组件：
- EnhancedParentDocumentRetriever: 自定义 ParentDocumentRetriever，增强元数据和返回值
- EnhancedParentDocumentRetrieverFactory: 根据文件类型配置好 parent_splitter 的 ParentDocumentRetriever 工厂
"""
import asyncio
import copy
import math
import os
import string
import uuid
from pathlib import Path
from typing import Any, Optional, Dict, List

import jieba
import nltk
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain_classic.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain_classic.retrievers.multi_vector import SearchType
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig, patch_config, ensure_config
from langchain_elasticsearch import ElasticsearchRetriever
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from typing_extensions import override

from app.config.global_config import global_config
from app.core.chunking import SplitterRegistry
from app.core.reranker import RerankerFactory
from app.exception.exception import CustomException
from app.models.common import EnhancedPDRetrieverAddDocumentsResult

_RAG_ROOT = Path(__file__).resolve().parent.parent.parent


# ======================== ElasticSearchRetriever ========================

class EnhancedElasticsearchRetriever(ElasticsearchRetriever):
    """
    增强版 ElasticsearchRetriever，在原版基础上：
    1. 支持动态构建查询体，添加 search_kwargs 参数，支持多种类型的元数据过滤（term、terms、range、exists）
    2. bulk_index_documents 方法将子文档的 id 设置为 ES 的系统 _id
    """
    def __init__(self, **kwargs):
        kwargs["document_mapper"] = self._docs_mapper
        kwargs["body_func"] = self._filter_query_func
        super().__init__(**kwargs)
        self._search_kwargs = {}

    @staticmethod
    def _docs_mapper(hit: Dict[str, Any]) -> Document:
        """
        将 Elasticsearch 的搜索结果(hit)转换为 Langchain Document 对象
        """
        # 1. 获取 ES 的原始数据内容
        source = hit.get("_source", {})

        # 2. 提取正文内容 (page_content)
        content = source.get("page_content", "")

        # 3. 提取元数据 (metadata)
        metadata = source.get("metadata", {}).copy()

        # 4. 返回标准的 Langchain Document
        return Document(
            page_content=content,
            metadata=metadata,
            id=source.get("id"),
            type=source.get("type"),
        )

    def _filter_query_func(self, query: str) -> Dict[str, Any]:
        """
        动态构建 Elasticsearch 查询体
        :param query: 用户输入的搜索词
        _search_kwargs: k 和 元数据过滤参数，元数据过滤参数会自动拼接metadata, 例如：
            {
                "k": 10
                "file_name": "参考资料.md",
                "file_extension": [".md", ".pdf"],
                "children_index": {"gte": 0, "lte": 50},
                "is_url": False
            }
        """
        k = self._search_kwargs.get("k", 10) if self._search_kwargs else 10

        # 基础结构：must 处理相关性评分，filter 处理硬性过滤
        bool_query = {
            "must": [
                {
                    "match": {
                        "page_content": {
                            "query": query,
                            "analyzer": "ik_smart"  # 确保使用中文分词
                        }
                    }
                }
            ],
            "filter": []
        }

        if self._search_kwargs:
            for field, value in self._search_kwargs.items():
                if field == "k": continue  # 跳过k参数
                field = "metadata." + field
                # 1. 如果是列表，使用 terms 匹配 (多选一)
                if isinstance(value, list):
                    bool_query["filter"].append({"terms": {field: value}})

                # 2. 如果是字典，检查是否为范围查询
                elif isinstance(value, dict):
                    # 支持 range: {"gt": x, "lt": y}
                    if any(k in value for k in ["gt", "lt", "gte", "lte"]):
                        bool_query["filter"].append({"range": {field: value}})
                    # 支持 exists 查询 (可选)
                    elif value.get("exists"):
                        bool_query["filter"].append({"exists": {"field": field}})

                # 3. 普通值，使用 term 精确匹配
                else:
                    bool_query["filter"].append({"term": {field: value}})

        enhanced_query = {
            "size": k,
            "query": {"bool": bool_query}
        }
        print(f"ELASTICSEARCH query: {enhanced_query}")
        return enhanced_query

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        bm25_search_kwargs: Dict[str, Any],
        **kwargs
    ) -> List[Document]:
        """
        重写父类方法，额外接受 bm25_search_kwargs 参数进行动态设置
        """
        self._search_kwargs = bm25_search_kwargs
        return super()._get_relevant_documents(query, run_manager=run_manager)


class ElasticSearchFactory:
    """
    ElasticSearch相关操作封装
    1. init 初始化
    3. bulk_index_documents 加载新文档到 ES，供 BM25 检索使用
    4. 相关删除方法
    """

    _es_client: Elasticsearch | None = None
    _index_name: str = "simple-rag"
    _es_retriever: ElasticsearchRetriever | None = None

    @classmethod
    def init(cls):
        """
        初始化，创建 Elasticsearch 客户端，创建index，设置 Retriever
        :return:
        """
        if cls._es_client is not None:
            print("ELASTICSEARCH: Already initialized...")
            return
        es_config = global_config.get("elasticsearch", {})
        es_url = es_config.get("url")
        if not es_url:
            raise RuntimeError("Elasticsearch URL not configured, please set in config.yml.")
        cls._es_client = Elasticsearch(
            [es_url],
            basic_auth=(es_config.get("username"), os.environ["ELASTIC_PASSWORD"]),
            ca_certs=str(_RAG_ROOT / "http_ca.crt"),
        )
        info = cls._es_client.info()
        print(f"ELASTICSEARCH: Client Initialized. Info {info}.")
        cls._create_child_index()
        cls._es_retriever = EnhancedElasticsearchRetriever(
            client=cls._es_client,
            index_name=cls._index_name,
        )
        print("ELASTICSEARCH: Initialized...")

    @classmethod
    def _create_child_index(cls):
        mappings = {
            "properties": {
                # 1. 基础字段
                "id": {"type": "keyword"},  # 文档唯一标识符
                "type": {"type": "keyword"},  # 区分 Document 类型

                # 2. 核心内容字段：用于 BM25 关键词检索
                "page_content": {
                    "type": "text",
                    "analyzer": "ik_smart",  # 强烈建议安装并使用 IK 分词器处理中文
                    "search_analyzer": "ik_smart"
                },

                # 3. 元数据字段：全部用于过滤和精准匹配
                "metadata": {
                    "properties": {
                        # 标识类 (用于过滤/关联)
                        "id": {"type": "keyword"},
                        "file_id": {"type": "keyword"},
                        "parent_id": {"type": "keyword"},  # 关键：用于回溯父文档

                        # 路径与文件信息 (用于范围缩小)
                        "path": {"type": "keyword"},
                        "file_name": {"type": "keyword"},
                        "file_directory": {"type": "keyword"},
                        "file_extension": {"type": "keyword"},
                        "document_loader": {"type": "keyword"},

                        # 逻辑与数值 (用于过滤/排序)
                        "is_url": {"type": "boolean"},
                        "child_index": {"type": "integer"},  # 记录在父文档中的顺序
                        "last_modified": {"type": "date"},

                        # 集合类 (ES 自动处理数组)
                        "text_splitters": {"type": "keyword"}
                    }
                }
            }
        }
        if cls._es_client.indices.exists(index=cls._index_name):
            print(f"ELASTICSEARCH: {cls._index_name} index already exists...")
        else:
            cls._es_client.indices.create(index=cls._index_name, mappings=mappings)
            print(f"ELASTICSEARCH: {cls._index_name} index created...")

    @classmethod
    def bulk_index_documents(cls, docs: list[Document]):
        """
        langchain_docs: 一个包含多个 Document 对象的列表
        """
        actions = []

        for doc in docs:
            # doc 结构参考你提供的 JSON
            action = {
                "_index": cls._index_name,
                "_id": doc.id,  # 【核心操作】将子文档的 id 设置为 ES 的系统 _id
                "_source": {
                    "id": doc.id,
                    "type": doc.type,
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
            }
            actions.append(action)

        # 执行批量写入
        success, failed = bulk(cls._es_client, actions)
        print(f"ELASTICSEARCH: bulk documents. Total: {len(docs)}, success: {success}, fail: {failed}")

    @classmethod
    def delete_by_parent_ids(cls, parent_doc_ids: list[str]) -> int:
        """
        根据父文档id进行删除
        :param parent_doc_ids:
        :return: 0 表示成功
        """
        if not parent_doc_ids:
            print("ELASTICSEARCH: Empty parent ids. Skip delete...")
            return 0
            # 构造查询体：使用 terms 查询匹配列表中的任意一个 ID
        try:
            # 执行按查询删除，分批次删除，每次删除1000条
            batch_size = 1000
            total_deleted = 0

            for start in range(0, len(parent_doc_ids), batch_size):
                batch_ids = parent_doc_ids[start: start + batch_size]
                query = {
                    "query": {
                        "terms": {
                            "metadata.parent_id": batch_ids
                        }
                    }
                }
                response = cls._es_client.delete_by_query(
                    index=cls._index_name,
                    body=query,
                    refresh=True
                )

                deleted = response.get("deleted", 0)
                conflicts = response.get("version_conflicts", 0)
                total_deleted += deleted

                print(
                    f"ELASTICSEARCH: batch delete done. "
                    f"range=[{start}, {start + len(batch_ids) - 1}], "
                    f"deleted={deleted}, conflicts={conflicts}"
                )

            print(f"ELASTICSEARCH: delete success! Total deleted: {total_deleted}")
            return 0

        except Exception as e:
            print(f"ELASTICSEARCH: delete error. {e}")
            raise e

    @classmethod
    def get_es_client(cls):
        """
        获取 Elasticsearch 客户端
        :return: Elasticsearch 客户端实例
        """
        if cls._es_client is None:
            raise RuntimeError("ELASTICSEARCH: client not initialized. Call ElasticSearchFactory.init() first.")
        return cls._es_client

    @classmethod
    def get_es_retriever(cls):
        """
        获取 Elasticsearch Retriever
        :return:
        """
        if cls._es_retriever is None:
            raise RuntimeError("ELASTICSEARCH: retriever not initialized. Call ElasticSearchFactory.init() first.")
        return cls._es_retriever




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
        :return: 合法的父文档列表(id和Document对象的元组有序列表, [(id, doc), ...]),会过滤掉不存在的父文档ID
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
            print(
                f"Warning: The following parent_doc_ids were not found in docstore and will be skipped:\n{invalid_ids}")
        return valid_result

    def get_child_docs(self, parent_doc_ids: list[str]) -> list[Document]:
        """
        根据父文档 ID 获取对应的子文档列表。无需配置parent_splitter
        :param parent_doc_ids:
        :return: 子文档列表
        """
        # 验证父文档 ID 是否存在, 并获取父文档Document对象
        result = self.get_parent_docs(parent_doc_ids)
        parent_docs = [doc for _, doc in result]
        # 获取父文档的 children_ids 元数据
        children_ids = []
        for parent_doc in parent_docs:
            children_ids.extend(parent_doc.metadata.get("children_ids", []))
        if not children_ids:
            raise ValueError(f"Parent documents <{parent_doc_ids}> have no children_ids metadata or empty.")
        # 根据 children_ids 获取子文档列表
        children_docs = self.vectorstore.get_by_ids(children_ids)
        if not children_docs:
            raise ValueError(f"No child documents found for parent_docs <{parent_doc_ids}>")
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
            print(
                f"DELETE CHUNKS: Deleted <{len(all_child_ids_to_delete)}> child documents. Result <{del_child_result}>.")
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
        full_docs = []  # 父文档列表，id和 Document 对象的元组列表
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
    ) -> tuple[list[str], list[Document]]:
        """
        添加文档到 docstore 和 vectorstore。
        Splitter切分时会添加text_splitters元数据。
        :return: (父文档 ID 有序列表, 添加的子文档列表)
        """
        children_docs, full_docs = self._split_docs_for_adding(
            documents,
            ids,
            add_to_docstore=add_to_docstore,
        )
        self.vectorstore.add_documents(children_docs, **kwargs)
        if add_to_docstore:
            self.docstore.mset(full_docs)
        return [_id for _id, _ in full_docs], children_docs

    async def aadd_documents(
            self,
            documents: list[Document],
            ids: Optional[list[str]] = None,
            add_to_docstore: bool = True,
            **kwargs: Any,
    ) -> tuple[list[str], list[Document]]:
        """
        异步添加文档到 docstore 和 vectorstore
        Splitter切分时会添加text_splitters元数据。
        :return: (父文档 ID 有序列表, 添加的子文档列表)
        """
        children_docs, full_docs = self._split_docs_for_adding(
            documents,
            ids,
            add_to_docstore=add_to_docstore,
        )
        await self.vectorstore.aadd_documents(children_docs, **kwargs)
        if add_to_docstore:
            await self.docstore.amset(full_docs)
        return [_id for _id, _ in full_docs], children_docs

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
            semantic_search_kwargs: Optional[dict] = None,
            **kwargs
    ) -> list[Document]:
        """
        重写父类方法，直接返回子文档列表
        :param query:
        :param run_manager:
        :param semantic_search_kwargs: 可选，搜索参数字典，会合并全局搜索参数，调用时传入的 search_kwargs 优先级高于全局搜索参数
                              参考链接 https://docs.trychroma.com/docs/querying-collections/metadata-filtering
        :return: 子文档有序列表
        """
        current_search_kwargs = copy.deepcopy(self.search_kwargs) if hasattr(self, "search_kwargs") else {}
        if semantic_search_kwargs:
            current_search_kwargs.update(semantic_search_kwargs)
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
            semantic_search_kwargs: Optional[dict] = None,
            **kwargs
    ) -> list[Document]:
        """
        重写父类方法，直接返回子文档列表
        :param query:
        :param run_manager:
        :param semantic_search_kwargs: 可选，搜索参数字典，会合并全局搜索参数，调用时传入的 search_kwargs 优先级高于全局搜索参数
        :return: 子文档有序列表
        """
        current_search_kwargs = copy.deepcopy(self.search_kwargs) if hasattr(self, "search_kwargs") else {}
        if semantic_search_kwargs:
            current_search_kwargs.update(semantic_search_kwargs)
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
            search_type=SearchType.mmr,  # 默认使用 MMR 检索
            id_key="parent_id"  # 父文档ID在子文档metadata中的键名，默认为parent_id
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
                      use_parent: bool = True, add_to_docstore: bool = True, **kwargs: Any, ) -> \
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
        parent_doc_ids, children_docs = pd_retriever.add_documents(documents, ids, add_to_docstore, **kwargs)
        return EnhancedPDRetrieverAddDocumentsResult(
            parent_doc_ids=parent_doc_ids,
            children_count=len(children_docs),
            children_docs=children_docs,
            parent_splitter_name=parent_splitter_name
        )

    @classmethod
    async def aadd_documents(cls, file_type: str, documents: list[Document], ids: Optional[list[str]] = None,
                             use_parent: bool = True, add_to_docstore: bool = True, **kwargs: Any, ) -> \
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
        parent_doc_ids, children_docs = await pd_retriever.aadd_documents(documents, ids, add_to_docstore, **kwargs)
        return EnhancedPDRetrieverAddDocumentsResult(
            parent_doc_ids=parent_doc_ids,
            children_count=len(children_docs),
            children_docs=children_docs,
            parent_splitter_name=parent_splitter_name
        )

    @classmethod
    def get_instance(cls) -> EnhancedParentDocumentRetriever:
        """获取 EnhancedParentDocumentRetriever 单例。"""
        if cls._pd_retriever is None:
            raise RuntimeError(
                "Retriever has not been initialized. Call EnhancedParentDocumentRetrieverFactory.init() first.")
        return cls._pd_retriever


class RetrievalKParams:
    """
    检索使用的k参数
    """

    def __init__(self, bm25_k: int, vector_k: int, vector_fetch_k: int, rrf_k: int, final_k: int):
        self.bm25_k = bm25_k
        self.vector_k = vector_k
        self.vector_fetch_k = vector_fetch_k
        self.rrf_k = rrf_k
        self.final_k = final_k


class SearchKwargs:
    """
    检索时使用的kwargs
    """

    def __init__(self, bm25_search_kwargs: Dict[str, Any], semantic_search_kwargs: Dict[str, Any]):
        self.bm25_search_kwargs = bm25_search_kwargs
        self.semantic_search_kwargs = semantic_search_kwargs


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

    def get_file_infos(self) -> list[dict]:
        """获取当前使用的文件名称集合"""
        return self._file_infos

    def _get_bm25_retriever(self) -> BaseRetriever:
        if self._bm25_retriever is None:
            raise RuntimeError("BM25Retriever has not been initialized. "
                               "Call reset_file_ids() first.")
        return self._bm25_retriever

    def _get_pd_retriever(self) -> EnhancedParentDocumentRetriever:
        if self._pd_retriever is None:
            raise RuntimeError("EnhancedParentDocumentRetriever has not been initialized. "
                               "Call reset_file_ids() first.")
        return self._pd_retriever

    def _get_retrieval_k_params(self) -> RetrievalKParams:
        if self._retrieval_k_params is None:
            raise RuntimeError("Retrieval parameters have not been initialized. "
                               "Call reset_file_ids() first.")
        return self._retrieval_k_params

    def _prepare_retrieval(self, input: str, k_params: RetrievalKParams, **kwargs) -> SearchKwargs:
        """
        invoke / ainvoke 的公共前置逻辑：
            1. 参数校验、配置两个 retriever 的检索参数。kwargs中的优先级更高
            2. 设置 self.retrievers，
            3. 返回 SearchKwargs。
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

        # 配置原始bm25检索参数
        bm25_search_kwargs = {
            "k": k_params.bm25_k,
            "file_id": list(file_ids),
        }

        # 配置原始语义向量检索参数
        semantic_search_kwargs = {
            "k": k_params.vector_k,
            "filter": {
                "file_id": {"$in": list(file_ids)}
            },
        }
        if pd_retriever.search_type == SearchType.mmr:
            semantic_search_kwargs["fetch_k"] = k_params.vector_fetch_k

        # 使用kwargs中的参数进行覆盖更新
        bm25_search_kwargs.update(kwargs.get("bm25_search_kwargs", {}))
        semantic_search_kwargs.update(kwargs.get("semantic_search_kwargs", {}))

        # 设置 EnsembleRetriever 的 retrievers
        self.retrievers = [bm25_retriever, pd_retriever]

        print(f"HYBRID PD RETRIEVER INVOKE: query='{input[:50]}...', "
              f"bm25_k={k_params.bm25_k}, vector_k={k_params.vector_k}, vector_fetch_k={k_params.vector_fetch_k}, rrf_k={k_params.rrf_k}, "
              f"rrf_c={self.c}, final_k={k_params.final_k}, file_ids_count={len(file_ids)}")

        return SearchKwargs(bm25_search_kwargs, semantic_search_kwargs)

    def reset_file_ids(self, file_ids: set[str], file_infos: list[dict],
                       child_docs: list[Document] | None = None) -> int:
        """
        重置已选择的文件ID集合，并创建对应的BM25Retriever实例（使用child文档，BM25Plus变体），
        重新设置search_kwargs以使用新的child文档集合进行检索。
        如果file_ids为空，则重置两个Retriever为None
        :param file_infos: 文档信息集合，用于大模型参考
        :param file_ids: 数据库中的文档ID列表
        :param child_docs: 可选，file_ids对应的child文档列表.
        :return: 0表示成功
        """
        self._file_ids = file_ids
        if not file_ids:
            self._file_infos = [{}]
            self._bm25_retriever = None
            self._pd_retriever = None
            self._retrieval_k_params = None
            print("HYBRID PD RETRIEVER: No file_ids provided, "
                  "BM25Retriever and EnhancedParentDocumentRetriever have been reset to None.")
            if child_docs:
                print("HYBRID PD RETRIEVER: Warning: child_docs provided but file_ids is empty, "
                      "child_docs will be ignored.")
            return 0
        if not child_docs:
            raise ValueError("HYBRID PD RETRIEVER: child_docs must be provided when file_ids is not empty.")

        # 设置文档名称集合
        self._file_infos = file_infos

        # 创建BM25Retriever实例，使用Elasticsearch
        self._bm25_retriever = ElasticSearchFactory.get_es_retriever()

        # 获取EnhancedParentDocumentRetriever（仅用于检索）
        self._pd_retriever = EnhancedParentDocumentRetrieverFactory.get_instance()

        # 设置检索参数
        k = math.ceil(min(50, max(10, math.ceil(len(child_docs) * 0.1))))  # k根据child_docs数量动态调整，范围在10-50之间
        self._retrieval_k_params = RetrievalKParams(
            bm25_k=k,
            vector_k=k,
            vector_fetch_k=math.ceil(min(len(child_docs), k * 3)),
            rrf_k=math.ceil(k * 0.5),
            final_k=self._retriever_config.get("final_k", 8)
        )

        return 0

    def merge_continuous_parent_docs(self, parent_docs: list[Document]) -> list[Document]:
        """
        合并连续的父文档块，如果它们的 parent_index 是连续的且属于同一个文件（file_id相同），则合并它们。
        合并到连续块中第一个父文档中，保持顺序。
        :param parent_docs:
        :return:
        """
        # todo

    def rank_fusion(
            self,
            query: str,
            run_manager: CallbackManagerForRetrieverRun,
            *,
            config: RunnableConfig | None = None,
            **kwargs: Any,
    ) -> list[Document]:
        """
        修改夫类 rank_fusion, 传递**kwargs
        :param query:
        :param run_manager:
        :param config:
        :param kwargs:
        :return:
        """
        # Get the results of all retrievers.
        retriever_docs = [
            retriever.invoke(
                query,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(tag=f"retriever_{i + 1}"),
                ),
                **kwargs,
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=cast("str", doc)) if isinstance(doc, str) else doc  # type: ignore[unreachable]
                for doc in retriever_docs[i]
            ]

        # apply rank fusion
        return self.weighted_reciprocal_rank(retriever_docs)

    async def arank_fusion(
            self,
            query: str,
            run_manager: AsyncCallbackManagerForRetrieverRun,
            *,
            config: RunnableConfig | None = None,
            **kwargs: Any,
    ) -> list[Document]:
        # Get the results of all retrievers.
        retriever_docs = await asyncio.gather(
            *[
                retriever.ainvoke(
                    query,
                    patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"retriever_{i + 1}"),
                    ),
                    **kwargs,
                )
                for i, retriever in enumerate(self.retrievers)
            ],
        )

        # Enforce that retrieved docs are Documents for each list in retriever_docs
        for i in range(len(retriever_docs)):
            retriever_docs[i] = [
                Document(page_content=doc) if not isinstance(doc, Document) else doc
                for doc in retriever_docs[i]
            ]

        # apply rank fusion
        return self.weighted_reciprocal_rank(retriever_docs)

    def super_invoke(
            self,
            input: str,
            config: RunnableConfig | None = None,
            **kwargs: Any,
    ) -> list[Document]:
        """
        修改负类，传递**kwargs给rank_fusion
        :param input:
        :param config:
        :param kwargs:
        :return:
        """
        from langchain_core.callbacks import CallbackManager

        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = self.rank_fusion(input, run_manager=run_manager, config=config, **kwargs)
        except Exception as e:
            run_manager.on_retriever_error(e)
            raise
        else:
            run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

    async def super_ainvoke(
            self,
            input: str,
            config: RunnableConfig | None = None,
            **kwargs: Any,
    ) -> list[Document]:
        from langchain_core.callbacks import AsyncCallbackManager

        config = ensure_config(config)
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=config.get("tags", []),
            local_tags=self.tags,
            inheritable_metadata=config.get("metadata", {}),
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            **kwargs,
        )
        try:
            result = await self.arank_fusion(
                input,
                run_manager=run_manager,
                config=config,
                **kwargs,
            )
        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise
        else:
            await run_manager.on_retriever_end(
                result,
                **kwargs,
            )
            return result

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
        k_params = self._get_retrieval_k_params()
        search_kwargs = self._prepare_retrieval(input, k_params, **kwargs)
        kwargs["bm25_search_kwargs"] = search_kwargs.bm25_search_kwargs
        kwargs["semantic_search_kwargs"] = search_kwargs.semantic_search_kwargs
        fused_results = self.super_invoke(input, config=config, **kwargs)

        # 获取 rrf_k 个子 Document 对应的父文档列表，保持顺序
        children_docs = fused_results[:k_params.rrf_k]
        parent_docs = self._get_pd_retriever().get_parent_docs_from_children(children_docs)

        # 可选的 rerank 逻辑
        if not self._retriever_config.get("reranker", {}).get("enabled", False):
            print("HYBRID PD RETRIEVER INVOKE: Reranker is disabled, skipping reranking step.")
            return parent_docs[:k_params.final_k]

        # 数量 <= final_k 时跳过重排，直接返回
        if len(parent_docs) <= k_params.final_k:
            print(
                f"HYBRID PD RETRIEVER INVOKE: parent_docs count ({len(parent_docs)}) <= final_k ({k_params.final_k}), "
                f"skipping reranking step.")
            return parent_docs

        print(f"HYBRID PD RETRIEVER INVOKE: Reranking {len(parent_docs)} parent docs...")
        reranker = RerankerFactory.get_instance()
        reranked_docs = reranker.compress_documents(parent_docs, input)
        return list(reranked_docs[:k_params.final_k])

    async def ainvoke(
            self,
            input: str,
            config: RunnableConfig | None = None,
            **kwargs: Any,
    ) -> list[Document]:
        """
        异步混合检索。
        RRF 融合后返回 rrf_k 个子 Document，获取对应父文档列表，执行可选的 rerank 后返回 final_k 个父Document。
        """
        k_params = self._get_retrieval_k_params()
        search_kwargs = self._prepare_retrieval(input, k_params, **kwargs)
        kwargs["bm25_search_kwargs"] = search_kwargs.bm25_search_kwargs
        kwargs["semantic_search_kwargs"] = search_kwargs.semantic_search_kwargs
        fused_results = await self.super_ainvoke(input, config=config, **kwargs)

        # 获取 rrf_k 个子 Document 对应的父文档列表，保持顺序
        children_docs = fused_results[:k_params.rrf_k]
        parent_docs = await self._get_pd_retriever().aget_parent_docs_from_children(children_docs)

        # 可选的 rerank 逻辑
        if not self._retriever_config.get("reranker", {}).get("enabled", False):
            print("HYBRID PD RETRIEVER AINVOKE: Reranker is disabled, skipping reranking step.")
            return parent_docs[:k_params.final_k]

        # 数量 <= final_k 时跳过重排，直接返回
        if len(parent_docs) <= k_params.final_k:
            print(
                f"HYBRID PD RETRIEVER AINVOKE: parent_docs count ({len(parent_docs)}) <= final_k ({k_params.final_k}), "
                f"skipping reranking step.")
            return parent_docs

        print(f"HYBRID PD RETRIEVER AINVOKE: Reranking {len(parent_docs)} parent docs...")
        reranker = RerankerFactory.get_instance()
        reranked_docs = await reranker.acompress_documents(parent_docs, input)
        return list(reranked_docs[:k_params.final_k])


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
        cls._instance._file_infos = [{}]
        cls._instance._bm25_retriever = None
        cls._instance._pd_retriever = None
        cls._instance._retrieval_k_params = None
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
