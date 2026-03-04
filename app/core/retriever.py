"""
retriever模块。

核心组件：
- EnhancedParentDocumentRetriever: 自定义 ParentDocumentRetriever，增强元数据和返回值
- RetrieverFactory: 根据文件类型配置好 parent_splitter 的 ParentDocumentRetriever 工厂
"""

import uuid
from typing import Any, Optional

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

from app.core.chunking import SplitterRegistry


# ======================== EnhancedParentDocumentRetriever ========================

class EnhancedParentDocumentRetriever(ParentDocumentRetriever):
    """
    增强版 ParentDocumentRetriever，在原版基础上：
    1. add_documents / aadd_documents 返回父文档 ID 列表
    2. 为每个父文档块添加 children_ids 和 children_count 元数据
    3. 为每个父文档块添加 parent_index，为每个子文档块添加 child_index 元数据
    4. 添加获取父/子文档块的方法
    5. 添加级联删除父子文档的方法
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
        - 手动为子文档生成并设置 id 属性
        - 父文档设置 id 属性，添加 children_ids, children_count, parent_index 元数据
        - 子文档添加 child_index 元数据
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
                _doc.metadata[self.id_key] = parent_id
                _doc.metadata["child_index"] = child_index
                children_ids.append(child_id)

            # 为父文档设置id属性，并添加增强元数据
            parent_doc.id = parent_id
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
    ) -> list[str]:
        """
        异步添加文档到 docstore 和 vectorstore
        :return: 父文档 ID 列表
        """
        children_docs, full_docs = self._split_docs_for_adding(
            documents,
            ids,
            add_to_docstore=add_to_docstore,
        )
        await self.vectorstore.aadd_documents(children_docs, **kwargs)
        if add_to_docstore:
            await self.docstore.amset(full_docs)
        return [_id for _id, _ in full_docs]


# ======================== RetrieverFactory ========================

class RetrieverFactory:
    """
    EnhancedParentDocumentRetriever 工厂(单例)。
    职责：根据文件类型，从 SplitterRegistry 获取对应的 parent_splitter，
    配置好 retriever.parent_splitter 并返回。
    child_splitter 固定为 RecursiveCharacterTextSplitter，所有文件类型共用，从 SplitterRegistry 获取。
    """
    _pd_retriever: EnhancedParentDocumentRetriever | None = None

    @classmethod
    def init(cls, vectorstore, docstore) -> EnhancedParentDocumentRetriever:
        """
        初始化 EnhancedParentDocumentRetriever。
        需先初始化SplitterRegistry，要从SplitterRegistry中获取Splitter
        """
        if cls._pd_retriever is not None:
            print("INIT RETRIEVER: Instance already exists, returning cached instance.")
            return cls._pd_retriever

        # child_splitter 固定：用于将父文档切成小块做向量检索
        child_splitter = SplitterRegistry.get_child_splitter()

        cls._pd_retriever = EnhancedParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            # parent_splitter 不在此设置，由 configure_pd_for_file_type() 动态设置
        )

        print("INIT RETRIEVER: Initialized successfully.")
        return cls._pd_retriever

    @classmethod
    def configure_pd_for_file_type(cls, file_type: str, use_parent: bool = True) -> \
            tuple[str, EnhancedParentDocumentRetriever]:
        """
        根据文件类型配置 ParentDocumentRetriever 的 parent_splitter 并返回。
        :param file_type: 文件后缀
        :param use_parent: 是否使用父分块器，False 则 parent_splitter=None（原始文档整体作为父文档）, splitter_name = None
        :return: (parent_splitter_name, EnhancedParentDocumentRetriever)
        """
        retriever = cls.get_pd_retriever()
        if use_parent:
            splitter_registry = SplitterRegistry.get_instance()
            splitter_name, parent_splitter = splitter_registry.get_splitter(file_type)
            retriever.parent_splitter = parent_splitter
        else:
            splitter_name = None
            retriever.parent_splitter = None
        return splitter_name, retriever

    @classmethod
    def get_pd_retriever(cls) -> EnhancedParentDocumentRetriever:
        """获取 EnhancedParentDocumentRetriever 单例。"""
        if cls._pd_retriever is None:
            raise RuntimeError("Retriever has not been initialized. Call RetrieverFactory.init() first.")
        return cls._pd_retriever
