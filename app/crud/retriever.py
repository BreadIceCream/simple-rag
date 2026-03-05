from sqlalchemy.ext.asyncio import AsyncSession

from app.core.retriever import EnhancedParentDocumentRetriever, HybridPDRetriever
from app.crud.document import get_document_by_ids


async def set_reference_documents(doc_ids: list[str] | None, db: AsyncSession | None,
                                  pd_retriever: EnhancedParentDocumentRetriever | None,
                                  hybrid_retriever: HybridPDRetriever):
    """
    设置参考文档，供后续问答调用
    :param doc_ids: 数据库中的文档ID列表
    :param db:
    :param pd_retriever:
    :param hybrid_retriever:
    :return: 0 表示成功
    """
    if not doc_ids:
        # 如果传入空列表，表示清空参考文档
        return hybrid_retriever.reset_file_ids(set(), None)
    # 1. 获取所有文档的子文档列表
    all_docs = await get_document_by_ids(doc_ids, db)
    all_child_docs = []
    for a_doc in all_docs:
        parent_ids = a_doc.parent_doc_ids
        for parent_id in parent_ids:
            child_docs = pd_retriever.get_child_docs(parent_id)
            all_child_docs.extend(child_docs)
    # 2. 传入HybridPDRetriever进行设置
    return hybrid_retriever.reset_file_ids(set(doc_ids), all_child_docs)


async def get_reference_documents(db: AsyncSession, hybrid_retriever: HybridPDRetriever):
    """
    获取当前设置的参考文档列表
    :param db:
    :param hybrid_retriever:
    :return:
    """
    doc_ids = hybrid_retriever.get_file_ids()
    if not doc_ids:
        return []
    return await get_document_by_ids(list(doc_ids), db)


async def clear_reference_documents(hybrid_retriever: HybridPDRetriever):
    """
    清空当前设置的参考文档
    :return: 0 表示成功
    """
    return await set_reference_documents(None, None, None, hybrid_retriever)


async def query(msg: str, hybrid_retriever: HybridPDRetriever):
    """
    使用检索器异步执行查询，返回相关文档列表
    :param msg:
    :param hybrid_retriever:
    :return: list[Document]，相关文档列表
    """
    return await hybrid_retriever.ainvoke(msg)