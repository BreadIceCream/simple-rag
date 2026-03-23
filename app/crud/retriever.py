from sqlalchemy.ext.asyncio import AsyncSession

from app.core.retriever import EnhancedParentDocumentRetriever, HybridPDRetriever, \
    EnhancedParentDocumentRetrieverFactory, HybridPDRetrieverFactory
from app.crud.document import get_document_by_ids


async def set_reference_documents(doc_ids: list[str] | None, db: AsyncSession | None):
    """
    设置参考文档，供后续问答调用
    :param doc_ids: 数据库中的文档ID列表
    :param db:
    :return: 0 表示成功
    """
    pd_retriever = EnhancedParentDocumentRetrieverFactory.get_instance()
    hybrid_retriever = HybridPDRetrieverFactory.get_instance()
    if not doc_ids:
        # 如果传入空列表，表示清空参考文档
        return hybrid_retriever.reset_file_ids(set(), [{}], None)
    # 1. 获取所有文档的子文档列表
    all_docs = await get_document_by_ids(doc_ids, db)
    all_child_docs = []
    valid_doc_ids = []
    valid_doc_infos = []
    for a_doc in all_docs:
        valid_doc_ids.append(a_doc.id)
        a_doc_info = {
            "file_id" : a_doc.id,
            "file_name" : a_doc.file_name if a_doc.file_name else a_doc.path,
            "file_summary": a_doc.file_summary
        }
        valid_doc_infos.append(a_doc_info)
        parent_ids = a_doc.parent_doc_ids
        all_child_docs.extend(pd_retriever.get_child_docs(parent_ids))
    # 2. 传入HybridPDRetriever进行设置
    return hybrid_retriever.reset_file_ids(set(valid_doc_ids), valid_doc_infos, all_child_docs)


async def get_reference_documents(db: AsyncSession):
    """
    获取当前设置的参考文档列表
    :param db:
    :return:
    """
    hybrid_retriever = HybridPDRetrieverFactory.get_instance()
    doc_ids = hybrid_retriever.get_file_ids()
    if not doc_ids:
        return []
    return await get_document_by_ids(list(doc_ids), db)


async def clear_reference_documents():
    """
    清空当前设置的参考文档
    :return: 0 表示成功
    """
    return await set_reference_documents(None, None)


async def query(msg: str):
    """
    使用检索器异步执行查询，返回相关文档列表
    :param msg:
    :return: list[Document]，相关文档列表
    """
    hybrid_retriever = HybridPDRetrieverFactory.get_instance()
    return await hybrid_retriever.ainvoke(msg)