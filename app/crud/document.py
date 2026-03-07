import os.path
import time
import uuid
from pathlib import Path

import filetype
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from app.core.chunking import SplitterRegistry
from app.core.document_loader import load_doc_to_vector_store
from app.core.retriever import EnhancedParentDocumentRetriever
from app.exception.exception import CustomException
from app.models.schemas import EmbeddedDocument


async def list_documents(db: AsyncSession):
    """
    获取所有文档
    :param db:
    :return: 文档列表
    """
    stmt = select(EmbeddedDocument).order_by(EmbeddedDocument.created_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_document_by_id(doc_id: str, db: AsyncSession) -> EmbeddedDocument:
    """
    获取文档详情
    :param doc_id: 数据库中的文档ID
    :param db:
    :return: 文档详情，若不存在抛出异常
    """
    stmt = select(EmbeddedDocument).where(EmbeddedDocument.id == doc_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if doc is None:
        raise CustomException(code=status.HTTP_400_BAD_REQUEST, message="文档不存在")
    return doc


async def get_document_by_ids(doc_ids: list[str], db: AsyncSession) -> list[EmbeddedDocument]:
    """
    获取多个文档详情
    :param doc_ids: 数据库中的文档ID列表
    :param db:
    :return: 合法的文档详情有序列表
    """
    stmt = select(EmbeddedDocument).where(EmbeddedDocument.id.in_(doc_ids))
    result = await db.execute(stmt)
    docs = result.scalars().all()
    # 按照输入的 doc_ids 顺序返回文档详情列表，未找到的文档直接过滤
    doc_dict = {doc.id: doc for doc in docs}
    ordered_docs = [doc_dict[doc_id] for doc_id in doc_ids if doc_id in doc_dict]
    return ordered_docs


async def upload_document(path: str, is_url: bool, pd_retriever: EnhancedParentDocumentRetriever, db: AsyncSession):
    """
    上传文档/读取网页，通过 ParentDocumentRetriever 分块并入库，文件信息存储在数据库中
    :param path:
    :param is_url:
    :param pd_retriever:
    :param db:
    :return: 0 表示成功
    """
    load_result = None
    try:
        # 1. 对文档进行分块（由 retriever.add_documents 自动完成）
        file_id = str(uuid.uuid4())
        load_result = await load_doc_to_vector_store(path, file_id, is_url)
        if load_result.error:
            raise Exception(f"Failed to load document <{path}>: {load_result.error}")
        # 2. 构造入库对象
        splitters = [load_result.parent_splitter, SplitterRegistry.child_splitter_name] if load_result.parent_splitter else \
                        [SplitterRegistry.child_splitter_name]
        load_metadata = {
            "embedding_model": pd_retriever.vectorstore.embeddings.__class__.__name__ + ":"
                               + pd_retriever.vectorstore.embeddings.model_name,
            "document_loader": load_result.document_loader,
            "text_splitters": splitters
        }
        if is_url:
            file_dir = None
            file_name = None
            last_modified = None
            file_ext = ".html"
            mime_type = "text/html"
        else:
            file = Path(path)
            file_dir = str(file.parent.resolve())
            file_name = file.name
            last_modified = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path)))
            file_ext = file.suffix
            mime_type = filetype.guess_mime(path)
        embedded_doc = EmbeddedDocument(
            id=file_id,
            path=path,
            is_url=is_url,
            file_directory=file_dir,
            file_name=file_name,
            file_extension=file_ext,
            mime_type=mime_type,
            last_modified=last_modified,
            parent_doc_ids=load_result.parent_doc_ids if load_result.parent_doc_ids else [],
            children_count=load_result.children_count,
            load_metadata=load_metadata
        )
        db.add(embedded_doc)
        await db.commit()
        return 0
    except Exception as e:
        print(f"UPLOAD DOCUMENT ERROR: Failed to upload document <{path}>: {e}")
        # 如果入库失败，应该删除已经添加的父子文档，以保持数据一致性
        if load_result and load_result.parent_doc_ids:
            print(f"UPLOAD DOCUMENT ERROR: Rolling back documents loaded for <{path}>")
            pd_retriever.delete_docs_cascade(load_result.parent_doc_ids)
        raise e


async def delete_document(doc_id: str, pd_retriever: EnhancedParentDocumentRetriever, db: AsyncSession):
    """
    删除文档，同时删除向量数据库中的相关数据
    :param doc_id: 数据库中的文档ID
    :param pd_retriever:
    :param db:
    :return: 0 表示成功
    """
    doc = await get_document_by_id(doc_id, db)
    # 1. 删除向量数据库中的相关数据
    if doc.parent_doc_ids:
        print(f"DELETE DOCUMENT: Deleting document <{doc_id}> in docstore and vector store: "
              f"{len(doc.parent_doc_ids)} parent docs and {doc.children_count} child docs...")
        pd_retriever.delete_docs_cascade(doc.parent_doc_ids)
    # 2. 删除数据库中的文档记录
    await db.delete(doc)
    await db.commit()
    return 0


async def get_parent_chunks(doc_id: str, offset: int, limit: int, pd_retriever: EnhancedParentDocumentRetriever, db: AsyncSession):
    """
    获取文件切分后的父文档列表，支持分页
    :param doc_id: 数据库中的文档ID
    :param offset:
    :param limit:
    :param pd_retriever:
    :param db:
    :return: 父文档总数和父文档列表
    """
    doc = await get_document_by_id(doc_id, db)
    total = len(doc.parent_doc_ids)
    if total == 0:
        return 0, []
    ids_to_search = doc.parent_doc_ids[offset: offset + limit]
    parent_docs_tuple = pd_retriever.get_parent_docs(ids_to_search)
    result_list = []
    for _, parent_doc in parent_docs_tuple:
        result_list.append(parent_doc)
    return total, result_list