import json
import os.path
import uuid

import filetype
from langchain_core.vectorstores import VectorStore
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from app.core.chunking import SplitterChain
from app.core.document_loader import load_doc_to_vector_store
from app.exception.exception import CustomException
from app.models.schemas import EmbeddedDocument


async def list_documents(db: AsyncSession):
    """
    获取所有文档
    :param db:
    :return: 文档列表
    """
    stmt = select(EmbeddedDocument).order_by(EmbeddedDocument.updated_at.desc())
    result = await db.execute(stmt)
    return result.scalars().all()


async def get_document_by_id(doc_id: str, db: AsyncSession):
    """
    获取文档详情
    :param doc_id:
    :param db:
    :return: 文档详情，若不存在抛出异常
    """
    stmt = select(EmbeddedDocument).where(EmbeddedDocument.id == doc_id)
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()
    if doc is None:
        raise CustomException(code=status.HTTP_400_BAD_REQUEST, message="文档不存在")
    return doc


async def upload_document(file_path: str, is_url: bool, db: AsyncSession,
                    splitter_chain: SplitterChain, vector_store: VectorStore):
    """
    上传文档/读取网页，分块并入库，文件信息存储在数据库中，分块后的文本和向量存储在向量数据库中
    :param file_path:
    :param is_url:
    :param db:
    :param splitter_chain:
    :param vector_store:
    :return:0 表示成功
    """
    load_result = None
    try:
        # 1. 对文档进行分块
        file_id = str(uuid.uuid4())
        load_result = await load_doc_to_vector_store(splitter_chain, vector_store, file_path, file_id, is_url)
        if load_result.error:
            raise Exception(f"Failed to load document <{file_path}>: {load_result.error}")
        # 2. 将文档信息存储在数据库中
        document_ids = load_result.document_ids if load_result.document_ids else []
        doc_metadata = {
            "embedding_model": vector_store.embeddings.__class__.__name__  + ":" + vector_store.embeddings.model_name,
            "document_loader": load_result.document_loader,
            "splitters": load_result.splitters,
        }
        if is_url:
            filename = file_path
            file_ext = ".html"
            file_type = "text/html"
        else:
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(file_path)[1].lower()
            file_type = filetype.guess(file_path)
        embedded_doc = EmbeddedDocument(
            id=file_id,
            filename=filename,
            file_path=file_path,
            file_extension=file_ext,
            filetype=file_type,
            chunk_count=len(document_ids),
            chunk_ids=document_ids,
            doc_metadata=doc_metadata
        )
        db.add(embedded_doc)
        await db.commit()
        return 0
    except Exception as e:
        print(f"UPLOAD DOCUMENT ERROR: Failed to upload document <{file_path}>: {e}")
        # 如果入库失败，应该删除已经添加到向量数据库中的文档，以保持数据一致性
        if load_result and load_result.document_ids:
            print("UPLOAD DOCUMENT ERROR: Rolling back vector store changes...")
            vector_store.delete(ids=load_result.document_ids)
        raise e


async def delete_document(doc_id: str, vector_store: VectorStore, db: AsyncSession):
    """
    删除文档，同时删除向量数据库中的相关数据
    :param doc_id:
    :param vector_store:
    :param db:
    :return:
    """
    doc = await get_document_by_id(doc_id, db)
    # 1. 删除向量数据库中的相关数据
    if doc.chunk_ids and len(doc.chunk_ids) > 0:
        print(f"DELETE DOCUMENT: Deleting chunks for document <{doc_id}> with {len(doc.chunk_ids)} chunks")
        vector_store.delete(ids=doc.chunk_ids)
    # 2. 删除数据库中的文档记录
    await db.delete(doc)
    await db.commit()
    return 0