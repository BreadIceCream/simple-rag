from fastapi import APIRouter, Depends, Path
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.db_config import DatabaseManager
from app.core.chunking import SplitterChainFactory
from app.core.vector_store import VectorStoreFactory
from app.crud import document
from app.models.common import Result
from app.models.schemas import EmbeddedDocumentVO

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.get("/list")
async def list_documents(db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    列出所有已上传的文档
    :return: 文档列表
    """
    documents_list = await document.list_documents(db)
    doc_vo_list = []
    for doc in documents_list:
        doc_vo = EmbeddedDocumentVO.model_validate(doc)
        doc_vo_list.append(doc_vo)
    return Result(code=200, message="文档列表获取成功", data=doc_vo_list)


@router.get("/{doc_id}")
async def get_document(
        doc_id: str = Path(..., regex=r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$",
                           description="文档uuid"),
        db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    获取文档详情
    :param doc_id:
    :param db:
    :return:
    """
    doc = await document.get_document_by_id(doc_id, db)
    return Result(code=200, message="文档详情获取成功", data=doc)


@router.post("/local")
async def upload_local_document(file_path: str, db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    上传文档，分块并入库
    :param file_path: 文件路径
    :param db:
    :return:
    """
    splitter_chain = SplitterChainFactory.get_instance()
    vector_store = VectorStoreFactory.get_instance()
    result = await document.upload_document(file_path, False, db, splitter_chain, vector_store)
    if result == 0:
        return Result(code=200, message="文档上传成功", data=None)
    else:
        raise Exception("文档上传失败")


@router.post("/url")
async def upload_url_document(url: str, db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    读取网页，分块并入库
    :param url: 网页 URL
    :param db:
    :return:
    """
    splitter_chain = SplitterChainFactory.get_instance()
    vector_store = VectorStoreFactory.get_instance()
    result = await document.upload_document(url, True, db, splitter_chain, vector_store)
    if result == 0:
        return Result(code=200, message="文档上传成功", data=None)
    else:
        raise Exception("文档上传失败")


@router.delete("/{doc_id}")
async def delete_document(
        doc_id: str = Path(..., regex=r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$",
                           description="文档uuid"),
        db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    删除文档
    :param doc_id:
    :param db:
    :return:
    """
    vector_store = VectorStoreFactory.get_instance()
    result = await document.delete_document(doc_id, vector_store, db)
    if result == 0:
        return Result(code=200, message="文档删除成功", data=None)
    else:
        raise Exception("文档删除失败")
