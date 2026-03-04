from fastapi import APIRouter, Depends, Path
from fastapi.params import Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.db_config import DatabaseManager
from app.core.retriever import RetrieverFactory
from app.crud import document
from app.models.common import Result
from app.models.schemas import EmbeddedDocumentVO

router = APIRouter(prefix="/api/documents", tags=["documents"])

uuid_regex = r"^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$"


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

@router.get("/parents")
async def get_parent_chunks(
        doc_id: str = Query(..., alias="docId", pattern=uuid_regex, description="文档uuid"),
        offset: int = Query(0, ge=0, description="分块列表的起始位置,包括"),
        limit: int = Query(10, gt=0, le=20, description="获取的分块个数,最多20个"),
        db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    获取父文档的分块列表，支持分页
    :param doc_id:
    :param offset:
    :param limit:
    :param db:
    :return: 父文档分块总数和分块列表
    """
    total, chunks = await document.get_parent_chunks(doc_id, offset, limit, RetrieverFactory.get_pd_retriever(), db)
    return Result(code=200, message="文档分块获取成功", data={"total": total, "chunks": chunks})

@router.post("/local")
async def upload_local_document(file_path: str, db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    上传文档，分块并入库
    :param file_path: 文件路径
    :param db:
    :return:
    """
    result = await document.upload_document(file_path, False, RetrieverFactory.get_pd_retriever(), db)
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
    result = await document.upload_document(url, True, RetrieverFactory.get_pd_retriever(), db)
    if result == 0:
        return Result(code=200, message="文档上传成功", data=None)
    else:
        raise Exception("文档上传失败")


@router.get("/{doc_id}")
async def get_document(
        doc_id: str = Path(..., pattern=uuid_regex, description="文档uuid"),
        db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    获取文档详情
    :param doc_id:
    :param db:
    :return:
    """
    doc = await document.get_document_by_id(doc_id, db)
    return Result(code=200, message="文档详情获取成功", data=doc)


@router.delete("/{doc_id}")
async def delete_document(
        doc_id: str = Path(..., pattern=uuid_regex, description="文档uuid"),
        db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    删除文档
    :param doc_id:
    :param db:
    :return:
    """
    result = await document.delete_document(doc_id, RetrieverFactory.get_pd_retriever(), db)
    if result == 0:
        return Result(code=200, message="文档删除成功", data=None)
    else:
        raise Exception("文档删除失败")
