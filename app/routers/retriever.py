from uuid import UUID

from fastapi import APIRouter, Depends, Body
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.db_config import DatabaseManager
from app.core.retriever import EnhancedParentDocumentRetrieverFactory, EnhancedParentDocumentRetriever, \
    HybridPDRetrieverFactory, HybridPDRetriever
from app.crud import retriever
from app.models.common import Result, EmbeddedDocumentVO


router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


@router.post("/references")
async def set_reference_documents(doc_ids: list[UUID] = Body(..., alias="docIds"),
                                  db: AsyncSession = Depends(DatabaseManager.get_db),
                                  pd_retriever: EnhancedParentDocumentRetriever = Depends(EnhancedParentDocumentRetrieverFactory.get_instance),
                                  hybrid_retriever: HybridPDRetriever = Depends(HybridPDRetrieverFactory.get_instance)):
    """
    设置检索器参考的文档列表
    :param doc_ids:
    :param db:
    :param pd_retriever:
    :param hybrid_retriever:
    :return:
    """
    doc_ids = [str(doc_id) for doc_id in doc_ids]
    result = await retriever.set_reference_documents(doc_ids, db, pd_retriever, hybrid_retriever)
    if result == 0:
        return Result(code=200, message="参考文档设置成功", data=None)
    else:
        raise Exception("参考文档设置失败")


@router.get("/references")
async def get_reference_documents(db: AsyncSession = Depends(DatabaseManager.get_db),
                                  hybrid_retriever: HybridPDRetriever = Depends(HybridPDRetrieverFactory.get_instance)):
    """
    获取当前设置的参考文档列表
    :param hybrid_retriever:
    :param db:
    :return:
    """
    doc_list = await retriever.get_reference_documents(db, hybrid_retriever)
    doc_vo_list = []
    for doc in doc_list:
        doc_vo = EmbeddedDocumentVO.model_validate(doc)
        doc_vo_list.append(doc_vo)
    return Result(code=200, message="参考文档列表获取成功", data=doc_vo_list)


@router.delete("/references")
async def clear_reference_documents(hybrid_retriever: HybridPDRetriever = Depends(HybridPDRetrieverFactory.get_instance)):
    """
    清空当前设置的参考文档列表
    :return:
    """
    result = await retriever.clear_reference_documents(hybrid_retriever)
    if result == 0:
        return Result(code=200, message="参考文档列表清空成功", data=None)
    else:
        raise Exception("参考文档列表清空失败")

@router.post("/query")
async def query(message: str = Body(..., embed=True),
                hybrid_retriever: HybridPDRetriever = Depends(HybridPDRetrieverFactory.get_instance)):
    """
    根据查询语句进行检索，返回相关文档列表
    :param hybrid_retriever:
    :param message:
    :return:
    """
    docs = await retriever.query(message, hybrid_retriever)
    return Result(code=200, message="查询成功", data=docs)