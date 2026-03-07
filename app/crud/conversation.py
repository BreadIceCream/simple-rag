"""
对话相关 CRUD 操作。
"""
from datetime import datetime
from typing import Literal

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from app.core.graph import Graph
from app.exception.exception import CustomException
from app.models.schemas import Conversation, ChatHistory


async def create_conversation(conversation_id: str, title: str, db: AsyncSession) -> Conversation:
    """
    创建新对话
    :param conversation_id: 对话 UUID（同时作为 LangGraph conversation_id）
    :param title: 对话标题
    :param db:
    :return: 创建的 Conversation 对象
    """
    conv = Conversation(id=conversation_id, title=title)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def get_conversation(conversation_id: str, db: AsyncSession) -> Conversation:
    """
    获取单个对话，不存在时抛出异常
    :param conversation_id: 对话 UUID
    :param db:
    :return: Conversation 对象
    :raise CustomException: 对话不存在
    """
    stmt = select(Conversation).where(Conversation.id == conversation_id)
    result = await db.execute(stmt)
    conv = result.scalar_one_or_none()
    if conv is None:
        raise CustomException(code=status.HTTP_400_BAD_REQUEST, message="对话不存在")
    return conv


async def list_conversations(db: AsyncSession) -> list[Conversation]:
    """
    获取所有对话列表，按最后更新时间降序
    """
    stmt = select(Conversation).order_by(Conversation.updated_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def update_conversation_title(conversation_id: str, title: str, db: AsyncSession) -> Conversation:
    """
    修改对话标题
    :param conversation_id: 对话 UUID
    :param title: 新标题
    :param db:
    :return: 更新后的 Conversation 对象
    """
    conv = await get_conversation(conversation_id, db)
    conv.title = title
    conv.updated_at = datetime.now()
    await db.commit()
    await db.refresh(conv)
    return conv


async def update_conversation_checkpoint_id(conversation_id: str, checkpoint_id: str | None, db: AsyncSession):
    """
    更新对话的 checkpoint ID
    """
    conv = await get_conversation(conversation_id, db)
    conv.checkpoint_id = checkpoint_id
    # 更新不需要修改 updated_at 或者可以依赖 onupdate
    await db.commit()


async def delete_conversation(conversation_id: str, db: AsyncSession):
    """
    删除对话：删除 ChatHistory 记录 + Conversation 记录 + Graph checkpointer 数据
    :param conversation_id: 对话 UUID
    :param db:
    """
    # 1. 删除关联的聊天记录
    await db.execute(delete(ChatHistory).where(ChatHistory.conversation_id == conversation_id))
    # 2. 删除对话记录
    conv = await get_conversation(conversation_id, db)
    await db.delete(conv)
    await db.commit()
    # 3. 删除 graph checkpointer 中的 thread 数据
    try:
        Graph.delete_thread(conversation_id)
    except Exception as e:
        print(f"CRUD: Warning: Failed to delete graph thread '{conversation_id}': {e}")


async def add_chat_message(
    conversation_id: str,
    role: Literal["user", "ai"],
    content: str,
    parent_doc_ids: list[str],
    db: AsyncSession,
) -> ChatHistory:
    """
    添加一条聊天记录，并更新对话的 updated_at
    :param conversation_id: 对话 ID
    :param role: "user" 或 "ai"
    :param content: 消息内容
    :param parent_doc_ids: 本次检索到的父文档 ID 列表（doc.id）
    :param db:
    :return: 创建的 ChatHistory 对象
    """
    msg = ChatHistory(
        conversation_id=conversation_id,
        role=role,
        content=content,
        parent_doc_ids=parent_doc_ids,
    )
    db.add(msg)
    # 更新对话的 updated_at
    conv = await get_conversation(conversation_id, db)
    conv.updated_at = datetime.now()
    await db.commit()
    await db.refresh(msg)
    return msg


async def get_chat_history(conversation_id: str, db: AsyncSession) -> list[ChatHistory]:
    """
    获取对话的聊天记录列表，按创建时间升序
    :param conversation_id: 对话 ID
    :param db:
    :return: ChatHistory 聊天记录列表
    """
    stmt = (
        select(ChatHistory)
        .where(ChatHistory.conversation_id == conversation_id)
        .order_by(ChatHistory.created_at.asc())
    )
    result = await db.execute(stmt)
    return list(result.scalars().all())
