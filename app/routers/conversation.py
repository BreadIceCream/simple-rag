"""
对话路由 — 基于 LangGraph 的 Agentic RAG 聊天接口。
使用 SSE (Server-Sent Events) 实现流式输出。
对话历史独立存储在 conversation / chat_history 表中，与 graph 内部消息隔离。
"""
import asyncio
import json
import uuid
from uuid import UUID

from fastapi import APIRouter, Body, Depends
from fastapi.params import Query, Path
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.db_config import DatabaseManager
from app.core.graph import Graph
from app.crud import conversation as conv_crud
from app.models.common import Result, ConversationVO, ChatMessageVO, ChatHistoryResponseVO

router = APIRouter(prefix="/api/conversation", tags=["conversation"])


@router.post("/chat")
async def chat(
    message: str = Body(..., embed=True),
    conversation_id: UUID = Body(None, embed=True),
    db: AsyncSession = Depends(DatabaseManager.get_db),
):
    """
    聊天接口（SSE 流式输出）

    使用 LangGraph 的 stream_mode="custom" 获取节点内部通过 writer 发出的自定义事件。
    同时使用 "updates" 模式获取最终的状态更新（包含 LLM 回答）。

    graph 执行完成后，将用户消息和 AI 最终回答存入 chat_history 表。
    新对话自动创建 conversation 记录，标题由 LLM 并发生成。

    :param message: 用户输入的消息
    :param conversation_id: 对话 ID。为空时自动创建新对话
    :param db: 数据库会话
    :return: SSE 流式响应
    """
    is_new_conversation = conversation_id is None
    conversation_id_str = str(uuid.uuid4()) if is_new_conversation else str(conversation_id)

    graph = Graph.get_compiled_graph()

    inputs = {
        "messages": [HumanMessage(content=message)],
        "original_message": message,
    }
    config: RunnableConfig = {
        "configurable": {"thread_id": conversation_id_str,},
    }

    # 新对话：创建 conversation 记录（并发生成标题）
    if is_new_conversation:
        title_task = asyncio.create_task(Graph.generate_title_async(message))
        title = await title_task
        await conv_crud.create_conversation(conversation_id_str, title, db)

    # 存储用户消息
    await conv_crud.add_chat_message(conversation_id_str, "user", message, [], db)

    async def event_stream():
        """SSE 事件流生成器"""
        final_answer = None
        parent_doc_ids = []

        for mode, chunk in graph.stream(
            inputs,
            config,
            stream_mode=["custom", "updates", "messages"],
        ):
            if mode == "messages":
                # tuple of (message, metadata)
                msg, metadata = chunk
                
                if ("stream_answer" in metadata.get("tags", []) and
                        metadata["langgraph_node"] in ["decide_retrieve_or_respond", "generate_answer"]):
                    # 只处理来自这两个节点的消息，且包含 "stream_answer" 标签
                    if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                        # AIMessage 且非工具调用且有内容，认为是 LLM 生成的回答，进行流式输出
                        sse_data = json.dumps({
                            "event": "Token streaming.",
                            "status": "progress",
                            "token": msg.content,
                        }, ensure_ascii=False)
                        yield f"data: {sse_data}\n\n"


            elif mode == "custom":
                # writer 发出的自定义事件 (dict)
                sse_data = json.dumps(chunk, ensure_ascii=False)
                yield f"data: {sse_data}\n\n"

            elif mode == "updates":
                # 节点状态更新: chunk 格式为 {node_name: {state_updates}}
                for node_name, updates in chunk.items():
                    # updates是状态更新信息，可能为空（即没有状态更新），需要判断
                    if not updates or not isinstance(updates, dict):
                        continue
                    # 提取检索到的文档 ID
                    if "documents" in updates:
                        for doc in updates["documents"]:
                            if hasattr(doc, "id") and doc.id and doc.id not in parent_doc_ids:
                                parent_doc_ids.append(doc.id)
                    # 提取最终 AI 回复
                    if "messages" in updates:
                        for msg in updates["messages"]:
                            if isinstance(msg, AIMessage) and msg.content:
                                final_answer = msg.content

        # ---- graph 执行完毕，持久化对话历史 ----

        # 存储 AI 回答，发送最终回答
        if final_answer:
            await conv_crud.add_chat_message(conversation_id_str, "ai", final_answer, parent_doc_ids, db)
            sse_data = json.dumps({
                "event": "Answer generated.",
                "status": "finished",
                "answer": final_answer,
                "thread_id": conversation_id_str,
            }, ensure_ascii=False)
            yield f"data: {sse_data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/list")
async def list_conversations(db: AsyncSession = Depends(DatabaseManager.get_db)):
    """
    获取所有对话列表
    """
    conversations = await conv_crud.list_conversations(db)
    data = [ConversationVO.model_validate(conv) for conv in conversations]
    return Result(code=200, message="对话列表获取成功", data=data)


@router.get("/history")
async def get_history(
    conversation_id: UUID = Query(..., description="对话 ID"),
    db: AsyncSession = Depends(DatabaseManager.get_db),
):
    """
    获取指定对话的聊天记录（从 chat_history 表读取，非 graph state）

    :param conversation_id: 对话 ID
    :param db:
    :return: 聊天记录列表
    """
    conversation_id_str = str(conversation_id)
    history = await conv_crud.get_chat_history(conversation_id_str, db)
    
    messages = [ChatMessageVO.model_validate(msg) for msg in history]
    data = ChatHistoryResponseVO(conversation_id=conversation_id_str, messages=messages)
    
    return Result(code=200, message="聊天记录获取成功", data=data)


@router.put("/{conversation_id}/title")
async def update_title(
    conversation_id: UUID = Path(..., description="对话 ID"),
    title: str = Body(..., embed=True),
    db: AsyncSession = Depends(DatabaseManager.get_db),
):
    """
    修改对话标题
    """
    conv = await conv_crud.update_conversation_title(str(conversation_id), title, db)
    return Result(code=200, message="标题修改成功", data={"id": conv.id, "title": conv.title})


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID = Path(..., description="对话 ID"),
    db: AsyncSession = Depends(DatabaseManager.get_db),
):
    """
    删除对话（含聊天记录和 graph checkpoint 数据）
    """
    await conv_crud.delete_conversation(str(conversation_id), db)
    return Result(code=200, message="对话删除成功", data=None)