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
from app.core.retriever import EnhancedParentDocumentRetrieverFactory
from app.crud import conversation as conv_crud, document
from app.exception.exception import CustomException
from app.models.common import Result, ConversationVO, ChatMessageVO, ChatHistoryResponseVO, ChatReferenceParentDocVO, \
    ChatReferenceVO, SseTokenVO, SseAnswerVO, SseErrorVO, SseDoneVO, SseConversationVO

router = APIRouter(prefix="/api/conversation", tags=["conversation"])


@router.post("/chat")
async def chat(
    message: str = Body(..., embed=True, description="用户输入的消息"),
    conversation_id: UUID = Body(None, embed=True, alias="conversationId", description="对话 ID，选填。为空时自动创建新对话"),
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
    }
    config: RunnableConfig = {
        "configurable": {"thread_id": conversation_id_str,},
    }

    # 新对话：创建 conversation 记录（并发生成标题）
    if is_new_conversation:
        title_task = asyncio.create_task(Graph.generate_title_async(message))
        title = await title_task
        new_conv = await conv_crud.create_conversation(conversation_id_str, title, db)

    # 检查是否需要从异常中恢复
    is_resuming = False
    if not is_new_conversation:
        conv = await conv_crud.get_conversation(conversation_id_str, db)
        if conv and conv.checkpoint_id:
            is_resuming = True
            config["configurable"]["checkpoint_id"] = conv.checkpoint_id

    # 存储用户消息，如果是恢复执行，说明已经是原来那条消息导致异常，消息已存储过，跳过存储
    if not is_resuming:
        await conv_crud.add_chat_message(conversation_id_str, "user", message, [], db)

    async def event_stream():
        """SSE 事件流生成器"""
        # 如果是新对话，在流的开头推送会话信息
        if is_new_conversation and new_conv:
            conv_vo = ConversationVO.model_validate(new_conv)
            sse_conv = SseConversationVO(conversation=conv_vo)
            yield f"data: {sse_conv.model_dump_json()}\n\n"

        final_answer = None
        parent_doc_ids = []  # 存储检索到的文档 ID 有序无重复列表
        parent_docs = [] # 存储检索到的文档对象列表，保持与 parent_doc_ids 顺序一致
        file_ids = [] # 存储涉及的文件 ID 有序无重复列表

        try:
            # 如果是异常恢复，inputs 传 None
            stream_inputs = None if is_resuming else inputs
            for mode, chunk in graph.stream(
                stream_inputs,
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
                            response_obj = SseTokenVO(token=msg.content)
                            yield f"data: {response_obj.model_dump_json()}\n\n"

                elif mode == "custom":
                    # writer 发出的自定义事件 (dict)
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                elif mode == "updates":
                    # 节点状态更新: chunk 格式为 {node_name: {state_updates}}
                    for node_name, updates in chunk.items():
                        # updates是状态更新信息，可能为空（即没有状态更新），需要判断
                        if not updates or not isinstance(updates, dict):
                            continue
                        # 提取检索到的文档 ID 和 文档，并添加 file_id
                        if "documents" in updates:
                            for doc in updates["documents"]:
                                if hasattr(doc, "id") and doc.id and doc.id not in parent_doc_ids:
                                    parent_doc_ids.append(doc.id)
                                    parent_docs.append(doc)
                                    if doc.metadata["file_id"] not in file_ids:
                                        file_ids.append(doc.metadata["file_id"])
                        # 提取最终 AI 回复
                        if "messages" in updates:
                            for msg in updates["messages"]:
                                if isinstance(msg, AIMessage) and msg.content:
                                    final_answer = msg.content

            # ---- graph 执行完毕，持久化对话历史 ----

            # 存储 AI 回答，发送最终回答
            if final_answer:
                await conv_crud.add_chat_message(conversation_id_str, "ai", final_answer, parent_doc_ids, db)
                # 获取参考的文档信息，转成VO对象列表，以文件为单位，合并相同文件的文档列表（保持顺序），文档间也保持顺序（先出现的优先）
                reference_vo_list = []
                if parent_doc_ids and parent_docs and file_ids:
                    embedded_files = await document.get_document_by_ids(file_ids, db)
                    for file in embedded_files:
                        # 获取该文件检索到的父文档列表，保持顺序
                        file_parent_docs = [doc for doc in parent_docs if doc.metadata["file_id"] == file.id]
                        vos = [ChatReferenceParentDocVO(id=doc.id, parent_index=doc.metadata["parent_index"], content=doc.page_content)
                                for doc in file_parent_docs] if file_parent_docs else []
                        # 构造 ChatReferenceVO 对象，转成dict后添加到列表中以便 JSON 序列化
                        reference_vo_list.append(ChatReferenceVO(
                            id=file.id, path=file.path, file_name=file.file_name, parent_docs=vos
                        ).model_dump())

                response_obj = SseAnswerVO(
                    answer=final_answer,
                    references=reference_vo_list,
                    conversation_id=conversation_id_str
                )
                yield f"data: {response_obj.model_dump_json()}\n\n"

            yield f"data: {SseDoneVO().model_dump_json()}\n\n"
            # 正常完成，清空 checkpoint_id
            await conv_crud.update_conversation_checkpoint_id(conversation_id_str, None, db)

        except CustomException as e:
            # 捕获自定义异常（如未关联知识库等）并推送错误信息给前端
            error_obj = SseErrorVO(
                message=e.message,
                code=e.code,
                conversation_id=conversation_id_str
            )
            yield f"data: {error_obj.model_dump_json()}\n\n"
            
            # 保存异常时的 checkpoint_id
            try:
                state_snapshot = graph.get_state(config)
                if state_snapshot and state_snapshot.config:
                    checkpoint_id = state_snapshot.config.get("configurable", {}).get("checkpoint_id")
                    if checkpoint_id:
                        await conv_crud.update_conversation_checkpoint_id(conversation_id_str, checkpoint_id, db)
            except Exception as save_e:
                print(f"Failed to save checkpoint for {conversation_id_str}: {save_e}")
        except Exception as e:
            # 捕获其他未知异常
            error_obj = SseErrorVO(
                message=f"Graph 执行出错: {str(e)}",
                conversation_id=conversation_id_str
            )
            yield f"data: {error_obj.model_dump_json()}\n\n"
            
            # 保存异常时的 checkpoint_id
            try:
                state_snapshot = graph.get_state(config)
                if state_snapshot and state_snapshot.config:
                    checkpoint_id = state_snapshot.config.get("configurable", {}).get("checkpoint_id")
                    if checkpoint_id:
                        await conv_crud.update_conversation_checkpoint_id(conversation_id_str, checkpoint_id, db)
            except Exception as save_e:
                print(f"Failed to save checkpoint for {conversation_id_str}: {save_e}")

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
    message_vos = []
    for msg in history:
        msg_vo = ChatMessageVO.model_validate(msg)
        # 获取消息相关的参考资料列表
        reference_vos = []
        if msg.parent_doc_ids:
            pd_retriever = EnhancedParentDocumentRetrieverFactory.get_instance()
            parent_docs_tuple_list = pd_retriever.get_parent_docs(msg.parent_doc_ids)
            # 依次获取对应的file_id，file_ids有序，每个file_id对应的parent_doc_vos列表也有序（先出现的优先）
            file_ids = []
            file_id_to_parent_doc_vos = {}
            for _, parent_doc in parent_docs_tuple_list:
                file_id = parent_doc.metadata["file_id"]
                if file_id not in file_ids:
                    file_ids.append(file_id)
                pd_vo = ChatReferenceParentDocVO(
                    id=parent_doc.id,
                    parent_index=parent_doc.metadata["parent_index"],
                    content=parent_doc.page_content
                )
                # 添加到字典中对应的列表里
                if file_id in file_id_to_parent_doc_vos:
                    file_id_to_parent_doc_vos[file_id].append(pd_vo)
                else:
                    file_id_to_parent_doc_vos[file_id] = [pd_vo]
            # 根据file_id获取文件信息
            files = await document.get_document_by_ids(file_ids, db)

            for file in files:
                parent_doc_vos = file_id_to_parent_doc_vos.get(file.id, [])
                reference_vo = ChatReferenceVO(
                    id=file.id, path=file.path, file_name=file.file_name, parent_docs=parent_doc_vos
                )
                reference_vos.append(reference_vo)

        msg_vo.references = reference_vos
        message_vos.append(msg_vo)

    data = ChatHistoryResponseVO(conversation_id=conversation_id_str, messages=message_vos)
    
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