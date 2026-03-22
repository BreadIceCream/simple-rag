"""
LangGraph 对话流程 — Agentic RAG 反思式生成。

流程:
  START → summarize_conversation → decide_retrieve_or_respond
    → (有 tool_calls) → retrieve_node → handle_retrieve_result
        → (检索异常) → decide_retrieve_or_respond
        → (相关) → generate_answer → check_hallucination
            → (无幻觉) → check_usefulness
                → (已解决) → END
                → (未解决) → rewrite_question → decide_retrieve_or_respond
            → (有幻觉) → generate_answer
        → (不相关) → rewrite_question → decide_retrieve_or_respond
    → (无 tool_calls, 直接回答) → END

注意: 所有配置读取和 LLM 初始化均在 Graph.build() 中执行，
     因为 global_config 在 FastAPI lifespan 中才初始化。
"""
import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, RemoveMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition, ToolRuntime, ToolNode
from langgraph.types import Command
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, Field

from app.config.db_config import DatabaseManager
from app.config.global_config import global_config
from app.core.retriever import HybridPDRetrieverFactory, EnhancedParentDocumentRetrieverFactory
from app.models.state import GraphState, GradeDocuments, HallucinationCheck, UsefulnessCheck

# ======================== 元数据过滤 ========================

# retrieve tool 返回的父文档中需要保留的 metadata 字段
KEPT_METADATA_KEYS = {"file_id", "file_extension", "parent_index", "Header 1"}


def _filter_metadata(metadata: dict) -> dict:
    """过滤父文档元数据，只保留对 LLM 有用的字段。"""
    return {k: v for k, v in metadata.items() if k in KEPT_METADATA_KEYS}


def _serialize_docs(docs: list) -> str:
    """将 Document 列表序列化为字符串，供 ToolMessage 和 Prompt 使用。"""
    if not docs:
        return "未检索到相关文档。"
    return "\n\n".join(
        f"Metadata: {_filter_metadata(doc.metadata)}\nContent: {doc.page_content}"
        for doc in docs
    )


# ======================== Tool Schema 定义 ========================


class GetDocsPageInput(BaseModel):
    file_id: str = Field(..., description="The ID of the file to retrieve documents for. UUID format.")
    offset: int = Field(0, ge=0, description="The starting position of the document list, inclusive.")
    limit: int = Field(5, gt=0, le=10, description="The number of documents to retrieve, maximum 10.")


# ======================== Retrieve Tool 定义 ========================
# 仅用于 LLM bind_tools，实际执行由 Graph.retrieve_node 完成


# ======================== Prompt Templates ========================

GRADE_DOCUMENTS_PROMPT = (
    "你是一个评估检索文档与用户问题相关性的评分员。\n"
    "以下是检索到的文档:\n\n{context}\n\n"
    "以下是用户问题: \n\n'{question}'\n\n"
    "如果文档包含与用户问题相关的关键词或语义信息，则评为相关。\n"
    "请直接给出 'yes' 或 'no' 的二元评分，表示文档是否与问题相关。"
)

REWRITE_QUESTION_PROMPT = (
    "你是一个问题重写器。请分析以下问题的语义意图，\n"
    "将其重写为一个更适合检索的优化问题。\n\n"
    "原始问题: '{question}'\n\n"
    "请直接输出优化后的问题:"
)

GENERATE_ANSWER_PROMPT = (
    "你是一个问答助手。请根据以下检索到的上下文来回答问题。\n"
    "如果你不知道答案，请直接说不知道。\n"
    "回答应当精炼准确。\n\n"
    "问题: \n'{question}'\n\n"
    "上下文: \n'{context}'"
)

CHECK_HALLUCINATION_PROMPT = (
    "你是一个评估 LLM 回答是否基于检索文档的评分员。\n"
    "以下是检索到的文档:\n\n{documents}\n\n"
    "以下是 LLM 的回答:\n\n'{generation}'\n\n"
    "请判断该回答是否基于/受支持于检索到的文档（无幻觉）。\n"
    "直接给出 'yes'（基于文档，无幻觉）或 'no'（存在幻觉）的评分。"
)

CHECK_USEFULNESS_PROMPT = (
    "你是一个评估 LLM 回答是否解决了用户问题的评分员。\n"
    "以下是用户问题: {question}\n\n"
    "以下是 LLM 的回答:\n\n{generation}\n\n"
    "请判断该回答是否充分解决了用户的问题。\n"
    "直接给出 'yes'（已解决）或 'no'（未解决）的评分。"
)

SUMMARIZE_CONVERSATION_PROMPT = (
    "请对以上对话内容进行简要总结，保留关键事实和用户偏好。"
)

DECIDE_SYSTEM_PROMPT = (
    "你是一个智能助手，可以回答用户的问题。\n"
    "当用户的问题需要查阅文档知识库时，请调用工具检索相关文档。\n"
    "对于日常问候、闲聊或你已知的通用知识，请直接回答，无需检索。\n"
    "可参考的文档信息: \n{file_info}\n"
)

GENERATE_TITLE_PROMPT = (
    "用户正在和大语言模型进行一轮新的对话，"
    "请根据以下用户消息，直接生成一个简短的对话标题（不超过20个字，不要加引号）：\n\n"
    "{message}"
)

TOOL_PLACEHOLDER = "This is a tool message and its content has been omitted."


# ======================== Graph 类 ========================


class Graph:
    """
    Agentic RAG Graph.
    将 retrieve tool、各 Node、Conditional Edges 和 Graph 构建逻辑封装在类中。

    所有配置读取和 LLM 初始化在 build() 中执行（因为 global_config 在 FastAPI lifespan 中才初始化）。
    Node / Edge 方法通过类变量 _response_llm / _light_llm / _chat_config 访问运行时资源。
    """

    _compiled_graph: CompiledStateGraph | None = None
    _connection_pool: ConnectionPool | None = None
    _checkpointer: PostgresSaver | None = None
    _retrieve_tools: list = []  # 用于检索的tools

    # ---- 运行时资源（在 build() 中初始化） ----
    _response_llm = None   # 用于生成查询/回答
    _light_llm = None     # 用于结构化输出评分等
    _chat_config: dict = {}  # chat 配置

    # stream output
    _event_description = "event"  # 当前的事件描述（如“正在生成回答...”）
    _status_key = "status" # 当前的状态["progress", "finished"]

    # ======================== Tools ========================

    @tool("retrieve", description="Retrieve relevant documents from the document store based on the query.")
    @staticmethod
    def _retrieve(query: str, runtime: ToolRuntime) -> Command:
        """从文档库中检索相关信息以帮助回答用户问题。当用户的问题需要依据文档知识来回答时，调用此工具。"""
        all_docs = []

        print(f"GRAPH: Tool retrieve: Executing retrieval for query='{query[:20]}...'")
        runtime.stream_writer({Graph._event_description: f"Searching documentation for '{query[:20]}'...",
                               Graph._status_key: "progress"})

        # 调用 HybridPDRetriever 获取父文档列表
        retriever = HybridPDRetrieverFactory.get_instance()
        docs = retriever.invoke(query)
        all_docs.extend(docs)

        # 序列化为字符串作为 ToolMessage 内容
        serialized = _serialize_docs(docs)

        return Command(
            update={
                "messages": [ToolMessage(
                    content=serialized,
                    tool_call_id=runtime.tool_call_id,
                )],
                "documents": all_docs,
            }
        )

    @tool("get_documents_by_file_id", description="Get the documents of the file based on the specific file_id with pagination support.", args_schema=GetDocsPageInput)
    @staticmethod
    def _get_docs_page(file_id: str, offset: int, limit: int, runtime: ToolRuntime) -> Command:
        """
        根据 file_id 分页获取文档内容。
        校验 file_id 合法性后，从数据库获取该文件的 parent_doc_ids 列表，
        按 offset/limit 分页取出对应的父文档内容并返回。
        """
        hybrid_retriever = HybridPDRetrieverFactory.get_instance()

        # 校验 file_id 是否在当前参考的文件中
        current_file_ids = hybrid_retriever.get_file_ids()
        if file_id not in current_file_ids:
            return Command(
                update={
                    "messages": [ToolMessage(
                        content=f"Error：文件 {file_id} 不在当前参考文档范围内.",
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )],
                }
            )

        runtime.stream_writer({Graph._event_description: f"Fetching documents for file '{file_id[:8]}...'...",
                               Graph._status_key: "progress"})

        # 同步获取文档的 parent_doc_ids
        from app.crud.document import get_document_by_id_sync
        doc = get_document_by_id_sync(file_id)

        total = len(doc.parent_doc_ids)
        if total == 0:
            return Command(
                update={
                    "messages": [ToolMessage(
                        content=f"文件 {file_id} 没有父文档分块（parent_doc_ids 为空）。",
                        tool_call_id=runtime.tool_call_id,
                    )],
                }
            )

        # 分页获取父文档
        ids_to_search = doc.parent_doc_ids[offset: offset + limit]
        pd_retriever = EnhancedParentDocumentRetrieverFactory.get_instance()
        parent_docs_tuple = pd_retriever.get_parent_docs(ids_to_search)
        parent_docs = [parent_doc for _, parent_doc in parent_docs_tuple]

        print(f"GRAPH: Tool get_docs_page: file_id='{file_id}', total={total}, offset={offset}, limit={limit}, fetched={len(parent_docs)}")

        # 序列化结果
        if parent_docs:
            docs_text = _serialize_docs(parent_docs)
            content = f"文件 {file_id} 的父文档分块（共 {total} 个，当前 offset={offset}, limit={limit}）:\n\n{docs_text}"
        else:
            content = f"文件 {file_id} 的 offset={offset} 超出范围（共 {total} 个父文档分块）。"

        return Command(
            update={
                "messages": [ToolMessage(
                    content=content,
                    tool_call_id=runtime.tool_call_id,
                )],
                "documents": parent_docs,
            }
        )

    # ======================== Nodes ========================

    @classmethod
    def decide_retrieve_or_respond(cls, state: GraphState):
        """
        入口节点: LLM 决定是调用 retrieve tool 检索文档，还是直接回答用户。
        通过 bind_tools 绑定 retrieve tool schema，LLM 自行判断是否需要检索。
        如果有历史摘要，将其作为系统消息注入上下文。
        """
        writer = get_stream_writer()
        writer({cls._event_description: "Deciding to retrieve documents or respond directly...", cls._status_key: "progress"})

        messages = state["messages"]

        # 系统提示词（含历史摘要）
        summary = state.get("summary", "")
        current_file_infos = HybridPDRetrieverFactory.get_instance().get_file_infos()
        system_content = DECIDE_SYSTEM_PROMPT.format(file_info=current_file_infos)
        if summary:
            system_content += f"\n\n以下是之前对话的摘要，请参考:\n{summary}"
        system_messages = [SystemMessage(content=system_content)]

        response = cls._response_llm.bind_tools(cls._retrieve_tools).invoke(system_messages + messages)
        return {"messages": [response]}

    @classmethod
    def generate_answer(cls, state: GraphState):
        """
        基于检索到的文档生成最终回答。
        使用 original_message 作为问题（不受 rewrite 影响）。
        """
        writer = get_stream_writer()
        writer({cls._event_description: "Generating answer based on retrieved documents...", cls._status_key: "progress"})

        messages = state["messages"]
        question = state.get("original_message", messages[0].content)

        # 上下文: 最近一条 ToolMessage 的内容
        context = ""
        for m in reversed(messages):
            if hasattr(m, "type") and m.type == "tool":
                context = m.content
                break

        prompt = GENERATE_ANSWER_PROMPT.format(question=question, context=context)
        response = cls._response_llm.invoke([HumanMessage(content=prompt)])
        return {
            "messages": [response],
            "generate_count": state.get("generate_count", 0) + 1,
        }

    @classmethod
    def rewrite_question(cls, state: GraphState):
        """
        问题重写节点: 当检索文档不相关或回答无用时，重写问题以优化检索。
        递增 rewrite_count 防止无限循环。使用 original_message 作为重写基准。
        """
        writer = get_stream_writer()

        question = state.get("original_message", state["messages"][0].content)

        prompt = REWRITE_QUESTION_PROMPT.format(question=question)
        response = cls._response_llm.invoke([HumanMessage(content=prompt)])

        writer({cls._event_description: "Rewriting question to improve retrieval...", cls._status_key: "progress"})

        return {
            "messages": [HumanMessage(content=response.content)],
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    @classmethod
    def init_graph_state(cls, state: GraphState):
        """
        初始化 state，重新设置 documents, rewrite_count, generate_count
        :param state:
        :return:
        """
        writer = get_stream_writer()
        writer({cls._event_description: "Init graph state...", cls._status_key: "progress"})
        return {"documents": [], "rewrite_count": 0, "generate_count": 0}

    @classmethod
    def compact_tool_messages(cls, state: GraphState):
        """
        替换历史轮次中的 ToolMessage 内容为占位文本。
        每次进入该节点时都会执行，最新一轮的 ToolMessage 不替换。
        一轮对话 = 一条 HumanMessage 及其后续所有非 HumanMessage 消息。
        """
        writer = get_stream_writer()
        writer({cls._event_description: "Compact previous tool messages...", cls._status_key: "progress"})

        messages = state["messages"]

        # 替换历史轮次的所有 ToolMessage 内容
        replaced_tool_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.content != TOOL_PLACEHOLDER:
                replaced_tool_messages.append(ToolMessage(
                    content=TOOL_PLACEHOLDER,
                    name=msg.name,
                    tool_call_id=msg.tool_call_id,
                    id=msg.id,
                ))

        if replaced_tool_messages:
            return {"messages": replaced_tool_messages}
        return {}


    @classmethod
    def summarize_conversation(cls, state: GraphState):
        """
        对话摘要节点: 当对话轮次超过 conversation_summarize_threshold 时，
        摘要旧消息以管理短期记忆。保留最近 2 轮原始消息，仅压缩更早的轮次。
        SystemMessage 不参与轮次计数和摘要。
        """
        messages = state["messages"]

        # 1. 剔除 SystemMessage，按 HumanMessage 分割对话轮次
        non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        rounds: list[list] = []
        for msg in non_system_msgs:
            if isinstance(msg, HumanMessage):
                rounds.append([msg])
            elif rounds:
                rounds[-1].append(msg)

        conversation_count = len(rounds)
        threshold = cls._chat_config.get("conversation_summarize_threshold", 10)
        if conversation_count <= threshold:
            return {}

        writer = get_stream_writer()
        writer({cls._event_description: "Summarizing conversation history...", cls._status_key: "progress"})

        # 2. 划分：需要压缩的轮次 vs 保留的最近 2 轮
        rounds_to_summarize = rounds[:-2]

        # 3. 构建摘要输入，ToolMessage 内容替换为占位符
        # 要 summarize 的消息和要 delete 相同
        msgs_to_summarize = []
        msgs_to_delete = []
        for r in rounds_to_summarize:
            msgs_to_summarize.extend(r)
            msgs_to_delete.extend(r)

        existing_summary = state.get("summary", "")
        if existing_summary:
            summary_request = (
                f"这是之前对话的摘要: {existing_summary}\n\n"
                "请结合以上新消息扩展该摘要:"
            )
        else:
            summary_request = SUMMARIZE_CONVERSATION_PROMPT

        msgs_to_summarize.append(HumanMessage(content=summary_request))
        response = cls._response_llm.invoke(msgs_to_summarize)

        # 4. 删除被压缩轮次的消息
        delete_messages = [RemoveMessage(id=m.id) for m in msgs_to_delete]

        print(f"GRAPH: summarize_conversation: {conversation_count} rounds, "
              f"summarized {len(rounds_to_summarize)} rounds, "
              f"deleted {len(msgs_to_delete)} messages")

        return {
            "summary": response.content,
            "messages": delete_messages,
        }

    # ======================== Conditional Edges ========================

    @classmethod
    def handle_retrieve_result(cls, state: GraphState) -> Literal["decide_retrieve_or_respond", "generate_answer", "rewrite_question"]:
        """
        处理检索结果，如果检索出现异常则回到decide_retrieve_or_respond节点，否则评估检索到的文档是否与用户问题相关。
        超过最大重试次数时直接进入 generate_answer。
        """
        writer = get_stream_writer()

        tool_message = state["messages"][-1]
        if tool_message.status == "error":
            writer({cls._event_description: "Tool execution error...", cls._status_key: "progress"})
            return "decide_retrieve_or_respond"

        writer({cls._event_description: "Evaluating document relevance...", cls._status_key: "progress"})

        max_rewrite = cls._chat_config.get("max_rewrite_time", 3)
        if state.get("rewrite_count", 0) >= max_rewrite:
            print(f"GRAPH: rewrite_count ({state['rewrite_count']}) >= max_rewrite_time ({max_rewrite}), "
                  f"forcing generate_answer.")
            return "generate_answer"

        question = state.get("original_message", state["messages"][0].content)

        prompt = GRADE_DOCUMENTS_PROMPT.format(question=question, context=tool_message.content)
        response = cls._light_llm.with_structured_output(GradeDocuments).invoke(
            [HumanMessage(content=prompt)]
        )

        if response.binary_score == "yes":
            print("GRAPH: grade_documents → 相关，进入 generate_answer")
            writer({cls._event_description: "Documents are relevant.", cls._status_key: "progress"})
            return "generate_answer"
        else:
            print("GRAPH: grade_documents → 不相关，进入 rewrite_question")
            writer({cls._event_description: "Documents not relevant, rewriting question...", cls._status_key: "progress"})
            return "rewrite_question"

    @classmethod
    def check_hallucination(cls, state: GraphState) -> Literal["check_usefulness", "generate_answer"]:
        """
        检查 LLM 生成的回答是否存在幻觉（是否基于检索到的文档）。
        超过最大重新生成次数时强制进入 check_usefulness。
        """
        writer = get_stream_writer()
        writer({cls._event_description: "Checking for hallucination...", cls._status_key: "progress"})

        # 防无限循环: 生成次数>=上限时强制通过
        max_generate = cls._chat_config.get("max_generate_time", 2)
        if state.get("generate_count", 0) >= max_generate:
            print(f"GRAPH: generate_count ({state['generate_count']}) > max_generate_time ({max_generate}), "
                  f"forcing check_usefulness.")
            writer({cls._event_description: "Max regeneration attempts reached, proceeding with current answer.", cls._status_key: "progress"})
            return "check_usefulness"

        messages = state["messages"]
        generation = messages[-1].content

        documents = ""
        for m in reversed(messages):
            if hasattr(m, "type") and m.type == "tool":
                documents = m.content
                break

        prompt = CHECK_HALLUCINATION_PROMPT.format(documents=documents, generation=generation)
        response = cls._light_llm.with_structured_output(HallucinationCheck).invoke(
            [HumanMessage(content=prompt)]
        )

        if response.binary_score == "yes":
            print("GRAPH: check_hallucination → 无幻觉，进入 check_usefulness")
            return "check_usefulness"
        else:
            print("GRAPH: check_hallucination → 存在幻觉，重新 generate_answer")
            writer({cls._event_description: "Hallucination detected, regenerating answer...", cls._status_key: "progress"})
            return "generate_answer"

    @classmethod
    def check_usefulness(cls, state: GraphState) -> Literal["__end__", "rewrite_question"]:
        """
        检查 LLM 的回答是否解决了用户的问题。
        超过最大重试次数时直接结束。
        """
        writer = get_stream_writer()
        writer({cls._event_description: "Checking answer usefulness...", cls._status_key: "progress"})

        max_rewrite = cls._chat_config.get("max_rewrite_time", 3)
        if state.get("rewrite_count", 0) >= max_rewrite:
            print(f"GRAPH: rewrite_count ({state['rewrite_count']}) >= max_rewrite_time ({max_rewrite}), "
                  f"forcing END.")
            return "__end__"

        question = state.get("original_message", state["messages"][0].content)
        generation = state["messages"][-1].content

        prompt = CHECK_USEFULNESS_PROMPT.format(question=question, generation=generation)
        response = cls._light_llm.with_structured_output(UsefulnessCheck).invoke(
            [HumanMessage(content=prompt)]
        )

        if response.binary_score == "yes":
            print("GRAPH: check_usefulness → 已解决，结束")
            writer({cls._event_description: "Answer verified, finishing...", cls._status_key: "finished"})
            return "__end__"
        else:
            print("GRAPH: check_usefulness → 未解决，进入 rewrite_question")
            writer({cls._event_description: "Answer insufficient, retrying...", cls._status_key: "progress"})
            return "rewrite_question"

    # ======================== Passthrough Node ========================

    @staticmethod
    def _passthrough(state: GraphState):
        """透传节点，不修改状态。用于在两个 conditional edge 之间做中转。"""
        return {}

    # ======================== 初始化与构建 ========================

    @classmethod
    def _init_llm(cls):
        """
        从 global_config 读取配置并初始化 LLM 实例。
        必须在 global_config.load() 之后调用。
        """
        if cls._response_llm is not None:
            return

        chat_model_config = global_config.get("chat_model", {})
        default_model = chat_model_config.get("default")
        light_model = chat_model_config.get("light")

        print(f"GRAPH: Initializing LLMs (default={default_model}, light={light_model})...")

        # response_llm: 用于生成查询/回答（可绑定 tools）
        cls._response_llm = init_chat_model(
            default_model,
            configurable_fields=("model", "model_provider", "temperature", "tags"),
            tags=["stream_answer"],
        )

        # light_llm: 用于结构化输出评分等简单任务（grade / hallucination / usefulness）
        cls._light_llm = init_chat_model(
            light_model,
            configurable_fields=("model", "model_provider", "temperature"),
            temperature=0.2
        )

        print("GRAPH: LLMs initialized successfully.")

    @classmethod
    def _init_checkpointer(cls):
        """
        从 global_config 读取数据库 URL，初始化 PostgresSaver checkpointer。
        将 asyncpg 驱动替换为 psycopg 驱动。
        需要安装: pip install "psycopg[binary,pool]" langgraph-checkpoint-postgres
        """
        if cls._checkpointer is not None:
            return

        db_url = global_config.get("database", {}).get("url", "")
        db_uri = db_url.replace("postgresql+asyncpg://", "postgresql://")

        print(f"GRAPH: Initializing PostgresSaver checkpointer...")
        cls._connection_pool = ConnectionPool(conninfo=db_uri, max_size=20, kwargs={"autocommit": True})
        cls._checkpointer = PostgresSaver(cls._connection_pool)
        # 首次使用时需要调用 setup() 创建必要的表
        cls._checkpointer.setup()
        print("GRAPH: PostgresSaver checkpointer initialized.")

    @classmethod
    def _init_tools(cls):
        """
        初始化工具列表。将所有 @tool 定义的类方法收集到 _retrieve_tools 中，
        供 ToolNode 和 bind_tools 使用。
        """
        cls._retrieve_tools = [cls._retrieve, cls._get_docs_page]
        print(f"GRAPH: Initialized {len(cls._retrieve_tools)} tools: {[t.name for t in cls._retrieve_tools]}")


    @classmethod
    def build(cls) -> CompiledStateGraph:
        """
        构建并编译 LangGraph StateGraph。
        从 global_config 读取配置，初始化 LLM 和 Checkpointer，构建图。
        必须在 global_config.load() 之后调用（通常在 FastAPI lifespan 中）。

        Graph 结构:
          START → summarize_conversation → decide_retrieve_or_respond
            → (tools_condition) → retrieve_node | END
          retrieve_node → (handle_retrieve_result) → decide_retrieve_or_respond| generate_answer | rewrite_question
          generate_answer → (check_hallucination) → check_usefulness_node | generate_answer
          check_usefulness_node → (check_usefulness) → END | rewrite_question
          rewrite_question → decide_retrieve_or_respond

        使用示例（在 FastAPI lifespan 中）:
            Graph.build()
            # 之后在路由中:
            graph = Graph.get_compiled_graph()
            result = graph.invoke(
                {"messages": [{"role": "user", "content": "..."}], "original_message": "..."},
                {"configurable": {"conversation_id": "session_123"}},
            )
        """
        if cls._compiled_graph is not None:
            print("GRAPH: Returning cached compiled graph.")
            return cls._compiled_graph

        # ---- 读取配置 ----
        cls._chat_config = global_config.get("chat", {})
        print(f"GRAPH: chat config loaded: max_rewrite_time={cls._chat_config.get('max_rewrite_time', 3)}, "
              f"max_generate_time={cls._chat_config.get('max_generate_time', 2)}, "
              f"message_summarize_threshold={cls._chat_config.get('message_summarize_threshold', 10)}")

        # ---- 初始化 Tools ----
        cls._init_tools()

        # ---- 初始化 LLM ----
        cls._init_llm()

        # ---- 初始化 Checkpointer ----
        cls._init_checkpointer()

        # ---- 构建图 ----
        print("GRAPH: Building graph...")
        workflow = StateGraph(GraphState)

        # ---- 添加节点 ----
        # 3个预处理节点
        workflow.add_node("init_graph_state", cls.init_graph_state)
        workflow.add_node("compact_tool_messages", cls.compact_tool_messages)
        workflow.add_node("summarize_conversation", cls.summarize_conversation)
        # 5个核心节点
        workflow.add_node("decide_retrieve_or_respond", cls.decide_retrieve_or_respond)
        workflow.add_node("retrieve", ToolNode(cls._retrieve_tools, handle_tool_errors=True))
        workflow.add_node("generate_answer", cls.generate_answer)
        workflow.add_node("rewrite_question", cls.rewrite_question)
        workflow.add_node("check_usefulness_node", cls._passthrough)

        # ---- 添加边 ----

        # START → init_graph_state → compact_tool_messages → summarize_conversation → decide_retrieve_or_respond
        workflow.add_edge(START, "init_graph_state")
        workflow.add_edge("init_graph_state", "compact_tool_messages")
        workflow.add_edge("compact_tool_messages", "summarize_conversation")
        workflow.add_edge("summarize_conversation", "decide_retrieve_or_respond")

        # decide_retrieve_or_respond → (tools_condition) → retrieve 或直接回答 END
        workflow.add_conditional_edges(
            "decide_retrieve_or_respond",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            },
        )

        # retrieve → (handle_retrieve_result) → decide_retrieve_or_respond 或 generate_answer 或 rewrite_question
        workflow.add_conditional_edges(
            "retrieve",
            cls.handle_retrieve_result,
        )

        # generate_answer → (check_hallucination) → check_usefulness_node 或 generate_answer
        workflow.add_conditional_edges(
            "generate_answer",
            cls.check_hallucination,
            {
                "check_usefulness": "check_usefulness_node",
                "generate_answer": "generate_answer",
            },
        )

        # check_usefulness_node → (check_usefulness) → END 或 rewrite_question
        workflow.add_conditional_edges(
            "check_usefulness_node",
            cls.check_usefulness,
        )

        # rewrite_question → decide_retrieve_or_respond（跳过 summarize，因为是内部循环）
        workflow.add_edge("rewrite_question", "decide_retrieve_or_respond")

        # 编译，传入内部初始化的 checkpointer
        cls._compiled_graph = workflow.compile(
            checkpointer=cls._checkpointer,
        )

        print("GRAPH: Graph built and compiled successfully.")

        # 将图绘制到项目根目录的 graph.md 中
        from pathlib import Path
        rag_root = Path(__file__).resolve().parent.parent.parent
        with open(rag_root / "graph.md", "w", encoding="utf-8") as f:
            f.write(cls._compiled_graph.get_graph().draw_mermaid())
            print(f"GRAPH: Graph visualization saved to {rag_root / 'graph.md'}")

        return cls._compiled_graph

    @classmethod
    def get_compiled_graph(cls) -> CompiledStateGraph:
        """获取已编译的 Graph 实例。未构建时抛出异常。"""
        if cls._compiled_graph is None:
            raise RuntimeError("Graph has not been built. Call Graph.build() first.")
        return cls._compiled_graph

    @classmethod
    def close(cls):
        """
        关闭 checkpointer 连接。
        应在 FastAPI lifespan shutdown 时调用。
        """
        if cls._connection_pool is not None:
            cls._connection_pool.close()
            cls._connection_pool = None
            cls._checkpointer = None
            print("GRAPH: PostgresSaver checkpointer closed.")

    @classmethod
    def delete_thread(cls, thread_id: str):
        """
        删除指定 thread 的所有 checkpoint 数据。
        在删除对话时调用，清理 graph 的短期记忆。
        """
        if cls._checkpointer is None:
            raise RuntimeError("Checkpointer not initialized.")
        cls._checkpointer.delete_thread(thread_id)
        print(f"GRAPH: Deleted thread '{thread_id}' from checkpointer.")

    @classmethod
    async def generate_title_async(cls, message: str) -> str:
        """
        调用 LLM 异步生成对话标题。
        使用 asyncio.to_thread 将同步 LLM 调用放到线程池中执行。
        """
        llm = cls._light_llm
        if llm is None:
            return message[:30]
        try:
            prompt = GENERATE_TITLE_PROMPT.format(message=message)
            response = await asyncio.to_thread(
                llm.invoke, [HumanMessage(content=prompt)]
            )
            title = response.content.strip()
            return title[:255] if title else message[:30]
        except Exception as e:
            print(f"GRAPH: Failed to generate title: {e}")
            return message[:30]
