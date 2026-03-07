"""
LangGraph 对话流程 — Agentic RAG 反思式生成。

流程:
  START → summarize_conversation → decide_retrieve_or_respond
    → (有 tool_calls) → retrieve_node → grade_documents
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
from psycopg_pool import ConnectionPool
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import tools_condition

from app.core.retriever import HybridPDRetrieverFactory
from app.config.global_config import global_config
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


# ======================== Retrieve Tool 定义 ========================
# 仅用于 LLM bind_tools，实际执行由 Graph.retrieve_node 完成


@tool(description="Retrieve relevant documents from the document store based on the query.")
def retrieve(query: str) -> str:
    """从文档库中检索相关信息以帮助回答用户问题。当用户的问题需要依据文档知识来回答时，调用此工具。"""
    # 实际执行逻辑在 Graph.retrieve_node 中，此处不会被直接调用
    pass


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
    "当用户的问题需要查阅文档知识库时，请调用 retrieve 工具检索相关文档。\n"
    "对于日常问候、闲聊或你已知的通用知识，请直接回答，无需检索。"
)

GENERATE_TITLE_PROMPT = (
    "用户正在和大语言模型进行一轮新的对话，"
    "请根据以下用户消息，直接生成一个简短的对话标题（不超过20个字，不要加引号）：\n\n"
    "{message}"
)


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
    _retrieve_tool = retrieve  # 仅用于 LLM bind_tools 的 tool schema

    # ---- 运行时资源（在 build() 中初始化） ----
    _response_llm = None   # 用于生成查询/回答
    _light_llm = None     # 用于结构化输出评分等
    _chat_config: dict = {}  # chat 配置

    # stream output
    _event_description = "event"  # 当前的事件描述（如“正在生成回答...”）
    _status_key = "status" # 当前的状态["progress", "finished"]

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
        system_content = DECIDE_SYSTEM_PROMPT
        if summary:
            system_content += f"\n\n以下是之前对话的摘要，请参考:\n{summary}"
        system_messages = [SystemMessage(content=system_content)]

        response = cls._response_llm.bind_tools([cls._retrieve_tool]).invoke(system_messages + messages)
        return {"messages": [response]}

    @classmethod
    def retrieve_node(cls, state: GraphState):
        """
        自定义检索节点（替代 ToolNode）。
        从 LLM 返回的最后一条消息中提取 tool_calls，执行检索，
        同时将 Document 原始对象存入 state["documents"]，将序列化内容存入 ToolMessage。
        """
        writer = get_stream_writer()

        last_message = state["messages"][-1]
        results = []
        all_docs = []

        for tool_call in last_message.tool_calls:
            if tool_call["name"] != cls._retrieve_tool.name:
                print(f"GRAPH: retrieve_node: Unknown tool '{tool_call['name']}', skipping.")
                results.append(ToolMessage(
                    content=f"Error: Unknown tool '{tool_call['name']}'",
                    tool_call_id=tool_call["id"],
                ))
                continue

            query = tool_call["args"].get("query", "")
            print(f"GRAPH: retrieve_node: Executing retrieval for query='{query[:20]}...'")
            writer({cls._event_description: f"Searching documentation for '{query[:20]}'...",
                    cls._status_key: "progress"})

            # 调用 HybridPDRetriever 获取父文档列表
            retriever = HybridPDRetrieverFactory.get_instance()
            docs = retriever.invoke(query)
            all_docs.extend(docs)

            # 序列化为字符串作为 ToolMessage 内容
            serialized = _serialize_docs(docs)
            results.append(ToolMessage(
                content=serialized,
                tool_call_id=tool_call["id"],
            ))

        return {
            "messages": results,
            "documents": all_docs,  # 将原始 Document 对象存入 state
        }

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
    def summarize_conversation(cls, state: GraphState):
        """
        对话摘要节点: 当 messages 过多时，摘要旧消息以管理短期记忆。
        保留最近 2 条消息，其余摘要后删除。
        仅在 messages 数量超过 message_summarize_threshold 时执行摘要。
        """
        threshold = cls._chat_config.get("message_summarize_threshold", 10)
        messages = state["messages"]
        if len(messages) <= threshold:
            return {}

        writer = get_stream_writer()
        writer({cls._event_description: "Summarizing conversation history...", cls._status_key: "progress"})

        # 获取已有摘要
        existing_summary = state.get("summary", "")
        if existing_summary:
            summary_request = (
                f"这是之前对话的摘要: {existing_summary}\n\n"
                "请结合以上新消息扩展该摘要:"
            )
        else:
            summary_request = SUMMARIZE_CONVERSATION_PROMPT

        # 让 LLM 生成摘要
        summary_messages = messages + [HumanMessage(content=summary_request)]
        response = cls._response_llm.invoke(summary_messages)

        # 删除除最近 2 条以外的所有消息
        delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]

        return {
            "summary": response.content,
            "messages": delete_messages,
        }

    # ======================== Conditional Edges ========================

    @classmethod
    def grade_documents(cls, state: GraphState) -> Literal["generate_answer", "rewrite_question"]:
        """
        评估检索到的文档是否与用户问题相关。
        超过最大重试次数时直接进入 generate_answer。
        """
        writer = get_stream_writer()
        writer({cls._event_description: "Evaluating document relevance...", cls._status_key: "progress"})

        max_rewrite = cls._chat_config.get("max_rewrite_time", 3)
        if state.get("rewrite_count", 0) >= max_rewrite:
            print(f"GRAPH: rewrite_count ({state['rewrite_count']}) >= max_rewrite_time ({max_rewrite}), "
                  f"forcing generate_answer.")
            return "generate_answer"

        question = state.get("original_message", state["messages"][0].content)
        context = state["messages"][-1].content

        prompt = GRADE_DOCUMENTS_PROMPT.format(question=question, context=context)
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
    def build(cls) -> CompiledStateGraph:
        """
        构建并编译 LangGraph StateGraph。
        从 global_config 读取配置，初始化 LLM 和 Checkpointer，构建图。
        必须在 global_config.load() 之后调用（通常在 FastAPI lifespan 中）。

        Graph 结构:
          START → summarize_conversation → decide_retrieve_or_respond
            → (tools_condition) → retrieve_node | END
          retrieve_node → (grade_documents) → generate_answer | rewrite_question
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

        # ---- 初始化 LLM ----
        cls._init_llm()

        # ---- 初始化 Checkpointer ----
        cls._init_checkpointer()

        # ---- 构建图 ----
        print("GRAPH: Building graph...")
        workflow = StateGraph(GraphState)

        # ---- 添加节点 ----
        workflow.add_node("summarize_conversation", cls.summarize_conversation)
        workflow.add_node("decide_retrieve_or_respond", cls.decide_retrieve_or_respond)
        workflow.add_node("retrieve", cls.retrieve_node)  # 自定义检索节点
        workflow.add_node("generate_answer", cls.generate_answer)
        workflow.add_node("rewrite_question", cls.rewrite_question)
        workflow.add_node("check_usefulness_node", cls._passthrough)

        # ---- 添加边 ----

        # START → summarize_conversation → decide_retrieve_or_respond
        workflow.add_edge(START, "summarize_conversation")
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

        # retrieve → (grade_documents) → generate_answer 或 rewrite_question
        workflow.add_conditional_edges(
            "retrieve",
            cls.grade_documents,
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
