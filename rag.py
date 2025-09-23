from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from flashrank import Ranker
from operator import add
from langgraph.graph import MessagesState
from typing import Annotated, Literal
import torch
import dotenv
import os


# 设置环境变量
def load_env():
    print("Loading environment variables...")
    dotenv.load_dotenv(override=True)
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
    if os.environ["LANGSMITH_TRACING"] == "true":
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")


# 创建LLM
def init_llm() -> ChatOpenAI:
    print("Initializing LLM...")
    model_name = os.getenv("MODEL_NAME")
    base_url = os.getenv("OPENAI_BASE_URL")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    if os.getenv("MODEL_PROVIDER") == "openai":
        return ChatOpenAI(
            model=model_name,
            temperature=0.3,
            verbose=True
        )
    else:
        return ChatOpenAI(
            model=model_name,
            temperature=0.3,
            verbose=True,
            base_url=base_url
        )

# 初始化嵌入模型
def init_embedding_model() -> HuggingFaceEmbeddings:
    print("Initializing embedding model...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": device})

# 初始化向量数据库
def init_vector_store(embeddings: HuggingFaceEmbeddings) -> Chroma:
    print("Initializing vector store...")
    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

# 初始化文本分词器
def init_text_splitter(chunk_size: int = 600, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    print("Initializing text splitter...")
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)

# 加载文档并获取retriever
def load_docs_and_get_retriever(text_splitter: RecursiveCharacterTextSplitter, vector_store: Chroma,
                                k: int = 50) -> dict[str, VectorStoreRetriever | list[str]]:
    """k: Amount of documents to return (Default: 50, to improve the precision rate)"""
    all_document_ids = []
    while True:
        file_path = input("Enter the file path, only supported .pdf (if ok, input 'done')：")
        if file_path == "done":
            break
        elif os.path.exists(file_path):
            print("Loading documents...")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_splits = text_splitter.split_documents(docs)
            document_ids = vector_store.add_documents(documents=all_splits)
            all_document_ids.extend(document_ids)
            print(f"Done! Added {len(all_splits)} documents to the vector store this time.")
        else:
            print("File not exist!")
    print(f"Added {len(all_document_ids)} documents to the vector store in total.\n"
          f"The number of documents in vector store is {vector_store._collection.count()}.\n"
          f"Retriever will retrieve {k} original documents.")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return {"retriever": retriever, "all_document_ids": all_document_ids}


# 初始化压缩Retriever，内置rerank
def init_compression_retriever(base_retriever: VectorStoreRetriever, top_n: int = 7) -> ContextualCompressionRetriever:
    """top_n: Number of documents to return."""
    print("Initializing compression retriever...")
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
    compressor = FlashrankRerank(client=ranker, top_n=top_n)
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)


# 图状态
class OverallState(MessagesState):
    user_question: str
    retrieved_docs: Annotated[list[str], add]

class OutputState(MessagesState):
    retrieved_docs: Annotated[list[str], add]


# 输入Schema
class RetrieveInputSchema(BaseModel):
    query: str = Field(..., description="The query to retrieve.")


# 创建工具
@tool(description="Retrieve relevant information to help answer a query", args_schema=RetrieveInputSchema)
def retrieve(query: str):
    """Retrieve information to help answer a query.
    Args:
        query: string.The query to retrieve.
    """
    retrieved_docs = compression_retriever.invoke(input=query)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized


# 创建工具节点
def tool_node(state: OverallState):
    """Performs the tool"""
    result = []
    all_retrieved_docs = []
    for tool_call in state["messages"][-1].tool_calls:
        content = ""
        if tool_call["name"] == retrieve.name:
            serialized = retrieve.invoke(tool_call["args"])
            content = serialized
            all_retrieved_docs.append(serialized)
        result.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
    return {"messages": result, "retrieved_docs": all_retrieved_docs}


def should_use_tool(state: OverallState) -> Literal["tool_node", "__end__"]:
    """Decides whether to use the tools. And send the state to relevant nodes."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    else:
        return "__end__"


# 创建生成查询的Node
QUERY_PROMPT_STR = """
You are a helpful RAG assistant.You can decide to use the tools.You can rewrite user's original question and generate some augmented questions that are better for retrieval.

Here is the original user question:
{user_question}
"""
query_promptTemplate = PromptTemplate.from_template(QUERY_PROMPT_STR, partial_variables={"user_question": "你好"})
def generate_query(state: OverallState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    prompt = query_promptTemplate.invoke(input={"user_question": state["user_question"], "tools": tools})
    response = (llm.bind_tools(tools)).invoke(prompt)
    return {"messages": [response]}


# 创建生成答案的Node
ANSWER_PROMPT_STR = """
You are an assistant for question-answering tasks.Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Question: {question}
Context: {context}
"""
answer_prompt = PromptTemplate.from_template(ANSWER_PROMPT_STR)
def generate_answer(state: OverallState):
    """Call the model to generate a final answer based on the current state."""
    chain = answer_prompt | llm
    response = chain.invoke(input={"question": state["user_question"], "context": state["retrieved_docs"]})
    return {"messages": [response]}


# 绘制图：
def draw_graph(graph: CompiledStateGraph):
    display(Image(graph.get_graph().draw_mermaid_png()))



if __name__ == "__main__":
    load_env()
    llm = init_llm()
    embeddings = init_embedding_model()
    vector_store = init_vector_store(embeddings=embeddings)
    text_splitter = init_text_splitter()
    retriever_and_docs_ids = load_docs_and_get_retriever(text_splitter=text_splitter, vector_store=vector_store)
    retriever = retriever_and_docs_ids["retriever"]
    docs_ids = retriever_and_docs_ids["all_document_ids"]
    compression_retriever = init_compression_retriever(base_retriever=retriever)

    tools = [retrieve]
    tools_by_name = {tool.name: tool for tool in tools}

    # 创建Graph
    workflow = StateGraph(OverallState, output_schema=OutputState)

    workflow.add_node(generate_query)
    workflow.add_node(tool_node)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query")
    workflow.add_conditional_edges("generate_query", should_use_tool)
    workflow.add_edge("tool_node", "generate_answer")
    workflow.add_edge("generate_answer", END)

    agent = workflow.compile()

    # 运行
    while True:
        question = input("==================\nAsk your questions (input 'exit' to stop)：")
        if question == "exit":
            if docs_ids:
                delete = input("Do you want to delete the documents used in this session? If not, they will persist in the current folder. (y/n) ")
                while delete not in ["y", "n"]:
                    delete = input("Invalid input. Please enter 'y' or 'n': ")
                if delete == "y":
                    vector_store.delete(ids=docs_ids)
                    print("Documents deleted.")
                else:
                    print("Documents will persist in the current folder.")
            print("Bye!")
            break
        for chunk in agent.stream(
                input={"user_question": question},
        ):
            for node, update in chunk.items():
                print("Update from node", node)
                update["messages"][-1].pretty_print()
                print("\n\n")