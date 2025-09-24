from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from flashrank import Ranker
from operator import add
from langgraph.graph import MessagesState
from typing import Annotated, Literal, Any, Coroutine
import nltk
import jieba
import asyncio
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
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
    load_env()
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
def init_vector_store() -> Chroma:
    embeddings = init_embedding_model()
    print("Initializing vector store...")
    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )

# 初始化文本分词器
async def init_text_splitter(chunk_size: int = 600, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    print("Initializing text splitter...")
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)


# bm25预处理函数
def nltk_resource_download():
    """Download NLTK resources"""
    print("Downloading NLTK resources...")
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")
stop_words = set(stopwords.words('english') + stopwords.words('chinese'))
punctuation = set(string.punctuation) | set("，。！？【】（）《》“”‘’：；、—…—")
lemmatizer = WordNetLemmatizer()
def is_english_word(s: str) -> bool:
    return s and 'a' <= s[0].lower() <= 'z'
def bilingual_preprocess_func(text: str) -> list[str]:
    """
    A powerful preprocessing function that supports mixed Chinese and English scenarios
    """
    # 1. transfer to lowercase(typically for English)
    text = text.lower()
    # 2. use jieba to cut text(useful for both Chinese and English)
    tokens = jieba.lcut(text, cut_all=False)
    processed_tokens = []
    for token in tokens:
        # 3. filter stop words and punctuation
        if token in stop_words or token in punctuation:
            continue
        # 4. lemmatize English words
        if is_english_word(token):
            token = lemmatizer.lemmatize(token)
        # 5. filter single character(mostly noises)
        if len(token) > 1:
            processed_tokens.append(token)
    return processed_tokens


# 加载文档并嵌入向量数据库
def load_doc_to_vector_store(text_splitter: RecursiveCharacterTextSplitter, vector_store: Chroma, file_path: str, task_id: int | None = None) -> \
dict[str, Exception | None | int] | dict[str, None | list[str] | int]:
    print(f"Loading task <{task_id}>, document: {file_path}...")
    try:
        start_time = time.time()
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        document_ids = vector_store.add_documents(documents=all_splits)
        end_time = time.time()
        print(f"Task <{task_id}> Done! Added {len(all_splits)} documents to the vector store this time, time cost: {end_time - start_time:.2f}s")
        return {"task_id": task_id, "error": None, "document_ids": document_ids}
    except Exception as e:
        return {"task_id": task_id, "error": e, "document_ids": None}


# 加载文档、设置并返回sparse_retriever和semantic_retriever
async def load_docs_and_get_retriever(text_splitter: RecursiveCharacterTextSplitter, vector_store: Chroma,
                                sparse_k: int = 30, semantic_k: int = 30) -> dict[
    str, BaseRetriever | list[str]]:
    """
    sparse_k: Amount of documents to return by sparse retriever, which retrieves using keywords (Default: 30, to improve the precision rate)
    semantic_k: Amount of documents to return by semantic retriever, which is provided by vector store and retrieves using similarity (Default: 30, to improve the precision rate)
    """
    print(f"Before adding documents, The number of documents in vector store is {vector_store._collection.count()}")
    all_document_ids = []
    task_list = []
    task_id = 1
    task_id_to_file_path = {}
    while True:
        file_path = input("Enter the file path, only supported .pdf (if ok, input 'done')：")
        if file_path == "done":
            break
        elif os.path.exists(file_path):
            # run task in another thread, map the task id to file path, and add task to task list
            task_id_to_file_path[task_id] = file_path
            task = asyncio.to_thread(load_doc_to_vector_store, text_splitter, vector_store, file_path, task_id)
            task_id += 1
            task_list.append(task)
        else:
            print("File not exist!")
    # wait for all tasks to complete, get the results and handle the exceptions
    results = await asyncio.gather(*task_list)
    for result in results:
        if result["error"]:
            print(f"Task <{result['task_id']}> failed \n file path: {task_id_to_file_path[result['task_id']]} \n error message : {result['error']}")
        else:
            all_document_ids.extend(result["document_ids"])
    docs_info = vector_store.get()
    bm25_retriever = BM25Retriever.from_texts(
        texts=docs_info["documents"], metadatas=docs_info["metadatas"],
        ids=docs_info["ids"], k=sparse_k, preprocess_func=bilingual_preprocess_func)
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": semantic_k})
    print(f"Added {len(all_document_ids)} documents to the vector store in total.\n"
          f"The number of documents in vector store is {len(docs_info["ids"])}.\n")
    return {"sparse_retriever": bm25_retriever, "semantic_retriever": semantic_retriever, "all_document_ids": all_document_ids}


# 创建hybrid_retriever，使用RRF进行融合
def init_hybrid_retriever(sparse_retriever: BaseRetriever, semantic_retriever: BaseRetriever, weights: list[float] = None) -> BaseRetriever:
    """weights: A list of weights corresponding to the retrievers. Defaults to equal weighting for all retrievers."""
    if weights is None:
        weights = [0.5, 0.5]
    print("Initializing hybrid retriever...")
    return EnsembleRetriever(retrievers=[sparse_retriever, semantic_retriever], weights=weights)


# 初始化CompressionRetriever，内置rerank
def init_compression_retriever(base_retriever: BaseRetriever, top_n: int = 7) -> BaseRetriever:
    """top_n: Number of documents to return by CompressionRetriever. Default 7."""
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



async def init_rag_application() -> dict[str, Any]:
    init_result = {}

    llm_init_task = asyncio.to_thread(init_llm)

    vector_store_init_task = asyncio.to_thread(init_vector_store)
    text_splitter_init_task = asyncio.create_task(init_text_splitter())
    nltk_resource_download_task = asyncio.to_thread(nltk_resource_download)
    results = await asyncio.gather(vector_store_init_task, text_splitter_init_task, nltk_resource_download_task)
    init_result["vector_store"] = results[0]
    init_result["text_splitter"] = results[1]

    retrievers_and_docs_ids = await asyncio.create_task(load_docs_and_get_retriever(text_splitter=init_result["text_splitter"], vector_store=init_result["vector_store"]))
    init_result["docs_ids"] = retrievers_and_docs_ids["all_document_ids"]
    hybrid_retriever = init_hybrid_retriever(sparse_retriever=retrievers_and_docs_ids["sparse_retriever"],
                                             semantic_retriever=retrievers_and_docs_ids["semantic_retriever"])
    compression_retriever = init_compression_retriever(base_retriever=hybrid_retriever)

    init_result["hybrid_retriever"] = hybrid_retriever
    init_result["compression_retriever"] = compression_retriever
    init_result["llm"] = await llm_init_task
    return init_result


if __name__ == "__main__":
    init_result = asyncio.run(init_rag_application())
    llm = init_result["llm"]
    vector_store = init_result["vector_store"]
    embeddings = vector_store.embeddings
    text_splitter = init_result["text_splitter"]
    docs_ids = init_result["docs_ids"]
    hybrid_retriever = init_result["hybrid_retriever"]
    compression_retriever = init_result["compression_retriever"]

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