import datetime
import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.embeddings import Embeddings
from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_core.tools import tool, StructuredTool
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from IPython.display import Image, display
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field, SecretStr
from flashrank import Ranker
from operator import add
from langgraph.graph import MessagesState
from typing import Annotated, Literal, Any
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import torch
import dotenv
import os
import sys
import inspect
import jieba
import asyncio
import time
from chromadb import ClientAPI


# 设置环境变量
def load_env():
    print("LOAD ENV: Loading environment variables...")
    dotenv.load_dotenv(override=True)
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
    if os.environ["LANGSMITH_TRACING"] == "true":
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["MODEL_PROVIDER"] = os.getenv("MODEL_PROVIDER")
    os.environ["MODEL_NAME"] = os.getenv("MODEL_NAME")
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["EMBEDDING_MODEL"] = os.getenv("EMBEDDING_MODEL")
    os.environ["OPENAI_EMBEDDING"] = os.getenv("OPENAI_EMBEDDING")
    if os.environ["OPENAI_EMBEDDING"] == "true":
        os.environ["OPENAI_EMBEDDING_API_BASE"] = os.getenv("OPENAI_EMBEDDING_BASE_URL")
        os.environ["OPENAI_EMBEDDING_API_KEY"] = os.getenv("OPENAI_EMBEDDING_API_KEY")

# 创建LLM
async def init_llm():
    print("INIT LLM: Initializing LLM...")
    llm = ChatOpenAI(model=os.environ["MODEL_NAME"], temperature=0.2)
    if os.environ["MODEL_PROVIDER"] != "openai":
        llm.openai_api_base = os.environ["OPENAI_API_BASE"]
    return llm.configurable_fields(temperature=ConfigurableField(
        id="temperature",
        name="Runtime Temperature",
        description="The runtime temperature provided by user"
    ))


# 初始化嵌入模型
def init_embedding_model() -> Embeddings:
    print("INIT EMBEDDING MODEL: Initializing embedding model...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("INIT EMBEDDING MODEL: CUDA is available")
    if os.environ["OPENAI_EMBEDDING"] == "true":
        print("INIT EMBEDDING MODEL: Using OpenAI embedding model...")
        return OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"],
                                base_url=os.environ["OPENAI_EMBEDDING_API_BASE"] if os.environ["OPENAI_EMBEDDING_API_BASE"] else None,
                                api_key=SecretStr(os.environ["OPENAI_EMBEDDING_API_KEY"]))
    print("INIT EMBEDDING MODEL: Using default HuggingFace embedding model...")
    return HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL"], model_kwargs={"device": device})


# 初始化向量数据库
async def init_vector_store(embeddings: Embeddings, client: ClientAPI, collection: Collection) -> Chroma:
    print("INIT VECTOR STORE: Initializing vector store...")
    return Chroma(
        client=client,
        collection_name=collection.name,
        embedding_function=embeddings
    )


def vector_store_use_or_create_collection(client: ClientAPI) -> Collection:
    collections = client.list_collections()
    exist_collections_name = []
    collection_name = "rag_default"
    if collections:
        print("INIT VECTOR STORE: History collections\n-------------------------------------------------")
        for c in collections:
            exist_collections_name.append(c.name)
            print(f"|   Collection: {c.name}, metadata:{c.metadata}   |")
        print("-------------------------------------------------")
    while True:
        collection_name = input(f"INIT VECTOR STORE: Please input a collection name to {'use or create a new one:' if exist_collections_name else 'create:'}")
        if collection_name in exist_collections_name:
            print(f"INIT VECTOR STORE: Using history collection {collection_name}")
        else:
            print(f"INIT VECTOR STORE: Creating collection {collection_name}")
        collection = client.get_or_create_collection(name=collection_name,
                                        metadata={
                                            "create_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M %A"),
                                            "embedding_model": os.environ["EMBEDDING_MODEL"],
                                            "hnsw:space": "cosine"
                                        })
        if collection.metadata["embedding_model"] != os.environ["EMBEDDING_MODEL"]:
            print(f"INIT VECTOR STORE: WARNING! Current embedding model {os.environ["EMBEDDING_MODEL"]} is not compatible with the collection, please delete the collection or create a new one.")
        else:
            return collection


# 初始化文本分词器
async def init_text_splitter(chunk_size: int = 512, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    print("INIT TEXT SPLITTER: Initializing text splitter...")
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)


# bm25预处理函数
def nltk_resource_download():
    """Download NLTK resources"""
    print("DOWNLOAD NLTK: Downloading NLTK resources...")
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
    print(f"LOADING DOCUMENTS TASK: Loading task <{task_id}>, document: {file_path}...")
    try:
        start_time = time.time()
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        # Essentially, aadd_documents calls the run_in_executor method of asyncio, which is equivalent to .to_thread(), executing in a separate thread.
        document_ids = vector_store.add_documents(documents=all_splits)
        end_time = time.time()
        print(f"LOADING DOCUMENTS TASK: Task <{task_id}> Done! Added {len(all_splits)} documents to the vector store this time, cost: {end_time - start_time:.2f}s")
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
    print(f"LOADING DOCUMENTS: Before adding documents, The number of documents in vector store is {vector_store._collection.count()}")
    # download NLTK resources for bilingual preprocessing
    nltk_resource_download_task = asyncio.to_thread(nltk_resource_download)
    all_document_ids = []
    load_task_list = []
    load_task_id = 1
    load_task_id_to_file_path = {}
    while True:
        file_path = input("LOADING DOCUMENTS: Enter the file path, only supported .pdf (if ok, input 'done')：")
        if file_path == "done":
            break
        elif os.path.exists(file_path):
            # run task in another process, map the task id to file path, and add task to task list
            load_task_id_to_file_path[load_task_id] = file_path
            task = asyncio.to_thread(load_doc_to_vector_store, text_splitter, vector_store, file_path, load_task_id)
            load_task_id += 1
            load_task_list.append(task)
        else:
            print("LOADING DOCUMENTS: File not exist!")
    # wait for all tasks to complete, get the results and handle the exceptions
    results = await asyncio.gather(*load_task_list, nltk_resource_download_task)
    for result in results:
        if result is None:
            continue
        elif result["error"]:
            print(f"LOADING DOCUMENTS: Task <{result['task_id']}> failed \n file path: {load_task_id_to_file_path[result['task_id']]} \n error message : {result['error']}")
        else:
            all_document_ids.extend(result["document_ids"])
    docs_info = vector_store.get()
    bm25_retriever = BM25Retriever.from_texts(
        texts=docs_info["documents"], metadatas=docs_info["metadatas"],
        ids=docs_info["ids"], k=sparse_k, preprocess_func=bilingual_preprocess_func)
    semantic_retriever = vector_store.as_retriever(search_kwargs={"k": semantic_k})
    print(f"LOADING DOCUMENTS: Added {len(all_document_ids)} documents to the vector store in total.\n"
          f"LOADING DOCUMENTS: The number of documents of current collection is NOW {len(docs_info["ids"])}.")
    return {"sparse_retriever": bm25_retriever, "semantic_retriever": semantic_retriever, "all_document_ids": all_document_ids}


# 创建hybrid_retriever，使用RRF进行融合
def init_hybrid_retriever(sparse_retriever: BaseRetriever, semantic_retriever: BaseRetriever, weights: list[float] = None) -> BaseRetriever:
    """weights: A list of weights corresponding to the retrievers. Defaults to equal weighting for all retrievers."""
    if weights is None:
        weights = [0.5, 0.5]
    print("INIT HYBRID RETRIEVER: Initializing hybrid retriever...")
    return EnsembleRetriever(retrievers=[sparse_retriever, semantic_retriever], weights=weights)


# 初始化CompressionRetriever，内置rerank
def init_compression_retriever(base_retriever: BaseRetriever, top_n: int = 7) -> BaseRetriever:
    """top_n: Number of documents to return by CompressionRetriever. Default 7."""
    print("INIT COMPRESSION RETRIEVER: Initializing compression retriever...")
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
    compressor = FlashrankRerank(client=ranker, top_n=top_n)
    return ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)

# 扫描当前模块中的所有StructuredTool对象，并将它们添加到tools列表中
tools = []
tools_without_retrieve = []
tools_by_name = {}
async def init_tool_infos():
    print("INIT TOOL INFO: Scanning current module for StructuredTool objects...")
    current_module = sys.modules['__main__']
    predicate = lambda member: isinstance(member, StructuredTool)
    tool_infos = inspect.getmembers(current_module, predicate)
    for name, tool in tool_infos:
        tools.append(tool)
        tools_by_name[tool.name] = tool
        if tool.name != retrieve.name:
            tools_without_retrieve.append(tool)
    print(f"INIT TOOL INFO: Find {len(tools)} tools")

# =========================================================================

# 图状态
class OverallState(MessagesState):
    command: Literal["retrieve", "direct"]
    user_question: str
    retrieved_docs: Annotated[list[str], add]

# class OutputState(MessagesState):
#     retrieved_docs: Annotated[list[str], add]

# 输入Schema
class RetrieveInputSchema(BaseModel):
    query: str = Field(..., description="The query to retrieve.")

# 创建工具
@tool(description="Retrieve relevant information from document store to help answer a question", args_schema=RetrieveInputSchema)
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
@tool(description="Add two number")
def add(a: int, b: int) -> str:
    """Add two number.
    Args:
        a: first int
        b: second int
    """
    return str(a + b)


# 创建工具节点
def tool_node(state: OverallState):
    """Performs the tool"""
    result = []
    all_retrieved_docs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name.get(tool_call["name"])
        if tool is None:
            print(f"Tool named '{tool_call['name']}' not found")
            continue
        content = tool.invoke(tool_call["args"])
        if tool_call["name"] == retrieve.name:
            all_retrieved_docs.append(content)
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
RETRIEVE_QUERY_PROMPT_STR = """
You are a helpful RAG assistant.You have access to a retriever tool.Use the tool to better answer user's question.If you decide to use retrieve tool, rewrite user's original question and generate some augmented questions that are better for retrieval.

Here is the original user question:
{user_question}
"""
QUERY_PROMPT_STR="""
You are a helpful assistant.You have access to some tools.Use these tools to better answer user's question.

Here is the original user question:
{user_question}
"""
retrieve_query_promptTemplate = PromptTemplate.from_template(RETRIEVE_QUERY_PROMPT_STR, partial_variables={"user_question": "你好"})
query_promptTemplate = PromptTemplate.from_template(QUERY_PROMPT_STR, partial_variables={"user_question": "你好"})
def generate_query_or_respond(state: OverallState):
    """Call the model to generate a response based on the current state. Based on state's command,
    decide to retrieve using the retriever tool, or simply respond to the user.
    """
    # In this node, we want llm to answer more randomly using a higher temperature, even retrieve task.
    response = ""
    if state["command"] == "retrieve":
        # through testing, with_config method should be used before bind_tools.In this case, configurable fields are passed to llm.
        prompt = retrieve_query_promptTemplate.invoke(input={"user_question": state["user_question"]})
        response = llm.with_config(configurable={"temperature": 0.6}).bind_tools([retrieve]).invoke(prompt)
    elif state["command"] == "direct":
        prompt = query_promptTemplate.invoke(input={"user_question": state["user_question"]})
        response = llm.with_config(configurable={"temperature": 0.6}).bind_tools(tools_without_retrieve).invoke(prompt)
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
    load_env()

    llm_init_task = asyncio.create_task(init_llm())
    text_splitter_init_task = asyncio.create_task(init_text_splitter())
    tool_init_task = asyncio.create_task(init_tool_infos())

    init_result["embeddings"] = await asyncio.to_thread(init_embedding_model)

    init_result["client"] = chromadb.PersistentClient("./chroma_langchain_db")
    init_result["collection"] = vector_store_use_or_create_collection(init_result["client"])
    vector_store_init_task = asyncio.create_task(init_vector_store(init_result["embeddings"], init_result["client"], init_result["collection"]))
    results = await asyncio.gather(text_splitter_init_task, vector_store_init_task)
    init_result["text_splitter"] = results[0]
    init_result["vector_store"] = results[1]

    retrievers_and_docs_ids = await asyncio.create_task(load_docs_and_get_retriever(init_result["text_splitter"], init_result["vector_store"]))
    init_result["docs_ids"] = retrievers_and_docs_ids["all_document_ids"]
    hybrid_retriever = init_hybrid_retriever(sparse_retriever=retrievers_and_docs_ids["sparse_retriever"],
                                             semantic_retriever=retrievers_and_docs_ids["semantic_retriever"])
    compression_retriever = init_compression_retriever(base_retriever=hybrid_retriever)
    init_result["hybrid_retriever"] = hybrid_retriever
    init_result["compression_retriever"] = compression_retriever

    results = await asyncio.gather(llm_init_task, tool_init_task)
    init_result["llm"] = results[0]
    return init_result


if __name__ == "__main__":
    init_result = asyncio.run(init_rag_application())
    llm = init_result["llm"]
    embeddings = init_result["embeddings"]
    vector_store = init_result["vector_store"]
    collection = init_result["collection"]
    client = init_result["client"]
    text_splitter = init_result["text_splitter"]
    docs_ids = init_result["docs_ids"]
    hybrid_retriever = init_result["hybrid_retriever"]
    compression_retriever = init_result["compression_retriever"]

    # 创建Graph
    workflow = StateGraph(OverallState)

    workflow.add_node(generate_query_or_respond)
    workflow.add_node(tool_node)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges("generate_query_or_respond", should_use_tool)
    workflow.add_edge("tool_node", "generate_answer")
    workflow.add_edge("generate_answer", END)

    agent = workflow.compile()

    # 运行
    command = "retrieve"
    while True:
        print(f"====================================================================")
        print(f"Using collection '{collection.name}'. Current mode '{command}'. Enter '/{['retrieve', 'direct'][command == 'retrieve']}' to switch to {['retrieve', 'direct'][command == 'retrieve']} mode.")
        user_input = input(f"Ask your questions (input 'exit' to stop)：")
        if user_input == "exit":
            if docs_ids:
                delete = input("Do you want to delete the documents added in this session? If not, they will persist in the current folder. (y/n) ")
                while delete not in ["y", "n"]:
                    delete = input("Invalid input. Please enter 'y' or 'n': ")
                if delete == "y":
                    vector_store.delete(ids=docs_ids)
                    print("Documents deleted.")
                else:
                    print("Documents will persist in the current folder.")
            print("Bye!")
            break
        elif user_input.startswith("/"):
            if user_input not in ["/retrieve", "/direct"]:
                print("Invalid command. Please enter '/retrieve' or '/direct'.")
                continue
            command = user_input[1:]
            print(f"Switched to {command} mode.\n")
            continue
        state = {"command": command, "user_question": user_input, "retrieved_docs": []}
        for chunk in agent.stream(
                input=state,
        ):
            for node, update in chunk.items():
                print("Update from node", node)
                update["messages"][-1].pretty_print()
                print("\n\n")
        if command == "retrieve" and state["retrieved_docs"]:
            print(f"Documents\n: {state["retrieved_docs"]}")