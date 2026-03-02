# import sys
# import inspect
# from typing import Literal
#
# from langchain_core.messages import ToolMessage
# from langchain_core.prompts import PromptTemplate
# from langchain_core.tools import tool, StructuredTool
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.state import CompiledStateGraph
# from IPython.display import Image, display
#
# from app.models.state import OverallState, RetrieveInputSchema
#
#
# # 扫描当前模块中的所有StructuredTool对象，并将它们添加到tools列表中
# tools = []
# tools_without_retrieve = []
# tools_by_name = {}
#
#
# async def init_tool_infos():
#     print("INIT TOOL INFO: Scanning current module for StructuredTool objects...")
#     current_module = sys.modules['__main__']
#     predicate = lambda member: isinstance(member, StructuredTool)
#     tool_infos = inspect.getmembers(current_module, predicate)
#     for name, tool in tool_infos:
#         tools.append(tool)
#         tools_by_name[tool.name] = tool
#         if tool.name != retrieve.name:
#             tools_without_retrieve.append(tool)
#     print(f"INIT TOOL INFO: Find {len(tools)} tools")
#
#
# # 创建工具
# @tool(description="Retrieve relevant information from document store to help answer a question",
#       args_schema=RetrieveInputSchema)
# def retrieve(query: str):
#     """Retrieve information to help answer a query.
#     Args:
#         query: string.The query to retrieve.
#     """
#     retrieved_docs = compression_retriever.invoke(input=query)
#     serialized = ""
#     origin_docs_map = {}
#     for doc in retrieved_docs:
#         doc_str = f"\nSource: {doc.metadata}\nContent: {doc.page_content}\n"
#         serialized += doc_str
#         origin_docs_map[doc.id] = doc_str
#     return {"serialized": serialized, "origin_docs_map": origin_docs_map}
#
#
# @tool(description="Add two number")
# def add(a: int, b: int) -> str:
#     """Add two number.
#     Args:
#         a: first int
#         b: second int
#     """
#     return str(a + b)
#
#
# # 创建工具节点
# def tool_node(state: OverallState):
#     """Performs the tool"""
#     result_messages = []
#     all_deduplicate_docs = {}
#     for tool_call in state["messages"][-1].tool_calls:
#         tool = tools_by_name.get(tool_call["name"])
#         if tool is None:
#             print(f"Tool named '{tool_call['name']}' not found")
#             continue
#         content = tool.invoke(tool_call["args"])
#         if tool_call["name"] == retrieve.name:
#             # the result of retrieve tool is a dict contains two elements: serialized and origin_docs_map
#             origin_docs_map = content["origin_docs_map"]  # list[Document]
#             content = content["serialized"]
#             for doc_id, doc_str in origin_docs_map.items():
#                 all_deduplicate_docs[doc_id] = doc_str
#         result_messages.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
#     return {"messages": result_messages, "retrieved_docs": set(list(all_deduplicate_docs.values()))}
#
#
# def should_use_tool(state: OverallState) -> Literal["tool_node", "__end__"]:
#     """Decides whether to use the tools. And send the state to relevant nodes."""
#     last_message = state["messages"][-1]
#     if last_message.tool_calls:
#         return "tool_node"
#     else:
#         return "__end__"
#
#
# # 创建生成查询的Node
# RETRIEVE_QUERY_PROMPT_STR = """
# You are a helpful RAG assistant.You have access to a retriever tool.Use the tool to better answer user's question.If you decide to use retrieve tool, rewrite user's original question and generate some augmented questions that are better for retrieval.
#
# Here is the original user question:
# {user_question}
# """
# QUERY_PROMPT_STR = """
# You are a helpful assistant.You have access to some tools.Use these tools to better answer user's question.
#
# Here is the original user question:
# {user_question}
# """
# retrieve_query_promptTemplate = PromptTemplate.from_template(RETRIEVE_QUERY_PROMPT_STR,
#                                                              partial_variables={"user_question": "你好"})
# query_promptTemplate = PromptTemplate.from_template(QUERY_PROMPT_STR, partial_variables={"user_question": "你好"})
#
#
# def generate_query_or_respond(state: OverallState):
#     """Call the model to generate a response based on the current state. Based on state's command,
#     decide to retrieve using the retriever tool, or simply respond to the user.
#     """
#     # In this node, we want llm to answer more randomly using a higher temperature, even retrieve task.
#     response = ""
#     if state["command"] == "retrieve":
#         # through testing, with_config method should be used before bind_tools.In this case, configurable fields are passed to llm.
#         prompt = retrieve_query_promptTemplate.invoke(input={"user_question": state["user_question"]})
#         response = llm.with_config(configurable={"temperature": 0.6}).bind_tools([retrieve]).invoke(prompt)
#     elif state["command"] == "direct":
#         prompt = query_promptTemplate.invoke(input={"user_question": state["user_question"]})
#         response = llm.with_config(configurable={"temperature": 0.6}).bind_tools(tools_without_retrieve).invoke(prompt)
#     return {"messages": [response]}
#
#
# # 创建生成答案的Node
# ANSWER_PROMPT_STR = """
# You are an assistant for question-answering tasks.Use the following pieces of retrieved context to answer the question.
# If you don't know the answer, just say that you don't know.
# Question: {question}
# Context: {context}
# """
# answer_prompt = PromptTemplate.from_template(ANSWER_PROMPT_STR)
#
#
# def generate_answer(state: OverallState):
#     """Call the model to generate a final answer based on the current state."""
#     chain = answer_prompt | llm
#     context = "".join(state["retrieved_docs"])
#     response = chain.invoke(input={"question": state["user_question"], "context": context})
#     return {"messages": [response]}
#
#
# # 绘制图：
# def draw_graph(graph: CompiledStateGraph):
#     display(Image(graph.get_graph().draw_mermaid_png()))
