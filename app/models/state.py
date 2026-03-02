# from langgraph.graph import MessagesState
# from typing import Annotated, Literal
# from pydantic import BaseModel, Field
#
#
# def union(a: set, b: set):
#     return a.union(b)
#
# # 图状态
# class OverallState(MessagesState):
#     command: Literal["retrieve", "direct"]
#     user_question: str
#     retrieved_docs: Annotated[set[str], union]
#
#
# # class OutputState(MessagesState):
# #     retrieved_docs: Annotated[list[str], add]
#
# # 输入Schema
# class RetrieveInputSchema(BaseModel):
#     query: str = Field(..., description="The query to retrieve.")
