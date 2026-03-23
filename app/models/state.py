"""
LangGraph 状态定义。
"""

from typing import Literal
from langchain_core.documents import Document
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class GraphState(MessagesState):
    """
    Graph 全局状态，继承 MessagesState 以支持多轮对话。
    MessagesState 自动管理 messages: Annotated[list[AnyMessage], add_messages]
    """
    original_question: str                       # 本次对话用户的原始问题（不随 rewrite 改变）
    documents: list[Document]                   # 检索到的文档列表（覆盖，每轮对话需要重置）
    rewrite_count: int                          # 问题重写计数（防 rewrite 无限循环，每轮对话需要重置）
    generate_count: int                         # 回答生成计数（防 hallucination 无限循环，每轮对话需要重置）
    summary: str                                # 历史对话摘要（summarize messages 使用）



# ======================== Structured Output Schemas ========================

class SetOriginalQuestion(BaseModel):
    """设置原始问题"""
    binary_score: Literal["keep", "update"] = Field(
        description="是否需要更新用户的原始问题：'keep'(保留) 或 'update'(更新)"
    )

class GradeDocuments(BaseModel):
    """检索文档相关性评分"""
    binary_score: Literal["yes", "no"] = Field(
        description="检索到的文档是否与用户问题相关: 'yes' 或 'no'"
    )


class HallucinationCheck(BaseModel):
    """幻觉检测评分"""
    binary_score: Literal["yes", "no"] = Field(
        description="LLM 生成的回答是否基于检索到的文档（无幻觉）: 'yes'(基于文档, 无幻觉) 或 'no'(存在幻觉)"
    )


class UsefulnessCheck(BaseModel):
    """回答有用性评分"""
    binary_score: Literal["yes", "no"] = Field(
        description="LLM 的回答是否解决了用户的问题: 'yes'(已解决) 或 'no'(未解决)"
    )
