from datetime import datetime

from langchain_core.documents import Document
from pydantic import Field, BaseModel, ConfigDict


class Result:
    """
    统一返回结果类
    """

    def __init__(self, code, message, data):
        self.code = code
        self.message = message
        self.data = data


class LoadDocToVectorStoreResult:
    """
    文档加载到向量数据库的结果类
        - file_id: 文件 ID
        - error: 错误信息，如果没有错误则为 None
        - document_loader: 使用的文档加载器名称，如果没有则为 None
        - parent_splitter: 使用的父分块器名称，如果没有则为 None
        - parent_doc_ids: 父文档 ID 列表，如果没有则为 None
        - children_count: 生成的子文档数量，如果没有则为 None
        - cost: 耗时(s)，如果没有则为 None
    """

    def __init__(self,
        file_id: str,
        error: Exception | None,
        document_loader: str | None = None,
        parent_splitter: str | None = None,
        parent_doc_ids: list[str] | None = None,
        children_count: int | None = None,
        cost: float | None = None,
    ):
        self.file_id = file_id
        self.error = error
        self.document_loader = document_loader
        self.parent_splitter = parent_splitter
        self.parent_doc_ids = parent_doc_ids
        self.children_count = children_count
        self.cost = cost

    @classmethod
    def error(cls, file_id: str, error: Exception):
        return cls(file_id=file_id,error=error)


class SplitResult:
    """
    分块结果类
     - splitters: 使用的分块器名称列表
     - chunks: 分块后的 Document 列表
    """

    def __init__(self, splitters: list[str], chunks: list[Document]):
        self.splitters = splitters
        self.chunks = chunks


class EnhancedPDRetrieverAddDocumentsResult:
    """
    EnhancedParentDocumentRetriever工厂添加文档的结果类
     - parent_splitter_name: 使用的父分块器名称，如果没有则为 None
     - parent_doc_ids: 父文档 ID 有序列表
     - children_count: 添加的子文档总数
    """

    def __init__(self, parent_doc_ids: list[str], children_count: int, parent_splitter_name: str | None = None):
        self.parent_doc_ids = parent_doc_ids
        self.children_count = children_count
        self.parent_splitter_name = parent_splitter_name

class SetReferencesDto:
    """
    设置参考文档的dto类
    """
    doc_ids: list[str] = Field(..., alias="docIds", description="文档ID列表")


class EmbeddedDocumentVO(BaseModel):
    """
    嵌入文档的视图对象（VO），用于 API 层与前端交互，不包含分块相关信息
    """

    id: str = Field(..., description="文档唯一标识(UUID)")
    path: str = Field(..., description="文件路径/url,唯一")
    is_url: bool = Field(..., description="是否为URL")
    file_directory: str | None = Field(None, description="文件目录,仅本地文件使用")
    file_name: str | None = Field(None, description="文件名")
    file_extension: str = Field(..., description="文件后缀名(如 .pdf)")
    mime_type: str | None = Field(None, description="文件mime类型")
    last_modified: datetime | None = Field(None, description="文件最后修改时间,仅本地文件使用")

    model_config = ConfigDict(
        from_attributes=True # 支持从 ORM 模型创建 Pydantic 模型
    )

# ======================== 对话相关的返回模型 ========================

class ConversationVO(BaseModel):
    """对话列表项的视图对象"""
    id: str = Field(..., description="对话 ID")
    title: str = Field(..., description="对话标题")
    created_at: datetime | None = Field(None, description="创建时间")
    updated_at: datetime | None = Field(None, description="更新时间")

    model_config = ConfigDict(from_attributes=True)

class ChatMessageVO(BaseModel):
    """聊天记录项的视图对象"""
    id: int = Field(..., description="消息 ID")
    role: str = Field(..., description="消息角色 (user/ai)")
    content: str = Field(..., description="消息内容")
    parent_doc_ids: list[str] = Field(default_factory=list, description="父文档 ID 列表")
    created_at: datetime | None = Field(None, description="创建时间")

    model_config = ConfigDict(from_attributes=True)

class ChatHistoryResponseVO(BaseModel):
    """获取聊天历史接口的外层视图对象"""
    conversation_id: str = Field(..., description="对话 ID")
    messages: list[ChatMessageVO] = Field(default_factory=list, description="消息列表")
