from langchain_core.documents import Document
from pydantic import Field


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