from langchain_core.documents import Document


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
    - error: 加载过程中发生的异常，若无异常则为 None
    - splitters: 参与分块的分块器名称列表，若未分块则为 None
    - document_loader: 使用的 DocumentLoader 名称，直接读取为DIRECT READ，若未加载则为 None
    - document_ids: 成功加载到向量数据库的 Document ID 列表，若未加载则为 None
    """
    def __init__(self, file_id: str, error: Exception | None,
                 splitters: list[str] | None, document_loader: str | None,
                 document_ids: list[str] | None):
        self.file_id = file_id
        self.error = error
        self.splitters = splitters
        self.document_loader = document_loader
        self.document_ids = document_ids

class SplitResult:
    """
    分块结果类
     - splitters: 使用的分块器名称列表
     - chunks: 分块后的 Document 列表
    """
    def __init__(self, splitters: list[str], chunks: list[Document]):
        self.splitters = splitters
        self.chunks = chunks