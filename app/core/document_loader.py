import os
import time
from abc import abstractmethod, ABC

from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, WebBaseLoader
from langchain_community.document_loaders.parsers import RapidOCRBlobParser
from langchain_core.documents import Document

from app.config.global_config import global_config
from app.core.chunking import EXT_TO_LANGUAGE
from app.core.retriever import EnhancedParentDocumentRetrieverFactory
from app.models.common import LoadDocToVectorStoreResult

# 支持的文件后缀
TEXT_EXTENSIONS = set(EXT_TO_LANGUAGE.keys()) | {".md", ".txt"}
PDF_EXTENSION = {".pdf"}
HTML_EXTENSIONS = {".html", ".htm"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSION | HTML_EXTENSIONS


def _get_file_extension(file_path: str) -> str:
    """获取文件后缀（含点号），如 '.pdf', '.md'"""
    return os.path.splitext(file_path)[1].lower()


def _validate_local_file(file_path: str) -> str:
    """
    校验文件是否存在且后缀受支持，返回文件后缀。
    :raises FileNotFoundError: 文件不存在
    :raises ValueError: 文件类型不受支持
    """
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{file_path}' does not exist or is not a file.")
    file_ext = _get_file_extension(file_path)
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{file_ext}' for file '{file_path}'.")
    return file_ext


# ======================== DocumentLoaderChain ========================

class DocumentLoader(ABC):
    """
    文档加载器抽象父类（责任链节点）。
    子类须实现 supports() 和 do_load() 方法。
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self, supported_file_types: set[str], order: int = 0):
        self.supported_file_types = supported_file_types
        self.order = order

    @abstractmethod
    def supports(self, file_type: str, is_url: bool = False) -> bool:
        """
        判断当前加载器是否支持该文件类型。
        :param file_type: 文件后缀（如 '.pdf', '.md'）
        :param is_url: 是否为 URL
        :return: 是否支持
        """

    @abstractmethod
    def do_load(self, path: str, file_type: str, is_url: bool = False) -> list[Document]:
        """
        执行文档加载，返回 Document 列表。
        :param path: 文件路径或 URL
        :param file_type: 文件后缀（如 '.pdf', '.md'）
        :param is_url: 是否为 URL
        :return: 加载后的 Document 列表
        """

    def _add_metadata(self, path: str, file_type: str, is_url: bool, docs: list[Document]) -> list[Document]:
        """
        根据文件信息为 Document 列表添加统一的元数据字段。
        默认实现添加 {
            "path",
            "file_directory"(非URL时添加),
            "file_name"(非URL时添加),
            "file_extension",
            "last_modified"(非URL时添加),
            "is_url",
            "document_loader",
        }
        :param path:
        :param file_type:
        :param is_url:
        :param docs:
        :return: 元数据更新后的 Document 列表
        """
        metadatas = {
            "path": path,
            "file_directory": os.path.dirname(os.path.abspath(path)) if not is_url else None,
            "file_name": os.path.basename(path) if not is_url else None,
            "file_extension": file_type,
            "last_modified": time.strftime("%Y-%m-%dT%H:%M:%S",
                                           time.localtime(os.path.getmtime(path))) if not is_url else None,
            "is_url": is_url,
            "document_loader": self.name,
        }
        for doc in docs:
            doc.metadata.update(metadatas)
        return docs


class PDFLoader(DocumentLoader):
    """
    PDF 文件加载器，使用 PyPDFLoader 加载 PDF.
    """

    def __init__(self, order: int = 10):
        super().__init__(PDF_EXTENSION, order)

    def supports(self, file_type: str, is_url: bool = False) -> bool:
        return not is_url and file_type in self.supported_file_types  # PDFLoader 不处理 URL

    def do_load(self, path: str, file_type: str, is_url: bool = False) -> list[Document]:
        try:
            loader = PyPDFLoader(
                file_path=path,
                mode="page",  # 按页加载，每页一个 Document
                images_inner_format="markdown-img",
                images_parser=RapidOCRBlobParser()  # 使用 rapidOCR 提取图片
            )
            docs = loader.load()
            return self._add_metadata(path, file_type, is_url, docs)
        except Exception as e:
            raise RuntimeError(f"PDFLoader failed to load '{path}': {e}")

    def _add_metadata(self, path: str, file_type: str, is_url: bool, docs: list[Document]) -> list[Document]:
        return super()._add_metadata(path, file_type, is_url, docs)


class HTMLLoader(DocumentLoader):
    """
    HTML 文件加载器
    - 使用 WebBaseLoader 加载URL
    - 使用 BSHTMLLoader 加载本地HTML文件
    """

    def __init__(self, order: int = 10):
        super().__init__(HTML_EXTENSIONS, order)

    def supports(self, file_type: str, is_url: bool = False) -> bool:
        # 可处理 HTML 文件和 URL，但仅限于 HTML 类型
        return file_type in self.supported_file_types

    def do_load(self, path: str, file_type: str, is_url: bool = False) -> list[Document]:
        try:
            if is_url:
                loader = WebBaseLoader(web_path=path)
            else:
                loader = BSHTMLLoader(file_path=path)
            docs = loader.load()
            return self._add_metadata(path, file_type, is_url, docs)
        except Exception as e:
            raise RuntimeError(f"HTMLLoader failed to load '{path}': {e}")

    def _add_metadata(self, path: str, file_type: str, is_url: bool, docs: list[Document]) -> list[Document]:
        return super()._add_metadata(path, file_type, is_url, docs)


class TextLoader(DocumentLoader):
    """
    文本文件加载器，处理 .txt, .md 等纯文本文件（包括代码文件），直接读取内容构造 Document
    """

    def __init__(self, order: int = 10):
        super().__init__(TEXT_EXTENSIONS, order)

    def supports(self, file_type: str, is_url: bool = False) -> bool:
        return not is_url and file_type in self.supported_file_types  # TextLoader 不处理 URL

    def do_load(self, path: str, file_type: str, is_url: bool = False) -> list[Document]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            doc = Document(
                page_content=content,
                metadata={"text_length": len(content)}
            )
            return self._add_metadata(path, file_type, is_url, [doc])
        except Exception as e:
            raise RuntimeError(f"TextLoader failed to load '{path}': {e}")

    def _add_metadata(self, path: str, file_type: str, is_url: bool, docs: list[Document]) -> list[Document]:
        return super()._add_metadata(path, file_type, is_url, docs)


class DocumentLoaderChain:
    """
    文档加载器责任链：注册多个 DocumentLoader 子类实例，
    do_load 时遍历注册列表，遇到第一个支持的加载器即执行其 do_load 方法。
    """
    _instance: "DocumentLoaderChain | None" = None

    def __init__(self):
        self._loaders: list[DocumentLoader] = []
        self._sorted = True

    @classmethod
    def init(cls) -> "DocumentLoaderChain":
        """
        初始化 DocumentLoaderChain 单例，注册所有内置的 DocumentLoader。
        """
        if cls._instance is not None:
            print("INIT LOADER CHAIN: Instance already exists, returning cached instance.")
            return cls._instance

        print("INIT LOADER CHAIN: Initializing...")
        chain = cls()

        # 注册内置的 DocumentLoader 子类
        chain.register(PDFLoader(order=10))
        chain.register(HTMLLoader(order=10))
        chain.register(TextLoader(order=10))

        cls._instance = chain

        print(f"INIT LOADER CHAIN: Initialized successfully, {len(chain._loaders)} loaders registered: "
              f"{', '.join(loader.name for loader in chain._loaders)}")
        return cls._instance

    def register(self, loader: DocumentLoader) -> "DocumentLoaderChain":
        """注册 DocumentLoader 子类实例，支持链式调用。order 越小越优先，相同 order 按注册顺序。"""
        if not isinstance(loader, DocumentLoader):
            raise TypeError(f"Expected DocumentLoader instance, got {type(loader).__name__}")
        self._loaders.append(loader)
        self._sorted = False
        return self

    def _ensure_sorted(self):
        """确保 loaders 按 order 升序 + 注册顺序排列（stable sort）"""
        if not self._sorted:
            self._loaders.sort(key=lambda l: l.order)
            self._sorted = True

    def do_load(self, path: str, file_type: str, is_url: bool = False) -> tuple[str, list[Document]]:
        """
        遍历注册的加载器，使用第一个支持该文件类型的加载器执行加载。
        会添加部分元数据，默认添加
        "path", "is_url", "document_loader", "file_extension",
        "file_directory"(非URL时添加), "file_name"(非URL时添加), "last_modified"(非URL时添加)。
        TextLoader 还会添加 "text_length"
        :param path: 文件路径或 URL
        :param file_type: 文件后缀
        :param is_url: 是否为 URL
        :return: (loader_name, documents)
        :raises ValueError: 没有匹配的加载器
        """
        self._ensure_sorted()
        for loader in self._loaders:
            if loader.supports(file_type, is_url):
                print(f"LOADER CHAIN: Selected '{loader.name}' (order={loader.order}) for file type '{file_type}'")
                docs = loader.do_load(path, file_type, is_url)
                print(f"LOADER CHAIN: '{loader.name}' loaded {len(docs)} documents for '{path}'")
                return loader.name, docs
        raise ValueError(f"No DocumentLoader supports file type: {file_type}")

    @property
    def loaders(self) -> list[DocumentLoader]:
        self._ensure_sorted()
        return list(self._loaders)

    @classmethod
    def get_instance(cls) -> "DocumentLoaderChain":
        """获取 DocumentLoaderChain 单例。未初始化时抛出异常。"""
        if cls._instance is None:
            raise RuntimeError("DocumentLoaderChain has not been initialized. Call DocumentLoaderChain.init() first.")
        return cls._instance


async def load_doc_to_vector_store(path: str, file_id: str, is_url: bool = False) -> \
        LoadDocToVectorStoreResult:
    """
    加载文档，分块与两层存储（子块向量 + 父文档持久化）。
        1. 使用DocumentLoaderChain加载文档，添加file_id元数据
        2. 使用 EnhancedParentDocumentRetrieverFactory 添加文档
    :param path: 文档路径/url
    :param file_id: 文档唯一标识（数据库id）
    :param is_url: 是否为url
    :return: LoadDocToVectorStoreResult
    """
    print(f"LOADING DOCUMENTS TASK: Loading document <{file_id}>: {path}...")
    try:
        start_time = time.time()

        # 1. 文件校验
        if file_id is None:
            raise ValueError("LOADING DOCUMENTS TASK: file_id cannot be None")
        if is_url:
            file_ext = ".html"
        else:
            # 本地文件校验：存在性 + 支持的后缀
            file_ext = _validate_local_file(path)

        # 2. 使用DocumentLoaderChain加载文档
        loader_chain = DocumentLoaderChain.get_instance()
        loader_name, docs = loader_chain.do_load(path, file_ext, is_url)
        if docs is None or len(docs) == 0:
            raise Exception(f"LOADING DOCUMENTS TASK: No documents loaded for <{file_id}> by loader '{loader_name}'")
        print(f"LOADING DOCUMENTS TASK: Loaded {len(docs)} documents for <{file_id}> using loader '{loader_name}'")

        # 3. 给加载后的全部文档添加 file_id 元数据
        for doc in docs:
            doc.metadata["file_id"] = file_id

        # 4. 判断是否启用父分块器，默认启用
        enable_parent_splitter = True
        if TextLoader.name == loader_name:
            # 纯文本文件，若长度小于阈值则不启用父分块器，整体作为一个父文档
            text_length = docs[0].metadata.get("text_length", 0)
            if text_length < global_config.get("text_file_length_threshold", 1000):
                enable_parent_splitter = False

        # 5. 使用 EnhancedParentDocumentRetrieverFactory 添加文档
        pd_add_docs_result = EnhancedParentDocumentRetrieverFactory.add_documents(file_type=file_ext, documents=docs,
                                                             use_parent=enable_parent_splitter, add_to_docstore=True)

        # 6. 获取结果
        parent_splitter_name = pd_add_docs_result.parent_splitter_name
        parent_doc_ids = pd_add_docs_result.parent_doc_ids
        children_count = pd_add_docs_result.children_count

        end_time = time.time()
        cost = end_time - start_time
        print(
            f"LOADING DOCUMENTS TASK: document <{file_id}> Done! "
            f"Loader: {loader_name}. Parent Splitter: {parent_splitter_name}, "
            f"Parent Doc count {len(parent_doc_ids)}. Children Doc count {children_count}. "
            f"Cost: {cost:.2f}s")
        return LoadDocToVectorStoreResult(file_id, None, loader_name, parent_splitter_name, parent_doc_ids,
                                          children_count, cost)
    except Exception as e:
        return LoadDocToVectorStoreResult.error(file_id, e)
