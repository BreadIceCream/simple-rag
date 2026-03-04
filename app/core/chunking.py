"""
chunking模块。配置和注册不同类型文件的 TextSplitter

核心组件：
- MarkdownTextSplitterAdapter / HTMLTextSplitterAdapter: TextSplitter 子类适配器
- SplitterEntry / SplitterRegistry: 按文件类型选择最优先的 parent TextSplitter
"""

import os
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, TextSplitter,
    MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter, Language,
)

# ======================== 常量 ========================

# 文件后缀 → Language 枚举的映射
EXT_TO_LANGUAGE: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".cs": Language.CSHARP,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".kt": Language.KOTLIN,
    ".php": Language.PHP,
    ".lua": Language.LUA,
    ".hs": Language.HASKELL,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".h": Language.CPP,
    ".latex": Language.LATEX,
}

# 通配符，匹配所有文件类型
MATCH_ALL = "*"

TEXT_SPLITTERS_METADATA_KEY = "text_splitters"


class TextSplitterMetadataMixin:
    """
    TextSplitter 元数据 Mixin：
    为分块后的 Document 追加 text_splitters 元数据（当前分块器类名）。所有自定义 TextSplitter 均应混入此类。
    """
    def add_metadata(self, original_metadata: dict) -> dict:
        """
        追加当前分块器类名到 text_splitters 元数据列表。
        :param original_metadata: 原始元数据字典
        :return: 更新后的元数据字典
        """
        text_splitters = original_metadata.get(TEXT_SPLITTERS_METADATA_KEY, [])
        text_splitters.append(self.__class__.__name__)
        original_metadata[TEXT_SPLITTERS_METADATA_KEY] = text_splitters
        return original_metadata


# ======================== TextSplitter 适配器 ========================
class MarkdownTextSplitterAdapter(TextSplitterMetadataMixin, TextSplitter):
    """
    Markdown 分割适配器：TextSplitter 子类，内部委托 MarkdownHeaderTextSplitter。

    MarkdownHeaderTextSplitter.split_text() 返回 list[Document]（含标题层级元数据），
    而 TextSplitter 的标准接口 split_text() 要求返回 list[str]。
    因此本适配器重写 split_text / split_documents / create_documents 三个核心方法，
    在保留元数据的同时满足 TextSplitter 接口约束。

    NOTE: 在 ParentDocumentRetriever 中，add_documents 会调用 parent_splitter.split_documents()，
    传入的每个 Document 的 page_content 即为完整的 Markdown 文件内容。
    """
    def __init__(self,
                 headers_to_split_on: list[tuple[str, str]] | None = None,
                 strip_headers: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self._md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on or [
                ("#", "Header 1"), ("##", "Header 2"),
            ],
            strip_headers=strip_headers,
        )

    def split_text(self, text: str) -> list[str]:
        """
        按 Markdown 标题层级切分，返回各分块的文本内容。
        无元数据保留。
        """
        docs = self._md_splitter.split_text(text)
        return [doc.page_content for doc in docs]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        按 Markdown 标题层级切分 Document 列表。
        保留原始 Document 的 metadata，并合并 MarkdownHeaderTextSplitter 产生的标题层级元数据。
        """
        result = []
        for doc in documents:
            sub_docs = self._md_splitter.split_text(doc.page_content)
            for sub_doc in sub_docs:
                # 合并父文档的 metadata + 切分产生的标题层级 metadata
                merged_metadata = {**doc.metadata, **sub_doc.metadata}
                merged_metadata = self.add_metadata(merged_metadata)
                result.append(Document(page_content=sub_doc.page_content, metadata=merged_metadata))
        return result

    def create_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        """将原始文本列表按 Markdown 标题层级切分为 Document 列表。"""
        result = []
        _metadatas = metadatas or [{}] * len(texts)
        for text, metadata in zip(texts, _metadatas):
            sub_docs = self._md_splitter.split_text(text)
            for sub_doc in sub_docs:
                merged_metadata = {**metadata, **sub_doc.metadata}
                merged_metadata = self.add_metadata(merged_metadata)
                result.append(Document(page_content=sub_doc.page_content, metadata=merged_metadata))
        return result




class HTMLTextSplitterAdapter(TextSplitterMetadataMixin, TextSplitter):
    """
    HTML 分割适配器：TextSplitter 子类，内部委托 HTMLHeaderTextSplitter。

    HTMLHeaderTextSplitter 有三种切分方式：split_text(html_str)、split_text_from_url(url)、
    split_text_from_file(file_path)。

    在 ParentDocumentRetriever 流程中，传入的 Document 的 page_content 可能是：
    1. HTML 文本内容
    2. 文件路径
    3. URL
    本适配器在 split_documents 中根据 page_content 的内容自动选择合适的切分方式。
    """

    def __init__(self, headers_to_split_on: list[tuple[str, str]] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._html_splitter = HTMLHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on or [
                ("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")
            ]
        )

    def _split_html_content(self, content: str) -> list[Document]:
        """
        根据内容类型选择合适的 HTMLHeaderTextSplitter 方法进行切分。
        :param content: HTML 文本 / 文件路径 / URL
        :return: 切分后的 Document 列表
        """
        if content.startswith("http://") or content.startswith("https://"):
            return self._html_splitter.split_text_from_url(content)
        elif os.path.exists(content) and os.path.isfile(content):
            return self._html_splitter.split_text_from_file(content)
        else:
            # 视为 HTML 文本内容
            return self._html_splitter.split_text(content)

    def split_text(self, text: str) -> list[str]:
        """按 HTML 标题层级切分，返回各分块的文本内容。"""
        docs = self._split_html_content(text)
        return [doc.page_content for doc in docs]

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """
        按 HTML 标题层级切分 Document 列表。
        保留原始 Document 的 metadata，并合并 HTMLHeaderTextSplitter 产生的标题层级元数据。
        """
        result = []
        for doc in documents:
            sub_docs = self._split_html_content(doc.page_content)
            for sub_doc in sub_docs:
                merged_metadata = {**doc.metadata, **sub_doc.metadata}
                merged_metadata = self.add_metadata(merged_metadata)
                result.append(Document(page_content=sub_doc.page_content, metadata=merged_metadata))
        return result

    def create_documents(self, texts: list[str], metadatas: list[dict] | None = None) -> list[Document]:
        """将原始文本/路径/URL 列表按 HTML 标题层级切分为 Document 列表。"""
        result = []
        _metadatas = metadatas or [{}] * len(texts)
        for text, metadata in zip(texts, _metadatas):
            sub_docs = self._split_html_content(text)
            for sub_doc in sub_docs:
                merged_metadata = {**metadata, **sub_doc.metadata}
                merged_metadata = self.add_metadata(merged_metadata)
                result.append(Document(page_content=sub_doc.page_content, metadata=merged_metadata))
        return result



class CodeTextSplitter(TextSplitterMetadataMixin, RecursiveCharacterTextSplitter):
    """
    代码分割器：RecursiveCharacterTextSplitter 子类
    """

    def __init__(self, **kwargs):
        super().__init__(is_separator_regex=True, **kwargs)

    def create_documents(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        metadatas_ = metadatas or [{}] * len(texts)
        for m in metadatas_:
            self.add_metadata(m)
        return super().create_documents(texts, metadatas_)

    def set_separators_for_language(self, language: Language):
        """根据语言设置适合的 separators"""
        self._separators = self.get_separators_for_language(language)
        self._is_separator_regex = True


class UniversalTextSplitter(TextSplitterMetadataMixin, RecursiveCharacterTextSplitter):
    """
    通用分割器：RecursiveCharacterTextSplitter 子类
    """

    def create_documents(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        metadatas_ = metadatas or [{}] * len(texts)
        for m in metadatas_:
            self.add_metadata(m)
        return super().create_documents(texts, metadatas_)


class ChildTextSplitter(TextSplitterMetadataMixin, RecursiveCharacterTextSplitter):
    """
    子分割器：RecursiveCharacterTextSplitter 子类
    """

    def create_documents(
        self, texts: list[str], metadatas: list[dict[Any, Any]] | None = None
    ) -> list[Document]:
        metadatas_ = metadatas or [{}] * len(texts)
        for m in metadatas_:
            self.add_metadata(m)
        return super().create_documents(texts, metadatas_)


# ======================== SplitterRegistry ========================

class SplitterEntry:
    """
    已注册的分块器条目。
    - text_splitter: TextSplitter 实例
    - supported_file_types: 支持的文件后缀集合，包含 MATCH_ALL("*") 表示匹配所有类型
    - order: 优先级，越小越优先（相同文件类型下，order 最小的优先选中）
    - name: 名称，默认使用传入的 TextSplitter 实例的类名
    """

    def __init__(self, text_splitter: TextSplitter, supported_file_types: set[str],
                 name: str | None = None, order: int = 0):
        self.text_splitter = text_splitter
        self.supported_file_types = supported_file_types
        self.name = name or text_splitter.__class__.__name__
        self.order = order

    def supports(self, file_type: str) -> bool:
        """判断是否支持该文件类型"""
        return MATCH_ALL in self.supported_file_types or file_type in self.supported_file_types


class SplitterRegistry:
    """
    分块器注册表（parent_splitter 选择器）：
    - 根据文件类型选择优先级最高的 TextSplitter 作为 parent_splitter
    - 代码分块器共用一个 RecursiveCharacterTextSplitter，使用前动态调整 separators
    - register() 注册 SplitterEntry，支持链式调用
    - get_splitter(file_type) 返回当前文件类型下 order 最小的 TextSplitter,
                            相同文件类型的多个 Splitter， order越小 + 越靠前注册的优先
    """

    _instance: "SplitterRegistry | None" = None
    # 代码分块器：所有语言共用，使用前动态调整 separators
    _code_splitter: RecursiveCharacterTextSplitter | None = None
    # 子分块器
    _child_splitter: RecursiveCharacterTextSplitter | None = None
    child_splitter_name: str | None = None

    def __init__(self):
        self._entries: list[SplitterEntry] = []
        self._sorted = True


    @classmethod
    def init(cls) -> "SplitterRegistry":
        """
        初始化 SplitterRegistry 单例，注册所有内置的 parent splitter。
        从 global_config 读取 chunking.parent 配置。
        """
        if cls._instance is not None:
            print("INIT SPLITTER REGISTRY: Instance already exists, returning cached instance.")
            return cls._instance

        from app.config.global_config import global_config
        parent_cfg = global_config.get("chunking", {}).get("parent", {})
        parent_chunk_size = parent_cfg.get("chunk_size", 512)
        parent_chunk_overlap = parent_cfg.get("chunk_overlap", 100)

        child_cfg = global_config.get("chunking", {}).get("child", {})
        child_chunk_size = child_cfg.get("chunk_size", 256)
        child_chunk_overlap = child_cfg.get("chunk_overlap", 50)

        print(f"INIT SPLITTER REGISTRY: Initializing (parent_chunk_size={parent_chunk_size}, parent_chunk_overlap={parent_chunk_overlap})...")
        print(f"INIT SPLITTER REGISTRY: Initializing (child_chunk_size={child_chunk_size}, child_chunk_overlap={child_chunk_overlap})...")

        registry = cls()

        # 1. Markdown 分块器（order=10，优先于通用分块器）
        md_splitter = MarkdownTextSplitterAdapter(strip_headers=False)
        registry.register(SplitterEntry(
            text_splitter=md_splitter,
            supported_file_types={".md"},
            order=10)
        )

        # 2. HTML 分块器（order=10）
        html_splitter = HTMLTextSplitterAdapter()
        registry.register(SplitterEntry(
            text_splitter=html_splitter,
            supported_file_types={".html", ".htm"},
            order=10)
        )

        # 3. 代码分块器：所有语言共用一个 CodeTextSplitter（order=10）
        #    使用前由 get_splitter() 调用 set_separators_for_language 动态调整
        cls._code_splitter = CodeTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )
        registry.register(SplitterEntry(
            text_splitter=cls._code_splitter,
            supported_file_types=set(EXT_TO_LANGUAGE.keys()),
            order=10)
        )

        # 4. 通用分块器（order=100，最后匹配）
        universal_splitter = UniversalTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap
        )
        registry.register(SplitterEntry(
            text_splitter=universal_splitter,
            supported_file_types={MATCH_ALL},
            order=100)
        )

        # 5. 子分块器，不注册到链上
        cls._child_splitter = ChildTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            add_start_index=True
        )
        cls.child_splitter_name = cls._child_splitter.__class__.__name__

        cls._instance = registry

        registered = registry.entries
        print(f"INIT SPLITTER REGISTRY: Initialized successfully, {len(registered)} entries registered. "
              f"Registry: {', '.join(f'{e.name}(order={e.order})' for e in registered)}. "
              f"Child splitter: {cls.child_splitter_name}")
        return cls._instance

    def register(self, entry: SplitterEntry) -> "SplitterRegistry":
        """注册 SplitterEntry，支持链式调用。"""
        if not isinstance(entry, SplitterEntry):
            raise TypeError(f"Expected SplitterEntry instance, got {type(entry).__name__}")
        self._entries.append(entry)
        self._sorted = False
        return self

    def _ensure_sorted(self):
        """确保 entries 按 order 升序 + 注册顺序排列（stable sort）"""
        if not self._sorted:
            self._entries.sort(key=lambda e: e.order)
            self._sorted = True

    @property
    def entries(self) -> list[SplitterEntry]:
        self._ensure_sorted()
        return list(self._entries)

    def get_splitter(self, file_type: str) -> tuple[str, TextSplitter]:
        """
        返回第一个支持该文件类型的 (splitter_name, TextSplitter)（order越小 + 越靠前注册的优先）。
        对于代码文件，会在返回前动态调整共用 splitter 的 separators。
        :param file_type: 文件后缀
        :return: 元组(splitter_name, TextSplitter)
        :raises ValueError: 没有匹配的 Splitter
        """
        self._ensure_sorted()
        for entry in self._entries:
            if entry.supports(file_type):
                # 如果是代码文件，动态设置该语言的 separators
                language = EXT_TO_LANGUAGE.get(file_type)
                if language is not None and isinstance(entry.text_splitter, CodeTextSplitter):
                    entry.text_splitter.set_separators_for_language(language)
                    name = f"{entry.name}({language.value})"
                    print(f"SPLITTER REGISTRY: Selected '{name}' for file type '{file_type}'")
                    return name, entry.text_splitter
                print(f"SPLITTER REGISTRY: Selected '{entry.name}' for file type '{file_type}'")
                return entry.name, entry.text_splitter
        raise ValueError(f"No splitter registered for file type: {file_type}")

    @classmethod
    def get_child_splitter(cls) -> TextSplitter:
        """获取子分块器实例。未初始化时抛出异常。"""
        if cls._child_splitter is None:
            raise RuntimeError("Child splitter has not been initialized. Call SplitterRegistry.init() first.")
        return cls._child_splitter

    @classmethod
    def get_instance(cls) -> "SplitterRegistry":
        """获取 SplitterRegistry 单例。未初始化时抛出异常。"""
        if cls._instance is None:
            raise RuntimeError("SplitterRegistry has not been initialized. Call SplitterRegistry.init() first.")
        return cls._instance
