from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, MarkdownHeaderTextSplitter, Language, \
    HTMLHeaderTextSplitter

from app.models.common import SplitResult

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

# 通配符，表示匹配所有文件类型
MATCH_ALL = "*"


class Splitter:
    """
    基础分块器(拦截器)。
    每个 Splitter 声明自己支持的文件类型集合，以及一个 order(执行优先级，越小越先执行)。
    supported_file_types 中包含 MATCH_ALL ("*") 表示匹配所有文件类型。
    """
    def __init__(
        self,
        text_splitter: TextSplitter | HTMLHeaderTextSplitter | MarkdownHeaderTextSplitter,
        supported_file_types: set[str],
        only_raw: bool = False,
        order: int = 0,
        name: str = "",
    ):
        self.text_splitter = text_splitter
        self.supported_file_types = supported_file_types
        self.only_raw = only_raw  # 是否只处理原始输入（不事先经过loader），而非 Document 列表
        self.order = order
        self.name = name or self.__class__.__name__

    def supports(self, file_type: str, file_is_raw: bool) -> bool:
        """判断是否支持处理该文件类型"""
        if self.only_raw and not file_is_raw:
            return False
        return MATCH_ALL in self.supported_file_types or file_type in self.supported_file_types

    def do_split(self, docs: list[Document] | list[str] | str, file_type: str = None) -> (bool, list[Document]):
        """执行分块，子类可重写"""
        try:
            if isinstance(docs, str):
                processed = self.text_splitter.split_text(docs)
            elif isinstance(docs, list) and isinstance(docs[0], str):
                processed = self.text_splitter.create_documents(docs)
            else:
                processed = self.text_splitter.split_documents(docs)
            return True, processed
        except Exception as e:
            print(f"SPLIT ERROR in {self.name} for file type '{file_type}': {e}")
            return False, docs


class MarkdownSplitter(Splitter):
    """
    Markdown 分块器：按标题层级切分，支持 .md 文件。
    order=10，优先于通用分块器。
    """
    def __init__(self, order: int = 10):
        super().__init__(
            text_splitter=MarkdownHeaderTextSplitter(
                [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4")],
                strip_headers=False,
            ),
            supported_file_types={".md"},
            only_raw=True, # 只处理原始输入
            order=order,
            name="MarkdownHeaderTextSplitter",
        )

    def do_split(self, docs: str, file_type: str = None) -> (bool, list[Document]):
        # 使用 MarkdownHeaderTextSplitter 进行分块，注意只能接收str，使用split_text方法
        if isinstance(docs, str):
            processed = self.text_splitter.split_text(docs)
            return True, processed
        return False, docs


class HTMLSplitter(Splitter):
    """
    HTML 分块器：按标题层级切分，支持 .html/.htm 文件。
    order=10，优先于通用分块器。
    """

    def __init__(self, order: int = 10):
        super().__init__(
            text_splitter=HTMLHeaderTextSplitter(
                [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3"), ("h4", "Header 4")]
            ),
            supported_file_types={".html", ".htm"},
            only_raw=True,  # 只处理原始输入(不经过loader)
            order=order,
            name="HTMLHeaderTextSplitter",
        )

    def do_split(self, docs: str, file_type: str = None) -> (bool, list[Document]):
        # 使用 HTMLHeaderTextSplitter 进行分块，注意只能接收str(url或者是file_path)
        if isinstance(docs, str):
            if docs.startswith("http"):
                processed = self.text_splitter.split_text_from_url(docs)
            else:
                processed = self.text_splitter.split_text_from_file(docs)
            return True, processed
        return False, docs


class CodeSplitter(Splitter):
    """
    代码分块器：持有单个 RecursiveCharacterTextSplitter 实例，
    在 do_split 时根据文件后缀动态加载对应语言的 separators 进行分块。
    """
    def __init__(self, text_splitter: RecursiveCharacterTextSplitter, order: int = 10):
        super().__init__(
            text_splitter=text_splitter,
            supported_file_types=set(EXT_TO_LANGUAGE.keys()),
            only_raw=False,
            order=order,
            name="CodeSplitter",
        )

    def do_split(self, docs: list[Document] | list[str] | str, file_type: str = None) -> list[Document]:
        language = EXT_TO_LANGUAGE.get(file_type)
        if language is None:
            raise ValueError(f"Unsupported code file type: {file_type}")
        # 动态加载该语言的 separators
        self.text_splitter._separators = RecursiveCharacterTextSplitter.get_separators_for_language(language)
        self.text_splitter._is_separator_regex = True
        return super().do_split(docs, file_type)


class SplitterChain:
    """
    分块器链(拦截器链)：
    - 通过 register() 注册 Splitter，支持动态扩展
    - 执行时，按 order 排序，所有匹配当前文件类型的 Splitter 依次加工
    - 前一个 Splitter 的输出作为后一个 Splitter 的输入(链式加工)
    - 若无任何 Splitter 匹配，返回原始文档列表
    """
    def __init__(self):
        self._splitters: list[Splitter] = []
        self._sorted = True  # 标记是否已按 order 排序

    def register(self, splitter: Splitter) -> "SplitterChain":
        """
        注册一个 Splitter 到链中。支持链式调用。
        :param splitter: 分块器实例
        :return: self，支持链式调用
        """
        self._splitters.append(splitter)
        self._sorted = False
        return self

    def _ensure_sorted(self):
        """确保 splitters 按 order 升序排列"""
        if not self._sorted:
            self._splitters.sort(key=lambda s: s.order)
            self._sorted = True

    @property
    def splitters(self) -> list[Splitter]:
        self._ensure_sorted()
        return list(self._splitters)

    def do_split(self, file_type: str, docs: list[Document] | list[str], file_is_raw: bool = False) -> SplitResult:
        """
        对文档执行链式分块：遍历所有已注册且匹配当前文件类型的 Splitter，
        前一个的输出作为后一个的输入。
        :param file_type: 文件后缀(如 ".md", ".py")
        :param docs: 待分块的文档列表
        :param file_is_raw: docs 是否为原始输入（未经过loader），而非 Document 列表
        :return: 最终分块结果
        """
        self._ensure_sorted()

        result = docs
        applied_splitters = []

        for splitter in self._splitters:
            if splitter.supports(file_type, file_is_raw):
                handled, result = splitter.do_split(result, file_type)
                if handled:
                    file_is_raw = False
                    applied_splitters.append(splitter.name)

        if not applied_splitters:
            print(f"SPLITTER CHAIN: No splitter matched for file type '{file_type}', returning original docs.")
        else:
            print(f"SPLITTER CHAIN: Applied {len(applied_splitters)} splitters for '{file_type}': {' → '.join(applied_splitters)}")

        return SplitResult(splitters=applied_splitters, chunks=result)


class SplitterChainFactory:
    """
    分块器链工厂(单例)：通过 init_text_splitters 创建并缓存唯一的 SplitterChain 实例。
    chunk_size 和 chunk_overlap 从 global_config 的 chunking 配置中读取。
    """
    _instance: SplitterChain | None = None

    @classmethod
    def init_text_splitters(cls) -> SplitterChain:
        """
        从 global_config 读取 chunking 配置，初始化文本分块器链并缓存为单例。
        若实例已存在则直接返回，不会重复创建。
        """
        if cls._instance is not None:
            print("INIT TEXT SPLITTERS: SplitterChain instance already exists, returning cached instance.")
            return cls._instance

        from app.config.global_config import global_config
        chunking_cfg = global_config.get("chunking", {})
        chunk_size = chunking_cfg.get("chunk_size", 512)
        chunk_overlap = chunking_cfg.get("chunk_overlap", 100)

        print(f"INIT TEXT SPLITTERS: Initializing (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")

        # 通用的 RecursiveCharacterTextSplitter，用于二次加工
        recursive_splitter = Splitter(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
            ),
            supported_file_types={MATCH_ALL},
            order=100,  # 最后执行，对所有类型生效
            name="RecursiveCharacterTextSplitter(fallback)",
        )

        # Markdown 分块器：按标题层级切分(order=10，优先于通用分块器)
        md_splitter = MarkdownSplitter(order=10)

        # HTML 分块器：按标题层级切分(order=10，优先于通用分块器)
        html_splitter = HTMLSplitter(order=10)

        # 代码分块器：按语言语法切分(order=10)
        code_splitter = CodeSplitter(
            text_splitter=RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
            ),
            order=10,
        )

        # 构建链：通过 register 注册，按 order 排序
        cls._instance = (SplitterChain()
            .register(md_splitter)
            .register(html_splitter)
            .register(code_splitter)
            .register(recursive_splitter)
        )

        registered = cls._instance.splitters
        print(f"INIT TEXT SPLITTERS: Initialized successfully, {len(registered)} splitters registered: "
              f"{', '.join(f'{s.name}(order={s.order})' for s in registered)}")
        return cls._instance

    @classmethod
    def get_instance(cls) -> SplitterChain:
        """获取 SplitterChain 单例。未初始化时抛出异常。"""
        if cls._instance is None:
            raise RuntimeError("SplitterChain has not been initialized. Call init_text_splitters() first.")
        return cls._instance