from typing import Sequence, Optional, Any

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document


class SimpleCompressor(BaseDocumentCompressor):
    """A simple compressor only return the top_n documents"""

    top_n: int = 7

    def __init__(self, /, **data: Any):
        super().__init__(**data)
        print(f"initializing Simple Compressor.Return top_n {self.top_n}...")

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks: Optional[Callbacks] = None) -> \
    Sequence[Document]:
        return documents[:self.top_n]
