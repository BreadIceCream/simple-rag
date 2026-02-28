from langchain_text_splitters import RecursiveCharacterTextSplitter


# 初始化文本分词器
async def init_text_splitter(chunk_size: int = 512, chunk_overlap: int = 100) -> RecursiveCharacterTextSplitter:
    print("INIT TEXT SPLITTER: Initializing text splitter...")
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
