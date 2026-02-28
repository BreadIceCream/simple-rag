import dotenv
import os


# 设置环境变量
def load_env():
    print("LOAD ENV: Loading environment variables...")
    dotenv.load_dotenv(override=True)
    os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING")
    if os.environ["LANGSMITH_TRACING"] == "true":
        os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
    os.environ["MODEL_PROVIDER"] = os.getenv("MODEL_PROVIDER")
    os.environ["MODEL_NAME"] = os.getenv("MODEL_NAME")
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["EMBEDDING_MODEL"] = os.getenv("EMBEDDING_MODEL")
    os.environ["OPENAI_EMBEDDING"] = os.getenv("OPENAI_EMBEDDING")
    if os.environ["OPENAI_EMBEDDING"] == "true":
        os.environ["OPENAI_EMBEDDING_API_BASE"] = os.getenv("OPENAI_EMBEDDING_BASE_URL")
        os.environ["OPENAI_EMBEDDING_API_KEY"] = os.getenv("OPENAI_EMBEDDING_API_KEY")
    os.environ["RERANKER_ENABLED"] = os.getenv("RERANKER_ENABLED")
    if os.environ["RERANKER_ENABLED"] == "true":
        os.environ["QWEN_RERANKER"] = os.getenv("QWEN_RERANKER")
