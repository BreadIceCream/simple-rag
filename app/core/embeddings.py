import torch
import os

from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import SecretStr


# 初始化嵌入模型
def init_embedding_model() -> Embeddings:
    print("INIT EMBEDDING MODEL: Initializing embedding model...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("INIT EMBEDDING MODEL: CUDA is available")
    if os.environ["OPENAI_EMBEDDING"] == "true":
        print("INIT EMBEDDING MODEL: Using OpenAI embedding model...")
        return OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"],
                                base_url=os.environ["OPENAI_EMBEDDING_API_BASE"] if os.environ[
                                    "OPENAI_EMBEDDING_API_BASE"] else None,
                                api_key=SecretStr(os.environ["OPENAI_EMBEDDING_API_KEY"]))
    print("INIT EMBEDDING MODEL: Using default HuggingFace embedding model...")
    return HuggingFaceEmbeddings(model_name=os.environ["EMBEDDING_MODEL"], model_kwargs={"device": device})
