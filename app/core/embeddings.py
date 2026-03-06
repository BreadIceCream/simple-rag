from getpass import getpass

import torch
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import SecretStr


class EmbeddingModelFactory:
    """
    Embedding 模型工厂(单例)：通过 init_embedding_model 创建并缓存唯一的 Embeddings 实例。
    配置从 global_config 读取。
    """
    _instance: Embeddings | None = None

    @classmethod
    def init_embedding_model(cls) -> Embeddings:
        """
        从 global_config 读取配置，初始化 Embedding 模型并缓存为单例。
        若实例已存在则直接返回，不会重复创建。
        """
        if cls._instance is not None:
            print("INIT EMBEDDING MODEL: Instance already exists, returning cached instance.")
            return cls._instance

        from app.config.global_config import global_config

        print("INIT EMBEDDING MODEL: Initializing embedding model...")
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        print(f"INIT EMBEDDING MODEL: device {device}")
        embedding_config = global_config.get("embedding")
        if embedding_config.get("openai_embedding") == "true":
            print("INIT EMBEDDING MODEL: Using OpenAI embedding model...")
            openai_embedding_config = global_config.get("openai").get("embedding")
            base_url = openai_embedding_config.get("base_url")
            cls._instance = OpenAIEmbeddings(
                model=openai_embedding_config.get("model"),
                base_url=base_url if base_url else None,
                api_key=SecretStr(openai_embedding_config.get("api_key")),
            )
        else:
            embedding_model = embedding_config.get("model")
            if embedding_config.get("huggingface_remote_inference").get("enabled"):
                print(f"INIT EMBEDDING MODEL REMOTE: Using HuggingFace Inference API embedding model <{embedding_model}>...")
                api_token = embedding_config.get("huggingface_remote_inference").get("api_token")
                if not api_token:
                    api_token = getpass("请输入 HuggingFace Inference API 的 API Token: ")
                cls._instance = HuggingFaceInferenceAPIEmbeddings(
                    api_key=api_token,
                    model_name=embedding_model,
                )
            else:
                print(f"INIT EMBEDDING MODEL LOCAL: Using HuggingFace embedding model <{embedding_model}>...")
                cls._instance = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    model_kwargs={"device": device},
                )
        print("INIT EMBEDDING MODEL: Initialized successfully.")
        return cls._instance

    @classmethod
    def get_instance(cls) -> Embeddings:
        """获取 Embeddings 单例。未初始化时抛出异常。"""
        if cls._instance is None:
            raise RuntimeError("Embedding model has not been initialized. Call init_embedding_model() first.")
        return cls._instance