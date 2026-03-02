from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.config.db_config import DatabaseManager
from app.config.global_config import global_config
from app.core.chunking import SplitterChainFactory
from app.core.embeddings import EmbeddingModelFactory
from app.core.vector_store import VectorStoreFactory
from app.exception.exception_handler import register_exception_handler
from app.routers import document


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：按依赖顺序初始化
    global_config.load()
    DatabaseManager.init()
    await DatabaseManager.init_db()
    SplitterChainFactory.init_text_splitters()
    EmbeddingModelFactory.init_embedding_model()
    VectorStoreFactory.init_vector_store(
        embeddings=EmbeddingModelFactory.get_instance(),
        collection_name=global_config.get("vector_store").get("collection_name")
    )
    yield
    # 关闭：清理资源
    await DatabaseManager.close()

app = FastAPI(lifespan=lifespan)

register_exception_handler(app)

origins = [
    "http://localhost:3000",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(document.router)
