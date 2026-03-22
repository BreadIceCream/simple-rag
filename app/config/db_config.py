from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker, Session

from app.models.schemas import Base

class DatabaseManager:
    """
    数据库管理器(单例)：延迟初始化，确保在 global_config 加载完成后再创建引擎和会话工厂。
    """
    _engine: AsyncEngine | None = None
    _session_factory: async_sessionmaker[AsyncSession] | None = None
    _sync_engine = None
    _sync_session_factory: sessionmaker | None = None

    @classmethod
    def init(cls) -> None:
        """
        从 global_config 读取数据库 URL，创建异步引擎和会话工厂。
        必须在 global_config.load() 之后调用。
        """
        if cls._engine is not None:
            print("DB CONFIG: Already initialized, skipping.")
            return

        from app.config.global_config import global_config
        db_url = global_config.get("database", {}).get("url")
        if not db_url:
            raise ValueError("Database URL not found in config.yml (database.url)")

        print(f"DB CONFIG: Initializing database engine...")
        cls._engine = create_async_engine(
            db_url,
            echo=True,
            pool_size=10,
            max_overflow=20,
        )
        cls._session_factory = async_sessionmaker(
            bind=cls._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # 同步引擎（将 asyncpg 替换为 psycopg2）
        sync_url = db_url.replace("postgresql+asyncpg://", "postgresql+psycopg://")
        cls._sync_engine = create_engine(sync_url, pool_size=5, max_overflow=10)
        cls._sync_session_factory = sessionmaker(bind=cls._sync_engine, expire_on_commit=False)

        print("DB CONFIG: Database engine initialized.")

    @classmethod
    async def init_db(cls) -> None:
        """在应用启动时创建表。"""
        if cls._engine is None:
            raise RuntimeError("DatabaseManager has not been initialized. Call init() first.")
        async with cls._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        print("DB CONFIG: Database tables created.")

    @classmethod
    async def get_db(cls):
        """依赖注入：获取数据库异步会话。"""
        if cls._session_factory is None:
            raise RuntimeError("DatabaseManager has not been initialized. Call init() first.")
        async with cls._session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    @classmethod
    def get_sync_db(cls) -> Session:
        """获取同步数据库会话，用于同步上下文（如线程池中的 Tool）。"""
        if cls._sync_session_factory is None:
            raise RuntimeError("DatabaseManager has not been initialized. Call init() first.")
        return cls._sync_session_factory()

    @classmethod
    async def close(cls) -> None:
        """关闭数据库引擎，用于应用关闭时清理资源。"""
        if cls._engine is not None:
            await cls._engine.dispose()
            print("DB CONFIG: Database engine closed.")
        if cls._sync_engine is not None:
            cls._sync_engine.dispose()
            print("DB CONFIG: Sync database engine closed.")