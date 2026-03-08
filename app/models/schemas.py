import uuid
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, JSON, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="创建时间")

class EmbeddedDocument(Base):
    __tablename__ = "embedded_document"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=str(uuid.uuid4), comment="文档唯一标识(UUID)")
    path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, comment="文件路径/url,唯一")
    is_url: Mapped[bool] = mapped_column(nullable=False, comment="是否为URL")
    file_directory: Mapped[str] = mapped_column(String(512), nullable=True, comment="文件目录,仅本地文件使用")
    file_name: Mapped[str] = mapped_column(String(255), nullable=True, comment="文件名,仅本地文件使用")
    file_extension: Mapped[str] = mapped_column(String(128), nullable=False, comment="后缀名(如 .pdf, .html)")
    mime_type: Mapped[str] = mapped_column(String(128), nullable=True, comment="文件mime类型")
    last_modified: Mapped[str] = mapped_column(String(64), nullable=True, comment="文件最后修改时间,仅本地文件使用")
    parent_doc_ids: Mapped[list[str]] = mapped_column(JSON, default=list, comment="父文档的 ID 列表")
    children_count: Mapped[int] = mapped_column(Integer, default=0, comment="子文档数量")
    load_metadata: Mapped[dict] = mapped_column(JSON, default=dict, comment="加载文档时的元信息，如使用的嵌入模型、加载器、分块器等")

    def __repr__(self):
        return f"<EmbeddedDocument(id={self.id}, file_path={self.path}, filename={self.file_name}), >"


# ======================== 对话相关模型 ========================


class Conversation(Base):
    """对话（线程）记录"""
    __tablename__ = "conversation"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, comment="对话 UUID，同时作为 LangGraph conversation_id")
    title: Mapped[str] = mapped_column(String(255), nullable=False, default="新对话", comment="对话标题（LLM 自动生成）")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, comment="最后更新时间")
    checkpoint_id: Mapped[str] = mapped_column(String(36), nullable=True, comment="LangGraph中断或结束时的checkpoint ID")

    def __repr__(self):
        return f"<Conversation(id={self.id}, title={self.title})>"


class ChatHistory(Base):
    """单条聊天记录"""
    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="消息自增 ID")
    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True, comment="关联的对话 ID（逻辑外键）")
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="消息角色: user / ai")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="消息内容")
    parent_doc_ids: Mapped[list[str]] = mapped_column(JSON, default=list, comment="本次检索到的父文档 doc.id 列表（直接回复时为空）")

    def __repr__(self):
        return f"<ChatHistory(id={self.id}, conversation_id={self.conversation_id}, role={self.role})>"
