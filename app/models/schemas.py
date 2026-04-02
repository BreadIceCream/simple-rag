import uuid
from datetime import datetime

from sqlalchemy import String, Integer, DateTime, JSON, Text, Float
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
    file_summary: Mapped[str] = mapped_column(Text, nullable=True, comment="文件内容摘要")
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


class OnlineEvalRun(Base):
    __tablename__ = "online_eval_run"

    request_id: Mapped[str] = mapped_column(String(64), primary_key=True, comment="Unique request id")
    conversation_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True, comment="Conversation id")
    thread_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True, comment="Thread id")
    event_date: Mapped[str] = mapped_column(String(10), nullable=False, index=True, comment="Request event date")
    request_created_at: Mapped[str] = mapped_column(String(32), nullable=False, comment="Request created timestamp")
    evaluation_created_at: Mapped[str] = mapped_column(String(32), nullable=True, comment="Evaluation created timestamp")
    query_type: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown", index=True, comment="Query type")
    hop_count: Mapped[str] = mapped_column(String(16), nullable=False, default="unknown", comment="Hop count")
    abstraction_level: Mapped[str] = mapped_column(String(16), nullable=False, default="unknown", comment="Abstraction level")
    query_type_source: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown", comment="Query type source")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown", index=True, comment="Evaluation status")
    latency_ms: Mapped[float] = mapped_column(Float, nullable=True, comment="Request latency milliseconds")
    rewrite_count: Mapped[int] = mapped_column(Integer, nullable=True, comment="Rewrite count")
    generate_count: Mapped[int] = mapped_column(Integer, nullable=True, comment="Generate count")
    retrieved_doc_count: Mapped[int] = mapped_column(Integer, nullable=True, comment="Retrieved document count")
    retrieved_file_count: Mapped[int] = mapped_column(Integer, nullable=True, comment="Retrieved file count")
    retrieved_context_count: Mapped[int] = mapped_column(Integer, nullable=True, comment="Retrieved context count")
    graph_message_count: Mapped[int] = mapped_column(Integer, nullable=True, comment="Graph message count")
    graph_event_count: Mapped[int] = mapped_column(Integer, nullable=True, comment="Graph event count")
    query_type_confidence: Mapped[float] = mapped_column(Float, nullable=True, comment="Query type confidence")
    successful_metric_count: Mapped[int] = mapped_column(Integer, nullable=True, comment="Successful metric count")
    metric_timeout_seconds: Mapped[float] = mapped_column(Float, nullable=True, comment="Metric timeout seconds")
    user_input: Mapped[str] = mapped_column(Text, nullable=False, comment="User input")
    actual_response: Mapped[str] = mapped_column(Text, nullable=False, comment="Actual response")
    reference_answer: Mapped[str] = mapped_column(Text, nullable=True, comment="Reference answer")
    error_message: Mapped[str] = mapped_column(Text, nullable=True, comment="Evaluation error message")
    actual_contexts: Mapped[list] = mapped_column(JSON, default=list, comment="Retrieved contexts")
    actual_doc_ids: Mapped[list] = mapped_column(JSON, default=list, comment="Retrieved document ids")
    actual_file_ids: Mapped[list] = mapped_column(JSON, default=list, comment="Retrieved file ids")
    graph_messages: Mapped[list] = mapped_column(JSON, default=list, comment="Graph messages")
    graph_events: Mapped[list] = mapped_column(JSON, default=list, comment="Graph events")
    reference_contexts: Mapped[list] = mapped_column(JSON, default=list, comment="Reference contexts")
    reference_context_ids: Mapped[list] = mapped_column(JSON, default=list, comment="Reference context ids")
    metrics: Mapped[dict] = mapped_column(JSON, default=dict, comment="Metric values")
    skipped_metrics: Mapped[list] = mapped_column(JSON, default=list, comment="Skipped metric names")
    query_type_reasons: Mapped[list] = mapped_column(JSON, default=list, comment="Query type reasons")
    metric_names: Mapped[list] = mapped_column(JSON, default=list, comment="Requested metric names")
    metric_failures: Mapped[dict] = mapped_column(JSON, default=dict, comment="Metric failures")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict, comment="Additional metadata")
