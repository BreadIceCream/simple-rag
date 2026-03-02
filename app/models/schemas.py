import uuid
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict
from sqlalchemy import String, Integer, DateTime, JSON, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, comment="创建时间")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now, comment="更新时间")

class EmbeddedDocument(Base):
    __tablename__ = "embedded_document"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=str(uuid.uuid4), comment="文档唯一标识(UUID)")
    filename: Mapped[str] = mapped_column(String(255), nullable=False, comment="文件名")
    file_path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False, comment="文件绝对路径，唯一")
    file_extension: Mapped[str] = mapped_column(String(20), nullable=False, comment="文件后缀名(如 .pdf)")
    filetype: Mapped[str] = mapped_column(String(100), nullable=True, comment="文件类型(如 pdf, md, txt, docx, html, csv, code)")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, comment="分块数量")
    chunk_ids: Mapped[list[str]] = mapped_column(JSON, default=list, comment="所有 chunk 的 ID 列表")
    doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict, comment="处理元数据(embedding 模型、分块参数等)")

    def __repr__(self):
        return f"<EmbeddedDocument(id={self.id}, filename={self.filename}), file_path={self.file_path}, chunk_count={self.chunk_count}>"

class EmbeddedDocumentVO(BaseModel):

    id: str = Field(..., description="文档唯一标识(UUID)")
    filename: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件绝对路径，唯一")
    file_extension: str = Field(..., description="文件后缀名(如 .pdf)")
    filetype: str | None = Field(None, description="文件类型(如 pdf, md, txt, docx, html, csv, code)")
    chunk_count: int = Field(0, description="分块数量")

    model_config = ConfigDict(
        from_attributes=True # 支持从 ORM 模型创建 Pydantic 模型
    )