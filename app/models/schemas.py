import uuid
from datetime import datetime

from pydantic import Field, BaseModel, ConfigDict
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

class EmbeddedDocumentVO(BaseModel):
    """
    嵌入文档的视图对象（VO），用于 API 层与前端交互，不包含分块相关信息
    """

    id: str = Field(..., description="文档唯一标识(UUID)")
    path: str = Field(..., description="文件路径/url,唯一")
    is_url: bool = Field(..., description="是否为URL")
    file_directory: str | None = Field(None, description="文件目录,仅本地文件使用")
    file_name: str | None = Field(None, description="文件名")
    file_extension: str = Field(..., description="文件后缀名(如 .pdf)")
    mime_type: str | None = Field(None, description="文件mime类型")
    last_modified: datetime | None = Field(None, description="文件最后修改时间,仅本地文件使用")

    model_config = ConfigDict(
        from_attributes=True # 支持从 ORM 模型创建 Pydantic 模型
    )