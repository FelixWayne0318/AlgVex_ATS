"""
快照模型
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Integer, String, Text

from ..database import Base


class Snapshot(Base):
    """数据快照模型"""

    __tablename__ = "snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_id = Column(String(100), unique=True, index=True, nullable=False)

    # 快照信息
    cutoff_time = Column(DateTime, index=True, nullable=False)
    symbols = Column(JSON, nullable=True)  # 包含的标的列表
    data_sources = Column(JSON, nullable=True)  # 数据源列表

    # 存储信息
    storage_path = Column(String(500), nullable=True)
    content_hash = Column(String(64), nullable=True)
    size_bytes = Column(Integer, nullable=True)

    # 元数据
    metadata = Column(JSON, nullable=True)
    description = Column(Text, nullable=True)

    # 状态
    is_valid = Column(Integer, default=1)  # 0: invalid, 1: valid
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Snapshot(id={self.id}, snapshot_id={self.snapshot_id})>"
