"""
策略模型
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from ..database import Base


class Strategy(Base):
    """策略模型"""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # 策略配置
    factors = Column(JSON, nullable=True)  # 使用的因子列表
    symbols = Column(JSON, nullable=True)  # 交易标的
    frequency = Column(String(10), default="5m")  # 时间框架

    # 状态
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关联关系
    owner = relationship("User", back_populates="strategies")
    versions = relationship("StrategyVersion", back_populates="strategy")
    backtests = relationship("Backtest", back_populates="strategy")

    def __repr__(self):
        return f"<Strategy(id={self.id}, name={self.name})>"


class StrategyVersion(Base):
    """策略版本模型"""

    __tablename__ = "strategy_versions"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    version = Column(String(20), nullable=False)  # 语义版本号

    # 版本配置
    config = Column(JSON, nullable=False)  # 完整配置
    config_hash = Column(String(64), nullable=False)  # 配置哈希

    # 元数据
    changelog = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关联关系
    strategy = relationship("Strategy", back_populates="versions")

    def __repr__(self):
        return f"<StrategyVersion(id={self.id}, version={self.version})>"
