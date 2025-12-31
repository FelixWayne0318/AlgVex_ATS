"""
信号模型
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import relationship

from ..database import Base


class Signal(Base):
    """信号模型"""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(String(100), unique=True, index=True, nullable=False)

    # 信号信息
    symbol = Column(String(20), index=True, nullable=False)
    frequency = Column(String(10), default="5m")
    bar_time = Column(DateTime, index=True, nullable=False)

    # 信号值
    raw_prediction = Column(Float, nullable=True)
    final_signal = Column(Integer, nullable=True)  # -1, 0, 1

    # 因子值
    factors = Column(JSON, nullable=True)  # {factor_name: value}

    # 哈希
    data_hash = Column(String(64), nullable=True)
    features_hash = Column(String(64), nullable=True)
    config_hash = Column(String(64), nullable=True)

    # 元数据
    mode = Column(String(10), default="live")  # live / replay
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关联关系
    traces = relationship("SignalTrace", back_populates="signal")

    def __repr__(self):
        return f"<Signal(id={self.id}, signal_id={self.signal_id})>"


class SignalTrace(Base):
    """信号追踪模型"""

    __tablename__ = "signal_traces"

    id = Column(Integer, primary_key=True, index=True)
    signal_id = Column(Integer, ForeignKey("signals.id"), nullable=False)

    # Trace 信息
    trace_type = Column(String(20), nullable=False)  # factor / model / risk / execution
    step_name = Column(String(50), nullable=False)

    # Trace 数据
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)

    # 性能
    duration_ms = Column(Float, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关联关系
    signal = relationship("Signal", back_populates="traces")

    def __repr__(self):
        return f"<SignalTrace(id={self.id}, type={self.trace_type})>"
