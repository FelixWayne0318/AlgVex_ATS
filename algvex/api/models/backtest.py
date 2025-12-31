"""
回测模型
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from ..database import Base


class BacktestStatus(str, Enum):
    """回测状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Backtest(Base):
    """回测任务模型"""

    __tablename__ = "backtests"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=True)

    # 回测参数
    start_date = Column(String(10), nullable=False)  # YYYY-MM-DD
    end_date = Column(String(10), nullable=False)
    symbols = Column(JSON, nullable=False)  # 交易标的列表
    initial_capital = Column(Float, default=100000.0)
    frequency = Column(String(10), default="5m")

    # 配置
    config = Column(JSON, nullable=True)  # 回测配置
    config_hash = Column(String(64), nullable=True)
    snapshot_id = Column(String(100), nullable=True)  # 使用的数据快照

    # 状态
    status = Column(String(20), default=BacktestStatus.PENDING)
    progress = Column(Float, default=0.0)  # 0-100
    error_message = Column(Text, nullable=True)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 关联关系
    owner = relationship("User", back_populates="backtests")
    strategy = relationship("Strategy", back_populates="backtests")
    result = relationship("BacktestResult", back_populates="backtest", uselist=False)

    def __repr__(self):
        return f"<Backtest(id={self.id}, status={self.status})>"


class BacktestResult(Base):
    """回测结果模型"""

    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    backtest_id = Column(Integer, ForeignKey("backtests.id"), unique=True, nullable=False)

    # 收益指标
    total_return = Column(Float, nullable=True)
    annual_return = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    sortino_ratio = Column(Float, nullable=True)
    calmar_ratio = Column(Float, nullable=True)

    # 风险指标
    max_drawdown = Column(Float, nullable=True)
    max_drawdown_duration = Column(Integer, nullable=True)  # 天数
    volatility = Column(Float, nullable=True)

    # 交易统计
    total_trades = Column(Integer, nullable=True)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    avg_trade_return = Column(Float, nullable=True)

    # 详细数据 (JSON)
    equity_curve = Column(JSON, nullable=True)  # 权益曲线
    monthly_returns = Column(JSON, nullable=True)  # 月度收益
    trade_list = Column(JSON, nullable=True)  # 交易列表

    # 元数据
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关联关系
    backtest = relationship("Backtest", back_populates="result")

    def __repr__(self):
        return f"<BacktestResult(id={self.id}, sharpe={self.sharpe_ratio})>"
