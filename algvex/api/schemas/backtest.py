"""
回测 Schema
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BacktestBase(BaseModel):
    """回测基础字段"""

    name: Optional[str] = Field(None, max_length=100)
    start_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="开始日期 YYYY-MM-DD")
    end_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="结束日期 YYYY-MM-DD")
    symbols: List[str] = Field(..., min_length=1, description="交易标的列表")
    initial_capital: float = Field(default=100000.0, gt=0)
    frequency: str = Field(default="5m")


class BacktestCreate(BacktestBase):
    """创建回测"""

    strategy_id: Optional[int] = Field(None, description="关联策略ID")
    config: Optional[Dict[str, Any]] = Field(None, description="回测配置")
    snapshot_id: Optional[str] = Field(None, description="使用的数据快照ID")


class BacktestResponse(BacktestBase):
    """回测响应"""

    id: int
    owner_id: int
    strategy_id: Optional[int]
    config_hash: Optional[str]
    snapshot_id: Optional[str]
    status: str
    progress: float
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class BacktestResultResponse(BaseModel):
    """回测结果响应"""

    id: int
    backtest_id: int

    # 收益指标
    total_return: Optional[float] = Field(None, description="总收益率")
    annual_return: Optional[float] = Field(None, description="年化收益率")
    sharpe_ratio: Optional[float] = Field(None, description="夏普比率")
    sortino_ratio: Optional[float] = Field(None, description="索提诺比率")
    calmar_ratio: Optional[float] = Field(None, description="卡尔玛比率")

    # 风险指标
    max_drawdown: Optional[float] = Field(None, description="最大回撤")
    max_drawdown_duration: Optional[int] = Field(None, description="最大回撤持续天数")
    volatility: Optional[float] = Field(None, description="波动率")

    # 交易统计
    total_trades: Optional[int] = Field(None, description="总交易次数")
    win_rate: Optional[float] = Field(None, description="胜率")
    profit_factor: Optional[float] = Field(None, description="盈亏比")
    avg_trade_return: Optional[float] = Field(None, description="平均交易收益")

    created_at: datetime

    class Config:
        from_attributes = True


class BacktestWithResult(BacktestResponse):
    """包含结果的回测响应"""

    result: Optional[BacktestResultResponse] = None


class BacktestDetailResponse(BacktestWithResult):
    """回测详情响应 (包含曲线数据)"""

    equity_curve: Optional[List[Dict[str, Any]]] = Field(None, description="权益曲线")
    monthly_returns: Optional[Dict[str, float]] = Field(None, description="月度收益")
    trade_list: Optional[List[Dict[str, Any]]] = Field(None, description="交易列表")
