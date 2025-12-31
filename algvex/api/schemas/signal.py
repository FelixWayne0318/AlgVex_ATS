"""
信号 Schema
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SignalBase(BaseModel):
    """信号基础字段"""

    symbol: str = Field(..., description="交易对")
    frequency: str = Field(default="5m", description="时间框架")
    bar_time: datetime = Field(..., description="K线时间")


class SignalCreate(SignalBase):
    """创建信号"""

    raw_prediction: Optional[float] = Field(None, description="原始预测值")
    final_signal: Optional[int] = Field(None, ge=-1, le=1, description="最终信号 (-1/0/1)")
    factors: Optional[Dict[str, float]] = Field(None, description="因子值")
    data_hash: Optional[str] = None
    features_hash: Optional[str] = None
    config_hash: Optional[str] = None
    mode: str = Field(default="live", description="模式 (live/replay)")


class SignalResponse(SignalBase):
    """信号响应"""

    id: int
    signal_id: str
    raw_prediction: Optional[float]
    final_signal: Optional[int]
    factors: Optional[Dict[str, float]]
    data_hash: Optional[str]
    features_hash: Optional[str]
    config_hash: Optional[str]
    mode: str
    created_at: datetime

    class Config:
        from_attributes = True


class SignalTraceResponse(BaseModel):
    """信号追踪响应"""

    id: int
    signal_id: int
    trace_type: str
    step_name: str
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    duration_ms: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class SignalWithTraces(SignalResponse):
    """包含追踪的信号响应"""

    traces: List[SignalTraceResponse] = Field(default_factory=list)


class SignalSummary(BaseModel):
    """信号摘要"""

    date: str
    total_signals: int
    buy_signals: int
    sell_signals: int
    neutral_signals: int
    symbols: List[str]


class AlignmentCheckRequest(BaseModel):
    """对齐检查请求"""

    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="检查日期")


class AlignmentCheckResponse(BaseModel):
    """对齐检查响应"""

    date: str
    passed: bool
    total_live: int
    total_replay: int
    matched: int
    missing_in_replay: int
    missing_in_live: int
    mismatched: int
    max_signal_diff: float
    failure_reasons: List[str]
    generated_at: datetime
