"""
API Pydantic Schemas

定义请求/响应的数据验证模型
"""

from .user import UserCreate, UserUpdate, UserResponse, UserLogin, Token
from .strategy import StrategyCreate, StrategyUpdate, StrategyResponse
from .backtest import BacktestCreate, BacktestResponse, BacktestResultResponse
from .signal import SignalResponse, SignalCreate
from .common import PaginatedResponse, ErrorResponse, HealthResponse

__all__ = [
    # User
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "UserLogin",
    "Token",
    # Strategy
    "StrategyCreate",
    "StrategyUpdate",
    "StrategyResponse",
    # Backtest
    "BacktestCreate",
    "BacktestResponse",
    "BacktestResultResponse",
    # Signal
    "SignalResponse",
    "SignalCreate",
    # Common
    "PaginatedResponse",
    "ErrorResponse",
    "HealthResponse",
]
