"""
AlgVex 因子基础类

设计原则:
- 所有因子继承 BaseFactor
- 统一的计算接口
- 支持可见性检查
- 支持增量计算
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


class FactorFamily(Enum):
    """因子族"""
    PRICE_VOLUME = "price_volume"        # 基础价量
    MOMENTUM = "momentum"                 # 动量
    VOLATILITY = "volatility"            # 波动率
    VOLUME = "volume"                     # 成交量
    PERPETUAL = "perpetual"              # 永续合约专用
    FUNDING = "funding"                   # 资金费率
    OPEN_INTEREST = "open_interest"       # 持仓量
    ORDER_FLOW = "order_flow"             # 订单流
    OPTIONS = "options"                   # 期权
    IMPLIED_VOL = "implied_vol"           # 隐含波动率
    DERIVATIVES = "derivatives"           # 衍生品结构
    ON_CHAIN = "on_chain"                 # 链上
    SENTIMENT = "sentiment"               # 情绪
    MACRO = "macro"                       # 宏观
    COMPOSITE = "composite"               # 复合/ML


class DataDependency(Enum):
    """数据依赖"""
    KLINES_5M = "klines_5m"
    KLINES_1H = "klines_1h"
    KLINES_1D = "klines_1d"
    OPEN_INTEREST = "open_interest"
    FUNDING_RATE = "funding_rate"
    LONG_SHORT_RATIO = "long_short_ratio"
    TAKER_BUY_SELL = "taker_buy_sell"
    LIQUIDATION = "liquidation"
    ORDER_BOOK = "order_book"
    OPTION_CHAIN = "option_chain"
    DVOL = "dvol"
    STABLECOIN = "stablecoin"
    DEFI_TVL = "defi_tvl"
    FEAR_GREED = "fear_greed"
    GOOGLE_TRENDS = "google_trends"
    DXY = "dxy"
    TREASURY = "treasury"
    SPX = "spx"


class HistoryTier(Enum):
    """历史数据等级"""
    A = "A"  # 多年历史，直接可得
    B = "B"  # 30天+历史
    C = "C"  # 需自建落盘


@dataclass
class FactorMetadata:
    """因子元数据"""
    factor_id: str
    name: str
    family: FactorFamily
    description: str
    data_dependencies: List[DataDependency]
    window: int = 1
    history_tier: HistoryTier = HistoryTier.A
    is_mvp: bool = False
    visibility_delay: timedelta = timedelta(0)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "factor_id": self.factor_id,
            "name": self.name,
            "family": self.family.value,
            "description": self.description,
            "data_dependencies": [d.value for d in self.data_dependencies],
            "window": self.window,
            "history_tier": self.history_tier.value,
            "is_mvp": self.is_mvp,
            "visibility_delay_seconds": self.visibility_delay.total_seconds(),
        }


@dataclass
class FactorResult:
    """因子计算结果"""
    factor_id: str
    value: Union[float, np.ndarray, pd.Series]
    timestamp: datetime
    data_time: datetime
    visible_time: datetime
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        value = self.value
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, pd.Series):
            value = value.to_dict()

        return {
            "factor_id": self.factor_id,
            "value": value,
            "timestamp": self.timestamp.isoformat(),
            "data_time": self.data_time.isoformat(),
            "visible_time": self.visible_time.isoformat(),
            "is_valid": self.is_valid,
            "metadata": self.metadata,
        }


class BaseFactor(ABC):
    """
    因子基类

    所有因子必须继承此类并实现 compute 方法。

    使用示例:
        class ReturnFactor(BaseFactor):
            def get_metadata(self) -> FactorMetadata:
                return FactorMetadata(
                    factor_id="return_1h",
                    name="1小时收益率",
                    family=FactorFamily.MOMENTUM,
                    ...
                )

            def compute(self, data: Dict, signal_time: datetime) -> FactorResult:
                # 计算逻辑
                ...
    """

    @abstractmethod
    def get_metadata(self) -> FactorMetadata:
        """获取因子元数据"""
        pass

    @abstractmethod
    def compute(
        self,
        data: Dict[str, pd.DataFrame],
        signal_time: datetime,
    ) -> FactorResult:
        """
        计算因子值

        Args:
            data: 数据字典 {数据类型: DataFrame}
            signal_time: 信号时间

        Returns:
            FactorResult
        """
        pass

    def validate_data(
        self,
        data: Dict[str, pd.DataFrame],
        required: List[DataDependency],
    ) -> bool:
        """验证数据完整性"""
        for dep in required:
            if dep.value not in data:
                return False
            if data[dep.value] is None or len(data[dep.value]) == 0:
                return False
        return True

    def create_invalid_result(
        self,
        signal_time: datetime,
        reason: str = "",
    ) -> FactorResult:
        """创建无效结果"""
        metadata = self.get_metadata()
        return FactorResult(
            factor_id=metadata.factor_id,
            value=np.nan,
            timestamp=signal_time,
            data_time=signal_time,
            visible_time=signal_time,
            is_valid=False,
            metadata={"reason": reason},
        )

    def create_result(
        self,
        value: Union[float, np.ndarray, pd.Series],
        signal_time: datetime,
        data_time: Optional[datetime] = None,
        metadata: Optional[Dict] = None,
    ) -> FactorResult:
        """创建有效结果"""
        factor_meta = self.get_metadata()
        data_time = data_time or signal_time
        visible_time = data_time + factor_meta.visibility_delay

        return FactorResult(
            factor_id=factor_meta.factor_id,
            value=value,
            timestamp=signal_time,
            data_time=data_time,
            visible_time=visible_time,
            is_valid=True,
            metadata=metadata or {},
        )

    @property
    def factor_id(self) -> str:
        """获取因子ID"""
        return self.get_metadata().factor_id

    @property
    def family(self) -> FactorFamily:
        """获取因子族"""
        return self.get_metadata().family


# 常用工具函数
def ema(values: np.ndarray, period: int) -> np.ndarray:
    """计算指数移动平均"""
    if len(values) == 0:
        return np.array([])
    alpha = 2 / (period + 1)
    result = np.zeros_like(values, dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def sma(values: np.ndarray, period: int) -> np.ndarray:
    """计算简单移动平均"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.convolve(values, np.ones(period) / period, mode='valid')
    return np.concatenate([np.full(period - 1, np.nan), result])


def rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    """计算滚动标准差"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = pd.Series(values).rolling(period).std().values
    return result


def rolling_max(values: np.ndarray, period: int) -> np.ndarray:
    """计算滚动最大值"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = pd.Series(values).rolling(period).max().values
    return result


def rolling_min(values: np.ndarray, period: int) -> np.ndarray:
    """计算滚动最小值"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = pd.Series(values).rolling(period).min().values
    return result


def zscore(values: np.ndarray, period: int) -> np.ndarray:
    """计算 Z-Score"""
    mean = sma(values, period)
    std = rolling_std(values, period)
    return (values - mean) / (std + 1e-10)


def rank(values: np.ndarray) -> np.ndarray:
    """计算排名 (0-1)"""
    return pd.Series(values).rank(pct=True).values


def returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
    """计算收益率"""
    if len(prices) <= periods:
        return np.full(len(prices), np.nan)
    result = np.zeros(len(prices))
    result[:periods] = np.nan
    result[periods:] = prices[periods:] / prices[:-periods] - 1
    return result


def log_returns(prices: np.ndarray, periods: int = 1) -> np.ndarray:
    """计算对数收益率"""
    if len(prices) <= periods:
        return np.full(len(prices), np.nan)
    result = np.zeros(len(prices))
    result[:periods] = np.nan
    result[periods:] = np.log(prices[periods:] / prices[:-periods])
    return result
