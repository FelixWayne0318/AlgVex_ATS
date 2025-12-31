"""
AlgVex 仓位管理器

根据风险和信号强度计算仓位大小。

支持多种仓位策略:
- Fixed: 固定仓位
- Kelly: Kelly 准则
- Volatility: 基于波动率
- Risk Parity: 风险平价

使用示例:
    config = PositionSizeConfig(
        method="kelly",
        max_position=0.3,  # 最大30%仓位
        kelly_fraction=0.5,  # 半Kelly
    )

    sizer = PositionSizer(config)
    size = sizer.calculate(
        signal_strength=0.8,
        win_rate=0.55,
        avg_win=0.02,
        avg_loss=0.01,
        current_volatility=0.02,
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np


class SizingMethod(Enum):
    """仓位计算方法"""
    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    RISK_PARITY = "risk_parity"
    SIGNAL_SCALED = "signal_scaled"


@dataclass
class PositionSizeConfig:
    """
    仓位配置

    Attributes:
        method: 仓位计算方法
        base_position: 基础仓位 (0-1)
        max_position: 最大仓位 (0-1)
        min_position: 最小仓位 (0-1)
        kelly_fraction: Kelly 分数 (0-1, 推荐 0.25-0.5)
        target_volatility: 目标波动率 (年化)
        risk_per_trade: 每笔交易风险 (占资金比例)
        leverage: 杠杆倍数
    """
    method: SizingMethod = SizingMethod.KELLY
    base_position: float = 0.1
    max_position: float = 0.3
    min_position: float = 0.01
    kelly_fraction: float = 0.5  # Half Kelly
    target_volatility: float = 0.15  # 15% 年化
    risk_per_trade: float = 0.02  # 2% 风险
    leverage: float = 1.0

    def __post_init__(self):
        if isinstance(self.method, str):
            self.method = SizingMethod(self.method)

        if not 0 < self.max_position <= 1:
            raise ValueError("max_position must be between 0 and 1")
        if not 0 <= self.min_position < self.max_position:
            raise ValueError("min_position must be between 0 and max_position")
        if not 0 < self.kelly_fraction <= 1:
            raise ValueError("kelly_fraction must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "method": self.method.value,
            "base_position": self.base_position,
            "max_position": self.max_position,
            "min_position": self.min_position,
            "kelly_fraction": self.kelly_fraction,
            "target_volatility": self.target_volatility,
            "risk_per_trade": self.risk_per_trade,
            "leverage": self.leverage,
        }


@dataclass
class PositionSize:
    """
    仓位计算结果

    Attributes:
        size: 仓位大小 (0-1)
        notional: 名义金额
        quantity: 数量
        method: 使用的方法
        raw_size: 原始计算结果 (未限制)
        metadata: 额外信息
    """
    size: float  # 0-1
    notional: Optional[float] = None
    quantity: Optional[float] = None
    method: str = "kelly"
    raw_size: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """仓位是否有效"""
        return 0 < self.size <= 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "size": self.size,
            "notional": self.notional,
            "quantity": self.quantity,
            "method": self.method,
            "raw_size": self.raw_size,
            "metadata": self.metadata,
        }


class PositionSizer:
    """
    仓位管理器

    使用示例:
        config = PositionSizeConfig(
            method="kelly",
            max_position=0.3,
            kelly_fraction=0.5,
        )

        sizer = PositionSizer(config)

        # 基于信号和统计数据计算仓位
        size = sizer.calculate(
            signal_strength=0.8,
            win_rate=0.55,
            avg_win=0.02,
            avg_loss=0.01,
        )
    """

    def __init__(self, config: PositionSizeConfig):
        """
        初始化仓位管理器

        Args:
            config: 仓位配置
        """
        self.config = config

    def calculate(
        self,
        signal_strength: float = 1.0,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        current_volatility: Optional[float] = None,
        capital: Optional[float] = None,
        price: Optional[float] = None,
    ) -> PositionSize:
        """
        计算仓位大小

        Args:
            signal_strength: 信号强度 (0-1)
            win_rate: 胜率 (0-1)
            avg_win: 平均盈利
            avg_loss: 平均亏损
            current_volatility: 当前波动率
            capital: 可用资金
            price: 当前价格

        Returns:
            仓位计算结果
        """
        method = self.config.method

        if method == SizingMethod.FIXED:
            raw_size = self._calculate_fixed(signal_strength)
        elif method == SizingMethod.KELLY:
            raw_size = self._calculate_kelly(
                signal_strength, win_rate, avg_win, avg_loss
            )
        elif method == SizingMethod.VOLATILITY:
            raw_size = self._calculate_volatility(
                signal_strength, current_volatility
            )
        elif method == SizingMethod.RISK_PARITY:
            raw_size = self._calculate_risk_parity(
                signal_strength, current_volatility
            )
        elif method == SizingMethod.SIGNAL_SCALED:
            raw_size = self._calculate_signal_scaled(signal_strength)
        else:
            raw_size = self.config.base_position

        # 应用限制
        size = self._apply_limits(raw_size)

        # 计算名义金额和数量
        notional = None
        quantity = None
        if capital is not None:
            notional = capital * size * self.config.leverage
            if price is not None and price > 0:
                quantity = notional / price

        return PositionSize(
            size=size,
            notional=notional,
            quantity=quantity,
            method=method.value,
            raw_size=raw_size,
            metadata={
                "signal_strength": signal_strength,
                "win_rate": win_rate,
                "current_volatility": current_volatility,
            },
        )

    def _calculate_fixed(self, signal_strength: float) -> float:
        """固定仓位"""
        return self.config.base_position * abs(signal_strength)

    def _calculate_kelly(
        self,
        signal_strength: float,
        win_rate: Optional[float],
        avg_win: Optional[float],
        avg_loss: Optional[float],
    ) -> float:
        """
        Kelly 准则

        Kelly% = W - (1-W)/R
        其中 W = 胜率, R = 盈亏比
        """
        if win_rate is None or avg_win is None or avg_loss is None:
            # 使用默认值
            win_rate = 0.5
            avg_win = 0.02
            avg_loss = 0.01

        if avg_loss <= 0:
            return self.config.base_position

        # 盈亏比
        win_loss_ratio = avg_win / abs(avg_loss)

        # Kelly 公式
        kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # 应用 Kelly 分数 (半 Kelly 等)
        kelly = kelly * self.config.kelly_fraction

        # 考虑信号强度
        kelly = kelly * abs(signal_strength)

        return max(0, kelly)

    def _calculate_volatility(
        self,
        signal_strength: float,
        current_volatility: Optional[float],
    ) -> float:
        """
        基于波动率的仓位

        目标波动率 / 当前波动率
        """
        if current_volatility is None or current_volatility <= 0:
            return self.config.base_position

        # 目标波动率调整
        vol_ratio = self.config.target_volatility / current_volatility

        # 基础仓位 * 波动率比 * 信号强度
        size = self.config.base_position * vol_ratio * abs(signal_strength)

        return size

    def _calculate_risk_parity(
        self,
        signal_strength: float,
        current_volatility: Optional[float],
    ) -> float:
        """
        风险平价

        仓位 = 风险预算 / (波动率 * 杠杆)
        """
        if current_volatility is None or current_volatility <= 0:
            return self.config.base_position

        # 每笔交易的风险预算
        risk_budget = self.config.risk_per_trade

        # 仓位 = 风险预算 / 波动率
        size = risk_budget / current_volatility

        # 考虑信号强度
        size = size * abs(signal_strength)

        return size

    def _calculate_signal_scaled(self, signal_strength: float) -> float:
        """
        信号强度线性缩放

        仓位 = base * signal_strength
        """
        return self.config.base_position * abs(signal_strength)

    def _apply_limits(self, size: float) -> float:
        """应用仓位限制"""
        if np.isnan(size) or np.isinf(size):
            size = self.config.min_position

        size = max(self.config.min_position, size)
        size = min(self.config.max_position, size)

        return size

    def calculate_quantity(
        self,
        capital: float,
        price: float,
        position_size: float,
    ) -> float:
        """
        计算交易数量

        Args:
            capital: 可用资金
            price: 当前价格
            position_size: 仓位比例 (0-1)

        Returns:
            交易数量
        """
        if price <= 0:
            raise ValueError("Price must be positive")

        notional = capital * position_size * self.config.leverage
        return notional / price


def create_conservative_sizer() -> PositionSizer:
    """创建保守型仓位管理器"""
    config = PositionSizeConfig(
        method=SizingMethod.KELLY,
        base_position=0.05,
        max_position=0.15,
        kelly_fraction=0.25,  # Quarter Kelly
        risk_per_trade=0.01,
    )
    return PositionSizer(config)


def create_moderate_sizer() -> PositionSizer:
    """创建中等型仓位管理器"""
    config = PositionSizeConfig(
        method=SizingMethod.KELLY,
        base_position=0.1,
        max_position=0.3,
        kelly_fraction=0.5,  # Half Kelly
        risk_per_trade=0.02,
    )
    return PositionSizer(config)


def create_aggressive_sizer() -> PositionSizer:
    """创建激进型仓位管理器"""
    config = PositionSizeConfig(
        method=SizingMethod.KELLY,
        base_position=0.2,
        max_position=0.5,
        kelly_fraction=0.75,  # 3/4 Kelly
        risk_per_trade=0.03,
    )
    return PositionSizer(config)
