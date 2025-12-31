"""
AlgVex 三重屏障风控

设计文档参考: Section 6.2 三重屏障风控 (Triple Barrier)

三重屏障:
1. 止损屏障 (Stop Loss) - 价格触及止损线时平仓
2. 止盈屏障 (Take Profit) - 价格触及止盈线时平仓
3. 时间屏障 (Time Limit) - 持仓时间超过限制时平仓

使用方式:
    config = TripleBarrierConfig(
        stop_loss=0.03,      # 3% 止损
        take_profit=0.06,    # 6% 止盈
        time_limit=86400,    # 24小时
        trailing_stop=0.02,  # 2% 移动止损
    )

    barrier = TripleBarrier(config)
    result = barrier.check(position, current_price, current_time)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class BarrierType(Enum):
    """屏障类型"""
    NONE = "none"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TIME_LIMIT = "time_limit"
    TRAILING_STOP = "trailing_stop"


class TriggerPrice(Enum):
    """触发价格类型"""
    MARK_PRICE = "mark_price"
    LAST_PRICE = "last_price"
    INDEX_PRICE = "index_price"


@dataclass
class TripleBarrierConfig:
    """
    三重屏障配置

    Attributes:
        stop_loss: 止损比例 (0.03 = 3%)
        take_profit: 止盈比例 (0.06 = 6%)
        time_limit: 时间限制 (秒)
        trailing_stop: 移动止损比例 (可选)
        trailing_step: 移动止损步进
        stop_loss_trigger: 止损触发价格类型
        take_profit_trigger: 止盈触发价格类型
        enabled: 是否启用
    """
    # 基本屏障
    stop_loss: Optional[float] = 0.03         # 3% 止损
    take_profit: Optional[float] = 0.06       # 6% 止盈
    time_limit: Optional[int] = 86400         # 24小时 (秒)

    # 移动止损
    trailing_stop: Optional[float] = None     # 移动止损比例
    trailing_step: float = 0.005              # 移动止损步进

    # 触发价格类型
    stop_loss_trigger: TriggerPrice = TriggerPrice.MARK_PRICE
    take_profit_trigger: TriggerPrice = TriggerPrice.LAST_PRICE

    # 开关
    enabled: bool = True

    def __post_init__(self):
        """验证配置"""
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("stop_loss must be positive")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("take_profit must be positive")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if self.trailing_stop is not None and self.trailing_stop <= 0:
            raise ValueError("trailing_stop must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "time_limit": self.time_limit,
            "trailing_stop": self.trailing_stop,
            "trailing_step": self.trailing_step,
            "stop_loss_trigger": self.stop_loss_trigger.value,
            "take_profit_trigger": self.take_profit_trigger.value,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TripleBarrierConfig":
        """从字典创建"""
        if "stop_loss_trigger" in data and isinstance(data["stop_loss_trigger"], str):
            data["stop_loss_trigger"] = TriggerPrice(data["stop_loss_trigger"])
        if "take_profit_trigger" in data and isinstance(data["take_profit_trigger"], str):
            data["take_profit_trigger"] = TriggerPrice(data["take_profit_trigger"])
        return cls(**data)


@dataclass
class BarrierCheckResult:
    """屏障检查结果"""
    triggered: bool
    barrier_type: BarrierType
    trigger_price: Optional[float] = None
    current_price: Optional[float] = None
    pnl_percentage: float = 0.0
    holding_time: Optional[timedelta] = None
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "triggered": self.triggered,
            "barrier_type": self.barrier_type.value,
            "trigger_price": self.trigger_price,
            "current_price": self.current_price,
            "pnl_percentage": self.pnl_percentage,
            "holding_time_seconds": self.holding_time.total_seconds() if self.holding_time else None,
            "message": self.message,
        }


@dataclass
class PositionState:
    """持仓状态 (用于三重屏障检查)"""
    symbol: str
    side: str                    # "long" or "short"
    entry_price: float
    quantity: float
    entry_time: datetime
    highest_price: float = 0.0   # 最高价 (用于移动止损)
    lowest_price: float = 0.0    # 最低价 (用于移动止损)
    trailing_stop_price: Optional[float] = None

    def __post_init__(self):
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price


class TripleBarrier:
    """
    三重屏障风控器

    使用示例:
        config = TripleBarrierConfig(
            stop_loss=0.03,
            take_profit=0.06,
            time_limit=86400,
        )

        barrier = TripleBarrier(config)

        # 每个 bar 检查
        result = barrier.check(
            position=position_state,
            current_price=50000,
            current_time=datetime.now(),
        )

        if result.triggered:
            # 执行平仓
            close_position(position, result.barrier_type)
    """

    def __init__(self, config: TripleBarrierConfig):
        """
        初始化三重屏障

        Args:
            config: 屏障配置
        """
        self.config = config

    def check(
        self,
        position: PositionState,
        current_price: float,
        current_time: datetime,
        price_data: Optional[Dict[str, float]] = None,
    ) -> BarrierCheckResult:
        """
        检查是否触发屏障

        Args:
            position: 持仓状态
            current_price: 当前价格
            current_time: 当前时间
            price_data: 价格数据 (包含 mark_price, last_price 等)

        Returns:
            BarrierCheckResult
        """
        if not self.config.enabled:
            return BarrierCheckResult(
                triggered=False,
                barrier_type=BarrierType.NONE,
                message="Triple barrier disabled",
            )

        # 计算盈亏
        pnl_percentage = self._calculate_pnl_percentage(
            position.entry_price,
            current_price,
            position.side,
        )

        # 计算持仓时间
        holding_time = current_time - position.entry_time

        # 更新最高/最低价
        self._update_extremes(position, current_price)

        # 1. 检查止损
        if self.config.stop_loss is not None:
            sl_result = self._check_stop_loss(position, current_price, pnl_percentage, price_data)
            if sl_result.triggered:
                return sl_result

        # 2. 检查止盈
        if self.config.take_profit is not None:
            tp_result = self._check_take_profit(position, current_price, pnl_percentage, price_data)
            if tp_result.triggered:
                return tp_result

        # 3. 检查移动止损
        if self.config.trailing_stop is not None:
            ts_result = self._check_trailing_stop(position, current_price, pnl_percentage)
            if ts_result.triggered:
                return ts_result

        # 4. 检查时间限制
        if self.config.time_limit is not None:
            tl_result = self._check_time_limit(position, holding_time, current_price, pnl_percentage)
            if tl_result.triggered:
                return tl_result

        # 未触发任何屏障
        return BarrierCheckResult(
            triggered=False,
            barrier_type=BarrierType.NONE,
            current_price=current_price,
            pnl_percentage=pnl_percentage,
            holding_time=holding_time,
        )

    def _calculate_pnl_percentage(
        self,
        entry_price: float,
        current_price: float,
        side: str,
    ) -> float:
        """计算盈亏百分比"""
        if entry_price == 0:
            return 0.0

        if side.lower() == "long":
            return (current_price - entry_price) / entry_price
        else:
            return (entry_price - current_price) / entry_price

    def _update_extremes(self, position: PositionState, current_price: float):
        """更新最高/最低价"""
        if current_price > position.highest_price:
            position.highest_price = current_price

            # 更新移动止损价
            if self.config.trailing_stop is not None and position.side.lower() == "long":
                new_trailing = current_price * (1 - self.config.trailing_stop)
                if position.trailing_stop_price is None or new_trailing > position.trailing_stop_price:
                    position.trailing_stop_price = new_trailing

        if current_price < position.lowest_price:
            position.lowest_price = current_price

            # 更新移动止损价 (空头)
            if self.config.trailing_stop is not None and position.side.lower() == "short":
                new_trailing = current_price * (1 + self.config.trailing_stop)
                if position.trailing_stop_price is None or new_trailing < position.trailing_stop_price:
                    position.trailing_stop_price = new_trailing

    def _check_stop_loss(
        self,
        position: PositionState,
        current_price: float,
        pnl_percentage: float,
        price_data: Optional[Dict[str, float]],
    ) -> BarrierCheckResult:
        """检查止损"""
        # 使用配置的触发价格
        trigger_price = current_price
        if price_data and self.config.stop_loss_trigger == TriggerPrice.MARK_PRICE:
            trigger_price = price_data.get("mark_price", current_price)

        # 计算止损价
        if position.side.lower() == "long":
            stop_price = position.entry_price * (1 - self.config.stop_loss)
            triggered = trigger_price <= stop_price
        else:
            stop_price = position.entry_price * (1 + self.config.stop_loss)
            triggered = trigger_price >= stop_price

        if triggered:
            return BarrierCheckResult(
                triggered=True,
                barrier_type=BarrierType.STOP_LOSS,
                trigger_price=stop_price,
                current_price=current_price,
                pnl_percentage=pnl_percentage,
                message=f"Stop loss triggered at {pnl_percentage:.2%}",
            )

        return BarrierCheckResult(triggered=False, barrier_type=BarrierType.NONE)

    def _check_take_profit(
        self,
        position: PositionState,
        current_price: float,
        pnl_percentage: float,
        price_data: Optional[Dict[str, float]],
    ) -> BarrierCheckResult:
        """检查止盈"""
        # 使用配置的触发价格
        trigger_price = current_price
        if price_data and self.config.take_profit_trigger == TriggerPrice.LAST_PRICE:
            trigger_price = price_data.get("last_price", current_price)

        # 计算止盈价
        if position.side.lower() == "long":
            tp_price = position.entry_price * (1 + self.config.take_profit)
            triggered = trigger_price >= tp_price
        else:
            tp_price = position.entry_price * (1 - self.config.take_profit)
            triggered = trigger_price <= tp_price

        if triggered:
            return BarrierCheckResult(
                triggered=True,
                barrier_type=BarrierType.TAKE_PROFIT,
                trigger_price=tp_price,
                current_price=current_price,
                pnl_percentage=pnl_percentage,
                message=f"Take profit triggered at {pnl_percentage:.2%}",
            )

        return BarrierCheckResult(triggered=False, barrier_type=BarrierType.NONE)

    def _check_trailing_stop(
        self,
        position: PositionState,
        current_price: float,
        pnl_percentage: float,
    ) -> BarrierCheckResult:
        """检查移动止损"""
        if position.trailing_stop_price is None:
            return BarrierCheckResult(triggered=False, barrier_type=BarrierType.NONE)

        if position.side.lower() == "long":
            triggered = current_price <= position.trailing_stop_price
        else:
            triggered = current_price >= position.trailing_stop_price

        if triggered:
            return BarrierCheckResult(
                triggered=True,
                barrier_type=BarrierType.TRAILING_STOP,
                trigger_price=position.trailing_stop_price,
                current_price=current_price,
                pnl_percentage=pnl_percentage,
                message=f"Trailing stop triggered at {current_price:.2f}",
            )

        return BarrierCheckResult(triggered=False, barrier_type=BarrierType.NONE)

    def _check_time_limit(
        self,
        position: PositionState,
        holding_time: timedelta,
        current_price: float,
        pnl_percentage: float,
    ) -> BarrierCheckResult:
        """检查时间限制"""
        time_limit = timedelta(seconds=self.config.time_limit)

        if holding_time >= time_limit:
            return BarrierCheckResult(
                triggered=True,
                barrier_type=BarrierType.TIME_LIMIT,
                current_price=current_price,
                pnl_percentage=pnl_percentage,
                holding_time=holding_time,
                message=f"Time limit reached ({holding_time})",
            )

        return BarrierCheckResult(triggered=False, barrier_type=BarrierType.NONE)

    def calculate_barrier_prices(
        self,
        entry_price: float,
        side: str,
    ) -> Dict[str, Optional[float]]:
        """
        计算屏障价格

        Args:
            entry_price: 入场价格
            side: 持仓方向

        Returns:
            各屏障价格
        """
        result = {
            "stop_loss_price": None,
            "take_profit_price": None,
        }

        if side.lower() == "long":
            if self.config.stop_loss:
                result["stop_loss_price"] = entry_price * (1 - self.config.stop_loss)
            if self.config.take_profit:
                result["take_profit_price"] = entry_price * (1 + self.config.take_profit)
        else:
            if self.config.stop_loss:
                result["stop_loss_price"] = entry_price * (1 + self.config.stop_loss)
            if self.config.take_profit:
                result["take_profit_price"] = entry_price * (1 - self.config.take_profit)

        return result


# 预设配置
def create_conservative_config() -> TripleBarrierConfig:
    """保守配置: 小止损，小止盈"""
    return TripleBarrierConfig(
        stop_loss=0.02,       # 2%
        take_profit=0.04,     # 4%
        time_limit=43200,     # 12小时
        trailing_stop=0.015,  # 1.5%
    )


def create_aggressive_config() -> TripleBarrierConfig:
    """激进配置: 大止损，大止盈"""
    return TripleBarrierConfig(
        stop_loss=0.05,       # 5%
        take_profit=0.15,     # 15%
        time_limit=172800,    # 48小时
        trailing_stop=0.03,   # 3%
    )


def create_scalping_config() -> TripleBarrierConfig:
    """剥头皮配置: 快进快出"""
    return TripleBarrierConfig(
        stop_loss=0.01,       # 1%
        take_profit=0.02,     # 2%
        time_limit=3600,      # 1小时
        trailing_stop=0.005,  # 0.5%
    )
