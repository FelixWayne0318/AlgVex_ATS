"""
AlgVex Controller

将 AlgVex 信号生成集成到 Hummingbot Strategy V2 框架

数据流:
SignalGenerator → AlgVexController → PositionExecutor → Connector
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """仓位方向"""
    LONG = "LONG"
    SHORT = "SHORT"


class ActionType(Enum):
    """动作类型"""
    CREATE_POSITION = "create_position"
    CLOSE_POSITION = "close_position"
    ADJUST_POSITION = "adjust_position"
    NO_ACTION = "no_action"


@dataclass
class AlgVexControllerConfig:
    """
    AlgVex Controller 配置

    配置参数控制信号到订单的转换逻辑
    """
    # 交易对列表
    trading_pairs: Set[str] = field(default_factory=set)

    # 信号阈值（绝对值超过此值才会开仓）
    signal_threshold: float = 0.5

    # 每个交易对最大仓位（以基础货币计）
    max_position_per_pair: Decimal = Decimal("0.1")

    # 杠杆倍数
    leverage: int = 1

    # 止损百分比
    stop_loss_pct: Decimal = Decimal("0.03")  # 3%

    # 止盈百分比
    take_profit_pct: Decimal = Decimal("0.06")  # 6%

    # 滑点容忍度
    slippage_tolerance: Decimal = Decimal("0.001")  # 0.1%

    # 最小仓位大小
    min_position_size: Decimal = Decimal("0.001")

    # 冷却时间（秒）- 同一交易对连续信号间隔
    cooldown_seconds: int = 60

    # 是否启用追踪止损
    trailing_stop: bool = False

    # 追踪止损偏移
    trailing_stop_offset: Decimal = Decimal("0.02")  # 2%

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "trading_pairs": list(self.trading_pairs),
            "signal_threshold": self.signal_threshold,
            "max_position_per_pair": str(self.max_position_per_pair),
            "leverage": self.leverage,
            "stop_loss_pct": str(self.stop_loss_pct),
            "take_profit_pct": str(self.take_profit_pct),
            "slippage_tolerance": str(self.slippage_tolerance),
            "min_position_size": str(self.min_position_size),
            "cooldown_seconds": self.cooldown_seconds,
            "trailing_stop": self.trailing_stop,
            "trailing_stop_offset": str(self.trailing_stop_offset),
        }


@dataclass
class PositionConfig:
    """仓位配置"""
    trading_pair: str
    side: PositionSide
    amount: Decimal
    leverage: int = 1
    stop_loss: Optional[Decimal] = None  # 止损价
    take_profit: Optional[Decimal] = None  # 止盈价
    entry_price: Optional[Decimal] = None
    trailing_stop: bool = False
    trailing_stop_offset: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "trading_pair": self.trading_pair,
            "side": self.side.value,
            "amount": str(self.amount),
            "leverage": self.leverage,
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit": str(self.take_profit) if self.take_profit else None,
            "entry_price": str(self.entry_price) if self.entry_price else None,
            "trailing_stop": self.trailing_stop,
            "trailing_stop_offset": str(self.trailing_stop_offset) if self.trailing_stop_offset else None,
        }


@dataclass
class ControllerAction:
    """控制器动作"""
    action_type: ActionType
    trading_pair: str
    position_config: Optional[PositionConfig] = None
    signal: Optional[Any] = None
    reason: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "action_type": self.action_type.value,
            "trading_pair": self.trading_pair,
            "position_config": self.position_config.to_dict() if self.position_config else None,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class SignalGeneratorInterface(ABC):
    """信号生成器接口"""

    @abstractmethod
    async def get_signal(self, trading_pair: str) -> Optional[Any]:
        """获取交易对信号"""
        pass


@dataclass
class Signal:
    """信号数据结构"""
    signal_id: str
    trading_pair: str
    final_signal: float  # -1 到 1 之间
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    factors: Dict[str, float] = field(default_factory=dict)


class AlgVexController:
    """
    AlgVex 信号控制器

    将 AlgVex SignalGenerator 集成到 Hummingbot V2 架构

    核心职责:
    1. 定期获取 AlgVex 信号
    2. 根据信号决定仓位操作
    3. 生成 PositionConfig 供执行器使用

    设计原则:
    - 信号强度决定仓位大小
    - 支持止损止盈
    - 支持冷却时间防止过度交易
    - 与 Hummingbot PositionExecutor 解耦
    """

    def __init__(
        self,
        config: AlgVexControllerConfig,
        signal_generator: Optional[SignalGeneratorInterface] = None,
    ):
        """
        Args:
            config: 控制器配置
            signal_generator: 信号生成器实例
        """
        self.config = config
        self.signal_generator = signal_generator

        # 最新信号缓存
        self._latest_signals: Dict[str, Signal] = {}

        # 当前仓位
        self._current_positions: Dict[str, PositionConfig] = {}

        # 最后信号时间（用于冷却）
        self._last_signal_time: Dict[str, float] = {}

        # 统计
        self._action_history: List[ControllerAction] = []
        self._max_history = 1000

        logger.info(f"AlgVexController initialized with {len(config.trading_pairs)} pairs")

    def set_signal_generator(self, generator: SignalGeneratorInterface):
        """设置信号生成器"""
        self.signal_generator = generator

    async def update_processed_data(self):
        """
        更新处理后的数据

        每个 tick 调用，获取最新的 AlgVex 信号

        对应 Hummingbot ControllerBase.update_processed_data
        """
        if not self.signal_generator:
            logger.warning("No signal generator configured")
            return

        for trading_pair in self.config.trading_pairs:
            try:
                signal = await self.signal_generator.get_signal(trading_pair)
                if signal:
                    self._latest_signals[trading_pair] = signal
                    logger.debug(f"Updated signal for {trading_pair}: {signal.final_signal}")
            except Exception as e:
                logger.warning(f"Failed to get signal for {trading_pair}: {e}")

    def determine_executor_actions(self) -> List[ControllerAction]:
        """
        确定执行器动作

        基于 AlgVex 信号决定是否开仓/平仓

        对应 Hummingbot ControllerBase.determine_executor_actions

        Returns:
            List[ControllerAction]: 需要执行的动作列表
        """
        actions = []
        current_time = time.time()

        for trading_pair in self.config.trading_pairs:
            signal = self._latest_signals.get(trading_pair)
            current_pos = self._current_positions.get(trading_pair)

            # 检查冷却时间
            last_time = self._last_signal_time.get(trading_pair, 0)
            if current_time - last_time < self.config.cooldown_seconds:
                continue

            action = self._determine_action_for_pair(trading_pair, signal, current_pos)

            if action.action_type != ActionType.NO_ACTION:
                actions.append(action)
                self._last_signal_time[trading_pair] = current_time

                # 记录历史
                self._action_history.append(action)
                if len(self._action_history) > self._max_history:
                    self._action_history = self._action_history[-self._max_history:]

        return actions

    def _determine_action_for_pair(
        self,
        trading_pair: str,
        signal: Optional[Signal],
        current_pos: Optional[PositionConfig],
    ) -> ControllerAction:
        """确定单个交易对的动作"""

        # 无信号
        if signal is None:
            return ControllerAction(
                action_type=ActionType.NO_ACTION,
                trading_pair=trading_pair,
                reason="no_signal",
            )

        signal_value = signal.final_signal

        # 信号强度不足
        if abs(signal_value) < self.config.signal_threshold:
            # 如果有仓位且信号反转，考虑平仓
            if current_pos:
                if (current_pos.side == PositionSide.LONG and signal_value < 0) or \
                   (current_pos.side == PositionSide.SHORT and signal_value > 0):
                    return ControllerAction(
                        action_type=ActionType.CLOSE_POSITION,
                        trading_pair=trading_pair,
                        signal=signal,
                        reason="signal_reversed_below_threshold",
                    )

            return ControllerAction(
                action_type=ActionType.NO_ACTION,
                trading_pair=trading_pair,
                reason="signal_below_threshold",
            )

        # 确定方向
        target_side = PositionSide.LONG if signal_value > 0 else PositionSide.SHORT

        # 已有仓位
        if current_pos:
            if current_pos.side == target_side:
                # 同方向，可以加仓或保持
                return ControllerAction(
                    action_type=ActionType.NO_ACTION,
                    trading_pair=trading_pair,
                    reason="already_positioned_same_side",
                )
            else:
                # 反方向，先平仓
                return ControllerAction(
                    action_type=ActionType.CLOSE_POSITION,
                    trading_pair=trading_pair,
                    signal=signal,
                    reason="close_for_reversal",
                )

        # 无仓位，创建新仓位
        position_size = self._calculate_position_size(signal)

        if position_size < self.config.min_position_size:
            return ControllerAction(
                action_type=ActionType.NO_ACTION,
                trading_pair=trading_pair,
                reason="position_size_too_small",
            )

        # 计算止损止盈价格（需要当前价格，这里先用占位符）
        position_config = PositionConfig(
            trading_pair=trading_pair,
            side=target_side,
            amount=position_size,
            leverage=self.config.leverage,
            trailing_stop=self.config.trailing_stop,
            trailing_stop_offset=self.config.trailing_stop_offset if self.config.trailing_stop else None,
        )

        return ControllerAction(
            action_type=ActionType.CREATE_POSITION,
            trading_pair=trading_pair,
            position_config=position_config,
            signal=signal,
            reason=f"signal_{target_side.value.lower()}",
        )

    def _calculate_position_size(self, signal: Signal) -> Decimal:
        """
        计算仓位大小

        基于信号强度和配置的最大仓位
        """
        base_size = self.config.max_position_per_pair
        signal_weight = Decimal(str(abs(signal.final_signal)))
        confidence_weight = Decimal(str(signal.confidence))

        # 仓位 = 最大仓位 * 信号强度 * 置信度
        size = base_size * signal_weight * confidence_weight

        return max(size, Decimal("0"))

    def update_position(self, trading_pair: str, position: Optional[PositionConfig]):
        """
        更新本地仓位记录

        由执行器调用，同步实际仓位状态
        """
        if position:
            self._current_positions[trading_pair] = position
            logger.debug(f"Position updated for {trading_pair}: {position.side.value} {position.amount}")
        elif trading_pair in self._current_positions:
            del self._current_positions[trading_pair]
            logger.debug(f"Position closed for {trading_pair}")

    def get_current_position(self, trading_pair: str) -> Optional[PositionConfig]:
        """获取当前仓位"""
        return self._current_positions.get(trading_pair)

    def get_all_positions(self) -> Dict[str, PositionConfig]:
        """获取所有仓位"""
        return dict(self._current_positions)

    def get_latest_signal(self, trading_pair: str) -> Optional[Signal]:
        """获取最新信号"""
        return self._latest_signals.get(trading_pair)

    def get_action_history(self, limit: int = 50) -> List[Dict]:
        """获取动作历史"""
        return [a.to_dict() for a in self._action_history[-limit:]]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        action_counts = {}
        for action in self._action_history:
            action_type = action.action_type.value
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

        return {
            "config": self.config.to_dict(),
            "active_pairs": len(self.config.trading_pairs),
            "current_positions": len(self._current_positions),
            "signals_cached": len(self._latest_signals),
            "action_counts": action_counts,
            "total_actions": len(self._action_history),
        }


class MockSignalGenerator(SignalGeneratorInterface):
    """
    模拟信号生成器

    用于测试和 Paper Trading
    """

    def __init__(self, signals: Optional[Dict[str, float]] = None):
        """
        Args:
            signals: 预设信号 {trading_pair: signal_value}
        """
        self._signals = signals or {}
        self._signal_counter = 0

    def set_signal(self, trading_pair: str, value: float):
        """设置信号值"""
        self._signals[trading_pair] = value

    async def get_signal(self, trading_pair: str) -> Optional[Signal]:
        """获取信号"""
        value = self._signals.get(trading_pair)
        if value is None:
            return None

        self._signal_counter += 1

        return Signal(
            signal_id=f"mock_{self._signal_counter}",
            trading_pair=trading_pair,
            final_signal=value,
            confidence=1.0,
        )
