"""
AlgVex 风险管理器

提供实时风险监控和控制:
- 仓位限制
- 回撤控制
- 风险敞口管理
- 相关性风险

使用示例:
    config = RiskConfig(
        max_drawdown=0.15,
        max_position=0.3,
        max_daily_loss=0.05,
    )

    risk_manager = RiskManager(config)

    # 检查是否允许交易
    can_trade, reason = risk_manager.check_trade(
        current_drawdown=0.08,
        current_position=0.2,
        daily_pnl=-0.02,
    )
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    """风险应对动作"""
    ALLOW = "allow"  # 允许交易
    REDUCE = "reduce"  # 减少仓位
    HEDGE = "hedge"  # 对冲
    STOP = "stop"  # 停止交易
    CLOSE_ALL = "close_all"  # 平掉所有仓位


@dataclass
class RiskConfig:
    """
    风险管理配置

    Attributes:
        max_drawdown: 最大回撤限制
        max_daily_loss: 最大日内亏损
        max_position: 最大仓位
        max_leverage: 最大杠杆
        max_concentration: 单标的最大集中度
        correlation_limit: 相关性限制
        var_limit: VaR 限制 (95%)
        recovery_period: 恢复期 (小时)
    """
    max_drawdown: float = 0.15  # 15%
    max_daily_loss: float = 0.05  # 5%
    max_position: float = 0.5  # 50%
    max_leverage: float = 5.0
    max_concentration: float = 0.3  # 单标的30%
    correlation_limit: float = 0.8
    var_limit: float = 0.02  # 2% 日度 VaR
    recovery_period: int = 24  # 24小时恢复期

    def __post_init__(self):
        if not 0 < self.max_drawdown <= 1:
            raise ValueError("max_drawdown must be between 0 and 1")
        if not 0 < self.max_daily_loss <= 1:
            raise ValueError("max_daily_loss must be between 0 and 1")
        if not 0 < self.max_position <= 1:
            raise ValueError("max_position must be between 0 and 1")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "max_drawdown": self.max_drawdown,
            "max_daily_loss": self.max_daily_loss,
            "max_position": self.max_position,
            "max_leverage": self.max_leverage,
            "max_concentration": self.max_concentration,
            "correlation_limit": self.correlation_limit,
            "var_limit": self.var_limit,
            "recovery_period": self.recovery_period,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RiskConfig":
        """从字典创建"""
        return cls(**data)


@dataclass
class RiskMetrics:
    """
    风险指标

    Attributes:
        current_drawdown: 当前回撤
        daily_pnl: 日内盈亏
        total_position: 总仓位
        leverage: 当前杠杆
        var_95: 95% VaR
        risk_level: 风险等级
        action: 建议动作
        timestamp: 计算时间
    """
    current_drawdown: float = 0.0
    daily_pnl: float = 0.0
    total_position: float = 0.0
    leverage: float = 1.0
    var_95: Optional[float] = None
    risk_level: RiskLevel = RiskLevel.LOW
    action: RiskAction = RiskAction.ALLOW
    timestamp: datetime = field(default_factory=datetime.now)
    messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "current_drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "total_position": self.total_position,
            "leverage": self.leverage,
            "var_95": self.var_95,
            "risk_level": self.risk_level.value,
            "action": self.action.value,
            "timestamp": self.timestamp.isoformat(),
            "messages": self.messages,
        }


@dataclass
class RiskEvent:
    """风险事件"""
    event_type: str
    risk_level: RiskLevel
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)


class RiskManager:
    """
    风险管理器

    使用示例:
        config = RiskConfig(
            max_drawdown=0.15,
            max_position=0.3,
        )

        manager = RiskManager(config)

        # 检查交易
        can_trade, reason = manager.check_trade(
            current_drawdown=0.08,
            current_position=0.2,
        )

        # 获取风险指标
        metrics = manager.calculate_metrics(
            equity_curve=equity_curve,
            positions=positions,
        )
    """

    def __init__(self, config: RiskConfig = None):
        """
        初始化风险管理器

        Args:
            config: 风险配置 (可选，默认使用 RiskConfig())
        """
        self.config = config or RiskConfig()
        self._risk_events: List[RiskEvent] = []
        self._stop_trading_until: Optional[datetime] = None
        self._daily_loss: float = 0.0
        self._last_reset_date: Optional[datetime] = None

    def check_trade(
        self,
        current_drawdown: float = 0.0,
        current_position: float = 0.0,
        daily_pnl: float = 0.0,
        proposed_position: float = 0.0,
        leverage: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        检查是否允许交易

        Args:
            current_drawdown: 当前回撤
            current_position: 当前仓位
            daily_pnl: 日内盈亏
            proposed_position: 建议仓位变化
            leverage: 当前杠杆

        Returns:
            (是否允许, 原因)
        """
        # 检查是否在停止交易期间
        if self._stop_trading_until is not None:
            if datetime.now() < self._stop_trading_until:
                remaining = (self._stop_trading_until - datetime.now()).seconds // 3600
                return False, f"Trading stopped for {remaining} more hours"
            else:
                self._stop_trading_until = None

        # 检查回撤
        if current_drawdown >= self.config.max_drawdown:
            self._trigger_stop(RiskLevel.CRITICAL, "Max drawdown exceeded")
            return False, f"Max drawdown exceeded: {current_drawdown:.2%}"

        # 检查日内亏损
        if daily_pnl <= -self.config.max_daily_loss:
            self._trigger_stop(RiskLevel.HIGH, "Daily loss limit exceeded")
            return False, f"Daily loss limit exceeded: {daily_pnl:.2%}"

        # 检查仓位
        new_position = current_position + proposed_position
        if new_position > self.config.max_position:
            return False, f"Position limit exceeded: {new_position:.2%}"

        # 检查杠杆
        if leverage > self.config.max_leverage:
            return False, f"Leverage limit exceeded: {leverage:.1f}x"

        # 风险警告
        if current_drawdown >= self.config.max_drawdown * 0.7:
            self._add_event(
                RiskLevel.MEDIUM,
                f"Drawdown warning: {current_drawdown:.2%}",
            )

        return True, "Trade allowed"

    def calculate_metrics(
        self,
        equity_curve: Optional[np.ndarray] = None,
        positions: Optional[Dict[str, float]] = None,
        returns: Optional[np.ndarray] = None,
    ) -> RiskMetrics:
        """
        计算风险指标

        Args:
            equity_curve: 权益曲线
            positions: 当前持仓 {symbol: position_size}
            returns: 收益率序列

        Returns:
            风险指标
        """
        metrics = RiskMetrics()
        messages = []

        # 计算回撤
        if equity_curve is not None and len(equity_curve) > 0:
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            metrics.current_drawdown = float(drawdown[-1])

        # 计算总仓位
        if positions is not None:
            metrics.total_position = sum(abs(p) for p in positions.values())

        # 计算 VaR
        if returns is not None and len(returns) > 20:
            metrics.var_95 = float(np.percentile(returns, 5))

        # 确定风险等级
        risk_level = RiskLevel.LOW

        if metrics.current_drawdown >= self.config.max_drawdown:
            risk_level = RiskLevel.CRITICAL
            messages.append("CRITICAL: Max drawdown exceeded")
        elif metrics.current_drawdown >= self.config.max_drawdown * 0.7:
            risk_level = RiskLevel.HIGH
            messages.append("HIGH: Approaching max drawdown")
        elif metrics.current_drawdown >= self.config.max_drawdown * 0.5:
            risk_level = RiskLevel.MEDIUM
            messages.append("MEDIUM: Drawdown warning")

        if metrics.total_position > self.config.max_position:
            risk_level = max(risk_level, RiskLevel.HIGH, key=lambda x: x.value)
            messages.append("HIGH: Position limit exceeded")

        metrics.risk_level = risk_level
        metrics.messages = messages

        # 确定建议动作
        if risk_level == RiskLevel.CRITICAL:
            metrics.action = RiskAction.CLOSE_ALL
        elif risk_level == RiskLevel.HIGH:
            metrics.action = RiskAction.REDUCE
        else:
            metrics.action = RiskAction.ALLOW

        return metrics

    def get_position_adjustment(
        self,
        current_position: float,
        risk_level: RiskLevel,
    ) -> float:
        """
        获取仓位调整建议

        Args:
            current_position: 当前仓位
            risk_level: 风险等级

        Returns:
            建议的仓位调整 (负数表示减仓)
        """
        if risk_level == RiskLevel.CRITICAL:
            return -current_position  # 全平

        if risk_level == RiskLevel.HIGH:
            return -current_position * 0.5  # 减半

        if risk_level == RiskLevel.MEDIUM:
            return -current_position * 0.25  # 减25%

        return 0.0

    def check_concentration(
        self,
        positions: Dict[str, float],
    ) -> Tuple[bool, List[str]]:
        """
        检查仓位集中度

        Args:
            positions: 持仓 {symbol: position_size}

        Returns:
            (是否合规, 超限标的列表)
        """
        total = sum(abs(p) for p in positions.values())
        if total == 0:
            return True, []

        over_limit = []
        for symbol, position in positions.items():
            concentration = abs(position) / total
            if concentration > self.config.max_concentration:
                over_limit.append(symbol)

        return len(over_limit) == 0, over_limit

    def _trigger_stop(self, level: RiskLevel, message: str):
        """触发停止交易"""
        self._stop_trading_until = datetime.now() + timedelta(
            hours=self.config.recovery_period
        )
        self._add_event(level, message)

    def _add_event(self, level: RiskLevel, message: str):
        """添加风险事件"""
        event = RiskEvent(
            event_type="risk_warning" if level != RiskLevel.CRITICAL else "risk_breach",
            risk_level=level,
            message=message,
            timestamp=datetime.now(),
        )
        self._risk_events.append(event)

    def get_recent_events(self, hours: int = 24) -> List[RiskEvent]:
        """获取最近的风险事件"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self._risk_events if e.timestamp >= cutoff]

    def reset_daily(self):
        """重置日内计数器"""
        self._daily_loss = 0.0
        self._last_reset_date = datetime.now()


def create_conservative_risk_manager() -> RiskManager:
    """创建保守型风险管理器"""
    config = RiskConfig(
        max_drawdown=0.10,
        max_daily_loss=0.03,
        max_position=0.3,
        max_leverage=2.0,
        max_concentration=0.2,
    )
    return RiskManager(config)


def create_moderate_risk_manager() -> RiskManager:
    """创建中等型风险管理器"""
    config = RiskConfig(
        max_drawdown=0.15,
        max_daily_loss=0.05,
        max_position=0.5,
        max_leverage=5.0,
        max_concentration=0.3,
    )
    return RiskManager(config)


def create_aggressive_risk_manager() -> RiskManager:
    """创建激进型风险管理器"""
    config = RiskConfig(
        max_drawdown=0.25,
        max_daily_loss=0.08,
        max_position=0.7,
        max_leverage=10.0,
        max_concentration=0.4,
    )
    return RiskManager(config)
