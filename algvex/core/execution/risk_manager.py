"""
AlgVex 风控管理器
多层风控体系，集成 Hummingbot 企业级风控能力
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskAction(Enum):
    ALLOW = "allow"
    REDUCE_SIZE = "reduce_size"
    BLOCK = "block"
    CLOSE_ALL = "close_all"


@dataclass
class RiskConfig:
    """风控配置"""
    # 仓位限制
    max_position_value: float = 10000.0  # 单仓最大价值
    max_total_exposure: float = 50000.0  # 总敞口上限
    max_positions: int = 10  # 最大持仓数
    max_leverage: int = 10  # 最大杠杆

    # 亏损限制
    max_daily_loss: float = 0.05  # 日最大亏损 5%
    max_weekly_loss: float = 0.10  # 周最大亏损 10%
    max_drawdown: float = 0.15  # 最大回撤 15%

    # 单笔限制
    max_single_trade_risk: float = 0.02  # 单笔最大风险 2%
    min_risk_reward_ratio: float = 1.5  # 最小风险收益比

    # 止损设置
    default_stop_loss: float = 0.03  # 默认止损 3%
    trailing_stop: float = 0.02  # 移动止损 2%

    # 资金费率
    max_funding_rate: float = 0.001  # 最大资金费率 0.1%

    # 流动性
    min_volume_24h: float = 1000000  # 24h最小成交量


@dataclass
class RiskState:
    """风控状态"""
    current_exposure: float = 0.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    positions_count: int = 0
    blocked_symbols: List[str] = field(default_factory=list)
    last_check: datetime = field(default_factory=datetime.now)


class RiskManager:
    """
    风控管理器

    多层风控体系:
    1. 预交易风控 (Pre-Trade)
    2. 交易中风控 (In-Trade)
    3. 事后风控 (Post-Trade)
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.state = RiskState()
        self._alerts: List[Dict] = []

        logger.info("RiskManager initialized")

    def check_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        leverage: int,
    ) -> Dict[str, Any]:
        """
        预交易风控检查

        Args:
            symbol: 交易对
            side: 买卖方向
            quantity: 数量
            price: 价格
            leverage: 杠杆

        Returns:
            检查结果
        """
        checks = []
        action = RiskAction.ALLOW
        adjusted_quantity = quantity

        # 1. 杠杆检查
        if leverage > self.config.max_leverage:
            checks.append({
                "check": "leverage",
                "status": "failed",
                "message": f"Leverage {leverage}x exceeds max {self.config.max_leverage}x",
            })
            action = RiskAction.BLOCK
        else:
            checks.append({"check": "leverage", "status": "passed"})

        # 2. 单仓价值检查
        position_value = quantity * price
        if position_value > self.config.max_position_value:
            adjusted_quantity = self.config.max_position_value / price
            checks.append({
                "check": "position_value",
                "status": "adjusted",
                "message": f"Reduced from {quantity:.4f} to {adjusted_quantity:.4f}",
            })
            action = RiskAction.REDUCE_SIZE
        else:
            checks.append({"check": "position_value", "status": "passed"})

        # 3. 总敞口检查
        new_exposure = self.state.current_exposure + position_value
        if new_exposure > self.config.max_total_exposure:
            checks.append({
                "check": "total_exposure",
                "status": "failed",
                "message": f"Total exposure would exceed {self.config.max_total_exposure}",
            })
            if action != RiskAction.BLOCK:
                action = RiskAction.BLOCK
        else:
            checks.append({"check": "total_exposure", "status": "passed"})

        # 4. 持仓数量检查
        if self.state.positions_count >= self.config.max_positions:
            checks.append({
                "check": "positions_count",
                "status": "failed",
                "message": f"Max positions {self.config.max_positions} reached",
            })
            if action != RiskAction.BLOCK:
                action = RiskAction.BLOCK
        else:
            checks.append({"check": "positions_count", "status": "passed"})

        # 5. 日亏损检查
        if self.state.daily_pnl < -self.config.max_daily_loss:
            checks.append({
                "check": "daily_loss",
                "status": "failed",
                "message": f"Daily loss limit reached: {self.state.daily_pnl:.2%}",
            })
            action = RiskAction.BLOCK
        else:
            checks.append({"check": "daily_loss", "status": "passed"})

        # 6. 黑名单检查
        if symbol in self.state.blocked_symbols:
            checks.append({
                "check": "blocked_symbol",
                "status": "failed",
                "message": f"{symbol} is blocked",
            })
            action = RiskAction.BLOCK
        else:
            checks.append({"check": "blocked_symbol", "status": "passed"})

        result = {
            "allowed": action == RiskAction.ALLOW or action == RiskAction.REDUCE_SIZE,
            "action": action.value,
            "checks": checks,
            "original_quantity": quantity,
            "adjusted_quantity": adjusted_quantity if action == RiskAction.REDUCE_SIZE else quantity,
        }

        if not result["allowed"]:
            self._add_alert("order_blocked", symbol, result)

        return result

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        atr: Optional[float] = None,
    ) -> float:
        """
        计算止损价格

        Args:
            entry_price: 入场价格
            side: 持仓方向
            atr: ATR (可选，用于动态止损)
        """
        if atr:
            # 基于 ATR 的动态止损
            stop_distance = atr * 2
        else:
            # 固定百分比止损
            stop_distance = entry_price * self.config.default_stop_loss

        if side.lower() == "long":
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        side: str,
    ) -> float:
        """
        计算止盈价格 (基于风险收益比)
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * self.config.min_risk_reward_ratio

        if side.lower() == "long":
            return entry_price + reward
        else:
            return entry_price - reward

    def check_funding_rate(
        self,
        symbol: str,
        funding_rate: float,
        side: str,
    ) -> Dict[str, Any]:
        """
        检查资金费率风险

        Args:
            symbol: 交易对
            funding_rate: 当前资金费率
            side: 持仓方向
        """
        # 负费率对多头有利，正费率对空头有利
        is_favorable = (funding_rate < 0 and side == "long") or \
                       (funding_rate > 0 and side == "short")

        is_excessive = abs(funding_rate) > self.config.max_funding_rate

        return {
            "symbol": symbol,
            "funding_rate": funding_rate,
            "is_favorable": is_favorable,
            "is_excessive": is_excessive,
            "recommendation": "hold" if is_favorable else ("close" if is_excessive else "monitor"),
        }

    def update_pnl(self, pnl: float, is_realized: bool = False) -> None:
        """更新 PnL 状态"""
        if is_realized:
            self.state.daily_pnl += pnl

        # 更新回撤
        current_equity = self.state.peak_equity + self.state.daily_pnl
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        else:
            drawdown = (self.state.peak_equity - current_equity) / self.state.peak_equity
            self.state.max_drawdown = max(self.state.max_drawdown, drawdown)

        # 检查是否触发熔断
        if self.state.max_drawdown > self.config.max_drawdown:
            self._add_alert("max_drawdown", "ALL", {
                "drawdown": self.state.max_drawdown,
                "threshold": self.config.max_drawdown,
            })

    def update_exposure(self, symbol: str, exposure: float) -> None:
        """更新敞口"""
        self.state.current_exposure = exposure
        self.state.last_check = datetime.now()

    def update_positions_count(self, count: int) -> None:
        """更新持仓数量"""
        self.state.positions_count = count

    def block_symbol(self, symbol: str, reason: str) -> None:
        """拉黑交易对"""
        if symbol not in self.state.blocked_symbols:
            self.state.blocked_symbols.append(symbol)
            self._add_alert("symbol_blocked", symbol, {"reason": reason})
            logger.warning(f"Symbol blocked: {symbol} - {reason}")

    def unblock_symbol(self, symbol: str) -> None:
        """解除拉黑"""
        if symbol in self.state.blocked_symbols:
            self.state.blocked_symbols.remove(symbol)
            logger.info(f"Symbol unblocked: {symbol}")

    def should_close_all(self) -> bool:
        """是否应该全部平仓 (熔断)"""
        return (
            self.state.daily_pnl < -self.config.max_daily_loss or
            self.state.max_drawdown > self.config.max_drawdown
        )

    def get_risk_level(self) -> RiskLevel:
        """获取当前风险等级"""
        # 基于多个因素评估风险等级
        score = 0

        # 敞口占比
        exposure_ratio = self.state.current_exposure / self.config.max_total_exposure
        if exposure_ratio > 0.9:
            score += 3
        elif exposure_ratio > 0.7:
            score += 2
        elif exposure_ratio > 0.5:
            score += 1

        # 日亏损
        if self.state.daily_pnl < -self.config.max_daily_loss * 0.8:
            score += 3
        elif self.state.daily_pnl < -self.config.max_daily_loss * 0.5:
            score += 2
        elif self.state.daily_pnl < 0:
            score += 1

        # 回撤
        if self.state.max_drawdown > self.config.max_drawdown * 0.8:
            score += 3
        elif self.state.max_drawdown > self.config.max_drawdown * 0.5:
            score += 2

        # 评定等级
        if score >= 6:
            return RiskLevel.CRITICAL
        elif score >= 4:
            return RiskLevel.HIGH
        elif score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def get_alerts(self, clear: bool = False) -> List[Dict]:
        """获取风险告警"""
        alerts = self._alerts.copy()
        if clear:
            self._alerts = []
        return alerts

    def _add_alert(self, alert_type: str, symbol: str, data: Dict) -> None:
        """添加告警"""
        alert = {
            "type": alert_type,
            "symbol": symbol,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "risk_level": self.get_risk_level().value,
        }
        self._alerts.append(alert)
        logger.warning(f"Risk alert: {alert_type} - {symbol}")

    def get_status(self) -> Dict[str, Any]:
        """获取风控状态"""
        return {
            "risk_level": self.get_risk_level().value,
            "current_exposure": self.state.current_exposure,
            "max_exposure": self.config.max_total_exposure,
            "exposure_ratio": self.state.current_exposure / self.config.max_total_exposure,
            "daily_pnl": self.state.daily_pnl,
            "max_daily_loss": self.config.max_daily_loss,
            "max_drawdown": self.state.max_drawdown,
            "positions_count": self.state.positions_count,
            "max_positions": self.config.max_positions,
            "blocked_symbols": self.state.blocked_symbols,
            "should_close_all": self.should_close_all(),
            "alerts_count": len(self._alerts),
        }

    def reset_daily(self) -> None:
        """每日重置 (UTC 0:00)"""
        logger.info(f"Daily reset - Previous PnL: {self.state.daily_pnl:.2%}")
        self.state.daily_pnl = 0.0
        self._alerts = []
