"""
风险管理模块

风控规则:
1. 最大持仓限制
2. 最大杠杆限制
3. 止损/止盈
4. 每日最大亏损
5. 频率限制
6. 资金费率预警
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from .adapter import TradeInstruction, TradeAction


@dataclass
class RiskConfig:
    """风控配置"""
    # 仓位限制
    max_position_per_symbol: float = 0.2     # 单标的最大仓位占比
    max_total_position: float = 0.8          # 总仓位最大占比
    max_leverage: float = 5.0                # 最大杠杆

    # 止损止盈
    stop_loss_pct: float = 0.05              # 止损比例 5%
    take_profit_pct: float = 0.1             # 止盈比例 10%

    # 亏损限制
    max_daily_loss_pct: float = 0.05         # 每日最大亏损 5%
    max_drawdown_pct: float = 0.15           # 最大回撤 15%

    # 频率限制
    min_trade_interval_seconds: int = 60     # 最小交易间隔(秒)
    max_trades_per_hour: int = 20            # 每小时最大交易次数

    # 资金费率预警
    funding_rate_warning: float = 0.001      # 资金费率预警阈值 0.1%


class RiskManager:
    """风险管理器"""

    def __init__(self, config: RiskConfig = None, capital: float = 100000.0):
        self.config = config or RiskConfig()
        self.capital = capital

        # 状态追踪
        self.positions: Dict[str, float] = {}           # 当前持仓
        self.position_entry_prices: Dict[str, float] = {}  # 开仓价
        self.daily_pnl: float = 0.0                     # 当日盈亏
        self.peak_equity: float = capital               # 峰值权益
        self.current_equity: float = capital            # 当前权益

        # 交易记录
        self.trade_history: List[Dict] = []
        self.last_trade_time: Dict[str, datetime] = {}

    def update_equity(self, equity: float):
        """更新权益"""
        self.current_equity = equity
        self.peak_equity = max(self.peak_equity, equity)

    def update_daily_pnl(self, pnl: float):
        """更新当日盈亏"""
        self.daily_pnl += pnl

    def reset_daily_stats(self):
        """重置每日统计"""
        self.daily_pnl = 0.0

    def check_trade(self, instruction: TradeInstruction, current_price: float) -> tuple:
        """
        检查交易是否符合风控规则

        Args:
            instruction: 交易指令
            current_price: 当前价格

        Returns:
            (通过, 拒绝原因)
        """
        symbol = instruction.symbol

        # 1. 检查杠杆
        if instruction.leverage > self.config.max_leverage:
            return False, f"杠杆超限: {instruction.leverage} > {self.config.max_leverage}"

        # 2. 检查单标的仓位
        trade_value = instruction.quantity * current_price
        current_pos_value = abs(self.positions.get(symbol, 0)) * current_price
        new_pos_value = current_pos_value + trade_value

        if new_pos_value > self.capital * self.config.max_position_per_symbol:
            return False, f"单标的仓位超限: {new_pos_value:.0f} > {self.capital * self.config.max_position_per_symbol:.0f}"

        # 3. 检查总仓位
        total_pos_value = sum(
            abs(pos) * current_price for pos in self.positions.values()
        ) + trade_value

        if total_pos_value > self.capital * self.config.max_total_position:
            return False, f"总仓位超限: {total_pos_value:.0f} > {self.capital * self.config.max_total_position:.0f}"

        # 4. 检查每日亏损
        if self.daily_pnl < -self.capital * self.config.max_daily_loss_pct:
            return False, f"每日亏损超限: {self.daily_pnl:.0f}"

        # 5. 检查最大回撤
        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        if current_drawdown > self.config.max_drawdown_pct:
            return False, f"最大回撤超限: {current_drawdown:.2%}"

        # 6. 检查交易频率
        now = datetime.now()
        if symbol in self.last_trade_time:
            time_since_last = (now - self.last_trade_time[symbol]).total_seconds()
            if time_since_last < self.config.min_trade_interval_seconds:
                return False, f"交易过于频繁: {time_since_last:.0f}s < {self.config.min_trade_interval_seconds}s"

        # 7. 检查每小时交易次数
        hour_ago = now - timedelta(hours=1)
        recent_trades = [t for t in self.trade_history if t["time"] > hour_ago]
        if len(recent_trades) >= self.config.max_trades_per_hour:
            return False, f"每小时交易次数超限: {len(recent_trades)} >= {self.config.max_trades_per_hour}"

        return True, ""

    def calculate_stop_loss(self, symbol: str, entry_price: float, is_long: bool) -> float:
        """计算止损价"""
        if is_long:
            return entry_price * (1 - self.config.stop_loss_pct)
        else:
            return entry_price * (1 + self.config.stop_loss_pct)

    def calculate_take_profit(self, symbol: str, entry_price: float, is_long: bool) -> float:
        """计算止盈价"""
        if is_long:
            return entry_price * (1 + self.config.take_profit_pct)
        else:
            return entry_price * (1 - self.config.take_profit_pct)

    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """检查是否触发止损"""
        if symbol not in self.positions or symbol not in self.position_entry_prices:
            return False

        pos = self.positions[symbol]
        entry_price = self.position_entry_prices[symbol]

        if pos > 0:  # 多头
            stop_price = self.calculate_stop_loss(symbol, entry_price, is_long=True)
            return current_price <= stop_price
        elif pos < 0:  # 空头
            stop_price = self.calculate_stop_loss(symbol, entry_price, is_long=False)
            return current_price >= stop_price

        return False

    def check_take_profit(self, symbol: str, current_price: float) -> bool:
        """检查是否触发止盈"""
        if symbol not in self.positions or symbol not in self.position_entry_prices:
            return False

        pos = self.positions[symbol]
        entry_price = self.position_entry_prices[symbol]

        if pos > 0:  # 多头
            tp_price = self.calculate_take_profit(symbol, entry_price, is_long=True)
            return current_price >= tp_price
        elif pos < 0:  # 空头
            tp_price = self.calculate_take_profit(symbol, entry_price, is_long=False)
            return current_price <= tp_price

        return False

    def check_funding_rate_risk(self, symbol: str, funding_rate: float, position: float) -> Optional[str]:
        """
        检查资金费率风险

        Args:
            symbol: 交易对
            funding_rate: 当前资金费率
            position: 当前持仓 (正=多, 负=空)

        Returns:
            风险警告信息 (None表示无风险)
        """
        if abs(funding_rate) < self.config.funding_rate_warning:
            return None

        if funding_rate > 0 and position > 0:
            return f"警告: {symbol} 资金费率 {funding_rate:.4%} 较高，多头将支付资金费用"
        elif funding_rate < 0 and position < 0:
            return f"警告: {symbol} 资金费率 {funding_rate:.4%} 为负，空头将支付资金费用"

        return None

    def record_trade(self, instruction: TradeInstruction, price: float):
        """记录交易"""
        symbol = instruction.symbol
        self.last_trade_time[symbol] = datetime.now()
        self.trade_history.append({
            "time": datetime.now(),
            "symbol": symbol,
            "action": instruction.action.value,
            "quantity": instruction.quantity,
            "price": price,
        })

        # 更新持仓
        if instruction.action in [TradeAction.OPEN_LONG]:
            self.positions[symbol] = self.positions.get(symbol, 0) + instruction.quantity
            if symbol not in self.position_entry_prices:
                self.position_entry_prices[symbol] = price
        elif instruction.action in [TradeAction.OPEN_SHORT]:
            self.positions[symbol] = self.positions.get(symbol, 0) - instruction.quantity
            if symbol not in self.position_entry_prices:
                self.position_entry_prices[symbol] = price
        elif instruction.action in [TradeAction.CLOSE_LONG]:
            self.positions[symbol] = self.positions.get(symbol, 0) - instruction.quantity
            if self.positions.get(symbol, 0) <= 0:
                self.position_entry_prices.pop(symbol, None)
        elif instruction.action in [TradeAction.CLOSE_SHORT]:
            self.positions[symbol] = self.positions.get(symbol, 0) + instruction.quantity
            if self.positions.get(symbol, 0) >= 0:
                self.position_entry_prices.pop(symbol, None)

    def get_risk_report(self) -> Dict:
        """获取风险报告"""
        current_drawdown = (self.peak_equity - self.current_equity) / self.peak_equity if self.peak_equity > 0 else 0

        total_pos_value = sum(abs(v) for v in self.positions.values())

        return {
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown": current_drawdown,
            "daily_pnl": self.daily_pnl,
            "total_position_value": total_pos_value,
            "position_count": len([p for p in self.positions.values() if abs(p) > 0]),
            "trades_last_hour": len([t for t in self.trade_history if t["time"] > datetime.now() - timedelta(hours=1)]),
        }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    from .adapter import TradeInstruction, TradeAction

    # 创建风控管理器
    config = RiskConfig(
        max_position_per_symbol=0.2,
        max_leverage=5.0,
        stop_loss_pct=0.03,
    )
    risk_manager = RiskManager(config, capital=100000)

    # 模拟交易检查
    instruction = TradeInstruction(
        symbol="btcusdt",
        action=TradeAction.OPEN_LONG,
        quantity=0.5,
        leverage=3.0,
    )

    passed, reason = risk_manager.check_trade(instruction, current_price=40000)
    print(f"交易检查: {'通过' if passed else '拒绝'}, 原因: {reason}")

    # 记录交易
    if passed:
        risk_manager.record_trade(instruction, 40000)

    # 获取风险报告
    report = risk_manager.get_risk_report()
    print("\n风险报告:")
    for k, v in report.items():
        print(f"  {k}: {v}")
