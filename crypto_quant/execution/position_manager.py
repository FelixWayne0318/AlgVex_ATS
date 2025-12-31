"""
仓位管理模块

职责:
1. 跟踪当前持仓状态
2. 同步交易所持仓
3. 计算盈亏
4. 持仓调整建议
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    amount: float                  # 持仓量 (正=多, 负=空)
    entry_price: float             # 开仓均价
    current_price: float = 0.0     # 当前价格
    leverage: float = 1.0          # 杠杆
    margin: float = 0.0            # 占用保证金
    unrealized_pnl: float = 0.0    # 未实现盈亏
    realized_pnl: float = 0.0      # 已实现盈亏
    funding_paid: float = 0.0      # 累计支付资金费用
    open_time: datetime = None     # 开仓时间

    @property
    def side(self) -> str:
        """持仓方向"""
        if self.amount > 0:
            return "long"
        elif self.amount < 0:
            return "short"
        return "none"

    @property
    def notional_value(self) -> float:
        """名义价值"""
        return abs(self.amount) * self.current_price

    @property
    def pnl_pct(self) -> float:
        """盈亏百分比"""
        if self.entry_price == 0:
            return 0
        if self.amount > 0:
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

    def update_price(self, price: float):
        """更新价格"""
        self.current_price = price
        if self.amount > 0:
            self.unrealized_pnl = (price - self.entry_price) * self.amount
        else:
            self.unrealized_pnl = (self.entry_price - price) * abs(self.amount)


class PositionManager:
    """
    仓位管理器

    管理所有持仓的生命周期
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, PositionInfo] = {}
        self.closed_positions: List[PositionInfo] = []
        self.position_history: List[Dict] = []

    @property
    def total_equity(self) -> float:
        """总权益 = 现金 + 未实现盈亏"""
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return self.cash + unrealized

    @property
    def total_margin(self) -> float:
        """总占用保证金"""
        return sum(p.margin for p in self.positions.values())

    @property
    def available_margin(self) -> float:
        """可用保证金"""
        return self.cash - self.total_margin

    @property
    def total_unrealized_pnl(self) -> float:
        """总未实现盈亏"""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def open_position(
        self,
        symbol: str,
        amount: float,
        price: float,
        leverage: float = 1.0,
    ) -> Optional[PositionInfo]:
        """
        开仓

        Args:
            symbol: 交易对
            amount: 数量 (正=多, 负=空)
            price: 开仓价
            leverage: 杠杆

        Returns:
            持仓信息
        """
        notional = abs(amount) * price
        margin_required = notional / leverage

        # 检查保证金
        if margin_required > self.available_margin:
            logger.warning(f"Insufficient margin for {symbol}: required {margin_required:.2f}, available {self.available_margin:.2f}")
            return None

        # 创建或更新持仓
        if symbol in self.positions:
            pos = self.positions[symbol]

            # 同向加仓
            if (pos.amount > 0 and amount > 0) or (pos.amount < 0 and amount < 0):
                total_notional = abs(pos.amount * pos.entry_price) + abs(amount * price)
                total_amount = abs(pos.amount) + abs(amount)
                pos.entry_price = total_notional / total_amount
                pos.amount += amount
                pos.margin += margin_required
                pos.leverage = leverage
            else:
                # 反向开仓 - 先平掉原仓位
                self.close_position(symbol, abs(pos.amount), price)
                # 再开新仓
                return self.open_position(symbol, amount, price, leverage)
        else:
            pos = PositionInfo(
                symbol=symbol,
                amount=amount,
                entry_price=price,
                current_price=price,
                leverage=leverage,
                margin=margin_required,
                open_time=datetime.now(),
            )
            self.positions[symbol] = pos

        # 扣除保证金
        self.cash -= margin_required

        # 记录历史
        self._record_history("open", symbol, amount, price)

        logger.info(f"Opened position: {symbol} {amount:.4f} @ {price:.2f}, margin={margin_required:.2f}")
        return pos

    def close_position(
        self,
        symbol: str,
        amount: float,
        price: float,
    ) -> Optional[float]:
        """
        平仓

        Args:
            symbol: 交易对
            amount: 平仓数量
            price: 平仓价

        Returns:
            实现盈亏
        """
        if symbol not in self.positions:
            logger.warning(f"No position to close: {symbol}")
            return None

        pos = self.positions[symbol]
        close_amount = min(abs(amount), abs(pos.amount))

        # 计算盈亏
        if pos.amount > 0:
            pnl = (price - pos.entry_price) * close_amount
        else:
            pnl = (pos.entry_price - price) * close_amount

        # 释放保证金
        margin_ratio = close_amount / abs(pos.amount)
        released_margin = pos.margin * margin_ratio

        # 更新持仓
        if pos.amount > 0:
            pos.amount -= close_amount
        else:
            pos.amount += close_amount

        pos.margin -= released_margin
        pos.realized_pnl += pnl

        # 更新现金
        self.cash += released_margin + pnl

        # 记录历史
        self._record_history("close", symbol, close_amount, price, pnl)

        # 清理空仓
        if abs(pos.amount) < 1e-8:
            self.closed_positions.append(pos)
            del self.positions[symbol]
            logger.info(f"Closed position: {symbol}, total PnL={pos.realized_pnl:.2f}")
        else:
            logger.info(f"Reduced position: {symbol} {close_amount:.4f} @ {price:.2f}, PnL={pnl:.2f}")

        return pnl

    def update_prices(self, prices: Dict[str, float]):
        """更新所有持仓价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def apply_funding(self, symbol: str, funding_rate: float, mark_price: float):
        """
        应用资金费率

        Args:
            symbol: 交易对
            funding_rate: 资金费率
            mark_price: 标记价格
        """
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        position_value = abs(pos.amount) * mark_price
        funding_cost = position_value * funding_rate

        if pos.amount > 0:
            # 多头支付正费率
            self.cash -= funding_cost
            pos.funding_paid += funding_cost
        else:
            # 空头收取正费率
            self.cash += funding_cost
            pos.funding_paid -= funding_cost

        logger.debug(f"Funding applied: {symbol} {funding_cost:.4f}")

    def _record_history(self, action: str, symbol: str, amount: float, price: float, pnl: float = 0):
        """记录历史"""
        self.position_history.append({
            "time": datetime.now(),
            "action": action,
            "symbol": symbol,
            "amount": amount,
            "price": price,
            "pnl": pnl,
            "equity": self.total_equity,
        })

    def get_position_summary(self) -> pd.DataFrame:
        """获取持仓汇总"""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for symbol, pos in self.positions.items():
            data.append({
                "symbol": symbol,
                "side": pos.side,
                "amount": pos.amount,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealized_pnl": pos.unrealized_pnl,
                "pnl_pct": pos.pnl_pct,
                "leverage": pos.leverage,
                "margin": pos.margin,
                "notional": pos.notional_value,
            })

        return pd.DataFrame(data)

    def get_equity_curve(self) -> pd.DataFrame:
        """获取权益曲线"""
        if not self.position_history:
            return pd.DataFrame()

        return pd.DataFrame(self.position_history)

    def get_performance_metrics(self) -> Dict:
        """计算绩效指标"""
        equity_curve = self.get_equity_curve()
        if equity_curve.empty:
            return {}

        equity = equity_curve["equity"]
        returns = equity.pct_change().dropna()

        total_return = (equity.iloc[-1] / self.initial_capital - 1) if len(equity) > 0 else 0
        max_drawdown = (equity / equity.cummax() - 1).min()
        sharpe = returns.mean() / returns.std() * np.sqrt(365 * 24) if returns.std() > 0 else 0

        # 交易统计
        closed_pnls = [p.realized_pnl for p in self.closed_positions]
        winning = len([p for p in closed_pnls if p > 0])
        total = len(closed_pnls)

        return {
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
            "win_rate": winning / total if total > 0 else 0,
            "total_trades": total,
            "total_pnl": sum(closed_pnls),
            "current_equity": self.total_equity,
        }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建仓位管理器
    pm = PositionManager(initial_capital=100000)

    # 开多仓
    pm.open_position("btcusdt", 1.0, 40000, leverage=2.0)

    # 更新价格
    pm.update_prices({"btcusdt": 41000})

    # 查看持仓
    print("当前持仓:")
    print(pm.get_position_summary())
    print(f"\n总权益: {pm.total_equity:.2f}")
    print(f"未实现盈亏: {pm.total_unrealized_pnl:.2f}")

    # 平仓
    pnl = pm.close_position("btcusdt", 1.0, 41000)
    print(f"\n平仓盈亏: {pnl:.2f}")
    print(f"最终权益: {pm.total_equity:.2f}")

    # 绩效指标
    print("\n绩效指标:")
    metrics = pm.get_performance_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
