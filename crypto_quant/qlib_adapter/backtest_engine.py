"""
永续合约回测引擎 - 支持做空和资金费率

扩展Qlib回测能力:
1. 支持做空 (负持仓)
2. 模拟资金费率扣除
3. 杠杆和保证金计算
4. 爆仓检测
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    amount: float = 0.0          # 正=多头, 负=空头
    entry_price: float = 0.0     # 开仓均价
    leverage: float = 1.0        # 杠杆倍数
    margin: float = 0.0          # 占用保证金
    unrealized_pnl: float = 0.0  # 未实现盈亏


@dataclass
class Trade:
    """交易记录"""
    datetime: pd.Timestamp
    symbol: str
    side: str           # "open_long", "open_short", "close_long", "close_short"
    amount: float
    price: float
    fee: float = 0.0
    funding_cost: float = 0.0
    pnl: float = 0.0


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_capital: float = 100000.0    # 初始资金
    leverage: float = 1.0                 # 默认杠杆
    max_leverage: float = 10.0            # 最大杠杆
    taker_fee: float = 0.0004             # Taker费率 0.04%
    maker_fee: float = 0.0002             # Maker费率 0.02%
    slippage: float = 0.0001              # 滑点 0.01%
    funding_rate_interval: int = 8        # 资金费率间隔(小时)
    liquidation_threshold: float = 0.8    # 爆仓阈值 (保证金率)


class CryptoPerpetualBacktest:
    """
    永续合约回测引擎

    特性:
    - 支持多空双向交易
    - 资金费率模拟
    - 杠杆和保证金计算
    - 爆仓检测
    - 完整的交易记录
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """重置回测状态"""
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.current_time: pd.Timestamp = None

    # ==================== 交易执行 ====================
    def open_long(
        self,
        symbol: str,
        amount: float,
        price: float,
        leverage: float = None,
    ) -> Trade:
        """开多"""
        leverage = leverage or self.config.leverage
        return self._open_position(symbol, amount, price, leverage, "long")

    def open_short(
        self,
        symbol: str,
        amount: float,
        price: float,
        leverage: float = None,
    ) -> Trade:
        """开空"""
        leverage = leverage or self.config.leverage
        return self._open_position(symbol, -amount, price, leverage, "short")

    def close_long(self, symbol: str, amount: float, price: float) -> Trade:
        """平多"""
        return self._close_position(symbol, amount, price, "close_long")

    def close_short(self, symbol: str, amount: float, price: float) -> Trade:
        """平空"""
        return self._close_position(symbol, amount, price, "close_short")

    def _open_position(
        self,
        symbol: str,
        amount: float,  # 正=多, 负=空
        price: float,
        leverage: float,
        side: str,
    ) -> Trade:
        """开仓"""
        # 计算费用
        notional = abs(amount) * price
        fee = notional * self.config.taker_fee
        slippage_cost = notional * self.config.slippage

        # 计算保证金
        margin = notional / leverage

        # 检查资金
        if margin + fee > self.cash:
            logger.warning(f"Insufficient margin for {symbol}")
            return None

        # 更新持仓
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        pos = self.positions[symbol]

        # 同向加仓 or 新开仓
        if (pos.amount >= 0 and amount >= 0) or (pos.amount <= 0 and amount <= 0):
            total_notional = abs(pos.amount * pos.entry_price) + abs(amount * price)
            total_amount = abs(pos.amount) + abs(amount)
            pos.entry_price = total_notional / total_amount if total_amount > 0 else price
            pos.amount += amount
            pos.leverage = leverage
            pos.margin += margin
        else:
            # 反向开仓 (先平后开)
            # 简化处理: 直接替换持仓
            pos.entry_price = price
            pos.amount = amount
            pos.leverage = leverage
            pos.margin = margin

        # 扣除费用
        self.cash -= (margin + fee + slippage_cost)

        # 记录交易
        trade = Trade(
            datetime=self.current_time,
            symbol=symbol,
            side=f"open_{side}",
            amount=abs(amount),
            price=price,
            fee=fee,
        )
        self.trades.append(trade)

        return trade

    def _close_position(
        self,
        symbol: str,
        amount: float,
        price: float,
        side: str,
    ) -> Trade:
        """平仓"""
        if symbol not in self.positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        pos = self.positions[symbol]

        # 计算实际平仓量
        close_amount = min(abs(amount), abs(pos.amount))
        if close_amount == 0:
            return None

        # 计算盈亏
        if pos.amount > 0:  # 平多
            pnl = (price - pos.entry_price) * close_amount
        else:  # 平空
            pnl = (pos.entry_price - price) * close_amount

        # 计算费用
        notional = close_amount * price
        fee = notional * self.config.taker_fee

        # 释放保证金
        released_margin = pos.margin * (close_amount / abs(pos.amount))

        # 更新持仓
        if pos.amount > 0:
            pos.amount -= close_amount
        else:
            pos.amount += close_amount

        pos.margin -= released_margin

        # 更新现金
        self.cash += released_margin + pnl - fee

        # 清理空仓
        if abs(pos.amount) < 1e-8:
            del self.positions[symbol]

        # 记录交易
        trade = Trade(
            datetime=self.current_time,
            symbol=symbol,
            side=side,
            amount=close_amount,
            price=price,
            fee=fee,
            pnl=pnl,
        )
        self.trades.append(trade)

        return trade

    # ==================== 资金费率 ====================
    def apply_funding_rate(self, symbol: str, funding_rate: float, mark_price: float):
        """
        应用资金费率

        资金费率规则:
        - 正费率: 多头付给空头
        - 负费率: 空头付给多头
        """
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        if abs(pos.amount) < 1e-8:
            return

        # 计算资金费用
        position_value = abs(pos.amount) * mark_price
        funding_cost = position_value * funding_rate

        if pos.amount > 0:
            # 多头: 正费率付出, 负费率收入
            self.cash -= funding_cost
        else:
            # 空头: 正费率收入, 负费率付出
            self.cash += funding_cost

        # 记录
        if abs(funding_cost) > 0:
            trade = Trade(
                datetime=self.current_time,
                symbol=symbol,
                side="funding",
                amount=0,
                price=mark_price,
                funding_cost=funding_cost,
            )
            self.trades.append(trade)

    # ==================== 爆仓检测 ====================
    def check_liquidation(self, prices: Dict[str, float]) -> List[str]:
        """检查爆仓"""
        liquidated = []

        for symbol, pos in list(self.positions.items()):
            if abs(pos.amount) < 1e-8:
                continue

            price = prices.get(symbol, pos.entry_price)

            # 计算未实现盈亏
            if pos.amount > 0:
                unrealized_pnl = (price - pos.entry_price) * pos.amount
            else:
                unrealized_pnl = (pos.entry_price - price) * abs(pos.amount)

            # 计算保证金率
            equity = pos.margin + unrealized_pnl
            margin_ratio = equity / pos.margin if pos.margin > 0 else 1.0

            # 爆仓
            if margin_ratio < (1 - self.config.liquidation_threshold):
                logger.warning(f"Liquidation: {symbol}, margin_ratio={margin_ratio:.2%}")

                # 强制平仓
                if pos.amount > 0:
                    self.close_long(symbol, abs(pos.amount), price)
                else:
                    self.close_short(symbol, abs(pos.amount), price)

                liquidated.append(symbol)

        return liquidated

    def _dict_to_multiindex(self, data_dict: Dict, col_name: str) -> pd.DataFrame:
        """将字典格式转换为MultiIndex DataFrame"""
        all_data = []
        for symbol, series in data_dict.items():
            df = series.to_frame(name=col_name)
            df["instrument"] = symbol
            df.index.name = "datetime"
            all_data.append(df.reset_index())

        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.set_index(["datetime", "instrument"])
        return combined

    # ==================== 回测运行 ====================
    def run(
        self,
        signals,  # pd.DataFrame or Dict[str, pd.Series]
        prices,   # pd.DataFrame or Dict[str, pd.Series]
        funding_rates = None,  # pd.DataFrame or Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        运行回测

        Args:
            signals: 交易信号 DataFrame
                - index: (datetime, instrument)
                - columns: signal (-1=做空, 0=空仓, 1=做多), 或具体持仓权重
            prices: 价格数据
                - index: (datetime, instrument)
                - columns: $close, $open, $high, $low
            funding_rates: 资金费率 (可选)
                - index: (datetime, instrument)
                - columns: $funding_rate

        Returns:
            回测结果DataFrame
        """
        self.reset()

        # 支持字典格式输入
        if isinstance(signals, dict):
            signals = self._dict_to_multiindex(signals, "signal")
        if isinstance(prices, dict):
            prices = self._dict_to_multiindex(prices, "$close")
        if funding_rates is not None and isinstance(funding_rates, dict):
            funding_rates = self._dict_to_multiindex(funding_rates, "$funding_rate")

        # 获取所有时间点
        all_times = signals.index.get_level_values("datetime").unique().sort_values()

        for t in all_times:
            self.current_time = t

            # 获取当前信号和价格
            try:
                current_signals = signals.xs(t, level="datetime")
                current_prices = prices.xs(t, level="datetime")
            except KeyError:
                continue

            # 构建价格字典
            price_dict = {}
            for inst in current_prices.index:
                price_dict[inst] = current_prices.loc[inst, "$close"]

            # 检查爆仓
            self.check_liquidation(price_dict)

            # 应用资金费率 (每8小时)
            if funding_rates is not None and t.hour % self.config.funding_rate_interval == 0:
                try:
                    current_funding = funding_rates.xs(t, level="datetime")
                    for inst in current_funding.index:
                        if inst in price_dict:
                            fr = current_funding.loc[inst, "$funding_rate"]
                            self.apply_funding_rate(inst, fr, price_dict[inst])
                except KeyError:
                    pass

            # 执行交易信号
            for inst in current_signals.index:
                if inst not in price_dict:
                    continue

                signal = current_signals.loc[inst, "signal"] if "signal" in current_signals.columns else current_signals.loc[inst]
                price = price_dict[inst]

                # 获取当前持仓
                current_pos = self.positions.get(inst, Position(symbol=inst)).amount

                # 目标持仓 (简化: signal * 固定仓位)
                target_pos = signal * (self.config.initial_capital * 0.1 / price)  # 10%资金

                # 计算需要的交易
                diff = target_pos - current_pos

                if abs(diff) < 1e-8:
                    continue

                if diff > 0:
                    if current_pos < 0:
                        # 先平空
                        self.close_short(inst, abs(current_pos), price)
                    # 开多
                    self.open_long(inst, abs(diff), price)
                else:
                    if current_pos > 0:
                        # 先平多
                        self.close_long(inst, abs(current_pos), price)
                    # 开空
                    self.open_short(inst, abs(diff), price)

            # 记录净值
            total_equity = self.cash
            for symbol, pos in self.positions.items():
                if symbol in price_dict:
                    price = price_dict[symbol]
                    if pos.amount > 0:
                        unrealized = (price - pos.entry_price) * pos.amount
                    else:
                        unrealized = (pos.entry_price - price) * abs(pos.amount)
                    total_equity += pos.margin + unrealized

            self.equity_curve.append({
                "datetime": t,
                "cash": self.cash,
                "equity": total_equity,
                "positions": len(self.positions),
            })

        return self.get_results()

    def get_results(self) -> pd.DataFrame:
        """获取回测结果"""
        equity_df = pd.DataFrame(self.equity_curve)
        if equity_df.empty:
            return equity_df

        equity_df = equity_df.set_index("datetime")

        # 计算指标
        equity_df["returns"] = equity_df["equity"].pct_change()
        equity_df["cumulative_returns"] = (1 + equity_df["returns"]).cumprod() - 1
        equity_df["drawdown"] = equity_df["equity"] / equity_df["equity"].cummax() - 1
        equity_df["max_drawdown"] = equity_df["drawdown"].cummin()

        return equity_df

    def get_metrics(self) -> Dict:
        """计算回测指标"""
        equity_df = self.get_results()
        if equity_df.empty:
            return {}

        returns = equity_df["returns"].dropna()
        total_return = equity_df["equity"].iloc[-1] / self.config.initial_capital - 1
        max_drawdown = equity_df["max_drawdown"].min()

        # 交易统计
        trades_df = pd.DataFrame([t.__dict__ for t in self.trades])
        if not trades_df.empty:
            winning_trades = len(trades_df[trades_df["pnl"] > 0])
            total_trades = len(trades_df[trades_df["pnl"] != 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_fees = trades_df["fee"].sum()
            total_funding = trades_df["funding_cost"].sum()
        else:
            win_rate = 0
            total_fees = 0
            total_funding = 0
            total_trades = 0

        # 年化收益和夏普
        periods_per_year = 24 * 365  # 小时级数据
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1 if len(returns) > 0 else 0
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "total_fees": total_fees,
            "total_funding_cost": total_funding,
            "final_equity": equity_df["equity"].iloc[-1],
        }


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建示例数据
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    instruments = ["btcusdt", "ethusdt"]

    # 价格数据
    price_data = []
    for inst in instruments:
        base_price = 40000 if inst == "btcusdt" else 2000
        for i, dt in enumerate(dates):
            price = base_price * (1 + np.sin(i / 10) * 0.1 + np.random.randn() * 0.01)
            price_data.append({
                "datetime": dt,
                "instrument": inst,
                "$close": price,
                "$open": price * 0.999,
                "$high": price * 1.01,
                "$low": price * 0.99,
            })

    prices = pd.DataFrame(price_data).set_index(["datetime", "instrument"])

    # 信号数据 (简单均值回归)
    signal_data = []
    for inst in instruments:
        inst_prices = prices.xs(inst, level="instrument")["$close"]
        ma = inst_prices.rolling(20).mean()
        signal = np.sign(ma - inst_prices)  # 价格低于均线做多

        for dt, sig in zip(dates, signal):
            signal_data.append({
                "datetime": dt,
                "instrument": inst,
                "signal": sig if not np.isnan(sig) else 0,
            })

    signals = pd.DataFrame(signal_data).set_index(["datetime", "instrument"])

    # 资金费率
    funding_data = []
    for inst in instruments:
        for dt in dates:
            funding_data.append({
                "datetime": dt,
                "instrument": inst,
                "$funding_rate": 0.0001 + np.random.randn() * 0.0002,
            })

    funding_rates = pd.DataFrame(funding_data).set_index(["datetime", "instrument"])

    # 运行回测
    config = BacktestConfig(
        initial_capital=100000,
        leverage=2.0,
        taker_fee=0.0004,
    )
    engine = CryptoPerpetualBacktest(config)
    results = engine.run(signals, prices, funding_rates)

    print("\n回测结果:")
    print(results.tail(10))

    print("\n回测指标:")
    metrics = engine.get_metrics()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
