"""
AlgVex 永续合约回测引擎

核心回测引擎，支持:
- 永续合约特性 (无到期日)
- 资金费率模拟
- 强平逻辑
- 杠杆保证金计算
- 动态滑点和手续费

设计文档参考: Section 5.1 CryptoPerpetualBacktest
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd

from .config import BacktestConfig
from .models import (
    Account, Position, Trade, Signal,
    PositionSide, TradeType, OrderType, TradeStatus,
)
from .metrics import BacktestMetrics, BacktestResult
from .triple_barrier import (
    TripleBarrier, TripleBarrierConfig, PositionState, BarrierType
)

# 导入 P0 组件
from shared.price_semantics import PriceSemantics, PriceScenario, PriceData
from shared.funding_rate import FundingRateHandler
from shared.execution_models import DynamicSlippageModel, FeeModel, VIPLevel

logger = logging.getLogger(__name__)


@dataclass
class BarData:
    """K线数据"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = ""
    mark_price: Optional[float] = None
    index_price: Optional[float] = None
    funding_rate: Optional[float] = None


class CryptoPerpetualBacktest:
    """
    永续合约回测引擎

    使用示例:
        from algvex.core.backtest import BacktestConfig, CryptoPerpetualBacktest

        config = BacktestConfig(
            initial_capital=100000.0,
            leverage=3.0,
            taker_fee=0.0004,
            maker_fee=0.0002,
        )

        engine = CryptoPerpetualBacktest(config)
        results = engine.run(signals, prices, funding_rates)
    """

    def __init__(
        self,
        config: BacktestConfig,
        triple_barrier_config: Optional[TripleBarrierConfig] = None,
    ):
        """
        初始化回测引擎

        Args:
            config: 回测配置
            triple_barrier_config: 三重屏障配置 (可选)
        """
        self.config = config
        self.triple_barrier_config = triple_barrier_config

        # 初始化账户
        self.account = Account(
            balance=config.initial_capital,
            initial_balance=config.initial_capital,
        )

        # P0 组件
        self.price_semantics = PriceSemantics()
        self.funding_handler = FundingRateHandler()
        self.slippage_model = DynamicSlippageModel(
            base_slippage=config.slippage,
            max_slippage=config.execution_config.max_slippage,
        )
        self.fee_model = FeeModel(
            exchange=config.execution_config.exchange,
            vip_level=config.execution_config.vip_level,
            custom_fees={
                "maker": config.maker_fee,
                "taker": config.taker_fee,
            },
        )

        # 三重屏障风控
        self.triple_barrier: Optional[TripleBarrier] = None
        if triple_barrier_config:
            self.triple_barrier = TripleBarrier(triple_barrier_config)

        # 回测指标计算器
        self.metrics_calculator = BacktestMetrics()

        # 状态跟踪
        self.current_time: Optional[datetime] = None
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.all_trades: List[Trade] = []
        self.all_positions: List[Position] = []

        # 三重屏障触发统计
        self.barrier_triggers: Dict[str, int] = {
            "stop_loss": 0,
            "take_profit": 0,
            "time_limit": 0,
            "trailing_stop": 0,
        }

        # 资金费率数据
        self._funding_rates: Dict[str, Dict[datetime, float]] = {}

        # 市场数据缓存
        self._market_data: Dict[str, pd.DataFrame] = {}

        logger.info(f"CryptoPerpetualBacktest initialized with capital={config.initial_capital}")
        if triple_barrier_config:
            logger.info(f"Triple barrier enabled: SL={triple_barrier_config.stop_loss}, TP={triple_barrier_config.take_profit}")

    def reset(self):
        """重置引擎状态"""
        self.account = Account(
            balance=self.config.initial_capital,
            initial_balance=self.config.initial_capital,
        )
        self.current_time = None
        self.equity_curve = []
        self.all_trades = []
        self.all_positions = []
        self.barrier_triggers = {
            "stop_loss": 0,
            "take_profit": 0,
            "time_limit": 0,
            "trailing_stop": 0,
        }
        self._funding_rates = {}
        self._market_data = {}

    def run(
        self,
        signals: Union[List[Signal], pd.DataFrame],
        prices: Union[Dict[str, pd.DataFrame], pd.DataFrame],
        funding_rates: Optional[Dict[str, Dict[datetime, float]]] = None,
    ) -> BacktestResult:
        """
        运行回测

        Args:
            signals: 信号列表或 DataFrame
            prices: 价格数据 {symbol: DataFrame} 或 合并的 DataFrame
            funding_rates: 资金费率 {symbol: {time: rate}}

        Returns:
            BacktestResult
        """
        # 转换输入格式
        signal_list = self._normalize_signals(signals)
        self._market_data = self._normalize_prices(prices)
        self._funding_rates = funding_rates or {}

        if not signal_list:
            logger.warning("No signals provided")
            return self._create_empty_result()

        # 获取时间范围
        start_time = min(s.timestamp for s in signal_list)
        end_time = max(s.timestamp for s in signal_list)

        # 获取所有 bar 时间点
        all_times = self._get_all_bar_times()
        if not all_times:
            logger.warning("No price data available")
            return self._create_empty_result()

        logger.info(f"Running backtest from {start_time} to {end_time}")
        logger.info(f"Total signals: {len(signal_list)}, Total bars: {len(all_times)}")

        # 按时间戳索引信号
        signals_by_time = {}
        for signal in signal_list:
            if signal.timestamp not in signals_by_time:
                signals_by_time[signal.timestamp] = []
            signals_by_time[signal.timestamp].append(signal)

        # 主回测循环
        for bar_time in sorted(all_times):
            self.current_time = bar_time

            # 1. 更新价格
            self._update_prices(bar_time)

            # 2. 检查资金费结算
            if self.config.enable_funding:
                self._process_funding_settlement(bar_time)

            # 3. 检查强平
            self._check_liquidations(bar_time)

            # 4. 检查三重屏障
            if self.triple_barrier:
                self._check_triple_barriers(bar_time)

            # 5. 处理信号
            if bar_time in signals_by_time:
                for signal in signals_by_time[bar_time]:
                    self._process_signal(signal, bar_time)

            # 6. 更新权益曲线
            self._update_equity_curve(bar_time)

        # 关闭所有持仓 (回测结束时)
        self._close_all_positions(end_time)

        # 计算指标
        result = self.metrics_calculator.calculate_metrics(
            account=self.account,
            equity_curve=self.equity_curve,
            trades=self.all_trades,
            positions=self.all_positions,
            start_time=start_time,
            end_time=end_time,
            config=self.config.to_dict(),
        )

        logger.info(f"Backtest completed: return={result.total_return:.2%}, trades={result.total_trades}")
        return result

    def _normalize_signals(self, signals: Union[List[Signal], pd.DataFrame]) -> List[Signal]:
        """标准化信号格式"""
        if isinstance(signals, list):
            return signals

        # DataFrame 转换
        signal_list = []
        for _, row in signals.iterrows():
            signal = Signal(
                symbol=row.get("symbol", ""),
                signal_type=row.get("signal_type", row.get("type", "")),
                strength=row.get("strength", row.get("score", 0)),
                timestamp=pd.to_datetime(row.get("timestamp", row.name)),
                price=row.get("price"),
            )
            signal_list.append(signal)

        return signal_list

    def _normalize_prices(
        self,
        prices: Union[Dict[str, pd.DataFrame], pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """标准化价格数据格式"""
        if isinstance(prices, dict):
            return prices

        # 单个 DataFrame，按 symbol 分组
        if "symbol" in prices.columns:
            return {
                symbol: group.copy()
                for symbol, group in prices.groupby("symbol")
            }

        # 假设单一标的
        return {"default": prices}

    def _get_all_bar_times(self) -> List[datetime]:
        """获取所有 bar 时间点"""
        all_times = set()
        for df in self._market_data.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_times.update(df.index.to_pydatetime())
            elif "timestamp" in df.columns:
                all_times.update(pd.to_datetime(df["timestamp"]).to_pydatetime())
            elif "datetime" in df.columns:
                all_times.update(pd.to_datetime(df["datetime"]).to_pydatetime())

        return sorted(all_times)

    def _get_bar_data(self, symbol: str, time: datetime) -> Optional[BarData]:
        """获取指定时间的 bar 数据"""
        if symbol not in self._market_data:
            return None

        df = self._market_data[symbol]

        # 尝试不同的索引方式
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                if time in df.index:
                    row = df.loc[time]
                else:
                    return None
            else:
                # 使用时间列
                time_col = "timestamp" if "timestamp" in df.columns else "datetime"
                mask = pd.to_datetime(df[time_col]) == pd.to_datetime(time)
                if not mask.any():
                    return None
                row = df[mask].iloc[0]

            return BarData(
                timestamp=time,
                open=float(row.get("open", row.get("Open", 0))),
                high=float(row.get("high", row.get("High", 0))),
                low=float(row.get("low", row.get("Low", 0))),
                close=float(row.get("close", row.get("Close", 0))),
                volume=float(row.get("volume", row.get("Volume", 0))),
                symbol=symbol,
                mark_price=float(row.get("mark_price")) if "mark_price" in row else None,
                index_price=float(row.get("index_price")) if "index_price" in row else None,
                funding_rate=float(row.get("funding_rate")) if "funding_rate" in row else None,
            )
        except Exception as e:
            logger.debug(f"Failed to get bar data for {symbol} at {time}: {e}")
            return None

    def _update_prices(self, time: datetime):
        """更新所有持仓的价格"""
        for symbol, position in self.account.positions.items():
            bar = self._get_bar_data(symbol, time)
            if bar:
                # 使用正确的价格语义 (mark_price 用于 PnL 计算)
                price_data = PriceData(
                    mark_price=bar.mark_price or bar.close,
                    close_price=bar.close,
                    last_price=bar.close,
                )
                mark_price = self.price_semantics.get_price(
                    PriceScenario.PNL_CALCULATION,
                    price_data,
                    fallback=True,
                )
                position.update_price(mark_price, time)

        # 更新账户未实现盈亏
        self.account.unrealized_pnl = sum(
            p.unrealized_pnl for p in self.account.positions.values()
        )
        self.account.update_equity_stats()

    def _process_funding_settlement(self, time: datetime):
        """处理资金费结算"""
        # 检查是否是结算时间
        if time.hour not in self.funding_handler.settlement_hours:
            return
        if time.minute != 0 or time.second != 0:
            return

        for symbol, position in self.account.positions.items():
            if not position.is_open:
                continue

            # 获取资金费率
            rate = None
            if symbol in self._funding_rates:
                rate = self._funding_rates[symbol].get(time)

            # 尝试从 bar 数据获取
            if rate is None:
                bar = self._get_bar_data(symbol, time)
                if bar and bar.funding_rate is not None:
                    rate = bar.funding_rate

            if rate is None:
                continue

            # 计算资金费支付
            payment = self.funding_handler.calculate_funding_payment(
                symbol=symbol,
                position_value=position.position_value,
                side=position.side.value,
                entry_time=position.entry_time,
                current_time=time,
                funding_rate=rate,
            )

            if payment:
                # 记录资金费
                position.add_funding(payment.payment)
                self.account.total_funding += payment.payment
                self.account.balance -= payment.payment

                logger.debug(
                    f"Funding paid: {symbol} {payment.payment:.4f} "
                    f"(rate={rate:.4%}, value={position.position_value:.2f})"
                )

    def _check_liquidations(self, time: datetime):
        """检查并处理强平"""
        symbols_to_liquidate = []

        for symbol, position in self.account.positions.items():
            if not position.is_open:
                continue

            if position.check_liquidation(self.config.maintenance_margin_rate):
                symbols_to_liquidate.append(symbol)
                logger.warning(f"Liquidation triggered for {symbol}")

        for symbol in symbols_to_liquidate:
            self._liquidate_position(symbol, time)

    def _liquidate_position(self, symbol: str, time: datetime):
        """强制平仓"""
        position = self.account.positions.get(symbol)
        if not position or not position.is_open:
            return

        bar = self._get_bar_data(symbol, time)
        if not bar:
            return

        # 使用当前价格强平
        close_price = bar.close

        # 标记为强平
        position.liquidated = True

        # 平仓
        trade = position.close(
            close_price=close_price,
            close_time=time,
            fee=self._calculate_fee(position.position_value, is_maker=False),
            slippage=0,  # 强平不计算滑点
        )

        # 更新账户
        self._finalize_position_close(position, trade)

        logger.warning(
            f"Position liquidated: {symbol}, "
            f"loss={trade.pnl:.2f}, price={close_price:.4f}"
        )

    def _check_triple_barriers(self, time: datetime):
        """检查三重屏障触发"""
        if not self.triple_barrier:
            return

        symbols_to_close = []

        for symbol, position in self.account.positions.items():
            if not position.is_open:
                continue

            bar = self._get_bar_data(symbol, time)
            if not bar:
                continue

            # 创建持仓状态 (用于屏障检查)
            position_state = PositionState(
                symbol=symbol,
                side=position.side.value,
                entry_price=position.entry_price,
                quantity=position.quantity,
                entry_time=position.entry_time,
                highest_price=getattr(position, 'highest_price', position.entry_price),
                lowest_price=getattr(position, 'lowest_price', position.entry_price),
                trailing_stop_price=getattr(position, 'trailing_stop_price', None),
            )

            # 准备价格数据
            price_data = {
                "mark_price": bar.mark_price or bar.close,
                "last_price": bar.close,
                "index_price": bar.index_price or bar.close,
            }

            # 检查屏障
            result = self.triple_barrier.check(
                position=position_state,
                current_price=bar.close,
                current_time=time,
                price_data=price_data,
            )

            if result.triggered:
                symbols_to_close.append((symbol, result.barrier_type, bar.close))

                # 更新持仓的 highest/lowest 价格 (用于移动止损)
                position.highest_price = position_state.highest_price
                position.lowest_price = position_state.lowest_price
                position.trailing_stop_price = position_state.trailing_stop_price

                logger.info(
                    f"Triple barrier triggered: {symbol} "
                    f"barrier={result.barrier_type.value}, pnl={result.pnl_percentage:.2%}"
                )

        # 平仓
        for symbol, barrier_type, close_price in symbols_to_close:
            self._close_position_by_barrier(symbol, close_price, time, barrier_type)
            self.barrier_triggers[barrier_type.value] += 1

    def _close_position_by_barrier(
        self,
        symbol: str,
        price: float,
        time: datetime,
        barrier_type: BarrierType,
    ):
        """因屏障触发平仓"""
        position = self.account.positions.get(symbol)
        if not position or not position.is_open:
            return

        position_value = position.quantity * price

        # 计算滑点
        slippage = self._calculate_slippage(symbol, position_value)
        slippage_cost = position_value * slippage

        # 调整成交价格 (平多头减滑点，平空头加滑点)
        if position.side == PositionSide.LONG:
            actual_price = price * (1 - slippage)
        else:
            actual_price = price * (1 + slippage)

        # 计算手续费
        fee = self._calculate_fee(position_value, is_maker=False)

        # 执行平仓
        trade = position.close(
            close_price=actual_price,
            close_time=time,
            fee=fee,
            slippage=slippage_cost,
        )

        # 记录屏障类型
        trade.barrier_type = barrier_type.value

        # 更新账户
        self._finalize_position_close(position, trade)

        logger.info(
            f"Position closed by {barrier_type.value}: {symbol} "
            f"pnl={trade.pnl:.2f}, price={actual_price:.4f}"
        )

    def _process_signal(self, signal: Signal, time: datetime):
        """处理交易信号"""
        symbol = signal.symbol

        # 获取当前价格
        bar = self._get_bar_data(symbol, time)
        if not bar:
            logger.debug(f"No price data for {symbol} at {time}")
            return

        # 使用正确的价格语义 (回测成交使用 close_price)
        price_data = PriceData(
            close_price=bar.close,
            mark_price=bar.mark_price or bar.close,
            last_price=bar.close,
        )
        fill_price = self.price_semantics.get_price(
            PriceScenario.BACKTEST_FILL,
            price_data,
            fallback=True,
        )

        # 检查现有持仓
        existing_position = self.account.positions.get(symbol)

        if signal.is_close:
            # 平仓信号
            if existing_position and existing_position.is_open:
                self._close_position(symbol, fill_price, time, signal)
        elif signal.is_long:
            # 多头信号
            if existing_position and existing_position.is_open:
                if existing_position.side == PositionSide.SHORT:
                    # 先平空仓
                    self._close_position(symbol, fill_price, time, signal)
                else:
                    # 已有多仓，可以加仓或忽略
                    return

            self._open_position(symbol, PositionSide.LONG, fill_price, time, signal)

        elif signal.is_short:
            # 空头信号
            if existing_position and existing_position.is_open:
                if existing_position.side == PositionSide.LONG:
                    # 先平多仓
                    self._close_position(symbol, fill_price, time, signal)
                else:
                    # 已有空仓
                    return

            self._open_position(symbol, PositionSide.SHORT, fill_price, time, signal)

    def _open_position(
        self,
        symbol: str,
        side: PositionSide,
        price: float,
        time: datetime,
        signal: Signal,
    ):
        """开仓"""
        # 计算仓位大小 (基于可用资金和杠杆)
        position_value = self._calculate_position_size(symbol, signal.strength)
        if position_value <= 0:
            logger.debug(f"Position size too small for {symbol}")
            return

        quantity = position_value / price

        # 计算滑点
        slippage = self._calculate_slippage(symbol, position_value)
        slippage_cost = position_value * slippage

        # 调整成交价格 (多头加滑点，空头减滑点)
        if side == PositionSide.LONG:
            actual_price = price * (1 + slippage)
        else:
            actual_price = price * (1 - slippage)

        # 计算手续费 (市价单通常是 taker)
        fee = self._calculate_fee(position_value, is_maker=False)

        # 计算所需保证金
        margin = position_value / self.config.leverage

        # 检查可用资金
        if margin + fee + slippage_cost > self.account.free_margin:
            logger.debug(f"Insufficient margin for {symbol}")
            return

        # 创建持仓
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=actual_price,
            current_price=actual_price,
            leverage=self.config.leverage,
            entry_time=time,
            last_update_time=time,
            margin=margin,
            initial_margin=margin,
        )

        # 创建交易记录
        trade = Trade(
            symbol=symbol,
            trade_type=TradeType.OPEN_LONG if side == PositionSide.LONG else TradeType.OPEN_SHORT,
            side=side,
            quantity=quantity,
            price=actual_price,
            value=position_value,
            fee=fee,
            slippage=slippage_cost,
            timestamp=time,
            signal_id=id(signal),
            position_id=position.id,
        )

        # 更新账户
        self.account.balance -= (margin + fee + slippage_cost)
        self.account.used_margin += margin
        self.account.total_fees += fee
        self.account.total_slippage += slippage_cost
        self.account.positions[symbol] = position
        self.account.trades.append(trade)
        self.account.num_trades += 1

        # 记录
        position.trades.append(trade)
        position.total_fees = fee
        position.total_slippage = slippage_cost
        self.all_trades.append(trade)

        logger.debug(
            f"Opened {side.value} position: {symbol} "
            f"qty={quantity:.6f}, price={actual_price:.4f}, margin={margin:.2f}"
        )

    def _close_position(
        self,
        symbol: str,
        price: float,
        time: datetime,
        signal: Optional[Signal] = None,
    ):
        """平仓"""
        position = self.account.positions.get(symbol)
        if not position or not position.is_open:
            return

        position_value = position.quantity * price

        # 计算滑点
        slippage = self._calculate_slippage(symbol, position_value)
        slippage_cost = position_value * slippage

        # 调整成交价格 (平多头减滑点，平空头加滑点)
        if position.side == PositionSide.LONG:
            actual_price = price * (1 - slippage)
        else:
            actual_price = price * (1 + slippage)

        # 计算手续费
        fee = self._calculate_fee(position_value, is_maker=False)

        # 执行平仓
        trade = position.close(
            close_price=actual_price,
            close_time=time,
            fee=fee,
            slippage=slippage_cost,
        )

        if signal:
            trade.signal_id = id(signal)

        # 更新账户
        self._finalize_position_close(position, trade)

        logger.debug(
            f"Closed position: {symbol} "
            f"pnl={trade.pnl:.2f}, price={actual_price:.4f}"
        )

    def _finalize_position_close(self, position: Position, trade: Trade):
        """完成平仓后的账户更新"""
        # 返还保证金并计入盈亏
        returned_margin = position.initial_margin + trade.pnl

        self.account.balance += returned_margin
        self.account.used_margin -= position.initial_margin
        self.account.realized_pnl += trade.pnl
        self.account.total_fees += trade.fee
        self.account.total_slippage += trade.slippage

        # 统计
        if trade.pnl > 0:
            self.account.num_wins += 1
        elif trade.pnl < 0:
            self.account.num_losses += 1

        # 移动到已关闭持仓
        symbol = position.symbol
        if symbol in self.account.positions:
            del self.account.positions[symbol]
        self.account.closed_positions.append(position)
        self.all_positions.append(position)

        self.account.trades.append(trade)
        self.all_trades.append(trade)

    def _close_all_positions(self, time: datetime):
        """关闭所有持仓"""
        symbols = list(self.account.positions.keys())
        for symbol in symbols:
            bar = self._get_bar_data(symbol, time)
            if bar:
                self._close_position(symbol, bar.close, time)

    def _calculate_position_size(self, symbol: str, strength: float) -> float:
        """
        计算仓位大小

        Args:
            symbol: 交易对
            strength: 信号强度 (0-1)

        Returns:
            仓位价值 (USDT)
        """
        # 基于可用资金和信号强度
        available = self.account.free_margin * self.config.leverage

        # 限制单笔最大仓位
        max_position = self.config.max_position_size or (available * 0.2)

        # 基于信号强度调整仓位
        position_value = min(available * abs(strength), max_position)

        # 检查最小订单
        if position_value < self.config.min_order_value:
            return 0

        return position_value

    def _calculate_slippage(self, symbol: str, order_value: float) -> float:
        """计算滑点"""
        # 获取市场条件 (简化版)
        avg_volume = 10_000_000  # 默认日均成交量

        return self.slippage_model.get_slippage_for_backtest(
            symbol=symbol,
            order_size_usd=order_value,
            avg_daily_volume=avg_volume,
        )

    def _calculate_fee(self, order_value: float, is_maker: bool = False) -> float:
        """计算手续费"""
        return self.fee_model.calculate_fee(order_value, is_maker)

    def _update_equity_curve(self, time: datetime):
        """更新权益曲线"""
        equity = self.account.equity
        self.equity_curve.append((time, equity))

    def _create_empty_result(self) -> BacktestResult:
        """创建空结果"""
        return BacktestResult(
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            config=self.config.to_dict(),
        )

    # ===== 执行模型接口 (用于对齐验证) =====

    def get_fill_price_impl(self) -> str:
        """获取成交价格实现"""
        return self.config.execution_config.fill_price

    def get_partial_fill_impl(self) -> bool:
        """获取部分成交实现"""
        return self.config.execution_config.partial_fill

    def get_fee_model_impl(self) -> Dict[str, Any]:
        """获取费率模型实现"""
        return self.fee_model.get_fee_summary()

    def get_slippage_model_impl(self) -> Dict[str, Any]:
        """获取滑点模型实现"""
        return {
            "type": self.config.execution_config.slippage_model,
            "base": self.slippage_model.base_slippage,
            "max": self.slippage_model.max_slippage,
        }

    def get_position_mode_impl(self) -> str:
        """获取仓位模式实现"""
        return self.config.execution_config.position_mode.value

    def get_liquidation_logic_impl(self) -> Dict[str, Any]:
        """获取强平逻辑实现"""
        return {
            "use_mark_price": True,
            "maintenance_margin_rate": self.config.maintenance_margin_rate,
        }

    def get_leverage_handling_impl(self) -> Dict[str, Any]:
        """获取杠杆处理实现"""
        return {
            "max_leverage": self.config.max_leverage,
            "cross_margin": self.config.margin_mode.value == "cross",
        }


# Backward compatibility alias
BacktestEngine = CryptoPerpetualBacktest
