#!/usr/bin/env python
"""
实盘交易脚本

用法:
    python scripts/live_trade.py --paper    # 模拟盘
    python scripts/live_trade.py            # 实盘 (谨慎!)
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from data_collector.binance_collector import BinanceDataCollector
from qlib_adapter.feature_engine import CryptoFeatureEngine
from strategy.ml_strategy import MLStrategy
from strategy.signal_generator import SignalGenerator, MultiSymbolSignalGenerator, SignalType
from execution.adapter import SignalToTradeAdapter, TradeInstruction
from execution.risk_manager import RiskManager
from execution.position_manager import PositionManager


def setup_logging(log_level: str = "INFO", log_file: bool = True):
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level=log_level,
        colorize=True,
    )

    if log_file:
        log_path = Path("~/.cryptoquant/logs").expanduser()
        log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path / "live_trade_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level=log_level,
        )


def load_config() -> dict:
    """加载配置"""
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def load_symbols() -> list:
    """加载标的列表"""
    symbols_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
    if symbols_path.exists():
        with open(symbols_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return [s["symbol"] for s in config.get("primary", []) if s.get("enabled", True)]
    return ["btcusdt", "ethusdt"]


class LiveTrader:
    """
    实盘交易器

    整合数据采集、特征计算、信号生成、风控、执行
    """

    def __init__(
        self,
        symbols: List[str],
        paper_mode: bool = True,
        initial_capital: float = 100000,
        leverage: float = 3.0,
        config: dict = None,
    ):
        self.symbols = symbols
        self.paper_mode = paper_mode
        self.config = config or {}

        # 初始化组件
        self.data_collector = BinanceDataCollector()
        self.feature_engine = CryptoFeatureEngine()
        self.ml_strategy = MLStrategy()
        self.signal_generator = MultiSymbolSignalGenerator(
            symbols=symbols,
            max_positions=self.config.get("strategy", {}).get("max_positions", 5),
            long_threshold=self.config.get("strategy", {}).get("long_threshold", 0.6),
            short_threshold=self.config.get("strategy", {}).get("short_threshold", 0.4),
        )
        self.risk_manager = RiskManager(
            max_position_pct=self.config.get("risk", {}).get("max_position_pct", 0.2),
            max_leverage=self.config.get("risk", {}).get("max_total_leverage", 5.0),
            stop_loss_pct=self.config.get("risk", {}).get("stop_loss_pct", 0.05),
            max_daily_loss_pct=self.config.get("risk", {}).get("max_daily_loss_pct", 0.1),
        )
        self.position_manager = PositionManager(initial_capital=initial_capital)
        self.trade_adapter = SignalToTradeAdapter(
            account_equity=initial_capital,
            max_position_pct=self.config.get("risk", {}).get("max_position_pct", 0.2),
            leverage=leverage,
        )

        # 加载模型
        self._load_models()

        # 状态
        self.running = False
        self.last_rebalance = None
        self.daily_pnl = 0

        logger.info(f"LiveTrader initialized (paper_mode={paper_mode})")
        logger.info(f"Symbols: {symbols}")

    def _load_models(self):
        """加载所有模型"""
        for symbol in self.symbols:
            if not self.ml_strategy.load_model(symbol):
                logger.warning(f"Model not found for {symbol}, will use fallback signals")

    async def fetch_latest_data(self, symbol: str, lookback_hours: int = 200) -> pd.DataFrame:
        """获取最新数据"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=lookback_hours)

        try:
            # 获取K线
            klines = self.data_collector.fetch_klines(
                symbol=symbol,
                interval="1h",
                start_time=int(start_time.timestamp() * 1000),
                end_time=int(end_time.timestamp() * 1000),
            )

            if klines.empty:
                return pd.DataFrame()

            # 获取资金费率
            funding = self.data_collector.fetch_funding_rate(
                symbol=symbol,
                start_time=int(start_time.timestamp() * 1000),
                end_time=int(end_time.timestamp() * 1000),
            )

            # 获取持仓量
            oi = self.data_collector.fetch_open_interest_history(
                symbol=symbol,
                period="1h",
                start_time=int(start_time.timestamp() * 1000),
                end_time=int(end_time.timestamp() * 1000),
            )

            # 获取多空比
            ls_ratio = self.data_collector.fetch_long_short_ratio(
                symbol=symbol,
                period="1h",
                limit=lookback_hours,
            )

            # 合并数据
            df = klines.copy()
            df.index = pd.to_datetime(df.index)

            if not funding.empty:
                funding.index = pd.to_datetime(funding.index)
                df = df.merge(funding[["funding_rate"]], left_index=True, right_index=True, how="left")

            if not oi.empty:
                oi.index = pd.to_datetime(oi.index)
                df = df.merge(oi[["open_interest"]], left_index=True, right_index=True, how="left")

            if not ls_ratio.empty:
                ls_ratio.index = pd.to_datetime(ls_ratio.index)
                df = df.merge(ls_ratio[["long_short_ratio"]], left_index=True, right_index=True, how="left")

            # 前向填充
            df = df.ffill()

            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()

    def generate_prediction(self, symbol: str, df: pd.DataFrame) -> Optional[float]:
        """生成预测"""
        try:
            # 计算特征
            df_features = self.feature_engine.calculate_all_features(
                df,
                groups=["momentum", "volatility", "volume", "funding", "oi", "ls_ratio"],
            )

            if df_features.empty:
                return None

            # 准备特征
            feature_cols = [c for c in df_features.columns
                          if c not in ["close", "open", "high", "low", "volume", "label"]]

            # 取最新一行
            X = df_features[feature_cols].iloc[[-1]].dropna(axis=1)

            if X.empty:
                return None

            # 预测
            if symbol in self.ml_strategy.models:
                _, pred_proba = self.ml_strategy.predict(X, symbol)
                return float(pred_proba[0])
            else:
                # 使用随机信号
                return np.random.random()

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None

    async def execute_trade(self, instruction: TradeInstruction) -> bool:
        """执行交易"""
        if self.paper_mode:
            # 模拟执行
            logger.info(f"[PAPER] Executing: {instruction}")

            price = instruction.price
            if instruction.action in ["open_long", "close_short"]:
                price *= 1.001  # 模拟滑点
            else:
                price *= 0.999

            # 更新仓位管理器
            if instruction.action == "open_long":
                self.position_manager.open_position(
                    instruction.symbol,
                    instruction.quantity,
                    price,
                    instruction.leverage,
                )
            elif instruction.action == "open_short":
                self.position_manager.open_position(
                    instruction.symbol,
                    -instruction.quantity,
                    price,
                    instruction.leverage,
                )
            elif instruction.action in ["close_long", "close_short"]:
                self.position_manager.close_position(
                    instruction.symbol,
                    instruction.quantity,
                    price,
                )

            return True
        else:
            # 实盘执行 - 需要集成 ValueCell 或 CCXT
            logger.warning("Live trading not implemented yet!")
            return False

    async def rebalance(self):
        """再平衡"""
        logger.info("Starting rebalance cycle...")

        # 1. 获取当前仓位
        current_positions = {
            symbol: pos.amount / self.position_manager.total_equity
            for symbol, pos in self.position_manager.positions.items()
        }

        # 2. 获取最新数据和预测
        predictions = {}
        prices = {}
        market_data = {}

        for symbol in self.symbols:
            df = await self.fetch_latest_data(symbol)
            if df.empty:
                continue

            # 当前价格
            prices[symbol] = float(df["close"].iloc[-1])

            # 预测
            pred = self.generate_prediction(symbol, df)
            if pred is not None:
                predictions[symbol] = pred

            # 市场数据
            market_data[symbol] = {
                "funding_rate": float(df.get("funding_rate", pd.Series([0])).iloc[-1]),
                "volatility": float(df["close"].pct_change().std()) if len(df) > 1 else 0,
                "long_short_ratio": float(df.get("long_short_ratio", pd.Series([1])).iloc[-1]),
            }

        if not predictions:
            logger.warning("No predictions generated, skipping rebalance")
            return

        # 3. 生成信号
        signals = self.signal_generator.generate_all_signals(
            predictions=predictions,
            current_positions=current_positions,
            market_data=market_data,
        )

        # 4. 选择最佳信号
        top_signals = self.signal_generator.select_top_signals(signals)

        logger.info(f"Generated signals: {len(signals)}, selected: {len(top_signals)}")

        # 5. 更新价格
        self.position_manager.update_prices(prices)

        # 6. 检查止损
        for symbol in list(self.position_manager.positions.keys()):
            if symbol in prices:
                if self.risk_manager.check_stop_loss(symbol, prices[symbol]):
                    logger.warning(f"Stop loss triggered for {symbol}")
                    # 强制平仓
                    pos = self.position_manager.positions[symbol]
                    instruction = TradeInstruction(
                        symbol=symbol,
                        action="close_long" if pos.amount > 0 else "close_short",
                        quantity=abs(pos.amount),
                        price=prices[symbol],
                        leverage=pos.leverage,
                    )
                    await self.execute_trade(instruction)

        # 7. 生成交易指令
        for symbol, signal in top_signals.items():
            if symbol not in prices:
                continue

            # 计算目标仓位
            target_position = signal.target_position
            current_position = current_positions.get(symbol, 0)

            if abs(target_position - current_position) < 0.05:
                continue  # 变化太小，不交易

            # 确定交易动作
            if target_position > current_position:
                if current_position < 0:
                    # 先平空
                    instruction = TradeInstruction(
                        symbol=symbol,
                        action="close_short",
                        quantity=abs(current_position) * self.position_manager.total_equity / prices[symbol],
                        price=prices[symbol],
                        leverage=self.trade_adapter.leverage,
                    )
                    approved, msg = self.risk_manager.check_trade(instruction, prices[symbol])
                    if approved:
                        await self.execute_trade(instruction)
                    else:
                        logger.warning(f"Trade rejected: {msg}")

                if target_position > 0:
                    # 开多
                    instruction = TradeInstruction(
                        symbol=symbol,
                        action="open_long",
                        quantity=target_position * self.position_manager.total_equity / prices[symbol],
                        price=prices[symbol],
                        leverage=self.trade_adapter.leverage,
                    )
                    approved, msg = self.risk_manager.check_trade(instruction, prices[symbol])
                    if approved:
                        await self.execute_trade(instruction)
                    else:
                        logger.warning(f"Trade rejected: {msg}")

            elif target_position < current_position:
                if current_position > 0:
                    # 先平多
                    instruction = TradeInstruction(
                        symbol=symbol,
                        action="close_long",
                        quantity=current_position * self.position_manager.total_equity / prices[symbol],
                        price=prices[symbol],
                        leverage=self.trade_adapter.leverage,
                    )
                    approved, msg = self.risk_manager.check_trade(instruction, prices[symbol])
                    if approved:
                        await self.execute_trade(instruction)
                    else:
                        logger.warning(f"Trade rejected: {msg}")

                if target_position < 0:
                    # 开空
                    instruction = TradeInstruction(
                        symbol=symbol,
                        action="open_short",
                        quantity=abs(target_position) * self.position_manager.total_equity / prices[symbol],
                        price=prices[symbol],
                        leverage=self.trade_adapter.leverage,
                    )
                    approved, msg = self.risk_manager.check_trade(instruction, prices[symbol])
                    if approved:
                        await self.execute_trade(instruction)
                    else:
                        logger.warning(f"Trade rejected: {msg}")

        # 8. 输出状态
        self._print_status()

        self.last_rebalance = datetime.now()

    def _print_status(self):
        """打印状态"""
        logger.info("-" * 60)
        logger.info("Current Status:")
        logger.info(f"  Equity: ${self.position_manager.total_equity:,.2f}")
        logger.info(f"  Cash: ${self.position_manager.cash:,.2f}")
        logger.info(f"  Unrealized PnL: ${self.position_manager.total_unrealized_pnl:,.2f}")

        if self.position_manager.positions:
            logger.info("  Positions:")
            for symbol, pos in self.position_manager.positions.items():
                logger.info(f"    {symbol}: {pos.side} {abs(pos.amount):.4f} @ {pos.entry_price:.2f}, "
                           f"PnL: ${pos.unrealized_pnl:,.2f} ({pos.pnl_pct*100:.2f}%)")
        else:
            logger.info("  Positions: None")

        logger.info("-" * 60)

    async def run(self, rebalance_interval: int = 3600):
        """
        运行交易循环

        Args:
            rebalance_interval: 再平衡间隔 (秒)
        """
        self.running = True
        logger.info(f"Starting live trader (interval={rebalance_interval}s)")

        try:
            while self.running:
                try:
                    await self.rebalance()
                except Exception as e:
                    logger.error(f"Rebalance error: {e}")

                # 等待下次再平衡
                await asyncio.sleep(rebalance_interval)

        except KeyboardInterrupt:
            logger.info("Trader interrupted")
        finally:
            self.running = False
            logger.info("Trader stopped")

    def stop(self):
        """停止交易"""
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="CryptoQuant Live Trader")

    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Paper trading mode",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="Leverage",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Rebalance interval (seconds)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    # 安全检查
    if not args.paper:
        logger.warning("=" * 60)
        logger.warning("WARNING: You are about to run in LIVE mode!")
        logger.warning("This will execute real trades with real money!")
        logger.warning("Make sure you understand the risks.")
        logger.warning("=" * 60)
        confirm = input("Type 'CONFIRM' to proceed: ")
        if confirm != "CONFIRM":
            logger.info("Aborted.")
            return

    config = load_config()

    if args.symbols:
        symbols = [s.strip().lower() for s in args.symbols.split(",")]
    else:
        symbols = load_symbols()

    trader = LiveTrader(
        symbols=symbols,
        paper_mode=args.paper,
        initial_capital=args.capital,
        leverage=args.leverage,
        config=config,
    )

    asyncio.run(trader.run(rebalance_interval=args.interval))


if __name__ == "__main__":
    main()
