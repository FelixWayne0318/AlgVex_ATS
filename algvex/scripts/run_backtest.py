#!/usr/bin/env python3
"""
AlgVex 回测运行脚本

功能:
- 采集历史数据
- 计算因子
- 生成信号
- 运行回测
- 生成报告

使用方式:
    # 基本回测
    python scripts/run_backtest.py --symbol BTCUSDT --start 2024-01-01 --end 2024-06-30

    # 多标的回测
    python scripts/run_backtest.py --symbols BTCUSDT,ETHUSDT --start 2024-01-01

    # 自定义杠杆
    python scripts/run_backtest.py --symbol BTCUSDT --leverage 5
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtest import (
    BacktestConfig,
    CryptoPerpetualBacktest,
    BacktestResult,
)
from core.backtest.models import Signal
from core.data.collector import BinanceDataCollector
from production.factor_engine import MVPFactorEngine
from production.signal_generator import SignalGenerator


def run_backtest(
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    leverage: float = 3.0,
    top_k: int = 5,
    model_path: Optional[str] = None,
    verbose: bool = False,
) -> BacktestResult:
    """
    运行回测

    Args:
        symbols: 交易对列表
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
        leverage: 杠杆倍数
        top_k: Top-K 策略
        model_path: 模型路径
        verbose: 详细输出

    Returns:
        BacktestResult 回测结果
    """
    print(f"\n{'='*60}")
    print("AlgVex 回测引擎")
    print(f"{'='*60}")
    print(f"交易对: {symbols}")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"初始资金: ${initial_capital:,.0f}")
    print(f"杠杆: {leverage}x")

    # 1. 采集数据
    print("\n📊 Step 1: 采集数据...")
    collector = BinanceDataCollector(symbols=symbols)

    try:
        data = collector.collect_all(start_date, end_date, interval="1h")
        print(f"  K线数据: {len(data.get('klines', []))} 条")
        print(f"  资金费率: {len(data.get('funding', []))} 条")
        print(f"  持仓量: {len(data.get('oi', []))} 条")
    except Exception as e:
        print(f"  ⚠️ 数据采集失败: {e}")
        print("  使用模拟数据...")
        data = create_mock_data(symbols, start_date, end_date)

    # 2. 准备数据
    print("\n🔧 Step 2: 准备数据...")
    klines_data = {}
    if "klines" in data and not data["klines"].empty:
        for symbol in symbols:
            df = data["klines"][data["klines"]["symbol"] == symbol].copy()
            if not df.empty:
                df.set_index("datetime", inplace=True)
                klines_data[symbol] = df

    if not klines_data:
        print("  使用模拟K线数据...")
        mock_data = create_mock_data(symbols, start_date, end_date)
        for symbol in symbols:
            df = mock_data["klines"][mock_data["klines"]["symbol"] == symbol].copy()
            if not df.empty:
                df.set_index("datetime", inplace=True)
                klines_data[symbol] = df

    print(f"  有效标的: {list(klines_data.keys())}")

    # 3. 计算因子
    print("\n📈 Step 3: 计算因子...")
    factor_engine = MVPFactorEngine()
    all_factors = {}

    for symbol, klines in klines_data.items():
        factors = factor_engine.compute_all_factors(
            klines=klines,
            signal_time=datetime.strptime(end_date, "%Y-%m-%d"),
        )
        all_factors[symbol] = {
            k: v.value for k, v in factors.items() if v.is_valid
        }
        print(f"  {symbol}: {len(all_factors[symbol])} 个有效因子")

    # 4. 生成信号
    print("\n🎯 Step 4: 生成信号...")
    signal_generator = SignalGenerator(
        factor_engine=factor_engine,
        enable_trace=False,
    )

    # 为每个时间点生成信号
    signals = generate_signals_for_backtest(
        klines_data=klines_data,
        signal_generator=signal_generator,
        top_k=top_k,
    )

    print(f"  生成信号: {len(signals)} 个")
    if verbose and signals:
        for signal in signals[:5]:
            print(f"    {signal.symbol}: {signal.signal_type} (强度: {signal.strength:.4f})")
        if len(signals) > 5:
            print(f"    ... 还有 {len(signals) - 5} 个信号")

    # 5. 创建回测配置
    print("\n⚙️ Step 5: 配置回测引擎...")
    config = BacktestConfig(
        initial_capital=initial_capital,
        leverage=leverage,
        max_leverage=10.0,
        taker_fee=0.0004,
        maker_fee=0.0002,
        slippage=0.0001,
        enable_funding=True,
        frequency="1h",
        verbose=verbose,
    )

    # 6. 运行回测
    print("\n🚀 Step 6: 运行回测...")
    engine = CryptoPerpetualBacktest(config)

    # 准备资金费率数据
    funding_rates = prepare_funding_rates(data, symbols)

    # 运行回测
    result = engine.run(
        signals=signals,
        prices=klines_data,
        funding_rates=funding_rates,
    )

    # 7. 输出报告
    print(result.get_summary())

    return result


def generate_signals_for_backtest(
    klines_data: Dict[str, pd.DataFrame],
    signal_generator: SignalGenerator,
    top_k: int = 5,
) -> List[Signal]:
    """
    为回测生成信号序列

    Args:
        klines_data: K线数据
        signal_generator: 信号生成器
        top_k: 选取前 K 个

    Returns:
        信号列表
    """
    signals = []

    # 获取所有时间点
    all_times = set()
    for df in klines_data.values():
        all_times.update(df.index.to_pydatetime())

    all_times = sorted(all_times)

    # 每隔一定周期生成信号 (例如每24根K线)
    signal_interval = 24  # 每24小时生成一次信号

    for i, signal_time in enumerate(all_times):
        if i % signal_interval != 0:
            continue

        if i < 100:  # 需要足够的历史数据
            continue

        try:
            raw_signals = signal_generator.generate(
                symbols=list(klines_data.keys()),
                klines_data=klines_data,
                signal_time=signal_time,
            )

            # 转换为 Signal 对象
            for raw_signal in raw_signals:
                signal = Signal(
                    symbol=raw_signal.symbol,
                    signal_type=raw_signal.signal_type.value,
                    strength=raw_signal.strength,
                    timestamp=signal_time,
                    price=raw_signal.entry_price,
                )
                signals.append(signal)

        except Exception as e:
            continue

    return signals


def prepare_funding_rates(
    data: Dict,
    symbols: List[str],
) -> Dict[str, Dict[datetime, float]]:
    """准备资金费率数据"""
    funding_rates = {}

    if "funding" not in data or data["funding"].empty:
        return funding_rates

    funding_df = data["funding"]

    for symbol in symbols:
        symbol_funding = funding_df[funding_df["symbol"] == symbol]
        if symbol_funding.empty:
            continue

        rates = {}
        for _, row in symbol_funding.iterrows():
            time = pd.to_datetime(row["funding_time"])
            if time.tzinfo is None:
                time = time.replace(tzinfo=timezone.utc)
            rate = float(row["funding_rate"])
            rates[time] = rate

        funding_rates[symbol] = rates

    return funding_rates


def create_mock_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> Dict:
    """创建模拟数据"""
    dates = pd.date_range(start_date, end_date, freq="1h")
    n = len(dates)

    all_klines = []
    all_funding = []

    for symbol in symbols:
        np.random.seed(hash(symbol) % 2**32)

        # 模拟价格走势
        base_price = 50000 if "BTC" in symbol else 3000
        returns = np.random.randn(n) * 0.002
        prices = base_price * np.cumprod(1 + returns)

        klines = pd.DataFrame({
            "datetime": dates,
            "symbol": symbol,
            "open": prices * (1 + np.random.randn(n) * 0.001),
            "high": prices * (1 + abs(np.random.randn(n)) * 0.002),
            "low": prices * (1 - abs(np.random.randn(n)) * 0.002),
            "close": prices,
            "volume": 1000000 + np.random.randint(0, 500000, n),
        })
        klines["high"] = klines[["open", "high", "close"]].max(axis=1)
        klines["low"] = klines[["open", "low", "close"]].min(axis=1)
        all_klines.append(klines)

        # 模拟资金费率 (每8小时)
        funding_times = pd.date_range(start_date, end_date, freq="8h")
        funding = pd.DataFrame({
            "symbol": symbol,
            "funding_time": funding_times,
            "funding_rate": np.random.randn(len(funding_times)) * 0.0001,
        })
        all_funding.append(funding)

    return {
        "klines": pd.concat(all_klines, ignore_index=True),
        "funding": pd.concat(all_funding, ignore_index=True),
    }


def main():
    parser = argparse.ArgumentParser(description="AlgVex 回测")
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="单个交易对",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT",
        help="多个交易对 (逗号分隔)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="开始日期",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="结束日期",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="初始资金",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=3.0,
        help="杠杆倍数",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K 策略",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()

    if args.symbol:
        symbols = [args.symbol]
    else:
        symbols = [s.strip() for s in args.symbols.split(",")]

    result = run_backtest(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        leverage=args.leverage,
        top_k=args.top_k,
        model_path=args.model,
        verbose=args.verbose,
    )

    # 返回结果 (可用于编程调用)
    return result


if __name__ == "__main__":
    main()
