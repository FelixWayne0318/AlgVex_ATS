#!/usr/bin/env python
"""
回测脚本

用法:
    python scripts/backtest.py
    python scripts/backtest.py --start 2023-01-01 --end 2024-01-01
    python scripts/backtest.py --symbol btcusdt --leverage 3
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from qlib_adapter.data_handler import CryptoDataHandler
from qlib_adapter.feature_engine import CryptoFeatureEngine
from qlib_adapter.backtest_engine import CryptoPerpetualBacktest
from strategy.ml_strategy import MLStrategy
from strategy.signal_generator import SignalGenerator, SignalType


def setup_logging(log_level: str = "INFO"):
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level=log_level,
    )


def load_symbols() -> list:
    """加载标的列表"""
    symbols_path = Path(__file__).parent.parent / "config" / "symbols.yaml"
    if symbols_path.exists():
        with open(symbols_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return [s["symbol"] for s in config.get("primary", []) if s.get("enabled", True)]
    return ["btcusdt", "ethusdt"]


def run_backtest(
    symbols: list,
    start_date: str,
    end_date: str,
    initial_capital: float = 100000,
    leverage: float = 3.0,
    long_threshold: float = 0.6,
    short_threshold: float = 0.4,
    apply_funding: bool = True,
    data_dir: str = None,
    output_dir: str = None,
):
    """
    运行回测

    Args:
        symbols: 标的列表
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
        leverage: 杠杆
        long_threshold: 做多阈值
        short_threshold: 做空阈值
        apply_funding: 是否应用资金费率
        data_dir: 数据目录
        output_dir: 输出目录
    """
    if data_dir is None:
        data_dir = Path("~/.cryptoquant/data").expanduser()
    else:
        data_dir = Path(data_dir)

    if output_dir is None:
        output_dir = Path("~/.cryptoquant/backtest").expanduser()
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CryptoQuant Backtest")
    logger.info("=" * 60)
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Initial capital: ${initial_capital:,.2f}")
    logger.info(f"Leverage: {leverage}x")
    logger.info(f"Long threshold: {long_threshold}")
    logger.info(f"Short threshold: {short_threshold}")
    logger.info(f"Apply funding: {apply_funding}")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("\n1. Loading data...")
    data_handler = CryptoDataHandler(data_dir=str(data_dir))

    all_data = {}
    for symbol in symbols:
        try:
            df = data_handler.get_qlib_data(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                freq="1h",
            )
            if not df.empty:
                all_data[symbol] = df
                logger.info(f"  {symbol}: {len(df)} records")
        except Exception as e:
            logger.warning(f"  {symbol}: Failed to load - {e}")

    if not all_data:
        logger.error("No data loaded!")
        return

    # 2. 计算特征
    logger.info("\n2. Calculating features...")
    feature_engine = CryptoFeatureEngine()

    for symbol in list(all_data.keys()):
        try:
            df_features = feature_engine.calculate_all_features(
                all_data[symbol],
                groups=["momentum", "volatility", "volume", "funding", "oi", "ls_ratio"],
            )
            all_data[symbol] = df_features
            logger.info(f"  {symbol}: {len(df_features.columns)} features")
        except Exception as e:
            logger.warning(f"  {symbol}: Feature calculation failed - {e}")
            del all_data[symbol]

    # 3. 加载模型并生成预测
    logger.info("\n3. Loading models and generating predictions...")
    strategy = MLStrategy()
    signal_gen = SignalGenerator(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
    )

    predictions = {}
    for symbol in all_data.keys():
        if strategy.load_model(symbol):
            # 准备特征
            df = all_data[symbol]
            feature_cols = [c for c in df.columns if c not in ["$close", "$open", "$high", "$low", "$volume", "label"]]

            try:
                pred_class, pred_proba = strategy.predict(df[feature_cols].dropna(), symbol)
                predictions[symbol] = pd.Series(pred_proba, index=df[feature_cols].dropna().index)
                logger.info(f"  {symbol}: Generated {len(pred_proba)} predictions")
            except Exception as e:
                logger.warning(f"  {symbol}: Prediction failed - {e}")
        else:
            logger.warning(f"  {symbol}: Model not found, using random signals")
            # 使用随机信号作为演示
            df = all_data[symbol]
            np.random.seed(42)
            predictions[symbol] = pd.Series(np.random.random(len(df)), index=df.index)

    # 4. 运行回测
    logger.info("\n4. Running backtest...")
    backtest = CryptoPerpetualBacktest(
        initial_capital=initial_capital,
        leverage=leverage,
        fee_rate=0.0004,
    )

    # 获取价格数据
    prices = {}
    funding_rates = {}
    for symbol, df in all_data.items():
        if "$close" in df.columns:
            prices[symbol] = df["$close"]
        elif "close" in df.columns:
            prices[symbol] = df["close"]

        if "$funding_rate" in df.columns:
            funding_rates[symbol] = df["$funding_rate"]
        elif "funding_rate" in df.columns:
            funding_rates[symbol] = df["funding_rate"]

    # 生成信号
    signals = {}
    for symbol in predictions.keys():
        symbol_signals = []
        proba_series = predictions[symbol]

        for idx, proba in proba_series.items():
            signal = signal_gen.generate_signal(
                symbol=symbol,
                probability=proba,
                current_position=0,  # 简化处理
                market_data={},
            )
            symbol_signals.append({
                "time": idx if isinstance(idx, datetime) else idx[0],
                "signal": signal.target_position,
            })

        signals[symbol] = pd.DataFrame(symbol_signals).set_index("time")["signal"]

    # 执行回测
    equity_curve = backtest.run(
        signals=signals,
        prices=prices,
        funding_rates=funding_rates if apply_funding else None,
    )

    # 5. 计算绩效指标
    logger.info("\n5. Calculating performance metrics...")
    metrics = backtest.get_metrics()

    logger.info("\nPerformance Summary:")
    logger.info("-" * 40)
    logger.info(f"Total Return: {metrics['total_return']*100:.2f}%")
    logger.info(f"Annual Return: {metrics.get('annual_return', 0)*100:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
    logger.info(f"Total Trades: {metrics['total_trades']}")
    logger.info(f"Total PnL: ${metrics['total_pnl']:,.2f}")
    logger.info(f"Total Funding Paid: ${metrics.get('total_funding', 0):,.2f}")
    logger.info(f"Final Equity: ${metrics['final_equity']:,.2f}")
    logger.info("-" * 40)

    # 6. 保存结果
    logger.info("\n6. Saving results...")

    # 保存权益曲线
    equity_path = output_dir / f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    equity_curve.to_csv(equity_path)
    logger.info(f"  Equity curve saved to: {equity_path}")

    # 保存交易记录
    trades = backtest.get_trade_history()
    if trades:
        trades_df = pd.DataFrame(trades)
        trades_path = output_dir / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"  Trades saved to: {trades_path}")

    # 保存指标
    metrics_path = output_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    logger.info(f"  Metrics saved to: {metrics_path}")

    # 7. 绘制图表
    logger.info("\n7. Generating charts...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 权益曲线
    ax1 = axes[0]
    equity_curve.plot(ax=ax1, linewidth=1)
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Equity ($)")
    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # 回撤
    ax2 = axes[1]
    drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
    drawdown.plot(ax=ax2, color='red', linewidth=1)
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)

    # 月度收益
    ax3 = axes[2]
    if len(equity_curve) > 30:
        monthly_returns = equity_curve.resample("ME").last().pct_change().dropna() * 100
        colors = ['green' if x >= 0 else 'red' for x in monthly_returns.values]
        monthly_returns.plot(kind='bar', ax=ax3, color=colors)
        ax3.set_title("Monthly Returns")
        ax3.set_ylabel("Return (%)")
        ax3.set_xticklabels([d.strftime('%Y-%m') for d in monthly_returns.index], rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    chart_path = output_dir / f"backtest_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"  Chart saved to: {chart_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Backtest completed!")
    logger.info("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CryptoQuant Backtest")

    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (default: from config)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2023-01-01",
        help="Start date",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date",
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
        "--long-threshold",
        type=float,
        default=0.6,
        help="Long signal threshold",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        default=0.4,
        help="Short signal threshold",
    )
    parser.add_argument(
        "--no-funding",
        action="store_true",
        help="Disable funding rate",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.symbols:
        symbols = [s.strip().lower() for s in args.symbols.split(",")]
    else:
        symbols = load_symbols()

    run_backtest(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        initial_capital=args.capital,
        leverage=args.leverage,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        apply_funding=not args.no_funding,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
