"""
离线回测脚本 (v10.0.0)

与实盘使用完全相同的链路:
OHLCV → unified_features → normalizer → booster → signal → 仿真

用法:
    python scripts/backtest_offline.py --instruments btcusdt --test-start 2024-07-01

重要:
    - 使用已收盘bar生成信号 (与实盘一致)
    - 严格特征对齐 (缺列直接FAIL)
    - 支持手续费、滑点、止损止盈、时间限制、冷却
"""

import json
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import lightgbm as lgb

# 导入统一特征模块 (兼容不同运行目录)
try:
    from scripts.unified_features import (
        compute_unified_features,
        FeatureNormalizer,
        FEATURE_COLUMNS,
    )
except ImportError:
    from unified_features import (
        compute_unified_features,
        FeatureNormalizer,
        FEATURE_COLUMNS,
    )


@dataclass
class BacktestConfig:
    """回测配置 (与实盘 controller 配置一致)"""
    signal_threshold: float = 0.001
    stop_loss: float = 0.02
    take_profit: float = 0.03
    time_limit_hours: int = 24
    cooldown_bars: int = 1  # 同一根bar不重复交易
    fee_rate: float = 0.001  # 0.1% 手续费
    slippage: float = 0.0005  # 0.05% 滑点


@dataclass
class Position:
    """持仓"""
    side: int  # +1=long, -1=short
    entry_price: float
    entry_bar: int
    size: float = 1.0


def load_model(model_dir: Path):
    """加载模型和归一化器"""
    model = lgb.Booster(model_file=str(model_dir / "lgb_model.txt"))
    normalizer = FeatureNormalizer()
    normalizer.load(str(model_dir / "normalizer.pkl"))

    with open(model_dir / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

    return model, normalizer, feature_columns


def run_backtest(
    data_dir: str,
    model_dir: str,
    instruments: list,
    test_start: str,
    test_end: str,
    freq: str = "1h",
    config: BacktestConfig = None,
):
    """运行离线回测"""

    if config is None:
        config = BacktestConfig()

    data_path = Path(data_dir).expanduser()
    model_path = Path(model_dir).expanduser()

    # 加载模型
    print(f"Loading model from {model_path}...")
    model, normalizer, feature_columns = load_model(model_path)

    # 验证特征列 (v10.0.2: 更健壮的比较)
    if set(feature_columns) != set(FEATURE_COLUMNS):
        raise ValueError("Feature columns mismatch! Retrain model.")
    if feature_columns != FEATURE_COLUMNS:
        import warnings
        warnings.warn("Feature column order differs from code, using model's order")
        # 使用模型训练时的列顺序，不影响预测结果

    # 解析时间
    test_start_ts = pd.Timestamp(test_start, tz="UTC")
    test_end_ts = pd.Timestamp(test_end, tz="UTC")

    all_results = []

    for inst in instruments:
        print(f"\nBacktesting {inst}...")

        # 加载数据
        file_path = data_path / freq / f"{inst}.parquet"
        if not file_path.exists():
            print(f"  Skipping {inst}: file not found")
            continue

        df = pd.read_parquet(file_path)
        df = df[(df.index >= test_start_ts) & (df.index <= test_end_ts)]

        if len(df) < 61:  # MIN_BARS
            print(f"  Skipping {inst}: insufficient data ({len(df)} bars)")
            continue

        # 计算特征
        features = compute_unified_features(df)

        # 严格对齐 (缺列直接FAIL)
        missing = set(feature_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        features = features[feature_columns]

        # v10.0.4: 剔除滚动窗口 NaN (前 60 行)，避免 strict=True 报错
        valid_mask = ~features.isna().any(axis=1)
        features_valid = features[valid_mask]
        df_valid = df.loc[features_valid.index]  # 同步裁剪 df

        if len(features_valid) < 10:
            print(f"  Skipping {inst}: insufficient valid features ({len(features_valid)})")
            continue

        # 归一化 (strict=True) - 此时已无 NaN
        features_norm = normalizer.transform(features_valid, strict=True)

        # 预测
        predictions = model.predict(features_norm.values)

        # 仿真交易 (使用裁剪后的 df_valid，predictions 已对齐)
        result = simulate_trading(
            df_valid, predictions, config, inst
        )
        all_results.append(result)

    # 汇总结果
    if all_results:
        print_summary(all_results, config)

    return all_results


def simulate_trading(
    df: pd.DataFrame,
    predictions: np.ndarray,
    config: BacktestConfig,
    instrument: str,
) -> dict:
    """仿真交易逻辑 (与实盘信号规则一致)"""

    closes = df["close"].values
    n = len(closes)

    position: Optional[Position] = None
    trades = []
    equity = [1.0]
    last_trade_bar = -config.cooldown_bars

    # 从第 61 根bar开始 (需要 60 根历史)
    for i in range(60, n - 1):
        current_price = closes[i]

        # 使用已收盘bar的预测 (与实盘一致: iloc[-2])
        pred = predictions[i]

        # 检查是否需要平仓
        if position is not None:
            entry_price = position.entry_price
            bars_held = i - position.entry_bar

            # 计算收益
            if position.side == 1:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            # 止损/止盈/时间限制
            should_close = False
            close_reason = ""

            if pnl_pct <= -config.stop_loss:
                should_close = True
                close_reason = "stop_loss"
            elif pnl_pct >= config.take_profit:
                should_close = True
                close_reason = "take_profit"
            elif bars_held >= config.time_limit_hours:
                should_close = True
                close_reason = "time_limit"

            if should_close:
                # 扣除手续费和滑点
                exit_price = current_price * (1 - config.slippage * position.side)
                fee = config.fee_rate * 2  # 开仓+平仓

                if position.side == 1:
                    net_pnl = (exit_price - entry_price) / entry_price - fee
                else:
                    net_pnl = (entry_price - exit_price) / entry_price - fee

                trades.append({
                    "instrument": instrument,
                    "side": position.side,
                    "entry_bar": position.entry_bar,
                    "exit_bar": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": net_pnl,
                    "reason": close_reason,
                })

                equity.append(equity[-1] * (1 + net_pnl))
                position = None
                last_trade_bar = i

        # 检查是否开仓
        if position is None and (i - last_trade_bar) >= config.cooldown_bars:
            signal = 0
            if pred > config.signal_threshold:
                signal = 1
            elif pred < -config.signal_threshold:
                signal = -1

            if signal != 0:
                entry_price = current_price * (1 + config.slippage * signal)
                position = Position(
                    side=signal,
                    entry_price=entry_price,
                    entry_bar=i,
                )

    # 强制平仓
    if position is not None:
        exit_price = closes[-1]
        if position.side == 1:
            net_pnl = (exit_price - position.entry_price) / position.entry_price - config.fee_rate * 2
        else:
            net_pnl = (position.entry_price - exit_price) / position.entry_price - config.fee_rate * 2

        trades.append({
            "instrument": instrument,
            "side": position.side,
            "entry_bar": position.entry_bar,
            "exit_bar": n - 1,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "pnl_pct": net_pnl,
            "reason": "end_of_test",
        })
        equity.append(equity[-1] * (1 + net_pnl))

    # 计算指标
    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1] if len(equity) > 1 else np.array([])

    return {
        "instrument": instrument,
        "trades": trades,
        "equity": equity,
        "returns": returns,
    }


def print_summary(results: list, config: BacktestConfig):
    """打印回测汇总"""

    all_trades = []
    for r in results:
        all_trades.extend(r["trades"])

    if not all_trades:
        print("\n⚠️ No trades executed!")
        return

    df = pd.DataFrame(all_trades)

    # 基础统计
    total_trades = len(df)
    win_trades = (df["pnl_pct"] > 0).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0

    total_pnl = df["pnl_pct"].sum()
    avg_pnl = df["pnl_pct"].mean()

    # 年化 (假设1h bar, 8760 bars/year)
    periods_per_year = 8760
    if len(results[0]["returns"]) > 0:
        returns = np.concatenate([r["returns"] for r in results])
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(periods_per_year)

        # 最大回撤
        equity = np.concatenate([r["equity"] for r in results])
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = abs(drawdown.min())
    else:
        sharpe = 0.0
        max_dd = 0.0

    print("\n" + "=" * 60)
    print("回测结果 (Backtest Results)")
    print("=" * 60)
    print(f"总交易次数: {total_trades}")
    print(f"胜率: {win_rate:.2%}")
    print(f"总收益: {total_pnl:.2%}")
    print(f"平均收益: {avg_pnl:.4%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.2%}")

    # 按平仓原因统计
    print("\n平仓原因分布:")
    for reason, group in df.groupby("reason"):
        print(f"  {reason}: {len(group)} ({len(group)/total_trades:.1%})")

    # 验收判断
    print("\n" + "=" * 60)
    passed = True
    if sharpe < 0.5:
        print("⚠️ Sharpe < 0.5")
        passed = False
    if max_dd > 0.3:
        print("⚠️ Max Drawdown > 30%")
        passed = False
    if win_rate < 0.4:
        print("⚠️ Win Rate < 40%")
        passed = False

    if passed:
        print("✓ 回测通过，可进入 Paper Trading")
    else:
        print("✗ 回测未通过，建议优化模型或参数")


def main():
    parser = argparse.ArgumentParser(description="Offline Backtest (v10.0.0)")
    parser.add_argument("--data-dir", type=str, default="~/.algvex/data")
    parser.add_argument("--model-dir", type=str, default="~/.algvex/models/qlib_alpha")
    parser.add_argument("--instruments", type=str, nargs="+", default=["btcusdt"])
    parser.add_argument("--test-start", type=str, default="2024-07-01")
    parser.add_argument("--test-end", type=str, default="2024-12-31")
    parser.add_argument("--freq", type=str, default="1h")
    parser.add_argument("--signal-threshold", type=float, default=0.001)
    parser.add_argument("--stop-loss", type=float, default=0.02)
    parser.add_argument("--take-profit", type=float, default=0.03)

    args = parser.parse_args()

    config = BacktestConfig(
        signal_threshold=args.signal_threshold,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
    )

    run_backtest(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        instruments=args.instruments,
        test_start=args.test_start,
        test_end=args.test_end,
        freq=args.freq,
        config=config,
    )


if __name__ == "__main__":
    main()
