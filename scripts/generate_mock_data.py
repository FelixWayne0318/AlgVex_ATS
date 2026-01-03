#!/usr/bin/env python3
"""
AlgVex v10.0.4 - 模拟数据生成脚本

为教程和测试生成 BTC/ETH 的模拟 OHLCV 数据。
数据保存到 ~/.algvex/data/1h/ 目录。

用法:
    python scripts/generate_mock_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_mock_data():
    """生成模拟的加密货币 OHLCV 数据"""

    # 数据目录
    data_dir = Path.home() / '.algvex' / 'data' / '1h'
    data_dir.mkdir(parents=True, exist_ok=True)

    # 生成 2 年模拟数据 (2023-01-01 到 2024-12-31)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='1h', tz='UTC')
    n = len(dates)

    print("=" * 50)
    print("AlgVex v10.0.4 - 模拟数据生成")
    print("=" * 50)
    print(f"\n数据目录: {data_dir}")
    print(f"时间范围: 2023-01-01 ~ 2024-12-31")
    print(f"数据点数: {n} bars\n")

    symbols = [
        ('btcusdt', 30000, 42),  # symbol, base_price, random_seed
        ('ethusdt', 2000, 43),
    ]

    for symbol, base_price, seed in symbols:
        np.random.seed(seed)

        # 生成随机收益率 (约 2% 日波动)
        returns = np.random.randn(n) * 0.02
        close = base_price * np.exp(np.cumsum(returns))

        # 生成 OHLCV 数据
        df = pd.DataFrame({
            'open': close * (1 + np.random.randn(n) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(n)) * 0.01),
            'low': close * (1 - np.abs(np.random.randn(n)) * 0.01),
            'close': close,
            'volume': np.random.uniform(100, 1000, n),
        }, index=dates)

        # 确保 high >= max(open, close) 和 low <= min(open, close)
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)

        # 保存为 parquet 格式
        output_path = data_dir / f'{symbol}.parquet'
        df.to_parquet(output_path)

        print(f"Created {symbol}.parquet: {len(df)} bars")
        print(f"  - Price range: ${df['close'].min():.2f} ~ ${df['close'].max():.2f}")

    print(f"\nData saved to: {data_dir}")
    print("Done!")


if __name__ == '__main__':
    generate_mock_data()
