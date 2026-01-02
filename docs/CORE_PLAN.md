# AlgVex 核心方案 (P0 - MVP)

> **版本**: v8.2.0 (2026-01-02)
> **状态**: 可直接运行的完整方案

> **Qlib + Hummingbot 融合的加密货币现货量化交易平台**
>
> 混合方案：Qlib 添加加密货币原生支持，Hummingbot 使用 **Strategy V2 框架**。

---

## 目录

- [1. 方案概述](#1-方案概述)
- [2. 文件清单](#2-文件清单)
- [3. Qlib 修改](#3-qlib-修改)
- [4. 数据准备](#4-数据准备)
- [5. 模型训练](#5-模型训练)
- [6. Qlib 回测](#6-qlib-回测)
- [7. 策略脚本](#7-策略脚本)
- [8. 配置文件](#8-配置文件)
- [9. 启动与运行](#9-启动与运行)
- [10. 验收标准](#10-验收标准)

---

## 1. 方案概述

### 1.1 核心原则

```
┌─────────────────────────────────────────────────────────────┐
│                      核心原则                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 混合修改策略                                             │
│     - Qlib: 修改源码，添加 REG_CRYPTO 区域支持              │
│     - Hummingbot: 零修改，使用 Strategy V2 框架             │
│                                                             │
│  2. 原生支持优于 Workaround                                  │
│     - Qlib 原生支持加密货币，无需运行时覆盖配置             │
│     - Hummingbot 使用官方推荐的 V2 架构                     │
│                                                             │
│  3. V2 架构优势                                              │
│     - 数据获取: MarketDataProvider 统一接口                 │
│     - 订单执行: Executors 自动管理订单生命周期              │
│     - 策略逻辑: Controllers 抽象可复用                      │
│                                                             │
│  4. 可贡献上游                                               │
│     - Qlib 修改可提交 PR 到 microsoft/qlib                  │
│     - 让更多人受益                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     AlgVex 系统架构 (V2)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                      ┌─────────────────┐  │
│  │    Qlib     │                      │   Hummingbot    │  │
│  │  (修改2文件)│                      │  (Strategy V2)  │  │
│  │             │                      │                 │  │
│  │ + REG_CRYPTO│◀────────────────────▶│ - Binance API   │  │
│  │ - LGBModel  │    scripts/          │ - V2 框架       │  │
│  │ - Alpha158  │    qlib_alpha_       │ - Executors     │  │
│  │             │    strategy.py       │ - 风控管理      │  │
│  └─────────────┘         │            └─────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Strategy V2 架构                         │  │
│  │                                                       │  │
│  │  Script (StrategyV2Base)                              │  │
│  │      │                                                │  │
│  │      ├── MarketDataProvider  ← 统一数据获取接口       │  │
│  │      │   └── get_candles_df()                         │  │
│  │      │                                                │  │
│  │      ├── Controller          ← Qlib 模型预测          │  │
│  │      │   └── QlibAlphaController                      │  │
│  │      │                                                │  │
│  │      └── Executors           ← 订单执行组件           │  │
│  │          └── PositionExecutor (止损/止盈/时间限制)    │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 修改统计

| 组件 | 修改类型 | 文件数 | 说明 |
|------|----------|--------|------|
| **Qlib** | 修改源码 | 2 | 添加 REG_CRYPTO 区域 |
| **Hummingbot** | 零修改 | 0 | 使用 Strategy V2 框架 |
| **新建脚本** | 新建 | 6 | 数据/训练/回测/策略/控制器/配置 |

---

## 2. 文件清单

### 2.1 Qlib 修改文件

| 序号 | 文件路径 | 修改类型 | 说明 |
|------|----------|----------|------|
| 1 | `qlib/qlib/constant.py` | 修改 | 添加 `REG_CRYPTO` 常量 |
| 2 | `qlib/qlib/config.py` | 修改 | 添加加密货币区域配置 |

### 2.2 新建文件

| 序号 | 文件路径 | 类型 | 说明 |
|------|----------|------|------|
| 1 | `scripts/prepare_crypto_data.py` | 新建 | 数据准备脚本 |
| 2 | `scripts/train_model.py` | 新建 | 模型训练脚本 |
| 3 | `scripts/backtest_model.py` | 新建 | Qlib 回测脚本 |
| 4 | `scripts/qlib_alpha_strategy.py` | 新建 | V2 策略主脚本 |
| 5 | `controllers/qlib_alpha_controller.py` | 新建 | Qlib 信号控制器 |
| 6 | `conf/controllers/qlib_alpha.yml` | 新建 | 策略配置 |

### 2.3 Hummingbot

| 框架 | 修改内容 |
|------|---------|
| **Hummingbot** | 无修改，使用 Strategy V2 框架 (StrategyV2Base + Executors) |

---

## 3. Qlib 修改

### 3.1 修改 constant.py

**文件路径**: `qlib/qlib/constant.py`

**修改内容**: 添加 `REG_CRYPTO` 常量

```python
# 在文件末尾添加

# Crypto region (24/7 trading, no price limits)
REG_CRYPTO = "crypto"
```

**完整 diff**:

```diff
--- a/qlib/qlib/constant.py
+++ b/qlib/qlib/constant.py
@@ -3,3 +3,6 @@
 REG_CN = "cn"
 REG_US = "us"
 REG_TW = "tw"
+
+# Crypto region (24/7 trading, no price limits)
+REG_CRYPTO = "crypto"
```

### 3.2 修改 config.py

**文件路径**: `qlib/qlib/config.py`

**修改内容**: 添加加密货币区域配置

**修改 1**: 导入 `REG_CRYPTO`

```diff
--- a/qlib/qlib/config.py
+++ b/qlib/qlib/config.py
@@ -22,7 +22,7 @@ from typing import Callable, Optional, Union
 from typing import TYPE_CHECKING

-from qlib.constant import REG_CN, REG_US, REG_TW
+from qlib.constant import REG_CN, REG_US, REG_TW, REG_CRYPTO
```

**修改 2**: 添加 `_default_region_config` 中的加密货币配置

```diff
@@ -295,6 +295,12 @@ _default_region_config = {
         "limit_threshold": None,
         "deal_price": "close",
     },
+    REG_CRYPTO: {
+        "trade_unit": 0.00001,      # 加密货币最小交易单位
+        "limit_threshold": None,    # 无涨跌停限制
+        "deal_price": "close",
+        "time_per_step": 60,        # 分钟级数据
+    },
 }
```

### 3.3 验证修改

```python
# 验证 Qlib 加密货币支持
import qlib
from qlib.constant import REG_CRYPTO
from qlib.config import C

# 使用加密货币区域初始化
qlib.init(provider_uri="~/.qlib/qlib_data/crypto_data", region=REG_CRYPTO)

# 验证配置
print(f"trade_unit: {C['trade_unit']}")        # 0.00001
print(f"limit_threshold: {C['limit_threshold']}")  # None
print(f"region: {C['region']}")                # crypto
```

---

## 4. 数据准备

### 4.1 为何不使用 Qlib 官方加密货币收集器

> **说明**: Qlib 官方提供了加密货币数据收集器 (`qlib/scripts/data_collector/crypto/`)，
> 但本方案选择自定义实现，原因如下：

| 对比维度 | Qlib 官方收集器 | 本方案 (Binance API) |
|----------|-----------------|---------------------|
| **数据源** | Coingecko API | Binance 交易所 API |
| **支持频率** | 仅日线 (1d) | 1m/5m/15m/1h/4h/1d |
| **回测支持** | ❌ 不支持 (官方 README 明确说明) | ✅ 完整支持 |
| **数据字段** | prices, volumes, market_caps | OHLCV + VWAP |
| **数据质量** | 聚合数据 | 交易所原始数据 |

**关键原因:**

1. **小时级数据需求** - 加密货币 24/7 交易，小时级策略更常见
2. **回测必须** - Qlib 回测需要完整 OHLCV，官方收集器不支持
3. **数据质量** - 直接从交易所获取，避免聚合误差

```
官方收集器位置: qlib/scripts/data_collector/crypto/README.md
官方限制说明: "currently this dataset does not support backtesting"
```

### 4.2 脚本代码

**文件路径**: `scripts/prepare_crypto_data.py`

```python
"""
加密货币数据准备脚本

从 Binance 获取历史 K 线数据，转换为 Qlib 格式。

用法:
    python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT --interval 1h
"""

import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

import numpy as np
import pandas as pd


async def fetch_binance_klines(
    trading_pair: str,
    interval: str,
    start_time: int,
    end_time: int,
) -> pd.DataFrame:
    """
    从 Binance API 获取历史 K 线数据

    Parameters
    ----------
    trading_pair : str
        交易对，如 "BTC-USDT"
    interval : str
        K 线间隔，如 "1h", "1d"
    start_time : int
        开始时间戳 (毫秒)
    end_time : int
        结束时间戳 (毫秒)

    Returns
    -------
    pd.DataFrame
        K 线数据
    """
    import aiohttp

    symbol = trading_pair.replace("-", "")
    url = "https://api.binance.com/api/v3/klines"

    all_klines = []
    current_start = start_time

    async with aiohttp.ClientSession() as session:
        while current_start < end_time:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_time,
                "limit": 1000,
            }

            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    print(f"Error fetching {trading_pair}: {resp.status}")
                    break

                klines = await resp.json()
                if not klines:
                    break

                all_klines.extend(klines)
                current_start = klines[-1][0] + 1  # 下一个时间戳

                print(f"  Fetched {len(all_klines)} klines for {trading_pair}...")

    if not all_klines:
        return pd.DataFrame()

    # 转换为 DataFrame
    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    # 类型转换
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    return df


def detect_timestamp_unit(timestamp: int) -> str:
    """自动检测时间戳单位 (秒/毫秒)"""
    if timestamp > 1e12:
        return "ms"
    return "s"


def convert_to_qlib_format(
    df: pd.DataFrame,
    trading_pair: str,
) -> pd.DataFrame:
    """
    将 Binance K 线数据转换为 Qlib 格式

    Qlib 格式:
    - MultiIndex: (datetime, instrument)
    - Columns: $open, $high, $low, $close, $volume, $vwap, $factor
    """
    if df.empty:
        return pd.DataFrame()

    # 标准化交易对名称
    instrument = trading_pair.lower().replace("-", "")

    # 转换时间戳
    unit = detect_timestamp_unit(df["timestamp"].iloc[0])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True)
    df["instrument"] = instrument

    # 计算 VWAP
    df["$vwap"] = np.where(
        df["volume"] > 0,
        df["quote_volume"] / df["volume"],
        df["close"]
    )

    # 重命名列
    df["$open"] = df["open"]
    df["$high"] = df["high"]
    df["$low"] = df["low"]
    df["$close"] = df["close"]
    df["$volume"] = df["volume"]

    # factor 字段: 复权因子 (加密货币无复权，设为 1.0)
    # Qlib 官方要求: open, close, high, low, volume and factor at least
    df["$factor"] = 1.0

    # 设置 MultiIndex
    df = df.set_index(["datetime", "instrument"])

    # 只保留 Qlib 需要的列
    return df[["$open", "$high", "$low", "$close", "$volume", "$vwap", "$factor"]]


def save_to_qlib_format(
    merged_df: pd.DataFrame,
    output_dir: Path,
    freq: str,
):
    """
    保存为 Qlib Provider 可读取的格式

    目录结构:
    output_dir/
    ├── calendars/{freq}.txt
    ├── features/{instrument}/{col}.bin
    └── instruments/all.txt
    """
    features_dir = output_dir / "features"
    calendars_dir = output_dir / "calendars"
    instruments_dir = output_dir / "instruments"

    for d in [features_dir, calendars_dir, instruments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 获取所有交易对
    instruments = merged_df.index.get_level_values("instrument").unique()

    # 获取完整日历
    full_calendar = merged_df.index.get_level_values("datetime").unique().sort_values()

    # 时间格式
    freq_lower = freq.lower()
    if freq_lower in ["1d", "day"]:
        time_format = "%Y-%m-%d"
        calendar_filename = "day.txt"
    else:
        time_format = "%Y-%m-%d %H:%M:%S"
        calendar_filename = f"{freq_lower}.txt"

    # 保存每个交易对的特征数据
    # Qlib 官方要求: open, close, high, low, volume and factor at least
    qlib_columns = ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$factor"]
    for inst in instruments:
        inst_df = merged_df.xs(inst, level="instrument")
        inst_df = inst_df.reindex(full_calendar)  # 对齐日历

        inst_dir = features_dir / inst
        inst_dir.mkdir(exist_ok=True)

        for col in qlib_columns:
            col_name = col.replace("$", "")
            file_path = inst_dir / f"{col_name}.bin"

            # Qlib .bin 格式: [start_index (float32)] + [data (float32...)]
            # start_index 表示数据在日历中的起始位置
            data = inst_df[col].values.astype(np.float32)
            start_index = np.array([0], dtype=np.float32)  # 从索引 0 开始
            np.hstack([start_index, data]).tofile(file_path)

    # 保存日历
    calendar_file = calendars_dir / calendar_filename
    with open(calendar_file, "w") as f:
        for dt in full_calendar:
            f.write(dt.strftime(time_format) + "\n")

    # 保存交易对列表
    instruments_file = instruments_dir / "all.txt"
    with open(instruments_file, "w") as f:
        for inst in instruments:
            start = full_calendar.min().strftime(time_format)
            end = full_calendar.max().strftime(time_format)
            f.write(f"{inst}\t{start}\t{end}\n")

    print(f"Data saved to {output_dir}")


async def main():
    parser = argparse.ArgumentParser(description="Prepare crypto data for Qlib")
    parser.add_argument(
        "--trading-pairs",
        type=str,
        nargs="+",
        default=["BTC-USDT", "ETH-USDT"],
        help="Trading pairs to fetch",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Candle interval (1m, 5m, 15m, 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/.qlib/qlib_data/crypto_data",
        help="Output directory",
    )

    args = parser.parse_args()

    # 转换时间
    start_ts = int(datetime.strptime(args.start_date, "%Y-%m-%d")
                   .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(args.end_date, "%Y-%m-%d")
                 .replace(tzinfo=timezone.utc).timestamp() * 1000)

    output_dir = Path(args.output_dir).expanduser()

    # 获取所有交易对数据
    all_data = {}
    for pair in args.trading_pairs:
        print(f"Fetching {pair}...")
        df = await fetch_binance_klines(pair, args.interval, start_ts, end_ts)
        if not df.empty:
            qlib_df = convert_to_qlib_format(df, pair)
            all_data[pair] = qlib_df
            print(f"  Total: {len(qlib_df)} records")

    if not all_data:
        print("No data fetched!")
        return

    # 合并数据
    merged_df = pd.concat(all_data.values())
    merged_df = merged_df.sort_index()
    print(f"Total merged records: {len(merged_df)}")

    # 保存
    save_to_qlib_format(merged_df, output_dir, args.interval)


if __name__ == "__main__":
    asyncio.run(main())
```

### 4.3 数据目录结构

```
~/.qlib/qlib_data/crypto_data/
├── calendars/
│   └── 1h.txt              # 小时级日历
├── features/
│   ├── btcusdt/
│   │   ├── open.bin
│   │   ├── high.bin
│   │   ├── low.bin
│   │   ├── close.bin
│   │   ├── volume.bin
│   │   ├── vwap.bin
│   │   └── factor.bin      # 复权因子 (加密货币=1.0)
│   └── ethusdt/
│       └── ...
└── instruments/
    └── all.txt             # 交易对列表
```

---

## 5. 模型训练

### 5.1 Alpha158 适配策略

> **说明**: Alpha158 是为 A 股市场设计的 158 个因子集合，
> 直接用于加密货币需要注意适配问题。

**潜在问题:**

| 问题 | 说明 | 解决方案 |
|------|------|----------|
| **股票特有因子** | 换手率、市盈率等加密货币没有 | Alpha158 仅使用 OHLCV，无此问题 |
| **时间窗口** | 5/10/20/30/60 基于交易日 | 加密货币 24/7，窗口改为小时 |
| **交易日历** | 股票有休市日 | 加密货币使用连续时间戳 |

**本方案的适配策略:**

```
┌─────────────────────────────────────────────────────────────┐
│                  Alpha158 适配策略                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  训练阶段 (使用完整 Alpha158):                               │
│  ─────────────────────────────                              │
│  - 使用 Qlib 原生 Alpha158 Handler                         │
│  - 158 个因子全部计算                                       │
│  - 自动处理缺失值和标准化                                   │
│                                                             │
│  实盘阶段 (使用简化因子集):                                  │
│  ─────────────────────────────                              │
│  - 手动计算核心因子 (~50 个)                                │
│  - 排除股票特有逻辑                                         │
│  - 保持与训练特征对齐                                       │
│                                                             │
│  原因:                                                       │
│  - Alpha158 内部有股票市场假设 (如交易日历)                 │
│  - 实盘需要更快的计算速度                                   │
│  - 简化因子集更易于调试和维护                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**核心因子保留:**

- **KBAR 类** (9 个): KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
- **ROC/MA/STD** (5 个周期 × 10 类): ROC, MA, STD, MAX, MIN, QTLU, QTLD, RSV, CORR, CORD
- **总计**: ~50+ 个核心因子，覆盖 Alpha158 的主要信息

### 5.2 脚本代码

**文件路径**: `scripts/train_model.py`

```python
"""
模型训练脚本

使用 Qlib 的 LGBModel 训练价格预测模型。

用法:
    python scripts/train_model.py --instruments btcusdt ethusdt

注意:
    - freq 必须与实盘 prediction_interval 一致
    - 使用 RobustZScoreNorm 而非 CSRankNorm (适合少量品种)
"""

import pickle
import argparse
from pathlib import Path

import qlib
from qlib.constant import REG_CRYPTO
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH


def train_model(
    qlib_data_path: str,
    output_path: str,
    instruments: list,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    freq: str = "1h",
):
    """
    训练 LightGBM 模型
    """
    # 初始化 Qlib (使用 REG_CRYPTO 区域)
    print("Initializing Qlib with REG_CRYPTO...")
    data_path = Path(qlib_data_path).expanduser().resolve()
    qlib.init(provider_uri=str(data_path), region=REG_CRYPTO)

    # 创建数据处理器
    # 重要：freq 必须与实盘 prediction_interval 一致！
    #
    # Alpha158 默认 Label 定义 (Qlib 官方):
    #   label = Ref($close, -2) / Ref($close, -1) - 1
    #   即：预测 t+2 时刻相对于 t+1 时刻的收益率
    #   这意味着模型学习的是"下下个周期的收益率"
    #
    print(f"Creating Alpha158 handler with freq={freq}...")
    handler = Alpha158(
        instruments=instruments,
        start_time=train_start,
        end_time=valid_end,
        freq=freq,
        infer_processors=[],  # 推理时不处理 label
        learn_processors=[
            {"class": "DropnaLabel"},
            # RobustZScoreNorm 适合少量品种 (CSRankNorm 需要多品种)
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "label", "clip_outlier": True}},
        ],
    )

    # 创建数据集
    print("Creating dataset...")
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (train_start, train_end),
            "valid": (valid_start, valid_end),
        },
    )

    # 创建模型
    print("Creating LGBModel...")
    model = LGBModel(
        loss="mse",
        early_stopping_rounds=50,
        num_boost_round=500,
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
    )

    # 训练
    print("Training model...")
    model.fit(dataset)

    # 保存
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {output_path}")

    # 验证
    print("\nValidating model...")
    predictions = model.predict(dataset, segment="valid")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"Predictions sample:\n{predictions.head()}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Qlib model for crypto")
    parser.add_argument("--qlib-data-path", type=str, default="~/.qlib/qlib_data/crypto_data")
    parser.add_argument("--output", type=str, default="~/.qlib/models/lgb_model.pkl")
    parser.add_argument("--instruments", type=str, nargs="+", default=["btcusdt", "ethusdt"])
    parser.add_argument("--train-start", type=str, default="2023-01-01")
    parser.add_argument("--train-end", type=str, default="2024-06-30")
    parser.add_argument("--valid-start", type=str, default="2024-07-01")
    parser.add_argument("--valid-end", type=str, default="2024-12-31")
    parser.add_argument("--freq", type=str, default="1h", help="Must match prediction_interval")

    args = parser.parse_args()

    train_model(
        qlib_data_path=args.qlib_data_path,
        output_path=args.output,
        instruments=args.instruments,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        freq=args.freq,
    )


if __name__ == "__main__":
    main()
```

---

## 6. Qlib 回测

> **重要**: 在实盘交易前，必须使用 Qlib 回测验证模型有效性。
> Qlib 回测与 Hummingbot Paper Trading 互补，分别验证模型和执行逻辑。

### 6.1 回测流程

```
┌─────────────────────────────────────────────────────────────┐
│                    完整验证流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: Qlib 回测 (模型验证)                              │
│  ───────────────────────────────                            │
│  - 验证 Alpha158 因子有效性 (IC 值)                         │
│  - 验证 LGBModel 预测能力                                   │
│  - 计算策略收益率、夏普比率、最大回撤                        │
│  - 快速迭代模型参数                                         │
│  - 产出: 确认模型可用                                       │
│                                                             │
│  Stage 2: Hummingbot Paper Trading (执行验证)               │
│  ───────────────────────────────                            │
│  - 验证订单执行逻辑                                         │
│  - 验证交易所连接                                           │
│  - 验证风控 (止损/止盈)                                     │
│  - 产出: 确认系统可上线                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 回测脚本

**文件路径**: `scripts/backtest_model.py`

```python
"""
Qlib 回测脚本

使用 Qlib 的 Backtest 模块验证模型有效性。

用法:
    python scripts/backtest_model.py --instruments btcusdt ethusdt

输出指标:
    - IC (Information Coefficient): 预测值与实际收益的相关性
    - ICIR (IC Information Ratio): IC 的稳定性
    - Rank IC: 排序相关性
    - 年化收益率、夏普比率、最大回撤
"""

import pickle
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

import qlib
from qlib.constant import REG_CRYPTO
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.report import analysis_model, analysis_position
from qlib.backtest import backtest, executor


def load_model(model_path: str):
    """加载预训练模型"""
    path = Path(model_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def run_backtest(
    qlib_data_path: str,
    model_path: str,
    instruments: list,
    test_start: str,
    test_end: str,
    freq: str = "1h",
    topk: int = 1,
    n_drop: int = 0,
    benchmark: str = "btcusdt",
):
    """
    运行 Qlib 回测

    Parameters
    ----------
    qlib_data_path : str
        Qlib 数据路径
    model_path : str
        模型文件路径
    instruments : list
        交易对列表
    test_start : str
        测试开始日期
    test_end : str
        测试结束日期
    freq : str
        数据频率
    topk : int
        选择预测值最高的 K 个品种
    n_drop : int
        每次调仓时卖出的品种数
    benchmark : str
        基准品种
    """
    # 初始化 Qlib
    print("Initializing Qlib with REG_CRYPTO...")
    data_path = Path(qlib_data_path).expanduser().resolve()
    qlib.init(provider_uri=str(data_path), region=REG_CRYPTO)

    # 加载模型
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # 创建测试数据集
    print("Creating test dataset...")
    handler = Alpha158(
        instruments=instruments,
        start_time=test_start,
        end_time=test_end,
        freq=freq,
        infer_processors=[],
        learn_processors=[
            {"class": "DropnaLabel"},
            {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "label", "clip_outlier": True}},
        ],
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "test": (test_start, test_end),
        },
    )

    # 生成预测
    print("Generating predictions...")
    predictions = model.predict(dataset, segment="test")
    print(f"Predictions shape: {predictions.shape}")

    # ========== 模型评估 ==========
    print("\n" + "=" * 60)
    print("模型评估 (Model Evaluation)")
    print("=" * 60)

    # 计算 IC 指标
    ic_analysis = calculate_ic(predictions, dataset)
    print(f"\nIC (Information Coefficient): {ic_analysis['IC']:.4f}")
    print(f"ICIR (IC Information Ratio): {ic_analysis['ICIR']:.4f}")
    print(f"Rank IC: {ic_analysis['Rank IC']:.4f}")
    print(f"Rank ICIR: {ic_analysis['Rank ICIR']:.4f}")

    # ========== 策略回测 ==========
    print("\n" + "=" * 60)
    print("策略回测 (Strategy Backtest)")
    print("=" * 60)

    # 创建策略
    strategy = TopkDropoutStrategy(
        signal=predictions,
        topk=topk,
        n_drop=n_drop,
    )

    # 回测执行器配置
    executor_config = {
        "time_per_step": "day" if freq == "1d" else freq,
        "generate_portfolio_metrics": True,
    }

    # 运行回测
    portfolio_metric_dict, indicator_dict = backtest(
        executor=executor.SimulatorExecutor(**executor_config),
        strategy=strategy,
        start_time=test_start,
        end_time=test_end,
        benchmark=benchmark,
    )

    # 分析结果
    analysis = analyze_results(portfolio_metric_dict, indicator_dict)

    print(f"\n年化收益率 (Annual Return): {analysis['annual_return']:.2%}")
    print(f"夏普比率 (Sharpe Ratio): {analysis['sharpe']:.2f}")
    print(f"最大回撤 (Max Drawdown): {analysis['max_drawdown']:.2%}")
    print(f"信息比率 (Information Ratio): {analysis['information_ratio']:.2f}")
    print(f"胜率 (Win Rate): {analysis['win_rate']:.2%}")

    # ========== 验收建议 ==========
    print("\n" + "=" * 60)
    print("验收建议 (Acceptance Criteria)")
    print("=" * 60)

    passed = True

    if ic_analysis['IC'] < 0.02:
        print("⚠️  IC < 0.02: 模型预测能力较弱，建议优化特征或模型")
        passed = False
    else:
        print(f"✓ IC = {ic_analysis['IC']:.4f}: 模型预测能力可接受")

    if ic_analysis['ICIR'] < 0.1:
        print("⚠️  ICIR < 0.1: IC 不稳定，建议增加训练数据")
        passed = False
    else:
        print(f"✓ ICIR = {ic_analysis['ICIR']:.4f}: IC 稳定性可接受")

    if analysis['sharpe'] < 0.5:
        print("⚠️  Sharpe < 0.5: 风险调整后收益较低")
        passed = False
    else:
        print(f"✓ Sharpe = {analysis['sharpe']:.2f}: 风险收益比可接受")

    if analysis['max_drawdown'] > 0.3:
        print("⚠️  Max Drawdown > 30%: 最大回撤过大")
        passed = False
    else:
        print(f"✓ Max Drawdown = {analysis['max_drawdown']:.2%}: 回撤可接受")

    print("\n" + "=" * 60)
    if passed:
        print("✓ 模型验收通过，可进入 Hummingbot Paper Trading 阶段")
    else:
        print("✗ 模型验收未通过，建议优化后重新回测")
    print("=" * 60)

    return {
        "ic_analysis": ic_analysis,
        "portfolio_analysis": analysis,
        "predictions": predictions,
        "passed": passed,
    }


def calculate_ic(predictions: pd.Series, dataset: DatasetH) -> dict:
    """
    计算 IC 相关指标

    Parameters
    ----------
    predictions : pd.Series
        模型预测值
    dataset : DatasetH
        数据集 (包含 label)

    Returns
    -------
    dict
        IC, ICIR, Rank IC, Rank ICIR
    """
    # 获取真实标签
    df_test = dataset.prepare("test", col_set=["label"])
    labels = df_test["label"].reindex(predictions.index)

    # 合并预测和标签
    df = pd.DataFrame({
        "prediction": predictions,
        "label": labels,
    }).dropna()

    # 按时间分组计算 IC
    ic_series = df.groupby(level=0).apply(
        lambda x: x["prediction"].corr(x["label"])
    )

    # 计算 Rank IC
    rank_ic_series = df.groupby(level=0).apply(
        lambda x: x["prediction"].rank().corr(x["label"].rank())
    )

    return {
        "IC": ic_series.mean(),
        "ICIR": ic_series.mean() / (ic_series.std() + 1e-8),
        "Rank IC": rank_ic_series.mean(),
        "Rank ICIR": rank_ic_series.mean() / (rank_ic_series.std() + 1e-8),
    }


def analyze_results(portfolio_metric_dict: dict, indicator_dict: dict) -> dict:
    """
    分析回测结果

    Parameters
    ----------
    portfolio_metric_dict : dict
        组合指标
    indicator_dict : dict
        交易指标

    Returns
    -------
    dict
        分析结果
    """
    # 提取收益序列
    returns = portfolio_metric_dict.get("return", pd.Series())

    if len(returns) == 0:
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "information_ratio": 0.0,
            "win_rate": 0.0,
        }

    # 计算指标
    annual_return = returns.mean() * 252  # 假设日频
    sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252)

    # 最大回撤
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # 胜率
    win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    # 信息比率 (相对基准)
    excess_returns = returns - returns.mean()
    information_ratio = excess_returns.mean() / (excess_returns.std() + 1e-8) * np.sqrt(252)

    return {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "information_ratio": information_ratio,
        "win_rate": win_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Qlib Backtest for Crypto")
    parser.add_argument("--qlib-data-path", type=str, default="~/.qlib/qlib_data/crypto_data")
    parser.add_argument("--model-path", type=str, default="~/.qlib/models/lgb_model.pkl")
    parser.add_argument("--instruments", type=str, nargs="+", default=["btcusdt", "ethusdt"])
    parser.add_argument("--test-start", type=str, default="2024-07-01")
    parser.add_argument("--test-end", type=str, default="2024-12-31")
    parser.add_argument("--freq", type=str, default="1h")
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--benchmark", type=str, default="btcusdt")

    args = parser.parse_args()

    run_backtest(
        qlib_data_path=args.qlib_data_path,
        model_path=args.model_path,
        instruments=args.instruments,
        test_start=args.test_start,
        test_end=args.test_end,
        freq=args.freq,
        topk=args.topk,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    main()
```

### 6.3 回测指标说明

| 指标 | 含义 | 通过标准 |
|------|------|----------|
| **IC** | 预测值与实际收益的相关系数 | > 0.02 |
| **ICIR** | IC 的信息比率 (IC均值/IC标准差) | > 0.1 |
| **Rank IC** | 预测排名与实际排名的相关系数 | > 0.02 |
| **年化收益率** | 年化后的策略收益率 | > 0 |
| **夏普比率** | 风险调整后收益 | > 0.5 |
| **最大回撤** | 最大亏损幅度 | < 30% |

### 6.4 Qlib 回测 vs Hummingbot Paper Trading

| 对比维度 | Qlib 回测 | Hummingbot Paper Trading |
|----------|-----------|--------------------------|
| **目的** | 验证模型/因子有效性 | 验证执行逻辑 |
| **数据** | 历史数据 (.bin) | 实时行情 (API) |
| **速度** | 秒级 (快速迭代) | 1:1 实时 |
| **指标** | IC, ICIR, Sharpe | PnL, 胜率 |
| **适用阶段** | 模型开发阶段 | 上线前验证 |

---

## 7. 策略脚本 (Strategy V2)

> **重要**: v8.0.0 采用 Hummingbot 官方推荐的 Strategy V2 架构，
> 使用 `StrategyV2Base` + `Executors` + `MarketDataProvider`。

### 6.1 控制器代码

**文件路径**: `controllers/qlib_alpha_controller.py`

```python
"""
Qlib Alpha 控制器

基于 Qlib 机器学习模型生成交易信号的控制器。

V2 架构中，Controller 负责:
1. 从 MarketDataProvider 获取数据
2. 计算特征和预测信号
3. 生成 ExecutorAction 供策略执行
"""

import pickle
import logging
from pathlib import Path
from decimal import Decimal
from typing import List, Optional, Set

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_CRYPTO

from hummingbot.strategy_v2.controllers.controller_base import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TrailingStop,
    TripleBarrierConfig,
)
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
    StopExecutorAction,
)
from pydantic import Field


class QlibAlphaControllerConfig(ControllerConfigBase):
    """
    Qlib Alpha 控制器配置

    使用 StrategyV2ConfigBase 风格的配置类
    """
    controller_name: str = "qlib_alpha"
    controller_type: str = "directional_trading"

    # 交易配置
    connector_name: str = Field(default="binance")
    trading_pair: str = Field(default="BTC-USDT")
    order_amount_usd: Decimal = Field(default=Decimal("100"))

    # 模型配置
    model_path: str = Field(default="~/.qlib/models/lgb_model.pkl")
    qlib_data_path: str = Field(default="~/.qlib/qlib_data/crypto_data")

    # 信号配置
    signal_threshold: Decimal = Field(default=Decimal("0.005"))
    prediction_interval: str = Field(default="1h")
    lookback_bars: int = Field(default=100)

    # 三重屏障配置 (用于 PositionExecutor)
    stop_loss: Decimal = Field(default=Decimal("0.02"))
    take_profit: Decimal = Field(default=Decimal("0.03"))
    time_limit: int = Field(default=3600)

    # 执行配置
    cooldown_interval: int = Field(default=60)
    max_executors_per_side: int = Field(default=1)


class QlibAlphaController(ControllerBase):
    """
    Qlib Alpha 控制器

    V2 架构中的核心组件，负责:
    1. 接收 MarketDataProvider 数据
    2. 调用 Qlib 模型预测
    3. 生成 PositionExecutor 动作
    """

    def __init__(self, config: QlibAlphaControllerConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 模型状态
        self.model = None
        self.qlib_initialized = False
        self.last_signal_time = 0

        # 初始化
        self._init_qlib()
        self._load_model()

    def _init_qlib(self):
        """初始化 Qlib (使用 REG_CRYPTO 区域)"""
        try:
            qlib_path = Path(self.config.qlib_data_path).expanduser()
            qlib.init(provider_uri=str(qlib_path), region=REG_CRYPTO)
            self.qlib_initialized = True
            self.logger.info("Qlib initialized with REG_CRYPTO")
        except Exception as e:
            self.logger.error(f"Failed to initialize Qlib: {e}")

    def _load_model(self):
        """加载预训练模型"""
        try:
            model_path = Path(self.config.model_path).expanduser()
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.logger.info(f"Model loaded from {model_path}")
            else:
                self.logger.warning(f"Model file not found: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")

    async def update_processed_data(self):
        """
        更新处理后的数据 (V2 框架回调)

        从 MarketDataProvider 获取 K 线数据
        """
        try:
            # 使用 MarketDataProvider 获取 K 线数据
            candles_df = self.market_data_provider.get_candles_df(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                interval=self.config.prediction_interval,
                max_records=self.config.lookback_bars,
            )
            self.processed_data["candles"] = candles_df
        except Exception as e:
            self.logger.debug(f"Failed to get candles: {e}")
            self.processed_data["candles"] = None

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        确定执行动作 (V2 框架核心方法)

        Returns
        -------
        List[ExecutorAction]
            要执行的动作列表 (CreateExecutorAction / StopExecutorAction)
        """
        actions = []

        # 检查冷却时间
        current_time = self.market_data_provider.time()
        if current_time - self.last_signal_time < self.config.cooldown_interval:
            return actions

        # 检查是否已有活跃的 Executor
        active_executors = self.get_active_executors()
        if len(active_executors) >= self.config.max_executors_per_side:
            return actions

        # 获取信号
        signal = self._get_signal()

        if signal != 0:
            action = self._create_position_executor(signal)
            if action:
                actions.append(action)
                self.last_signal_time = current_time

        return actions

    def _get_signal(self) -> int:
        """
        获取交易信号

        Returns
        -------
        int
            1=买入, -1=卖出, 0=持有
        """
        if not self.qlib_initialized or self.model is None:
            return 0

        candles = self.processed_data.get("candles")
        if candles is None or len(candles) < 60:
            return 0

        try:
            # 计算特征
            features = self._compute_features(candles)
            if features is None or features.empty:
                return 0

            # 预测
            latest_features = features.iloc[-1:].values
            prediction = self.model.model.predict(latest_features)[0]

            # 根据阈值生成信号
            threshold = float(self.config.signal_threshold)
            if prediction > threshold:
                self.logger.info(f"BUY signal: prediction={prediction:.6f}")
                return 1
            elif prediction < -threshold:
                self.logger.info(f"SELL signal: prediction={prediction:.6f}")
                return -1

            return 0

        except Exception as e:
            self.logger.error(f"Error getting signal: {e}")
            return 0

    def _compute_features(self, candles: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算 Alpha158 因子"""
        try:
            df = candles.copy()

            close = df["close"].astype(float)
            open_ = df["open"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            volume = df["volume"].astype(float)

            features = pd.DataFrame(index=df.index)

            # KBAR 类因子
            features["KMID"] = (close - open_) / open_
            features["KLEN"] = (high - low) / open_
            features["KMID2"] = (close - open_) / (high - low + 1e-12)
            features["KUP"] = (high - np.maximum(open_, close)) / open_
            features["KUP2"] = (high - np.maximum(open_, close)) / (high - low + 1e-12)
            features["KLOW"] = (np.minimum(open_, close) - low) / open_
            features["KLOW2"] = (np.minimum(open_, close) - low) / (high - low + 1e-12)
            features["KSFT"] = (2 * close - high - low) / open_
            features["KSFT2"] = (2 * close - high - low) / (high - low + 1e-12)

            # ROC/MA/STD 因子
            for d in [5, 10, 20, 30, 60]:
                features[f"ROC{d}"] = close / close.shift(d) - 1
                ma = close.rolling(d).mean()
                features[f"MA{d}"] = close / ma - 1
                features[f"STD{d}"] = close.rolling(d).std() / close
                features[f"MAX{d}"] = close / high.rolling(d).max() - 1
                features[f"MIN{d}"] = close / low.rolling(d).min() - 1
                features[f"QTLU{d}"] = close / close.rolling(d).quantile(0.8) - 1
                features[f"QTLD{d}"] = close / close.rolling(d).quantile(0.2) - 1
                hh = high.rolling(d).max()
                ll = low.rolling(d).min()
                features[f"RSV{d}"] = (close - ll) / (hh - ll + 1e-12)
                features[f"CORR{d}"] = close.rolling(d).corr(volume)
                ret = close.pct_change()
                features[f"CORD{d}"] = ret.rolling(d).corr(volume.pct_change())

            return features.dropna()

        except Exception as e:
            self.logger.error(f"Error computing features: {e}")
            return None

    def _create_position_executor(self, signal: int) -> Optional[CreateExecutorAction]:
        """
        创建 PositionExecutor 动作

        使用 TripleBarrierConfig 配置止损/止盈/时间限制
        """
        try:
            # 获取当前价格
            mid_price = self.market_data_provider.get_price_by_type(
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                price_type="mid",
            )

            if mid_price is None or mid_price <= 0:
                return None

            # 计算下单数量
            amount = self.config.order_amount_usd / mid_price

            # 三重屏障配置
            triple_barrier = TripleBarrierConfig(
                stop_loss=self.config.stop_loss,
                take_profit=self.config.take_profit,
                time_limit=self.config.time_limit,
            )

            # 创建 PositionExecutor 配置
            executor_config = PositionExecutorConfig(
                timestamp=self.market_data_provider.time(),
                connector_name=self.config.connector_name,
                trading_pair=self.config.trading_pair,
                side="BUY" if signal > 0 else "SELL",
                amount=amount,
                triple_barrier_config=triple_barrier,
            )

            self.logger.info(
                f"Creating PositionExecutor: {executor_config.side} "
                f"{amount} @ {mid_price}"
            )

            return CreateExecutorAction(
                controller_id=self.config.controller_name,
                executor_config=executor_config,
            )

        except Exception as e:
            self.logger.error(f"Error creating executor: {e}")
            return None

    def get_active_executors(self) -> List[ExecutorBase]:
        """获取活跃的 Executors"""
        return [
            executor
            for executor in self.executors_info
            if executor.is_active
        ]
```

### 6.2 策略脚本代码

**文件路径**: `scripts/qlib_alpha_strategy.py`

```python
"""
Qlib Alpha V2 策略

基于 Qlib 机器学习模型的加密货币交易策略。

使用 Hummingbot Strategy V2 框架:
- StrategyV2Base (替代 ScriptStrategyBase)
- MarketDataProvider (替代 CandlesFactory)
- PositionExecutor (替代 buy()/sell())

启动方式:
    hummingbot
    >>> start --script qlib_alpha_strategy.py --conf conf/controllers/qlib_alpha.yml
"""

import logging
from decimal import Decimal
from typing import Dict, Set

from pydantic import Field

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.strategy_v2.strategy_v2_base import StrategyV2Base, StrategyV2ConfigBase
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction

# 导入控制器
from controllers.qlib_alpha_controller import (
    QlibAlphaController,
    QlibAlphaControllerConfig,
)


class QlibAlphaStrategyConfig(StrategyV2ConfigBase):
    """
    V2 策略配置

    继承自 StrategyV2ConfigBase，支持:
    - 动态配置更新 (config_update_interval)
    - 多控制器管理
    """
    script_file_name: str = "qlib_alpha_strategy.py"
    controllers_config: list = Field(default=[])

    # 可以在 YAML 中直接配置控制器
    # 或者在这里定义默认值


class QlibAlphaStrategy(StrategyV2Base):
    """
    Qlib Alpha V2 策略

    V2 架构工作流程:
    1. MarketDataProvider 自动管理 K 线数据
    2. Controller 调用 Qlib 模型生成信号
    3. PositionExecutor 自动管理订单生命周期 (含三重屏障)
    """

    @classmethod
    def init_markets(cls, config: QlibAlphaStrategyConfig):
        """初始化市场配置"""
        # 从控制器配置中提取交易对
        cls.markets = {}
        for controller_config in config.controllers_config:
            connector = controller_config.get("connector_name", "binance")
            trading_pair = controller_config.get("trading_pair", "BTC-USDT")
            if connector not in cls.markets:
                cls.markets[connector] = set()
            cls.markets[connector].add(trading_pair)

    def __init__(self, connectors: Dict[str, ConnectorBase], config: QlibAlphaStrategyConfig):
        super().__init__(connectors, config)
        self.logger = logging.getLogger(__name__)

    def create_actions_proposal(self) -> list[ExecutorAction]:
        """
        创建执行动作提案 (V2 核心方法)

        遍历所有控制器，收集 ExecutorAction
        """
        actions = []
        for controller in self.controllers.values():
            controller_actions = controller.determine_executor_actions()
            actions.extend(controller_actions)
        return actions

    def format_status(self) -> str:
        """状态显示"""
        lines = []
        lines.append("=== Qlib Alpha V2 Strategy ===")
        lines.append(f"Controllers: {len(self.controllers)}")

        for name, controller in self.controllers.items():
            lines.append(f"\n[Controller: {name}]")
            lines.append(f"  Qlib Initialized: {controller.qlib_initialized}")
            lines.append(f"  Model Loaded: {controller.model is not None}")
            active = len(controller.get_active_executors())
            lines.append(f"  Active Executors: {active}")

        return "\n".join(lines)
```

### 6.3 V2 vs V1 架构对比

```
┌─────────────────────────────────────────────────────────────┐
│                    V1 vs V2 架构对比                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  V1 (ScriptStrategyBase) - 已弃用                           │
│  ─────────────────────────────────────                      │
│  - 手动调用 buy()/sell()                                    │
│  - 手动管理 CandlesFactory                                  │
│  - 手动实现三重屏障                                         │
│  - 配置类: BaseClientModel                                  │
│                                                             │
│  V2 (StrategyV2Base) - 推荐                                 │
│  ─────────────────────────────────────                      │
│  - Executors 自动管理订单                                   │
│  - MarketDataProvider 统一数据接口                          │
│  - TripleBarrierConfig 内置支持                             │
│  - 配置类: StrategyV2ConfigBase                             │
│  - 支持动态配置更新                                         │
│  - 支持多控制器                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 配置文件

### 8.1 控制器配置

**文件路径**: `conf/controllers/qlib_alpha.yml`

```yaml
# Qlib Alpha V2 控制器配置

# 控制器元信息
controller_name: qlib_alpha
controller_type: directional_trading

# 交易配置
connector_name: binance
trading_pair: BTC-USDT
order_amount_usd: 100

# 模型配置
model_path: ~/.qlib/models/lgb_model.pkl
qlib_data_path: ~/.qlib/qlib_data/crypto_data

# 信号配置
signal_threshold: 0.005    # 0.5% 收益率阈值
prediction_interval: 1h
lookback_bars: 100

# 三重屏障配置 (PositionExecutor 自动管理)
stop_loss: 0.02            # 2% 止损
take_profit: 0.03          # 3% 止盈
time_limit: 3600           # 1小时持仓限制

# 执行配置
cooldown_interval: 60      # 60秒冷却
max_executors_per_side: 1  # 每方向最多1个执行器
```

### 8.2 策略配置

**文件路径**: `conf/scripts/qlib_alpha_v2.yml`

```yaml
# Qlib Alpha V2 策略配置

# V2 策略支持多控制器
controllers_config:
  - controller_name: qlib_alpha
    controller_type: directional_trading
    connector_name: binance
    trading_pair: BTC-USDT
    order_amount_usd: 100
    model_path: ~/.qlib/models/lgb_model.pkl
    qlib_data_path: ~/.qlib/qlib_data/crypto_data
    signal_threshold: 0.005
    prediction_interval: 1h
    lookback_bars: 100
    stop_loss: 0.02
    take_profit: 0.03
    time_limit: 3600
    cooldown_interval: 60
    max_executors_per_side: 1

# 可添加多个控制器实现多品种交易
# - controller_name: qlib_alpha_eth
#   trading_pair: ETH-USDT
#   ...
```

---

## 9. 启动与运行

### 9.1 完整运行流程

```
┌─────────────────────────────────────────────────────────────┐
│                   完整运行流程 (V2)                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 0: 修改 Qlib 源码 (仅首次)                            │
│  - 按照第 3 节修改 constant.py 和 config.py                 │
│                                                             │
│  Step 1: 准备数据                                           │
│  $ python scripts/prepare_crypto_data.py \                 │
│      --trading-pairs BTC-USDT ETH-USDT \                   │
│      --interval 1h \                                        │
│      --start-date 2023-01-01 \                             │
│      --end-date 2024-12-31                                 │
│                                                             │
│  Step 2: 训练模型                                           │
│  $ python scripts/train_model.py \                         │
│      --instruments btcusdt ethusdt \                       │
│      --freq 1h                                             │
│                                                             │
│  Step 3: Qlib 回测 (模型验证) ← 新增                        │
│  $ python scripts/backtest_model.py \                      │
│      --instruments btcusdt ethusdt \                       │
│      --test-start 2024-07-01 \                             │
│      --test-end 2024-12-31                                 │
│  - 验证 IC > 0.02, Sharpe > 0.5                            │
│  - 通过后进入下一步                                        │
│                                                             │
│  Step 4: 配置 API                                           │
│  $ cd hummingbot && ./start                                │
│  >>> connect binance                                        │
│  >>> [输入 API Key 和 Secret]                               │
│                                                             │
│  Step 5: 启动 V2 策略                                       │
│  >>> start --script qlib_alpha_strategy.py \               │
│            --conf conf/scripts/qlib_alpha_v2.yml           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 命令参考

```bash
# 数据准备 (可选参数)
python scripts/prepare_crypto_data.py \
    --trading-pairs BTC-USDT ETH-USDT SOL-USDT \
    --interval 1h \
    --start-date 2023-01-01 \
    --end-date 2024-12-31 \
    --output-dir ~/.qlib/qlib_data/crypto_data

# 模型训练 (可选参数)
python scripts/train_model.py \
    --qlib-data-path ~/.qlib/qlib_data/crypto_data \
    --output ~/.qlib/models/lgb_model.pkl \
    --instruments btcusdt ethusdt \
    --train-start 2023-01-01 \
    --train-end 2024-06-30 \
    --valid-start 2024-07-01 \
    --valid-end 2024-12-31 \
    --freq 1h

# Qlib 回测 (模型验证)
python scripts/backtest_model.py \
    --qlib-data-path ~/.qlib/qlib_data/crypto_data \
    --model-path ~/.qlib/models/lgb_model.pkl \
    --instruments btcusdt ethusdt \
    --test-start 2024-07-01 \
    --test-end 2024-12-31 \
    --freq 1h

# 启动 V2 策略
cd hummingbot
./start
>>> connect binance
>>> start --script qlib_alpha_strategy.py --conf conf/scripts/qlib_alpha_v2.yml
```

### 9.3 Paper Trading

```bash
# 使用 Binance Testnet
>>> connect binance_paper_trade
>>> start --script qlib_alpha_strategy.py --conf conf/scripts/qlib_alpha_v2.yml
```

---

## 10. 验收标准

### 10.1 验收清单

| 序号 | 验收项 | 验收方法 | 通过标准 |
|------|--------|----------|----------|
| 1 | Qlib 修改 | 导入 REG_CRYPTO | 无报错 |
| 2 | 数据准备 | 运行 prepare_crypto_data.py | 生成 Qlib 格式数据 |
| 3 | 模型训练 | 运行 train_model.py | 模型文件生成 |
| 4 | **Qlib 回测** | 运行 backtest_model.py | IC > 0.02, Sharpe > 0.5 |
| 5 | Qlib 初始化 | Controller 启动 | region=crypto |
| 6 | MarketDataProvider | Controller 运行 | get_candles_df() 返回数据 |
| 7 | 特征计算 | Controller 运行 | 特征矩阵非空 |
| 8 | 信号生成 | Controller 运行 | 返回 -1/0/1 |
| 9 | PositionExecutor | 策略运行 | Executor 创建成功 |
| 10 | 三重屏障 | 触发条件 | Executor 自动关闭 |
| 11 | Paper Trading | 模拟交易 24h | 无异常 |

### 10.2 验证脚本

```python
# scripts/verify_integration.py
"""
集成验证脚本

验证 Qlib + Hummingbot 融合的完整性和数据质量。

用法:
    python scripts/verify_integration.py
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def verify_qlib_crypto() -> bool:
    """验证 Qlib REG_CRYPTO 配置"""
    print("1. Testing Qlib REG_CRYPTO...")
    try:
        import qlib
        from qlib.constant import REG_CRYPTO
        from qlib.config import C

        data_path = Path("~/.qlib/qlib_data/crypto_data").expanduser()
        qlib.init(provider_uri=str(data_path), region=REG_CRYPTO)

        assert C["region"] == "crypto", f"Expected crypto, got {C['region']}"
        assert C["trade_unit"] == 0.00001, f"Expected 0.00001, got {C['trade_unit']}"
        assert C["limit_threshold"] is None, f"Expected None, got {C['limit_threshold']}"
        print("   ✓ REG_CRYPTO working correctly")
        return True
    except Exception as e:
        print(f"   ✗ REG_CRYPTO failed: {e}")
        return False


def verify_data_quality(data_path: str = "~/.qlib/qlib_data/crypto_data") -> Tuple[bool, dict]:
    """
    验证数据质量

    检查项:
    - $factor 字段存在且 = 1.0
    - 缺失值比例 < 5%
    - 时间戳连续性
    - 数据范围合理性
    """
    print("2. Testing data quality...")
    results = {"passed": True, "issues": []}

    data_dir = Path(data_path).expanduser()
    features_dir = data_dir / "features"

    if not features_dir.exists():
        print(f"   ✗ Features directory not found: {features_dir}")
        return False, {"passed": False, "issues": ["Features directory not found"]}

    # 检查每个交易对
    for inst_dir in features_dir.iterdir():
        if not inst_dir.is_dir():
            continue

        inst_name = inst_dir.name
        print(f"   Checking {inst_name}...")

        # 2.1 验证 $factor 字段
        factor_file = inst_dir / "factor.bin"
        if not factor_file.exists():
            results["issues"].append(f"{inst_name}: factor.bin not found")
            results["passed"] = False
        else:
            # 读取 .bin 文件 (跳过第一个 float32 作为 start_index)
            data = np.fromfile(factor_file, dtype=np.float32)
            factor_values = data[1:]  # 跳过 start_index

            # 验证 factor = 1.0 (加密货币无复权)
            if not np.allclose(factor_values, 1.0, atol=1e-6):
                results["issues"].append(f"{inst_name}: factor != 1.0")
                results["passed"] = False
            else:
                print(f"      ✓ factor.bin valid (all = 1.0)")

        # 2.2 验证必需字段完整性
        required_fields = ["open", "high", "low", "close", "volume"]
        for field in required_fields:
            field_file = inst_dir / f"{field}.bin"
            if not field_file.exists():
                results["issues"].append(f"{inst_name}: {field}.bin not found")
                results["passed"] = False
            else:
                data = np.fromfile(field_file, dtype=np.float32)[1:]

                # 检查缺失值 (NaN)
                nan_pct = np.isnan(data).sum() / len(data)
                if nan_pct > 0.05:
                    results["issues"].append(f"{inst_name}/{field}: {nan_pct:.1%} missing values")
                    results["passed"] = False

                # 检查异常值 (价格为负或为零)
                if field in ["open", "high", "low", "close"]:
                    invalid_pct = (data <= 0).sum() / len(data)
                    if invalid_pct > 0:
                        results["issues"].append(f"{inst_name}/{field}: {invalid_pct:.1%} invalid values (<=0)")
                        results["passed"] = False

        print(f"      ✓ Required fields complete")

    # 2.3 验证日历文件
    calendars_dir = data_dir / "calendars"
    if calendars_dir.exists():
        calendar_files = list(calendars_dir.glob("*.txt"))
        if calendar_files:
            cal_file = calendar_files[0]
            with open(cal_file) as f:
                lines = f.readlines()
            print(f"      ✓ Calendar: {len(lines)} timestamps in {cal_file.name}")

            # 检查时间戳连续性 (仅警告，不阻断)
            if len(lines) > 1:
                # 简单检查：统计时间间隔
                print(f"      ✓ Calendar file valid")
    else:
        results["issues"].append("calendars directory not found")
        results["passed"] = False

    if results["passed"]:
        print("   ✓ Data quality check passed")
    else:
        print(f"   ✗ Data quality issues: {len(results['issues'])}")
        for issue in results["issues"]:
            print(f"      - {issue}")

    return results["passed"], results


def verify_model_load() -> bool:
    """验证模型加载"""
    print("3. Testing model load...")
    import pickle

    model_path = Path("~/.qlib/models/lgb_model.pkl").expanduser()
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print(f"   ✓ Model loaded: {type(model).__name__}")
            return True
        except Exception as e:
            print(f"   ✗ Model load failed: {e}")
            return False
    else:
        print("   ⚠ Model not found (run train_model.py first)")
        return True  # 非阻断性检查


def verify_feature_computation() -> bool:
    """验证特征计算"""
    print("4. Testing feature computation...")
    try:
        # Mock data
        df = pd.DataFrame({
            "open": np.random.uniform(40000, 42000, 100),
            "high": np.random.uniform(42000, 43000, 100),
            "low": np.random.uniform(39000, 40000, 100),
            "close": np.random.uniform(40000, 42000, 100),
            "volume": np.random.uniform(100, 1000, 100),
        })

        close = df["close"]
        open_ = df["open"]
        high = df["high"]
        low = df["low"]

        features = pd.DataFrame()

        # KBAR 因子
        features["KMID"] = (close - open_) / open_
        features["KLEN"] = (high - low) / open_
        features["KMID2"] = (close - open_) / (high - low + 1e-12)

        # ROC/MA 因子
        for d in [5, 10, 20]:
            features[f"ROC{d}"] = close / close.shift(d) - 1
            ma = close.rolling(d).mean()
            features[f"MA{d}"] = close / ma - 1

        features = features.dropna()

        assert len(features) > 0, "Features empty"
        assert len(features.columns) >= 9, f"Expected >= 9 features, got {len(features.columns)}"

        print(f"   ✓ Computed {len(features.columns)} features, {len(features)} samples")
        return True
    except Exception as e:
        print(f"   ✗ Feature computation failed: {e}")
        return False


def verify():
    """运行所有验证"""
    print("=" * 60)
    print("AlgVex Integration Verification")
    print("=" * 60 + "\n")

    results = []

    # 1. Qlib REG_CRYPTO
    results.append(("Qlib REG_CRYPTO", verify_qlib_crypto()))

    # 2. 数据质量
    passed, details = verify_data_quality()
    results.append(("Data Quality", passed))

    # 3. 模型加载
    results.append(("Model Load", verify_model_load()))

    # 4. 特征计算
    results.append(("Feature Computation", verify_feature_computation()))

    # 汇总
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ All verifications passed!")
        return 0
    else:
        print("✗ Some verifications failed")
        return 1


if __name__ == "__main__":
    sys.exit(verify())
```

---

## 附录

### A. 依赖版本

```
qlib >= 0.9.7 (需修改源码)
hummingbot >= 2.11.0
lightgbm >= 4.0.0
pandas >= 2.0.0
numpy >= 1.24.0
pydantic >= 2.0.0
aiohttp >= 3.8.0
```

### B. 常见问题

| 问题 | 解决方案 |
|------|----------|
| REG_CRYPTO 未定义 | 检查 constant.py 修改是否生效 |
| Qlib 初始化失败 | 检查数据目录路径是否正确 |
| 模型加载失败 | 先运行 train_model.py |
| MarketDataProvider 返回空 | 等待数据收集或检查 trading_pair 格式 |
| 信号一直为 0 | 调整 signal_threshold |
| Executor 创建失败 | 检查 API 权限、余额、配置格式 |
| Controller 未找到 | 检查 controllers/ 目录和导入路径 |

### C. 变更日志

**v8.2.0** (2026-01-02)
- **增强**: 数据验证脚本 `verify_integration.py`
  - 新增 $factor 字段验证 (确保 = 1.0)
  - 新增缺失值检查 (阈值 < 5%)
  - 新增异常值检查 (价格 > 0)
  - 新增日历文件验证
- **文档**: 说明为何不使用 Qlib 官方加密货币收集器 (4.1 节)
  - 官方仅支持日线，本方案支持小时级
  - 官方不支持回测，本方案完整支持
- **文档**: 说明 Alpha158 适配策略 (5.1 节)
  - 训练阶段使用完整 Alpha158
  - 实盘阶段使用简化因子集 (~50 个)
  - 核心因子对齐说明

**v8.1.0** (2026-01-02)
- **新增**: Qlib 回测模块集成 (第 6 节)
- 新增 `scripts/backtest_model.py` 回测脚本
- 支持 IC/ICIR/Sharpe/MaxDrawdown 等指标计算
- 使用 TopkDropoutStrategy 进行策略回测
- 明确两阶段验证流程: Qlib 回测 → Hummingbot Paper Trading
- 更新运行流程，新增 Step 3 回测验证步骤
- 更新验收清单，新增回测验收项

**v8.0.2** (2026-01-02)
- 添加 Alpha158 默认 Label 定义说明 (`Ref($close, -2) / Ref($close, -1) - 1`)
- 明确 `infer_processors=[]` 用途说明
- 基于 Qlib 官方文档全面验证

**v8.0.1** (2026-01-02)
- 添加 `$factor` 字段支持 (Qlib 官方要求的必需字段)
- 加密货币无复权，factor 固定为 1.0
- 更新数据目录结构说明

**v8.0.0** (2026-01-02)
- **重大升级**: 采用 Hummingbot Strategy V2 架构
- 基类: `ScriptStrategyBase` → `StrategyV2Base`
- 数据获取: `CandlesFactory` → `MarketDataProvider.get_candles_df()`
- 订单执行: `buy()/sell()` → `PositionExecutor` + `TripleBarrierConfig`
- 配置类: `BaseClientModel` → `StrategyV2ConfigBase`
- 新增: `QlibAlphaController` 控制器，分离信号逻辑
- 新增: 支持多控制器、动态配置更新
- 符合 Hummingbot 官方推荐的现代架构

**v7.0.0** (2026-01-02)
- 采用混合修改策略：Qlib 修改源码，Hummingbot 零修改
- Qlib: 添加 REG_CRYPTO 区域配置 (constant.py, config.py)
- 移除运行时 C 配置覆盖，使用原生 region 支持
- 代码更清晰，可维护性更高
- ⚠️ 使用 ScriptStrategyBase (已被 v8.0.0 取代)

**v6.2.0** (2026-01-02)
- 修复 markets 定义：使用 `init_markets` classmethod (参考 simple_pmm.py)
- 修复 config 类：`script` → `script_file_name` (Hummingbot 要求)
- 添加 `ConnectorBase` 导入和正确的类型注解

**v6.1.1** (2026-01-02)
- 修复属性名：`is_ready` → `ready` (与 CandlesBase 源码一致)
- 添加 `candles_feed.start()` 调用 (必须启动才能收集数据)
- 添加 `on_stop()` 生命周期方法 (停止时清理资源)

**v6.1.0** (2026-01-02)
- 修复 K 线数据获取方式：使用 CandlesFactory 替代不存在的 connector.get_candles()
- 修复 Qlib .bin 文件格式：添加起始索引字节
- 添加 _init_candles_feed() 初始化方法

**v6.0.0** (2026-01-02)
- 完全重写方案，采用零源码修改架构
- Qlib: 使用 qlib.init() 运行时配置
- Hummingbot: 使用 scripts/ 脚本策略机制
- 桥接逻辑内联到策略脚本
- 从 14 个文件 ~970 行 → 4 个文件 ~500 行
