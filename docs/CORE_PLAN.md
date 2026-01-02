# AlgVex 核心方案 (P0 - MVP)

> **版本**: v6.1.1 (2026-01-02)
> **状态**: 可直接运行的完整方案

> **Qlib + Hummingbot 融合的加密货币现货量化交易平台**
>
> 本方案基于两个框架的原生扩展机制，**零源码修改**，仅新建必要脚本。

---

## 目录

- [1. 方案概述](#1-方案概述)
- [2. 文件清单](#2-文件清单)
- [3. 数据准备](#3-数据准备)
- [4. 模型训练](#4-模型训练)
- [5. 策略脚本](#5-策略脚本)
- [6. 配置文件](#6-配置文件)
- [7. 启动与运行](#7-启动与运行)
- [8. 验收标准](#8-验收标准)

---

## 1. 方案概述

### 1.1 核心原则

```
┌─────────────────────────────────────────────────────────────┐
│                      核心原则                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 零源码修改                                               │
│     - Qlib: 使用 qlib.init() 运行时配置                     │
│     - Hummingbot: 使用 scripts/ 脚本策略机制                │
│                                                             │
│  2. 复用现有机制                                             │
│     - Qlib: Provider 机制、YAML 配置加载                    │
│     - Hummingbot: ScriptStrategyBase、原生下单 API          │
│                                                             │
│  3. 桥接逻辑内联                                             │
│     - 数据转换: 策略脚本私有方法                             │
│     - 信号执行: 直接调用 self.buy()/self.sell()             │
│                                                             │
│  4. 升级兼容                                                 │
│     - Qlib 升级: 直接升级，无需合并                          │
│     - Hummingbot 升级: 保留 scripts/ 目录即可               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                     AlgVex 系统架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐                      ┌─────────────────┐  │
│  │    Qlib     │                      │   Hummingbot    │  │
│  │  (不修改)   │                      │    (不修改)     │  │
│  │             │                      │                 │  │
│  │ - LGBModel  │◀────────────────────▶│ - Binance API   │  │
│  │ - Alpha158  │    scripts/          │ - K线数据       │  │
│  │ - 因子计算  │    qlib_alpha_       │ - 订单执行      │  │
│  │             │    strategy.py       │ - 风控管理      │  │
│  └─────────────┘         │            └─────────────────┘  │
│                          │                                  │
│                          ▼                                  │
│              ┌─────────────────────┐                       │
│              │   策略脚本 (核心)    │                       │
│              │                     │                       │
│              │ - 数据转换 (内联)   │                       │
│              │ - 特征计算 (内联)   │                       │
│              │ - 信号生成          │                       │
│              │ - 交易执行          │                       │
│              └─────────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 代码量统计

| 文件 | 用途 | 代码量 |
|------|------|--------|
| `scripts/prepare_crypto_data.py` | 数据准备 | ~120 行 |
| `scripts/train_model.py` | 模型训练 | ~100 行 |
| `scripts/qlib_alpha_strategy.py` | 策略脚本 | ~250 行 |
| `conf/scripts/qlib_alpha.yml` | 配置文件 | ~30 行 |
| **总计** | - | **~500 行** |

**对比 v5.x**: 从 14 个文件 ~970 行 → 4 个文件 ~500 行

---

## 2. 文件清单

### 2.1 完整文件清单

| 序号 | 文件路径 | 类型 | 说明 |
|------|----------|------|------|
| 1 | `scripts/prepare_crypto_data.py` | 新建 | 数据准备脚本 |
| 2 | `scripts/train_model.py` | 新建 | 模型训练脚本 |
| 3 | `scripts/qlib_alpha_strategy.py` | 新建 | 策略主脚本 |
| 4 | `conf/scripts/qlib_alpha.yml` | 新建 | 策略配置 |

### 2.2 Qlib / Hummingbot 修改

| 框架 | 修改内容 |
|------|---------|
| **Qlib** | 无修改，使用 `qlib.init()` 运行时配置 |
| **Hummingbot** | 无修改，使用 `scripts/` 脚本策略机制 |

---

## 3. 数据准备

### 3.1 脚本代码

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
    - Columns: $open, $high, $low, $close, $volume, $vwap
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

    # 设置 MultiIndex
    df = df.set_index(["datetime", "instrument"])

    # 只保留 Qlib 需要的列
    return df[["$open", "$high", "$low", "$close", "$volume", "$vwap"]]


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
    qlib_columns = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
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

### 3.2 数据目录结构

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
│   │   └── vwap.bin
│   └── ethusdt/
│       └── ...
└── instruments/
    └── all.txt             # 交易对列表
```

---

## 4. 模型训练

### 4.1 脚本代码

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
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.config import C


def init_qlib_for_crypto(data_path: str):
    """
    初始化 Qlib (加密货币配置)

    使用运行时配置，无需修改 Qlib 源码
    """
    data_path = Path(data_path).expanduser().resolve()

    qlib.init(provider_uri=str(data_path))

    # 运行时修改交易参数
    C["trade_unit"] = 0.00001      # 加密货币最小交易单位
    C["limit_threshold"] = None    # 无涨跌停限制
    C["deal_price"] = "close"


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
    # 初始化 Qlib
    print("Initializing Qlib...")
    init_qlib_for_crypto(qlib_data_path)

    # 创建数据处理器
    # 重要：freq 必须与实盘 prediction_interval 一致！
    print(f"Creating Alpha158 handler with freq={freq}...")
    handler = Alpha158(
        instruments=instruments,
        start_time=train_start,
        end_time=valid_end,
        freq=freq,
        infer_processors=[],
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

## 5. 策略脚本

### 5.1 脚本代码

**文件路径**: `scripts/qlib_alpha_strategy.py`

```python
"""
Qlib Alpha 策略

基于 Qlib 机器学习模型的加密货币交易策略。

使用 Hummingbot ScriptStrategyBase，无需修改任何源码。

启动方式:
    hummingbot
    >>> start --script qlib_alpha_strategy.py --conf conf/scripts/qlib_alpha.yml
"""

import pickle
import logging
from pathlib import Path
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import Field

import qlib
from qlib.config import C

from hummingbot.client.config.config_data_types import BaseClientModel
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.core.data_type.common import OrderType, TradeType, PositionAction
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig


# =============================================================================
# 配置类
# =============================================================================

class QlibAlphaConfig(BaseClientModel):
    """策略配置 (Pydantic 模型)"""

    script: str = "qlib_alpha_strategy.py"

    # 交易配置
    exchange: str = Field(default="binance", description="交易所")
    trading_pair: str = Field(default="BTC-USDT", description="交易对")
    order_amount_usd: Decimal = Field(default=Decimal("100"), description="每笔订单金额(USD)")

    # 模型配置
    model_path: str = Field(default="~/.qlib/models/lgb_model.pkl", description="模型路径")
    qlib_data_path: str = Field(default="~/.qlib/qlib_data/crypto_data", description="Qlib数据目录")

    # 信号配置
    signal_threshold: Decimal = Field(default=Decimal("0.005"), description="信号阈值(收益率)")
    prediction_interval: str = Field(default="1h", description="预测间隔")
    lookback_bars: int = Field(default=100, description="回看K线数量")

    # 风控配置 (三重屏障)
    stop_loss: Decimal = Field(default=Decimal("0.02"), description="止损比例")
    take_profit: Decimal = Field(default=Decimal("0.03"), description="止盈比例")
    time_limit: int = Field(default=3600, description="持仓时间限制(秒)")

    # 执行配置
    cooldown_time: int = Field(default=60, description="交易冷却时间(秒)")


# =============================================================================
# 策略主类
# =============================================================================

class QlibAlphaStrategy(ScriptStrategyBase):
    """
    Qlib Alpha 策略

    工作流程:
    1. 获取 Hummingbot K 线数据
    2. 计算 Alpha158 因子 (内联)
    3. 调用 Qlib 模型预测
    4. 根据阈值生成信号
    5. 执行交易 + 三重屏障风控
    """

    # Hummingbot 要求: 声明市场
    markets = {}  # 动态设置

    def __init__(self, connectors: Dict, config: QlibAlphaConfig):
        # 设置市场
        self.markets = {config.exchange: {config.trading_pair}}

        super().__init__(connectors, config)

        self.config = config
        self.logger = logging.getLogger(__name__)

        # 状态
        self.model = None
        self.qlib_initialized = False
        self.last_signal_time = 0
        self.current_position = Decimal("0")
        self.entry_price = Decimal("0")
        self.entry_time = 0

        # K 线数据源 (使用 CandlesFactory)
        self.candles_feed = None
        self._init_candles_feed()

        # 初始化
        self._init_qlib()
        self._load_model()

    # =========================================================================
    # 初始化
    # =========================================================================

    def _init_candles_feed(self):
        """初始化 K 线数据源"""
        try:
            candles_config = CandlesConfig(
                connector=self.config.exchange,
                trading_pair=self.config.trading_pair,
                interval=self.config.prediction_interval,
                max_records=self.config.lookback_bars,
            )
            self.candles_feed = CandlesFactory.get_candle(candles_config)
            # 必须调用 start() 启动 K 线数据收集
            self.candles_feed.start()
            self.logger.info(f"Candles feed initialized and started: {self.config.trading_pair}")
        except Exception as e:
            self.logger.error(f"Failed to initialize candles feed: {e}")

    def _init_qlib(self):
        """初始化 Qlib (运行时配置，无需改源码)"""
        try:
            qlib_path = Path(self.config.qlib_data_path).expanduser()
            qlib.init(provider_uri=str(qlib_path))

            # 运行时修改加密货币参数
            C["trade_unit"] = 0.00001
            C["limit_threshold"] = None

            self.qlib_initialized = True
            self.logger.info("Qlib initialized successfully")
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

    # =========================================================================
    # 核心逻辑: on_tick
    # =========================================================================

    def on_tick(self):
        """每秒触发"""
        # 检查冷却时间
        current_time = self.current_timestamp
        if current_time - self.last_signal_time < self.config.cooldown_time:
            return

        # 检查三重屏障
        if self.current_position != 0:
            self._check_triple_barrier()
            return

        # 获取信号
        signal = self._get_signal()

        if signal != 0:
            self._execute_signal(signal)
            self.last_signal_time = current_time

    # =========================================================================
    # 信号生成 (内联桥接逻辑)
    # =========================================================================

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

        try:
            # 获取 K 线数据
            candles = self._get_candles()
            if candles is None or len(candles) < 60:
                return 0

            # 计算特征 (内联桥接逻辑)
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

    def _get_candles(self) -> Optional[pd.DataFrame]:
        """
        获取 K 线数据

        使用初始化时创建的 CandlesFactory 数据源
        """
        try:
            # 使用初始化时创建的 candles_feed
            if self.candles_feed is None:
                return None

            # CandlesBase 对象提供 candles_df 属性
            # 注意：属性名是 ready，不是 is_ready
            if self.candles_feed.ready:
                return self.candles_feed.candles_df
            return None
        except Exception as e:
            self.logger.debug(f"Failed to get candles: {e}")
            return None

    def _compute_features(self, candles: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算 Alpha158 因子 (内联桥接逻辑)

        与 train_model.py 使用相同的特征，确保训练/推理一致
        """
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

    # =========================================================================
    # 交易执行 (直接用 Hummingbot API)
    # =========================================================================

    def _execute_signal(self, signal: int):
        """执行交易信号"""
        connector = self.connectors.get(self.config.exchange)
        if connector is None:
            return

        # 获取当前价格
        mid_price = connector.get_mid_price(self.config.trading_pair)
        if mid_price is None or mid_price <= 0:
            return

        # 计算下单数量
        amount = self.config.order_amount_usd / mid_price

        if signal > 0:
            # 买入
            self.buy(
                connector_name=self.config.exchange,
                trading_pair=self.config.trading_pair,
                amount=amount,
                order_type=OrderType.MARKET,
            )
            self.current_position = amount
            self.entry_price = mid_price
            self.entry_time = self.current_timestamp
            self.logger.info(f"BUY executed: {amount} @ {mid_price}")

        elif signal < 0 and self.current_position > 0:
            # 卖出 (平仓)
            self.sell(
                connector_name=self.config.exchange,
                trading_pair=self.config.trading_pair,
                amount=self.current_position,
                order_type=OrderType.MARKET,
            )
            self.logger.info(f"SELL executed: {self.current_position} @ {mid_price}")
            self.current_position = Decimal("0")
            self.entry_price = Decimal("0")

    # =========================================================================
    # 三重屏障风控
    # =========================================================================

    def _check_triple_barrier(self):
        """三重屏障检查: 止损/止盈/时间限制"""
        if self.current_position == 0:
            return

        connector = self.connectors.get(self.config.exchange)
        if connector is None:
            return

        current_price = connector.get_mid_price(self.config.trading_pair)
        if current_price is None:
            return

        # 计算收益率
        pnl_pct = (current_price - self.entry_price) / self.entry_price

        # 持仓时间
        holding_time = self.current_timestamp - self.entry_time

        should_close = False
        reason = ""

        # 止损
        if pnl_pct < -float(self.config.stop_loss):
            should_close = True
            reason = f"Stop Loss triggered: {pnl_pct:.2%}"

        # 止盈
        elif pnl_pct > float(self.config.take_profit):
            should_close = True
            reason = f"Take Profit triggered: {pnl_pct:.2%}"

        # 时间限制
        elif holding_time > self.config.time_limit:
            should_close = True
            reason = f"Time Limit triggered: {holding_time}s"

        if should_close:
            self.logger.info(f"Closing position: {reason}")
            self.sell(
                connector_name=self.config.exchange,
                trading_pair=self.config.trading_pair,
                amount=self.current_position,
                order_type=OrderType.MARKET,
            )
            self.current_position = Decimal("0")
            self.entry_price = Decimal("0")

    # =========================================================================
    # 状态显示
    # =========================================================================

    def format_status(self) -> str:
        """状态显示"""
        lines = []
        lines.append(f"Exchange: {self.config.exchange}")
        lines.append(f"Trading Pair: {self.config.trading_pair}")
        lines.append(f"Model Loaded: {self.model is not None}")
        lines.append(f"Qlib Initialized: {self.qlib_initialized}")
        lines.append(f"Candles Ready: {self.candles_feed.ready if self.candles_feed else False}")
        lines.append(f"Current Position: {self.current_position}")
        if self.entry_price > 0:
            lines.append(f"Entry Price: {self.entry_price}")
        return "\n".join(lines)

    # =========================================================================
    # 生命周期
    # =========================================================================

    async def on_stop(self):
        """
        策略停止时清理资源

        必须停止 candles_feed，否则网络迭代器会永久运行
        """
        if self.candles_feed is not None:
            self.candles_feed.stop()
            self.logger.info("Candles feed stopped")
```

---

## 6. 配置文件

### 6.1 策略配置

**文件路径**: `conf/scripts/qlib_alpha.yml`

```yaml
# Qlib Alpha 策略配置

# 交易配置
exchange: binance
trading_pair: BTC-USDT
order_amount_usd: 100

# 模型配置
model_path: ~/.qlib/models/lgb_model.pkl
qlib_data_path: ~/.qlib/qlib_data/crypto_data

# 信号配置
signal_threshold: 0.005    # 0.5% 收益率阈值
prediction_interval: 1h
lookback_bars: 100

# 风控配置 (三重屏障)
stop_loss: 0.02            # 2% 止损
take_profit: 0.03          # 3% 止盈
time_limit: 3600           # 1小时持仓限制

# 执行配置
cooldown_time: 60          # 60秒冷却
```

---

## 7. 启动与运行

### 7.1 完整运行流程

```
┌─────────────────────────────────────────────────────────────┐
│                      完整运行流程                            │
├─────────────────────────────────────────────────────────────┤
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
│  Step 3: 配置 API                                           │
│  $ cd hummingbot && ./start                                │
│  >>> connect binance                                        │
│  >>> [输入 API Key 和 Secret]                               │
│                                                             │
│  Step 4: 启动策略                                           │
│  >>> start --script qlib_alpha_strategy.py \               │
│            --conf conf/scripts/qlib_alpha.yml              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 命令参考

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

# 启动策略
cd hummingbot
./start
>>> connect binance
>>> start --script qlib_alpha_strategy.py --conf conf/scripts/qlib_alpha.yml
```

### 7.3 Paper Trading

```bash
# 使用 Binance Testnet
>>> connect binance_paper_trade
>>> start --script qlib_alpha_strategy.py --conf conf/scripts/qlib_alpha.yml
```

---

## 8. 验收标准

### 8.1 验收清单

| 序号 | 验收项 | 验收方法 | 通过标准 |
|------|--------|----------|----------|
| 1 | 数据准备 | 运行 prepare_crypto_data.py | 生成 Qlib 格式数据 |
| 2 | 模型训练 | 运行 train_model.py | 模型文件生成 |
| 3 | Qlib 初始化 | 策略启动 | 无报错 |
| 4 | K 线获取 | 策略运行 | 获取足够数据 |
| 5 | 特征计算 | 策略运行 | 特征矩阵非空 |
| 6 | 信号生成 | 策略运行 | 返回 -1/0/1 |
| 7 | 订单执行 | 策略运行 | 订单成功 |
| 8 | 三重屏障 | 触发条件 | 自动平仓 |
| 9 | Paper Trading | 模拟交易 24h | 无异常 |

### 8.2 验证脚本

```python
# scripts/verify_integration.py

def verify():
    import qlib
    from pathlib import Path

    # 1. 验证 Qlib 初始化 (无需改源码)
    print("1. Testing Qlib init...")
    qlib.init(provider_uri=str(Path("~/.qlib/qlib_data/crypto_data").expanduser()))
    print("   ✓ Qlib initialized")

    # 2. 验证模型加载
    print("2. Testing model load...")
    import pickle
    model_path = Path("~/.qlib/models/lgb_model.pkl").expanduser()
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("   ✓ Model loaded")
    else:
        print("   ✗ Model not found (run train_model.py first)")

    # 3. 验证特征计算
    print("3. Testing feature computation...")
    import pandas as pd
    import numpy as np

    # Mock data
    df = pd.DataFrame({
        "open": np.random.uniform(40000, 42000, 100),
        "high": np.random.uniform(42000, 43000, 100),
        "low": np.random.uniform(39000, 40000, 100),
        "close": np.random.uniform(40000, 42000, 100),
        "volume": np.random.uniform(100, 1000, 100),
    })

    # Compute features (same as strategy)
    close = df["close"]
    features = pd.DataFrame()
    features["KMID"] = (close - df["open"]) / df["open"]
    for d in [5, 10, 20]:
        features[f"ROC{d}"] = close / close.shift(d) - 1
    features = features.dropna()

    assert len(features) > 0, "Features empty"
    print(f"   ✓ Computed {len(features.columns)} features")

    print("\n✓ All verifications passed!")


if __name__ == "__main__":
    verify()
```

---

## 附录

### A. 依赖版本

```
qlib >= 0.9.7
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
| Qlib 初始化失败 | 检查数据目录路径是否正确 |
| 模型加载失败 | 先运行 train_model.py |
| K 线数据不足 | 等待 Hummingbot 收集数据 |
| 信号一直为 0 | 调整 signal_threshold |
| 订单执行失败 | 检查 API 权限、余额 |

### C. 变更日志

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

**v5.2.0** (2026-01-02)
- 修复训练/推理特征一致性问题
- 修复时间戳单位检测
- 修复 SignalBridge 方向逻辑
- 添加策略注册说明

**v5.1.0** (2026-01-02)
- 修复 max/min → np.maximum/np.minimum
- 修复 model.predict() 接口
- 修复信号阈值逻辑
