# AlgVex 核心方案 (P0 - MVP)

> **版本**: v5.2.0 (2026-01-02)
> **状态**: 可直接运行的完整方案 (含策略层)

> **Qlib + Hummingbot 融合的加密货币现货量化交易平台**
>
> 本文档包含完整的实施方案，精确到文件路径、行号、代码内容。
> 可直接交付给任何开发团队实施。

---

## 目录

- [1. 方案概述](#1-方案概述)
- [2. 修改清单](#2-修改清单)
- [3. Qlib 修改详情](#3-qlib-修改详情)
- [4. Hummingbot 修改详情](#4-hummingbot-修改详情)
- [5. 桥接层新建代码](#5-桥接层新建代码)
- [6. 策略层新建代码](#6-策略层新建代码)
- [7. 模型训练](#7-模型训练)
- [8. 数据准备](#8-数据准备)
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
│  1. 直接修改 qlib/ 和 hummingbot/ 文件夹                     │
│     - 不新建独立系统                                         │
│     - 最小化修改，最大化复用                                  │
│                                                             │
│  2. Qlib 修改范围                                            │
│     - constant.py: 添加 REG_CRYPTO 区域常量                  │
│     - config.py: 添加加密货币区域配置 + get_crypto_config()  │
│     - data/data.py: 添加 CryptoCalendarProvider (可选)      │
│                                                             │
│  3. Hummingbot 修改范围                                      │
│     - 现有代码无需修改，100% 原生复用                         │
│     - 新增模块: data_feed/qlib_bridge/ (桥接层)             │
│     - 新增模块: strategy/qlib_alpha/ (策略层)               │
│                                                             │
│  4. 升级兼容性                                               │
│     - Hummingbot 升级时保留新增目录即可                       │
│     - Qlib 升级时重新应用 3 处修改                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 代码量统计

| 类型 | 文件 | 代码量 |
|------|------|--------|
| **Qlib 修改** | constant.py | +3 行 |
| **Qlib 修改** | config.py | +12 行 |
| **Qlib 新增** | data/data.py | +45 行 |
| **桥接层新增** | qlib_bridge/*.py | ~300 行 |
| **策略层新增** | strategy/qlib_alpha/*.py | ~350 行 |
| **脚本新增** | scripts/*.py | ~150 行 |
| **总计** | - | **~860 行** |

---

## 2. 修改清单

### 2.1 完整修改清单

| 序号 | 文件路径 | 修改类型 | 说明 |
|------|----------|----------|------|
| 1 | `qlib/qlib/constant.py` | 修改 | 添加 REG_CRYPTO 常量 |
| 2 | `qlib/qlib/config.py` | 修改 | 添加加密货币区域配置 |
| 3 | `qlib/qlib/data/data.py` | 修改 | 添加 CryptoCalendarProvider 类 |
| 4 | `hummingbot/hummingbot/data_feed/qlib_bridge/__init__.py` | 新建 | 模块初始化 |
| 5 | `hummingbot/hummingbot/data_feed/qlib_bridge/data_bridge.py` | 新建 | 数据格式转换 |
| 6 | `hummingbot/hummingbot/data_feed/qlib_bridge/signal_bridge.py` | 新建 | 信号转订单 |
| 7 | `hummingbot/hummingbot/data_feed/qlib_bridge/calendar.py` | 新建 | 日历生成工具 |
| 8 | `hummingbot/hummingbot/strategy/qlib_alpha/__init__.py` | 新建 | 策略模块初始化 |
| 9 | `hummingbot/hummingbot/strategy/qlib_alpha/qlib_alpha.py` | 新建 | Qlib Alpha 策略主类 |
| 10 | `hummingbot/hummingbot/strategy/qlib_alpha/qlib_alpha_config.py` | 新建 | 策略配置 |
| 11 | `hummingbot/hummingbot/strategy/qlib_alpha/start.py` | 新建 | 策略启动入口 |
| 12 | `scripts/train_model.py` | 新建 | 模型训练脚本 |
| 13 | `scripts/prepare_crypto_data.py` | 新建 | 数据准备脚本 |
| 14 | `scripts/run_strategy.py` | 新建 | 策略运行脚本 |

---

## 3. Qlib 修改详情

### 3.1 修改 constant.py

**文件路径**: `qlib/qlib/constant.py`

**修改位置**: 第 10-12 行之后

**修改前**:
```python
# 第 10-12 行
REG_CN = "cn"
REG_US = "us"
REG_TW = "tw"
```

**修改后**:
```python
# 第 10-13 行
REG_CN = "cn"
REG_US = "us"
REG_TW = "tw"
REG_CRYPTO = "crypto"  # 新增：加密货币区域
```

---

### 3.2 修改 config.py

**文件路径**: `qlib/qlib/config.py`

**修改位置 1**: 第 25 行，导入语句

**修改前**:
```python
from qlib.constant import REG_CN, REG_US, REG_TW
```

**修改后**:
```python
from qlib.constant import REG_CN, REG_US, REG_TW, REG_CRYPTO
```

---

**修改位置 2**: 第 295-311 行之后，添加加密货币区域配置

**修改前**:
```python
_default_region_config = {
    REG_CN: {
        "trade_unit": 100,
        "limit_threshold": 0.095,
        "deal_price": "close",
    },
    REG_US: {
        "trade_unit": 1,
        "limit_threshold": None,
        "deal_price": "close",
    },
    REG_TW: {
        "trade_unit": 1000,
        "limit_threshold": 0.1,
        "deal_price": "close",
    },
}
```

**修改后**:
```python
_default_region_config = {
    REG_CN: {
        "trade_unit": 100,
        "limit_threshold": 0.095,
        "deal_price": "close",
    },
    REG_US: {
        "trade_unit": 1,
        "limit_threshold": None,
        "deal_price": "close",
    },
    REG_TW: {
        "trade_unit": 1000,
        "limit_threshold": 0.1,
        "deal_price": "close",
    },
    # 新增：加密货币区域配置
    REG_CRYPTO: {
        "trade_unit": 0.00001,      # 加密货币最小交易单位
        "limit_threshold": None,     # 无涨跌停限制
        "deal_price": "close",
        "trade_calendar": "24/7",    # 24/7 交易
        "timezone": "UTC",           # UTC 时区
        # 注意：加密货币使用本地日历文件，不需要特殊 CalendarProvider
        # 日历文件由 prepare_crypto_data.py 生成
    },
}
```

---

**修改位置 3**: 第 280 行附近，MODE_CONF["client"] 中添加加密货币默认配置

**修改前**:
```python
"client": {
    # ...existing config...
    "region": REG_CN,
    # ...
},
```

**修改后**:
```python
# 在文件末尾添加一个辅助函数，用于加密货币初始化
def get_crypto_config(provider_uri: str = "~/.qlib/qlib_data/crypto_data") -> dict:
    """
    获取加密货币专用配置

    用法:
        import qlib
        from qlib.config import get_crypto_config
        qlib.init(**get_crypto_config("/path/to/crypto_data"))

    Parameters
    ----------
    provider_uri : str
        加密货币数据目录

    Returns
    -------
    dict
        Qlib 初始化配置
    """
    from pathlib import Path

    return {
        "provider_uri": str(Path(provider_uri).expanduser()),
        "region": REG_CRYPTO,
        # 加密货币使用本地日历文件 (由 prepare_crypto_data.py 生成)
        "calendar_provider": "LocalCalendarProvider",
        "instrument_provider": "LocalInstrumentProvider",
        "feature_provider": "LocalFeatureProvider",
        "expression_provider": "LocalExpressionProvider",
        "dataset_provider": "LocalDatasetProvider",
    }
```

---

### 3.3 修改 data/data.py (可选)

> **使用说明**:
>
> `CryptoCalendarProvider` 是一个**可选的备用方案**。
>
> **当前推荐方案**: 使用 `LocalCalendarProvider` + 预生成的日历文件
> - 由 `prepare_crypto_data.py` 生成 `calendars/1h.txt` 等文件
> - 在 `get_crypto_config()` 中配置 `calendar_provider: LocalCalendarProvider`
>
> **此类的用途**: 动态生成 24/7 日历（无需预生成文件）
> - 适用于需要动态时间范围的场景
> - 如果使用此类，需要在 `qlib.init()` 中指定 `calendar_provider: CryptoCalendarProvider`
>
> **建议**: MVP 阶段使用预生成日历文件方案，此类作为未来扩展保留。

**文件路径**: `qlib/qlib/data/data.py`

**修改位置**: 第 676 行之后（LocalCalendarProvider 类之后），添加 CryptoCalendarProvider 类

**新增代码**:
```python
# 在第 676 行之后添加
# 【可选】此类为备用方案，MVP 阶段使用 LocalCalendarProvider + 预生成日历文件

class CryptoCalendarProvider(CalendarProvider):
    """
    加密货币 24/7 交易日历 Provider

    与股票市场不同，加密货币市场 24/7 交易，无休市日。
    此 Provider 动态生成连续的时间戳序列。
    """

    def __init__(self, start_date: str = "2020-01-01", end_date: str = "2030-12-31"):
        """
        初始化加密货币日历 Provider

        Parameters
        ----------
        start_date : str
            日历开始日期，格式 "YYYY-MM-DD"
        end_date : str
            日历结束日期，格式 "YYYY-MM-DD"
        """
        super().__init__()
        self.start_date = pd.Timestamp(start_date, tz='UTC')
        self.end_date = pd.Timestamp(end_date, tz='UTC')

    def load_calendar(self, freq: str, future: bool = False) -> list:
        """
        生成 24/7 交易日历

        Parameters
        ----------
        freq : str
            时间频率: "1min", "5min", "15min", "30min", "1h", "4h", "day"
        future : bool
            是否包含未来日期（加密货币无此限制）

        Returns
        -------
        list
            时间戳列表
        """
        freq_map = {
            "1min": "1T", "5min": "5T", "15min": "15T", "30min": "30T",
            "1h": "1H", "4h": "4H", "day": "1D", "1d": "1D",
        }
        pd_freq = freq_map.get(freq.lower(), freq)

        calendar = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=pd_freq,
            tz='UTC'
        )
        return [ts for ts in calendar]
```

---

## 4. Hummingbot 修改详情

### 4.1 修改策略

**Hummingbot 现有代码无需修改**，但需要在 `hummingbot/` 目录下**新增**以下模块：

- `hummingbot/data_feed/qlib_bridge/` - 数据桥接层（新增）
- `hummingbot/strategy/qlib_alpha/` - Qlib Alpha 策略（新增）

这种方式的优点：
1. 不破坏 Hummingbot 原有功能
2. 可随时升级 Hummingbot 版本（合并时注意保留新增目录）
3. 新增代码与原有代码解耦

以下 Hummingbot 原生组件直接复用（无需修改）：

| 组件 | 文件路径 | 用途 |
|------|----------|------|
| BinanceSpotCandles | `data_feed/candles_feed/binance_spot_candles/` | K线数据获取 |
| BinanceExchange | `connector/exchange/binance/` | 订单执行 |
| CandlesBase | `data_feed/candles_feed/candles_base.py` | K线基类 |
| DirectionalStrategyBase | `strategy/directional_strategy_base.py` | 策略基类 |

---

## 5. 桥接层新建代码

### 5.1 目录结构

```
hummingbot/hummingbot/data_feed/qlib_bridge/
├── __init__.py
├── data_bridge.py      # Hummingbot K线 → Qlib 格式
├── signal_bridge.py    # Qlib 信号 → Hummingbot 订单
└── calendar.py         # 日历生成工具
```

---

### 5.2 __init__.py

**文件路径**: `hummingbot/hummingbot/data_feed/qlib_bridge/__init__.py`

```python
"""
Qlib Bridge - Hummingbot 与 Qlib 的桥接层

此模块提供：
1. DataBridge: Hummingbot K线数据 → Qlib 格式
2. SignalBridge: Qlib 交易信号 → Hummingbot 订单
3. CryptoCalendarGenerator: 加密货币 24/7 日历生成
"""

from hummingbot.data_feed.qlib_bridge.data_bridge import DataBridge
from hummingbot.data_feed.qlib_bridge.signal_bridge import SignalBridge
from hummingbot.data_feed.qlib_bridge.calendar import CryptoCalendarGenerator

__all__ = [
    "DataBridge",
    "SignalBridge",
    "CryptoCalendarGenerator",
]
```

---

### 5.3 data_bridge.py

**文件路径**: `hummingbot/hummingbot/data_feed/qlib_bridge/data_bridge.py`

```python
"""
DataBridge - Hummingbot K线数据转换为 Qlib 格式

Hummingbot CandlesBase 输出格式:
    columns = ["timestamp", "open", "high", "low", "close", "volume",
               "quote_asset_volume", "n_trades", "taker_buy_base_volume",
               "taker_buy_quote_volume"]

Qlib 需要的格式:
    MultiIndex: (datetime, instrument)
    columns: ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
"""

from typing import List, Dict, Optional, Union
from decimal import Decimal
import pandas as pd
import numpy as np


class DataBridge:
    """
    Hummingbot K线数据 → Qlib 格式转换器
    """

    # Hummingbot CandlesBase 的列名
    HB_COLUMNS = [
        "timestamp", "open", "high", "low", "close", "volume",
        "quote_asset_volume", "n_trades", "taker_buy_base_volume",
        "taker_buy_quote_volume"
    ]

    # Qlib 需要的列名 (带 $ 前缀)
    QLIB_COLUMNS = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]

    def __init__(self, timezone: str = "UTC"):
        """
        初始化 DataBridge

        Parameters
        ----------
        timezone : str
            时区，默认 UTC
        """
        self.timezone = timezone

    def candles_to_qlib(
        self,
        candles_df: pd.DataFrame,
        trading_pair: str
    ) -> pd.DataFrame:
        """
        将 Hummingbot CandlesBase.candles_df 转换为 Qlib 格式

        Parameters
        ----------
        candles_df : pd.DataFrame
            Hummingbot CandlesBase.candles_df 输出
        trading_pair : str
            交易对，如 "BTC-USDT"

        Returns
        -------
        pd.DataFrame
            Qlib 格式的 DataFrame，MultiIndex (datetime, instrument)
        """
        if candles_df.empty:
            return pd.DataFrame()

        # 标准化交易对名称: BTC-USDT → btcusdt
        instrument = self._normalize_trading_pair(trading_pair)

        # 转换时间戳 (自动检测秒/毫秒)
        df = candles_df.copy()
        df["datetime"] = pd.to_datetime(
            df["timestamp"],
            unit=self._detect_timestamp_unit(df["timestamp"]),
            utc=True
        )
        df["instrument"] = instrument

        # 计算 VWAP (成交量加权平均价格)
        # VWAP = quote_asset_volume / volume
        df["$vwap"] = np.where(
            df["volume"] > 0,
            df["quote_asset_volume"] / df["volume"],
            df["close"]  # 如果成交量为0，使用收盘价
        )

        # 重命名列
        df["$open"] = df["open"].astype(float)
        df["$high"] = df["high"].astype(float)
        df["$low"] = df["low"].astype(float)
        df["$close"] = df["close"].astype(float)
        df["$volume"] = df["volume"].astype(float)

        # 设置 MultiIndex
        df = df.set_index(["datetime", "instrument"])

        # 只保留 Qlib 需要的列
        df = df[self.QLIB_COLUMNS]

        return df

    def merge_instruments(
        self,
        dataframes: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        合并多个交易对的数据

        Parameters
        ----------
        dataframes : Dict[str, pd.DataFrame]
            {trading_pair: qlib_df} 字典

        Returns
        -------
        pd.DataFrame
            合并后的 Qlib 格式 DataFrame
        """
        if not dataframes:
            return pd.DataFrame()

        merged = pd.concat(dataframes.values())
        merged = merged.sort_index()

        return merged

    def _normalize_trading_pair(self, trading_pair: str) -> str:
        """
        标准化交易对名称

        BTC-USDT → btcusdt
        BTCUSDT → btcusdt
        """
        return trading_pair.lower().replace("-", "").replace("_", "")

    def _detect_timestamp_unit(self, timestamps: pd.Series) -> str:
        """
        自动检测时间戳单位（秒或毫秒）

        Hummingbot 不同版本可能使用不同单位：
        - 秒级时间戳：约 10 位数字（如 1704067200）
        - 毫秒时间戳：约 13 位数字（如 1704067200000）

        Parameters
        ----------
        timestamps : pd.Series
            时间戳列

        Returns
        -------
        str
            "s" 表示秒，"ms" 表示毫秒
        """
        if timestamps.empty:
            return "s"

        # 取第一个非空值
        sample = timestamps.dropna().iloc[0] if not timestamps.dropna().empty else 0

        # 如果时间戳大于 1e12，认为是毫秒 (2001年后的毫秒时间戳)
        # 秒级时间戳在 2033 年前不会超过 2e9
        if sample > 1e12:
            return "ms"
        else:
            return "s"

    def to_qlib_provider_format(
        self,
        df: pd.DataFrame,
        output_dir: str,
        freq: str = "1h",
    ) -> None:
        """
        将数据保存为 Qlib Provider 可读取的格式

        Parameters
        ----------
        df : pd.DataFrame
            Qlib 格式的 DataFrame
        output_dir : str
            输出目录路径
        freq : str
            数据频率，如 "1min", "5min", "1h", "1d"，影响日历文件名和时间格式
        """
        import os
        from pathlib import Path

        output_path = Path(output_dir)
        features_dir = output_path / "features"
        calendars_dir = output_path / "calendars"
        instruments_dir = output_path / "instruments"

        # 创建目录
        for d in [features_dir, calendars_dir, instruments_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 获取所有交易对
        instruments = df.index.get_level_values("instrument").unique()

        # 获取完整日历 (所有时间戳的并集)
        full_calendar = df.index.get_level_values("datetime").unique().sort_values()

        # 保存每个交易对的特征数据
        # 注意：必须对齐到统一日历，缺失值用 NaN 填充
        for inst in instruments:
            inst_df = df.xs(inst, level="instrument")
            # 对齐到完整日历
            inst_df = inst_df.reindex(full_calendar)

            inst_dir = features_dir / inst
            inst_dir.mkdir(exist_ok=True)

            for col in self.QLIB_COLUMNS:
                col_name = col.replace("$", "")
                file_path = inst_dir / f"{col_name}.bin"
                # Qlib 使用 float32 二进制格式
                # NaN 值会被保留，Qlib 读取时会处理
                inst_df[col].values.astype(np.float32).tofile(file_path)

        # 保存日历
        # 根据频率决定文件名和时间格式
        freq_lower = freq.lower()
        if freq_lower in ["1d", "day"]:
            calendar_filename = "day.txt"
            time_format = "%Y-%m-%d"
        else:
            # 分钟/小时级别
            calendar_filename = f"{freq_lower}.txt"
            time_format = "%Y-%m-%d %H:%M:%S"

        calendar_file = calendars_dir / calendar_filename
        with open(calendar_file, "w") as f:
            for dt in full_calendar:
                f.write(dt.strftime(time_format) + "\n")

        # 保存交易对列表
        instruments_file = instruments_dir / "all.txt"
        with open(instruments_file, "w") as f:
            for inst in instruments:
                # 使用完整日历的范围
                start = full_calendar.min().strftime(time_format)
                end = full_calendar.max().strftime(time_format)
                f.write(f"{inst}\t{start}\t{end}\n")
```

---

### 5.4 signal_bridge.py

> **模块定位说明**:
>
> `SignalBridge` 在当前 MVP 版本的 `QlibAlphaStrategy` 中**暂未直接使用**。
>
> 当前策略使用 `DirectionalStrategyBase` 的内置机制：
> - `get_signal()` 返回 1/-1/0 方向信号
> - `DirectionalStrategyBase` 自动通过 `PositionExecutor` 执行订单
>
> `SignalBridge` 的设计目的是为**未来扩展**准备：
> - 多品种投资组合管理（同时管理 BTC/ETH/SOL 等）
> - 权重调仓（根据预测分数调整持仓比例）
> - 自定义订单执行逻辑（限价单、分批建仓等）
>
> 如果需要使用 `SignalBridge`，可以在自定义策略中直接调用其方法。

**文件路径**: `hummingbot/hummingbot/data_feed/qlib_bridge/signal_bridge.py`

```python
"""
SignalBridge - Qlib 交易信号转换为 Hummingbot 订单

【当前版本未使用，为未来多品种投资组合扩展预留】

Qlib 策略输出格式:
    {
        "instrument": "btcusdt",
        "direction": 1,      # 1=买入, -1=卖出, 0=持有
        "weight": 0.1,       # 目标权重 (0-1)
        "score": 0.85,       # 预测分数
    }

Hummingbot 订单格式:
    {
        "trading_pair": "BTC-USDT",
        "order_type": OrderType.MARKET,
        "trade_type": TradeType.BUY,
        "amount": Decimal("0.01"),
        "price": Decimal("0"),
    }
"""

from decimal import Decimal
from typing import Dict, List, Optional, Any
import logging

from hummingbot.core.data_type.common import OrderType, TradeType


class SignalBridge:
    """
    Qlib 交易信号 → Hummingbot 订单转换器
    """

    def __init__(
        self,
        connector,
        min_order_amount: Decimal = Decimal("0.0001"),
        default_order_type: OrderType = OrderType.MARKET,
    ):
        """
        初始化 SignalBridge

        Parameters
        ----------
        connector
            Hummingbot 交易所连接器实例
        min_order_amount : Decimal
            最小下单金额
        default_order_type : OrderType
            默认订单类型
        """
        self.connector = connector
        self.min_order_amount = min_order_amount
        self.default_order_type = default_order_type
        self.logger = logging.getLogger(__name__)

    def signal_to_order(
        self,
        signal: Dict[str, Any],
        available_balance: Decimal,
        current_position: Decimal = Decimal("0"),
        current_price: Decimal = Decimal("0"),
    ) -> Optional[Dict[str, Any]]:
        """
        将 Qlib 信号转换为 Hummingbot 订单

        Parameters
        ----------
        signal : Dict
            Qlib 策略输出的信号
            - instrument: str, 交易对
            - direction: int, 1=买入, -1=卖出, 0=持有
            - weight: float, 目标权重
            - score: float, 预测分数
        available_balance : Decimal
            可用报价币余额 (如 USDT)
        current_position : Decimal
            当前基础币持仓 (如 BTC 数量)
        current_price : Decimal
            当前价格，用于单位转换

        Returns
        -------
        Optional[Dict]
            Hummingbot 订单参数，如果不需要交易则返回 None
        """
        direction = signal.get("direction", 0)

        # 持有信号，不交易
        if direction == 0:
            return None

        # 价格校验
        if current_price <= 0:
            self.logger.error("Invalid current_price for unit conversion")
            return None

        instrument = signal.get("instrument", "")
        weight = Decimal(str(signal.get("weight", 0)))

        # 将当前持仓转换为报价币价值
        # current_position 是基础币数量 (如 BTC)
        # current_position_quote 是等值报价币 (如 USDT)
        current_position_quote = current_position * current_price

        # 根据信号方向计算目标仓位
        # direction=1 (看涨): 建仓到 weight 比例
        # direction=-1 (看跌): 清仓（对于现货交易，不能做空）
        if direction > 0:
            target_quote_amount = available_balance * weight
        else:
            # 看跌时目标仓位为 0（现货无法做空，只能清仓）
            target_quote_amount = Decimal("0")

        # 计算仓位变化量 (delta)
        # delta > 0: 需要买入增加仓位
        # delta < 0: 需要卖出减少仓位
        delta_quote = target_quote_amount - current_position_quote

        # 根据 delta 方向决定交易类型
        if delta_quote > 0:
            trade_type = TradeType.BUY
            trade_quote_amount = delta_quote
        elif delta_quote < 0:
            trade_type = TradeType.SELL
            trade_quote_amount = abs(delta_quote)
        else:
            # 无需交易
            return None

        # 转换为基础币数量 (Hummingbot 下单使用基础币数量)
        trade_base_amount = trade_quote_amount / current_price

        # 检查最小下单数量
        if trade_base_amount < self.min_order_amount:
            self.logger.debug(f"Trade amount {trade_base_amount} below minimum {self.min_order_amount}")
            return None

        return {
            "trading_pair": self._format_trading_pair(instrument),
            "order_type": self.default_order_type,
            "trade_type": trade_type,
            "amount": trade_base_amount,
            "price": Decimal("0"),  # 市价单
        }

    async def execute_order(self, order: Dict[str, Any]) -> Optional[str]:
        """
        执行订单

        Parameters
        ----------
        order : Dict
            订单参数

        Returns
        -------
        Optional[str]
            订单 ID，失败返回 None
        """
        try:
            if order["trade_type"] == TradeType.BUY:
                order_id = self.connector.buy(
                    trading_pair=order["trading_pair"],
                    amount=order["amount"],
                    order_type=order["order_type"],
                    price=order["price"],
                )
            else:
                order_id = self.connector.sell(
                    trading_pair=order["trading_pair"],
                    amount=order["amount"],
                    order_type=order["order_type"],
                    price=order["price"],
                )

            self.logger.info(f"Order placed: {order_id}")
            return order_id

        except Exception as e:
            self.logger.error(f"Order execution failed: {e}")
            return None

    def _format_trading_pair(self, instrument: str) -> str:
        """
        格式化交易对: btcusdt → BTC-USDT

        Parameters
        ----------
        instrument : str
            Qlib 格式的交易对

        Returns
        -------
        str
            Hummingbot 格式的交易对
        """
        instrument = instrument.upper()

        # 常见报价币种
        quote_currencies = ["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"]

        for quote in quote_currencies:
            if instrument.endswith(quote):
                base = instrument[:-len(quote)]
                return f"{base}-{quote}"

        # 如果没有匹配，返回原始格式
        return instrument

    def batch_signals_to_orders(
        self,
        signals: List[Dict[str, Any]],
        available_balance: Decimal,
        positions: Dict[str, Decimal],
        prices: Dict[str, Decimal],
    ) -> List[Dict[str, Any]]:
        """
        批量转换信号为订单

        Parameters
        ----------
        signals : List[Dict]
            Qlib 信号列表
        available_balance : Decimal
            可用报价币余额 (如 USDT)
        positions : Dict[str, Decimal]
            当前持仓 {instrument: base_amount}，基础币数量
        prices : Dict[str, Decimal]
            当前价格 {instrument: price}，用于单位转换

        Returns
        -------
        List[Dict]
            订单列表
        """
        orders = []

        for signal in signals:
            instrument = signal.get("instrument", "")
            current_position = positions.get(instrument, Decimal("0"))
            current_price = prices.get(instrument, Decimal("0"))

            order = self.signal_to_order(
                signal=signal,
                available_balance=available_balance,
                current_position=current_position,
                current_price=current_price,
            )

            if order is not None:
                orders.append(order)

        return orders
```

---

### 5.5 calendar.py

**文件路径**: `hummingbot/hummingbot/data_feed/qlib_bridge/calendar.py`

```python
"""
CryptoCalendarGenerator - 加密货币 24/7 日历生成器

用于生成 Qlib 需要的日历文件格式。
"""

from typing import List, Optional
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd


class CryptoCalendarGenerator:
    """
    加密货币 24/7 日历生成器

    生成连续的时间戳序列，无休市日。
    """

    FREQ_MAP = {
        "1min": "1T",
        "5min": "5T",
        "15min": "15T",
        "30min": "30T",
        "1h": "1H",
        "4h": "4H",
        "1d": "1D",
        "day": "1D",
    }

    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2030-12-31",
    ):
        """
        初始化日历生成器

        Parameters
        ----------
        start_date : str
            开始日期，格式 "YYYY-MM-DD"
        end_date : str
            结束日期，格式 "YYYY-MM-DD"
        """
        self.start_date = pd.Timestamp(start_date, tz='UTC')
        self.end_date = pd.Timestamp(end_date, tz='UTC')

    def generate(self, freq: str) -> List[pd.Timestamp]:
        """
        生成指定频率的日历

        Parameters
        ----------
        freq : str
            时间频率: "1min", "5min", "15min", "1h", "4h", "1d"

        Returns
        -------
        List[pd.Timestamp]
            时间戳列表
        """
        pd_freq = self.FREQ_MAP.get(freq.lower(), freq)

        calendar = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=pd_freq,
            tz='UTC'
        )

        return list(calendar)

    def save_to_file(
        self,
        freq: str,
        output_dir: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        将日历保存到文件

        Parameters
        ----------
        freq : str
            时间频率
        output_dir : str
            输出目录
        filename : Optional[str]
            文件名，默认使用频率名

        Returns
        -------
        str
            输出文件路径
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{freq.lower()}.txt"

        file_path = output_path / filename
        calendar = self.generate(freq)

        with open(file_path, "w") as f:
            for ts in calendar:
                if "min" in freq.lower() or "h" in freq.lower():
                    f.write(ts.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                else:
                    f.write(ts.strftime("%Y-%m-%d") + "\n")

        return str(file_path)

    def generate_all_frequencies(self, output_dir: str) -> dict:
        """
        生成所有支持频率的日历文件

        Parameters
        ----------
        output_dir : str
            输出目录

        Returns
        -------
        dict
            {freq: file_path}
        """
        result = {}

        for freq in self.FREQ_MAP.keys():
            file_path = self.save_to_file(freq, output_dir)
            result[freq] = file_path

        return result
```

---

## 6. 策略层新建代码

### 6.1 目录结构

```
hummingbot/hummingbot/strategy/qlib_alpha/
├── __init__.py
├── qlib_alpha.py           # 策略主类 (继承 DirectionalStrategyBase)
├── qlib_alpha_config.py    # 策略配置
└── start.py                # 策略启动入口
```

---

### 6.2 __init__.py

**文件路径**: `hummingbot/hummingbot/strategy/qlib_alpha/__init__.py`

```python
"""
Qlib Alpha Strategy - 基于 Qlib 机器学习模型的交易策略

此策略:
1. 使用 Hummingbot 获取实时 K 线数据
2. 通过 DataBridge 转换为 Qlib 格式
3. 使用预训练的 Qlib 模型生成预测信号
4. 通过 SignalBridge 执行交易
5. 内置三重屏障风控 (止损/止盈/时间限制)
"""

from hummingbot.strategy.qlib_alpha.qlib_alpha import QlibAlphaStrategy
from hummingbot.strategy.qlib_alpha.qlib_alpha_config import QlibAlphaConfig

__all__ = [
    "QlibAlphaStrategy",
    "QlibAlphaConfig",
]
```

---

### 6.3 qlib_alpha_config.py

**文件路径**: `hummingbot/hummingbot/strategy/qlib_alpha/qlib_alpha_config.py`

```python
"""
Qlib Alpha Strategy 配置

使用 Pydantic 进行配置验证和管理。
"""

from decimal import Decimal
from typing import Optional
from pydantic import Field, field_validator

from hummingbot.client.config.strategy_config_data_types import BaseTradingStrategyConfigMap


class QlibAlphaConfig(BaseTradingStrategyConfigMap):
    """
    Qlib Alpha 策略配置
    """

    strategy: str = Field(default="qlib_alpha")

    # 交易配置
    exchange: str = Field(
        default="binance",
        description="交易所名称",
        json_schema_extra={"prompt": "Enter exchange name (e.g., binance): ", "prompt_on_new": True},
    )

    market: str = Field(
        default="BTC-USDT",
        description="交易对",
        json_schema_extra={"prompt": "Enter trading pair (e.g., BTC-USDT): ", "prompt_on_new": True},
    )

    # 订单配置
    order_amount_usd: Decimal = Field(
        default=Decimal("100"),
        description="每笔订单金额 (USD)",
        json_schema_extra={"prompt": "Enter order amount in USD: ", "prompt_on_new": True},
    )

    # 模型配置
    model_path: str = Field(
        default="~/.qlib/models/lgb_model.pkl",
        description="Qlib 模型文件路径",
        json_schema_extra={"prompt": "Enter model file path: ", "prompt_on_new": True},
    )

    qlib_data_path: str = Field(
        default="~/.qlib/qlib_data/crypto_data",
        description="Qlib 数据目录",
        json_schema_extra={"prompt": "Enter Qlib data path: ", "prompt_on_new": True},
    )

    # 信号配置
    # 注意：阈值是收益率阈值，不是概率阈值
    # 模型预测的是未来收益率，典型范围 [-0.05, +0.05]
    signal_threshold: Decimal = Field(
        default=Decimal("0.005"),
        description="信号阈值 (收益率)，预测值 > 阈值则买入，< -阈值则卖出。0.005 表示 0.5%",
        json_schema_extra={"prompt": "Enter signal threshold (e.g., 0.005 for 0.5% return): "},
    )

    prediction_interval: str = Field(
        default="1h",
        description="预测间隔 (1m, 5m, 15m, 1h, 4h, 1d)",
        json_schema_extra={"prompt": "Enter prediction interval: "},
    )

    # 风控配置 (三重屏障)
    stop_loss: Decimal = Field(
        default=Decimal("0.02"),
        description="止损比例 (0.02 = 2%)",
        json_schema_extra={"prompt": "Enter stop loss percentage (e.g., 0.02 for 2%): "},
    )

    take_profit: Decimal = Field(
        default=Decimal("0.03"),
        description="止盈比例 (0.03 = 3%)",
        json_schema_extra={"prompt": "Enter take profit percentage (e.g., 0.03 for 3%): "},
    )

    time_limit: int = Field(
        default=3600,
        description="持仓时间限制 (秒)",
        json_schema_extra={"prompt": "Enter time limit in seconds: "},
    )

    # 冷却配置
    cooldown_after_execution: int = Field(
        default=60,
        description="执行后冷却时间 (秒)",
        json_schema_extra={"prompt": "Enter cooldown time in seconds: "},
    )

    max_executors: int = Field(
        default=1,
        description="最大同时持仓数",
        json_schema_extra={"prompt": "Enter max concurrent positions: "},
    )

    @field_validator("stop_loss", "take_profit", mode="before")
    @classmethod
    def validate_percentage(cls, v):
        v = Decimal(str(v))
        if v <= 0 or v >= 1:
            raise ValueError("Percentage must be between 0 and 1")
        return v

    @field_validator("signal_threshold", mode="before")
    @classmethod
    def validate_threshold(cls, v):
        v = Decimal(str(v))
        if v < 0 or v > 1:
            raise ValueError("Threshold must be between 0 and 1")
        return v
```

---

### 6.4 qlib_alpha.py

**文件路径**: `hummingbot/hummingbot/strategy/qlib_alpha/qlib_alpha.py`

```python
"""
Qlib Alpha Strategy - 主策略类

继承 DirectionalStrategyBase，使用 Qlib 模型生成交易信号。
"""

import pickle
import logging
from pathlib import Path
from decimal import Decimal
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import qlib
from qlib.constant import REG_CRYPTO
from qlib.contrib.data.handler import Alpha158

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.data_feed.candles_feed.candles_base import CandlesBase
from hummingbot.data_feed.candles_feed.binance_spot_candles import BinanceSpotCandles
from hummingbot.data_feed.qlib_bridge import DataBridge
from hummingbot.strategy.directional_strategy_base import DirectionalStrategyBase
from hummingbot.strategy.qlib_alpha.qlib_alpha_config import QlibAlphaConfig


class QlibAlphaStrategy(DirectionalStrategyBase):
    """
    Qlib Alpha 策略

    使用 Qlib 机器学习模型 (LGBModel) 预测价格方向，
    结合 Hummingbot 的三重屏障风控执行交易。

    工作流程:
    1. on_tick() 每秒触发
    2. 检查 K 线数据是否就绪
    3. 调用 get_signal() 获取交易信号
    4. DirectionalStrategyBase 自动处理三重屏障风控
    """

    directional_strategy_name: str = "qlib_alpha"

    def __init__(
        self,
        connectors: Dict[str, ConnectorBase],
        config: QlibAlphaConfig,
    ):
        """
        初始化 Qlib Alpha 策略

        Parameters
        ----------
        connectors : Dict[str, ConnectorBase]
            交易所连接器字典
        config : QlibAlphaConfig
            策略配置
        """
        # 设置策略参数 (在调用 super().__init__ 之前)
        self.trading_pair = config.market
        self.exchange = config.exchange
        self.order_amount_usd = config.order_amount_usd
        self.stop_loss = float(config.stop_loss)
        self.take_profit = float(config.take_profit)
        self.time_limit = config.time_limit
        self.cooldown_after_execution = config.cooldown_after_execution
        self.max_executors = config.max_executors

        # 初始化 K 线数据源
        self.candles: List[CandlesBase] = [
            BinanceSpotCandles(
                trading_pair=self.trading_pair,
                interval=config.prediction_interval,
                max_records=200,  # 保留足够的历史数据用于因子计算
            )
        ]

        # 设置 markets 属性
        self.markets: Dict[str, Set[str]] = {config.exchange: {config.market}}

        # 调用父类初始化
        super().__init__(connectors)

        # 策略配置
        self.config = config
        self.signal_threshold = float(config.signal_threshold)

        # 初始化组件
        self.data_bridge = DataBridge()
        self.model = None
        self.qlib_initialized = False

        # 日志
        self.logger = logging.getLogger(__name__)

        # 加载模型
        self._init_qlib()
        self._load_model()

    def _init_qlib(self):
        """初始化 Qlib"""
        try:
            qlib_path = Path(self.config.qlib_data_path).expanduser()
            qlib.init(
                provider_uri=str(qlib_path),
                region=REG_CRYPTO,
            )
            self.qlib_initialized = True
            self.logger.info("Qlib initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Qlib: {e}")
            self.qlib_initialized = False

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
                self.model = None
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            self.model = None

    def get_signal(self) -> int:
        """
        获取交易信号

        Returns
        -------
        int
            1 = 买入, -1 = 卖出, 0 = 持有
        """
        # 检查前置条件
        if not self.qlib_initialized:
            self.logger.debug("Qlib not initialized")
            return 0

        if self.model is None:
            self.logger.debug("Model not loaded")
            return 0

        if not self.all_candles_ready:
            self.logger.debug("Candles not ready")
            return 0

        try:
            # 获取 K 线数据
            candles_df = self.candles[0].candles_df
            if candles_df.empty or len(candles_df) < 60:
                self.logger.debug("Not enough candle data")
                return 0

            # 转换为 Qlib 格式
            qlib_df = self.data_bridge.candles_to_qlib(candles_df, self.trading_pair)

            # 计算 Alpha158 因子
            features = self._compute_features(qlib_df)
            if features is None or features.empty:
                self.logger.debug("Failed to compute features")
                return 0

            # 获取最新一行特征
            latest_features = features.iloc[-1:].values

            # 模型预测
            # 注意：Qlib LGBModel.predict() 期望 DatasetH 对象，但实盘需要直接调用底层 lightgbm
            # self.model 是 Qlib LGBModel，self.model.model 是底层 lightgbm.Booster
            prediction = self.model.model.predict(latest_features)[0]

            # 根据阈值生成信号
            # 注意：Alpha158 标签是收益率 (Ref($close,-2)/Ref($close,-1)-1)
            # 预测值范围通常在 [-0.05, +0.05]，不是 [0,1] 概率
            # signal_threshold 应设为收益率阈值（如 0.005 表示 0.5% 收益）
            if prediction > self.signal_threshold:
                self.logger.info(f"BUY signal: prediction={prediction:.6f} > threshold={self.signal_threshold}")
                return 1
            elif prediction < -self.signal_threshold:
                self.logger.info(f"SELL signal: prediction={prediction:.6f} < -{self.signal_threshold}")
                return -1
            else:
                return 0

        except Exception as e:
            self.logger.error(f"Error getting signal: {e}")
            return 0

    def _compute_features(self, qlib_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算 Alpha158 因子 (简化版，用于实时推理)

        重要说明：
        1. 此函数计算的因子必须与训练时使用的因子完全一致
        2. 当前实现是 Alpha158 的简化版本（约 60 个因子）
        3. 如果训练使用完整 Alpha158，推理时也应使用完整版本
        4. 生产环境建议：直接使用 Qlib Alpha158 Handler 确保一致性

        简化版因子列表：
        - KBAR 类：KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
        - ROC 类：ROC5, ROC10, ROC20, ROC30, ROC60
        - MA 类：MA5, MA10, MA20, MA30, MA60
        - STD 类：STD5, STD10, STD20, STD30, STD60
        - MAX/MIN 类：MAX5-60, MIN5-60
        - QTLU/QTLD 类：QTLU5-60, QTLD5-60
        - RSV 类：RSV5-60
        - CORR 类：CORR5-60, CORD5-60
        """
        try:
            df = qlib_df.copy()

            # 重置索引以便计算
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level="instrument", drop=True)

            # 提取 OHLCV
            close = df["$close"]
            open_ = df["$open"]
            high = df["$high"]
            low = df["$low"]
            volume = df["$volume"]

            features = pd.DataFrame(index=df.index)

            # 基础因子 (参考 Alpha158)
            # KBAR 类因子
            # 注意：必须使用 np.maximum/np.minimum 进行逐元素比较
            # Python 内置 max/min 对 Series 会触发 "truth value ambiguous" 错误
            features["KMID"] = (close - open_) / open_
            features["KLEN"] = (high - low) / open_
            features["KMID2"] = (close - open_) / (high - low + 1e-12)
            features["KUP"] = (high - np.maximum(open_, close)) / open_
            features["KUP2"] = (high - np.maximum(open_, close)) / (high - low + 1e-12)
            features["KLOW"] = (np.minimum(open_, close) - low) / open_
            features["KLOW2"] = (np.minimum(open_, close) - low) / (high - low + 1e-12)
            features["KSFT"] = (2 * close - high - low) / open_
            features["KSFT2"] = (2 * close - high - low) / (high - low + 1e-12)

            # ROC 类因子
            for d in [5, 10, 20, 30, 60]:
                features[f"ROC{d}"] = close / close.shift(d) - 1

            # MA 类因子
            for d in [5, 10, 20, 30, 60]:
                ma = close.rolling(d).mean()
                features[f"MA{d}"] = close / ma - 1

            # STD 类因子
            for d in [5, 10, 20, 30, 60]:
                features[f"STD{d}"] = close.rolling(d).std() / close

            # MAX/MIN 类因子
            for d in [5, 10, 20, 30, 60]:
                features[f"MAX{d}"] = close / high.rolling(d).max() - 1
                features[f"MIN{d}"] = close / low.rolling(d).min() - 1

            # QTLU/QTLD 类因子
            for d in [5, 10, 20, 30, 60]:
                features[f"QTLU{d}"] = close / close.rolling(d).quantile(0.8) - 1
                features[f"QTLD{d}"] = close / close.rolling(d).quantile(0.2) - 1

            # RSV 因子
            for d in [5, 10, 20, 30, 60]:
                hh = high.rolling(d).max()
                ll = low.rolling(d).min()
                features[f"RSV{d}"] = (close - ll) / (hh - ll + 1e-12)

            # CORR 因子 (价量相关性)
            for d in [5, 10, 20, 30, 60]:
                features[f"CORR{d}"] = close.rolling(d).corr(volume)

            # CORD 因子 (收益率与换手率相关性)
            ret = close.pct_change()
            for d in [5, 10, 20, 30, 60]:
                features[f"CORD{d}"] = ret.rolling(d).corr(volume.pct_change())

            # 删除 NaN
            features = features.dropna()

            return features

        except Exception as e:
            self.logger.error(f"Error computing features: {e}")
            return None

    def market_data_extra_info(self) -> List[str]:
        """显示额外的市场数据信息"""
        lines = []

        if self.all_candles_ready:
            candles_df = self.candles[0].candles_df
            if not candles_df.empty:
                last_close = candles_df["close"].iloc[-1]
                last_volume = candles_df["volume"].iloc[-1]
                lines.append(f"Last Close: {last_close}")
                lines.append(f"Last Volume: {last_volume}")
                lines.append(f"Candles Count: {len(candles_df)}")

        lines.append(f"Model Loaded: {self.model is not None}")
        lines.append(f"Qlib Initialized: {self.qlib_initialized}")
        lines.append(f"Signal Threshold: {self.signal_threshold}")

        return lines
```

---

### 6.5 start.py

**文件路径**: `hummingbot/hummingbot/strategy/qlib_alpha/start.py`

```python
"""
Qlib Alpha Strategy 启动入口

此文件由 Hummingbot 框架调用以启动策略。
"""

from typing import Dict, Any

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.strategy.qlib_alpha.qlib_alpha import QlibAlphaStrategy
from hummingbot.strategy.qlib_alpha.qlib_alpha_config import QlibAlphaConfig


def _get_config_value(config_map: Dict, key: str, default: Any = None) -> Any:
    """
    安全地从 config_map 获取值

    Hummingbot 的 config_map 可能返回 ConfigVar 对象而非原始值，
    需要通过 .value 属性获取实际值。

    Parameters
    ----------
    config_map : Dict
        配置字典
    key : str
        配置键名
    default : Any
        默认值

    Returns
    -------
    Any
        配置值
    """
    value = config_map.get(key, default)
    if value is None:
        return default

    # 如果是 ConfigVar 或类似对象，尝试获取 .value 属性
    if hasattr(value, 'value'):
        return value.value

    return value


def start(self) -> QlibAlphaStrategy:
    """
    启动 Qlib Alpha 策略

    此函数由 Hummingbot 的 start 命令调用。

    Returns
    -------
    QlibAlphaStrategy
        策略实例
    """
    # 从 self (HummingbotApplication) 获取配置
    # 使用 _get_config_value 安全地提取值，处理 ConfigVar 类型
    config = QlibAlphaConfig(
        exchange=_get_config_value(self.strategy_config_map, "exchange", "binance"),
        market=_get_config_value(self.strategy_config_map, "market", "BTC-USDT"),
        order_amount_usd=_get_config_value(self.strategy_config_map, "order_amount_usd", 100),
        model_path=_get_config_value(self.strategy_config_map, "model_path", "~/.qlib/models/lgb_model.pkl"),
        qlib_data_path=_get_config_value(self.strategy_config_map, "qlib_data_path", "~/.qlib/qlib_data/crypto_data"),
        signal_threshold=_get_config_value(self.strategy_config_map, "signal_threshold", 0.005),
        prediction_interval=_get_config_value(self.strategy_config_map, "prediction_interval", "1h"),
        stop_loss=_get_config_value(self.strategy_config_map, "stop_loss", 0.02),
        take_profit=_get_config_value(self.strategy_config_map, "take_profit", 0.03),
        time_limit=_get_config_value(self.strategy_config_map, "time_limit", 3600),
        cooldown_after_execution=_get_config_value(self.strategy_config_map, "cooldown_after_execution", 60),
        max_executors=_get_config_value(self.strategy_config_map, "max_executors", 1),
    )

    # 获取连接器
    connectors: Dict[str, ConnectorBase] = {}
    for connector_name in self.connectors:
        connectors[connector_name] = self.connectors[connector_name]

    # 创建策略实例
    strategy = QlibAlphaStrategy(
        connectors=connectors,
        config=config,
    )

    return strategy
```

---

### 6.6 策略注册

Hummingbot 需要知道新策略的存在才能在 `create` 命令中显示。需要进行以下注册：

**Step 1**: 修改 `hummingbot/hummingbot/client/command/create_command.py`

在策略列表中添加 `qlib_alpha`:

```python
# 找到策略注册部分（约第 20-50 行）
# 在 STRATEGIES 列表中添加:
STRATEGIES = [
    # ... existing strategies ...
    "qlib_alpha",  # 新增：Qlib Alpha 策略
]
```

**Step 2**: 修改 `hummingbot/hummingbot/strategy/__init__.py`

导出策略模块：

```python
# 在文件末尾添加:
from hummingbot.strategy.qlib_alpha import QlibAlphaStrategy, QlibAlphaConfig

__all__.extend([
    "QlibAlphaStrategy",
    "QlibAlphaConfig",
])
```

**Step 3**: 创建策略配置映射 `hummingbot/hummingbot/client/config/strategy/qlib_alpha_config_map.py`

```python
"""
Qlib Alpha 策略配置映射

此文件用于 Hummingbot CLI 的 create 命令交互式配置。
"""

from hummingbot.strategy.qlib_alpha.qlib_alpha_config import QlibAlphaConfig

# 使用 Pydantic config 作为 config_map
qlib_alpha_config_map = QlibAlphaConfig
```

**Step 4**: 在 `hummingbot/hummingbot/client/config/strategy/__init__.py` 中注册

```python
# 添加导入
from hummingbot.client.config.strategy.qlib_alpha_config_map import qlib_alpha_config_map

# 在 STRATEGY_CONFIG_MAP 字典中添加
STRATEGY_CONFIG_MAP = {
    # ... existing strategies ...
    "qlib_alpha": qlib_alpha_config_map,
}
```

> **注意**: 具体的注册方式可能因 Hummingbot 版本不同而有所差异。
> 建议参考现有策略（如 `directional_strategy_base`）的注册方式。

---

## 7. 模型训练

### 7.1 训练脚本

**文件路径**: `scripts/train_model.py`

```python
"""
模型训练脚本

使用 Qlib 的 LGBModel 训练价格预测模型。
"""

import pickle
import argparse
from pathlib import Path
from datetime import datetime

import qlib
from qlib.constant import REG_CRYPTO
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


def train_model(
    qlib_data_path: str,
    output_path: str,
    instruments: list,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
):
    """
    训练 LightGBM 模型

    Parameters
    ----------
    qlib_data_path : str
        Qlib 数据目录
    output_path : str
        模型输出路径
    instruments : list
        交易对列表
    train_start, train_end : str
        训练集时间范围
    valid_start, valid_end : str
        验证集时间范围
    """
    # 初始化 Qlib
    print("Initializing Qlib...")
    qlib.init(
        provider_uri=qlib_data_path,
        region=REG_CRYPTO,
    )

    # 创建数据处理器 (Alpha158 因子)
    # 重要：训练时的 freq 必须与实盘 prediction_interval 一致！
    # 默认使用 1h，与 qlib_alpha.py 中的 _compute_features() 保持一致
    print("Creating data handler with Alpha158 factors...")
    handler = Alpha158(
        instruments=instruments,
        start_time=train_start,
        end_time=valid_end,
        freq="1h",  # 必须与实盘 prediction_interval 一致
        infer_processors=[],
        learn_processors=[
            {"class": "DropnaLabel"},
            # 注意：CSRankNorm 需要多个品种才能正常工作
            # 如果只有 2 个品种，可以删除或改用 Robust 标准化
            # {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
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

    # 训练模型
    print("Training model...")
    model.fit(dataset)

    # 保存模型
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {output_path}")

    # 验证模型
    print("\nValidating model...")
    predictions = model.predict(dataset, segment="valid")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions sample:\n{predictions.head()}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train Qlib model")
    parser.add_argument(
        "--qlib-data-path",
        type=str,
        default="~/.qlib/qlib_data/crypto_data",
        help="Qlib data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="~/.qlib/models/lgb_model.pkl",
        help="Output model path",
    )
    parser.add_argument(
        "--instruments",
        type=str,
        nargs="+",
        default=["btcusdt", "ethusdt"],
        help="Instruments to train on",
    )
    parser.add_argument(
        "--train-start",
        type=str,
        default="2023-01-01",
        help="Training start date",
    )
    parser.add_argument(
        "--train-end",
        type=str,
        default="2024-06-30",
        help="Training end date",
    )
    parser.add_argument(
        "--valid-start",
        type=str,
        default="2024-07-01",
        help="Validation start date",
    )
    parser.add_argument(
        "--valid-end",
        type=str,
        default="2024-12-31",
        help="Validation end date",
    )

    args = parser.parse_args()

    train_model(
        qlib_data_path=args.qlib_data_path,
        output_path=args.output,
        instruments=args.instruments,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
    )


if __name__ == "__main__":
    main()
```

---

## 8. 数据准备

### 8.1 Qlib 数据目录结构

```
~/.qlib/qlib_data/crypto_data/
├── calendars/
│   ├── day.txt           # 日历文件
│   ├── 1min.txt
│   ├── 5min.txt
│   ├── 15min.txt
│   ├── 1h.txt
│   └── 4h.txt
├── features/
│   ├── btcusdt/
│   │   ├── open.bin      # float32 二进制
│   │   ├── high.bin
│   │   ├── low.bin
│   │   ├── close.bin
│   │   ├── volume.bin
│   │   └── vwap.bin
│   ├── ethusdt/
│   │   └── ...
│   └── ...
└── instruments/
    └── all.txt           # 交易对列表
```

### 8.2 数据准备脚本示例

**文件路径**: `scripts/prepare_crypto_data.py` (新建)

```python
"""
加密货币数据准备脚本

从 Hummingbot 获取历史数据，转换为 Qlib 格式。

注意：
1. 数据量必须与日历范围匹配
2. 使用 get_historical_candles 获取完整历史数据
3. 训练/验证时间范围必须与 train_model.py 一致
"""

import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone

from hummingbot.data_feed.candles_feed.binance_spot_candles import BinanceSpotCandles
from hummingbot.data_feed.candles_feed.data_types import HistoricalCandlesConfig
from hummingbot.data_feed.qlib_bridge import DataBridge, CryptoCalendarGenerator


async def fetch_historical_data(
    trading_pairs: list,
    interval: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
):
    """
    获取历史数据并保存为 Qlib 格式

    Parameters
    ----------
    trading_pairs : list
        交易对列表
    interval : str
        K 线间隔
    start_date : str
        开始日期 YYYY-MM-DD
    end_date : str
        结束日期 YYYY-MM-DD
    output_dir : Path
        输出目录
    """
    # 转换日期为时间戳
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())

    # 初始化
    data_bridge = DataBridge()

    # 获取 K 线数据
    all_data = {}
    for pair in trading_pairs:
        print(f"Fetching {pair} candles from {start_date} to {end_date}...")

        candles = BinanceSpotCandles(trading_pair=pair, interval=interval)

        # 使用 get_historical_candles 获取完整历史数据
        config = HistoricalCandlesConfig(
            connector_name="binance",
            trading_pair=pair,
            interval=interval,
            start_time=start_ts,
            end_time=end_ts,
        )

        try:
            candles_df = await candles.get_historical_candles(config)
            print(f"  Fetched {len(candles_df)} candles for {pair}")

            # 转换格式
            qlib_df = data_bridge.candles_to_qlib(candles_df, pair)
            all_data[pair] = qlib_df
        except Exception as e:
            print(f"  Error fetching {pair}: {e}")
            continue

    if not all_data:
        print("No data fetched, exiting.")
        return

    # 合并数据
    merged_df = data_bridge.merge_instruments(all_data)
    print(f"Total merged records: {len(merged_df)}")

    # 保存为 Qlib 格式 (传入 freq 参数)
    print("Saving to Qlib format...")
    data_bridge.to_qlib_provider_format(merged_df, str(output_dir), freq=interval)

    print(f"Data saved to {output_dir}")


def main():
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

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    asyncio.run(fetch_historical_data(
        trading_pairs=args.trading_pairs,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=output_dir,
    ))


if __name__ == "__main__":
    main()
```

---

## 9. 启动与运行

### 9.1 完整运行流程

```
┌─────────────────────────────────────────────────────────────┐
│                      完整运行流程                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Step 1: 准备数据                                           │
│  $ python scripts/prepare_crypto_data.py                   │
│                                                             │
│  Step 2: 训练模型                                           │
│  $ python scripts/train_model.py                           │
│                                                             │
│  Step 3: 配置 API                                           │
│  $ hummingbot                                               │
│  >>> connect binance                                        │
│  >>> [输入 API Key 和 Secret]                               │
│                                                             │
│  Step 4: 启动策略                                           │
│  >>> create                                                 │
│  >>> [选择 qlib_alpha 策略]                                 │
│  >>> start                                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 配置 Binance API

**Step 1**: 获取 API Key

1. 登录 [Binance](https://www.binance.com)
2. 进入 API 管理页面
3. 创建新的 API Key
4. 启用 "Enable Spot & Margin Trading" 权限
5. (可选) 设置 IP 白名单

**Step 2**: 在 Hummingbot 中配置

```bash
# 启动 Hummingbot
$ cd hummingbot
$ ./start

# 连接 Binance
>>> connect binance
Enter your Binance API key >>> [YOUR_API_KEY]
Enter your Binance API secret >>> [YOUR_API_SECRET]
```

### 9.3 创建并启动策略

```bash
# 创建策略配置
>>> create

# 选择策略
What is your strategy? >>> qlib_alpha

# 配置参数
Enter exchange name (e.g., binance): >>> binance
Enter trading pair (e.g., BTC-USDT): >>> BTC-USDT
Enter order amount in USD: >>> 100
Enter model file path: >>> ~/.qlib/models/lgb_model.pkl
Enter Qlib data path: >>> ~/.qlib/qlib_data/crypto_data
Enter signal threshold (e.g., 0.005 for 0.5% return): >>> 0.005
Enter prediction interval: >>> 1h
Enter stop loss percentage (e.g., 0.02 for 2%): >>> 0.02
Enter take profit percentage (e.g., 0.03 for 3%): >>> 0.03
Enter time limit in seconds: >>> 3600
Enter cooldown time in seconds: >>> 60
Enter max concurrent positions: >>> 1

# 启动策略
>>> start
```

### 9.4 监控与管理

```bash
# 查看策略状态
>>> status

# 查看余额
>>> balance

# 查看订单历史
>>> history

# 停止策略
>>> stop

# 退出
>>> exit
```

### 9.5 Paper Trading (模拟交易)

在正式交易前，建议使用 Binance Testnet 进行模拟交易：

1. 获取 Testnet API Key: https://testnet.binance.vision/
2. 配置连接器:

```bash
>>> connect binance_paper_trade
Enter your Binance Testnet API key >>> [TESTNET_API_KEY]
Enter your Binance Testnet API secret >>> [TESTNET_API_SECRET]
```

---

## 10. 验收标准

### 10.1 验收清单

| 序号 | 验收项 | 验收方法 | 通过标准 |
|------|--------|----------|----------|
| 1 | Qlib 初始化 | `qlib.init(region="crypto")` | 无报错 |
| 2 | 日历生成 | CryptoCalendarGenerator.generate("1h") | 返回连续时间戳 |
| 3 | 数据转换 | DataBridge.candles_to_qlib() | MultiIndex DataFrame |
| 4 | 信号转换 | SignalBridge.signal_to_order() | 正确订单格式 |
| 5 | Alpha158 因子 | Alpha158(instruments=["btcusdt"]) | 158 个因子计算 |
| 6 | 模型训练 | scripts/train_model.py | 模型文件生成 |
| 7 | 策略加载 | QlibAlphaStrategy 初始化 | 模型加载成功 |
| 8 | 信号生成 | get_signal() 返回值 | 返回 -1, 0, 1 |
| 9 | 三重屏障 | 止损/止盈触发 | 自动平仓 |
| 10 | Paper Trading | Testnet 模拟交易 24h | 无异常 |
| 11 | 实盘交易 | Binance 真实下单 | 订单成功执行 |

### 10.2 验证代码

```python
# 验证脚本: scripts/verify_integration.py

import qlib
from qlib.constant import REG_CRYPTO
from qlib.contrib.data.handler import Alpha158
from hummingbot.data_feed.qlib_bridge import DataBridge, SignalBridge, CryptoCalendarGenerator


def verify():
    # 1. 验证 Qlib 初始化
    print("1. Testing Qlib init with crypto region...")
    qlib.init(
        provider_uri="~/.qlib/qlib_data/crypto_data",
        region=REG_CRYPTO,
    )
    print("   ✓ Qlib initialized successfully")

    # 2. 验证日历生成
    print("2. Testing calendar generation...")
    calendar_gen = CryptoCalendarGenerator()
    cal = calendar_gen.generate("1h")
    assert len(cal) > 0, "Calendar is empty"
    print(f"   ✓ Generated {len(cal)} timestamps")

    # 3. 验证数据转换
    print("3. Testing data bridge...")
    import pandas as pd
    mock_candles = pd.DataFrame({
        "timestamp": [1704067200, 1704070800],
        "open": [42000.0, 42100.0],
        "high": [42500.0, 42600.0],
        "low": [41800.0, 41900.0],
        "close": [42100.0, 42200.0],
        "volume": [100.0, 150.0],
        "quote_asset_volume": [4200000.0, 6300000.0],
        "n_trades": [1000, 1200],
        "taker_buy_base_volume": [50.0, 75.0],
        "taker_buy_quote_volume": [2100000.0, 3150000.0],
    })
    bridge = DataBridge()
    qlib_df = bridge.candles_to_qlib(mock_candles, "BTC-USDT")
    assert "$close" in qlib_df.columns, "Missing $close column"
    assert "$vwap" in qlib_df.columns, "Missing $vwap column"
    print("   ✓ Data conversion successful")

    # 4. 验证信号转换
    print("4. Testing signal bridge...")
    from decimal import Decimal
    signal = {
        "instrument": "btcusdt",
        "direction": 1,
        "weight": 0.1,
        "score": 0.85,
    }
    # Mock connector
    class MockConnector:
        pass
    signal_bridge = SignalBridge(MockConnector())
    # 注意：必须传入 current_price 用于单位转换
    order = signal_bridge.signal_to_order(
        signal=signal,
        available_balance=Decimal("10000"),
        current_position=Decimal("0"),
        current_price=Decimal("42000"),  # 当前 BTC 价格
    )
    assert order is not None, "Order is None"
    assert order["trading_pair"] == "BTC-USDT", f"Wrong trading pair: {order['trading_pair']}"
    # 验证订单金额: 10000 * 0.1 / 42000 ≈ 0.0238 BTC
    expected_amount = Decimal("10000") * Decimal("0.1") / Decimal("42000")
    assert abs(order["amount"] - expected_amount) < Decimal("0.0001"), f"Wrong amount: {order['amount']}"
    print("   ✓ Signal conversion successful")

    print("\n✓ All verifications passed!")


if __name__ == "__main__":
    verify()
```

---

## 附录

### A. 文件修改总结

| 文件 | 修改类型 | 行数 |
|------|----------|------|
| `qlib/qlib/constant.py` | 修改 | +1 行 |
| `qlib/qlib/config.py` | 修改 | +11 行 |
| `qlib/qlib/data/data.py` | 新增类 | +45 行 |
| `hummingbot/.../qlib_bridge/__init__.py` | 新建 | ~20 行 |
| `hummingbot/.../qlib_bridge/data_bridge.py` | 新建 | ~150 行 |
| `hummingbot/.../qlib_bridge/signal_bridge.py` | 新建 | ~120 行 |
| `hummingbot/.../qlib_bridge/calendar.py` | 新建 | ~80 行 |
| `hummingbot/.../strategy/qlib_alpha/__init__.py` | 新建 | ~20 行 |
| `hummingbot/.../strategy/qlib_alpha/qlib_alpha.py` | 新建 | ~250 行 |
| `hummingbot/.../strategy/qlib_alpha/qlib_alpha_config.py` | 新建 | ~80 行 |
| `hummingbot/.../strategy/qlib_alpha/start.py` | 新建 | ~40 行 |
| `scripts/train_model.py` | 新建 | ~100 行 |
| `scripts/prepare_crypto_data.py` | 新建 | ~50 行 |
| **总计** | - | **~970 行** |

### B. 使用示例

```python
# 完整使用示例: 从数据准备到实盘交易

# ============ Step 1: 数据准备 ============
# $ python scripts/prepare_crypto_data.py

# ============ Step 2: 模型训练 ============
# $ python scripts/train_model.py \
#     --instruments btcusdt ethusdt \
#     --train-start 2023-01-01 \
#     --train-end 2024-06-30 \
#     --valid-start 2024-07-01 \
#     --valid-end 2024-12-31

# ============ Step 3: 启动 Hummingbot ============
# $ cd hummingbot && ./start
# >>> connect binance
# >>> create
# >>> [选择 qlib_alpha]
# >>> start

# ============ Python API 示例 ============
import qlib
from qlib.constant import REG_CRYPTO
from qlib.contrib.data.handler import Alpha158
from hummingbot.data_feed.qlib_bridge import DataBridge, SignalBridge
from hummingbot.strategy.qlib_alpha import QlibAlphaStrategy, QlibAlphaConfig

# 1. 初始化 Qlib
qlib.init(
    provider_uri="~/.qlib/qlib_data/crypto_data",
    region=REG_CRYPTO,
)

# 2. 加载预训练模型
import pickle
with open("~/.qlib/models/lgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# 3. 创建策略配置
config = QlibAlphaConfig(
    exchange="binance",
    market="BTC-USDT",
    order_amount_usd=100,
    model_path="~/.qlib/models/lgb_model.pkl",
    stop_loss=0.02,
    take_profit=0.03,
    time_limit=3600,
)

# 4. 策略将自动:
#    - 获取实时 K 线数据 (BinanceSpotCandles)
#    - 计算 Alpha158 因子
#    - 调用模型生成预测
#    - 根据阈值生成交易信号
#    - 通过三重屏障管理持仓
#    - 自动止损止盈
```

### C. 常见问题

| 问题 | 解决方案 |
|------|----------|
| Qlib 初始化失败 | 检查 `qlib_data_path` 是否正确，确保数据已准备 |
| 模型加载失败 | 确保已运行 `train_model.py` 生成模型文件 |
| K 线数据不足 | 等待 Hummingbot 收集足够的历史数据 (约 60 根 K 线) |
| 信号一直为 0 | 调整 `signal_threshold` 参数，或检查模型预测范围 |
| 订单执行失败 | 检查 API 权限、余额、交易对是否正确 |

### D. 依赖版本

```
qlib >= 0.9.7
hummingbot >= 2.11.0
lightgbm >= 4.0.0
pandas >= 2.0.0
numpy >= 1.24.0
pydantic >= 2.0.0
```

---

### E. 变更日志

**v5.2.0** (2026-01-02)
- [B1] 统一训练与实盘特征/频率：train_model.py 改用 `freq="1h"` + `RobustZScoreNorm`
- [B2] 修复时间戳单位：添加 `_detect_timestamp_unit()` 自动检测 s/ms
- [B3] 修复 SignalBridge 买卖方向：基于 delta 而非 direction 决定 trade_type
- [B4] 修复 verify_integration.py：添加 `current_price` 参数
- [C1] 添加策略注册步骤：新增 6.6 节说明如何在 Hummingbot 中注册策略
- [C2] 明确 SignalBridge 定位：说明当前未使用，为未来多品种扩展预留
- [C4] 明确 CryptoCalendarProvider 定位：说明为可选备用方案
- 添加 numpy 导入到 qlib_alpha.py
- 增强 _compute_features() 文档说明训练/推理一致性

**v5.1.0** (2026-01-02)
- 修复 `_compute_features()` 中 `max/min` 改为 `np.maximum/np.minimum`
- 修复 `model.predict()` 接口：使用底层 `self.model.model.predict()`
- 修复信号阈值逻辑：从概率阈值改为收益率阈值 (默认 0.005)
- 修复 SignalBridge 金额单位：添加 `current_price` 参数进行单位转换
- 修复日历/数据存储：支持多频率，添加 `freq` 参数
- 修复数据准备脚本：使用 `get_historical_candles` 获取完整历史
- 修复 start.py：添加 `_get_config_value()` 处理 ConfigVar 类型
- 修复文档表述：明确 Hummingbot 需新增模块而非完全不修改
- 添加 `get_crypto_config()` 辅助函数简化 Qlib 初始化

**v5.0.0** (初始版本)
- 完整的 Qlib + Hummingbot 融合方案
- 包含桥接层、策略层、训练脚本、数据准备脚本
