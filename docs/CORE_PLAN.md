# AlgVex 核心方案 (P0 - MVP)

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
- [6. 数据准备](#6-数据准备)
- [7. 验收标准](#7-验收标准)

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
│     - config.py: 添加加密货币区域配置                         │
│     - data/data.py: 添加 CryptoCalendarProvider             │
│                                                             │
│  3. Hummingbot 修改范围                                      │
│     - 无需修改，100% 原生使用                                 │
│                                                             │
│  4. 新建桥接层                                               │
│     - hummingbot/hummingbot/data_feed/qlib_bridge/          │
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
| **总计** | - | **~360 行** |

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
    },
}
```

---

### 3.3 修改 data/data.py

**文件路径**: `qlib/qlib/data/data.py`

**修改位置**: 第 676 行之后（LocalCalendarProvider 类之后），添加 CryptoCalendarProvider 类

**新增代码**:
```python
# 在第 676 行之后添加

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

### 4.1 无需修改

Hummingbot 100% 原生使用，无需修改任何现有代码。

以下组件直接使用：

| 组件 | 文件路径 | 用途 |
|------|----------|------|
| BinanceSpotCandles | `data_feed/candles_feed/binance_spot_candles/` | K线数据获取 |
| BinanceExchange | `connector/exchange/binance/` | 订单执行 |
| CandlesBase | `data_feed/candles_feed/candles_base.py` | K线基类 |

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

        # 转换时间戳
        df = candles_df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp"], unit='s', utc=True)
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

    def to_qlib_provider_format(
        self,
        df: pd.DataFrame,
        output_dir: str
    ) -> None:
        """
        将数据保存为 Qlib Provider 可读取的格式

        Parameters
        ----------
        df : pd.DataFrame
            Qlib 格式的 DataFrame
        output_dir : str
            输出目录路径
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

        # 保存每个交易对的特征数据
        for inst in instruments:
            inst_df = df.xs(inst, level="instrument")
            inst_dir = features_dir / inst
            inst_dir.mkdir(exist_ok=True)

            for col in self.QLIB_COLUMNS:
                col_name = col.replace("$", "")
                file_path = inst_dir / f"{col_name}.bin"
                # Qlib 使用 float32 二进制格式
                inst_df[col].values.astype(np.float32).tofile(file_path)

        # 保存日历
        calendar = df.index.get_level_values("datetime").unique()
        calendar_file = calendars_dir / "day.txt"
        with open(calendar_file, "w") as f:
            for dt in calendar:
                f.write(dt.strftime("%Y-%m-%d") + "\n")

        # 保存交易对列表
        instruments_file = instruments_dir / "all.txt"
        with open(instruments_file, "w") as f:
            for inst in instruments:
                dates = df.xs(inst, level="instrument").index
                start = dates.min().strftime("%Y-%m-%d")
                end = dates.max().strftime("%Y-%m-%d")
                f.write(f"{inst}\t{start}\t{end}\n")
```

---

### 5.4 signal_bridge.py

**文件路径**: `hummingbot/hummingbot/data_feed/qlib_bridge/signal_bridge.py`

```python
"""
SignalBridge - Qlib 交易信号转换为 Hummingbot 订单

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
            可用余额
        current_position : Decimal
            当前持仓

        Returns
        -------
        Optional[Dict]
            Hummingbot 订单参数，如果不需要交易则返回 None
        """
        direction = signal.get("direction", 0)

        # 持有信号，不交易
        if direction == 0:
            return None

        instrument = signal.get("instrument", "")
        weight = Decimal(str(signal.get("weight", 0)))

        # 计算目标金额
        target_amount = available_balance * weight

        # 计算需要交易的金额
        if direction > 0:  # 买入
            trade_amount = target_amount - current_position
            trade_type = TradeType.BUY
        else:  # 卖出
            trade_amount = current_position - target_amount
            trade_type = TradeType.SELL

        # 检查最小下单金额
        if abs(trade_amount) < self.min_order_amount:
            self.logger.debug(f"Trade amount {trade_amount} below minimum {self.min_order_amount}")
            return None

        trade_amount = abs(trade_amount)

        return {
            "trading_pair": self._format_trading_pair(instrument),
            "order_type": self.default_order_type,
            "trade_type": trade_type,
            "amount": trade_amount,
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
    ) -> List[Dict[str, Any]]:
        """
        批量转换信号为订单

        Parameters
        ----------
        signals : List[Dict]
            Qlib 信号列表
        available_balance : Decimal
            可用余额
        positions : Dict[str, Decimal]
            当前持仓 {instrument: amount}

        Returns
        -------
        List[Dict]
            订单列表
        """
        orders = []

        for signal in signals:
            instrument = signal.get("instrument", "")
            current_position = positions.get(instrument, Decimal("0"))

            order = self.signal_to_order(
                signal=signal,
                available_balance=available_balance,
                current_position=current_position,
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

## 6. 数据准备

### 6.1 Qlib 数据目录结构

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

### 6.2 数据准备脚本示例

**文件路径**: `scripts/prepare_crypto_data.py` (新建)

```python
"""
加密货币数据准备脚本

从 Hummingbot 获取历史数据，转换为 Qlib 格式。
"""

import asyncio
from pathlib import Path
from hummingbot.data_feed.candles_feed.binance_spot_candles import BinanceSpotCandles
from hummingbot.data_feed.qlib_bridge import DataBridge, CryptoCalendarGenerator


async def main():
    # 配置
    trading_pairs = ["BTC-USDT", "ETH-USDT"]
    interval = "1h"
    output_dir = Path.home() / ".qlib/qlib_data/crypto_data"

    # 初始化
    data_bridge = DataBridge()
    calendar_gen = CryptoCalendarGenerator(
        start_date="2023-01-01",
        end_date="2024-12-31"
    )

    # 生成日历
    print("Generating calendars...")
    calendar_gen.generate_all_frequencies(str(output_dir / "calendars"))

    # 获取 K 线数据
    all_data = {}
    for pair in trading_pairs:
        print(f"Fetching {pair} candles...")
        candles = BinanceSpotCandles(trading_pair=pair, interval=interval, max_records=1000)
        await candles.start_network()

        # 等待数据就绪
        while not candles.ready:
            await asyncio.sleep(1)

        # 转换格式
        qlib_df = data_bridge.candles_to_qlib(candles.candles_df, pair)
        all_data[pair] = qlib_df

        await candles.stop_network()

    # 合并数据
    merged_df = data_bridge.merge_instruments(all_data)

    # 保存为 Qlib 格式
    print("Saving to Qlib format...")
    data_bridge.to_qlib_provider_format(merged_df, str(output_dir))

    print(f"Data saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 7. 验收标准

### 7.1 验收清单

| 序号 | 验收项 | 验收方法 | 通过标准 |
|------|--------|----------|----------|
| 1 | Qlib 初始化 | `qlib.init(region="crypto")` | 无报错 |
| 2 | 日历生成 | CryptoCalendarGenerator.generate("1h") | 返回连续时间戳 |
| 3 | 数据转换 | DataBridge.candles_to_qlib() | MultiIndex DataFrame |
| 4 | 信号转换 | SignalBridge.signal_to_order() | 正确订单格式 |
| 5 | Alpha158 因子 | Alpha158(instruments=["btcusdt"]) | 158 个因子计算 |
| 6 | 回测运行 | backtest_loop() | 完整回测报告 |
| 7 | Paper Trading | 模拟下单 24h | 无异常 |

### 7.2 验证代码

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
    order = signal_bridge.signal_to_order(
        signal=signal,
        available_balance=Decimal("10000"),
        current_position=Decimal("0"),
    )
    assert order is not None, "Order is None"
    assert order["trading_pair"] == "BTC-USDT", f"Wrong trading pair: {order['trading_pair']}"
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
| **总计** | - | **~430 行** |

### B. 使用示例

```python
# 完整使用示例

import qlib
from qlib.constant import REG_CRYPTO
from qlib.contrib.data.handler import Alpha158
from qlib.backtest import backtest_loop
from hummingbot.data_feed.qlib_bridge import DataBridge, SignalBridge

# 1. 初始化 Qlib
qlib.init(
    provider_uri="~/.qlib/qlib_data/crypto_data",
    region=REG_CRYPTO,
)

# 2. 加载 Alpha158 因子
handler = Alpha158(
    instruments=["btcusdt", "ethusdt"],
    start_time="2024-01-01",
    end_time="2024-12-31",
    freq="day",
)

# 3. 训练模型并回测
# ... (使用 Qlib 标准流程)

# 4. 实盘交易时使用 SignalBridge
from hummingbot.connector.exchange.binance import BinanceExchange

connector = BinanceExchange(
    binance_api_key="...",
    binance_api_secret="...",
)
signal_bridge = SignalBridge(connector)

# 执行信号
order = signal_bridge.signal_to_order(signal, balance, position)
await signal_bridge.execute_order(order)
```

---

> **版本**: v4.0.0 (2026-01-02)
> **状态**: 可直接实施的详细方案
