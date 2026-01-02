# AlgVex 核心方案 (P0 - MVP)

> **Qlib + Hummingbot 融合的加密货币现货量化交易平台**
>
> 本文档仅包含 MVP 核心功能，专注于**加密货币现货交易**。
>
> 相关文档：
> - [扩展功能 (P2)](./EXTENSION_PLAN.md) - 永续合约扩展、因子扩展等
> - [未来规划 (P3)](./FUTURE_PLAN.md) - 开发路线图、更新日志等

---

## 目录

- [1. 核心原则](#1-核心原则)
- [2. MVP 范围定义](#2-mvp-范围定义)
- [3. 系统架构](#3-系统架构)
- [4. Qlib 适配层](#4-qlib-适配层)
- [5. 因子系统](#5-因子系统)
- [6. 数据层](#6-数据层)
- [7. 回测层](#7-回测层)
- [8. 执行层](#8-执行层)
- [9. 桥接层](#9-桥接层)
- [10. 验收标准](#10-验收标准)

---

## 1. 核心原则

```
┌─────────────────────────────────────────────────────────────┐
│                      核心原则                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Qlib 是生产系统核心                                      │
│     - 数据层: Qlib Provider                                  │
│     - 因子层: Qlib 表达式引擎 + Alpha158                     │
│     - 回测层: Qlib Exchange (原生)                           │
│                                                             │
│  2. Hummingbot 负责订单执行                                  │
│     - 实时数据: Hummingbot CandlesBase (WebSocket)          │
│     - 订单执行: Hummingbot Connector                         │
│     - 风控执行: 止损止盈                                     │
│                                                             │
│  3. 桥接层是唯一新建组件 (~300行)                             │
│     - 数据桥接: Hummingbot K线 → Qlib 格式                   │
│     - 信号桥接: Qlib 信号 → Hummingbot 订单                  │
│                                                             │
│  4. 只做现货，不做合约                                        │
│     - 无做空、无杠杆、无资金费率、无爆仓                       │
│     - Qlib Exchange 原生支持，无需扩展                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 设计哲学

| 原则 | 说明 |
|------|------|
| **最小适配** | 只做必要适配，最大化复用 Qlib/Hummingbot 原生能力 |
| **现货优先** | 先验证现货架构，永续合约留待 Phase 2 |
| **Qlib 核心** | Qlib 作为因子/回测核心，不自建因子引擎 |
| **Hummingbot 执行** | Hummingbot 作为执行层，不自建交易连接器 |

### 1.2 技术选型

| 层级 | 组件 | 版本 | 说明 |
|------|------|------|------|
| **数据层** | Qlib Provider | 0.9.7 | 统一数据管理 |
| **因子层** | Qlib Alpha158 | 0.9.7 | 原生因子表达式 |
| **回测层** | Qlib Exchange | 0.9.7 | 原生回测引擎 |
| **执行层** | Hummingbot | 2.11.0 | 现货订单执行 |
| **桥接层** | AlgVexBridge | 自建 | 唯一新建组件 |

---

## 2. MVP 范围定义

### 2.1 MVP 边界

| 条件 | MVP 定义 | 说明 |
|------|----------|------|
| **交易类型** | 仅现货 | 不做永续合约 |
| **交易方向** | 仅做多 | 买入/卖出，无做空 |
| **杠杆** | 无 | 1x 无杠杆 |
| **标的范围** | 20-50 个现货 | BTC, ETH + Top 流动性币种 |
| **时间框架** | 多时间框架 | 1m, 5m, 15m, 1h, 4h, 1d |
| **因子系统** | Alpha158 | Qlib 原生 OHLCV 因子 |

### 2.2 现货 vs 永续对比

| 特性 | 现货 (MVP) | 永续 (Phase 2) |
|------|-----------|----------------|
| 做空 | ❌ 不需要 | 需要扩展 |
| 杠杆 | ❌ 不需要 | 需要扩展 |
| 资金费率 | ❌ 不需要 | 需要扩展 |
| 爆仓检测 | ❌ 不需要 | 需要扩展 |
| Qlib 适配 | ~100 行 | ~550 行 |
| 工期 | 1 周 | 3-4 周 |

### 2.3 MVP 数据源

```yaml
# MVP 仅使用 OHLCV 数据
mvp_data_sources:
  - source_id: klines
    description: "K线数据 (OHLCV)"
    fields: [open, high, low, close, volume]
    frequencies: [1m, 5m, 15m, 1h, 4h, 1d]
    visibility: bar_close
```

---

## 3. 系统架构

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AlgVex 现货交易系统                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Qlib 核心层                                  │   │
│  │                    (研究 + 生产 统一使用)                             │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │   │
│  │   │   Provider  │───▶│  Alpha158   │───▶│   Model     │            │   │
│  │   │  (数据层)   │    │  (因子层)   │    │  (预测层)   │            │   │
│  │   └─────────────┘    └─────────────┘    └──────┬──────┘            │   │
│  │                                                │                    │   │
│  │   ┌─────────────┐    ┌─────────────┐          │                    │   │
│  │   │  Exchange   │◀───│  Strategy   │◀─────────┘                    │   │
│  │   │  (回测层)   │    │  (策略层)   │                               │   │
│  │   └─────────────┘    └─────────────┘                               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      AlgVexBridge (桥接层)                           │   │
│  │                        唯一新建组件 ~300行                            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │   ┌─────────────┐                      ┌─────────────┐              │   │
│  │   │ DataBridge  │   Hummingbot K线 ──▶ │ Qlib 格式   │              │   │
│  │   └─────────────┘                      └─────────────┘              │   │
│  │   ┌─────────────┐                      ┌─────────────┐              │   │
│  │   │SignalBridge │   Qlib 信号 ──────▶  │ HB 订单     │              │   │
│  │   └─────────────┘                      └─────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Hummingbot 执行层                               │   │
│  │                        (原生，无需修改)                               │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                     │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │   │
│  │   │ CandlesBase │    │  Connector  │    │ RiskMgmt   │            │   │
│  │   │ (实时K线)   │    │ (订单执行)  │    │ (止损止盈) │            │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘            │   │
│  │                                                                     │   │
│  │   支持交易所: Binance, Gate.io, OKX, Bybit, KuCoin 等              │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据流                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【回测模式】                                                                │
│                                                                             │
│   历史数据 ──▶ Qlib Provider ──▶ Alpha158 ──▶ Model ──▶ Strategy           │
│                                                            │                │
│                                                            ▼                │
│                                              Qlib Exchange (回测)           │
│                                                            │                │
│                                                            ▼                │
│                                                     回测报告                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【实盘模式】                                                                │
│                                                                             │
│   Hummingbot ──▶ DataBridge ──▶ Qlib Provider ──▶ Alpha158 ──▶ Model       │
│   (WebSocket)    (格式转换)                                    │            │
│                                                                ▼            │
│                                                           Strategy          │
│                                                                │            │
│                                                                ▼            │
│   Hummingbot ◀── SignalBridge ◀───────────────────────── Qlib Signal       │
│   (订单执行)     (信号转换)                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Qlib 适配层

### 4.1 适配范围

| 适配项 | 工作量 | 说明 |
|--------|--------|------|
| **CryptoCalendar** | ~50 行 | 24/7 交易日历 |
| **UTC 时区** | ~50 行 | 全系统强制 UTC |
| **合计** | **~100 行** | - |

### 4.2 CryptoCalendar

```python
# algvex/adapters/crypto_calendar.py

from qlib.data.data import Cal

class CryptoCalendar:
    """
    加密货币 24/7 交易日历

    替换 Qlib 默认的股票交易日历
    """

    def __init__(self, start_date: str, end_date: str, freq: str = "1min"):
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq

    def get_calendar(self) -> list:
        """
        生成 24/7 交易日历

        Returns:
            完整的时间戳列表
        """
        import pandas as pd

        return pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=self._freq_to_pandas(self.freq),
            tz='UTC'
        ).tolist()

    def _freq_to_pandas(self, freq: str) -> str:
        """转换频率格式"""
        mapping = {
            "1min": "1T", "5min": "5T", "15min": "15T",
            "1h": "1H", "4h": "4H", "1d": "1D"
        }
        return mapping.get(freq, freq)
```

### 4.3 时区统一

```python
# algvex/adapters/timezone.py

import pandas as pd
from datetime import timezone

# 全系统强制 UTC
SYSTEM_TIMEZONE = timezone.utc

def ensure_utc(dt) -> pd.Timestamp:
    """确保时间戳为 UTC"""
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    if dt.tz is None:
        return dt.tz_localize('UTC')
    return dt.tz_convert('UTC')
```

---

## 5. 因子系统

### 5.1 Alpha158 因子

使用 Qlib 原生 Alpha158 因子集，无需自建因子。

```python
# 使用 Qlib 原生 Alpha158
from qlib.contrib.data.handler import Alpha158

handler = Alpha158(
    instruments="crypto_spot",  # 自定义标的池
    start_time="2023-01-01",
    end_time="2024-12-31",
    freq="day",
    infer_processors=[],
    learn_processors=[],
)
```

### 5.2 Alpha158 因子类别

| 类别 | 因子数量 | 说明 |
|------|---------|------|
| **KBAR** | 6 | K线形态因子 |
| **PRICE** | 11 | 价格相关因子 |
| **VOLUME** | 6 | 成交量因子 |
| **CORR** | 6 | 相关性因子 |
| **STD** | 6 | 波动率因子 |
| **BETA** | 6 | Beta 因子 |
| **RSRS** | 6 | RSRS 因子 |
| **ROC** | 6 | 变化率因子 |
| **MA** | 15 | 均线因子 |
| **HIGH/LOW** | 20 | 高低点因子 |
| **其他** | 70 | 其他技术因子 |
| **合计** | **158** | - |

### 5.3 多时间框架支持

```python
# 多时间框架因子计算
timeframes = ["5min", "15min", "1h", "4h", "1d"]

for tf in timeframes:
    handler = Alpha158(
        instruments="crypto_spot",
        start_time="2023-01-01",
        end_time="2024-12-31",
        freq=tf,  # 不同时间框架
    )
```

---

## 6. 数据层

### 6.1 数据源

| 数据源 | 字段 | 频率 | 说明 |
|--------|------|------|------|
| **币安现货** | OHLCV | 1m-1d | 主要数据源 |
| **Gate.io** | OHLCV | 1m-1d | 备用数据源 |
| **OKX** | OHLCV | 1m-1d | 备用数据源 |

### 6.2 Qlib 数据格式

```
MultiIndex: (datetime, instrument)
字段前缀: $ (如 $open, $close, $volume)
时区: 全系统强制 UTC

                           $open    $close    $high     $low   $volume
datetime    instrument
2024-01-01  btcusdt      42000.0  42100.0  42500.0  41800.0   1000.0
            ethusdt       2200.0   2210.0   2250.0   2180.0    500.0
```

### 6.3 数据管理器

```python
# algvex/data/manager.py

from qlib.data import D

class CryptoDataManager:
    """加密货币数据管理器"""

    def __init__(self, provider_uri: str):
        self.provider_uri = provider_uri

    def get_data(
        self,
        instruments: list,
        fields: list,
        start_time: str,
        end_time: str,
        freq: str = "1d"
    ):
        """获取数据"""
        return D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
        )
```

---

## 7. 回测层

### 7.1 Qlib Exchange (原生)

现货交易使用 Qlib 原生 Exchange，无需扩展。

```python
from qlib.backtest import Exchange

exchange = Exchange(
    freq="day",
    start_time="2024-01-01",
    end_time="2024-12-31",
    codes=["btcusdt", "ethusdt"],
    deal_price="$close",
    open_cost=0.001,   # 开仓费率 0.1%
    close_cost=0.001,  # 平仓费率 0.1%
    min_cost=0,
)
```

### 7.2 回测配置

```python
from qlib.backtest import backtest

result = backtest(
    start_time="2024-01-01",
    end_time="2024-12-31",
    strategy=my_strategy,
    executor=executor,
    exchange=exchange,
    account=100000,  # 初始资金 $100k
)
```

### 7.3 回测指标

| 收益指标 | 风险指标 | 交易指标 |
|----------|----------|----------|
| 总收益率 | 最大回撤 | 胜率 |
| 年化收益 | 波动率 | 盈亏比 |
| 夏普比率 | 索提诺比率 | 平均持仓时间 |
| 卡尔玛比率 | VaR | 交易次数 |

---

## 8. 执行层

### 8.1 Hummingbot 现货连接器

使用 Hummingbot 原生现货连接器，无需修改。

```python
# 支持的现货交易所
SPOT_EXCHANGES = [
    "binance",
    "gate_io",
    "okx",
    "bybit",
    "kucoin",
    "htx",
    "bitmart",
]
```

### 8.2 订单类型

| 订单类型 | 支持 | 说明 |
|----------|------|------|
| **Market** | ✅ | 市价单 |
| **Limit** | ✅ | 限价单 |
| **Stop Loss** | ✅ | 止损单 |
| **Take Profit** | ✅ | 止盈单 |

### 8.3 风控

```python
# 简单止损止盈 (无需三重屏障)
risk_config = {
    "stop_loss": 0.05,      # 止损 5%
    "take_profit": 0.10,    # 止盈 10%
    "max_position_size": 0.1,  # 最大仓位 10%
}
```

---

## 9. 桥接层

### 9.1 桥接层概述

桥接层是唯一新建组件，约 300 行代码。

```
┌─────────────────────────────────────────────────────────────┐
│                    AlgVexBridge (~300行)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   DataBridge (~100行)                                       │
│   ├── hb_to_qlib()      # Hummingbot K线 → Qlib 格式       │
│   └── update_realtime() # 实时数据更新                      │
│                                                             │
│   SignalBridge (~100行)                                     │
│   ├── signal_to_order() # Qlib 信号 → Hummingbot 订单      │
│   └── execute_order()   # 执行订单                         │
│                                                             │
│   Utils (~100行)                                            │
│   ├── ensure_utc()      # 时区转换                         │
│   └── format_symbol()   # 交易对格式转换                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 DataBridge

```python
# algvex/bridge/data_bridge.py

import pandas as pd

class DataBridge:
    """Hummingbot 数据 → Qlib 格式"""

    def hb_to_qlib(self, hb_candles: list) -> pd.DataFrame:
        """
        将 Hummingbot K线转换为 Qlib 格式

        Args:
            hb_candles: Hummingbot CandlesBase 数据

        Returns:
            Qlib 格式的 DataFrame (MultiIndex)
        """
        records = []
        for candle in hb_candles:
            records.append({
                "datetime": pd.Timestamp(candle.timestamp, unit='s', tz='UTC'),
                "instrument": candle.trading_pair.lower().replace("-", ""),
                "$open": float(candle.open),
                "$high": float(candle.high),
                "$low": float(candle.low),
                "$close": float(candle.close),
                "$volume": float(candle.volume),
            })

        df = pd.DataFrame(records)
        df = df.set_index(["datetime", "instrument"])
        return df
```

### 9.3 SignalBridge

```python
# algvex/bridge/signal_bridge.py

from decimal import Decimal
from hummingbot.core.data_type.common import OrderType, TradeType

class SignalBridge:
    """Qlib 信号 → Hummingbot 订单"""

    def __init__(self, connector):
        self.connector = connector

    def signal_to_order(self, signal: dict) -> dict:
        """
        将 Qlib 信号转换为 Hummingbot 订单

        Args:
            signal: Qlib 策略输出的信号
                - symbol: 交易对
                - direction: 1 买入, -1 卖出
                - amount: 数量

        Returns:
            Hummingbot 订单参数
        """
        return {
            "trading_pair": self._format_pair(signal["symbol"]),
            "order_type": OrderType.MARKET,
            "trade_type": TradeType.BUY if signal["direction"] > 0 else TradeType.SELL,
            "amount": Decimal(str(signal["amount"])),
            "price": Decimal("0"),  # 市价单
        }

    async def execute_order(self, order: dict) -> str:
        """执行订单"""
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
        return order_id

    def _format_pair(self, symbol: str) -> str:
        """格式化交易对: btcusdt → BTC-USDT"""
        symbol = symbol.upper()
        if "USDT" in symbol:
            base = symbol.replace("USDT", "")
            return f"{base}-USDT"
        return symbol
```

---

## 10. 验收标准

### 10.1 MVP 验收清单

| 验收项 | 标准 | 优先级 |
|--------|------|--------|
| **Qlib 适配** | 24/7 日历 + UTC 时区正常工作 | P0 |
| **Alpha158** | 158 个因子计算正确 | P0 |
| **多时间框架** | 5m/15m/1h/4h/1d 因子计算正常 | P0 |
| **回测运行** | Qlib Exchange 回测完整运行 | P0 |
| **数据桥接** | Hummingbot K线正确转换为 Qlib 格式 | P0 |
| **信号桥接** | Qlib 信号正确转换为 Hummingbot 订单 | P0 |
| **Paper Trading** | 模拟交易 24h 无异常 | P0 |

### 10.2 迭代计划

```
Week 1: 基础适配
├── Day 1-2: CryptoCalendar + UTC 适配
├── Day 3-4: Alpha158 验证
└── Day 5: 多时间框架测试

Week 2: 桥接层 + 集成
├── Day 1-2: DataBridge 实现
├── Day 3-4: SignalBridge 实现
└── Day 5: Paper Trading 验证
```

### 10.3 代码量估算

| 组件 | 代码量 | 说明 |
|------|--------|------|
| CryptoCalendar | ~50 行 | 24/7 日历 |
| 时区适配 | ~50 行 | UTC 统一 |
| DataBridge | ~100 行 | 数据格式转换 |
| SignalBridge | ~100 行 | 信号→订单 |
| **合计** | **~300 行** | 唯一新建代码 |

---

## 附录

### A. 目录结构

```
algvex/
├── adapters/
│   ├── crypto_calendar.py    # 24/7 日历
│   └── timezone.py           # UTC 适配
├── bridge/
│   ├── data_bridge.py        # 数据桥接
│   └── signal_bridge.py      # 信号桥接
└── config/
    └── settings.py           # 配置
```

### B. 与 Phase 2 的关系

Phase 2 将扩展支持永续合约：

| 功能 | Phase 1 (现货) | Phase 2 (永续) |
|------|---------------|----------------|
| 做空 | ❌ | ✅ PerpetualPosition |
| 杠杆 | ❌ | ✅ 保证金计算 |
| 资金费率 | ❌ | ✅ FundingRateHandler |
| 爆仓 | ❌ | ✅ 爆仓检测 |
| 代码量 | ~300 行 | +~450 行 |

---

> **版本**: v3.0.0 (2026-01-02)
> **状态**: MVP 现货版本
