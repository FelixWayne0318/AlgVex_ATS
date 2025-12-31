# AlgVex 重构方案 v3.0

> **核心目标**: 先跑通 Qlib + Hummingbot + 加密货币，完全保留原版功能
>
> **MVP 原则**: 只用标准 OHLCV，使用 Qlib 原版 Alpha158，不加自定义因子
>
> **参考文档**: `AlgVex_Qlib_Hummingbot_Platform.md` (完整系统设计方案)

---

## 1. MVP 目标

### 1.1 核心目标

```
┌─────────────────────────────────────────────────────────────────┐
│                     MVP: 先跑通系统                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Qlib 原版功能 100% 保留                                      │
│     • 58 个运算符 (ops.py)                                      │
│     • 14 个 Processor                                           │
│     • 35 个模型 (LGBModel, XGBModel, Transformer...)            │
│     • Alpha158, Alpha360 Handler                                │
│     • 回测框架、评估函数、工作流                                  │
│                                                                 │
│  2. Hummingbot 原版功能 100% 保留                                │
│     • 12 个交易所连接器                                          │
│     • 6 个执行器 (PositionExecutor, DCA, TWAP, Grid...)          │
│     • Strategy V2 框架                                          │
│                                                                 │
│  3. 最小必要适配                                                 │
│     • 24/7 日历 (代替股票 252 天日历)                            │
│     • 标准 OHLCV 数据转换为 Qlib bin 格式                        │
│     • SignalBridge 连接 Qlib 信号 → Hummingbot 执行              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 不在 MVP 范围 (后期再做)

| 后期内容 | 说明 |
|----------|------|
| 自定义因子 | funding_rate, open_interest 等加密货币特有因子 |
| 自定义 Processor | 加密货币数据特殊处理 |
| 订单簿特征 | OBI, spread, depth, impact 等 |
| 杠杆/爆仓模拟 | CryptoPerpetualExchange 扩展 |
| 双向持仓 | BidirectionalPosition |
| 保证金模式 | MarginMode (isolated/cross) |

---

## 2. 方案选择

### 2.1 方案对比

| 方面 | 方案A: pip依赖 | 方案B: 复制源码 ✅ |
|------|---------------|-------------------|
| **代码位置** | `pip install qlib` | `algvex/qlib/` |
| **修改能力** | 只能继承/包装 | ✅ 可以改任何代码 |
| **功能完整** | 依赖接口是否开放 | ✅ 100% 可控 |
| **遗漏风险** | 可能漏掉内部功能 | ✅ 不会遗漏 |

### 2.2 选择方案B的理由

1. **不会遗漏** - 源码全部在，想用什么都有
2. **深度定制** - 加密货币需要改 Qlib 底层（24/7 日历）
3. **稳定可控** - 不怕上游版本变化破坏兼容性

---

## 3. 目录结构

```
algvex/
│
├── qlib/                        # 【复制】Qlib 0.9.7 完整源码 (原版保留)
│   ├── data/
│   │   ├── ops.py               # 58+ 运算符 (原版)
│   │   └── dataset/
│   │       ├── handler.py       # DataHandler (原版)
│   │       └── processor.py     # 14个 Processor (原版)
│   │
│   ├── model/                   # 模型模块 (原版保留)
│   │
│   ├── contrib/
│   │   ├── model/               # 35个模型 (原版)
│   │   ├── data/handler.py      # Alpha158, Alpha360 (原版)
│   │   └── evaluate/            # risk_analysis, calc_ic (原版)
│   │
│   ├── backtest/                # 回测模块 (原版保留)
│   ├── strategy/                # 策略模块 (原版保留)
│   └── workflow/                # 工作流模块 (原版保留)
│
├── hummingbot/                  # 【复制】Hummingbot 2.11.0 完整源码 (原版保留)
│   ├── connector/               # 12个交易所连接器 (原版)
│   ├── strategy_v2/             # 6个执行器 (原版)
│   │   └── executors/
│   │       ├── position_executor/
│   │       ├── dca_executor/
│   │       └── twap_executor/
│   │
│   └── strategy/
│       └── algvex_strategy/     # 【新增】AlgVex 信号驱动策略
│           ├── __init__.py
│           └── algvex_strategy.py
│
├── integration/                 # 【新增】整合层
│   ├── __init__.py
│   ├── bridge.py                # Qlib 信号 → Hummingbot 执行
│   └── api.py                   # 统一 API 入口
│
├── scripts/                     # 【新增】脚本
│   ├── convert_ohlcv_to_qlib.py # OHLCV → Qlib bin 格式
│   └── generate_crypto_calendar.py # 24/7 日历生成
│
└── tests/
    └── integration/             # 集成测试
```

---

## 4. 数据层设计 (最小适配)

### 4.1 只用标准 OHLCV

```
~/.qlib/qlib_data/crypto_data/
├── calendars/
│   └── 5min.txt                 # 24/7 全天候日历
├── instruments/
│   └── perpetual.txt            # BTCUSDT, ETHUSDT...
└── features/
    └── BTCUSDT/
        ├── $open.5min.bin       # 标准 OHLCV 只有这 5 个字段
        ├── $high.5min.bin
        ├── $low.5min.bin
        ├── $close.5min.bin
        └── $volume.5min.bin
```

### 4.2 24/7 日历生成

> **唯一必须的适配**: 股票 252 天/年 → 加密货币 365 天/年 × 24 小时

```python
# scripts/generate_crypto_calendar.py
import pandas as pd

def build_crypto_calendar(start: str, end: str, freq: str = "5min") -> pd.DatetimeIndex:
    """生成 24/7 全天候日历"""
    return pd.date_range(start=start, end=end, freq=freq, tz="UTC")

# 输出格式: calendars/5min.txt
# 2024-01-01 00:00:00
# 2024-01-01 00:05:00
# ... (每天 288 条)
```

### 4.3 OHLCV 数据转换

```python
# scripts/convert_ohlcv_to_qlib.py
import numpy as np

def save_to_bin(values: np.ndarray, path: str) -> None:
    """写入 Qlib bin 格式"""
    arr = np.asarray(values, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(arr.tobytes(order="C"))

def convert_ohlcv(df, symbol, output_dir):
    """
    输入: DataFrame with columns [timestamp, open, high, low, close, volume]
    输出: Qlib bin 文件
    """
    for field in ["open", "high", "low", "close", "volume"]:
        path = f"{output_dir}/features/{symbol}/${field}.5min.bin"
        save_to_bin(df[field].values, path)
```

---

## 5. 因子层设计

### 5.1 使用 Qlib 原版 Alpha158

> **不自定义因子**，直接使用 Qlib 原版 Alpha158 (158 个因子)

```python
# 使用 Qlib 原版 DataHandler
from qlib.contrib.data.handler import Alpha158

# Alpha158 包含 158 个因子，基于 OHLCV 计算:
# - KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
# - ROC_5, ROC_10, ROC_20, ROC_30, ROC_60
# - MA_5, MA_10, MA_20, MA_30, MA_60
# - STD_5, STD_10, STD_20, STD_30, STD_60
# - BETA_5, BETA_10, BETA_20, BETA_30, BETA_60
# - RSQR_5, RSQR_10, RSQR_20, RSQR_30, RSQR_60
# - RESI_5, RESI_10, RESI_20, RESI_30, RESI_60
# - MAX_5, MAX_10, MAX_20, MAX_30, MAX_60
# - MIN_5, MIN_10, MIN_20, MIN_30, MIN_60
# - QTLU_5, QTLU_10, QTLU_20, QTLU_30, QTLU_60
# - QTLD_5, QTLD_10, QTLD_20, QTLD_30, QTLD_60
# - RANK_5, RANK_10, RANK_20, RANK_30, RANK_60
# - RSV_5, RSV_10, RSV_20, RSV_30, RSV_60
# - IMAX_5, IMAX_10, IMAX_20, IMAX_30, IMAX_60
# - IMIN_5, IMIN_10, IMIN_20, IMIN_30, IMIN_60
# - IMXD_5, IMXD_10, IMXD_20, IMXD_30, IMXD_60
# - CORR_5, CORR_10, CORR_20, CORR_30, CORR_60
# - CORD_5, CORD_10, CORD_20, CORD_30, CORD_60
# - CNTP_5, CNTP_10, CNTP_20, CNTP_30, CNTP_60
# - CNTN_5, CNTN_10, CNTN_20, CNTN_30, CNTN_60
# - CNTD_5, CNTD_10, CNTD_20, CNTD_30, CNTD_60
# - SUMP_5, SUMP_10, SUMP_20, SUMP_30, SUMP_60
# - SUMN_5, SUMN_10, SUMN_20, SUMN_30, SUMN_60
# - SUMD_5, SUMD_10, SUMD_20, SUMD_30, SUMD_60
# - VMA_5, VMA_10, VMA_20, VMA_30, VMA_60
# - VSTD_5, VSTD_10, VSTD_20, VSTD_30, VSTD_60
# - WVMA_5, WVMA_10, WVMA_20, WVMA_30, WVMA_60
# - VSUMP_5, VSUMP_10, VSUMP_20, VSUMP_30, VSUMP_60
# - VSUMN_5, VSUMN_10, VSUMN_20, VSUMN_30, VSUMN_60
# - VSUMD_5, VSUMD_10, VSUMD_20, VSUMD_30, VSUMD_60
```

### 5.2 使用 Qlib 原版 Processor

```python
# 使用 Qlib 原版 Processor (14个)
from qlib.data.dataset.processor import (
    ZScoreNorm,      # z-score 标准化
    RobustZScoreNorm, # 鲁棒 z-score
    CSZScoreNorm,    # 截面 z-score
    CSRankNorm,      # 截面排名
    MinMaxNorm,      # 归一化
    Fillna,          # 缺失值填充
    DropnaLabel,     # 删除缺失标签
    DropnaFeature,   # 删除缺失特征
    # ... 等
)
```

---

## 6. 执行层设计

### 6.1 使用 Hummingbot 原版执行器

```python
# 使用 Hummingbot 原版 PositionExecutor
from hummingbot.strategy_v2.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TripleBarrierConfig,
)

config = PositionExecutorConfig(
    connector_name="binance_perpetual",
    trading_pair="BTC-USDT",
    triple_barrier_config=TripleBarrierConfig(
        stop_loss=Decimal("0.03"),          # 止损 3%
        take_profit=Decimal("0.06"),        # 止盈 6%
        time_limit=60 * 60 * 24,            # 时间限制 24小时
    )
)
```

### 6.2 信号桥接 (SignalBridge)

```python
# algvex/integration/bridge.py
class SignalBridge:
    """
    Qlib 信号 → Hummingbot 执行

    这是唯一需要新增的整合代码
    """

    def __init__(self, connector):
        self.connector = connector

    async def execute_signal(self, signal) -> dict:
        """执行信号"""
        if signal.direction > 0:
            order_id = self.connector.buy(...)
        else:
            order_id = self.connector.sell(...)

        return {"status": "submitted", "order_id": order_id}
```

---

## 7. 模型训练

### 7.1 使用 Qlib 原版 LGBModel

```python
# 使用 Qlib 原版模型 (35个模型全部可用)
from qlib.contrib.model.gbdt import LGBModel

model = LGBModel(
    loss="mse",
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=8,
    n_estimators=200,
    num_leaves=100,
)

# 训练
model.fit(dataset)

# 预测
pred = model.predict(dataset)
```

### 7.2 使用 Qlib 原版回测

```python
# 使用 Qlib 原版回测
from qlib.backtest import backtest_loop
from qlib.contrib.evaluate import risk_analysis

# 运行回测
report = backtest_loop(...)

# 评估
metrics = risk_analysis(report["return"])
print(f"Sharpe: {metrics['information_ratio']:.2f}")
print(f"MaxDD: {metrics['max_drawdown']:.2%}")
```

---

## 8. 实施步骤

### Phase 1: 复制源码 (1天)

| 步骤 | 任务 | 命令 |
|------|------|------|
| 1.1 | 备份现有代码 | `git branch backup-before-refactor` |
| 1.2 | 复制 Qlib 源码 | `cp -r qlib-0.9.7/qlib algvex/qlib` |
| 1.3 | 复制 Hummingbot 源码 | `cp -r hummingbot-2.11.0/hummingbot algvex/hummingbot` |
| 1.4 | 验证文件完整性 | `find algvex/qlib -name "*.py" \| wc -l` |

### Phase 2: 数据准备 (1天)

| 步骤 | 任务 | 说明 |
|------|------|------|
| 2.1 | 生成 24/7 日历 | `python scripts/generate_crypto_calendar.py` |
| 2.2 | 下载 OHLCV 数据 | 从 Binance 下载 BTC, ETH 的 5min K线 |
| 2.3 | 转换为 Qlib 格式 | `python scripts/convert_ohlcv_to_qlib.py` |

### Phase 3: 验证 Qlib 功能 (1天)

| 步骤 | 任务 | 验证标准 |
|------|------|----------|
| 3.1 | Alpha158 因子计算 | 158 个因子全部计算成功 |
| 3.2 | LGBModel 训练 | `model.fit()` 正常完成 |
| 3.3 | 回测运行 | `backtest_loop()` 正常完成 |
| 3.4 | 评估指标 | `risk_analysis()` 输出正确 |

### Phase 4: 验证 Hummingbot 功能 (1天)

| 步骤 | 任务 | 验证标准 |
|------|------|----------|
| 4.1 | 连接器测试 | `binance_perpetual` 连接成功 |
| 4.2 | 执行器测试 | `PositionExecutor` 正常工作 |
| 4.3 | 模拟下单 | Paper trading 正常 |

### Phase 5: 信号桥接 (1天)

| 步骤 | 任务 | 说明 |
|------|------|------|
| 5.1 | 实现 SignalBridge | `integration/bridge.py` |
| 5.2 | 实现 AlgVexStrategy | `hummingbot/strategy/algvex_strategy/` |
| 5.3 | 端到端测试 | Qlib 信号 → Hummingbot 订单 |

---

## 9. 验收标准

### 9.1 功能验收

| 功能 | 验收标准 |
|------|----------|
| Qlib 因子计算 | Alpha158 所有因子可用 |
| Qlib 模型训练 | LGBModel.fit() 正常 |
| Qlib 回测 | backtest_loop() 正常 |
| Qlib 评估 | risk_analysis, calc_ic 正常 |
| Hummingbot 连接 | binance_perpetual 连接成功 |
| Hummingbot 执行 | PositionExecutor 正常 |
| 信号桥接 | Qlib 信号 → Hummingbot 订单 |

### 9.2 数据验收

| 项目 | 验收标准 |
|------|----------|
| 24/7 日历 | 每天 288 条 (5min 频率) |
| OHLCV 数据 | 无缺失，时间对齐 |
| bin 文件 | float32, little-endian |

---

## 10. Qlib 原版功能清单

### 10.1 运算符 (58个)

```
ops.py: Ref, Mean, Std, Var, Skew, Kurt, Max, Min, Med, Mad, Rank,
        Count, Sum, Prod, Delta, Abs, Sign, Log, Power, Add, Sub,
        Mul, Div, Greater, Less, And, Or, Not, If, Feature,
        Corr, Cov, EMA, WMA, SMA, MACD, RSI, ROC, BETA, RSQR,
        RESI, MAX, MIN, QTLU, QTLD, RANK, RSV, IMAX, IMIN, IMXD,
        CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA, VSTD
```

### 10.2 Processor (14个)

```
processor.py: ZScoreNorm, RobustZScoreNorm, CSZScoreNorm, CSRankNorm,
              MinMaxNorm, Fillna, DropnaLabel, DropnaFeature,
              FilterCol, TanhProcess, ProcessInf, HashStockFormat,
              TimeRangeFlt, ChangeInstrument
```

### 10.3 模型 (35个)

```
LGBModel, XGBModel, CatBoostModel, LinearModel, ElasticNetModel,
LSTM, GRU, ALSTM, Transformer, TRA, LocalFormer,
GATs, GATsTS, TCN, TCTS,
HIST, SFM, TabNet, AdaRNN, ADD, ADARNN, IGMTF,
DoubleEnsemble, DEnsemble, KRNN, ...
```

### 10.4 DataHandler

```
Alpha158 - 158 个因子 (基于 OHLCV)
Alpha360 - 360 个因子 (扩展版)
```

---

## 11. Hummingbot 原版功能清单

### 11.1 交易所连接器 (12个)

```
CEX: Binance, Bybit, OKX, KuCoin, Gate.io, Bitget, BitMart
DEX: Hyperliquid, dYdX v4, Injective v2, Derive, Vertex
```

### 11.2 执行器 (6个)

```
PositionExecutor  - 单仓位执行 (三重屏障)
DCAExecutor       - 定投/分批建仓
TWAPExecutor      - 时间加权平均价格执行
GridExecutor      - 网格交易执行
ArbitrageExecutor - 跨交易所套利
XEMMExecutor      - 跨交易所做市
```

---

## 12. 文件清单

### 12.1 需要复制

```bash
qlib-0.9.7/qlib/ → algvex/qlib/              # 全部复制，原版保留
hummingbot-2.11.0/hummingbot/ → algvex/hummingbot/  # 全部复制，原版保留
```

### 12.2 需要新增

```bash
# 整合层 (最小新增)
algvex/integration/__init__.py
algvex/integration/bridge.py         # SignalBridge
algvex/integration/api.py            # 统一 API

# Hummingbot 策略
algvex/hummingbot/strategy/algvex_strategy/__init__.py
algvex/hummingbot/strategy/algvex_strategy/algvex_strategy.py

# 数据脚本
scripts/convert_ohlcv_to_qlib.py     # OHLCV → Qlib bin
scripts/generate_crypto_calendar.py  # 24/7 日历
```

---

*方案版本: v3.0 | 日期: 2025-12-24*
*核心目标: 先跑通系统，只用标准 OHLCV + Qlib 原版 Alpha158*
*原则: 完全保留 Qlib + Hummingbot 原版功能，最小必要适配*

---

## 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v3.0 | 2025-12-24 | 大幅简化: 聚焦跑通系统。删除后期内容(自定义因子/Processor/订单簿/杠杆爆仓)，只用标准OHLCV + Alpha158 |
| v2.7 | 2025-12-24 | 完善数据层设计(Section 4): Qlib数据格式规范、bin文件结构、24/7日历生成、字段对齐策略 |
| v2.6 | 2025-12-24 | 修复一致性问题: 架构图命名统一、backtest/目录结构完善 |
| v2.5 | 2025-12-24 | 完善第11章: 补充 Qlib 全部模块清单 (35个模型、strategy、workflow、rl) |
| v2.0 | 2025-12-24 | 选择方案B (复制源码)，融入 P1-P10 原则 |
| v1.0 | 2025-12-24 | 初始版本 |
