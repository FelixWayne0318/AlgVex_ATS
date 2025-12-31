# 币安数字货币数据获取与Qlib/ValueCell融合指南

## 一、币安永续合约特有数据

### 1.1 币安USDT-M永续合约可获取的数据类型

| 数据类型 | API端点 | 说明 | 更新频率 |
|---------|---------|------|---------|
| **K线数据 (OHLCV)** | `GET /fapi/v1/klines` | 开高低收量 | 实时 |
| **资金费率** | `GET /fapi/v1/fundingRate` | 多空平衡费率 | 每8小时 |
| **持仓量 (Open Interest)** | `GET /fapi/v1/openInterest` | 未平仓合约量 | 实时 |
| **多空比** | `GET /futures/data/globalLongShortAccountRatio` | 账户多空比例 | 5分钟 |
| **大户持仓比** | `GET /futures/data/topLongShortAccountRatio` | 头部账户多空比 | 5分钟 |
| **爆仓数据** | `GET /futures/data/forceOrders` | 强平订单 | 实时 |
| **标记价格** | `GET /fapi/v1/premiumIndex` | 用于计算资金费率 | 实时 |
| **成交量统计** | `GET /futures/data/takerlongshortRatio` | 主动买卖比 | 5分钟 |
| **深度数据** | `GET /fapi/v1/depth` | 订单簿 | 实时 |

### 1.2 通过CCXT获取币安数据的方法

```python
import ccxt.async_support as ccxt

async def fetch_binance_futures_data():
    exchange = ccxt.binance({
        'options': {'defaultType': 'future'}  # USDT-M永续
    })
    await exchange.load_markets()

    # 1. K线数据
    ohlcv = await exchange.fetch_ohlcv('BTC/USDT', '1h', limit=100)

    # 2. 资金费率
    funding_rate = await exchange.fetch_funding_rate('BTC/USDT')

    # 3. 持仓量
    open_interest = await exchange.fetch_open_interest('BTC/USDT')

    # 4. 资金费率历史
    funding_history = await exchange.fetch_funding_rate_history('BTC/USDT', limit=100)

    # 5. 订单簿
    orderbook = await exchange.fetch_order_book('BTC/USDT')

    await exchange.close()
    return {
        'ohlcv': ohlcv,
        'funding_rate': funding_rate,
        'open_interest': open_interest,
        'funding_history': funding_history,
        'orderbook': orderbook
    }
```

### 1.3 CCXT支持的币安期货数据获取方法

| 方法名 | 返回数据 | Qlib可分析 |
|-------|---------|-----------|
| `fetch_ohlcv()` | K线数据 | ✅ 完全兼容 |
| `fetch_funding_rate()` | 当前资金费率 | ✅ 需转换 |
| `fetch_funding_rate_history()` | 历史资金费率 | ✅ 需转换 |
| `fetch_open_interest()` | 持仓量 | ✅ 需转换 |
| `fetch_open_interest_history()` | 历史持仓量 | ✅ 需转换 |
| `fetch_ticker()` | 行情快照 | ✅ 部分字段 |
| `fetch_order_book()` | 订单簿深度 | ⚠️ 需预处理 |
| `fetch_trades()` | 逐笔成交 | ⚠️ 需聚合 |
| `fetch_positions()` | 当前持仓 | ❌ 账户数据 |
| `fetch_balance()` | 账户余额 | ❌ 账户数据 |

---

## 二、Qlib数据格式规范

### 2.1 Qlib标准字段

Qlib使用 `$` 前缀表示原始特征:

```python
# 必需字段 (回测必须)
$open      # 开盘价
$close     # 收盘价
$high      # 最高价
$low       # 最低价
$volume    # 成交量
$factor    # 复权因子 (永续合约可设为1.0)

# 可选字段
$vwap      # 成交量加权平均价
$change    # 涨跌幅
$amount    # 成交额
```

### 2.2 Qlib数据存储格式

```
~/.qlib/qlib_data/
├── calendars/
│   └── day.txt          # 交易日历
├── instruments/
│   └── all.txt          # 标的列表
└── features/
    └── btcusdt/         # 每个标的一个目录
        ├── open.day.bin
        ├── close.day.bin
        ├── high.day.bin
        ├── low.day.bin
        └── volume.day.bin
```

### 2.3 扩展字段 (永续合约特有)

可以添加自定义字段供Qlib分析:

```python
# 永续合约扩展字段
$funding_rate     # 资金费率 (每8小时)
$open_interest    # 持仓量
$long_short_ratio # 多空比
$liquidation_vol  # 爆仓量
$mark_price       # 标记价格
$index_price      # 指数价格
$basis            # 基差 = 标记价格 - 指数价格
```

---

## 三、Qlib与ValueCell融合方案

### 3.1 数据格式对比

| 字段 | Qlib格式 | ValueCell格式 | 转换难度 |
|-----|---------|--------------|---------|
| 时间索引 | `datetime` (MultiIndex) | `timestamp` (单列) | 低 |
| 标的标识 | `instrument` (MultiIndex) | `symbol` (单列) | 低 |
| OHLCV | `$open/$close/$high/$low/$volume` | `open/close/high/low/volume` | 低 |
| 资金费率 | 需扩展 | 实时获取 | 中 |
| 持仓数据 | `Position`类 | `TradeInstruction` | 中 |

### 3.2 融合架构

```
┌─────────────────────────────────────────────────────────────┐
│                     统一数据层                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐     ┌───────────────┐                   │
│  │  CCXT数据获取  │────▶│  数据转换器    │                   │
│  │  (币安API)     │     │  (适配器)      │                   │
│  └───────────────┘     └───────┬───────┘                   │
│                                │                            │
│         ┌──────────────────────┴──────────────────────┐    │
│         ▼                                              ▼    │
│  ┌─────────────┐                              ┌────────────┐│
│  │   Qlib      │                              │  ValueCell ││
│  │  数据格式    │                              │   执行层   ││
│  │             │                              │            ││
│  │ - 回测分析   │◀─────── 信号 ─────────────────│ - 实盘下单 ││
│  │ - 特征工程   │                              │ - 持仓同步 ││
│  │ - ML训练    │                              │ - 风控检查 ││
│  └─────────────┘                              └────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 数据转换器实现

```python
# adapter/binance_to_qlib.py

import pandas as pd
import numpy as np
from typing import Dict, List
import ccxt.async_support as ccxt

class BinanceToQlibAdapter:
    """将币安CCXT数据转换为Qlib格式"""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.exchange = None

    async def init_exchange(self):
        self.exchange = ccxt.binance({
            'options': {'defaultType': 'future'}
        })
        await self.exchange.load_markets()

    async def fetch_ohlcv_qlib_format(
        self,
        symbol: str,
        timeframe: str = '1d',
        limit: int = 1000
    ) -> pd.DataFrame:
        """获取OHLCV并转换为Qlib格式"""

        ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['instrument'] = symbol.replace('/', '').lower()

        # 转换为Qlib命名
        df = df.rename(columns={
            'open': '$open',
            'high': '$high',
            'low': '$low',
            'close': '$close',
            'volume': '$volume'
        })

        # 永续合约无复权，factor=1
        df['$factor'] = 1.0
        df['$change'] = df['$close'].pct_change()

        # 设置MultiIndex
        df = df.set_index(['datetime', 'instrument'])

        return df[['$open', '$high', '$low', '$close', '$volume', '$factor', '$change']]

    async def fetch_funding_rate_qlib_format(
        self,
        symbol: str,
        limit: int = 500
    ) -> pd.DataFrame:
        """获取资金费率并转换为Qlib格式"""

        funding = await self.exchange.fetch_funding_rate_history(symbol, limit=limit)

        df = pd.DataFrame(funding)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['instrument'] = symbol.replace('/', '').lower()

        # Qlib自定义字段
        df['$funding_rate'] = df['fundingRate']

        df = df.set_index(['datetime', 'instrument'])

        return df[['$funding_rate']]

    async def fetch_open_interest_qlib_format(
        self,
        symbol: str
    ) -> Dict:
        """获取持仓量"""

        oi = await self.exchange.fetch_open_interest(symbol)

        return {
            'datetime': pd.Timestamp(oi['timestamp'], unit='ms'),
            'instrument': symbol.replace('/', '').lower(),
            '$open_interest': oi['openInterestAmount'],
            '$open_interest_value': oi['openInterestValue']
        }

    async def build_full_dataset(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """构建完整的Qlib数据集"""

        all_data = []

        for symbol in self.symbols:
            # OHLCV
            ohlcv = await self.fetch_ohlcv_qlib_format(symbol)

            # 资金费率
            funding = await self.fetch_funding_rate_qlib_format(symbol)

            # 合并
            merged = ohlcv.join(funding, how='left')
            merged['$funding_rate'] = merged['$funding_rate'].fillna(0)

            all_data.append(merged)

        return pd.concat(all_data)

    async def close(self):
        if self.exchange:
            await self.exchange.close()
```

### 3.4 Qlib特征工程示例

```python
# 使用Qlib表达式计算永续合约特有因子

PERPETUAL_FEATURES = [
    # 基础价格因子
    ("$close / Ref($close, 1) - 1", "return_1d"),
    ("$close / Ref($close, 5) - 1", "return_5d"),

    # 波动率因子
    ("Std($close / Ref($close, 1) - 1, 20)", "volatility_20d"),

    # 成交量因子
    ("$volume / Mean($volume, 20)", "volume_ratio"),

    # 资金费率因子 (永续合约特有)
    ("$funding_rate", "funding_rate"),
    ("Mean($funding_rate, 7)", "funding_rate_ma7"),
    ("Sum($funding_rate, 30)", "funding_cost_30d"),  # 30天资金费率成本

    # 持仓量因子
    ("$open_interest / Ref($open_interest, 1) - 1", "oi_change"),
    ("$open_interest / Mean($open_interest, 20)", "oi_ratio"),

    # 多空博弈因子
    ("($close - $low) / ($high - $low + 1e-8)", "close_position"),  # 收盘位置
    ("($high - $open) / ($open - $low + 1e-8)", "upper_lower_ratio"),
]
```

---

## 四、Qlib分析永续合约数据的能力

### 4.1 Qlib可以分析的内容

| 分析类型 | 支持程度 | 说明 |
|---------|---------|------|
| **时序预测** | ✅ 完全支持 | LSTM/Transformer/LightGBM等模型 |
| **因子挖掘** | ✅ 完全支持 | Alpha表达式、遗传算法 |
| **特征工程** | ✅ 完全支持 | 150+内置算子 |
| **回测验证** | ⚠️ 需扩展 | 需添加做空/杠杆/资金费率 |
| **滚动验证** | ✅ 完全支持 | RollingGen WFA |
| **IC/IR分析** | ✅ 完全支持 | 信号评估 |
| **组合优化** | ⚠️ 需扩展 | 需支持空头权重 |

### 4.2 Qlib内置算子 (可用于永续合约分析)

```python
# 统计类
Mean($close, 20)      # 20日均值
Std($close, 20)       # 20日标准差
Skew($close, 20)      # 20日偏度
Kurt($close, 20)      # 20日峰度
Quantile($close, 20, 0.5)  # 20日中位数

# 时序类
Ref($close, 1)        # 前1日收盘价
Diff($close, 1)       # 1日差分
Delta($close, 5)      # 5日变化
Slope($close, 20)     # 20日线性斜率
Rsquare($close, 20)   # 20日R方

# 排序类
Rank($close)          # 横截面排名
TsRank($close, 20)    # 时序排名

# 极值类
Max($high, 20)        # 20日最高
Min($low, 20)         # 20日最低
IdxMax($high, 20)     # 20日最高位置
IdxMin($low, 20)      # 20日最低位置

# 逻辑类
If($close > Ref($close, 1), 1, 0)  # 条件判断
And/Or/Not                          # 逻辑运算
```

### 4.3 永续合约分析示例

```python
from qlib.data import D

# 初始化Qlib
qlib.init(provider_uri='~/.qlib/crypto_data')

# 获取数据
df = D.features(
    instruments=['btcusdt', 'ethusdt'],
    fields=[
        '$close',
        '$volume',
        '$funding_rate',
        '$open_interest',
        'Mean($funding_rate, 7)',  # 资金费率7日均值
        'Std($close / Ref($close, 1) - 1, 20)',  # 20日波动率
    ],
    start_time='2024-01-01',
    end_time='2024-12-31'
)

# 分析资金费率与收益的关系
correlation = df.groupby('instrument').apply(
    lambda x: x['$funding_rate'].corr(x['$close'].pct_change())
)
```

---

## 五、融合实施路线图

### 5.1 第一阶段: 数据层打通 (1-2周)

```
1. 实现 BinanceToQlibAdapter
2. 定时获取币安OHLCV + 资金费率
3. 存储为Qlib bin格式
4. 验证数据完整性
```

### 5.2 第二阶段: 分析层增强 (2-3周)

```
1. 添加永续合约特有因子
2. 实现资金费率成本回测
3. 扩展Qlib回测支持做空
4. 验证因子IC/IR
```

### 5.3 第三阶段: 执行层对接 (1-2周)

```
1. Qlib信号 → ValueCell指令转换
2. 持仓状态双向同步
3. 执行结果反馈到Qlib
4. 端到端测试
```

---

## 六、总结

### 6.1 可行性结论

| 问题 | 答案 |
|------|------|
| Qlib能否获取币安数据? | ✅ 通过CCXT可以获取全部数据 |
| Qlib能否分析永续合约? | ✅ 完全支持,需添加自定义字段 |
| Qlib能否做永续合约回测? | ⚠️ 需扩展做空/杠杆/资金费率 |
| Qlib与ValueCell格式兼容? | ✅ 格式简单,易于转换 |

### 6.2 推荐技术栈

```
数据获取: CCXT (异步)
数据存储: Qlib bin格式 + 自定义字段
特征工程: Qlib表达式引擎
模型训练: Qlib内置模型 (LightGBM/XGBoost/NN)
信号生成: Qlib策略
回测验证: Qlib回测 (需扩展)
实盘执行: ValueCell CCXTExecutionGateway
```

### 6.3 代码改动量估计

| 模块 | 代码量 | 复杂度 |
|------|--------|--------|
| 数据适配器 | 300行 | 低 |
| 永续合约因子 | 200行 | 低 |
| 回测扩展(做空) | 400行 | 中 |
| 资金费率模拟 | 200行 | 中 |
| 信号转换器 | 150行 | 低 |
| **合计** | **~1250行** | **中等** |
