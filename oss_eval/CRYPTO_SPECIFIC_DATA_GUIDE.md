# 数字货币特有数据与Qlib融合方案

## 一、数字货币特有数据全景图

### 1.1 数据分类总览

| 类别 | 数据类型 | 原版Qlib支持 | 融合难度 | 价值评级 |
|------|---------|-------------|---------|---------|
| **交易所数据** | 资金费率 | ❌ | 低 | ⭐⭐⭐⭐⭐ |
| | 持仓量(OI) | ❌ | 低 | ⭐⭐⭐⭐⭐ |
| | 爆仓数据 | ❌ | 低 | ⭐⭐⭐⭐ |
| | 多空比 | ❌ | 低 | ⭐⭐⭐⭐ |
| | 大户持仓比 | ❌ | 低 | ⭐⭐⭐⭐ |
| | 主动买卖比 | ❌ | 低 | ⭐⭐⭐⭐ |
| | 基差/溢价 | ❌ | 低 | ⭐⭐⭐⭐ |
| **链上数据** | 交易所净流入 | ❌ | 中 | ⭐⭐⭐⭐⭐ |
| | 巨鲸地址追踪 | ❌ | 中 | ⭐⭐⭐⭐ |
| | 活跃地址数 | ❌ | 中 | ⭐⭐⭐ |
| | 算力/难度 | ❌ | 中 | ⭐⭐⭐ |
| | MVRV/NVT | ❌ | 中 | ⭐⭐⭐⭐ |
| **稳定币数据** | USDT发行量 | ❌ | 中 | ⭐⭐⭐⭐ |
| | 稳定币交易所流入 | ❌ | 中 | ⭐⭐⭐⭐ |
| **DeFi数据** | TVL (锁仓量) | ❌ | 中 | ⭐⭐⭐ |
| | DEX交易量 | ❌ | 中 | ⭐⭐⭐ |
| | 借贷利率 | ❌ | 中 | ⭐⭐⭐ |
| **情绪数据** | 恐惧贪婪指数 | ❌ | 低 | ⭐⭐⭐ |
| | 社交媒体热度 | ❌ | 高 | ⭐⭐ |

---

## 二、交易所特有数据 (最高优先级)

### 2.1 资金费率 (Funding Rate)

**数据说明**: 永续合约特有，每8小时结算一次，用于锚定现货价格

**数据源**:
```python
# CCXT获取
funding = await exchange.fetch_funding_rate('BTC/USDT')
funding_history = await exchange.fetch_funding_rate_history('BTC/USDT', limit=500)

# 币安原生API
# GET /fapi/v1/fundingRate
```

**Qlib字段映射**:
```python
$funding_rate        # 当前资金费率 (每8小时)
$funding_rate_1d     # 日累计资金费率
$funding_rate_7d     # 7日累计
$funding_rate_30d    # 30日累计 (年化约 = 30d * 3 * 365 / 30)
```

**分析因子**:
```python
FUNDING_FEATURES = [
    # 资金费率原始值
    ("$funding_rate", "fr"),

    # 资金费率变化
    ("$funding_rate - Ref($funding_rate, 1)", "fr_change"),

    # 资金费率均值回归
    ("$funding_rate - Mean($funding_rate, 30)", "fr_deviation"),

    # 极端资金费率 (>0.1% 或 <-0.1%)
    ("Abs($funding_rate) > 0.001", "fr_extreme"),

    # 持续正/负费率天数
    ("TsCount($funding_rate > 0, 30)", "fr_positive_days"),

    # 资金费率与收益率相关性
    ("Corr($funding_rate, $close / Ref($close, 1) - 1, 20)", "fr_return_corr"),
]
```

### 2.2 持仓量 (Open Interest)

**数据说明**: 未平仓合约总量，反映市场参与度和杠杆水平

**数据源**:
```python
# CCXT获取
oi = await exchange.fetch_open_interest('BTC/USDT')
oi_history = await exchange.fetch_open_interest_history('BTC/USDT', '1h', limit=500)

# 币安原生API
# GET /fapi/v1/openInterest
# GET /futures/data/openInterestHist
```

**Qlib字段映射**:
```python
$open_interest       # 持仓量 (张数)
$open_interest_value # 持仓价值 (USDT)
```

**分析因子**:
```python
OI_FEATURES = [
    # 持仓量变化率
    ("$open_interest / Ref($open_interest, 1) - 1", "oi_change"),

    # 持仓量与价格背离 (价涨仓降 = 空头回补)
    ("Corr($open_interest, $close, 20)", "oi_price_corr"),

    # 持仓量相对历史位置
    ("($open_interest - Min($open_interest, 30)) / (Max($open_interest, 30) - Min($open_interest, 30))", "oi_percentile"),

    # 持仓量异常放大
    ("$open_interest / Mean($open_interest, 20)", "oi_ratio"),

    # 价涨仓增 (趋势确认)
    ("($close > Ref($close, 1)) & ($open_interest > Ref($open_interest, 1))", "oi_trend_confirm"),
]
```

### 2.3 爆仓数据 (Liquidations)

**数据说明**: 强制平仓订单，反映市场极端波动和杠杆清洗

**数据源**:
```python
# 币安原生API
# GET /fapi/v1/forceOrders (需要API权限)
# 或使用第三方聚合: Coinglass API

# Coinglass示例
import requests
resp = requests.get('https://open-api.coinglass.com/public/v2/liquidation_history',
                    params={'symbol': 'BTC', 'interval': '1h'})
```

**Qlib字段映射**:
```python
$liq_long_volume     # 多头爆仓量
$liq_short_volume    # 空头爆仓量
$liq_long_value      # 多头爆仓价值
$liq_short_value     # 空头爆仓价值
```

**分析因子**:
```python
LIQUIDATION_FEATURES = [
    # 爆仓净额 (正=多头爆仓多)
    ("$liq_long_value - $liq_short_value", "liq_net"),

    # 爆仓不平衡
    ("($liq_long_value - $liq_short_value) / ($liq_long_value + $liq_short_value + 1)", "liq_imbalance"),

    # 爆仓量异常
    ("($liq_long_value + $liq_short_value) / Mean($liq_long_value + $liq_short_value, 24)", "liq_surge"),

    # 大额爆仓后反转信号
    ("Ref($liq_long_value > Quantile($liq_long_value, 24, 0.9), 1)", "liq_reversal_signal"),
]
```

### 2.4 多空比 (Long/Short Ratio)

**数据说明**: 持仓账户的多空比例

**数据源**:
```python
# 币安原生API
# GET /futures/data/globalLongShortAccountRatio  (全市场)
# GET /futures/data/topLongShortAccountRatio    (大户)
# GET /futures/data/topLongShortPositionRatio   (大户持仓)
```

**Qlib字段映射**:
```python
$ls_ratio_global     # 全市场多空账户比
$ls_ratio_top        # 大户多空账户比
$ls_position_top     # 大户多空持仓比
```

**分析因子**:
```python
LS_FEATURES = [
    # 多空比绝对值
    ("$ls_ratio_global", "ls_global"),

    # 多空比变化
    ("$ls_ratio_global - Ref($ls_ratio_global, 1)", "ls_change"),

    # 散户与大户背离 (逆向指标)
    ("$ls_ratio_global - $ls_ratio_top", "ls_divergence"),

    # 极端多空比 (反转信号)
    ("$ls_ratio_global > Quantile($ls_ratio_global, 30, 0.9)", "ls_extreme_long"),
    ("$ls_ratio_global < Quantile($ls_ratio_global, 30, 0.1)", "ls_extreme_short"),
]
```

### 2.5 主动买卖比 (Taker Buy/Sell Ratio)

**数据说明**: 主动买入与卖出的成交量比例

**数据源**:
```python
# 币安原生API
# GET /futures/data/takerlongshortRatio
```

**Qlib字段映射**:
```python
$taker_buy_volume    # 主动买入量
$taker_sell_volume   # 主动卖出量
$taker_ratio         # 买卖比
```

**分析因子**:
```python
TAKER_FEATURES = [
    # 主动买卖不平衡
    ("($taker_buy_volume - $taker_sell_volume) / ($taker_buy_volume + $taker_sell_volume)", "taker_imbalance"),

    # 主动买入占比趋势
    ("Mean($taker_ratio, 5) - Mean($taker_ratio, 20)", "taker_momentum"),
]
```

### 2.6 基差/溢价 (Basis/Premium)

**数据说明**: 永续合约价格与现货价格的差异

**数据源**:
```python
# 计算方式
# basis = 永续价格 - 现货价格
# premium = (永续价格 - 现货价格) / 现货价格

# 获取现货价格
spot = await exchange.fetch_ticker('BTC/USDT', params={'type': 'spot'})
# 获取永续价格
perp = await exchange.fetch_ticker('BTC/USDT', params={'type': 'future'})
```

**Qlib字段映射**:
```python
$spot_price          # 现货价格
$perp_price          # 永续价格
$basis               # 基差 = perp - spot
$premium_rate        # 溢价率 = (perp - spot) / spot
```

**分析因子**:
```python
BASIS_FEATURES = [
    # 溢价率
    ("$premium_rate * 10000", "premium_bps"),

    # 溢价率变化
    ("$premium_rate - Ref($premium_rate, 1)", "premium_change"),

    # 溢价率与资金费率关系
    ("Corr($premium_rate, $funding_rate, 24)", "premium_fr_corr"),

    # 溢价率均值回归
    ("$premium_rate - Mean($premium_rate, 24)", "premium_deviation"),
]
```

---

## 三、链上数据 (On-Chain Data)

### 3.1 交易所净流入

**数据说明**: BTC/ETH等流入交易所的净额，反映抛售压力

**数据源**:
```python
# Glassnode API (付费)
# CryptoQuant API (付费)
# IntoTheBlock API

import requests
# Glassnode示例
resp = requests.get('https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net',
                    params={'a': 'BTC', 'api_key': 'YOUR_KEY'})
```

**Qlib字段映射**:
```python
$exchange_inflow     # 交易所流入
$exchange_outflow    # 交易所流出
$exchange_netflow    # 净流入 = inflow - outflow
$exchange_reserve    # 交易所储备
```

**分析因子**:
```python
EXCHANGE_FLOW_FEATURES = [
    # 净流入 (正=抛售压力)
    ("$exchange_netflow", "netflow"),

    # 净流入异常
    ("$exchange_netflow / Std($exchange_netflow, 30)", "netflow_zscore"),

    # 交易所储备变化
    ("$exchange_reserve / Ref($exchange_reserve, 7) - 1", "reserve_change_7d"),

    # 大额流入预警
    ("$exchange_inflow > Quantile($exchange_inflow, 30, 0.95)", "inflow_spike"),
]
```

### 3.2 巨鲸地址追踪

**数据说明**: 持有大量币的地址行为

**数据源**:
```python
# Whale Alert API
# Glassnode whale metrics
# Santiment API
```

**Qlib字段映射**:
```python
$whale_count         # 巨鲸地址数量 (>1000 BTC)
$whale_balance       # 巨鲸总持仓
$whale_transactions  # 巨鲸交易笔数
```

### 3.3 链上估值指标

**MVRV (Market Value to Realized Value)**:
```python
$mvrv_ratio          # MVRV = 市值 / 已实现市值
# MVRV > 3.5 通常是周期顶部
# MVRV < 1 通常是周期底部
```

**NVT (Network Value to Transactions)**:
```python
$nvt_ratio           # NVT = 市值 / 链上交易量
# 类似股票P/E，高NVT可能高估
```

---

## 四、稳定币数据

### 4.1 稳定币发行量

**数据说明**: USDT/USDC等稳定币总供应量变化

**数据源**:
```python
# Glassnode / DefiLlama
# 直接从区块链读取
```

**Qlib字段映射**:
```python
$usdt_supply         # USDT总供应量
$usdc_supply         # USDC总供应量
$stablecoin_total    # 稳定币总量
```

**分析因子**:
```python
STABLECOIN_FEATURES = [
    # 稳定币增发 (买入力量)
    ("$usdt_supply / Ref($usdt_supply, 7) - 1", "usdt_growth_7d"),

    # 稳定币市值占比
    ("$stablecoin_total / $btc_marketcap", "stablecoin_ratio"),
]
```

### 4.2 稳定币交易所流入

**Qlib字段映射**:
```python
$stablecoin_exchange_inflow  # 稳定币流入交易所 (买入信号)
```

---

## 五、DeFi数据

### 5.1 TVL (Total Value Locked)

**数据源**:
```python
# DefiLlama API (免费)
import requests
resp = requests.get('https://api.llama.fi/v2/historicalChainTvl/Ethereum')
```

**Qlib字段映射**:
```python
$tvl_total           # 总锁仓量
$tvl_change          # TVL变化
```

### 5.2 DEX交易量

```python
$dex_volume          # DEX日交易量
$dex_cex_ratio       # DEX/CEX交易量比值
```

---

## 六、情绪数据

### 6.1 恐惧贪婪指数

**数据源**:
```python
# Alternative.me API (免费)
import requests
resp = requests.get('https://api.alternative.me/fng/?limit=30')
```

**Qlib字段映射**:
```python
$fear_greed_index    # 0-100, 0=极度恐惧, 100=极度贪婪
```

**分析因子**:
```python
SENTIMENT_FEATURES = [
    # 恐惧贪婪指数
    ("$fear_greed_index", "fgi"),

    # 极端恐惧 (买入信号)
    ("$fear_greed_index < 20", "extreme_fear"),

    # 极端贪婪 (卖出信号)
    ("$fear_greed_index > 80", "extreme_greed"),

    # 情绪变化
    ("$fear_greed_index - Ref($fear_greed_index, 7)", "fgi_change"),
]
```

---

## 七、融合架构

### 7.1 数据收集层

```python
# data_collector/crypto_extended_collector.py

class CryptoExtendedCollector:
    """扩展数据收集器"""

    def __init__(self, symbols: list):
        self.symbols = symbols
        self.ccxt_exchange = None

    async def collect_all(self, date: str) -> pd.DataFrame:
        """收集所有扩展数据"""

        data = {}

        # 1. 交易所数据 (CCXT)
        data.update(await self._collect_exchange_data())

        # 2. 链上数据 (Glassnode/CryptoQuant)
        data.update(await self._collect_onchain_data())

        # 3. 稳定币数据
        data.update(await self._collect_stablecoin_data())

        # 4. 情绪数据
        data.update(await self._collect_sentiment_data())

        return pd.DataFrame(data)

    async def _collect_exchange_data(self) -> dict:
        """收集交易所数据"""
        result = {}

        for symbol in self.symbols:
            # 资金费率
            fr = await self.ccxt_exchange.fetch_funding_rate(symbol)
            result[f'{symbol}_funding_rate'] = fr['fundingRate']

            # 持仓量
            oi = await self.ccxt_exchange.fetch_open_interest(symbol)
            result[f'{symbol}_open_interest'] = oi['openInterestAmount']

        return result
```

### 7.2 完整数据Schema

```python
# 扩展Qlib字段 (永续合约完整版)
CRYPTO_PERPETUAL_SCHEMA = {
    # === 基础OHLCV ===
    '$open': 'float',
    '$high': 'float',
    '$low': 'float',
    '$close': 'float',
    '$volume': 'float',
    '$factor': 'float',  # 永续=1.0

    # === 订单簿 ===
    '$bid1': 'float', '$bsize1': 'float',
    '$ask1': 'float', '$asize1': 'float',
    # ... up to bid20/ask20

    # === 永续合约特有 ===
    '$funding_rate': 'float',
    '$open_interest': 'float',
    '$open_interest_value': 'float',

    # === 多空数据 ===
    '$ls_ratio_global': 'float',
    '$ls_ratio_top': 'float',
    '$taker_buy_volume': 'float',
    '$taker_sell_volume': 'float',

    # === 爆仓数据 ===
    '$liq_long_value': 'float',
    '$liq_short_value': 'float',

    # === 基差 ===
    '$spot_price': 'float',
    '$premium_rate': 'float',

    # === 链上数据 ===
    '$exchange_netflow': 'float',
    '$exchange_reserve': 'float',
    '$mvrv_ratio': 'float',

    # === 稳定币 ===
    '$usdt_supply': 'float',
    '$stablecoin_exchange_inflow': 'float',

    # === 情绪 ===
    '$fear_greed_index': 'float',
}
```

---

## 八、实施优先级

| 优先级 | 数据类型 | 数据源 | 代码量 | 价值 |
|-------|---------|--------|--------|------|
| **P0** | 资金费率 | CCXT | 50行 | 极高 |
| **P0** | 持仓量 | CCXT | 50行 | 极高 |
| **P0** | 多空比 | 币安API | 80行 | 高 |
| **P1** | 爆仓数据 | Coinglass | 100行 | 高 |
| **P1** | 主动买卖比 | 币安API | 50行 | 高 |
| **P1** | 基差/溢价 | CCXT计算 | 30行 | 高 |
| **P2** | 交易所流入 | Glassnode | 100行 | 高 |
| **P2** | 稳定币供应 | DefiLlama | 80行 | 中 |
| **P3** | 恐惧贪婪指数 | Alternative.me | 30行 | 中 |
| **P3** | TVL | DefiLlama | 50行 | 低 |

---

## 九、总结

### 9.1 核心扩展 (必须)

1. **资金费率** - 永续合约最核心指标
2. **持仓量** - 市场杠杆和参与度
3. **多空比** - 市场情绪

### 9.2 增强分析 (推荐)

4. **爆仓数据** - 极端行情指标
5. **交易所流入** - 链上抛压信号
6. **主动买卖比** - 微观结构

### 9.3 补充指标 (可选)

7. **稳定币数据** - 宏观资金面
8. **情绪指数** - 辅助判断

**预估总代码量**: ~800行 (P0+P1)
