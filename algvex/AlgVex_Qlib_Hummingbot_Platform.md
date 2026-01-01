# AlgVex 实施方案

> **Qlib + Hummingbot 融合的专业加密货币量化交易平台**
>
> 网站: algvex.com
> 版本: 2.0.0
> 更新: 2025-12-31
>
> **v3.10.0 更新**: 完善 Hummingbot 执行层集成设计，新增 Iteration-4 交付计划

---

## 目录

- [0. 硬约束层（必须先过门槛）](#0-硬约束层-版本化配置哈希审计)
  - [0.0 MVP Scope 定义](#00-mvp-scope-定义)
  - [0.1 系统规约原则 (P1-P10) + 落地机制](#01-系统规约原则-p1-p10--落地机制)
  - [0.2 S1: 时间+快照契约](#02-s1-时间快照契约-time--snapshot-contract)
  - [0.3 S2: 数据契约模板](#03-s2-数据契约模板-data-contract-template)
  - [0.4 S3: 预算与降级策略](#04-s3-预算与降级策略-budget--degrade-policy)
  - [0.5 S4: 因子治理](#05-s4-因子治理-factor-governance)
  - [0.6 S5: 对齐与归因 + Daily Replay](#06-s5-对齐与归因--daily-replay-alignment--attribution)
  - [0.7 S6: 验收测试](#07-s6-验收测试-acceptance-tests)
  - [0.8 S7: 物理边界隔离](#08-s7-物理边界隔离-p0-1)
  - [0.9 S8: DataManager唯一入口](#09-s8-datamanager唯一入口-p0-2)
  - [0.10 S9: Canonical Hashing规范](#010-s9-canonical-hashing规范-p0-3)
  - [0.11 S10: Replay确定性保障](#011-s10-replay确定性保障-p0-4)
  - [0.12 Iteration-1/2/3/4 交付计划](#012-iteration-1234-交付计划)
  - [0.13 硬约束层检查清单](#013-硬约束层检查清单)
  - [0.14 逻辑一致性审查](#014-逻辑一致性审查v397-增补)
  - [0.15 复杂度证据化](#015-复杂度证据化把复杂写成可验证的代价与收益)
  - [0.16 合理决策](#016-合理决策如何保证复杂度不是乱加的不砍功能但把高配变成可控开关)
- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 数据层](#3-数据层)
- [4. 信号层](#4-信号层)
- [5. 回测层](#5-回测层)
- [6. 执行层](#6-执行层)
- [7. 风控层](#7-风控层)
- [8. 技术栈](#8-技术栈)
- [9. 目录结构](#9-目录结构)
- [10. 部署方案](#10-部署方案)
- [11. P0 验收标准](#11-p0-验收标准-上线前必须完成)
- [12. 开发路线图](#12-开发路线图)
- [文档总结](#文档总结)

---

## 📋 v2.0.0 更新日志 (2025-12-31)

### 🆕 重大更新: Qlib + Hummingbot 完整功能实现

#### 1. Qlib 模型封装层 (`research/qlib_models.py`)

完整封装 Qlib 0.9.7 所有 25+ 模型:

| 模型类别 | 模型列表 |
|----------|----------|
| **GBDT** | LightGBM, XGBoost, CatBoost |
| **线性模型** | Linear, Ridge, Lasso |
| **基础DL** | LSTM, GRU, MLP, TCN |
| **高级DL** | Transformer, ALSTM, TabNet, GATS, SFM, HIST, TRA |
| **集成模型** | DoubleEnsemble |
| **其他** | GAT, IGMTF, ADD, ADARNN, TCTS, Localformer |

```python
from algvex.research.qlib_models import ModelFactory, ModelType

# 创建模型
model = ModelFactory.create(ModelType.TRANSFORMER, d_model=64, n_heads=8)
model.fit(dataset)
predictions = model.predict(dataset)
```

#### 2. 交易所连接器 (`core/execution/exchange_connectors.py`)

支持多交易所永续合约交易:

| 交易所 | 功能 |
|--------|------|
| **Binance Perpetual** | 订单、持仓、账户、K线、资金费率 |
| **Bybit Perpetual** | 订单、持仓、账户、K线、资金费率 |
| **OKX** (预留) | 架构已支持 |
| **Gate.io** (预留) | 架构已支持 |

```python
from algvex.core.execution.exchange_connectors import (
    BinancePerpetualConnector, BybitPerpetualConnector
)

connector = BinancePerpetualConnector(api_key, api_secret)
await connector.connect()
order = await connector.create_order(OrderRequest(...))
positions = await connector.get_positions()
```

#### 3. 执行策略 (`core/execution/executors.py`)

实现 5 种专业执行算法:

| 策略 | 说明 | 用途 |
|------|------|------|
| **TWAP** | 时间加权平均价格 | 大单拆分，减少冲击 |
| **VWAP** | 成交量加权平均价格 | 跟踪市场成交分布 |
| **Grid** | 网格交易 | 震荡行情盈利 |
| **DCA** | 定投策略 | 分批建仓 |
| **Iceberg** | 冰山订单 | 隐藏大单意图 |

```python
from algvex.core.execution.executors import TWAPExecutor, GridExecutor

# TWAP 执行
executor = TWAPExecutor(connector, OrderRequest(...), duration=3600, slices=12)
result = await executor.execute()

# 网格交易
executor = GridExecutor(connector, symbol, total_amount=10000, 
                        lower_price=40000, upper_price=45000, grids=10)
result = await executor.execute()
```

#### 4. HummingbotBridge v2.0.0 重写

完全重写的执行桥接层:

- 多交易所支持
- 多执行策略支持
- 异步订单管理
- 自动重连机制
- 完整的状态同步

---

## 📋 v5.1.0 更新日志 (2025-12-23)

### 🆕 新增功能

#### 1. 跨截面处理器 (Qlib 原版适配)

| 处理器 | 说明 | 用法 |
|--------|------|------|
| `CSZScoreNorm` | 跨截面 Z-Score 标准化 | 每个时间点独立计算 z-score |
| `CSRankNorm` | 跨截面排名标准化 | 公式: (rank(pct=True) - 0.5) * 3.46 |
| `CSFillna` | 跨截面缺失值填充 | 用同一时间点的均值填充 |
| `TanhProcess` | Tanh 去噪处理 | 压缩极端值 |
| `ProcessInf` | 无穷值处理 | 替换 inf/-inf |
| `FilterCol` | 列过滤器 | 保留指定列 |
| `DropCol` | 列删除器 | 删除指定列 |

```python
from algvex.core.factor import CSZScoreNorm, CSRankNorm, TanhProcess

# 使用示例
processors = ProcessorChain([
    CSZScoreNorm(),      # 跨截面 z-score
    CSRankNorm(),        # 跨截面排名
    TanhProcess(),       # tanh 去噪
])
```

#### 2. 评估模块 (Qlib 原版)

| 函数 | 说明 |
|------|------|
| `risk_analysis(returns)` | 年化收益、夏普比率、最大回撤 |
| `calc_ic(pred, label)` | IC 和 Rank IC 计算 |
| `calc_long_short_return()` | 多空收益分析 |
| `calc_long_short_prec()` | 多空精度分析 |
| `generate_report()` | 综合评估报告 |

```python
from algvex.core import risk_analysis, calc_ic, generate_report

# 风险分析
metrics = risk_analysis(returns, freq='day')
print(f"夏普比率: {metrics['information_ratio']:.2f}")

# IC 分析
ic, rank_ic = calc_ic(predictions, labels)

# 综合报告
report = generate_report(predictions, labels, returns)
```

#### 3. Qlib 风格模型接口

| 模型 | 说明 |
|------|------|
| `LGBModel` | LightGBM，支持 fit/predict/finetune |
| `XGBModel` | XGBoost，带特征重要性 |
| `LinearModel` | OLS, NNLS, Ridge, Lasso |
| `get_model()` | 便捷工厂函数 |

```python
from algvex.core.model import LGBModel, LinearModel, get_model

# LightGBM 模型
model = LGBModel(num_leaves=64, learning_rate=0.05)
model.fit(dataset)
predictions = model.predict(dataset, segment='test')

# 线性模型
model = LinearModel(estimator='ridge', alpha=0.1)
model.fit(dataset)

# 微调
model.finetune(new_dataset, num_boost_round=10)
```

---

## 0. 硬约束层 (版本化配置+哈希审计)

> **核心原则**: 本章节定义的所有规则为**硬约束**，通过**版本化配置文件+哈希审计**实现可追溯性。
>
> **配置版本化机制**:
> - 每个配置文件都有 `config_version` 和 `config_hash`
> - 任何配置变更必须走 Git PR，自动计算新的 hash
> - 运行时校验 config_hash，发现不匹配立即报警
> - 历史运行可通过 trace 中的 config_hash 精确复现
>
> **硬约束 vs 实现**: 硬约束层定义"必须遵守的规则"，实现层定义"如何做到"。硬约束通过配置版本化保证一致性，实现可以渐进替换。

---

### 0.0 MVP Scope 定义

> **MVP 必须满足以下条件才能上线 Paper Trading**

#### 0.0.1 MVP 边界 (必须满足 A-D)

| 条件 | MVP 定义 | 验收标准 |
|------|----------|----------|
| **A) 单一时间框架** | 仅 5 分钟 Bar | 所有因子/信号/执行基于 5m Bar |
| **B) 有限标的** | 20-50 个永续合约 | 初始: BTCUSDT, ETHUSDT + Top-18 流动性 |
| **C) 每日Replay对齐** | Live vs Replay 偏差 < 阈值 | Daily job 自动比对并告警 |
| **D) 配置可追溯** | 所有配置版本化+哈希 | trace 记录 config_hash |

#### 0.0.2 MVP 数据源 (最小集)

```yaml
# MVP 仅使用以下3个数据源
mvp_data_sources:
  - source_id: klines_5m
    description: "5分钟K线 (OHLCV)"
    visibility: bar_close
    tier: A

  - source_id: open_interest_5m
    description: "5分钟持仓量快照"
    visibility: bar_close+5min
    tier: B

  - source_id: funding_8h
    description: "资金费率 (每8小时)"
    visibility: scheduled
    tier: A
```

#### 0.0.3 因子体系分层 (统一叙事)

> **因子层级说明**: 明确因子数量的层级关系。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          因子体系分层 (统一叙事)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【层级1: MVP生产因子】11个 ← 当前阶段只用这些                               │
│  ├── 动量族 (5个): return_5m, return_1h, ma_cross, breakout_20d,           │
│  │                 trend_strength                                          │
│  ├── 波动率族 (3个): atr_288, realized_vol_1d, vol_regime                  │
│  └── 订单流族 (3个): oi_change_rate, funding_momentum,                     │
│                      oi_funding_divergence                                 │
│                                                                             │
│  【层级2: 研究因子库】180个 ← 仅存在于 research/ 目录                        │
│  ├── 基于 klines + funding + oi 的扩展因子                                  │
│  ├── 用于因子研究、模型训练、策略探索                                        │
│  └── 验证通过后可晋升到 MVP                                                 │
│                                                                             │
│  【层级3: 扩展因子库】+21个 (P1扩展) ← Phase 2 考虑                         │
│  ├── L2深度因子 (8个): 需要深度数据                                         │
│  ├── 清算因子 (5个): 需要清算数据                                           │
│  └── 跨所Basis (8个): 需要多交易所数据                                      │
│                                                                             │
│  【读者指引】                                                               │
│  - Part A: 只关注 11个MVP因子                                               │
│  - Part B Section 4: 标注了哪些是MVP、哪些是研究侧                           │
│  - 实施顺序: MVP-11 → 验证稳定 → 逐步扩展                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**MVP因子详细定义**:

> **关键约束**:
> - K线数据窗口以 bar 数量计，5m 频率下 288 bars = 1 day
> - 派生因子窗口 (如 vol_regime 的 MA) 以样本数量计，需明确说明
> - OI 因子使用可见数据 (因 5min 延迟，OI[t-1] 是 signal_time=t 时的最新可见值)

| 因子ID | 因子族 | 计算公式 | 数据依赖 | 可见性 |
|--------|--------|----------|----------|--------|
| return_5m | 动量 | close[t] / close[t-1] - 1 | klines_5m | bar_close |
| return_1h | 动量 | close[t] / close[t-12] - 1 | klines_5m | bar_close |
| ma_cross | 动量 | MA(close, 5) / MA(close, 20) - 1 | klines_5m | bar_close |
| breakout_20d | 动量 | (close - rolling_max(high, 5760)) / atr_288 | klines_5m | bar_close |
| trend_strength | 动量 | ADX(14 bars) | klines_5m | bar_close |
| atr_288 | 波动率 | ATR(288 bars) = 1 day | klines_5m | bar_close |
| realized_vol_1d | 波动率 | std(return_5m, 288 bars) = 1 day | klines_5m | bar_close |
| vol_regime | 波动率 | realized_vol_1d / MA(realized_vol_1d, 30 days) | klines_5m | bar_close |
| oi_change_rate | 订单流 | (OI[t-1] - OI[t-2]) / OI[t-2] | open_interest_5m | bar_close + 5min |
| funding_momentum | 订单流 | MA_settlement(3) - MA_settlement(8) | funding_8h (settled) | settlement_time |
| oi_funding_divergence | 订单流 | sign(oi_change) != sign(funding) | aligned_asof join | max(oi_visible, funding_visible) |

> **v1.1.0 重要修正**:
> - `oi_change_rate`: 公式改为 `(OI[t-1] - OI[t-2]) / OI[t-2]`，因为 OI 有 5min 发布延迟
>   - 在 signal_time=t 时，OI[t] 的 visible_time = t+5min > t，不可用
>   - OI[t-1] 的 visible_time = t-5min+5min = t，刚好可用
>   - 因此计算变化率需用 OI[t-1] 和 OI[t-2]
> - `vol_regime`: 明确 MA(30 days) 是 30 天的日波动率平均值，需 30×288 = 8640 bars 数据

**因子可见性规则**:

```yaml
# ============================================================
# funding_momentum 窗口定义 (关键: 以结算次数计, 非 5m bars!)
# ============================================================
funding_momentum:
  # 窗口类型: settlement_events (不是 bars!)
  # MA(3) = 最近3次结算的平均值 (24小时)
  # MA(8) = 最近8次结算的平均值 (64小时)
  window_type: "settlement_events"
  fast_n: 3   # 3次结算 = 24h
  slow_n: 8   # 8次结算 = 64h

  # 禁止: 对 forward-fill 后的 5m 序列用 rolling(n_bars)
  # 错误示例: funding.rolling(3).mean() 在 5m 序列上 = 15min 而非 24h!
  prohibited_patterns:
    - "rolling(n_bars) on forward-filled 5m series"

  # 正确实现:
  implementation: |
    # 只取结算时刻的值 (00:00, 08:00, 16:00 UTC)
    settlements = funding_8h.loc[funding_8h.index.hour.isin([0, 8, 16])]
    ma_fast = settlements.rolling(3).mean()
    ma_slow = settlements.rolling(8).mean()
    funding_momentum = ma_fast - ma_slow
    # 然后 forward-fill 到 5m 时间轴

# ============================================================
# funding 数据的可见性关键说明
# ============================================================
funding_8h:
  # MVP只使用"已结算的 realized funding"，不使用预测 funding
  # 可见时间 = 8h 结算时刻 (00:00, 08:00, 16:00 UTC)
  semantic: "realized_funding_after_settlement"
  visible_after: "settlement_time"  # 00:00/08:00/16:00 UTC
  forward_fill_to_5m: true  # 填充到 5m 时间轴但窗口计算用原始序列

# ============================================================
# oi_funding_divergence 对齐规则
# ============================================================
oi_funding_divergence:
  # OI 可见延迟 5min，funding 是定时结算
  # 必须用 aligned_asof 合并，取 max(oi_visible_time, funding_visible_time)
  alignment: "asof_join"
  oi_delay: "bar_close + 5min"
  funding_delay: "settlement_time"
```

#### 0.0.4 Qlib 在 MVP 中的定位

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Qlib 定位: 研究工具，非生产核心                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【研究/回测阶段】✅ 使用 Qlib                                               │
│  - 因子计算与研究                                                           │
│  - 模型训练 (LightGBM / XGBoost)                                           │
│  - 历史回测                                                                 │
│  - Walk-forward 验证                                                       │
│                                                                             │
│  【MVP生产管道】❌ 不依赖 Qlib                                               │
│  - 原因: Qlib 是研究框架，不是实时交易框架                                    │
│  - 替代: 自建轻量级因子计算管道 (factors/core/)                              │
│  - 模型: 导出 Qlib 训练的模型权重，用独立推理代码加载                         │
│                                                                             │
│  【边界清晰化】                                                             │
│  - Qlib → 输出: model_weights.pkl, factor_definitions.yaml                 │
│  - 生产管道 → 输入: 上述文件 + 实时数据 → 输出: 交易信号                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 0.1 系统规约原则 (P1-P10) + 落地机制

> **原则落地**: 每条原则必须有具体的检查方式、责任人、违反后果。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AlgVex 系统规约原则 (10条)                            │
│                     违反任何一条将导致系统不可信/不可用                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【P1 可见性原则 No Lookahead】★★★ 最硬                                      │
│  任何数据/因子先问：在 signal_time 之前，是否真实可见？                         │
│  - 有发布延迟 → 必须显式记录 delay，回测做 shift                              │
│  - 会后补/修订 → 必须引入 revision_id，规定回测用哪个版本                      │
│  - 口径不稳定 → 禁用或仅作风控提示，不进主信号                                 │
│                                                                             │
│  【P2 单一真相源 Single Source of Truth】                                    │
│  所有上层计算必须可追溯到同一套 L0 事实源 + 同一条 snapshot_cutoff             │
│  - Redis/缓存只是性能层，不能成为"第二事实源"                                  │
│  - 任何因子必须能复算，不能只存在于内存/临时缓存                                │
│                                                                             │
│  【P3 增量证明 Marginal Utility First】                                      │
│  每新增数据/因子必须写清楚增量属于哪一类：                                      │
│  - A) 提升收益质量 (Sharpe/回撤/稳定性)                                       │
│  - B) 提升风险控制 (尾部损失/爆仓概率下降)                                     │
│  - C) 提升执行一致性 (回测-实盘偏差下降)                                       │
│  说不清属于哪一类 → 99% 是噪声税 (noise tax)，禁止入库                         │
│                                                                             │
│  【P4 成本预算 Budgeted Data】                                               │
│  每个数据源必须给预算：带宽/消息量/CPU/存储/维护成本                            │
│  预算超标 → 必须自动降级：降频、缩 universe、只采关键字段                       │
│  否则极端行情系统会因为"数据太全"而崩                                          │
│                                                                             │
│  【P5 稳定优先 Stability > Novelty】                                         │
│  优先采集"口径稳定、可落盘、可复现"的数据                                      │
│  新奇但不稳定的数据：先做观测与风控，不进核心模型                               │
│  等稳定+落盘完整再进入主因子                                                  │
│                                                                             │
│  【P6 可复现与可追责 Reproducibility & Audit】                               │
│  任何一次信号、任何一笔交易，必须能回答：                                       │
│  - 当时看到的数据快照是什么？                                                 │
│  - 因子值怎么来的？                                                          │
│  - 订单怎么下的？成交怎么回来的？                                              │
│  - PnL里每一分钱怎么扣的？(手续费/滑点/资金费率/冲击)                          │
│                                                                             │
│  【P7 去共线与去重复 Redundancy Control】                                    │
│  冗余因子 → 拟合更强、泛化更差、权重不稳定、信号抖动                            │
│  必须建立"因子族"：每一族只保留少数代表，或做正交化/降维                        │
│                                                                             │
│  【P8 稳健性优先 Robustness Across Regimes】                                 │
│  因子必须在不同市场状态下可解释：趋势/震荡、高/低波动、流动性充足/枯竭          │
│  只能在一个 regime 下漂亮 → 实盘变成"踩雷检测器"                               │
│                                                                             │
│  【P9 延迟与采样一致 Latency & Sampling Consistency】                        │
│  明确系统是 bar-based / event-based / hybrid                                │
│  OI/CVD/深度数据如何对齐到 bar_close 必须有唯一规则                           │
│  对齐不一致 → "数据越多，误差越大"                                            │
│                                                                             │
│  【P10 先对齐后增强 Alignment First】                                        │
│  优先把"回测=实盘"做到最接近，再做更复杂的因子与模型                            │
│  否则永远不知道收益变化来自：因子变强 还是 对齐/执行/扣费方式变了               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 0.1.1 原则落地机制 (每条原则如何检查)

| 原则 | 检查方式 | 检查时机 | 违反后果 |
|------|----------|----------|----------|
| **P1 可见性** | T1_VisibilityTest 自动测试 | PR门禁 + 每日Replay | PR拒绝 / 告警 |
| **P2 单一真相源** | DataService接口强制 + 导入扫描 | PR门禁 | PR拒绝 |
| **P3 增量证明** | 因子准入审查表 (见下方模板) | 因子入库前 | 不填表不入库 |
| **P4 成本预算** | budget.yaml 预算定义 + 运行时监控 | 启动时 + 运行时 | 自动降级 |
| **P5 稳定优先** | 数据源tier分级 (A/B/C) | 数据源接入时 | B/C档不进MVP |
| **P6 可复现** | Trace Schema 强制记录 | 每条信号 | 无trace的信号丢弃 |
| **P7 去冗余** | 因子相关性检查脚本 | 因子入库前 | 相关性>0.8拒绝 |
| **P8 稳健性** | Walk-forward + 分regime回测 | 因子入库前 | 单regime因子降级 |
| **P9 采样一致** | visibility.yaml 统一规则 | 配置校验 | 配置不一致拒启动 |
| **P10 先对齐** | Daily Replay差异 < 阈值 | 每日 | 差异大暂停交易 |

#### 0.1.2 P3增量证明: 因子/数据源准入审查表

```yaml
# 每新增因子或数据源必须填写此表，否则禁止入库
factor_admission_form:
  # 基本信息
  factor_id: "new_factor_xxx"
  proposer: "developer_name"
  date: "2025-12-22"

  # P3 增量证明 (必填，三选一)
  marginal_utility:
    category: "A"  # A=收益提升 / B=风险控制 / C=执行一致性
    evidence: |
      在6个月回测中，加入此因子后:
      - Sharpe从1.2提升到1.35 (+12.5%)
      - 最大回撤从15%降到12%
    backtest_report_link: "reports/factor_xxx_backtest.html"

  # P5 稳定性评估 (必填)
  stability:
    data_source_tier: "A"  # A/B/C
    history_available_days: 365
    schema_change_risk: "low"  # low/medium/high

  # P7 冗余检查 (必填)
  redundancy:
    max_correlation_with_existing: 0.65  # 必须 < 0.7
    most_correlated_factor: "existing_factor_yyy"

  # P8 稳健性 (必填)
  robustness:
    regimes_tested: ["trending", "ranging", "high_volatility"]
    positive_in_all_regimes: true  # 必须为true

  # 审批
  approved_by: ""  # 审批人签名
  approved_date: ""
```

#### 0.1.3 原则违反自动告警

```python
# algvex/core/principle_monitor.py
class PrincipleMonitor:
    """监控原则违反情况，自动告警"""

    def check_p1_visibility(self, signal: Signal, snapshot_cutoff: datetime) -> bool:
        """
        P1: 检查信号是否使用了未来数据

        关键修正: 判断基准必须是 visible_time (不是 data_time!)
        规则: visible_time <= snapshot_cutoff 才能使用
        """
        for factor_value in signal.factors_used:
            # 关键: 使用 visible_time 而非 data_time
            # visible_time = event_time + publication_delay (由 visibility.yaml 定义)
            if factor_value.visible_time > snapshot_cutoff:
                self.alert(
                    principle="P1",
                    severity="critical",
                    message=f"Future data detected: {factor_value.factor_id} "
                            f"visible_time={factor_value.visible_time} > "
                            f"snapshot_cutoff={snapshot_cutoff}"
                )
                return False
        return True

    def check_p4_budget(self, metrics: SystemMetrics) -> bool:
        """P4: 检查资源使用是否超预算"""
        if metrics.api_calls_per_min > self.budget.max_api_calls_per_min * 0.8:
            self.alert(
                principle="P4",
                severity="warning",
                message=f"API usage at {metrics.api_calls_per_min}, approaching limit"
            )
            self.trigger_degrade()
            return False
        return True

    def check_p10_alignment(self, live_signal: Signal, replay_signal: Signal) -> bool:
        """P10: 检查回测-实盘对齐"""
        diff = abs(live_signal.value - replay_signal.value)
        if diff > self.alignment_threshold:
            self.alert(
                principle="P10",
                severity="critical",
                message=f"Alignment failed: live={live_signal.value}, "
                        f"replay={replay_signal.value}, diff={diff}"
            )
            return False
        return True

    def alert(self, principle: str, severity: str, message: str):
        """发送告警"""
        log.error(f"[{principle}][{severity}] {message}")
        if severity == "critical":
            # 发送紧急通知 (Slack/邮件/短信)
            notify_oncall(principle, message)
```

---

### 0.2 S1: 时间+快照契约 (Time & Snapshot Contract)

> **版本化配置**: 本节定义的时间语义和可见性规则通过 `visibility.yaml` 配置文件管理，任何变更需走 Git PR。

#### 0.2.1 时间字段定义 (全系统统一)

| 字段名 | 定义 | 时区 | 用途 |
|--------|------|------|------|
| `event_time` | 事件发生的真实时间 | UTC | 原始数据时间戳 |
| `collected_at` | 系统采集到数据的时间 | UTC | 延迟监控、审计 |
| `bar_open_time` | K线开始时间 | UTC | Bar 标识 |
| `bar_close_time` | K线结束时间 | UTC | 信号生成基准时间 |
| `signal_time` | 信号生成时间 = `bar_close_time` | UTC | 策略决策时间 |
| `snapshot_cutoff` | 快照截止时间 | UTC | 可用数据边界 |

#### 0.2.2 可见性规则 (visibility.yaml)

```yaml
# ============================================================
# 文件: config/visibility.yaml
# 说明: 可见性规则配置 (版本化，任何变更需走 Git PR)
# ============================================================
config_version: "1.1.0"
config_hash: "sha256:abc123..."  # 自动计算，CI检查一致性

# ============================================================
# 唯一判定公式 (全系统统一，不允许其他模块自定义)
# ============================================================
#
# 任何数据/字段/因子可用 当且仅当: visible_time <= snapshot_cutoff
#
# 定义:
#   signal_time = bar_close_time (MVP固定)
#   snapshot_cutoff = signal_time (对于 bar_close 类型数据)
#   visible_time = 按数据类型计算 (见下方 visibility_types)
#
# 重要说明 (v1.1.0 修正):
#   - bar_close 数据: visible_time = bar_close_time = signal_time
#     因此 snapshot_cutoff 必须 >= signal_time，即 safety_margin = 0
#   - bar_close_delayed 数据 (如 OI): visible_time = bar_close_time + delay
#     在 signal_time=t 时，OI[t] 的 visible_time = t + 5min > t，不可用
#     OI[t-1] 的 visible_time = (t-5min) + 5min = t，刚好可用 (<=)
#   - funding 只能用最近一次已结算的值
#
# ============================================================
snapshot_cutoff_rule: "signal_time - safety_margin"
safety_margin: "0s"  # v1.1.0修正: 改为0s，避免 bar_close 数据不可见

visibility_types:
  realtime:
    description: "实时数据，几乎无延迟"
    rule: "event_time + ${latency_buffer}"
    latency_buffer: "1s"
    examples:
      - mark_price
      - last_price
      - best_bid
      - best_ask

  bar_close:
    description: "Bar聚合数据，bar收盘后完整可见"
    rule: "bar_close_time + 0s"
    examples:
      - ohlcv
      - taker_volume
      - cvd_5m
      - depth_aggregated

  bar_close_delayed:
    description: "Bar聚合+延迟数据"
    rule: "bar_close_time + ${publication_delay}"
    publication_delay: "5min"  # 可按数据源覆盖
    examples:
      - oi_change
      - long_short_ratio

  scheduled:
    description: "定时发布数据"
    rule: "scheduled_time + ${publication_delay}"
    publication_delay: "0s"
    examples:
      - funding_rate
      - fear_greed_index
      - macro_data

# 数据源 -> 可见性类型 映射
source_visibility_map:
  klines_5m: bar_close
  open_interest_5m: bar_close_delayed
  funding_8h: scheduled
  mark_price: realtime
  liquidations: bar_close_delayed

# 安全边际配置
safety_margins:
  default: "0s"
  conservative: "5s"
  # 注意: 生产环境禁止使用负值! 负值仅允许在研究环境中使用
  # aggressive: "-1s" # 已禁用 - 任何负边际都会导致 lookahead 风险
```

#### 0.2.3 配置哈希验证

```python
# 运行时校验 visibility.yaml 的 config_hash
def validate_visibility_config():
    """启动时校验配置文件哈希，防止未经审批的修改"""
    config = load_yaml("config/visibility.yaml")
    expected_hash = config["config_hash"]
    actual_hash = compute_hash(config, exclude=["config_hash"])

    if expected_hash != actual_hash:
        raise ConfigIntegrityError(
            f"visibility.yaml 已被修改但未更新 config_hash! "
            f"expected={expected_hash}, actual={actual_hash}"
        )
```

#### 0.2.3 快照契约 (Snapshot Contract)

```python
@dataclass(frozen=True)  # frozen=True 表示不可变
class SnapshotContract:
    """快照契约 - 定义数据快照的不可变规则"""

    # 快照ID生成规则: snapshot_id = f"snap_{cutoff_time}_{content_hash[:16]}"

    # 快照必须包含的元数据
    REQUIRED_METADATA = [
        "snapshot_id",           # 唯一标识
        "cutoff_time",           # 截止时间
        "symbols",               # 标的列表
        "data_sources",          # 数据源及其版本
        "visibility_config",     # 可见性配置的hash
        "content_hash",          # 数据内容hash
        "created_at",            # 创建时间
    ]

    # 快照不可变性规则
    # 1. 快照一旦创建，内容永不修改
    # 2. 同一 cutoff_time 可能有多个快照，用 snapshot_id 区分
    # 3. 回测/训练必须指定 snapshot_id，不能用 "latest"
    # 4. 快照文件必须有校验和，加载时验证
```

---

### 0.3 S2: 数据契约模板 (Data Contract Template)

> **版本化配置**: 每个数据源的契约存储在 `config/data_contracts/{source_id}.yaml`，变更需走 Git PR。

#### 0.3.1 数据契约模板

```yaml
# ============================================================
# 文件: config/data_contracts/klines_5m.yaml
# 说明: 数据源契约 (版本化，任何变更需走 Git PR)
# ============================================================
config_version: "1.0.0"
config_hash: "sha256:def456..."  # 自动计算，CI检查一致性

data_contract:
  source_id: "klines_5m"
  source_name: "币安永续合约5分钟K线"
  exchange: "binance"
  instrument_type: "perpetual"

  # 字段定义
  schema:
    primary_key: ["symbol", "bar_time"]
    time_field: "bar_time"
    time_zone: "UTC"
    fields:
      - name: open
        type: float
        nullable: false
      - name: high
        type: float
        nullable: false
      - name: low
        type: float
        nullable: false
      - name: close
        type: float
        nullable: false
      - name: volume
        type: float
        nullable: false
      - name: quote_volume
        type: float
        nullable: false
      - name: taker_buy_volume
        type: float
        nullable: false

  # 可见性 (引用 visibility.yaml 中定义的类型)
  visibility:
    type: "bar_close"  # 引用 visibility.yaml
    publication_delay: "0s"
    revision_policy: "no_revision"

  # 可得性分级
  availability:
    tier: "A"
    history_window: "unlimited"
    backfill_support: true
    schema_stability: "high"
    free_tier: true  # 免费数据

  # 数据稳定性评估
  stability_assessment:
    api_change_frequency: "low"  # low / medium / high
    last_schema_change: "2023-01-01"
    deprecation_risk: "low"
    backup_sources: []  # 无备用数据源

  # 预算与降级
  budget:
    max_symbols: 50
    max_frequency: "5m"
    max_api_calls_per_min: 100

  degrade_policy:
    - trigger: "api_calls > 80%"
      action: "reduce_symbols to top-20"
    - trigger: "latency > 10s"
      action: "skip_non_critical_symbols"

  # 质量验收
  acceptance:
    tests:
      - "completeness > 99%"
      - "latency_p99 < 5s"
      - "no_duplicate_bars"
    on_failure: "block_ingestion"
```

#### 0.3.2 MVP 数据源契约状态

| 数据源 | 契约文件 | 状态 | 可见性 | 可得性 | 免费 |
|--------|----------|------|--------|--------|------|
| klines_5m | `data_contracts/klines_5m.yaml` | ✅ 已定义 | bar_close | A档 | ✅ |
| open_interest_5m | `data_contracts/open_interest_5m.yaml` | ✅ 已定义 | bar_close+5min | B档 | ✅ |
| funding_8h | `data_contracts/funding_8h.yaml` | ✅ 已定义 | scheduled+0s | A档 | ✅ |

**OI 数据契约 (强制 asof_join)**:

```yaml
# config/data_contracts/open_interest_5m.yaml
source_id: "open_interest_5m"
config_version: "1.0.0"

visibility:
  type: "bar_close_delayed"
  publication_delay: "5min"
  # visible_time = bar_close_time + 5min

# ============================================================
# 关键约束: 禁止直接取同 bar 的 OI (会导致 lookahead)
# ============================================================
alignment:
  method: "asof_join_on_visible_time"
  # 必须通过 DataService.asof_get() 获取 last_visible 值
  # 在 signal_time=bar_close_time 时, OI[t] 不可用, 只能用 OI[t-1]
  max_staleness: "15min"  # 超过则标记缺失/降级
  fallback: "last_valid"  # 缺失时用最近有效值

# 生产代码禁止:
prohibited_access_patterns:
  - "直接 JOIN ON bar_time (会产生 lookahead)"
  - "不经 visible_time 检查直接取值"
```

**OI 数据访问强制规则**:

```python
# algvex/shared/data_service.py
class DataService(ABC):
    @abstractmethod
    def asof_get(
        self,
        source_id: str,
        cutoff_time: datetime,
        symbol: str
    ) -> Optional[DataPoint]:
        """
        获取 cutoff_time 之前最近的可见数据点

        关键: 这是唯一允许的 OI 数据访问方式!
        禁止: 直接按 bar_time 等值 JOIN
        """
        pass

# 使用示例
def get_oi_for_signal(signal_time: datetime, symbol: str) -> float:
    """正确的 OI 获取方式"""
    # v1.1.0修正: snapshot_cutoff = signal_time (safety_margin = 0s)
    snapshot_cutoff = signal_time

    # 必须用 asof_get, 不能直接取 OI[bar_time=signal_time]
    # 在 signal_time=t 时:
    #   - OI[t] 的 visible_time = t + 5min > t，不可用
    #   - OI[t-1] 的 visible_time = (t-5min) + 5min = t，刚好可用
    oi_point = data_service.asof_get(
        source_id="open_interest_5m",
        cutoff_time=snapshot_cutoff,
        symbol=symbol
    )

    if oi_point is None:
        raise DataNotAvailableError(f"No visible OI for {symbol} at {snapshot_cutoff}")

    # 验证: oi_point.visible_time <= snapshot_cutoff
    assert oi_point.visible_time <= snapshot_cutoff
    return oi_point.value
```

#### 0.3.3 契约变更审计

```python
# 每次启动时校验所有数据契约
def validate_all_data_contracts():
    """校验所有数据契约的 config_hash"""
    contracts_dir = Path("config/data_contracts")
    for yaml_file in contracts_dir.glob("*.yaml"):
        config = load_yaml(yaml_file)
        validate_config_hash(config, yaml_file.name)

    # 记录当前使用的契约版本到 trace
    return {
        f.stem: config["config_hash"]
        for f in contracts_dir.glob("*.yaml")
        for config in [load_yaml(f)]
    }
```

---

### 0.4 S3: 预算与降级策略 (Budget & Degrade Policy)

> **规则**: 系统必须有资源预算和自动降级机制。

#### 0.4.1 全局预算

```yaml
global_budget:
  network:
    max_websocket_connections: 50
    max_rest_calls_per_second: 100
  compute:
    max_factor_compute_time_per_bar: "30s"
  universe:
    default_symbols: ["BTCUSDT", "ETHUSDT"]
    max_symbols_normal: 20
    max_symbols_degraded: 5
```

#### 0.4.2 降级触发条件

| Level | 触发条件 | 动作 | 告警 |
|-------|----------|------|------|
| L1 | api_usage > 70% | 缩减 universe (TOP-20 → TOP-10) | warning |
| L2 | api_usage > 90% | 仅保留 BTC/ETH + 关闭非核心因子 | critical |
| L3 | latency > 30s | 仅 BTC/ETH + 仅开仓/加仓暂停 | emergency |
| L4 | 数据丢失/交易所错误 | 保护模式 (仅平仓/不开新仓) | emergency |

> **注意**: MVP 已固定 5m 频率，降级不能改变 bar 频率。降级只能在以下维度操作:
> - Universe 范围 (标的数量)
> - 因子集合 (关闭非核心因子)
> - 交易动作 (暂停开仓/仅平仓)
> - 下游推送频率 (UI 刷新频率，非信号频率)

---

### 0.5 S4: 因子治理 (Factor Governance)

> **版本化配置**: 因子准入门槛存储在 `config/factor_governance.yaml`，阈值可按市场状态动态调整。

#### 0.5.1 因子准入门槛 (动态阈值)

```yaml
# ============================================================
# 文件: config/factor_governance.yaml
# 说明: 使用动态基准替代固定阈值
# ============================================================
config_version: "1.0.0"
config_hash: "sha256:ghi789..."

factor_admission:
  # 基础门槛 (必须满足)
  basic_requirements:
    min_history_days: 180          # 至少6个月
    max_missing_rate: 0.05         # <5%
    max_correlation_with_existing: 0.7  # 防止冗余

  # IC 门槛 (动态基准)
  ic_thresholds:
    # 问题: 固定 IC>2% 对加密货币高波动市场不适用
    # 解决: 使用相对于基准的动态阈值
    method: "relative_to_baseline"
    baseline: "rolling_mean_ic_30d"  # 过去30天所有因子IC均值
    min_ic_above_baseline: 0.005    # IC需高于基准0.5%
    min_ic_ir: 0.3                  # IC_IR (IC/std(IC)) 最低要求

    # 备选: 按市场状态分档
    regime_specific:
      trending:
        min_ic: 0.03
      ranging:
        min_ic: 0.015
      high_volatility:
        min_ic: 0.01  # 高波动期间IC自然较低

  # 稳健性要求
  robustness:
    min_regimes_positive: 3        # 至少在3种市场状态下IC为正
    max_single_regime_contribution: 0.6  # 单一状态贡献不超过60%
```

#### 0.5.2 因子族与去冗余

```
MVP-11 因子族定义 (与 Section 0.0.3 一致):
├── momentum族 (5个): return_5m, return_1h, ma_cross, breakout_20d, trend_strength
├── volatility族 (3个): atr_288, realized_vol_1d, vol_regime
├── orderflow族 (3个): oi_change_rate, funding_momentum, oi_funding_divergence
└── 总计: 11个生产因子 (MVP Gate 强制白名单)

去冗余方法:
- 相关性 > 0.8 → 只保留 IC_IR 最高的
- 每族最多保留5个代表因子
- 新增因子必须证明与现有因子的增量价值 (P3原则)
```

#### 0.5.3 因子退出条件

| 退出条件 | 动作 |
|----------|------|
| IC连续3个月低于基准1% | 降权50% → 观察 → 退出 |
| IC_IR < 0.2 持续2个月 | 稳定性检查 |
| 单一regime贡献 > 70% | 降为风控因子 |
| 数据源不稳定/停服 | 立即暂停使用 |

---

### 0.6 S5: 对齐与归因 + Daily Replay (Alignment & Attribution)

> **核心目标**: 增加可验证的每日Replay闭环对齐，确保回测与实盘行为一致。

#### 0.6.1 Trace Schema (完整追踪)

> **重要**: Trace 落盘必须使用确定性 JSON 序列化，确保 data_hash 一致性

```yaml
# ============================================================
# 文件: config/alignment.yaml
# 说明: 对齐与追踪配置 (版本化)
# ============================================================
config_version: "1.0.0"
config_hash: "sha256:jkl012..."

trace_schema:
  # 每条信号/交易必须记录的追踪信息
  required_fields:
    - trace_id           # 唯一追踪ID
    - run_mode           # "live" | "replay" | "backtest"
    - timestamp          # 执行时间
    - contract_hash      # visibility.yaml + data_contracts/*.yaml 的联合hash
    - config_hash        # 所有配置文件的联合hash
    - code_hash          # 代码版本 (git commit hash)
    - data_hash          # 输入数据的hash

  # 信号追踪
  signal_trace:
    - signal_id
    - snapshot_id
    - factors_used       # 使用的因子及其值
    - model_version
    - raw_prediction
    - final_signal

  # 执行追踪
  execution_trace:
    - order_intent
    - fill_reports
    - slippage_actual
    - commission_actual

  # 序列化规范
  serialization:
    format: "json"
    sort_keys: true
    separators: [",", ":"]   # 无空格，紧凑格式
    ensure_ascii: false
```

#### 0.6.1.1 确定性 Trace 序列化器

```python
# algvex/core/trace_serializer.py
"""
确定性 Trace 序列化

确保 JSON 序列化稳定: json.dumps(sort_keys=True, separators=(',', ':'))
"""
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict
import hashlib

class DeterministicTraceSerializer:
    """
    确定性 Trace 序列化器

    保证:
    1. 相同数据 → 相同 JSON 字符串
    2. 相同 JSON 字符串 → 相同 hash
    """

    def serialize(self, trace: Dict[str, Any]) -> str:
        """
        序列化 trace 为确定性 JSON 字符串

        Args:
            trace: trace 字典

        Returns:
            确定性 JSON 字符串
        """
        # 1. 递归规范化所有值
        normalized = self._normalize(trace)

        # 2. 确定性序列化
        # 关键: sort_keys=True + separators 无空格
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
        )

    def compute_hash(self, trace: Dict[str, Any]) -> str:
        """计算 trace 的确定性 hash"""
        serialized = self.serialize(trace)
        hash_value = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        return f"sha256:{hash_value[:16]}"

    def _normalize(self, obj: Any) -> Any:
        """递归规范化值"""
        if isinstance(obj, dict):
            # 递归处理，key 排序
            return {k: self._normalize(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, list):
            return [self._normalize(item) for item in obj]
        elif isinstance(obj, set):
            # set → sorted list
            return sorted([self._normalize(item) for item in obj])
        elif isinstance(obj, datetime):
            # datetime → ISO8601 UTC 字符串
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        elif isinstance(obj, Decimal):
            # Decimal → 字符串 (保留精度)
            return str(obj.normalize())
        elif isinstance(obj, float):
            # float → 8位精度字符串
            return f"{obj:.8f}".rstrip('0').rstrip('.')
        elif obj is None:
            return None
        else:
            return obj

    def deserialize(self, json_str: str) -> Dict[str, Any]:
        """反序列化 JSON 字符串"""
        return json.loads(json_str)


# ============== Trace Writer ==============

class TraceWriter:
    """
    确定性 Trace 写入器

    用于写入 live_output_{date}.jsonl 和 replay_output_{date}.jsonl
    """

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.serializer = DeterministicTraceSerializer()

    def write(self, trace: Dict[str, Any]):
        """写入一条 trace (JSONL 格式)"""
        # 使用确定性序列化
        line = self.serializer.serialize(trace)

        with open(self.output_path, 'a') as f:
            f.write(line + '\n')

    def compute_file_hash(self) -> str:
        """计算整个文件的 hash (用于验证)"""
        with open(self.output_path, 'rb') as f:
            content = f.read()
        hash_value = hashlib.sha256(content).hexdigest()
        return f"sha256:{hash_value[:16]}"


# ============== 使用示例 ==============

def create_signal_trace(
    signal_id: str,
    factors_used: Dict[str, float],
    **kwargs
) -> Dict[str, Any]:
    """创建信号 trace"""
    serializer = DeterministicTraceSerializer()

    trace = {
        "trace_id": signal_id,
        "timestamp": datetime.now(timezone.utc),  # 强制 UTC
        "factors_used": factors_used,
        **kwargs,
    }

    # 计算 data_hash (用于后续验证)
    trace["data_hash"] = serializer.compute_hash({"factors": factors_used})

    return trace
```

#### 0.6.1.2 Trace 对比验证

```python
# scripts/verify_trace_determinism.py
"""验证 trace 序列化确定性"""

from algvex.core.trace_serializer import DeterministicTraceSerializer
from datetime import datetime, timezone
from decimal import Decimal

def test_trace_determinism():
    """测试相同数据产生相同序列化结果"""
    serializer = DeterministicTraceSerializer()

    # 测试用例: 不同顺序的 dict
    trace1 = {
        "b": 2,
        "a": 1,
        "factors": {"x": 0.1, "y": 0.2},
        "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    }

    trace2 = {
        "a": 1,
        "factors": {"y": 0.2, "x": 0.1},
        "b": 2,
        "timestamp": datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    }

    s1 = serializer.serialize(trace1)
    s2 = serializer.serialize(trace2)

    assert s1 == s2, f"失败: 相同数据产生不同序列化\n{s1}\n{s2}"

    h1 = serializer.compute_hash(trace1)
    h2 = serializer.compute_hash(trace2)

    assert h1 == h2, f"失败: 相同数据产生不同 hash\n{h1}\n{h2}"

    print("✅ 验证通过: trace 序列化确定性正常")

if __name__ == "__main__":
    test_trace_determinism()
```

#### 0.6.2 Daily Replay Alignment (每日闭环验证)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Daily Replay Alignment 闭环验证                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【工作流程】                                                               │
│                                                                             │
│  1. Live运行 (T日)                                                         │
│     ├─ 记录: signal_trace + execution_trace                                │
│     └─ 存储: live_output_{date}.jsonl                                      │
│                                                                             │
│  2. Replay运行 (T+1日凌晨)                                                  │
│     ├─ 输入: T日的 snapshot_id (固定数据)                                   │
│     ├─ 输入: T日的 config_hash (固定配置)                                   │
│     ├─ 输入: T日的 code_hash (固定代码)                                     │
│     └─ 输出: replay_output_{date}.jsonl                                    │
│                                                                             │
│  3. 对比验证 (自动化)                                                       │
│     ├─ 比对: live vs replay 的 signal_trace                                │
│     ├─ 差异 > 阈值 → 立即告警                                               │
│     └─ 记录: alignment_report_{date}.json                                  │
│                                                                             │
│  【关键约束】                                                               │
│  - Replay 必须使用与 Live 相同的 snapshot_id                                 │
│  - Replay 必须使用与 Live 相同的 config_hash                                 │
│  - 任何差异都说明系统存在不确定性 (需要调查)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 0.6.3 Trace Schema 规范

> **关键约束**: JSONL 每行是一个完整的 trace，顶层包含所有必需字段，不要嵌套 `{"trace": {...}}`

```python
# Trace 必需字段 (JSONL 每行顶层)
REQUIRED_TRACE_FIELDS = [
    "signal_id",        # 唯一标识: f"{symbol}|{bar_close_time_iso}|{strategy_id}"
    "trace_id",         # 可等于 signal_id
    "run_mode",         # "live" 或 "replay"
    "timestamp",        # ISO8601 UTC
    "symbol",           # 交易对
    "bar_close_time",   # ISO8601 UTC
    "raw_prediction",   # 模型原始输出
    "final_signal",     # 最终信号 (-1 到 1)
    "factors_used",     # Dict[str, float]
    "data_hash",        # 输入数据哈希
    "config_hash",      # 配置哈希
    "snapshot_id",      # 快照ID (可选，但 replay 必需)
]

# signal_id 生成规则
def make_signal_id(symbol: str, bar_close_time: datetime, strategy_id: str) -> str:
    return f"{symbol}|{bar_close_time.isoformat()}|{strategy_id}"
```

#### 0.6.4 Replay 对齐脚本

```python
# scripts/daily_replay_alignment.py
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class AlignmentReport:
    """对齐报告结构"""
    date: str
    total_live: int
    total_replay: int
    matched: int
    missing_in_replay: List[str]      # signal_ids 在 live 有但 replay 没有
    missing_in_live: List[str]        # signal_ids 在 replay 有但 live 没有
    mismatched: List[Dict[str, Any]]  # 匹配但字段不一致
    max_signal_diff: float
    config_hash: str
    snapshot_id: str

def run_daily_alignment(date: str) -> AlignmentReport:
    """每日 Replay 对齐验证"""

    # 1. 加载 Live 记录 (JSONL 每行是顶层 trace)
    live_traces = load_jsonl(f"logs/live_output_{date}.jsonl")

    # 2. 按 signal_id 建立索引 (去重: 保留最后一条)
    live_by_id = {}
    for trace in live_traces:
        signal_id = trace["signal_id"]
        live_by_id[signal_id] = trace  # 后出现的覆盖

    # 3. 获取配置 (取第一条的配置)
    first_trace = live_traces[0]
    snapshot_id = first_trace.get("snapshot_id", "")
    config_hash = first_trace["config_hash"]
    code_hash = first_trace.get("code_hash", "")

    # 4. Replay 运行
    replay_traces = run_replay(
        date=date,
        snapshot_id=snapshot_id,
        config_hash=config_hash,
        code_hash=code_hash,
    )
    replay_by_id = {t["signal_id"]: t for t in replay_traces}

    # 5. 按 signal_id 做 join
    all_signal_ids = set(live_by_id.keys()) | set(replay_by_id.keys())

    missing_in_replay = []
    missing_in_live = []
    mismatched = []
    max_diff = 0.0

    for sid in sorted(all_signal_ids):
        live_t = live_by_id.get(sid)
        replay_t = replay_by_id.get(sid)

        if live_t and not replay_t:
            missing_in_replay.append(sid)
        elif replay_t and not live_t:
            missing_in_live.append(sid)
        else:
            # 比较关键字段
            diff = compare_trace_fields(live_t, replay_t)
            if diff["signal_diff"] > SIGNAL_DIFF_THRESHOLD:
                mismatched.append({
                    "signal_id": sid,
                    "live_signal": live_t["final_signal"],
                    "replay_signal": replay_t["final_signal"],
                    "diff": diff,
                })
            max_diff = max(max_diff, diff["signal_diff"])

    report = AlignmentReport(
        date=date,
        total_live=len(live_by_id),
        total_replay=len(replay_by_id),
        matched=len(all_signal_ids) - len(missing_in_replay) - len(missing_in_live),
        missing_in_replay=missing_in_replay,
        missing_in_live=missing_in_live,
        mismatched=mismatched,
        max_signal_diff=max_diff,
        config_hash=config_hash,
        snapshot_id=snapshot_id,
    )

    # 6. 告警
    if max_diff > SIGNAL_DIFF_THRESHOLD or missing_in_replay or missing_in_live:
        send_alert(f"Replay alignment issues for {date}: {report}")

    return report
```

#### 0.6.5 对齐验收标准 (分层)

> **关键修正**: 浮点运算/不同 BLAS 实现会产生微小差异，不应报警。
> 采用分层验收标准：硬指标必须 100% 一致，软指标允许容差。

**分层验收标准**:

| 层级 | 对齐项 | 要求 | 容差 | 说明 |
|------|--------|------|------|------|
| **L1 (硬)** | data_hash | 100% 一致 | 0 | 输入数据必须完全相同 |
| **L2 (硬)** | features_hash | 100% 一致 | 0 | 特征向量必须完全相同 |
| **L3 (软)** | raw_prediction | 允许微小差异 | atol=1e-8, rtol=1e-6 | 模型推理浮点误差 |
| **L4 (硬)** | final_signal | 100% 一致 | 0 | 离散化后的最终决策 |
| **L5 (软)** | execution_trace | 允许差异 | 10 bps | 执行滑点/时序差异 |

```python
# 对齐验证逻辑
def verify_alignment(live: Trace, replay: Trace) -> AlignmentResult:
    """分层验证 Live vs Replay 对齐"""

    # L1: data_hash 必须 100% 一致 (硬)
    if live.data_hash != replay.data_hash:
        return AlignmentResult(passed=False, level="L1",
                               reason=f"data_hash mismatch: {live.data_hash} vs {replay.data_hash}")

    # L2: features_hash 必须 100% 一致 (硬)
    if live.features_hash != replay.features_hash:
        return AlignmentResult(passed=False, level="L2",
                               reason=f"features_hash mismatch")

    # L3: raw_prediction 允许微小浮点差异 (软)
    # 原因: 不同 BLAS 实现、CPU/GPU 差异可能产生 1e-12 级误差
    if not np.allclose(live.raw_prediction, replay.raw_prediction,
                       atol=1e-8, rtol=1e-6):
        # 记录差异但不一定 fail
        diff = abs(live.raw_prediction - replay.raw_prediction)
        if diff > 1e-4:  # 超过万分之一才 fail
            return AlignmentResult(passed=False, level="L3",
                                   reason=f"raw_prediction diff={diff}")

    # L4: final_signal (离散化后) 必须 100% 一致 (硬)
    if live.final_signal != replay.final_signal:
        return AlignmentResult(passed=False, level="L4",
                               reason=f"final_signal mismatch: {live.final_signal} vs {replay.final_signal}")

    return AlignmentResult(passed=True, level="all", reason="OK")
```

**执行层差异容差**:

| 对齐项 | 最大偏差 | 告警级别 | 说明 |
|--------|----------|----------|------|
| 成交价差 | 10 bps | warning | 市场流动性/时序差异 |
| 手续费差 | 5% | warning | 费率阶梯差异 |
| 滑点差 | 20 bps | warning | 深度/时序差异 |
| 总PnL差 | 10% | critical | 综合差异 |

#### 0.6.6 PnL归因

```python
@dataclass
class PnLAttribution:
    trade_id: str
    trace_id: str               # 关联到完整trace
    gross_pnl: Decimal          # 毛收益
    trading_fee: Decimal        # 手续费
    funding_fee: Decimal        # 资金费率
    slippage_cost: Decimal      # 滑点成本
    net_pnl: Decimal            # 净收益

    def validate(self) -> bool:
        """校验: net = gross - all_costs"""
        expected = self.gross_pnl - self.trading_fee - self.funding_fee - self.slippage_cost
        return abs(expected - self.net_pnl) < Decimal("0.01")
```

---

### 0.7 S6: 验收测试 (Acceptance Tests)

> **规则**: 每个模块上线前必须通过以下测试。

#### 0.7.1 测试类型

| 测试类型 | 说明 | 必须通过 |
|----------|------|----------|
| T1: 可见性测试 | 绝不使用未来数据 | ✅ PR必须 |
| T2: 复现测试 | 同快照同输出 | ✅ 发布必须 |
| T3: 对齐测试 | 回测-实盘偏差在阈值内 | ✅ 发布必须 |
| T4: 稳定性测试 | 断线/补数/重复幂等 | ✅ PR必须 |
| T5: 因子质量测试 | IC/稳定性/冗余度 | ✅ 因子上线必须 |

#### 0.7.2 CI/CD门禁

```yaml
test_gates:
  pr_required: [T1, T4]
  release_required: [T1, T2, T3, T4]
  factor_admission: [T5, Walk-Forward]
  coverage:
    min_line_coverage: 80%
    critical_modules: 95%
```

---

### 0.8 S7: 物理边界隔离 (P0-1)

> **要求**: research 与 production 必须目录隔离，用 CI 门禁强制执行。

#### 0.8.1 目录隔离规范

```
algvex/
├── production/          # 生产代码 (MVP)
│   ├── factors/         # 仅MVP-11因子
│   │   ├── __init__.py
│   │   ├── momentum.py      # return_5m, return_1h, ma_cross, breakout_20d, trend_strength
│   │   ├── volatility.py    # atr_288, realized_vol_1d, vol_regime
│   │   └── orderflow.py     # oi_change_rate, funding_momentum, oi_funding_divergence
│   ├── engine/          # 生产因子计算引擎
│   │   ├── factor_engine.py     # 不依赖Qlib
│   │   └── model_runner.py      # 加载导出的模型权重
│   ├── signal/          # 信号生成
│   └── execution/       # 执行层接口
│
├── research/            # 研究代码 (可选)
│   ├── qlib_adapter.py  # Qlib适配器 (仅此处可import qlib)
│   ├── alpha180/        # 180因子研究
│   ├── alpha201/        # 201因子研究 (含P1扩展)
│   └── experiments/     # 实验代码
│
└── shared/              # 共享代码 (严格审查)
    ├── data_service.py  # DataManager接口
    ├── time_provider.py # 时间服务
    └── types.py         # 类型定义
```

#### 0.8.2 导入规则 (强制)

```python
# ============================================================
# 规则1: production/ 禁止 import qlib
# ============================================================
# ❌ 禁止 (CI会失败)
from qlib.data import D
import qlib

# ✅ 允许
from algvex.production.factors import momentum
from algvex.shared.data_service import DataService

# ============================================================
# 规则2: production/ 禁止 import research/
# ============================================================
# ❌ 禁止
from algvex.research.alpha180 import factors

# ✅ 允许
from algvex.production.factors import momentum

# ============================================================
# 规则3: research/ 可以 import production/ (共用基础设施)
# ============================================================
# ✅ 允许
from algvex.production.factors import momentum  # 研究可复用生产因子
from algvex.shared.data_service import DataService
```

#### 0.8.3 CI 门禁脚本

```python
# scripts/ci/check_import_boundary.py
"""CI门禁: 检查 production/ 的非法导入"""

import ast
import sys
from pathlib import Path

FORBIDDEN_IMPORTS_IN_PRODUCTION = [
    "qlib",
    "algvex.research",
]

def check_file(filepath: Path) -> list[str]:
    """检查单个文件的非法导入"""
    violations = []
    with open(filepath) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(alias.name.startswith(pkg) for pkg in FORBIDDEN_IMPORTS_IN_PRODUCTION):
                    violations.append(f"{filepath}:{node.lineno} - import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and any(node.module.startswith(pkg) for pkg in FORBIDDEN_IMPORTS_IN_PRODUCTION):
                violations.append(f"{filepath}:{node.lineno} - from {node.module}")

    return violations

def main():
    production_dir = Path("algvex/production")
    all_violations = []

    for py_file in production_dir.rglob("*.py"):
        all_violations.extend(check_file(py_file))

    if all_violations:
        print("❌ 边界违规! production/ 包含非法导入:")
        for v in all_violations:
            print(f"  {v}")
        sys.exit(1)

    print("✅ 边界检查通过")
    sys.exit(0)

if __name__ == "__main__":
    main()
```

#### 0.8.4 CI 配置

```yaml
# .github/workflows/boundary-check.yml
name: Boundary Check
on: [push, pull_request]

jobs:
  check-imports:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Check import boundaries
        run: python scripts/ci/check_import_boundary.py

      # 额外验证: production 即使 qlib 未安装也能运行
      - name: Verify production runs without qlib
        run: |
          pip install -e ".[production-only]"  # 不装qlib
          python -c "from algvex.production.engine import factor_engine; print('✅ production无需qlib')"
```

---

### 0.9 S8: 数据层唯一入口 (P0-2)

> **要求**: 禁止任何模块直接读 DB/Redis/文件，所有数据访问必须经过 DataService 接口。
>
> **v1.1.0 术语澄清**:
> - **DataService**: 抽象接口 (abstract interface)，定义数据访问方法，外部模块只能看到这个
> - **DataManager**: 具体实现 (concrete implementation)，内部持有 DB/Redis 连接信息
> - 外部模块通过依赖注入获取 DataService 接口，无法访问 DataManager 的连接信息

#### 0.9.1 唯一入口架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      数据层架构 (接口与实现分离)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  外部模块只能看到 (DataService 接口):                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  interface DataService:  # 抽象接口，无连接信息                       │  │
│  │    get_bars(symbol, start, end, freq) -> DataFrame                   │  │
│  │    get_snapshot(snapshot_id) -> Snapshot                             │  │
│  │    get_factor(factor_id, symbol, bar_time) -> float                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  内部实现 (DataManager，对外不可见):                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐               │
│  │  TimescaleDB   │  │     Redis      │  │   Parquet      │               │
│  │  (L0 事实源)   │  │   (L2 缓存)    │  │  (L1 快照)     │               │
│  └────────────────┘  └────────────────┘  └────────────────┘               │
│                                                                             │
│  关键: 连接信息只在 DataManager 内部，外部模块通过 DataService 接口访问      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 0.9.2 接口定义

```python
# algvex/shared/data_service.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import pandas as pd

class DataService(ABC):
    """数据服务接口 - 所有数据访问的唯一入口"""

    @abstractmethod
    def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        freq: str = "5m",
        snapshot_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取K线数据"""
        pass

    @abstractmethod
    def get_latest_bar(self, symbol: str, freq: str = "5m") -> pd.Series:
        """获取最新K线"""
        pass

    @abstractmethod
    def get_funding_rate(self, symbol: str, bar_time: datetime) -> float:
        """获取资金费率"""
        pass

    @abstractmethod
    def get_open_interest(self, symbol: str, bar_time: datetime) -> float:
        """获取持仓量"""
        pass

# ============================================================
# 接口与实现分离原则
# ============================================================
# ✅ 正确做法: DataManager 实现 DataService 接口，通过依赖注入使用
# class DataManager(DataService):
#     def __init__(self, db_url, redis_url):  # 连接信息只在 DataManager 内部
#         self._db = connect(db_url)
#         self._redis = connect(redis_url)
#
# def create_app():
#     manager = DataManager(db_url=os.getenv("DB_URL"), ...)
#     engine = FactorEngine(data_service=manager)  # 注入为 DataService 类型
#
# ❌ 禁止做法: 外部模块直接访问连接信息
# engine = FactorEngine(db_url=..., redis_url=...)  # 不应该让外部知道连接信息
```

#### 0.9.3 依赖注入

```python
# algvex/production/engine/factor_engine.py
class FactorEngine:
    """因子计算引擎 - 通过接口获取数据，不知道数据来源"""

    def __init__(self, data_service: DataService):
        # ✅ 只拿接口，不拿连接信息
        self._data = data_service

    def compute_return_5m(self, symbol: str, bar_time: datetime) -> float:
        bars = self._data.get_bars(symbol, bar_time - timedelta(minutes=5), bar_time, "5m")
        return (bars.iloc[-1]["close"] / bars.iloc[0]["close"]) - 1

# 应用启动时注入
# main.py
def create_app():
    # 连接信息只在这一处，外部模块看不到
    data_manager = DataManager(
        db_url=os.getenv("DB_URL"),      # 只有这里知道
        redis_url=os.getenv("REDIS_URL"), # 只有这里知道
    )
    factor_engine = FactorEngine(data_service=data_manager)
    return App(factor_engine)
```

#### 0.9.4 导入扫描门禁

```python
# scripts/ci/check_data_access.py
"""
CI门禁: 检查非法数据库/Redis直接访问

关键: 使用 ALLOWED_IMPL_PREFIXES (目录) 而非单个文件
避免误杀: API层、migrations、infrastructure 目录
"""

# 禁止直接导入的数据库/缓存库
FORBIDDEN_IMPORTS = [
    "psycopg2",
    "sqlalchemy",
    "redis",
    "asyncpg",
    "databases",
    "pymongo",
    "sqlite3",
]

# 允许直接访问 DB/Redis 的目录/文件前缀
# 这些模块负责实现数据访问，其他模块必须通过 DataService 接口
ALLOWED_IMPL_PREFIXES = [
    "algvex/infrastructure/",       # 数据访问实现层
    "api/database.py",              # FastAPI 数据库配置
    "api/models/",                  # SQLAlchemy 模型定义
    "migrations/",                  # Alembic 迁移脚本
    "alembic/",                     # 另一种迁移目录名
    "tests/",                       # 测试可以直接访问
]

def is_allowed_file(file_path: str) -> bool:
    """检查文件是否在允许访问 DB 的白名单中"""
    return any(file_path.startswith(prefix) for prefix in ALLOWED_IMPL_PREFIXES)

def check_data_access_violations():
    """扫描非法数据访问 (排除白名单目录)"""
    import ast
    import sys
    from pathlib import Path

    violations = []

    # 扫描所有 Python 文件
    for py_file in Path(".").rglob("*.py"):
        rel_path = str(py_file)

        # 跳过允许访问 DB 的目录
        if is_allowed_file(rel_path):
            continue

        # 跳过非 algvex 目录 (只检查核心代码)
        if not rel_path.startswith("algvex/"):
            continue

        content = py_file.read_text()
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for banned in FORBIDDEN_IMPORTS:
                        if banned in alias.name:
                            violations.append({
                                "file": rel_path,
                                "line": node.lineno,
                                "type": "direct_import",
                                "module": alias.name,
                            })

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for banned in FORBIDDEN_IMPORTS:
                        if banned in node.module:
                            violations.append({
                                "file": rel_path,
                                "line": node.lineno,
                                "type": "from_import",
                                "module": node.module,
                            })

    if violations:
        print(f"❌ 发现 {len(violations)} 处数据访问违规:")
        for v in violations:
            print(f"  {v['file']}:{v['line']} - 非法导入 {v['module']}")
        print("\n提示: 业务代码应通过 DataService 接口访问数据")
        sys.exit(1)
    else:
        print("✅ 数据访问检查通过")
        return True

if __name__ == "__main__":
    check_data_access_violations()
```

---

### 0.10 S9: Canonical Hashing规范 (P0-3)

> **要求**: config_hash 必须基于规范化内容，定义统一的序列化/排序/精度规则。

#### 0.10.1 Canonical 规范

> **关键说明**: hash 基于"解析后对象的 canonical JSON string"，不是 YAML 文本

```yaml
# config/hashing_spec.yaml
hashing_specification:
  version: "1.1.0"

  # 序列化规则 (用于hash计算，不是文件格式)
  serialization:
    # 使用JSON而非YAML，因为JSON更稳定确定
    format: "json"
    json_options:
      sort_keys: true           # 强制按key排序
      separators: [",", ":"]    # 无空格，最紧凑
      ensure_ascii: false       # 保留unicode

  # 浮点数规则
  float_precision:
    max_decimals: 8             # 最多8位小数
    rounding: "half_even"       # 银行家舍入
    representation: "string"    # 转为字符串避免精度问题

  # 时间格式
  datetime_format: "ISO8601"    # 2025-12-22T00:00:00+00:00

  # 排除字段 (这些字段不参与hash计算)
  excluded_fields:
    - "config_hash"             # hash本身不参与
    - "_comments"               # 注释不参与
    - "_meta"                   # 元信息不参与

  # hash算法
  algorithm: "sha256"
  truncate_to: 32               # 取前32字符 (128位，推荐用于审计)
  # 或使用 full 保存完整hash: truncate_to: 64
```

#### 0.10.2 Canonical Hash 实现

> **说明**: 使用 JSON 计算 hash，使用 ruamel.yaml 写回配置文件

**依赖版本锁定** (requirements.txt):
```
# 锁定版本，确保跨环境一致性
PyYAML==6.0.1          # 用于读取
ruamel.yaml==0.18.6    # 用于写回 (保留注释/格式)
```

```python
# algvex/core/canonical_hash.py
"""
Canonical Hashing 完整实现

关键稳定性保证:
1. PyYAML 版本锁定 (6.0.1)
2. 浮点数转 Decimal 字符串 (不转回 float)
3. datetime 强制 ISO8601 UTC
4. dict/list 递归排序
5. set 转排序 list
"""
import hashlib
import yaml
from decimal import Decimal, ROUND_HALF_EVEN
from datetime import datetime, timezone
from typing import Any, Union
import json

class CanonicalHasher:
    """规范化哈希计算器"""

    EXCLUDED_FIELDS = {"config_hash", "_comments", "_meta"}
    FLOAT_PRECISION = 8

    def compute_hash(self, config: dict) -> str:
        """计算配置的规范化哈希"""
        # 1. 移除排除字段
        cleaned = self._remove_excluded(config)

        # 2. 规范化值 (转为稳定字符串表示)
        normalized = self._normalize_values(cleaned)

        # 3. 递归排序 (确保key顺序稳定)
        sorted_obj = self._deep_sort(normalized)

        # 使用JSON而非YAML，因为JSON更稳定
        # json.dumps 保证: sort_keys + separators 确定性
        canonical_str = json.dumps(
            sorted_obj,
            sort_keys=True,
            separators=(',', ':'),  # 无空格，最紧凑
            ensure_ascii=False,
        )

        # 4. 计算hash (截断32字符=128位，满足审计需求)
        full_hash = hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()
        return f"sha256:{full_hash[:32]}"

    def _remove_excluded(self, obj: Any) -> Any:
        """递归移除排除字段"""
        if isinstance(obj, dict):
            return {
                k: self._remove_excluded(v)
                for k, v in obj.items()
                if k not in self.EXCLUDED_FIELDS
            }
        elif isinstance(obj, list):
            return [self._remove_excluded(item) for item in obj]
        return obj

    def _normalize_values(self, obj: Any) -> Any:
        """
        规范化值

        关键: 浮点数转为 Decimal 字符串，不转回 float
        避免 float 的精度问题导致 hash 不一致
        """
        if isinstance(obj, dict):
            return {k: self._normalize_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_values(item) for item in obj]
        elif isinstance(obj, set):
            # set 转排序 list
            return sorted([self._normalize_values(item) for item in obj])
        elif isinstance(obj, float):
            # 浮点数 -> Decimal -> 字符串 (不转回float!)
            d = Decimal(str(obj)).quantize(
                Decimal(f"1e-{self.FLOAT_PRECISION}"),
                rounding=ROUND_HALF_EVEN
            )
            # 返回规范化字符串，如 "0.12345678"
            return str(d.normalize())
        elif isinstance(obj, Decimal):
            # 已经是 Decimal，直接规范化
            d = obj.quantize(
                Decimal(f"1e-{self.FLOAT_PRECISION}"),
                rounding=ROUND_HALF_EVEN
            )
            return str(d.normalize())
        elif isinstance(obj, datetime):
            # 强制 UTC ISO8601
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=timezone.utc)
            return obj.strftime("%Y-%m-%dT%H:%M:%SZ")
        return obj

    def _deep_sort(self, obj: Any) -> Any:
        """
        递归深度排序，确保嵌套结构稳定

        关键规则:
        - dict key 排序 ✅
        - set 转 sorted list ✅
        - list 保持原顺序 ✅ (不排序! 顺序是语义的一部分)

        不排序 list 的原因:
        - degrade_policy 的优先级顺序
        - rules 的匹配顺序
        - pipelines 的执行顺序
        这些顺序变化应该被视为真实的配置变更并导致 hash 变化
        """
        if isinstance(obj, dict):
            # dict key 排序
            return {k: self._deep_sort(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, set):
            # set 转 sorted list (set 本身无序，需排序)
            return sorted(self._deep_sort(x) for x in obj)
        elif isinstance(obj, list):
            # list 保持原顺序! 不排序!
            return [self._deep_sort(x) for x in obj]
        return obj

    def verify_hash(self, config: dict) -> bool:
        """验证配置的hash是否正确"""
        expected = config.get("config_hash", "")
        actual = self.compute_hash(config)
        return expected == actual

    @staticmethod
    def self_test() -> bool:
        """自测确保序列化稳定"""
        hasher = CanonicalHasher()

        # 测试用例: 相同语义，不同表示
        test_cases = [
            ({"a": 1.0, "b": 2.0}, {"b": 2.0, "a": 1.0}),  # 顺序不同
            ({"x": 0.1 + 0.2}, {"x": 0.3}),  # 浮点精度
            ({"t": datetime(2024, 1, 1, 0, 0, 0)},
             {"t": datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)}),
        ]

        for a, b in test_cases:
            hash_a = hasher.compute_hash(a)
            hash_b = hasher.compute_hash(b)
            if hash_a != hash_b:
                print(f"❌ 自测失败: {a} != {b}")
                return False

        print("✅ Canonical Hash 自测通过")
        return True
```

#### 0.10.3 CI 自动更新 Hash

> **说明**: 添加 write 权限 + 防死循环 guard

```yaml
# .github/workflows/update-config-hash.yml
name: Update Config Hashes
on:
  push:
    paths:
      - 'config/**/*.yaml'
    # 排除bot自己的提交，防止死循环
    branches-ignore:
      - 'dependabot/**'

# 必须有 write 权限才能 push
permissions:
  contents: write

jobs:
  update-hashes:
    runs-on: ubuntu-latest
    # 跳过自动更新触发的提交
    if: "!contains(github.event.head_commit.message, 'chore: auto-update')"
    steps:
      - uses: actions/checkout@v4
        with:
          # 使用 token 以便 push
          token: ${{ secrets.GITHUB_TOKEN }}

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install ruamel.yaml==0.18.6  # 保留注释/格式，与脚本import一致

      - name: Update config hashes
        id: update
        run: |
          python scripts/ci/update_config_hashes.py
          # 检查是否有实际变更
          if git diff --quiet config/; then
            echo "no_changes=true" >> $GITHUB_OUTPUT
          else
            echo "no_changes=false" >> $GITHUB_OUTPUT
          fi

      - name: Commit updated hashes
        # 只在有变更时提交
        if: steps.update.outputs.no_changes == 'false'
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add config/
          # commit message 带特殊标记，用于上面的 if 判断
          git commit -m "chore: auto-update config hashes [skip ci]"
          git push
```

```python
# scripts/ci/update_config_hashes.py
"""
配置哈希自动更新脚本

关键: 使用 ruamel.yaml RoundTrip 模式保留注释和格式!
- 读: ruamel.yaml RoundTripLoader
- 写: ruamel.yaml RoundTripDumper
- hash 计算: 基于解析后对象的 canonical JSON (不依赖 YAML 文本)
"""

import sys
from pathlib import Path
from ruamel.yaml import YAML
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algvex.core.canonical_hash import CanonicalHasher

def update_all_config_hashes():
    """更新所有配置文件的哈希 (保留注释和格式)"""
    hasher = CanonicalHasher()
    config_dir = Path("config")
    updated_count = 0

    # 使用 ruamel.yaml RoundTrip 模式 (保留注释/顺序/格式)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    for yaml_file in config_dir.rglob("*.yaml"):
        with open(yaml_file, 'r') as f:
            config = yaml.load(f)

        if config is None:
            continue

        # 计算新哈希 (基于解析后的 dict, 不是 YAML 文本)
        # 注意: ruamel 返回的是 CommentedMap, 需转为普通 dict 计算 hash
        config_dict = dict(config) if hasattr(config, 'items') else config
        new_hash = hasher.compute_hash(config_dict)
        old_hash = config.get("config_hash", "")

        # 幂等检查 - 只在哈希真正变化时更新
        if old_hash == new_hash:
            continue

        # 更新哈希 (只改这一个字段，保留其他一切)
        config["config_hash"] = new_hash

        # 写回 (ruamel 保留注释/顺序/格式，PR diff 最小化)
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f)

        print(f"✅ 更新 {yaml_file}: {old_hash[:20] if old_hash else 'None'}... -> {new_hash}")
        updated_count += 1

    if updated_count == 0:
        print("ℹ️ 所有配置哈希已是最新，无需更新")
    else:
        print(f"✅ 共更新 {updated_count} 个配置文件")

    return updated_count

if __name__ == "__main__":
    update_all_config_hashes()
```

#### 0.10.4 Contract Hash 定义

```python
# contract_hash = 所有契约配置的联合hash
def compute_contract_hash() -> str:
    """计算所有契约配置的联合hash"""
    hasher = CanonicalHasher()
    contract_files = [
        "config/visibility.yaml",
        "config/data_contracts/klines_5m.yaml",
        "config/data_contracts/open_interest_5m.yaml",
        "config/data_contracts/funding_8h.yaml",
    ]

    combined = {}
    for filepath in sorted(contract_files):  # 排序保证稳定
        with open(filepath) as f:
            config = yaml.safe_load(f)
            combined[filepath] = hasher._normalize_values(
                hasher._remove_excluded(config)
            )

    return hasher.compute_hash(combined)
```

---

### 0.11 S10: Replay确定性保障 (P0-4)

> **要求**: 消除 Replay 中的非确定性来源，保证 Live vs Replay 可精确对比。

#### 0.11.1 非确定性来源与对策

> **说明**: 除了 random/time，还需处理线程并行、set顺序、浮点精度

| 非确定性来源 | 对策 |
|--------------|------|
| `datetime.now()` | 统一使用 TimeProvider，Replay 固定时钟 |
| `random` / `np.random` | 统一使用 SeededRandom，固定种子 |
| numpy/pandas 线程并行 | 固定 MKL/OPENBLAS 线程数=1 |
| set 迭代顺序 | 转为 sorted list 后迭代 |
| 并发队列顺序 | 使用确定性优先级队列 |
| 浮点计算误差 | 关键路径使用 Decimal |
| Live用WS vs Replay用落盘 | Replay 必须使用 Live 记录的 data_hash |
| 字典迭代顺序 | Python 3.7+ 默认有序，无需处理 |

#### 0.11.1.1 环境确定性配置

```python
# algvex/core/determinism.py
"""
完整的确定性环境配置

关键约束:
- PYTHONHASHSEED 必须在进程启动前设置 (shell/cron 启动脚本)
- 线程变量也应在启动前设置，但可在 import numpy 前补救
- 本模块只能"验证"，不能"补救" PYTHONHASHSEED
"""
import os
import sys
import warnings
from typing import Optional, List

class DeterministicEnvError(Exception):
    """确定性环境配置错误 - 生产环境必须拒绝启动"""
    pass

def setup_deterministic_env(seed: int = 42, num_threads: int = 1, strict: bool = True):
    """
    配置确定性环境

    必须在 import numpy/pandas 之前调用!

    Args:
        seed: 全局随机种子
        num_threads: 线程数 (1=单线程，确保确定性)
        strict: True=生产环境(违规抛异常), False=研究环境(违规警告)
    """
    issues = []

    # ============ 检查 PYTHONHASHSEED (只能检查，不能补救!) ============
    # Python hash seed 在解释器启动时就确定了
    # 运行时设置 os.environ["PYTHONHASHSEED"] 不会改变当前进程的 hash 行为!
    current_hashseed = os.environ.get("PYTHONHASHSEED")
    if current_hashseed is None:
        issues.append("PYTHONHASHSEED 未在启动前设置! 请在 shell/cron 启动脚本中设置")
    elif current_hashseed != str(seed):
        issues.append(f"PYTHONHASHSEED={current_hashseed} 与预期 {seed} 不符")

    # ============ 设置线程变量 (必须在 import numpy 之前) ============
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"

    # ============ 导入 numpy 并设置随机种子 ============
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)

    try:
        import mkl
        mkl.set_num_threads(num_threads)
    except ImportError:
        pass

    # ============ 处理问题 ============
    if issues:
        msg = f"确定性环境问题: {issues}"
        if strict:
            raise DeterministicEnvError(msg)
        else:
            warnings.warn(msg, UserWarning)
            return False

    print(f"✅ 确定性环境已验证: seed={seed}, threads={num_threads}")
    return True


def verify_deterministic_env() -> bool:
    """验证确定性环境是否正确配置"""
    issues = []

    # 检查线程配置
    thread_vars = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"
    ]
    for var in thread_vars:
        val = os.environ.get(var)
        if val != "1":
            issues.append(f"{var}={val} (应为1)")

    # 检查 hash seed
    if os.environ.get("PYTHONHASHSEED") is None:
        issues.append("PYTHONHASHSEED 未设置")

    if issues:
        warnings.warn(f"确定性环境问题: {issues}", UserWarning)
        return False

    return True
```

#### 0.11.1.2 Set/Collection 确定性处理

```python
# algvex/shared/deterministic_collections.py
"""
确定性集合操作

问题: set 迭代顺序在不同运行/环境下可能不同
解决: 所有需要迭代 set 的地方，先转为 sorted list
"""
from typing import Set, List, Any, Callable, TypeVar

T = TypeVar('T')

def sorted_set_iter(s: Set[T], key: Callable[[T], Any] = None) -> List[T]:
    """
    确定性迭代 set

    用法:
        for item in sorted_set_iter(my_set):
            process(item)
    """
    if key:
        return sorted(s, key=key)
    return sorted(s)


def deterministic_dict_keys(d: dict) -> List[str]:
    """确定性获取 dict keys (虽然 Python 3.7+ 有序，但显式排序更安全)"""
    return sorted(d.keys())


class DeterministicPriorityQueue:
    """
    确定性优先级队列

    问题: heapq 在相同优先级时顺序不确定
    解决: 添加序列号作为 tiebreaker
    """
    import heapq

    def __init__(self):
        self._queue = []
        self._counter = 0

    def push(self, priority: float, item: Any):
        """插入元素 (priority 越小越优先)"""
        import heapq
        # counter 作为 tiebreaker，确保 FIFO
        heapq.heappush(self._queue, (priority, self._counter, item))
        self._counter += 1

    def pop(self) -> Any:
        """弹出最高优先级元素"""
        import heapq
        _, _, item = heapq.heappop(self._queue)
        return item

    def __len__(self):
        return len(self._queue)
```

#### 0.11.1.3 关键路径 Decimal 规范

```python
# algvex/shared/decimal_utils.py
"""
关键路径 Decimal 规范

关键路径 = 任何影响信号/订单的数值计算
"""
from decimal import Decimal, ROUND_HALF_EVEN, getcontext
from typing import Union

# 设置全局 Decimal 精度
def setup_decimal_context():
    """必须在应用启动时调用"""
    ctx = getcontext()
    ctx.prec = 28  # 28位精度 (足够覆盖加密货币价格)
    ctx.rounding = ROUND_HALF_EVEN  # 银行家舍入

# 强制 Decimal 的关键计算
def safe_divide(a: Union[Decimal, float], b: Union[Decimal, float]) -> Decimal:
    """安全除法，避免浮点误差"""
    a_dec = Decimal(str(a)) if isinstance(a, float) else a
    b_dec = Decimal(str(b)) if isinstance(b, float) else b
    return a_dec / b_dec

def safe_multiply(a: Union[Decimal, float], b: Union[Decimal, float]) -> Decimal:
    """安全乘法"""
    a_dec = Decimal(str(a)) if isinstance(a, float) else a
    b_dec = Decimal(str(b)) if isinstance(b, float) else b
    return a_dec * b_dec

# 因子计算必须返回 Decimal
class DecimalFactor:
    """因子值封装 - 确保精度"""

    def __init__(self, value: Union[float, Decimal, str]):
        if isinstance(value, float):
            self.value = Decimal(str(value))
        elif isinstance(value, str):
            self.value = Decimal(value)
        else:
            self.value = value

    def __float__(self) -> float:
        return float(self.value)

    def __repr__(self):
        return f"DecimalFactor({self.value})"
```

#### 0.11.1.4 启动时强制检查

```python
# algvex/main.py (示例)
"""应用入口 - 必须首先配置确定性环境"""

# ⚠️ 必须在其他 import 之前!
from algvex.core.determinism import setup_deterministic_env
setup_deterministic_env(seed=42, num_threads=1)

# 现在可以安全导入其他模块
import numpy as np
import pandas as pd
from algvex.core.mvp_scope_enforcer import MvpScopeEnforcer
# ...
```

#### 0.11.2 TimeProvider 实现

```python
# algvex/shared/time_provider.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

class TimeProvider(ABC):
    """时间服务接口 - 消除 datetime.now() 的非确定性"""

    @abstractmethod
    def now(self) -> datetime:
        """获取当前时间"""
        pass

    @abstractmethod
    def sleep(self, seconds: float) -> None:
        """等待指定秒数"""
        pass


class LiveTimeProvider(TimeProvider):
    """生产环境: 使用真实时间"""

    def now(self) -> datetime:
        return datetime.utcnow()

    def sleep(self, seconds: float) -> None:
        import time
        time.sleep(seconds)


class ReplayTimeProvider(TimeProvider):
    """Replay环境: 使用固定时间序列"""

    def __init__(self, timestamps: list[datetime]):
        self._timestamps = iter(timestamps)
        self._current: Optional[datetime] = None

    def now(self) -> datetime:
        if self._current is None:
            self._current = next(self._timestamps)
        return self._current

    def advance(self) -> None:
        """推进到下一个时间点"""
        self._current = next(self._timestamps)

    def sleep(self, seconds: float) -> None:
        # Replay 中 sleep 是空操作
        pass
```

#### 0.11.3 SeededRandom 实现

```python
# algvex/shared/seeded_random.py
import random
import numpy as np
from typing import Optional

class SeededRandom:
    """确定性随机数生成器"""

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._py_random = random.Random(seed)
        self._np_random = np.random.RandomState(seed)

    def random(self) -> float:
        return self._py_random.random()

    def randint(self, a: int, b: int) -> int:
        return self._py_random.randint(a, b)

    def choice(self, seq):
        return self._py_random.choice(seq)

    def numpy_random(self) -> np.random.RandomState:
        return self._np_random

    def get_seed(self) -> int:
        return self._seed


# 全局单例 (在启动时设置)
_global_random: Optional[SeededRandom] = None

def set_global_random(seed: int):
    global _global_random
    _global_random = SeededRandom(seed)

def get_global_random() -> SeededRandom:
    if _global_random is None:
        raise RuntimeError("SeededRandom not initialized. Call set_global_random() first.")
    return _global_random
```

#### 0.11.4 Replay 输入数据要求

```python
# Replay 必须使用与 Live 完全相同的输入数据
@dataclass
class ReplayInput:
    """Replay 所需的确定性输入"""

    # 必须来自 Live 记录
    date: str
    snapshot_id: str          # Live 使用的快照ID
    config_hash: str          # Live 使用的配置hash
    code_hash: str            # Live 使用的代码hash

    # 确定性控制
    random_seed: int          # 随机种子
    timestamps: list[datetime]  # 时间序列

    # 数据验证
    data_hash: str            # 输入数据的hash (必须与Live一致)

def validate_replay_input(replay_input: ReplayInput, live_trace: dict) -> bool:
    """验证 Replay 输入与 Live 记录一致"""
    return (
        replay_input.snapshot_id == live_trace["snapshot_id"] and
        replay_input.config_hash == live_trace["config_hash"] and
        replay_input.code_hash == live_trace["code_hash"] and
        replay_input.data_hash == live_trace["data_hash"]
    )
```

#### 0.11.5 关键路径 Decimal

```python
# algvex/production/signal/position_calculator.py
from decimal import Decimal, ROUND_DOWN

class PositionCalculator:
    """仓位计算 - 使用 Decimal 保证精度"""

    def calculate_quantity(
        self,
        capital: Decimal,
        price: Decimal,
        leverage: int,
        risk_pct: Decimal,
    ) -> Decimal:
        """计算下单数量 (Decimal精度)"""
        risk_capital = capital * risk_pct
        notional = risk_capital * Decimal(leverage)
        quantity = (notional / price).quantize(Decimal("0.001"), rounding=ROUND_DOWN)
        return quantity
```

---

### 0.12 Iteration-1/2/3/4 交付计划

> 将大范围改造拆分为4个可验证的小迭代。

#### 0.12.1 Iteration-1: 契约 + 可见性 + Trace + 边界隔离 (2周)

```
Iteration-1 交付物:
├── 配置文件
│   ├── config/visibility.yaml
│   ├── config/data_contracts/klines_5m.yaml
│   ├── config/data_contracts/open_interest_5m.yaml
│   ├── config/data_contracts/funding_8h.yaml
│   └── config/factor_governance.yaml
├── 核心代码
│   ├── algvex/core/config_validator.py    # 配置哈希校验
│   ├── algvex/core/trace_logger.py        # Trace记录
│   └── algvex/core/visibility_checker.py  # 可见性检查
└── 测试
    ├── tests/test_config_hash.py
    ├── tests/test_visibility.py
    └── tests/test_trace_schema.py

验收标准:
✅ 所有配置文件有 config_version + config_hash
✅ 启动时自动校验所有配置hash
✅ 每条信号记录完整trace (包含config_hash, code_hash)
✅ T1可见性测试100%通过
├── 边界隔离 (P0-1)
│   ├── algvex/production/           # 生产目录 (MVP-11因子)
│   ├── algvex/research/             # 研究目录 (alpha180/201)
│   └── scripts/ci/check_import_boundary.py
├── 数据入口 (P0-2)
│   ├── algvex/shared/data_service.py   # 接口定义
│   └── scripts/ci/check_data_access.py
└── 确定性 (P0-4)
    ├── algvex/shared/time_provider.py
    └── algvex/shared/seeded_random.py

验收标准:
✅ 所有配置文件有 config_version + config_hash
✅ 启动时自动校验所有配置hash (canonical hashing)
✅ production/ 导入边界检查通过 (CI门禁)
✅ 非DataManager模块禁止直接访问DB/Redis
✅ T1可见性测试100%通过
```

#### 0.12.2 Iteration-2: Daily Replay 对齐 + 确定性 (2周)

```
Iteration-2 交付物:
├── 脚本
│   ├── scripts/daily_replay_alignment.py
│   ├── scripts/snapshot_manager.py
│   └── scripts/alignment_reporter.py
├── 核心代码
│   ├── algvex/core/snapshot_store.py      # 快照存储
│   ├── algvex/core/replay_runner.py       # Replay执行器 (使用TimeProvider)
│   └── algvex/core/alignment_checker.py   # 对齐检查
├── 确定性保障
│   ├── 统一TimeProvider (禁止datetime.now())
│   ├── 统一SeededRandom (固定随机种子)
│   └── 关键路径Decimal (仓位/价格计算)
├── 定时任务
│   └── cron/daily_alignment_job.sh
└── 测试
    ├── tests/test_snapshot.py
    ├── tests/test_replay.py
    ├── tests/test_alignment.py
    └── tests/test_determinism.py  # 确定性验证

验收标准:
✅ Live运行产生 live_output_{date}.jsonl
✅ Replay使用相同 snapshot_id + data_hash 产生 replay_output_{date}.jsonl
✅ 自动比对报告 alignment_report_{date}.json
✅ 信号差异 > 0.1% 自动告警
✅ 连续7天Replay对齐通过 (信号完全一致)
```

#### 0.12.3 Iteration-3: 数据快照 + Qlib边界 (2周)

```
Iteration-3 交付物:
├── 数据快照
│   ├── algvex/data/snapshot_creator.py    # 快照生成
│   ├── algvex/data/snapshot_loader.py     # 快照加载
│   └── data/snapshots/                    # 快照存储目录
├── Qlib边界 (物理隔离)
│   ├── algvex/research/qlib_adapter.py    # Qlib适配器(仅研究用)
│   ├── algvex/production/factor_engine.py # 生产因子计算(不依赖Qlib)
│   └── algvex/production/model_loader.py  # 模型加载(从Qlib导出)
├── 模型导出
│   ├── scripts/export_qlib_model.py       # 导出Qlib模型权重
│   └── models/exported/                   # 导出的模型文件
└── 测试
    ├── tests/test_snapshot_integrity.py
    ├── tests/test_production_factors.py
    └── tests/test_qlib_boundary.py

验收标准:
✅ 快照可以被完整存储和恢复
✅ 使用相同快照的多次运行产生相同结果 (hash一致)
✅ 生产因子计算不依赖Qlib (pip uninstall qlib后仍可运行)
✅ Qlib模型可以成功导出并被生产代码加载
✅ CI门禁: production/ 不允许 import qlib
```

#### 0.12.4 Iteration-4: Hummingbot 执行层集成 (2周)

> **目标**: 完成 AlgVex 与 Hummingbot 的深度集成，实现 Paper Trading 验证

```
Iteration-4 交付物:
├── 执行层核心
│   ├── algvex/core/execution/hummingbot_bridge.py     # Hummingbot 桥接层
│   ├── algvex/core/execution/order_tracker.py         # 订单追踪器
│   ├── algvex/core/execution/state_synchronizer.py    # 状态同步器
│   └── algvex/core/execution/event_handlers.py        # 事件处理器
├── Strategy V2 集成
│   ├── algvex/core/execution/controllers/
│   │   ├── __init__.py
│   │   └── algvex_controller.py                       # AlgVex Controller
│   └── config/hummingbot_connector.yaml               # 连接器配置
├── Paper Trading
│   ├── scripts/paper_trading.py                       # Paper Trading 启动脚本
│   └── logs/paper_trading/                            # Paper Trading 日志
└── 测试
    ├── tests/execution/test_hummingbot_bridge.py      # Bridge 单元测试
    ├── tests/execution/test_order_idempotency.py      # 幂等性测试
    ├── tests/execution/test_state_sync.py             # 状态同步测试
    └── tests/execution/test_event_handlers.py         # 事件处理测试

验收标准:
✅ HummingbotBridge 信号 → 订单转换正确
✅ 幂等性: 相同信号重复调用返回相同 order_id
✅ InFlightOrder 订单状态追踪正确
✅ 状态同步: 仓位对账无差异
✅ 事件处理: 所有事件正确写入 trace
✅ AlgVexController V2 策略集成正常
✅ 断线恢复: 断线后能正确恢复状态
✅ Paper Trading: 24h 模拟运行无异常
```

#### 0.12.5 迭代验证矩阵

| 迭代 | 核心验证 | 通过标准 |
|------|----------|----------|
| Iter-1 | 配置可追溯 + 边界隔离 | config_hash校验通过 + 导入边界CI通过 |
| Iter-2 | 每日对齐 + 确定性 | 连续7天Live-Replay信号差异=0 |
| Iter-3 | 快照可复现 + Qlib隔离 | 同快照3次运行hash相同 + 无Qlib可运行 |
| **Iter-4** | **执行层集成 + Paper Trading** | **24h Paper Trading无异常 + 状态同步100%** |

---

### 0.13 硬约束层检查清单

| 契约 | 配置文件 | 状态 | 验收测试 |
|------|----------|------|----------|
| S0: MVP Scope | `mvp_scope.yaml` | ✅ 已定义 | 边界检查 |
| S1: 时间+快照契约 | `visibility.yaml` | ✅ 已定义 | T1_VisibilityTests |
| S2: 数据契约模板 | `data_contracts/*.yaml` | ⬜ Iter-1交付 | 数据源审查 |
| S3: 预算与降级策略 | `budget.yaml` | ✅ 已定义 | 压力测试 |
| S4: 因子治理 | `factor_governance.yaml` | ✅ 已定义 | T5_FactorTests |
| S5: 对齐与归因 | `alignment.yaml` | ⬜ Iter-2交付 | T3_AlignmentTests |
| S6: 验收测试 | - | ✅ 已定义 | CI/CD集成 |
| **S7: 物理边界隔离** | - | ⬜ Iter-1交付 | 导入扫描门禁 |
| **S8: DataManager唯一入口** | - | ⬜ Iter-1交付 | 数据访问扫描 |
| **S9: Canonical Hashing** | `hashing_spec.yaml` | ⬜ Iter-1交付 | Hash验证测试 |
| **S10: Replay确定性** | - | ⬜ Iter-2交付 | 确定性测试 |

#### 0.13.1 配置文件结构

```
config/
├── visibility.yaml              # S1: 可见性规则
├── alignment.yaml               # S5: 对齐配置
├── factor_governance.yaml       # S4: 因子治理
├── budget.yaml                  # S3: 预算配置
├── hashing_spec.yaml            # S9: Canonical Hashing规范
├── mvp_scope.yaml               # S0: MVP范围配置 (P0-5)
└── data_contracts/
    ├── klines_5m.yaml           # S2: K线数据契约
    ├── open_interest_5m.yaml    # S2: 持仓量数据契约
    └── funding_8h.yaml          # S2: 资金费率数据契约
```

#### 0.13.2 MVP Scope 配置开关 (P0-5)

```yaml
# config/mvp_scope.yaml
# MVP范围配置 - 生产环境必须遵守
config_version: "1.0.0"
config_hash: "sha256:..."

mvp_constraints:
  # 时间框架限制
  allowed_frequencies:
    - "5m"  # MVP仅允许5分钟
  forbidden_frequencies:
    - "1m"
    - "15m"
    - "1h"

  # 标的限制
  universe:
    max_symbols: 50
    default_symbols: ["BTCUSDT", "ETHUSDT"]
    # 动态扩展需要审批

  # 因子限制
  factor_set: "mvp11"  # 仅MVP-11因子
  forbidden_factors:
    - "alpha180/*"  # 禁止使用研究因子
    - "alpha201/*"

  # 数据源限制
  allowed_data_sources:
    - "klines_5m"
    - "open_interest_5m"
    - "funding_8h"
  forbidden_data_sources:
    - "depth_l2"      # B/C档数据
    - "liquidations"
    - "options_*"

  # 运行时强制检查
  enforcement:
    on_violation: "reject"  # reject / warn / log
    check_at_startup: true
    check_on_signal: true
```

#### 0.13.3 MvpScopeEnforcer 实现

> 必须在启动和每次信号入口处强制检查MVP边界

```python
# algvex/core/mvp_scope_enforcer.py
"""
MVP Scope 强制检查器

关键检查点:
1. 应用启动时 (check_at_startup)
2. 每次信号生成前 (check_on_signal)
3. 每次因子计算前 (check_factor)
4. 每次数据请求前 (check_data_source)
"""
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import List, Set, Optional
from enum import Enum
import fnmatch

class ViolationAction(Enum):
    REJECT = "reject"    # 拒绝并抛异常
    WARN = "warn"        # 警告但继续
    LOG = "log"          # 仅记录

@dataclass
class MvpViolation:
    """MVP范围违规"""
    category: str       # frequency / symbol / factor / data_source
    value: str          # 违规的值
    message: str        # 详细信息

class MvpScopeEnforcer:
    """
    MVP范围强制检查器

    用法:
    1. 应用启动时初始化
    2. 在 SignalGenerator.__init__ 中调用 check_startup()
    3. 在 SignalGenerator.generate() 入口调用 check_signal()
    """

    def __init__(self, config_path: str = "config/mvp_scope.yaml"):
        self.config = self._load_config(config_path)
        self.violations: List[MvpViolation] = []

        # 预解析配置
        constraints = self.config.get("mvp_constraints", {})
        self.allowed_frequencies: Set[str] = set(
            constraints.get("allowed_frequencies", ["5m"])
        )

        # Universe 配置
        universe = constraints.get("universe", {})
        self.max_symbols: int = universe.get("max_symbols", 50)
        self.allowed_symbols: Set[str] = set(universe.get("allowed_symbols", []))

        # 因子配置 - 使用白名单而非仅黑名单
        self.allowed_factors: Set[str] = set(
            constraints.get("allowed_factors", [])  # MVP-11 因子白名单
        )
        self.forbidden_factors: List[str] = constraints.get("forbidden_factors", [])

        # 数据源配置 - 白名单是权威
        self.allowed_data_sources: Set[str] = set(
            constraints.get("allowed_data_sources", [])
        )
        self.forbidden_data_sources: List[str] = constraints.get("forbidden_data_sources", [])

        enforcement = constraints.get("enforcement", {})
        self.on_violation = ViolationAction(enforcement.get("on_violation", "reject"))
        self.check_at_startup = enforcement.get("check_at_startup", True)
        self.check_on_signal = enforcement.get("check_on_signal", True)

    def _load_config(self, path: str) -> dict:
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"MVP Scope配置不存在: {path}")
        with open(config_file) as f:
            return yaml.safe_load(f)

    def check_startup(self, active_symbols: List[str], active_factors: List[str]):
        """
        启动时检查

        必须在 SignalGenerator.__init__ 或 main() 中调用
        """
        self.violations.clear()

        # 检查标的数量
        if len(active_symbols) > self.max_symbols:
            self.violations.append(MvpViolation(
                category="symbol",
                value=str(len(active_symbols)),
                message=f"标的数量 {len(active_symbols)} 超过MVP限制 {self.max_symbols}"
            ))

        # 检查标的是否在白名单 (如果配置了白名单)
        if self.allowed_symbols:
            for symbol in active_symbols:
                if symbol not in self.allowed_symbols:
                    self.violations.append(MvpViolation(
                        category="symbol",
                        value=symbol,
                        message=f"标的 {symbol} 不在MVP允许列表中"
                    ))

        # 检查因子 - 白名单优先，不在白名单的一律拒绝
        if self.allowed_factors:
            for factor in active_factors:
                if factor not in self.allowed_factors:
                    self.violations.append(MvpViolation(
                        category="factor",
                        value=factor,
                        message=f"因子 {factor} 不在MVP-11允许列表中"
                    ))
        else:
            # 没有白名单时，使用黑名单 (向后兼容)
            for factor in active_factors:
                if self._matches_forbidden(factor, self.forbidden_factors):
                    self.violations.append(MvpViolation(
                        category="factor",
                        value=factor,
                        message=f"因子 {factor} 在MVP禁止列表中"
                    ))

        return self._handle_violations("startup")

    def check_signal(self, frequency: str, symbol: str, factors_used: List[str]):
        """
        每次信号生成前检查

        必须在 SignalGenerator.generate() 入口调用
        """
        self.violations.clear()

        # 检查时间框架
        if frequency not in self.allowed_frequencies:
            self.violations.append(MvpViolation(
                category="frequency",
                value=frequency,
                message=f"时间框架 {frequency} 不在MVP允许列表 {self.allowed_frequencies}"
            ))

        # 检查标的 (如果配置了白名单)
        if self.allowed_symbols and symbol not in self.allowed_symbols:
            self.violations.append(MvpViolation(
                category="symbol",
                value=symbol,
                message=f"标的 {symbol} 不在MVP允许列表中"
            ))

        # 检查因子 - 白名单优先
        if self.allowed_factors:
            for factor in factors_used:
                if factor not in self.allowed_factors:
                    self.violations.append(MvpViolation(
                        category="factor",
                        value=factor,
                        message=f"信号使用了非MVP因子 {factor}"
                    ))
        else:
            for factor in factors_used:
                if self._matches_forbidden(factor, self.forbidden_factors):
                    self.violations.append(MvpViolation(
                        category="factor",
                        value=factor,
                        message=f"信号使用了禁止因子 {factor}"
                    ))

        return self._handle_violations(f"signal:{symbol}")

    def check_data_source(self, data_source: str):
        """
        数据请求前检查

        必须在 DataManager.get_* 方法入口调用

        关键逻辑: 只要不在 allowed_data_sources 就拒绝!
        forbidden_data_sources 仅用于提供更清晰的错误信息
        """
        if data_source not in self.allowed_data_sources:
            # 不在白名单 = 一律拒绝 (这是MVP边界的核心!)
            if self._matches_forbidden(data_source, self.forbidden_data_sources):
                message = f"数据源 {data_source} 在MVP禁止列表中"
            else:
                message = f"数据源 {data_source} 不在MVP允许列表中 (需审批后加入)"

            violation = MvpViolation(
                category="data_source",
                value=data_source,
                message=message
            )
            self.violations = [violation]
            return self._handle_violations(f"data:{data_source}")

        return True

    def _matches_forbidden(self, value: str, patterns: List[str]) -> bool:
        """检查是否匹配禁止模式 (支持通配符)"""
        for pattern in patterns:
            if fnmatch.fnmatch(value, pattern):
                return True
        return False

    def _handle_violations(self, context: str) -> bool:
        """处理违规"""
        if not self.violations:
            return True

        for v in self.violations:
            msg = f"[MVP违规] {context} - {v.category}: {v.message}"

            if self.on_violation == ViolationAction.REJECT:
                raise MvpScopeViolationError(msg)
            elif self.on_violation == ViolationAction.WARN:
                import warnings
                warnings.warn(msg, UserWarning)
            else:  # LOG
                import logging
                logging.warning(msg)

        return self.on_violation != ViolationAction.REJECT


class MvpScopeViolationError(Exception):
    """MVP范围违规异常 - 生产环境必须拒绝"""
    pass


# ============== 集成示例 ==============

class SignalGenerator:
    """信号生成器 - 展示MvpScopeEnforcer的集成方式"""

    def __init__(self, symbols: List[str], factors: List[str]):
        # 启动时强制检查
        self.enforcer = MvpScopeEnforcer()
        if self.enforcer.check_at_startup:
            self.enforcer.check_startup(symbols, factors)

        self.symbols = symbols
        self.factors = factors

    def generate(self, symbol: str, frequency: str = "5m") -> dict:
        # 每次信号生成前强制检查
        if self.enforcer.check_on_signal:
            self.enforcer.check_signal(frequency, symbol, self.factors)

        # ... 实际信号生成逻辑 ...
        return {"symbol": symbol, "signal": 0.5}
```

**CI门禁测试**

```python
# tests/p0/test_mvp_scope_enforcer.py
import pytest
from algvex.core.mvp_scope_enforcer import (
    MvpScopeEnforcer, MvpScopeViolationError
)

class TestMvpScopeEnforcer:
    """P0测试: MVP范围强制检查"""

    def test_reject_forbidden_frequency(self):
        """测试禁止的时间框架被拒绝"""
        enforcer = MvpScopeEnforcer()
        with pytest.raises(MvpScopeViolationError):
            enforcer.check_signal(
                frequency="15m",  # 禁止
                symbol="BTCUSDT",
                factors_used=["return_5m"]
            )

    def test_reject_non_allowed_data_source(self):
        """核心测试: 不在allowed列表的数据源必须拒绝 (即使不在forbidden)"""
        enforcer = MvpScopeEnforcer()
        with pytest.raises(MvpScopeViolationError):
            # "new_source" 既不在 allowed 也不在 forbidden，但必须拒绝!
            enforcer.check_data_source("new_source_xyz")

    def test_reject_forbidden_data_source(self):
        """测试明确禁止的数据源被拒绝"""
        enforcer = MvpScopeEnforcer()
        with pytest.raises(MvpScopeViolationError):
            enforcer.check_data_source("depth_l2")  # 在禁止列表

    def test_reject_non_allowed_factor(self):
        """核心测试: 不在MVP-11白名单的因子必须拒绝"""
        enforcer = MvpScopeEnforcer()
        with pytest.raises(MvpScopeViolationError):
            enforcer.check_startup(
                active_symbols=["BTCUSDT"],
                active_factors=["some_new_factor"]  # 不在 MVP-11 白名单
            )

    def test_reject_non_allowed_symbol(self):
        """核心测试: 不在universe白名单的标的必须拒绝"""
        enforcer = MvpScopeEnforcer()
        with pytest.raises(MvpScopeViolationError):
            enforcer.check_signal(
                frequency="5m",
                symbol="UNKNOWN_COIN",  # 不在 allowed_symbols
                factors_used=["return_5m"]
            )

    def test_allow_mvp_config(self):
        """测试MVP配置通过"""
        enforcer = MvpScopeEnforcer()
        assert enforcer.check_signal(
            frequency="5m",
            symbol="BTCUSDT",
            factors_used=["return_5m", "oi_change_rate"]
        )
```

---

# Part B: 实现层 (可渐进替换)

> 以下章节为实现层，可以根据需要渐进替换，但必须遵守 Part A 定义的硬约束。

---


### 0.14 逻辑一致性审查（v3.9.7 增补）

> 目的：把“复杂”拆成可验证的约束与代价，避免为了复杂而复杂；同时把潜在的逻辑漏洞提前暴露出来，防止实盘阶段才踩坑。

#### 0.14.1 审查维度（你提出的 7 项）

| 维度 | 结论 | 主要风险点（如果不补） | 本方案对应机制 | 需要补的证据/动作（最小集） |
|---|---|---|---|---|
| 一致性（定义/口径/命名） | **基本满足** | 因子/特征/标签口径漂移导致“同名不同义”与回测-实盘割裂 | S2 数据契约 + S4 因子治理 + S9 Hash规范 | 建立 **Metric/Factor Dictionary**（版本化）；CI 校验同名字段/同名因子 hash 不变（或显式 bump 版本） |
| 时间点数据（asof/水位线/快照） | **满足（设计已给）** | 外部数据延迟/缺失时产生“未来函数”或隐形漏数 | S1 时间+快照契约（watermark/asof）+ S3 降级 | 增加 2 类验收：① 回测/实盘同一日 replay 结果一致；② 故意制造延迟/缺失时触发降级并可观测 |
| 数据流闭环（输入→决策→执行→回写→归因） | **满足（需要把回写做成硬门槛）** | 只做“下单”，不做“回写+归因”，最终无法定位策略问题 | S5 对齐与归因 + Daily Replay | 把 **Fill/Order 回写** 写入 DoD：无回写则策略不可上线；增加 trace_id 贯穿（signal_id→order_id→fill_id） |
| 幂等/状态机（重试/重放/断点续跑） | **部分满足（需要补证据）** | 断线重连/重复事件导致重复下单、重复记账、重复信号 | S8 唯一入口 + S9 Canonical Hash + S10 Replay确定性 | 明确 **Order 幂等键**（client_order_id，Hummingbot内置支持）；Fills 通过 trade_id 去重；补"重放/重复事件"测试与演练脚本 |
| 降级链路（从正常到兜底的可控退化） | **满足（设计已给，但要防“静默降级”）** | 无声降级导致性能/胜率变化但不可见，误判策略有效性 | S3 预算与降级策略 + S6 验收 | 增加 **降级事件日志+指标**（degrade_level、原因、持续时长）；在回测输出中标注降级占比 |
| 接口签名/责任边界（Qlib vs 执行引擎） | **基本满足** | 研究/回测/实盘接口分叉，后期维护成本爆炸 | 明确 DataManager / Collector 接口，策略侧只消费“统一数据层” | 在仓库层面加 **import boundary**（禁止从执行层反向 import 研究层）；用 mypy/ruff 规则做门禁（可先轻量） |
| 循环依赖/口径漂移风险 | **需要补证据** | 随迭代出现“跨层引用”“字段含义悄悄改变”，导致长期不可维护 | 分层原则（P 系列）+ S4 因子治理 | 生成模块依赖图（CI 每次 PR 产出）；新增“字段/因子版本变更”必须更新 CHANGELOG 与验收记录 |

#### 0.14.2 本次审查结论：哪里复杂是“必要的”，哪里复杂必须补证据

- **必要复杂度（必须保留）**：S1/S2/S3/S4/S5/S6 + S8/S9/S10（它们是“防未来函数、口径漂移、实盘不可追责”的最低门槛）。
- **需要补证据才能成立的复杂度（先不删，但必须证据化/开关化）**：  
  1) 存储与组件选型的“高配版”（例如 Timescale/多缓存层/多队列等）——需要压测/成本收益证明；  
  2) 过度提前的“多交易所/多经纪商/多账户编排”——需要明确业务里程碑与真实需求；  
  3) 过度细粒度的微结构特征在 **与 S1 快照契约** 冲突时——需要定义在什么频率下仍可保持“时间点一致性”。

---

### 0.15 复杂度证据化（把复杂写成“可验证的代价与收益”）

> 规则：每个模块必须写清楚 **风险 → 机制 → 验收 → 不做代价 → 依赖 → 运行成本**，并标注“必要复杂度 / 证据不足复杂度”。

下面给出 **模块证据卡（v3.9.7 增补）**。你后续每次新增模块，都按同一模板补齐；否则默认视为“为了复杂而复杂”。

#### 0.15.1 S1 时间+快照契约（Time & Snapshot Contract）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 未来函数、跨源延迟造成信号不可复现；回测/实盘一致性破产 |
| 机制 | watermark/asof、快照版本、缺失字段显式标注；触发 S3 降级 |
| 验收 | ① 同一交易日 replay 一致；② 注入延迟/缺失触发降级且可观测；③ 不同采样频率下不越过 asof |
| 不做代价 | 任何回测收益都无法证明；上线后亏损无法定位责任链 |
| 依赖 | DataManager（S8），统一时间源，外部数据采集器 |
| 运行成本 | 增加缓存/存储与校验开销；但远小于排查“未来函数”的人力成本 |

#### 0.15.2 S2 数据契约模板（Data Contract Template）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 字段漂移、同名不同义、缺失字段静默吞掉导致策略行为突变 |
| 机制 | schema_version + required/optional + nullability + units + timezone；入库/出库校验 |
| 验收 | 合同测试：字段新增/删除/单位变化必须显式版本变更；旧版本数据仍可被读取（或明确迁移） |
| 不做代价 | 规模化后“只要一改字段就全系统炸”；回测与实盘口径长期漂移 |
| 依赖 | 因子治理（S4）、哈希规范（S9）、验收测试（S6） |
| 运行成本 | 多一次校验与元数据维护；但换来可控迭代与可追责变更 |

#### 0.15.3 S3 预算与降级策略（Budget & Degrade Policy）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 数据缺失/延迟时系统“假正常”；或者直接停摆，错过风险控制窗口 |
| 机制 | degrade_level（0/1/2…）+ 明确 fallback：停信号/只做风控/只做减仓；所有降级可观测 |
| 验收 | 强制注入异常：网络抖动、交易所限流、外部数据缺失；验证降级路径与恢复路径 |
| 不做代价 | 实盘最常见的不是策略问题，而是“数据/执行不稳定”引发的非线性损失 |
| 依赖 | S1 时间快照、S6 验收、监控告警 |
| 运行成本 | 需要更多状态与指标，但能显著降低事故概率与排查时间 |

#### 0.15.4 S4 因子治理（Factor Governance）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 因子实现/参数/归一化方式悄悄变化，导致模型漂移与不可复现 |
| 机制 | 因子注册表（name+version+hash+deps）、变更审计、冻结窗口 |
| 验收 | ① 因子 hash 稳定；② 变更必须 bump 版本并更新影响面；③ 回测/实盘同一版本因子输出一致 |
| 不做代价 | 模型训练与线上信号无法对齐，最终退化成“玄学调参” |
| 依赖 | 数据契约（S2）、哈希规范（S9）、replay（S10） |
| 运行成本 | 维护注册表与版本；但这是规模化研究的前提 |

#### 0.15.5 S5 对齐与归因 + Daily Replay（Alignment & Attribution）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 无法回答“赚/亏是哪个策略、哪个因子、哪个执行延迟导致的” |
| 机制 | trace_id 链路贯穿（signal→order→fill→pnl）；每日 replay 输出差异报告 |
| 验收 | ① 任意一笔成交能追溯到生成它的信号与当时快照；② daily replay 差异可解释（≤阈值） |
| 不做代价 | 策略迭代无法收敛；事故复盘无证据链 |
| 依赖 | S1/S8/S9/S10；执行回写（Hummingbot） |
| 运行成本 | 存储与计算额外开销；但这就是“可运营”的成本 |

#### 0.15.6 S6 验收测试（Acceptance Tests）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 方案写得再好，没有硬门槛就会在实现阶段被“偷工减料” |
| 机制 | DoD：一致性、降级、replay、回写、幂等、性能门槛 |
| 验收 | CI 必跑；出具报告（pass/fail + 差异解释） |
| 不做代价 | 后期 bug 与一致性问题指数级增长 |
| 依赖 | 全模块 |
| 运行成本 | CI 时长上升；但可通过分层（快测/慢测/夜间测）控制 |

#### 0.15.7 S7 物理边界隔离（研究/执行/密钥）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 密钥泄露、研究代码误触发实盘、执行层被研究层污染 |
| 机制 | 物理隔离（进程/权限）、密钥只在执行侧可见、只读数据出口 |
| 验收 | ① 研究侧无法读取密钥；② 研究侧无法直接调用下单 API；③ 执行侧重启不影响研究数据一致性 |
| 不做代价 | 安全与资金风险，且一旦发生通常是“不可逆” |
| 依赖 | 部署规范、权限体系 |
| 运行成本 | 运维复杂度略增；但属于安全基线 |

#### 0.15.8 S8 DataManager 唯一入口（单一事实源）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 多入口多缓存导致数据口径分裂；debug 成本爆炸 |
| 机制 | 所有读取/写入走 DataManager；缓存策略集中；统一 key 与 TTL |
| 验收 | 代码扫描：禁止绕过 DataManager；运行时 metrics：cache hit/miss、延迟、回源次数 |
| 不做代价 | “一半时间在找数据到底从哪来的” |
| 依赖 | S2 数据契约；S9 key/hash |
| 运行成本 | 集中化后组件更清晰，长期反而降低成本 |

#### 0.15.9 S9 Canonical Hashing 规范（统一 key / 去重 / 对齐）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 去重失败导致重复信号/重复订单；对齐失败导致归因失真 |
| 机制 | canonical serialize + stable hash；用于数据行、因子、信号、订单事件 |
| 验收 | hash 稳定性测试；跨语言/跨进程一致；升级必须版本化 |
| 不做代价 | 幂等/归因/重放都无法可靠实现 |
| 依赖 | S4 因子治理；S10 replay |
| 运行成本 | 计算开销极小；收益巨大 |

#### 0.15.10 S10 Replay 确定性保障（可复现 = 可运营）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）** |
| 风险 | 线上问题无法复现；策略迭代变成“试错赌博” |
| 机制 | 固定随机种子/版本锁定/数据快照；daily replay 差异报告 |
| 验收 | 同输入同版本输出严格一致；差异必须可解释且在阈值内 |
| 不做代价 | 系统永远停留在“个人项目”水平，无法规模化 |
| 依赖 | S1/S2/S4/S9 |
| 运行成本 | 存储与计算增加；可通过分层 replay（抽样/全量）控制 |

#### 0.15.11 执行引擎（Hummingbot 集成）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（必须保留）**（但“多交易所/多账户编排”属于**待证据复杂度**） |
| 风险 | 下单/撤单/部分成交/重连不一致导致资金损失；回测无法逼近实盘 |
| 机制 | 订单生命周期事件化（order/fill 回写）；幂等键；失败重试与限流；与 S1/S5 对齐 |
| 验收 | ① 模拟撮合/沙盒回放；② 断线重连/重复事件不重复下单；③ 延迟/滑点统计与阈值 |
| 不做代价 | “策略再好也赚不到钱”，而且事故通常不可控 |
| 依赖 | 交易所连接、密钥安全（S7）、监控告警 |
| 运行成本 | 需要维护连接与适配器；可先单交易所单账户 MVP，逐步扩展 |

#### 0.15.12 观测与运维（Observability & Ops）

| 项 | 内容 |
|---|---|
| 必要复杂度 | **必要复杂度（保留，但可分阶段）** |
| 风险 | 不知道系统是否降级/是否漏单/是否数据延迟；事故发现晚 |
| 机制 | 关键 SLO：数据延迟、降级等级、订单失败率、replay 差异；告警与仪表盘 |
| 验收 | ① 指标齐全且有报警阈值；② 事故演练（限流/断网/数据缺失）能被捕捉 |
| 不做代价 | 实盘运行等同“盲飞” |
| 依赖 | 日志/指标组件、CI 报告 |
| 运行成本 | 组件与存储成本；MVP 可先做最小集（核心指标 + 关键报警） |

---

### 0.16 合理决策：如何保证复杂度不是乱加的（不砍功能，但把“高配”变成可控开关）

1) **先把“必要复杂度”过门槛**：只要涉及一致性、时间点数据、回写归因、幂等重放、安全隔离，这些复杂度都是“买保险”，不能省。  
2) **把“证据不足复杂度”全部开关化**：默认 off，只有在满足证据条件（压测/成本收益/真实需求）后才开启。  
3) **给每个开关写清楚三件事**：开启条件（证据）、回退路径（降级）、上线影响面（指标/告警）。  
4) **扩展路线建议（不砍，分阶段）**：  
   - 阶段 A：单交易所 + 单账户 + 单策略族（把 S1-S10+回写归因跑通）  
   - 阶段 B：增加策略族/参数扫描（证明研究效率提升）  
   - 阶段 C：再做多交易所/多账户/更复杂的执行路由（用真实收益与运维成本证明它值得）


## 1. 项目概述

> ✅ **MVP包含** - 此节为核心设计原则，MVP必须遵守。

### 1.1 核心原则

> **"成熟度和稳定性优先，不以系统复杂度为基准"**

### 1.2 技术选型

| 层级 | 组件 | 版本 | 说明 |
|------|------|------|------|
| **数据层** | MultiSourceDataManager | 自建 | 统一管理数据源，支持快照存储 |
| **研究层** | Microsoft Qlib | 0.9.7 | **仅用于研究/回测** (见 Section 0.0.4) |
| **生产信号层** | FactorEngine | 自建 | 轻量级因子计算，**不依赖Qlib** |
| **回测层** | CryptoPerpetualBacktest | 自建 | 资金费率模拟、爆仓检测 |
| **执行层** | Hummingbot | 2.11.0 | 15k stars，企业级执行引擎 |
| **风控层** | RiskManager + PositionManager | 自建 | 多层风控、智能仓位分配 |

> **关键澄清**: Qlib 是研究工具，不是生产系统核心。MVP生产管道使用自建的 FactorEngine，不依赖 Qlib。详见 Section 0.0.4。

### 1.3 源码位置

```
/home/user/
├── qlib/                    # 用户项目仓库 (AlgVex)
│   └── algvex/              # AlgVex 核心代码
├── hummingbot/              # Hummingbot v2.11.0 源码
└── microsoft-qlib/          # Microsoft Qlib v0.9.7 源码
```

---

## 2. 系统架构

> ✅ **MVP包含** - 双链路架构为核心设计，MVP生产链路只用FactorEngine(11因子)，研究链路可后期扩展。

### 2.1 架构图 (生产/研究双链路)

> 生产链路不依赖Qlib，研究链路独立。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              algvex.com                                     │
│                            (Cloudflare CDN)                                 │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                         ┌────────▼────────┐
                         │      Nginx      │
                         │   (反向代理)    │
                         └────────┬────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
┌───────▼───────┐        ┌───────▼───────┐        ┌───────▼───────┐
│   Frontend    │        │  Backend API  │        │   WebSocket   │
│    (React)    │        │   (FastAPI)   │        │   (实时推送)  │
│   Port:3000   │        │   Port:8000   │        │   Port:8001   │
└───────────────┘        └───────┬───────┘        └───────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼───────┐       ┌───────▼───────┐       ┌───────▼───────┐
│    Celery     │       │     Redis     │       │  PostgreSQL   │
│   (任务队列)  │       │  (缓存/消息)  │       │ +TimescaleDB  │
└───────┬───────┘       └───────────────┘       └───────────────┘
        │
        │  ┌──────────────────────────────────────────────────────────┐
        │  │                    AlgVex Core Engine                    │
        │  ├──────────────────────────────────────────────────────────┤
        └──▶                                                          │
           │                                                          │
           │  ╔══════════════════════════════════════════════════╗   │
           │  ║            生产链路 (Production)                 ║   │
           │  ║            ❌ 不依赖 Qlib                         ║   │
           │  ╠══════════════════════════════════════════════════╣   │
           │  ║                                                  ║   │
           │  ║  ┌────────────┐  ┌────────────┐  ┌────────────┐ ║   │
           │  ║  │DataService │→│FactorEngine│→│ ModelRunner│ ║   │
           │  ║  │ (接口)     │  │ (MVP-11)   │  │(导出权重) │ ║   │
           │  ║  └────────────┘  └─────┬──────┘  └────────────┘ ║   │
           │  ║                        │                        ║   │
           │  ║                        ▼                        ║   │
           │  ║  ┌────────────┐  ┌────────────┐  ┌────────────┐ ║   │
           │  ║  │ RiskManager│←│ PositionMgr│←│SignalGen   │ ║   │
           │  ║  └─────┬──────┘  └────────────┘  └────────────┘ ║   │
           │  ║        ▼                                        ║   │
           │  ║  ┌─────────────────────────────────────────┐   ║   │
           │  ║  │         执行层 (Hummingbot)             │   ║   │
           │  ║  └─────────────────────────────────────────┘   ║   │
           │  ╚══════════════════════════════════════════════════╝   │
           │                                                          │
           │  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐      │
           │  │           研究链路 (Research)                 │      │
           │  │           ✅ 使用 Qlib                        │      │
           │  ├ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤      │
           │  │                                               │      │
           │  │ DataManager → QlibAdapter → Alpha180/201 →   │      │
           │  │ QlibDataset → Trainer → ExportModelArtifact  │      │
           │  │                    ↓                          │      │
           │  │              models/exported/                 │      │
           │  │              (供生产链路使用)                  │      │
           │  │                                               │      │
           │  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘      │
           └──────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MVP 外部数据源 (仅3个)                            │
├─────────────────────────┬─────────────────────┬────────────────────────┤
│ 币安 klines_5m          │ 币安 open_interest  │ 币安 funding_rate      │
│ (K线数据)               │ (持仓量)            │ (资金费率)             │
└─────────────────────────┴─────────────────────┴────────────────────────┘
```

### 2.2 统一数据管理架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        统一数据管理器 (DataManager)                          │
│                     整个系统的唯一数据入口和出口                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    1. 数据采集层 (Collectors)                        │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │ Binance  │ │ Deribit  │ │DefiLlama │ │Sentiment │ │  Macro   │  │   │
│  │  │Collector │ │Collector │ │Collector │ │Collector │ │Collector │  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│  │       │            │            │            │            │         │   │
│  │       └────────────┴────────────┼────────────┴────────────┘         │   │
│  │                                 ▼                                    │   │
│  │                    ┌─────────────────────┐                          │   │
│  │                    │  统一数据格式转换    │                          │   │
│  │                    │  (Qlib MultiIndex)  │                          │   │
│  │                    └──────────┬──────────┘                          │   │
│  └───────────────────────────────┼──────────────────────────────────────┘   │
│                                  ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    2. 数据存储层 (Storage)                           │   │
│  │                                                                      │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │   │
│  │   │ TimescaleDB │    │   Redis     │    │  Qlib本地   │             │   │
│  │   │  (历史数据)  │    │ (实时缓存)  │    │  (特征文件) │             │   │
│  │   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘             │   │
│  │          └──────────────────┼──────────────────┘                     │   │
│  │                             ▼                                        │   │
│  └─────────────────────────────┼────────────────────────────────────────┘   │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    3. 数据服务层 (DataService)                       │   │
│  │                    统一API，供所有下游模块调用                         │   │
│  │                                                                      │   │
│  │   get_historical()  get_realtime()  get_features()  get_labels()    │   │
│  │                                                                      │   │
│  └───────────┬─────────────────┬─────────────────┬──────────────────────┘   │
│              │                 │                 │                          │
└──────────────┼─────────────────┼─────────────────┼──────────────────────────┘
               ▼                 ▼                 ▼
     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
     │    因子计算      │ │     回测引擎    │ │    实盘执行      │
     │  (FactorEngine) │ │   (Backtest)    │ │  (LiveTrader)   │
     └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 2.3 数据流详解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              完整数据流                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【研究/回测模式】                                                           │
│                                                                             │
│   外部API ──→ Collector ──→ Storage ──→ DataService.get_historical()       │
│                                              │                              │
│                                              ▼                              │
│                                       FactorEngine (因子计算)               │
│                                              │                              │
│                                              ▼                              │
│                                       Qlib Dataset (训练数据)               │
│                                              │                              │
│                                              ▼                              │
│                                       ML Model (训练/预测)                  │
│                                              │                              │
│                                              ▼                              │
│                                       Backtest (回测验证)                   │
│                                              │                              │
│                                              ▼                              │
│                                       Report (回测报告)                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【实盘模式】                                                                │
│                                                                             │
│   外部API ──→ Collector ──→ Redis ──→ DataService.get_realtime()           │
│                (WebSocket)              │                                   │
│                                         ▼                                   │
│                                  FactorEngine (实时因子)                    │
│                                         │                                   │
│                                         ▼                                   │
│                                  ML Model (实时预测)                        │
│                                         │                                   │
│                                         ▼                                   │
│                                  SignalGenerator (信号生成)                 │
│                                         │                                   │
│                                         ▼                                   │
│                                  RiskManager (风控检查)                     │
│                                         │                                   │
│                                         ▼                                   │
│                                  Hummingbot (订单执行)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 数据层

> ✅ **MVP包含** - 但仅需实现**A档数据源**(币安klines_5m, open_interest_5m, funding_rate)。
> ⏸️ **MVP不包含** - B档(Deribit期权, Google Trends等)和C档(自建落盘)延后到Phase 2。

> **原则**: 仅使用免费数据源；历史可得性分A/B/C三档，B/C档需自建落盘才能形成长期可回放历史。

### 3.1 数据可得性分级 (关键！)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   数据可得性三级分类 (决定回测可信度)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【A档】可稳定回溯多年 - 直接可用于长期回测                                    │
│  ├─ 币安 OHLCV (K线)          历史窗口: 无限          口径稳定: ★★★         │
│  ├─ Yahoo Finance (宏观)      历史窗口: 多年          口径稳定: ★★★         │
│  ├─ FRED (利率/经济)          历史窗口: 多年          口径稳定: ★★★         │
│  └─ DefiLlama (TVL/稳定币)    历史窗口: 2年+          口径稳定: ★★☆         │
│                                                                             │
│  【B档】历史窗口有限 - 需自建落盘积累长期数据                                  │
│  ├─ 币安 OI/多空比/大户比     历史窗口: 30-90天       口径稳定: ★★☆         │
│  ├─ 币安 Taker Buy/Sell       历史窗口: 500条/请求    口径稳定: ★★★         │
│  ├─ Deribit 期权数据          历史窗口: 有限          口径稳定: ★★☆         │
│  └─ Alternative Fear&Greed    历史窗口: 多年          口径稳定: ★★★         │
│                                                                             │
│  【C档】必须自建落盘 - 无官方历史API或不稳定                                   │
│  ├─ Google Trends             历史窗口: 抓取依赖      口径稳定: ★☆☆         │
│  ├─ 实时WebSocket数据         历史窗口: 无            口径稳定: N/A          │
│  └─ 跨交易所价差 (需自行计算)  历史窗口: 取决于源     口径稳定: ★★☆         │
│                                                                             │
│  ⚠️ 重要: B/C档数据必须从现在开始自建落盘，才能形成长期可回放的历史！          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 数据源详细配置

| 类别 | 数据源 | 字段 | 频率 | 延迟 | 历史窗口 | 口径稳定性 | 补洞策略 |
|------|--------|------|------|------|----------|-----------|----------|
| **交易所基础** | 币安永续 | $open, $close, $high, $low, $volume | 1m-1d | bar收盘 | 无限 (A档) | High | backfill |
| | 币安永续 | $funding_rate | 8h | 结算后 | 多年 (A档) | High | backfill |
| | 币安永续 | $open_interest | 5m | ~5min | 30天 (B档) | Medium | 需落盘 |
| | 币安永续 | $long_short_ratio, $top_long_short_ratio | 5m | ~5min | 30天 (B档) | Medium | 需落盘 |
| | 币安永续 | $taker_buy_volume, $taker_sell_volume | 1m-1d | bar收盘 | 500条 (B档) | High | 需落盘 |
| **期权** | Deribit | $dvol_index (BTC/ETH) | 1h | ~1min | 有限 (B档) | Medium | 需落盘 |
| | Deribit | $iv_atm, $put_call_ratio, $max_pain | 1h | ~5min | 有限 (B档) | Medium | 需落盘 |
| **衍生品结构** | 多交易所 | $basis (spot - perp) | 1m | bar收盘 | 需计算 (C档) | Medium | 需落盘 |
| | 币安 | $insurance_fund | 1d | T+1 | 多年 (A档) | High | backfill |
| **链上** | DefiLlama | $stablecoin_supply, $stablecoin_change | 1d | ~1h | 2年+ (A档) | Medium | backfill |
| | DefiLlama | $defi_tvl, $tvl_change | 1d | ~1h | 2年+ (A档) | Medium | backfill |
| **情绪** | Alternative | $fear_greed (0-100) | 1d | ~1h | 多年 (A档) | High | backfill |
| | Google | $btc_trend, $crypto_trend | 1d | ~1d | 抓取 (C档) | Low | 需落盘 |
| **宏观** | Yahoo/FRED | $dxy, $us10y, $us02y, $gold, $spx, $vix | 1d | ~1h | 多年 (A档) | High | backfill |

### 3.3 统一数据管理器 (DataManager)

> **核心设计**: DataManager 是整个系统的唯一数据入口，所有模块通过它获取数据，确保数据一致性。

```python
from algvex.core.data import DataManager

# 初始化数据管理器（整个系统只需初始化一次）
dm = DataManager(
    db_url="postgresql://localhost/algvex",
    redis_url="redis://localhost:6379",
    qlib_path="~/.algvex/qlib_data",
)

# ==================== 研究/回测模式 ====================

# 1. 下载/更新历史数据（首次或定期更新）
dm.update_historical(
    start_date="2023-01-01",
    end_date="2024-12-31",
    symbols=["BTCUSDT", "ETHUSDT"],
)

# 2. 获取历史数据（因子计算、回测使用）
df = dm.get_historical(
    symbols=["BTCUSDT", "ETHUSDT"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    freq="1h",
    fields=["$open", "$close", "$funding_rate", "$dvol", "$fear_greed"],
)

# 3. 获取 Qlib Dataset（模型训练使用）
dataset = dm.get_qlib_dataset(
    symbols=["BTCUSDT", "ETHUSDT"],
    segments={
        "train": ("2023-01-01", "2024-06-30"),
        "valid": ("2024-07-01", "2024-09-30"),
        "test": ("2024-10-01", "2024-12-31"),
    },
    handler="CryptoAlpha180",  # 180因子处理器
)

# ==================== 实盘模式 ====================

# 4. 启动实时数据流
await dm.start_realtime(symbols=["BTCUSDT", "ETHUSDT"])

# 5. 获取实时数据（实盘信号生成使用）
realtime_df = dm.get_realtime(
    symbols=["BTCUSDT"],
    lookback="24h",  # 最近24小时数据
)

# 6. 获取最新因子值（实盘预测使用）
latest_features = dm.get_latest_features(
    symbols=["BTCUSDT", "ETHUSDT"],
)
```

### 3.4 数据服务接口 (DataService API)

| 方法 | 用途 | 调用方 |
|------|------|--------|
| `update_historical()` | 下载/更新历史数据 | 定时任务 |
| `get_historical()` | 获取历史数据 | 因子计算、回测 |
| `get_qlib_dataset()` | 获取Qlib格式数据集 | 模型训练 |
| `start_realtime()` | 启动实时数据流 | 实盘服务 |
| `get_realtime()` | 获取实时数据 | 实时因子 |
| `get_latest_features()` | 获取最新因子值 | 实时预测 |

### 3.5 数据采集器接口 (Collector Interface)

```python
from abc import ABC, abstractmethod

class BaseCollector(ABC):
    """所有数据采集器的基类"""

    @abstractmethod
    async def fetch_historical(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """获取历史数据"""
        pass

    @abstractmethod
    async def subscribe_realtime(self, symbol: str, callback: Callable) -> None:
        """订阅实时数据"""
        pass

    @abstractmethod
    def to_qlib_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换为Qlib格式"""
        pass

# 各数据源采集器继承统一接口
class BinanceCollector(BaseCollector): ...
class DeribitCollector(BaseCollector): ...
class DefiLlamaCollector(BaseCollector): ...
class SentimentCollector(BaseCollector): ...
class MacroCollector(BaseCollector): ...
```

### 3.6 Qlib 数据格式

```
MultiIndex: (datetime, instrument)
字段前缀: $ (如 $open, $close, $funding_rate)
时区: 全系统强制 UTC (尤其 funding 结算 0/8/16 UTC)
日历: 加密货币 24/7 (需自定义，不使用股票交易日)

                           $open    $close  $funding_rate  $cvd      $dvol  $fear_greed
datetime    instrument
2024-01-01  btcusdt      42000.0  42100.0        0.0001   15000.0    55.2          65
            ethusdt       2200.0   2210.0        0.00015   8000.0    62.1          65
```

### 3.7 数据事实源与版本化 (研究可复现的关键)

> **核心问题**: 多存储层（TimescaleDB + Qlib文件 + Redis）如果不定义事实源，将导致数据不一致和实验不可复现。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        数据层级与事实源定义                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  【L0 原始数据层】 事实源 = TimescaleDB                                       │
│  ├─ 特点: 不可变、可追溯、带时间戳                                            │
│  ├─ 存储: 原始API响应 (JSON/Parquet)                                        │
│  └─ 规则: 一旦写入不可修改，只能追加                                          │
│                                                                             │
│  【L1 派生特征层】 Qlib 本地文件 / Parquet                                    │
│  ├─ 特点: 由 L0 计算生成，必须版本化                                         │
│  ├─ 版本化: feature_set_id = hash(因子代码 + 参数 + L0快照ID)                │
│  └─ 规则: 每次重新计算生成新版本，旧版本保留                                   │
│                                                                             │
│  【L2 实时缓存层】 Redis                                                      │
│  ├─ 特点: 易失性，仅用于实盘低延迟访问                                        │
│  ├─ 规则: 不作为可复现实验的数据来源                                          │
│  └─ 刷新: 定期从 L0/L1 同步                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
# 数据快照与血缘记录 (每次训练/回测必须记录)
@dataclass
class DataSnapshot:
    """数据快照 - 确保实验可复现"""
    snapshot_id: str                    # 唯一ID
    created_at: datetime                # 创建时间

    # 数据范围
    symbols: List[str]                  # 标的列表
    start_date: str                     # 开始日期
    end_date: str                       # 结束日期

    # 各数据源截至时间与版本
    source_versions: Dict[str, str]     # {"binance_ohlcv": "2024-12-21T00:00:00Z", ...}

    # 延迟配置版本
    delay_config_hash: str              # PUBLICATION_DELAYS 配置的 hash

    # 补洞策略版本
    backfill_strategy_hash: str         # 补洞逻辑的 hash

@dataclass
class ExperimentRecord:
    """实验记录 - 完整血缘链"""
    experiment_id: str

    # 数据血缘
    data_snapshot_id: str               # 使用的数据快照
    feature_set_id: str                 # 因子版本 (代码hash + 参数hash)

    # 模型血缘
    model_config_hash: str              # 模型配置 hash
    random_seed: int                    # 随机种子

    # 结果
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]

    # 可追溯
    git_commit: str                     # 代码版本
    created_at: datetime
```

---

## 4. 信号层

> ✅ **MVP包含** - 但仅实现**MVP-11因子**(见Section 0.0.3)，使用自建FactorEngine。
> ⏸️ **MVP不包含** - 180因子研究库(含Qlib Alpha180)延后到研究阶段，不进入生产链路。

### 4.1 因子体系 (AlgVex 自建 201个，全部基于免费数据)

> **重要说明**: Qlib 本身不提供加密货币因子。以下 201 个因子均为 AlgVex 基于 Qlib 框架自建，专为永续合约设计。**数据源全部免费，但历史可得性分A/B/C三档（见 Section 3.1），B/C档需自建落盘才能形成长期可回放历史。**
>
> **因子构成**: 180个核心因子 + 21个P1扩展因子 (L2深度8个 + 清算5个 + Basis8个)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   AlgVex 自建因子体系 (180个，仅免费数据)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ═══════════════════════ 一、基础价量因子 (50个) ═══════════════════════    │
│                                                                             │
│  [价格动量] 20个                    [波动率] 15个                           │
│  ├─ return_1h/4h/24h/7d            ├─ volatility_12h/24h/7d               │
│  ├─ mom_12h/24h/72h                ├─ atr_24h/7d                          │
│  ├─ ma_cross_12_24/24_72           ├─ skewness/kurtosis                   │
│  ├─ price_position_52w             ├─ volatility_ratio                    │
│  └─ breakout_20d/60d               └─ realized_vol / rv_ratio             │
│                                                                             │
│  [成交量] 15个                                                              │
│  ├─ volume_ratio_12h/24h/7d        ├─ volume_trend                        │
│  ├─ price_volume_corr              ├─ obv / obv_change                    │
│  ├─ volume_breakout                └─ relative_volume                     │
│                                                                             │
│  ═══════════════════ 二、永续合约专用因子 (45个) ═══════════════════════    │
│                                                                             │
│  [资金费率] 12个 ★永续专用                                                  │
│  ├─ funding_rate                   ├─ funding_rate_ma_8h/24h             │
│  ├─ funding_premium                ├─ funding_momentum                   │
│  ├─ funding_zscore                 ├─ funding_extreme (>0.1% 或 <-0.05%) │
│  ├─ funding_cumsum_24h/7d          └─ funding_reversal_signal            │
│                                                                             │
│  [持仓量 OI] 12个 ★永续专用                                                 │
│  ├─ oi_change_1h/4h/24h            ├─ oi_volume_ratio                    │
│  ├─ oi_price_divergence            ├─ oi_momentum                        │
│  ├─ oi_zscore                      ├─ oi_concentration                   │
│  └─ oi_funding_interaction         └─ oi_breakout                        │
│                                                                             │
│  [多空博弈+CVD] 21个 ★永续专用                                              │
│  ├─ long_short_ratio               ├─ top_trader_long_short_ratio        │
│  ├─ top_trader_position_ratio      ├─ ls_momentum                        │
│  ├─ ls_extreme                     ├─ ls_reversal_signal                 │
│  ├─ taker_buy_volume               ├─ taker_sell_volume                  │
│  ├─ taker_buy_sell_ratio           ├─ taker_delta                        │
│  ├─ cvd (累计成交量差)              ├─ cvd_change_1h/4h/24h              │
│  ├─ cvd_price_divergence           ├─ cvd_momentum                       │
│  └─ cvd_zscore                     └─ net_taker_flow                     │
│                                                                             │
│  ══════════════════ 三、期权/波动率因子 (20个) ═════════════════════════    │
│                                                                             │
│  [隐含波动率] 10个 (Deribit 免费API)                                        │
│  ├─ dvol_btc / dvol_eth            ├─ dvol_change_24h                    │
│  ├─ iv_atm                         ├─ iv_skew (put vs call)              │
│  ├─ iv_term_structure              ├─ iv_rv_spread (IV - RV)             │
│  ├─ iv_percentile                  └─ vol_risk_premium                   │
│                                                                             │
│  [期权持仓] 10个 (Deribit 免费API)                                          │
│  ├─ put_call_ratio                 ├─ put_call_oi_ratio                  │
│  ├─ max_pain                       ├─ max_pain_distance                  │
│  ├─ gamma_exposure                 ├─ option_volume_spike                │
│  ├─ large_put_oi                   └─ large_call_oi                      │
│                                                                             │
│  ══════════════════ 四、衍生品结构因子 (15个) ══════════════════════════    │
│                                                                             │
│  [基差] 8个                                                                 │
│  ├─ basis (spot - perp)            ├─ basis_percentage                   │
│  ├─ basis_ma_24h                   ├─ basis_zscore                       │
│  ├─ basis_momentum                 ├─ annualized_basis                   │
│  ├─ basis_funding_corr             └─ basis_extreme                      │
│                                                                             │
│  [市场结构] 7个                                                             │
│  ├─ cross_exchange_spread          ├─ binance_premium                    │
│  ├─ insurance_fund_change          ├─ market_depth_ratio                 │
│  ├─ exchange_dominance             └─ volume_concentration               │
│                                                                             │
│  ═══════════════════ 五、链上因子 (10个，DefiLlama免费) ═══════════════    │
│                                                                             │
│  [稳定币] 5个 (DefiLlama 免费)                                              │
│  ├─ stablecoin_supply              ├─ stablecoin_supply_change_7d        │
│  ├─ stablecoin_dominance           ├─ usdt_usdc_ratio                    │
│  └─ stablecoin_momentum                                                    │
│                                                                             │
│  [DeFi TVL] 5个 (DefiLlama 免费)                                            │
│  ├─ defi_tvl_total                 ├─ defi_tvl_change_7d                 │
│  ├─ eth_tvl_dominance              ├─ tvl_mcap_ratio                     │
│  └─ tvl_momentum                                                           │
│                                                                             │
│  ═══════════════════ 六、情绪因子 (10个) ═══════════════════════════════    │
│                                                                             │
│  [Fear & Greed] 5个 (alternative.me 免费)                                   │
│  ├─ fear_greed_index               ├─ fear_greed_ma_7d                   │
│  ├─ fear_greed_momentum            ├─ fear_greed_extreme                 │
│  └─ fear_greed_reversal                                                    │
│                                                                             │
│  [Google Trends] 5个 (免费)                                                 │
│  ├─ btc_search_trend               ├─ crypto_search_trend                │
│  ├─ search_trend_change            ├─ search_spike_detection             │
│  └─ search_price_divergence                                                │
│                                                                             │
│  ═══════════════════ 七、宏观关联因子 (15个) ═══════════════════════════    │
│                                                                             │
│  [美元/利率] 8个 (Yahoo/FRED 免费)                                          │
│  ├─ dxy_index                      ├─ dxy_change_5d                      │
│  ├─ btc_dxy_corr_30d               ├─ us10y_yield                        │
│  ├─ us02y_yield                    ├─ yield_curve (10y - 2y)             │
│  ├─ rate_sensitivity               └─ real_yield                         │
│                                                                             │
│  [风险资产] 7个 (Yahoo 免费)                                                │
│  ├─ spx_return_5d                  ├─ btc_spx_corr_30d                   │
│  ├─ nasdaq_return_5d               ├─ vix_index                          │
│  ├─ vix_change                     ├─ gold_return_5d                     │
│  └─ btc_gold_corr_30d                                                      │
│                                                                             │
│  ═══════════════════ 八、复合/ML因子 (15个) ════════════════════════════    │
│                                                                             │
│  [复合因子] 15个                                                            │
│  ├─ alpha_momentum                 ├─ alpha_mean_reversion               │
│  ├─ alpha_orderflow                ├─ alpha_sentiment                    │
│  ├─ alpha_volatility               ├─ alpha_structure                    │
│  ├─ risk_on_off_score              ├─ regime_indicator                   │
│  ├─ trend_strength                 ├─ mean_reversion_score               │
│  ├─ breakout_probability           ├─ crash_risk_indicator               │
│  ├─ momentum_quality               ├─ factor_momentum                    │
│  └─ ml_ensemble_score                                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│               核心因子: 180 个 (全部免费，可直接获取历史数据)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ═══════════════════ 九、P1扩展因子 (21个，需自建落盘) ═══════════════════  │
│                                                                             │
│  [L2深度因子] 8个 (Step 9, C档)                                             │
│  ├─ bid_ask_spread                ├─ order_book_imbalance                 │
│  ├─ depth_1pct_bid/ask            ├─ depth_slope_bid/ask                  │
│  └─ impact_cost_buy/sell                                                   │
│                                                                             │
│  [清算因子] 5个 (Step 10, B档)                                              │
│  ├─ liquidation_volume_long/short ├─ liquidation_imbalance                │
│  ├─ liquidation_spike             └─ liquidation_momentum                 │
│                                                                             │
│  [多交易所Basis] 8个 (Step 11, C档)                                         │
│  ├─ basis_binance/bybit/okx       ├─ basis_consensus                      │
│  ├─ basis_dispersion              ├─ cross_exchange_spread                │
│  ├─ price_discovery_leader        └─ arbitrage_pressure                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│               总计: 201 个因子 (180核心 + 21 P1扩展)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 因子分类统计

| 类别 | 因子数 | 数据来源 | 历史数据 | 备注 |
|------|--------|----------|----------|------|
| 基础价量 | 50 | 币安 REST | ✅ 多年 (A档) | 核心 |
| 永续合约专用 | 45 | 币安 REST | ✅ 30天+ (B档) | 核心 |
| 期权/波动率 | 20 | Deribit | ✅ 有历史 (B档) | 核心 |
| 衍生品结构 | 15 | 多交易所 | ✅ 可计算 (C档) | 核心 |
| 链上 | 10 | DefiLlama | ✅ 多年 (A档) | 核心 |
| 情绪 | 10 | Alternative/Google | ✅ 多年 (A/C档) | 核心 |
| 宏观关联 | 15 | Yahoo/FRED | ✅ 多年 (A档) | 核心 |
| 复合/ML | 15 | 自建 | ✅ 基于以上 | 核心 |
| **核心小计** | **180** | | | |
| ★ L2深度 | 8 | 币安 WebSocket | ⚠️ 需自建 (C档) | Step 9 |
| ★ 清算 | 5 | 币安 WebSocket | ⚠️ 需自建 (B档) | Step 10 |
| ★ 多交易所Basis | 8 | Binance/Bybit/OKX | ⚠️ 需自建 (C档) | Step 11 |
| **P1扩展小计** | **21** | | | |
| **总计** | **201** | **全部免费** | | |

### 4.3 ML模型配置

```yaml
model:
  type: LGBMModel  # 或 XGBoostModel, DNNModel
  lgbm_params:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 8
    num_leaves: 64
    feature_fraction: 0.8
    bagging_fraction: 0.8
    min_child_samples: 50
  training:
    loss: mse
    early_stopping_rounds: 50
    validation_split: 0.2
  feature_selection:
    method: importance
    top_k: 50
```

---

## 5. 回测层

> ✅ **MVP包含** - 核心回测引擎必须实现，支持资金费率、滑点、爆仓检测。
> 📝 **MVP精简** - 仅需单交易所(币安)、单时间框架(5m)、基础指标(Sharpe/MaxDD/胜率)。

### 5.1 CryptoPerpetualBacktest

```python
from algvex.core.backtest import BacktestConfig, CryptoPerpetualBacktest

config = BacktestConfig(
    initial_capital=100000.0,     # 初始资金 $100k
    leverage=3.0,                  # 默认杠杆 3x
    max_leverage=10.0,             # 最大杠杆 10x
    taker_fee=0.0004,              # Taker费率 0.04%
    maker_fee=0.0002,              # Maker费率 0.02%
    slippage=0.0001,               # 滑点 0.01%
    funding_rate_interval=8,       # 资金费率间隔 8小时
    liquidation_threshold=0.8,     # 爆仓阈值 80%保证金率
)

engine = CryptoPerpetualBacktest(config)
results = engine.run(signals, prices, funding_rates)
```

### 5.2 回测指标

| 收益指标 | 风险指标 | 交易指标 | 永续专用 |
|----------|----------|----------|----------|
| 总收益率 | 最大回撤 | 胜率 | 资金费用总额 |
| 年化收益 | 波动率 | 盈亏比 | 资金费用占比 |
| 夏普比率 | 索提诺比率 | 平均持仓时间 | 爆仓次数 |
| 卡尔玛比率 | VaR | 交易次数 | 保证金利用率 |

---

## 6. 执行层

> ✅ **MVP包含** - Hummingbot集成为核心，必须实现订单执行和三重屏障。
> 📝 **MVP精简** - 仅需PositionExecutor执行器，其他5种执行器(DCA/TWAP/Grid/Arbitrage/XEMM)延后。

### 6.1 Hummingbot v2.11.0 Strategy V2 框架

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hummingbot Strategy V2 架构                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [控制器层] ControllerConfigBase                                        │
│  ├─ controller_name: 策略名称                                           │
│  ├─ candles_config: K线数据配置                                         │
│  ├─ initial_positions: 初始持仓配置                                     │
│  └─ manual_kill_switch: 手动停止开关                                    │
│                                                                         │
│  [执行器层] 6种内置执行器                                                │
│  ├─ PositionExecutor: 单仓位执行 (三重屏障: 止损/止盈/时间限制)          │
│  ├─ DCAExecutor: 定投/分批建仓                                          │
│  ├─ TWAPExecutor: 时间加权平均价格执行                                   │
│  ├─ GridExecutor: 网格交易执行                                          │
│  ├─ ArbitrageExecutor: 跨交易所套利                                      │
│  └─ XEMMExecutor: 跨交易所做市                                          │
│                                                                         │
│  [回测层] strategy_v2/backtesting/                                      │
│  ├─ BacktestingEngineBase: 回测引擎基类                                  │
│  ├─ BacktestingDataProvider: 回测数据提供者                              │
│  └─ ExecutorSimulator: 执行器模拟器                                      │
│                                                                         │
│  [永续交易所] 12个 CEX/DEX                                               │
│  ├─ CEX: Binance, Bybit, OKX, KuCoin, Gate.io, Bitget, BitMart          │
│  └─ DEX: Hyperliquid, dYdX v4, Injective v2, Derive                     │
│                                                                         │
│  [数据源] 21个K线数据源                                                  │
│  ├─ 支持间隔: 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d    │
│  └─ 字段: timestamp, open, high, low, close, volume, quote_asset_volume │
│                                                                         │
│  [清算数据] liquidations_feed (风控增强)                                 │
│  └─ BinancePerpetualLiquidations: 实时市场清算事件                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 三重屏障风控 (Triple Barrier)

```python
from hummingbot.strategy_v2.executors.position_executor.data_types import (
    PositionExecutorConfig,
    TripleBarrierConfig,
)

config = PositionExecutorConfig(
    connector_name="binance_perpetual",
    trading_pair="BTC-USDT",
    side=TradeType.BUY,
    amount=Decimal("0.1"),
    triple_barrier_config=TripleBarrierConfig(
        stop_loss=Decimal("0.03"),          # 止损 3%
        take_profit=Decimal("0.06"),        # 止盈 6%
        time_limit=60 * 60 * 24,            # 时间限制 24小时
        stop_loss_order_type=OrderType.MARKET,
        take_profit_order_type=OrderType.LIMIT,
    )
)
```

### 6.3 HummingbotBridge 详细实现

> **核心职责**: 连接 AlgVex SignalGenerator 和 Hummingbot Connector，管理订单生命周期，状态同步

```python
# algvex/core/execution/hummingbot_bridge.py

from decimal import Decimal
from typing import Dict, Optional, List
from datetime import datetime
import asyncio

from hummingbot.connector.connector_base import ConnectorBase
from hummingbot.core.data_type.in_flight_order import InFlightOrder
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate

from algvex.production.signal.signal_generator import Signal
from algvex.shared.trace_serializer import DeterministicTraceSerializer


class HummingbotBridge:
    """
    AlgVex 与 Hummingbot 的桥接层

    核心功能:
    1. 信号 → 订单转换 (Signal → Order)
    2. 订单生命周期管理 (InFlightOrder)
    3. 状态同步 (Position Reconciliation)
    4. 事件追踪 (Order Events → Trace)

    ⚠️ 幂等性保障: 相同信号生成相同 client_order_id
    """

    def __init__(
        self,
        connector: ConnectorBase,
        trace_writer: Optional['TraceWriter'] = None,
        exchange: str = "binance_perpetual",
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        max_leverage: int = 10,
    ):
        self.connector = connector
        self.trace_writer = trace_writer
        self.serializer = DeterministicTraceSerializer()
        self.exchange = exchange
        self.testnet = testnet
        self.max_leverage = max_leverage

        # 订单跟踪 (幂等性关键)
        self._signal_to_order: Dict[str, str] = {}  # signal_id -> client_order_id
        self._order_to_signal: Dict[str, str] = {}  # client_order_id -> signal_id
        self._pending_signals: Dict[str, Signal] = {}

    async def connect(self):
        """连接交易所"""
        await self.connector.start()

    async def execute_signal(self, signal: Signal) -> Dict:
        """
        执行 AlgVex 信号

        ⚠️ 幂等性: 相同信号重复调用返回相同结果

        Args:
            signal: AlgVex 信号对象

        Returns:
            执行结果 dict
        """
        # 1. 生成幂等的 client_order_id
        client_order_id = self._generate_idempotent_order_id(signal)

        # 2. 检查是否已处理过 (幂等保护)
        if client_order_id in self.connector.in_flight_orders:
            existing_order = self.connector.in_flight_orders[client_order_id]
            return {
                "status": "duplicate",
                "client_order_id": client_order_id,
                "order_state": existing_order.current_state.name,
            }

        # 3. 转换为 Hummingbot OrderCandidate
        order_candidate = self._signal_to_order_candidate(signal, client_order_id)

        # 4. 预算检查 (使用 Hummingbot BudgetChecker)
        adjusted_candidate = self.connector.budget_checker.adjust_candidate(
            order_candidate, all_or_none=False
        )
        if adjusted_candidate.amount == Decimal("0"):
            return {
                "status": "rejected",
                "reason": "insufficient_funds",
                "signal_id": signal.signal_id,
            }

        # 5. 下单
        try:
            if signal.direction > 0:
                order_id = self.connector.buy(
                    trading_pair=signal.symbol,
                    amount=adjusted_candidate.amount,
                    order_type=OrderType.MARKET,
                    price=Decimal("0"),
                )
            else:
                order_id = self.connector.sell(
                    trading_pair=signal.symbol,
                    amount=adjusted_candidate.amount,
                    order_type=OrderType.MARKET,
                    price=Decimal("0"),
                )

            # 6. 记录映射
            self._signal_to_order[signal.signal_id] = order_id
            self._order_to_signal[order_id] = signal.signal_id
            self._pending_signals[order_id] = signal

            # 7. 写入 trace
            if self.trace_writer:
                self._write_order_trace(signal, order_id, "submitted")

            return {
                "status": "submitted",
                "client_order_id": order_id,
                "signal_id": signal.signal_id,
                "amount": str(adjusted_candidate.amount),
            }

        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "signal_id": signal.signal_id,
            }

    def _generate_idempotent_order_id(self, signal: Signal) -> str:
        """
        生成幂等的订单 ID

        基于信号内容 hash，相同信号生成相同 ID
        ⚠️ 这是防止重复下单的核心机制
        """
        content = {
            "symbol": signal.symbol,
            "direction": signal.direction,
            "bar_close_time": signal.bar_close_time.isoformat(),
            "final_signal": signal.final_signal,
        }
        hash_str = self.serializer.compute_hash(content)
        return f"algvex_{hash_str[:16]}"

    def _signal_to_order_candidate(
        self, signal: Signal, client_order_id: str
    ) -> OrderCandidate:
        """将 AlgVex Signal 转换为 Hummingbot OrderCandidate"""
        return OrderCandidate(
            trading_pair=signal.symbol,
            is_maker=False,  # Market order = taker
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY if signal.direction > 0 else TradeType.SELL,
            amount=Decimal(str(signal.quantity)),
            price=Decimal("0"),
        )

    def signal_to_order(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float = 0.02,
        leverage: int = 1
    ) -> OrderCandidate:
        """
        信号转订单 (含仓位计算)

        Args:
            signal: 信号对象
            capital: 总资金
            risk_per_trade: 单笔风险比例
            leverage: 杠杆倍数
        """
        # 计算仓位大小
        position_value = capital * risk_per_trade * leverage
        quantity = position_value / signal.price if signal.price > 0 else Decimal("0")

        return OrderCandidate(
            trading_pair=signal.symbol,
            is_maker=False,
            order_type=OrderType.MARKET,
            order_side=TradeType.BUY if signal.score > 0 else TradeType.SELL,
            amount=Decimal(str(quantity)),
            price=Decimal("0"),
        )

    async def execute_order(self, order: OrderCandidate) -> Dict:
        """执行订单"""
        # 预算检查
        adjusted = self.connector.budget_checker.adjust_candidate(order, all_or_none=False)
        if adjusted.amount == Decimal("0"):
            return {"status": "rejected", "reason": "insufficient_funds"}

        # 下单
        if order.order_side == TradeType.BUY:
            order_id = self.connector.buy(
                trading_pair=order.trading_pair,
                amount=adjusted.amount,
                order_type=order.order_type,
                price=order.price,
            )
        else:
            order_id = self.connector.sell(
                trading_pair=order.trading_pair,
                amount=adjusted.amount,
                order_type=order.order_type,
                price=order.price,
            )

        return {"status": "submitted", "order_id": order_id}

    def _write_order_trace(self, signal: Signal, order_id: str, status: str):
        """写入订单追踪"""
        trace = {
            "type": "order_event",
            "signal_id": signal.signal_id,
            "client_order_id": order_id,
            "status": status,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.trace_writer.write(trace)

    # ==================== 事件处理 ====================

    async def on_order_filled(self, event: 'OrderFilledEvent'):
        """处理订单成交事件"""
        order_id = event.order_id
        signal_id = self._order_to_signal.get(order_id)

        if signal_id and self.trace_writer:
            trace = {
                "type": "order_filled",
                "signal_id": signal_id,
                "client_order_id": order_id,
                "exchange_order_id": event.exchange_order_id,
                "price": str(event.price),
                "amount": str(event.amount),
                "trade_fee": str(event.trade_fee.flat_fees) if event.trade_fee else "0",
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            self.trace_writer.write(trace)

        # 清理
        if order_id in self._pending_signals:
            del self._pending_signals[order_id]

    async def on_order_cancelled(self, event: 'OrderCancelledEvent'):
        """处理订单取消事件"""
        order_id = event.order_id
        signal_id = self._order_to_signal.get(order_id)

        if signal_id and self.trace_writer:
            trace = {
                "type": "order_cancelled",
                "signal_id": signal_id,
                "client_order_id": order_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            self.trace_writer.write(trace)

    # ==================== 状态同步 ====================

    async def sync_positions(self) -> Dict:
        """与交易所同步仓位"""
        exchange_positions = {}

        for trading_pair in self.connector.trading_pairs:
            position = self.connector.get_position(trading_pair)
            if position:
                exchange_positions[trading_pair] = {
                    "amount": str(position.amount),
                    "entry_price": str(position.entry_price),
                    "leverage": position.leverage,
                    "unrealized_pnl": str(position.unrealized_pnl),
                }

        return exchange_positions

    async def reconcile(self) -> Dict:
        """
        对账: 比较本地状态和交易所状态

        用于检测状态不一致并告警
        """
        exchange_positions = await self.sync_positions()
        # TODO: 与本地 PositionManager 状态对比
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "exchange_positions": len(exchange_positions),
            "aligned": True,  # 简化版
        }
```

### 6.4 InFlightOrder 集成

> Hummingbot 的 `InFlightOrder` 是订单生命周期管理的核心，AlgVex 必须正确集成。

```python
# Hummingbot InFlightOrder 状态机
# hummingbot/core/data_type/in_flight_order.py

class InFlightOrder:
    """
    订单生命周期追踪

    状态流转:
    PENDING_CREATE → OPEN → FILLED / CANCELLED / FAILED
    """
    client_order_id: str          # 客户端订单ID (幂等键)
    exchange_order_id: Optional[str]
    trading_pair: str
    order_type: OrderType
    trade_type: TradeType
    price: Decimal
    amount: Decimal
    creation_timestamp: float
    current_state: OrderState     # PENDING, OPEN, FILLED, CANCELLED, FAILED

    # 状态转换方法
    def update_with_order_update(self, order_update: OrderUpdate)
    def update_with_trade_update(self, trade_update: TradeUpdate)
```

**AlgVex 与 InFlightOrder 的集成**:

```python
# algvex/core/execution/order_tracker.py

class AlgVexOrderTracker:
    """
    订单追踪器 - 集成 Hummingbot InFlightOrder

    功能:
    1. 订单状态查询
    2. 订单超时检测
    3. 断线恢复
    """

    def __init__(self, bridge: HummingbotBridge):
        self.bridge = bridge
        self._order_timeout = 60  # 60秒超时

    def get_order_status(self, client_order_id: str) -> Optional[Dict]:
        """获取订单状态"""
        order = self.bridge.connector.in_flight_orders.get(client_order_id)
        if order:
            return {
                "client_order_id": client_order_id,
                "exchange_order_id": order.exchange_order_id,
                "state": order.current_state.name,
                "filled_amount": str(order.executed_amount_base),
                "avg_price": str(order.average_executed_price),
            }
        return None

    async def check_timeout_orders(self) -> List[str]:
        """检查超时订单"""
        timeout_orders = []
        current_time = time.time()

        for order_id, order in self.bridge.connector.in_flight_orders.items():
            if order.current_state == OrderState.PENDING_CREATE:
                age = current_time - order.creation_timestamp
                if age > self._order_timeout:
                    timeout_orders.append(order_id)

        return timeout_orders
```

### 6.5 状态同步器

```python
# algvex/core/execution/state_synchronizer.py

class StateSynchronizer:
    """
    状态同步器

    职责:
    1. 定期同步仓位状态
    2. 检测并处理状态不一致
    3. 处理断线重连
    """

    def __init__(
        self,
        bridge: HummingbotBridge,
        position_manager: 'PositionManager',
        sync_interval: float = 60.0,  # 60秒同步一次
    ):
        self.bridge = bridge
        self.position_manager = position_manager
        self.sync_interval = sync_interval
        self._running = False

    async def start(self):
        """启动同步循环"""
        self._running = True
        while self._running:
            try:
                await self.sync()
            except Exception as e:
                logger.error(f"Sync failed: {e}")
            await asyncio.sleep(self.sync_interval)

    async def sync(self):
        """执行一次同步"""
        # 1. 获取交易所仓位
        exchange_positions = await self.bridge.sync_positions()

        # 2. 获取本地仓位
        local_positions = self.position_manager.get_all_positions()

        # 3. 对比并处理差异
        for symbol in set(exchange_positions.keys()) | set(local_positions.keys()):
            await self._sync_symbol(symbol, exchange_positions, local_positions)

    async def _sync_symbol(
        self,
        symbol: str,
        exchange_positions: Dict,
        local_positions: Dict,
    ):
        """同步单个品种"""
        exchange_pos = exchange_positions.get(symbol)
        local_pos = local_positions.get(symbol)

        if exchange_pos and not local_pos:
            # 交易所有仓位，本地没有 -> 更新本地
            logger.warning(f"Missing local position for {symbol}, syncing")
            self.position_manager.update_position(symbol, exchange_pos)

        elif local_pos and not exchange_pos:
            # 本地有仓位，交易所没有 -> 可能已平仓
            logger.warning(f"Position {symbol} closed on exchange")
            self.position_manager.close_position(symbol)

        elif exchange_pos and local_pos:
            # 两边都有，检查数量是否一致
            exchange_amt = Decimal(exchange_pos["amount"])
            local_amt = local_pos["amount"]

            if abs(exchange_amt - local_amt) > Decimal("0.00001"):
                logger.error(f"Position mismatch for {symbol}")
                # 以交易所为准
                self.position_manager.update_position(symbol, exchange_pos)

    async def on_disconnect(self):
        """处理断线"""
        logger.warning("Connector disconnected, entering protection mode")
        self.position_manager.enter_protection_mode()

    async def on_reconnect(self):
        """处理重连"""
        logger.info("Connector reconnected, performing full sync")
        await self.sync()
        self.position_manager.exit_protection_mode()
```

### 6.6 AlgVexController (Strategy V2 集成)

> 将 AlgVex 信号生成集成到 Hummingbot Strategy V2 框架

```python
# algvex/core/execution/controllers/algvex_controller.py

from hummingbot.smart_components.controllers.controller_base import ControllerBase
from hummingbot.smart_components.executors.position_executor.data_types import PositionConfig


class AlgVexControllerConfig:
    """AlgVex Controller 配置"""
    trading_pairs: Set[str]
    signal_threshold: float = 0.5
    max_position_per_pair: Decimal = Decimal("0.1")
    leverage: int = 1


class AlgVexController(ControllerBase):
    """
    AlgVex 信号控制器

    将 AlgVex SignalGenerator 集成到 Hummingbot V2 架构

    数据流:
    SignalGenerator → AlgVexController → PositionExecutor → Connector
    """

    def __init__(
        self,
        config: AlgVexControllerConfig,
        signal_generator: 'SignalGenerator',
    ):
        super().__init__(config)
        self.config = config
        self.signal_generator = signal_generator
        self._latest_signals = {}

    async def update_processed_data(self):
        """
        更新处理后的数据

        每个 tick 调用，获取最新的 AlgVex 信号
        """
        for trading_pair in self.config.trading_pairs:
            try:
                signal = await self.signal_generator.get_signal(trading_pair)
                self._latest_signals[trading_pair] = signal
            except Exception as e:
                self.logger().warning(f"Failed to get signal for {trading_pair}: {e}")

    def determine_executor_actions(self) -> List:
        """
        确定执行器动作

        基于 AlgVex 信号决定是否开仓/平仓
        """
        actions = []

        for trading_pair, signal in self._latest_signals.items():
            if signal is None:
                continue

            # 检查信号强度
            if abs(signal.final_signal) < self.config.signal_threshold:
                continue

            # 确定方向
            is_long = signal.final_signal > 0

            # 创建仓位配置
            position_config = PositionConfig(
                trading_pair=trading_pair,
                side="LONG" if is_long else "SHORT",
                amount=self._calculate_position_size(signal),
                leverage=self.config.leverage,
                stop_loss=Decimal("0.03"),   # 3% 止损
                take_profit=Decimal("0.06"), # 6% 止盈
            )

            actions.append({
                "type": "create_position",
                "config": position_config,
                "signal": signal,
            })

        return actions

    def _calculate_position_size(self, signal) -> Decimal:
        """计算仓位大小"""
        base_size = self.config.max_position_per_pair
        signal_weight = Decimal(str(abs(signal.final_signal)))
        return base_size * signal_weight
```

### 6.7 事件处理器

```python
# algvex/core/execution/event_handlers.py

from hummingbot.core.event.events import (
    BuyOrderCreatedEvent,
    SellOrderCreatedEvent,
    OrderFilledEvent,
    OrderCancelledEvent,
    MarketOrderFailureEvent,
    FundingPaymentCompletedEvent,
)


class AlgVexEventHandler:
    """
    Hummingbot 事件处理器

    事件类型映射:
    - BuyOrderCreatedEvent → order_created trace
    - OrderFilledEvent → order_filled trace, 更新仓位
    - OrderCancelledEvent → order_cancelled trace
    - MarketOrderFailureEvent → order_failed trace, 告警
    - FundingPaymentCompletedEvent → funding_payment trace
    """

    def __init__(self, bridge: HummingbotBridge, trace_writer: 'TraceWriter'):
        self.bridge = bridge
        self.trace_writer = trace_writer

    def register_events(self, connector: ConnectorBase):
        """注册事件监听"""
        connector.add_listener(BuyOrderCreatedEvent, self.on_buy_order_created)
        connector.add_listener(SellOrderCreatedEvent, self.on_sell_order_created)
        connector.add_listener(OrderFilledEvent, self.on_order_filled)
        connector.add_listener(OrderCancelledEvent, self.on_order_cancelled)
        connector.add_listener(MarketOrderFailureEvent, self.on_order_failure)
        connector.add_listener(FundingPaymentCompletedEvent, self.on_funding_payment)

    async def on_buy_order_created(self, event: BuyOrderCreatedEvent):
        """买单创建"""
        await self._handle_order_created(event, "BUY")

    async def on_sell_order_created(self, event: SellOrderCreatedEvent):
        """卖单创建"""
        await self._handle_order_created(event, "SELL")

    async def _handle_order_created(self, event, side: str):
        """处理订单创建"""
        trace = {
            "type": "order_created",
            "client_order_id": event.order_id,
            "trading_pair": event.trading_pair,
            "side": side,
            "amount": str(event.amount),
            "price": str(event.price) if event.price else None,
            "order_type": event.type.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.trace_writer.write(trace)

    async def on_order_filled(self, event: OrderFilledEvent):
        """订单成交"""
        await self.bridge.on_order_filled(event)

    async def on_order_cancelled(self, event: OrderCancelledEvent):
        """订单取消"""
        await self.bridge.on_order_cancelled(event)

    async def on_order_failure(self, event: MarketOrderFailureEvent):
        """订单失败 - 需要告警"""
        trace = {
            "type": "order_failed",
            "order_id": event.order_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.trace_writer.write(trace)
        # TODO: 发送告警

    async def on_funding_payment(self, event: FundingPaymentCompletedEvent):
        """资金费率支付"""
        trace = {
            "type": "funding_payment",
            "trading_pair": event.trading_pair,
            "amount": str(event.amount),
            "funding_rate": str(event.funding_rate),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.trace_writer.write(trace)
```

### 6.8 执行层验收标准

| 验收项 | 描述 | 测试方法 | 状态 |
|--------|------|----------|------|
| HummingbotBridge | 信号 → 订单转换正确 | 单元测试 | ⬜ |
| 幂等性 | 相同信号生成相同 order_id | 重复调用测试 | ⬜ |
| InFlightOrder | 订单状态追踪正确 | 集成测试 | ⬜ |
| 状态同步 | 仓位对账无差异 | 对账测试 | ⬜ |
| 事件处理 | 所有事件正确写入 trace | 事件模拟测试 | ⬜ |
| AlgVexController | V2 策略集成正常 | 集成测试 | ⬜ |
| 断线恢复 | 断线后能正确恢复状态 | 断线模拟测试 | ⬜ |
| Paper Trading | 模拟交易验证通过 | 24h 模拟运行 | ⬜ |

---

## 7. 风控层

> ✅ **MVP包含** - 三层风控(预交易/交易中/事后)为核心，**不可精简**。
> 📝 **关键** - 风控是系统稳定性的关键保障，必须完整实现。

### 7.1 多层风控体系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           风控层级结构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Level 1: 预交易风控 (Pre-Trade)                                        │
│  ├─ 杠杆检查: leverage <= max_leverage                                  │
│  ├─ 仓位价值检查: position_value <= max_position_value                  │
│  ├─ 总敞口检查: total_exposure <= max_total_exposure                    │
│  ├─ 持仓数量检查: positions_count <= max_positions                      │
│  ├─ 日亏损检查: daily_pnl > -max_daily_loss                            │
│  └─ 黑名单检查: symbol not in blocked_symbols                          │
│                                                                         │
│  Level 2: 交易中风控 (In-Trade)                                         │
│  ├─ 止损设置: 默认 3% (可基于ATR动态调整)                               │
│  ├─ 止盈设置: 基于风险收益比 1.5:1                                       │
│  ├─ 移动止损: 跟踪价格 2%                                               │
│  └─ 资金费率监控: funding_rate > threshold 时预警                       │
│                                                                         │
│  Level 3: 事后风控 (Post-Trade)                                         │
│  ├─ 回撤监控: max_drawdown > 15% 时熔断                                 │
│  ├─ 日亏损限制: daily_loss > 5% 时停止交易                              │
│  ├─ 周亏损限制: weekly_loss > 10% 时暂停一天                            │
│  └─ 异常检测: 大额亏损、频繁交易等                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 RiskManager

```python
from algvex.core.execution import RiskManager, RiskConfig

config = RiskConfig(
    max_position_value=10000.0,     # 单仓最大价值 $10k
    max_total_exposure=50000.0,     # 总敞口上限 $50k
    max_positions=10,                # 最大持仓数 10个
    max_leverage=10,                 # 最大杠杆 10x
    max_daily_loss=0.05,            # 日最大亏损 5%
    max_weekly_loss=0.10,           # 周最大亏损 10%
    max_drawdown=0.15,              # 最大回撤 15%
    max_single_trade_risk=0.02,     # 单笔最大风险 2%
    min_risk_reward_ratio=1.5,      # 最小风险收益比 1.5
    default_stop_loss=0.03,         # 默认止损 3%
    trailing_stop=0.02,             # 移动止损 2%
    max_funding_rate=0.001,         # 最大资金费率 0.1%
)

risk_manager = RiskManager(config)
```

### 7.3 PositionManager

```python
from algvex.core.execution import PositionManager, RebalanceMethod

pm = PositionManager(
    total_capital=100000,
    max_positions=10,
    min_position_weight=0.05,
    max_position_weight=0.25,
    rebalance_threshold=0.05,
)

targets = pm.calculate_targets(signals=signals, method=RebalanceMethod.SIGNAL_WEIGHT)
orders = pm.generate_rebalance_orders(current_prices)
```

---

## 8. 技术栈

> ✅ **MVP包含** - 核心技术栈必须确定，版本锁定。
> 📝 **MVP精简** - torch为可选，MVP仅用LightGBM；前端可Phase 2再精细打磨。

### 8.1 后端

```yaml
核心语言: Python 3.11+

量化引擎:
  - qlib: 0.9.7              # Microsoft 量化框架
  - hummingbot: 2.11.0       # 执行引擎
  - ccxt: 4.4+               # 交易所API

Web框架:
  - fastapi: 0.100+
  - uvicorn: 0.23+
  - websockets: 11.0+

数据库:
  - sqlalchemy: 2.0+
  - alembic: 1.12+
  - asyncpg: 0.28+

任务队列:
  - celery: 5.3+
  - redis: 4.6+

数据处理:
  - pandas: 2.0+
  - numpy: 1.24+
  - pyarrow: 13.0+

ML框架:
  - lightgbm: 4.0+
  - xgboost: 2.0+
  - torch: 2.0+ (可选)
```

### 8.2 前端

```yaml
框架:
  - react: 18+
  - typescript: 5.0+
  - vite: 5.0+

UI组件:
  - tailwindcss: 3.4+
  - shadcn/ui: latest

图表:
  - lightweight-charts: 4.0+    # TradingView K线
  - recharts: 2.10+

状态管理:
  - zustand: 4.4+

数据获取:
  - axios: 1.6+
  - @tanstack/react-query: 5.0+
  - socket.io-client: 4.7+
```

### 8.3 基础设施

```yaml
数据库:
  - PostgreSQL 15 + TimescaleDB 2.12
  - Redis 7

Web服务器:
  - Nginx 1.25+

CDN/DNS:
  - Cloudflare

监控:
  - Prometheus + Grafana
  - Sentry
```

---

## 9. 目录结构

> ✅ **MVP包含** - 目录结构为参考，可根据实际需求调整。
> 📝 **MVP精简** - 仅需production/和core/目录，research/目录可Phase 2建立。

```
algvex/
├── core/                           # 核心引擎
│   ├── __init__.py
│   ├── engine.py                   # AlgVex主引擎
│   ├── data/                       # 数据层
│   │   ├── __init__.py
│   │   ├── collector.py            # 币安数据采集
│   │   ├── handler.py              # Qlib格式转换
│   │   ├── validator.py            # 数据验证
│   │   ├── realtime.py             # WebSocket实时数据
│   │   └── multi_source_manager.py # 多数据源管理
│   ├── factor/                     # 因子层
│   │   ├── __init__.py
│   │   └── engine.py               # 因子计算引擎
│   ├── model/                      # 模型层
│   │   ├── __init__.py
│   │   ├── qlib_models.py          # Qlib模型集成 (v2.0.0新增)
│   │   └── trainer.py              # ML模型训练
│   ├── backtest/                   # 回测层
│   │   ├── __init__.py
│   │   └── engine.py               # 永续合约回测
│   ├── execution/                  # 执行层
│   │   ├── __init__.py
│   │   ├── exchange_connectors.py  # 多交易所连接器 (v2.0.0新增)
│   │   ├── executors.py            # 执行策略 TWAP/VWAP/Grid (v2.0.0新增)
│   │   ├── hummingbot_bridge.py    # Hummingbot桥接 (v2.0.0重写)
│   │   ├── risk_manager.py         # 风控管理
│   │   └── position_manager.py     # 仓位管理
│   └── strategy/                   # 策略层
│       ├── __init__.py
│       └── signal.py               # 信号生成
├── api/                            # API层
│   ├── __init__.py
│   ├── main.py                     # FastAPI入口
│   ├── config.py                   # 配置
│   ├── database.py                 # 数据库连接
│   └── routers/                    # API路由
│       ├── auth.py
│       ├── users.py
│       ├── strategies.py
│       ├── backtests.py
│       ├── trades.py
│       └── market.py
├── web/                            # 前端
│   ├── src/
│   ├── package.json
│   └── ...
├── scripts/                        # 脚本
│   ├── init_server.sh
│   ├── setup.sh
│   └── deploy.sh
├── requirements.txt
├── .env.example
└── README.md
```

---

## 10. 部署方案

> ✅ **MVP包含** - 直接 Python 部署即可，单实例运行。
> 📝 **MVP精简** - 使用最低配置(4核8G)启动，无需高可用架构。

### 10.1 服务器要求

```yaml
最低配置:
  CPU: 4核
  RAM: 8GB
  SSD: 100GB
  带宽: 100Mbps

推荐配置:
  CPU: 8核
  RAM: 16GB
  SSD: 500GB
  带宽: 1Gbps

操作系统: Ubuntu 22.04 LTS / Windows 10+
Python: 3.10+
```

### 10.2 直接部署 (推荐)

**优点**: 简单、调试方便、资源开销小、适合单实例场景

```bash
# 1. 克隆项目
git clone https://github.com/your-org/algvex.git
cd algvex

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件

# 5. 运行回测
python scripts/run_backtest.py --symbols BTCUSDT,ETHUSDT

# 6. 运行 API 服务 (可选)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 7. 运行定时任务 (可选)
python scripts/daily_alignment.py
```

### 10.3 后台运行 (Linux)

```bash
# 使用 nohup
nohup python scripts/run_backtest.py > backtest.log 2>&1 &

# 使用 screen
screen -S algvex
python scripts/run_backtest.py
# Ctrl+A, D 分离

# 使用 systemd (推荐生产环境)
# 创建 /etc/systemd/system/algvex.service
```

### 10.4 systemd 服务配置 (可选)

```ini
# /etc/systemd/system/algvex.service
[Unit]
Description=AlgVex Quantitative Trading Service
After=network.target

[Service]
Type=simple
User=algvex
WorkingDirectory=/home/algvex/algvex
Environment="PYTHONHASHSEED=42"
ExecStart=/home/algvex/algvex/venv/bin/python scripts/run_backtest.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 启用服务
sudo systemctl enable algvex
sudo systemctl start algvex
sudo systemctl status algvex
```

### 10.5 数据库配置 (可选)

如果需要持久化存储，可以本地安装 PostgreSQL 和 Redis：

```bash
# Ubuntu
sudo apt install postgresql redis-server

# 创建数据库
sudo -u postgres createdb algvex
sudo -u postgres createuser algvex

# 配置 .env
DATABASE_URL=postgresql://algvex:password@localhost/algvex
REDIS_URL=redis://localhost:6379
```

---

## 11. P0 验收标准 (上线前必须完成)

> ✅ **MVP关键** - P0验收标准是MVP上线的**硬性门槛**，每一项都不可跳过。
> 📝 **执行要求** - 每项必须带单元测试 + Replay对齐测试，否则不得上线。

> **关键原则**: 以下 4 项 P0 标准是系统从"看起来专业"变成"真的专业、可长期赚钱"的关键。每项必须带单元测试/回放对齐测试。

### 11.1 P0-1: 数据可见性与泄露防护

**核心问题**: 回测时必须确保模型只能"看到"当时真实可获得的数据，防止未来数据泄露。

**⚠️ 特别注意: Bar聚合特征的可见性陷阱**

> CVD、taker_delta、OI变化、basis 等"bar聚合特征"容易产生微妙的未来泄露：
> - 1h bar 的 CVD 只有在 bar 收盘后才完整可见
> - 如果把它当作"实时流"合并，会把"未来一小时的成交"混进当前特征
> - **规则**: 所有 bar 聚合特征的可见时间 = bar_close_time + publication_delay

```python
# 数据可见性检查器
class DataVisibilityChecker:
    """确保每个特征在信号生成时刻是真实可见的"""

    # ==================== 关键: 区分实时数据 vs Bar聚合数据 ====================

    # 实时数据源延迟 (可在bar内部更新)
    REALTIME_DELAYS = {
        "binance_mark_price": timedelta(seconds=1),    # 标记价格实时
        "binance_last_price": timedelta(seconds=1),    # 最新成交实时
    }

    # Bar聚合数据延迟 (只有bar收盘后才完整可见！)
    BAR_AGGREGATED_DELAYS = {
        "binance_ohlcv": "bar_close",           # K线数据 = bar收盘后可见
        "binance_taker_volume": "bar_close",    # Taker成交量 = bar收盘后可见 (CVD基于此！)
        "binance_oi_change": "bar_close + 5min", # OI变化 = bar收盘 + 5分钟延迟
        "binance_ls_ratio": "bar_close + 5min", # 多空比 = bar收盘 + 5分钟延迟
    }

    # 定时发布数据延迟
    SCHEDULED_DELAYS = {
        "binance_funding_rate": timedelta(hours=8),  # 每8小时结算后可见
        "defilama_onchain": timedelta(hours=1),      # 约1小时延迟
        "deribit_options": timedelta(minutes=5),     # 约5分钟延迟
        "fear_greed_index": timedelta(hours=24),     # 每日更新
        "macro_dxy": timedelta(hours=1),             # 约1小时延迟
    }

    def get_visible_time(
        self,
        data_source: str,
        bar_freq: str,
        bar_close_time: datetime,
        event_time: Optional[datetime] = None,
        scheduled_time: Optional[datetime] = None
    ) -> datetime:
        """
        计算特征的可见时间 - 这是防泄露的核心！

        关键区分:
        - realtime: 基于 event_time (数据产生时间)
        - bar_agg: 基于 bar_close_time (bar收盘时间)
        - scheduled: 基于 scheduled_time (定时发布时间)
        """

        if data_source in self.REALTIME_DELAYS:
            # 实时数据: 可见时间 = event_time + 固定延迟
            # 关键修复: 不能用bar_close_time! 必须用event_time!
            base_time = event_time if event_time else bar_close_time
            return base_time + self.REALTIME_DELAYS[data_source]

        elif data_source in self.BAR_AGGREGATED_DELAYS:
            # Bar聚合数据: 可见时间 = bar_close_time + 额外延迟
            delay_spec = self.BAR_AGGREGATED_DELAYS[data_source]
            if delay_spec == "bar_close":
                return bar_close_time  # bar收盘即可见
            elif "+" in delay_spec:
                extra_delay = self._parse_delay(delay_spec.split("+")[1].strip())
                return bar_close_time + extra_delay

        elif data_source in self.SCHEDULED_DELAYS:
            # 定时数据: 基于 scheduled_time
            base_time = scheduled_time if scheduled_time else bar_close_time
            return base_time + self.SCHEDULED_DELAYS[data_source]

        else:
            # 未知数据源，保守处理 (24小时延迟)
            return bar_close_time + timedelta(hours=24)

# ==================== Bar聚合特征的正确合并方式 ====================

def safe_merge_bar_features(signal_df, feature_df, bar_freq: str):
    """
    安全合并 Bar 聚合特征 - 防止 CVD 等特征泄露

    关键规则:
    - signal_time 定义为 bar_close_time (UTC)
    - 特征只能使用 <= signal_time 的已收盘 bar 的数据
    - 例如: 1h 信号在 12:00 生成时，只能使用 11:00 收盘的 bar 的 CVD
    """
    # 确保特征时间戳是 bar_close_time
    # 信号时间 = 当前 bar 收盘时间
    # 可用特征 = 上一个 bar 的数据 (因为当前 bar 还没收盘)

    bar_duration = _parse_bar_freq(bar_freq)

    # 核心: 信号时刻只能看到 "前一个bar" 的聚合数据
    # 因为 "当前bar" 还在进行中，CVD 不完整
    return pd.merge_asof(
        signal_df.sort_index(),
        feature_df.sort_index(),
        left_index=True,
        right_index=True,
        direction='backward',
        tolerance=bar_duration  # 只匹配最近一个 bar
    )

# CVD 特征的正确计算方式
def calculate_cvd_safe(df: pd.DataFrame, bar_freq: str = "1h") -> pd.Series:
    """
    计算 CVD (累计成交量差) - 确保不泄露

    重要: CVD = cumsum(taker_buy_volume - taker_sell_volume)
    - taker_buy_volume 和 taker_sell_volume 是 bar 聚合数据
    - 只有 bar 收盘后才知道这个 bar 的完整成交量
    - 因此 CVD(T) 只能在 T bar 收盘后才能计算
    """
    cvd = (df['taker_buy_volume'] - df['taker_sell_volume']).cumsum()

    # 标记: 这个 CVD 的时间戳是 bar_close_time
    # 在信号生成时，只能使用 <= 当前时间 - bar_duration 的 CVD
    cvd.name = 'cvd'
    cvd.attrs['visibility'] = 'bar_close'
    cvd.attrs['bar_freq'] = bar_freq

    return cvd
```

**验证测试**:
```python
def test_no_future_data_leakage():
    """泄露检测测试 - 必须通过"""
    # 1. 准备测试数据：T 时刻的信号 (假设 1h bar)
    signal_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    # 2. 验证所有 bar 聚合特征使用的是 "上一个bar" 的数据
    for feature in bar_aggregated_features:
        # CVD 等特征应该使用 11:00 收盘的 bar 的数据
        assert feature.bar_close_time <= signal_time - timedelta(hours=1), \
            f"Bar聚合特征泄露! {feature.name} 使用了未收盘bar的数据"

    # 3. 验证实时特征的延迟
    for feature in realtime_features:
        assert feature.timestamp + REALTIME_DELAYS[feature.source] <= signal_time, \
            f"实时特征泄露: {feature.name}"

    # 4. 随机抽样回放验证 (更严格)
    for _ in range(1000):
        random_time = get_random_historical_time()
        features = get_features_at_time(random_time)
        for f in features:
            visible_time = checker.get_visible_time(f.source, f.bar_freq, f.bar_close_time)
            assert visible_time <= random_time, \
                f"发现泄露: {f.name} visible_time={visible_time} > signal_time={random_time}"

def test_cvd_visibility():
    """CVD 可见性专项测试"""
    # 假设 1h bar，信号在 12:00 生成
    signal_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    # CVD 应该使用 11:00 收盘 bar 的累计值
    cvd_data = get_cvd_at_time(signal_time, bar_freq="1h")

    # 最新可用的 CVD 应该是 11:00 bar 的值
    assert cvd_data.index.max() == datetime(2024, 6, 15, 11, 0, 0, tzinfo=timezone.utc), \
        "CVD 使用了当前未收盘 bar 的数据，存在泄露！"
```

---

### 11.2 P0-2: 永续合约价格语义统一

**核心问题**: 永续合约有多种价格 (mark/index/last/close)，必须明确每个场景使用哪种价格。

```python
class PriceSemantics:
    """价格语义统一器 - 永续合约专用"""

    # 价格类型定义
    PRICE_TYPES = {
        "mark_price": "标记价格 - 用于计算盈亏和强平",
        "index_price": "指数价格 - 多交易所加权",
        "last_price": "最新成交价 - 实际交易价格",
        "close_price": "收盘价 - K线收盘",
    }

    # 场景-价格映射 (必须严格遵守)
    PRICE_USAGE_MAP = {
        "pnl_calculation": "mark_price",      # PnL计算用mark
        "liquidation_check": "mark_price",    # 爆仓检测用mark
        "entry_exit_signal": "close_price",   # 入场出场信号用close
        "order_execution": "last_price",      # 下单时用last
        "backtest_fill": "close_price",       # 回测成交用close
        "funding_settlement": "mark_price",   # 资金费率结算用mark
    }

    def get_price(self, scenario: str, data: dict) -> float:
        """根据场景获取正确的价格"""
        price_type = self.PRICE_USAGE_MAP[scenario]
        return data[price_type]

# 资金费率时间戳对齐
class FundingRateHandler:
    """资金费率处理器 - 必须精确对齐结算时间"""

    SETTLEMENT_HOURS = [0, 8, 16]  # UTC 结算时间

    def get_applicable_funding_rate(self, position_time: datetime) -> float:
        """获取适用于当前持仓的资金费率"""
        # 找到下一个结算时间
        next_settlement = self._next_settlement_time(position_time)

        # 持仓必须跨越结算时间才需支付资金费
        if self.position_held_through(next_settlement):
            return self._get_rate_at_settlement(next_settlement)
        return 0.0

    def validate_backtest_funding(self, backtest_results):
        """验证回测资金费率计算是否正确"""
        for trade in backtest_results.trades:
            expected_funding = self._calculate_expected_funding(trade)
            actual_funding = trade.funding_paid
            assert abs(expected_funding - actual_funding) < 1e-6, \
                f"资金费率计算错误: expected={expected_funding}, actual={actual_funding}"
```

**价格一致性测试**:
```python
def test_price_semantics_consistency():
    """价格语义一致性测试"""
    # 1. 验证回测和实盘使用相同的价格语义
    backtest_engine = CryptoPerpetualBacktest()
    live_engine = HummingbotBridge()

    for scenario in PriceSemantics.PRICE_USAGE_MAP:
        bt_price_type = backtest_engine.get_price_type(scenario)
        live_price_type = live_engine.get_price_type(scenario)
        assert bt_price_type == live_price_type, \
            f"价格语义不一致: {scenario} 回测用 {bt_price_type}, 实盘用 {live_price_type}"
```

---

### 11.3 P0-3: 订单/持仓强一致性

**核心问题**: 分布式系统中订单和持仓状态必须强一致，防止重复下单和状态不同步。

```python
class OrderConsistencyManager:
    """订单一致性管理器"""

    def __init__(self):
        self.order_cache = {}  # client_order_id -> order_state
        self.position_cache = {}  # symbol -> position

    # P0-3.1: 客户端订单ID幂等性
    def create_order(self, signal: Signal) -> Order:
        """创建订单 - 必须使用幂等的 client_order_id"""
        # 生成确定性的 client_order_id (基于信号内容hash)
        client_order_id = self._generate_idempotent_id(signal)

        # 检查是否已处理过
        if client_order_id in self.order_cache:
            existing = self.order_cache[client_order_id]
            if existing.status in ["FILLED", "PENDING"]:
                return existing  # 幂等返回，不重复下单

        # 新订单
        order = Order(
            client_order_id=client_order_id,
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
        )
        self.order_cache[client_order_id] = order
        return order

    def _generate_idempotent_id(self, signal: Signal) -> str:
        """生成幂等ID - 相同信号必须生成相同ID"""
        content = f"{signal.symbol}_{signal.side}_{signal.timestamp}_{signal.score}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # P0-3.2: 定期对账
    def reconcile_positions(self):
        """与交易所定期对账 - 每5分钟执行"""
        exchange_positions = self.exchange.get_positions()
        local_positions = self.position_cache

        for symbol in set(exchange_positions.keys()) | set(local_positions.keys()):
            exchange_qty = exchange_positions.get(symbol, {}).get("quantity", 0)
            local_qty = local_positions.get(symbol, {}).get("quantity", 0)

            if abs(exchange_qty - local_qty) > 1e-8:
                self._handle_position_mismatch(symbol, exchange_qty, local_qty)

    def _handle_position_mismatch(self, symbol, exchange_qty, local_qty):
        """处理持仓不一致"""
        logger.error(f"持仓不一致! {symbol}: 交易所={exchange_qty}, 本地={local_qty}")

        # 以交易所为准，强制同步
        self.position_cache[symbol]["quantity"] = exchange_qty

        # 触发告警
        self.alert_manager.send_critical(
            f"持仓不一致已自动修复: {symbol}"
        )

    # P0-3.3: 网络断开保护
    def enter_protection_mode(self):
        """网络断开时进入保护模式"""
        logger.warning("网络断开，进入保护模式")

        # 1. 停止新信号处理
        self.signal_processor.pause()

        # 2. 不主动平仓 (避免重连后状态混乱)
        # 3. 记录断开时刻的本地状态
        self.save_snapshot()

        # 4. 重连后强制对账
        self.on_reconnect = self.force_reconcile
```

**一致性测试**:
```python
def test_order_idempotency():
    """订单幂等性测试"""
    signal = Signal(symbol="BTCUSDT", score=0.8, timestamp=now())

    order1 = manager.create_order(signal)
    order2 = manager.create_order(signal)  # 相同信号

    assert order1.client_order_id == order2.client_order_id
    assert order1 is order2  # 应该返回同一个对象

def test_position_reconciliation():
    """持仓对账测试"""
    # 模拟本地和交易所状态不一致
    manager.position_cache["BTCUSDT"] = {"quantity": 1.0}
    mock_exchange.set_position("BTCUSDT", quantity=1.1)  # 交易所多了0.1

    manager.reconcile_positions()

    # 应该以交易所为准
    assert manager.position_cache["BTCUSDT"]["quantity"] == 1.1
```

---

### 11.4 P0-4: 研究验证标准 (Walk-Forward)

**核心问题**: 时序数据不能随机切分，必须使用 Walk-Forward 验证。

```python
class WalkForwardValidator:
    """Walk-Forward 验证器 - 防止过拟合"""

    def __init__(self,
                 train_months: int = 12,
                 test_months: int = 3,
                 min_train_samples: int = 1000):
        self.train_months = train_months
        self.test_months = test_months
        self.min_train_samples = min_train_samples

    def create_folds(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """创建 Walk-Forward 折叠 - 严禁随机切分"""
        folds = []
        start_date = data.index.min()
        end_date = data.index.max()

        current_train_start = start_date

        while True:
            train_end = current_train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            if test_end > end_date:
                break

            train_data = data[current_train_start:train_end]
            test_data = data[test_start:test_end]

            # 验证训练集样本量
            if len(train_data) >= self.min_train_samples:
                folds.append((train_data, test_data))

            # 滚动向前
            current_train_start += pd.DateOffset(months=self.test_months)

        return folds

    def validate_model(self, model, data: pd.DataFrame) -> dict:
        """执行 Walk-Forward 验证"""
        folds = self.create_folds(data)
        results = []

        for i, (train, test) in enumerate(folds):
            # 在训练集上训练
            model.fit(train)

            # 在测试集上评估 (完全 Out-of-Sample)
            metrics = model.evaluate(test)
            metrics["fold"] = i
            metrics["train_period"] = f"{train.index.min()} ~ {train.index.max()}"
            metrics["test_period"] = f"{test.index.min()} ~ {test.index.max()}"
            results.append(metrics)

        return self._aggregate_results(results)

# 过拟合检测
class OverfittingDetector:
    """过拟合检测器"""

    MAX_TRAIN_TEST_GAP = 0.3  # 训练集和测试集夏普比差距上限

    def check_overfitting(self, train_sharpe: float, test_sharpe: float) -> bool:
        """检测是否过拟合"""
        gap = train_sharpe - test_sharpe

        if gap > self.MAX_TRAIN_TEST_GAP:
            logger.warning(f"疑似过拟合! 训练夏普={train_sharpe:.2f}, 测试夏普={test_sharpe:.2f}")
            return True
        return False

    def calculate_deflated_sharpe(self, sharpe: float, num_trials: int) -> float:
        """计算 Deflated Sharpe Ratio (调整后的夏普比)"""
        # 考虑多次尝试的影响 (防止数据挖掘偏差)
        from scipy import stats

        # 使用 Bailey-Lopez de Prado 公式
        expected_max_sharpe = stats.norm.ppf(1 - 1/num_trials)
        deflated = sharpe - expected_max_sharpe
        return max(0, deflated)
```

**验证标准检查**:
```python
def test_walk_forward_validation():
    """Walk-Forward 验证测试"""
    validator = WalkForwardValidator(train_months=12, test_months=3)

    # 使用2年数据
    data = load_data("2022-01-01", "2024-01-01")
    folds = validator.create_folds(data)

    # 应该有约 4 个折叠 (24个月 / 3个月步长 - 初始12个月训练)
    assert len(folds) >= 3

    for train, test in folds:
        # 训练集必须在测试集之前
        assert train.index.max() < test.index.min()

        # 不能有重叠
        assert len(set(train.index) & set(test.index)) == 0

def test_no_random_split():
    """禁止随机切分测试"""
    # 如果代码中使用了 train_test_split(shuffle=True)，测试应该失败
    with pytest.raises(ValueError, match="禁止随机切分时序数据"):
        model.split_data(shuffle=True)
```

---

### 11.5 P0-5: 数据血缘与快照 (研究可复现)

**核心问题**: 没有数据血缘，无法证明某次收益来自哪份数据与哪份因子，线上崩了也无法回滚。

```python
class DataLineageManager:
    """数据血缘管理器 - 确保实验可复现"""

    def create_snapshot(self,
                       symbols: List[str],
                       start_date: str,
                       end_date: str) -> str:
        """创建数据快照 - 每次训练/回测前必须调用"""

        snapshot = DataSnapshot(
            snapshot_id=generate_uuid(),
            created_at=datetime.now(timezone.utc),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            source_versions=self._get_all_source_versions(),
            delay_config_hash=hash_config(DataVisibilityChecker.get_all_delays()),
            backfill_strategy_hash=hash_config(self.backfill_strategies),
        )

        # 持久化到数据库
        self.db.save_snapshot(snapshot)
        return snapshot.snapshot_id

    def record_experiment(self,
                         snapshot_id: str,
                         feature_set_id: str,
                         model_config: dict,
                         train_metrics: dict,
                         test_metrics: dict) -> str:
        """记录实验 - 建立完整血缘链"""

        record = ExperimentRecord(
            experiment_id=generate_uuid(),
            data_snapshot_id=snapshot_id,
            feature_set_id=feature_set_id,
            model_config_hash=hash_config(model_config),
            random_seed=model_config.get('random_seed', 42),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            git_commit=get_current_git_commit(),
            created_at=datetime.now(timezone.utc),
        )

        self.db.save_experiment(record)
        return record.experiment_id

    def reproduce_experiment(self, experiment_id: str) -> dict:
        """复现实验 - 使用相同的数据快照和配置"""
        record = self.db.get_experiment(experiment_id)
        snapshot = self.db.get_snapshot(record.data_snapshot_id)

        # 使用历史快照重新加载数据
        data = self.load_snapshot_data(snapshot)

        # 使用相同配置重新训练
        # ...

        return {"original": record.test_metrics, "reproduced": new_metrics}
```

**验证测试**:
```python
def test_experiment_reproducibility():
    """实验可复现性测试"""
    # 1. 创建快照并训练
    snapshot_id = lineage.create_snapshot(["BTCUSDT"], "2023-01-01", "2024-01-01")
    model1 = train_model(snapshot_id, random_seed=42)
    exp1_id = lineage.record_experiment(snapshot_id, ...)

    # 2. 使用相同快照重新训练
    model2 = train_model(snapshot_id, random_seed=42)

    # 3. 结果应该完全一致
    assert model1.test_sharpe == model2.test_sharpe, \
        "相同快照+相同种子，结果不一致！"

def test_snapshot_immutability():
    """快照不可变性测试"""
    snapshot_id = lineage.create_snapshot(["BTCUSDT"], "2023-01-01", "2024-01-01")
    data1 = lineage.load_snapshot_data(snapshot_id)

    # 即使原始数据更新了
    update_raw_data()

    # 快照数据应该不变
    data2 = lineage.load_snapshot_data(snapshot_id)
    assert data1.equals(data2), "快照数据被修改！"
```

---

### 11.6 P0-6: 回测-实盘成交对齐

**核心问题**: 回测的成交模型如果与实盘不一致，回测收益就是幻觉。

```python
class ExecutionModelValidator:
    """成交模型验证器 - 确保回测与实盘一致"""

    # 需要对齐的成交模型要素
    ALIGNMENT_CHECKLIST = {
        "fill_price": "成交价格模型 (last vs close vs mid)",
        "partial_fill": "部分成交处理",
        "fee_model": "费率模型 (maker/taker, VIP等级)",
        "slippage_model": "滑点模型 (静态 vs 动态 vs 冲击成本)",
        "reduce_only": "仅减仓订单处理",
        "position_mode": "仓位模式 (单向 vs 双向)",
        "trigger_logic": "触发单逻辑 (止损/止盈触发条件)",
        "leverage_handling": "杠杆处理 (保证金计算)",
        "liquidation_logic": "爆仓逻辑 (与交易所一致)",
    }

    def validate_alignment(self, backtest_engine, live_engine) -> dict:
        """验证回测与实盘的成交模型对齐"""
        results = {}

        for item, description in self.ALIGNMENT_CHECKLIST.items():
            bt_impl = getattr(backtest_engine, f"get_{item}_impl")()
            live_impl = getattr(live_engine, f"get_{item}_impl")()

            results[item] = {
                "description": description,
                "backtest": bt_impl,
                "live": live_impl,
                "aligned": bt_impl == live_impl,
            }

        return results

# 更真实的滑点模型 (静态 0.01% 太乐观)
class DynamicSlippageModel:
    """动态滑点模型 - 考虑市场条件"""

    def estimate_slippage(self,
                         symbol: str,
                         order_size_usd: float,
                         market_conditions: dict) -> float:
        """估计滑点 - 基于订单大小和市场条件"""

        # 基础滑点
        base_slippage = 0.0001  # 0.01%

        # 订单大小影响 (大单冲击成本)
        avg_daily_volume = market_conditions['avg_daily_volume']
        size_ratio = order_size_usd / avg_daily_volume
        size_impact = size_ratio * 0.1  # 占日成交量比例的10%

        # 波动率影响
        current_volatility = market_conditions['volatility']
        normal_volatility = 0.02  # 假设正常波动率2%
        vol_multiplier = current_volatility / normal_volatility

        # 流动性影响
        bid_ask_spread = market_conditions['bid_ask_spread']
        spread_impact = bid_ask_spread / 2

        total_slippage = (base_slippage + size_impact + spread_impact) * vol_multiplier

        return min(total_slippage, 0.01)  # 上限1%

# 费率模型 (考虑VIP等级)
class FeeModel:
    """费率模型 - 考虑VIP等级和maker/taker"""

    FEE_TIERS = {
        "VIP0": {"maker": 0.0002, "taker": 0.0004},
        "VIP1": {"maker": 0.00016, "taker": 0.0004},
        "VIP2": {"maker": 0.00014, "taker": 0.00035},
        "VIP3": {"maker": 0.00012, "taker": 0.00032},
        # ...
    }

    def __init__(self, vip_level: str = "VIP0"):
        self.vip_level = vip_level

    def get_fee(self, is_maker: bool) -> float:
        tier = self.FEE_TIERS[self.vip_level]
        return tier["maker"] if is_maker else tier["taker"]
```

**验证测试**:
```python
def test_execution_model_alignment():
    """成交模型对齐测试"""
    validator = ExecutionModelValidator()
    results = validator.validate_alignment(BacktestEngine(), LiveEngine())

    for item, result in results.items():
        assert result["aligned"], \
            f"成交模型不一致: {item} - 回测={result['backtest']}, 实盘={result['live']}"

def test_slippage_realistic():
    """滑点模型真实性测试"""
    model = DynamicSlippageModel()

    # 小单滑点应该很小
    small_slippage = model.estimate_slippage("BTCUSDT", 1000, normal_conditions)
    assert small_slippage < 0.0005, "小单滑点过大"

    # 大单滑点应该更大
    large_slippage = model.estimate_slippage("BTCUSDT", 1000000, normal_conditions)
    assert large_slippage > small_slippage, "大单滑点应大于小单"

    # 高波动时滑点应该更大
    volatile_slippage = model.estimate_slippage("BTCUSDT", 1000, high_vol_conditions)
    assert volatile_slippage > small_slippage, "高波动时滑点应更大"
```

---

### 11.7 P0 验收清单

| P0 标准 | 描述 | 验收条件 | 测试方法 |
|---------|------|----------|----------|
| P0-1 | 数据可见性 | 所有特征都有明确的发布延迟配置 | 泄露检测单元测试 |
| P0-1 | Bar聚合可见性 | CVD等bar特征只用已收盘bar数据 | CVD可见性测试 |
| P0-1 | As-of Merge | 所有特征合并使用 merge_asof | 代码审查 + 单元测试 |
| P0-2 | 价格语义 | mark/index/last/close 使用场景明确 | 价格一致性测试 |
| P0-2 | 资金费率 | 结算时间精确对齐 (0/8/16 UTC) | 回放对齐测试 |
| P0-3 | 幂等性 | client_order_id 基于信号内容hash | 幂等性单元测试 |
| P0-3 | 对账 | 每5分钟与交易所对账 | 对账测试 |
| P0-3 | 保护模式 | 网络断开时正确进入保护模式 | 网络中断模拟测试 |
| P0-4 | Walk-Forward | 禁止随机切分，固定OOS周期 | 验证流程测试 |
| P0-4 | 过拟合检测 | 训练/测试夏普差 < 0.3 | 过拟合检测测试 |
| P0-5 | 数据快照 | 每次训练记录完整数据血缘 | 可复现性测试 |
| P0-5 | 实验记录 | snapshot_id + feature_set_id + model_config | 血缘链完整性测试 |
| P0-6 | 成交模型 | 回测与实盘使用相同成交逻辑 | 对齐测试 |
| P0-6 | 滑点模型 | 考虑订单大小/波动率/流动性 | 真实性测试 |
| P0-6 | 费率模型 | 考虑VIP等级和maker/taker | 费率一致性测试 |

---

## 12. 开发路线图

> 📋 **参考文档** - 此路线图为全景规划，**MVP仅需完成Phase 0 + Phase 1核心部分**。
> ⏸️ **MVP不包含** - Phase 2/3的180因子扩展、链上数据、社媒新闻等延后实现。

> **原则**: 先让系统"可信"（数据可复现），再做"可用"（最小可交易），最后做"丰富"（180因子）。

### 12.1 阶段总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          开发阶段与依赖关系                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Phase 0: 数据基础设施 (让系统"可信")                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 1: 数据采集器实现                                               │   │
│  │ Step 2: B/C档数据落盘                                                │   │
│  │ Step 3: 数据血缘与快照                                               │   │
│  │ Step 7: 数据质量监控                                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  Phase 1: 回测可信性 + P1数据扩展 (让回测"可信")                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 4: 回测-实盘成交对齐 (DynamicSlippageModel)                     │   │
│  │ Step 5: Walk-Forward验证流程                                         │   │
│  │ ★ Step 9: L2深度聚合 + 滑点校准 (CalibratedSlippageModel)            │   │
│  │ ★ Step 10: 清算数据 (LiquidationCollector + 级联检测)                │   │
│  │ ★ Step 11: 多交易所Basis (Binance/Bybit/OKX 价差矩阵)                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  Phase 2: P0验收与CI/CD                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Step 6: P0单元测试                                                   │   │
│  │ Step 8: CI/CD集成                                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              ↓                                              │
│  Phase 3: 最小可交易系统                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 使用30-60个稳定因子训练baseline模型                                  │   │
│  │ 实盘影子模式 (paper trading) 验证                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄ 后续扩展 (见12.14) ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄        │
│  P2: 链上流向交易所 | P2: 更细IV结构 | P3: 社媒/新闻                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Step 9 为何优先**:
1. **直接提升 P0-6 可信度** - 当前 DynamicSlippageModel 基于经验公式估算，有真实深度数据可验证/校准
2. **工程复杂度可控** - 只需 bar 聚合版 (1m/5m)，不需要毫秒级
3. **回测/实盘双向受益** - 既能改进回测滑点模型，也能用于实盘下单前预估

### 12.2 Step 1: 数据采集器实现

**目标**: 实现所有 Collector 类，确保 fetch_historical 和 subscribe_realtime 方法符合 DataManager 规范。

**文件位置**: `algvex/core/data/collectors/`

```python
# 需要实现的采集器
algvex/core/data/collectors/
├── __init__.py
├── base.py              # BaseCollector 抽象类
├── binance.py           # BinanceCollector (OHLCV, OI, LS, Taker, Funding)
├── deribit.py           # DeribitCollector (DVOL, IV, Put/Call, MaxPain)
├── defilama.py          # DefiLlamaCollector (TVL, Stablecoin)
├── sentiment.py         # SentimentCollector (Fear&Greed, Google Trends)
└── macro.py             # MacroCollector (DXY, Yields, SPX, VIX)
```

**关键实现要点**:

```python
class BinanceCollector(BaseCollector):
    """币安数据采集器"""

    # 1. API限流配置 (必须遵守，否则会被封IP)
    RATE_LIMITS = {
        "klines": {"weight": 1, "limit": 1200, "window": 60},  # 1200/分钟
        "openInterest": {"weight": 1, "limit": 1200, "window": 60},
        "topLongShortRatio": {"weight": 1, "limit": 1200, "window": 60},
    }

    # 2. 重试配置
    RETRY_CONFIG = {
        "max_retries": 3,
        "backoff_factor": 2,  # 2s, 4s, 8s
        "retry_on": [429, 500, 502, 503, 504],
    }

    # 3. 错误处理
    async def fetch_historical(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """获取历史数据 - 带限流和重试"""
        try:
            await self._check_rate_limit("klines")
            data = await self._fetch_with_retry(...)
            return self._validate_and_clean(data)
        except RateLimitExceeded:
            await asyncio.sleep(self._get_backoff_time())
            return await self.fetch_historical(symbol, start, end)
        except Exception as e:
            self.logger.error(f"Fetch failed: {e}")
            raise DataFetchError(f"Failed to fetch {symbol}: {e}")

    # 4. 数据验证
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证数据完整性"""
        # 检查缺失值
        missing_ratio = df.isnull().sum() / len(df)
        if missing_ratio.max() > 0.1:  # 超过10%缺失
            self.logger.warning(f"High missing ratio: {missing_ratio.max():.2%}")

        # 检查异常值
        # 检查时间连续性
        # ...
        return df
```

**验收标准**:
- [ ] 所有 5 个 Collector 实现完成
- [ ] 每个 Collector 有对应的单元测试
- [ ] API 限流逻辑通过压力测试
- [ ] 错误重试逻辑覆盖常见异常

---

### 12.3 Step 2: B/C档数据落盘

**目标**: 将 B/C 档数据源的数据定期拉取并存入 TimescaleDB，形成长期历史。

**调度方案**: 使用 Celery Beat 定时任务

```python
# algvex/tasks/data_collection.py

from celery import Celery
from celery.schedules import crontab

app = Celery('algvex')

# 定时任务配置
app.conf.beat_schedule = {
    # B档数据: 每5分钟采集一次
    'collect-oi-every-5min': {
        'task': 'tasks.collect_open_interest',
        'schedule': crontab(minute='*/5'),
        'args': (['BTCUSDT', 'ETHUSDT'],),
    },
    'collect-ls-ratio-every-5min': {
        'task': 'tasks.collect_long_short_ratio',
        'schedule': crontab(minute='*/5'),
        'args': (['BTCUSDT', 'ETHUSDT'],),
    },
    'collect-taker-volume-every-1min': {
        'task': 'tasks.collect_taker_volume',
        'schedule': crontab(minute='*/1'),
        'args': (['BTCUSDT', 'ETHUSDT'],),
    },

    # B档数据: 每小时采集
    'collect-deribit-every-hour': {
        'task': 'tasks.collect_deribit_options',
        'schedule': crontab(minute=5),  # 每小时第5分钟
        'args': (['BTC', 'ETH'],),
    },

    # C档数据: 每日采集
    'collect-google-trends-daily': {
        'task': 'tasks.collect_google_trends',
        'schedule': crontab(hour=1, minute=0),  # 每天凌晨1点
        'args': (['bitcoin', 'crypto'],),
    },

    # 数据质量检查: 每小时
    'check-data-quality-hourly': {
        'task': 'tasks.check_data_quality',
        'schedule': crontab(minute=30),
    },
}

@app.task(bind=True, max_retries=3)
def collect_open_interest(self, symbols: List[str]):
    """采集持仓量数据"""
    try:
        collector = BinanceCollector()
        for symbol in symbols:
            data = collector.fetch_open_interest(symbol)
            storage.save_to_timescale(data, table='binance_oi')
            logger.info(f"Collected OI for {symbol}: {len(data)} rows")
    except Exception as e:
        logger.error(f"OI collection failed: {e}")
        self.retry(exc=e, countdown=60)  # 1分钟后重试
```

**TimescaleDB 表结构**:

```sql
-- B档数据表 (需要长期积累的数据)
CREATE TABLE binance_oi (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    open_interest DOUBLE PRECISION,
    open_interest_value DOUBLE PRECISION,
    collected_at TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('binance_oi', 'time');

CREATE TABLE binance_ls_ratio (
    time        TIMESTAMPTZ NOT NULL,
    symbol      TEXT NOT NULL,
    long_short_ratio DOUBLE PRECISION,
    long_account DOUBLE PRECISION,
    short_account DOUBLE PRECISION,
    collected_at TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('binance_ls_ratio', 'time');

-- 数据落盘元数据表 (追踪采集状态)
CREATE TABLE data_collection_log (
    id          SERIAL PRIMARY KEY,
    source      TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    start_time  TIMESTAMPTZ NOT NULL,
    end_time    TIMESTAMPTZ NOT NULL,
    rows_collected INTEGER,
    status      TEXT,  -- 'success', 'partial', 'failed'
    error_message TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
```

**验收标准**:
- [ ] Celery Beat 定时任务配置完成
- [ ] TimescaleDB 表结构创建完成
- [ ] B档数据 (OI/LS/Taker) 每5分钟自动采集
- [ ] C档数据 (Google Trends) 每日自动采集
- [ ] 采集日志可追溯

---

### 12.4 Step 3: 数据血缘与快照

**目标**: 实现 DataSnapshot 和 ExperimentRecord，确保每次训练/回测可复现。

**文件位置**: `algvex/core/data/lineage.py`

**快照存储方案**:
- 快照元数据 → PostgreSQL
- 快照数据文件 → 本地 Parquet (未来可迁移到 S3)

```python
# algvex/core/data/lineage.py

import hashlib
from pathlib import Path

class DataLineageManager:
    """数据血缘管理器"""

    SNAPSHOT_DIR = Path("~/.algvex/snapshots").expanduser()

    def __init__(self, db_url: str):
        self.db = Database(db_url)
        self.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def create_snapshot(self,
                       symbols: List[str],
                       start_date: str,
                       end_date: str,
                       data_manager: DataManager) -> str:
        """
        创建数据快照

        1. 从 DataManager 获取数据
        2. 计算数据内容 hash
        3. 保存到 Parquet 文件
        4. 记录元数据到数据库
        """
        # 1. 获取数据
        df = data_manager.get_historical(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            fields="all",
        )

        # 2. 生成快照ID (基于内容hash)
        content_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16]
        snapshot_id = f"snap_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{content_hash}"

        # 3. 保存数据文件
        snapshot_path = self.SNAPSHOT_DIR / f"{snapshot_id}.parquet"
        df.to_parquet(snapshot_path, compression='zstd')

        # 4. 记录元数据
        snapshot = DataSnapshot(
            snapshot_id=snapshot_id,
            created_at=datetime.now(timezone.utc),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            source_versions=data_manager.get_source_versions(),
            delay_config_hash=self._hash_delay_config(),
            backfill_strategy_hash=self._hash_backfill_config(),
            file_path=str(snapshot_path),
            content_hash=content_hash,
            row_count=len(df),
            column_count=len(df.columns),
        )
        self.db.save_snapshot(snapshot)

        logger.info(f"Created snapshot: {snapshot_id} ({len(df)} rows)")
        return snapshot_id

    def load_snapshot(self, snapshot_id: str) -> pd.DataFrame:
        """加载历史快照 - 确保数据不可变"""
        snapshot = self.db.get_snapshot(snapshot_id)
        df = pd.read_parquet(snapshot.file_path)

        # 验证数据完整性
        current_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:16]

        if current_hash != snapshot.content_hash:
            raise DataIntegrityError(
                f"Snapshot {snapshot_id} has been corrupted! "
                f"Expected hash: {snapshot.content_hash}, Got: {current_hash}"
            )

        return df

    def record_experiment(self,
                         snapshot_id: str,
                         feature_set_id: str,
                         model_config: dict,
                         train_metrics: dict,
                         test_metrics: dict) -> str:
        """记录实验 - 完整血缘链"""

        experiment = ExperimentRecord(
            experiment_id=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}",
            data_snapshot_id=snapshot_id,
            feature_set_id=feature_set_id,
            model_config_hash=hashlib.sha256(
                json.dumps(model_config, sort_keys=True).encode()
            ).hexdigest()[:16],
            random_seed=model_config.get('random_seed', 42),
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            git_commit=self._get_git_commit(),
            created_at=datetime.now(timezone.utc),
        )

        self.db.save_experiment(experiment)
        return experiment.experiment_id
```

**验收标准**:
- [ ] create_snapshot() 可正常创建快照
- [ ] load_snapshot() 可加载并验证快照完整性
- [ ] record_experiment() 可记录完整血缘链
- [ ] 快照数据不可变测试通过

---

### 12.5 Step 4: 回测-实盘成交对齐

**目标**: 确保回测的 fill_price, fee_model, slippage_model 与实盘一致。

**文件位置**: `algvex/core/backtest/execution_model.py`

**关键对齐项**:

```python
# algvex/core/backtest/execution_model.py

class ExecutionModel:
    """统一成交模型 - 回测和实盘共用"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.fee_model = FeeModel(config.vip_level)
        self.slippage_model = DynamicSlippageModel()

    def calculate_fill_price(self,
                            side: str,
                            order_type: str,
                            market_data: dict) -> float:
        """
        计算成交价格

        规则 (回测和实盘必须一致):
        - MARKET 单: last_price + slippage
        - LIMIT 单: limit_price (假设完全成交)
        """
        if order_type == "MARKET":
            base_price = market_data['last_price']
            slippage = self.slippage_model.estimate(
                symbol=market_data['symbol'],
                order_size_usd=market_data['order_size_usd'],
                conditions=market_data,
            )
            if side == "BUY":
                return base_price * (1 + slippage)
            else:
                return base_price * (1 - slippage)
        else:
            return market_data['limit_price']

    def calculate_fee(self,
                     notional: float,
                     is_maker: bool) -> float:
        """计算手续费"""
        return notional * self.fee_model.get_fee(is_maker)

# 确保回测引擎使用相同的成交模型
class CryptoPerpetualBacktest:
    def __init__(self, config: BacktestConfig):
        # 使用统一的成交模型
        self.execution_model = ExecutionModel(config.execution_config)

# 确保实盘桥接器使用相同的成交模型
class HummingbotBridge:
    def __init__(self, config: LiveConfig):
        # 使用相同的成交模型进行预估
        self.execution_model = ExecutionModel(config.execution_config)
```

**验收标准**:
- [ ] ExecutionModel 类实现完成
- [ ] BacktestEngine 和 LiveEngine 使用同一个 ExecutionModel
- [ ] 成交价格对齐测试通过
- [ ] 手续费对齐测试通过

---

### 12.6 Step 5: Walk-Forward 验证流程

**目标**: 实现 Walk-Forward 验证，禁止随机切分时序数据。

**文件位置**: `algvex/core/model/validation.py`

```python
# algvex/core/model/validation.py

class WalkForwardValidator:
    """Walk-Forward 验证器"""

    def __init__(self,
                 train_months: int = 12,
                 test_months: int = 3,
                 min_train_samples: int = 1000,
                 purge_days: int = 7):  # 训练集和测试集之间的隔离天数
        self.train_months = train_months
        self.test_months = test_months
        self.min_train_samples = min_train_samples
        self.purge_days = purge_days

    def create_folds(self, data: pd.DataFrame) -> List[WalkForwardFold]:
        """创建 Walk-Forward 折叠"""
        folds = []
        # ... 实现逻辑 (已在 P0-4 中定义)
        return folds

    def validate(self,
                model_class,
                model_params: dict,
                data: pd.DataFrame,
                target_col: str) -> WalkForwardResult:
        """执行 Walk-Forward 验证"""
        folds = self.create_folds(data)
        results = []

        for fold in folds:
            # 训练
            model = model_class(**model_params)
            model.fit(fold.train_data, fold.train_data[target_col])

            # 预测
            predictions = model.predict(fold.test_data)

            # 计算指标
            metrics = self._calculate_metrics(
                fold.test_data[target_col],
                predictions,
            )
            results.append(metrics)

        return WalkForwardResult(
            folds=folds,
            metrics=results,
            aggregate=self._aggregate_metrics(results),
        )

# 强制禁止随机切分
def train_test_split(*args, shuffle=False, **kwargs):
    """重写 train_test_split，禁止 shuffle=True"""
    if shuffle:
        raise ValueError(
            "禁止随机切分时序数据！请使用 WalkForwardValidator。"
        )
    return sklearn_train_test_split(*args, shuffle=False, **kwargs)
```

**验收标准**:
- [ ] WalkForwardValidator 实现完成
- [ ] 禁止 shuffle=True 的保护逻辑生效
- [ ] Walk-Forward 结果可视化报告

---

### 12.7 Step 6: P0 单元测试

**目标**: 为所有 P0 标准编写单元测试。

**文件位置**: `tests/p0/`

```
tests/p0/
├── __init__.py
├── test_p0_1_data_visibility.py     # 数据可见性测试
├── test_p0_2_price_semantics.py     # 价格语义测试
├── test_p0_3_order_consistency.py   # 订单一致性测试
├── test_p0_4_walk_forward.py        # Walk-Forward 测试
├── test_p0_5_data_lineage.py        # 数据血缘测试
├── test_p0_6_execution_alignment.py # 成交对齐测试
└── conftest.py                      # pytest fixtures
```

```python
# tests/p0/test_p0_1_data_visibility.py

import pytest
from datetime import datetime, timezone, timedelta

class TestP0_1_DataVisibility:
    """P0-1: 数据可见性测试"""

    def test_bar_aggregated_visibility(self, data_manager):
        """测试 bar 聚合特征的可见性"""
        signal_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        # 获取 CVD 数据
        cvd = data_manager.get_cvd_at_time(signal_time, bar_freq="1h")

        # CVD 应该使用 11:00 bar 的数据，不能使用 12:00 bar
        assert cvd.index.max() <= signal_time - timedelta(hours=1), \
            "CVD 使用了未收盘 bar 的数据，存在泄露！"

    def test_no_future_leakage(self, data_manager, sample_signals):
        """测试无未来数据泄露"""
        for signal in sample_signals:
            features = data_manager.get_features_at_time(signal.time)
            for f in features:
                visible_time = data_manager.get_visible_time(f)
                assert visible_time <= signal.time, \
                    f"发现泄露: {f.name} visible_time > signal_time"

    def test_merge_asof_used(self):
        """测试是否使用 merge_asof 而非普通 merge"""
        import ast
        from pathlib import Path

        # 扫描所有因子计算和数据合并相关代码
        target_dirs = [
            Path("algvex/core/factor"),
            Path("algvex/core/data"),
        ]

        violations = []
        for target_dir in target_dirs:
            if not target_dir.exists():
                continue

            for py_file in target_dir.rglob("*.py"):
                content = py_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    # 检查 pd.merge() 调用
                    if isinstance(node, ast.Call):
                        func = node.func
                        # 检查 pd.merge 或 DataFrame.merge
                        if isinstance(func, ast.Attribute):
                            if func.attr == "merge":
                                # 检查是否在时序数据合并场景
                                # merge_asof 的特征: direction 参数
                                has_direction = any(
                                    kw.arg == "direction"
                                    for kw in node.keywords
                                )
                                if not has_direction:
                                    # 检查注释是否有豁免标记
                                    violations.append({
                                        "file": str(py_file),
                                        "line": node.lineno,
                                        "issue": "使用 merge 而非 merge_asof",
                                    })

        # 报告结果
        if violations:
            for v in violations:
                # 允许通过 # noqa: ASOF 豁免
                print(f"⚠️ {v['file']}:{v['line']} - {v['issue']}")
                print("   请确认是否需要改为 merge_asof (时序数据合并场景)")

        # 至少检查核心文件存在 merge_asof 调用
        core_files = list(Path("algvex/core/factor").rglob("*.py"))
        has_merge_asof = False
        for f in core_files:
            if "merge_asof" in f.read_text():
                has_merge_asof = True
                break

        assert has_merge_asof, "核心因子模块必须使用 merge_asof 进行时序数据合并"
```

**验收标准**:
- [ ] 每个 P0 标准至少有 3 个测试用例
- [ ] 测试覆盖率 > 80%
- [ ] 所有 P0 测试通过

---

### 12.8 Step 7: 数据质量监控 (补充)

**目标**: 监控数据源健康状态，及时发现问题。

**文件位置**: `algvex/core/data/quality.py`

```python
# algvex/core/data/quality.py

class DataQualityMonitor:
    """数据质量监控器"""

    # 监控指标阈值
    THRESHOLDS = {
        "missing_rate": 0.05,       # 缺失率 > 5% 告警
        "delay_seconds": 300,       # 延迟 > 5分钟告警
        "schema_change": True,      # 字段变化告警
        "value_range_violation": 0.01,  # 异常值 > 1% 告警
    }

    def check_data_source(self, source: str) -> DataQualityReport:
        """检查单个数据源的健康状态"""
        report = DataQualityReport(source=source)

        # 1. 检查最新数据时间 (延迟检测)
        latest_time = self.db.get_latest_time(source)
        delay = datetime.now(timezone.utc) - latest_time
        if delay.total_seconds() > self.THRESHOLDS["delay_seconds"]:
            report.add_alert(
                level="WARNING",
                message=f"Data delay: {delay.total_seconds()}s",
            )

        # 2. 检查缺失率
        missing_rate = self.db.get_missing_rate(source, window="24h")
        if missing_rate > self.THRESHOLDS["missing_rate"]:
            report.add_alert(
                level="ERROR",
                message=f"High missing rate: {missing_rate:.2%}",
            )

        # 3. 检查字段变化
        current_schema = self.get_current_schema(source)
        expected_schema = self.get_expected_schema(source)
        if current_schema != expected_schema:
            report.add_alert(
                level="CRITICAL",
                message=f"Schema changed: {current_schema}",
            )

        return report

    def run_all_checks(self) -> List[DataQualityReport]:
        """运行所有数据源检查"""
        reports = []
        for source in self.ALL_SOURCES:
            report = self.check_data_source(source)
            reports.append(report)

            # 发送告警
            if report.has_critical():
                self.alert_manager.send_critical(report)
            elif report.has_error():
                self.alert_manager.send_error(report)

        return reports
```

**Celery 定时检查**:

```python
@app.task
def check_data_quality():
    """每小时检查数据质量"""
    monitor = DataQualityMonitor()
    reports = monitor.run_all_checks()

    # 生成报告
    summary = DataQualitySummary(reports)
    logger.info(f"Data quality check: {summary.status}")

    # 如果有严重问题，暂停相关数据采集
    if summary.has_critical():
        pause_data_collection(summary.critical_sources)
```

**验收标准**:
- [ ] 数据延迟监控正常
- [ ] 缺失率监控正常
- [ ] 告警通知可达 (Slack/邮件)

---

### 12.9 Step 8: CI/CD 集成 (补充)

**目标**: 将 P0 测试集成到 CI/CD 流程，确保每次提交都通过验收。

**GitHub Actions 配置**:

```yaml
# .github/workflows/p0-tests.yml

name: P0 Verification Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  p0-tests:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: timescale/timescaledb:latest-pg15
        env:
          POSTGRES_DB: algvex_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Run P0 Tests
        run: |
          pytest tests/p0/ -v --tb=short --cov=algvex --cov-report=xml

      - name: Check P0 Coverage
        run: |
          # P0 测试覆盖率必须 > 80%
          coverage report --fail-under=80

      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
```

**验收标准**:
- [ ] GitHub Actions 配置完成
- [ ] PR 必须通过 P0 测试才能合并
- [ ] 覆盖率报告自动生成

---

### 12.10 开发检查清单

| Phase | Step | 描述 | 状态 | 负责人 |
|-------|------|------|------|--------|
| 0 | 1 | 数据采集器实现 (5个 Collector) | ⬜ | - |
| 0 | 2 | B/C档数据落盘 (Celery + TimescaleDB) | ⬜ | - |
| 0 | 3 | 数据血缘与快照 | ⬜ | - |
| 0 | 7 | 数据质量监控 | ⬜ | - |
| 1 | 4 | 回测-实盘成交对齐 (DynamicSlippageModel) | ⬜ | - |
| 1 | 5 | Walk-Forward 验证流程 | ⬜ | - |
| **1** | **9** | **★ L2深度聚合 + 滑点校准** | ⬜ | - |
| **1** | **10** | **★ 清算数据 + 级联检测** | ⬜ | - |
| **1** | **11** | **★ 多交易所Basis (Binance/Bybit/OKX)** | ⬜ | - |
| 2 | 6 | P0 单元测试 (6组) | ⬜ | - |
| 2 | 8 | CI/CD 集成 | ⬜ | - |
| 3 | - | Baseline模型训练 (30-60稳定因子) | ⬜ | - |
| 3 | - | 实盘影子模式 (Paper Trading) | ⬜ | - |

---

### 12.11 Step 9: L2 深度聚合 + 滑点模型校准 (优先实施)

> **为什么优先**: 这是第一个数据扩展，直接解决 P0-6 滑点模型"估算"的问题，用真实深度数据校准。
>
> **工程复杂度**: 可控。只做 1m/5m bar 聚合，不做毫秒级 orderbook 快照。
>
> **双向受益**: 回测滑点更真实 + 实盘下单前可预估冲击成本。

#### 12.11.1 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Step 9: L2 深度聚合 + 滑点校准 架构                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    1. DepthCollector (WebSocket)                    │   │
│  │                                                                      │   │
│  │   Binance WS ──→ 原始深度快照 ──→ 1m/5m 聚合 ──→ TimescaleDB       │   │
│  │   (100ms更新)     (内存buffer)      (bar_close)    (持久化)         │   │
│  │                                                                      │   │
│  │   ⚠️ 可见性: bar_close (只有当bar结束后，聚合数据才可用)              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    2. 深度因子计算 (8个核心指标)                      │   │
│  │                                                                      │   │
│  │   bid_ask_spread, order_book_imbalance, depth_1pct_bid/ask,         │   │
│  │   depth_slope_bid/ask, impact_cost_buy/sell                         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    3. CalibratedSlippageModel                        │   │
│  │                                                                      │   │
│  │   Step 4 DynamicSlippageModel (估算) ──升级──→ 真实深度校准           │   │
│  │                                                                      │   │
│  │   回测: 用历史 impact_cost 代替经验公式                               │   │
│  │   实盘: 用实时深度预估下单冲击                                        │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 12.11.2 文件结构

```
algvex/core/data/collectors/
├── depth.py                    # DepthCollector (新增)
│
algvex/core/data/features/
├── depth_features.py           # 8个深度因子计算 (新增)
│
algvex/core/backtest/
├── slippage_model.py           # DynamicSlippageModel (已有, Step 4)
├── calibrated_slippage.py      # CalibratedSlippageModel (新增, Step 9)
│
tests/p0/
├── test_depth_collector.py     # 深度采集测试 (新增)
├── test_calibrated_slippage.py # 校准滑点测试 (新增)
```

#### 12.11.3 DepthCollector 实现

**文件位置**: `algvex/core/data/collectors/depth.py`

```python
# algvex/core/data/collectors/depth.py

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import numpy as np
import pandas as pd
import websockets

from .base import BaseCollector


@dataclass
class DepthSnapshot:
    """单次深度快照"""
    timestamp: datetime
    symbol: str
    bids: List[List[float]]  # [[price, qty], ...]
    asks: List[List[float]]  # [[price, qty], ...]

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else float('inf')

    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2


@dataclass
class AggregatedDepthBar:
    """聚合后的深度 Bar (1m/5m)"""
    bar_time: datetime           # bar 开始时间
    symbol: str

    # 聚合统计 (bar 内所有快照的均值/加权均值)
    avg_bid_ask_spread: float    # 平均 spread (bps)
    avg_imbalance: float         # 平均 imbalance (-1 to 1)
    avg_depth_1pct_bid: float    # 1% 范围内平均买单量 (USD)
    avg_depth_1pct_ask: float    # 1% 范围内平均卖单量 (USD)
    avg_depth_slope_bid: float   # 买单量衰减斜率
    avg_depth_slope_ask: float   # 卖单量衰减斜率

    # 冲击成本 (关键! 用于滑点校准)
    impact_cost_10k_buy: float   # 买入 $10k 的冲击成本 (bps)
    impact_cost_10k_sell: float  # 卖出 $10k 的冲击成本 (bps)
    impact_cost_50k_buy: float   # 买入 $50k 的冲击成本 (bps)
    impact_cost_50k_sell: float  # 卖出 $50k 的冲击成本 (bps)
    impact_cost_100k_buy: float  # 买入 $100k 的冲击成本 (bps)
    impact_cost_100k_sell: float # 卖出 $100k 的冲击成本 (bps)

    # 元数据
    snapshot_count: int          # bar 内采集的快照数
    visibility: str = "bar_close"  # 可见性规则


class DepthCollector(BaseCollector):
    """
    币安 L2 深度采集器 (WebSocket)

    ⚠️ 可见性规则: bar_close
    - 深度数据在 bar 结束后才能用于因子计算
    - 防止未来信息泄露

    存储策略: 只存聚合后的 bar 数据，不存原始快照 (太大)
    """

    # 币安 WebSocket 配置
    WS_URL = "wss://fstream.binance.com/ws"
    DEPTH_LEVELS = 20  # 前20档
    UPDATE_SPEED = "100ms"  # 100ms 更新频率

    # 聚合配置
    BAR_FREQUENCIES = ["1m", "5m"]  # 支持的聚合周期

    # 冲击成本计算的订单规模 (USD)
    IMPACT_SIZES = [10_000, 50_000, 100_000]

    def __init__(self,
                 symbols: List[str],
                 bar_freq: str = "1m",
                 on_bar_complete: Optional[Callable] = None):
        """
        Args:
            symbols: 要订阅的交易对列表 (如 ["btcusdt", "ethusdt"])
            bar_freq: 聚合频率 ("1m" 或 "5m")
            on_bar_complete: bar 完成时的回调函数
        """
        self.symbols = [s.lower() for s in symbols]
        self.bar_freq = bar_freq
        self.on_bar_complete = on_bar_complete

        # 内存缓冲区: symbol -> 当前 bar 的快照列表
        self._buffers: Dict[str, List[DepthSnapshot]] = defaultdict(list)

        # 当前 bar 开始时间
        self._current_bar_start: Dict[str, datetime] = {}

        # 运行状态
        self._running = False
        self._ws = None

    async def start(self):
        """启动 WebSocket 连接和数据采集"""
        self._running = True

        # 构建订阅 streams
        streams = [f"{s}@depth{self.DEPTH_LEVELS}@{self.UPDATE_SPEED}"
                   for s in self.symbols]
        url = f"{self.WS_URL}/stream?streams={'/'.join(streams)}"

        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._ws = ws
                    await self._receive_loop()
            except Exception as e:
                if self._running:
                    print(f"WebSocket disconnected: {e}, reconnecting in 5s...")
                    await asyncio.sleep(5)

    async def _receive_loop(self):
        """接收并处理深度更新"""
        async for message in self._ws:
            data = json.loads(message)

            # 解析深度数据
            stream = data.get("stream", "")
            symbol = stream.split("@")[0]
            depth_data = data.get("data", {})

            snapshot = DepthSnapshot(
                timestamp=datetime.now(timezone.utc),
                symbol=symbol,
                bids=[[float(p), float(q)] for p, q in depth_data.get("b", [])],
                asks=[[float(p), float(q)] for p, q in depth_data.get("a", [])],
            )

            # 添加到缓冲区
            self._add_to_buffer(snapshot)

    def _add_to_buffer(self, snapshot: DepthSnapshot):
        """添加快照到缓冲区，检查是否需要聚合"""
        symbol = snapshot.symbol

        # 计算当前 bar 开始时间
        bar_start = self._get_bar_start(snapshot.timestamp)

        # 检查是否需要完成上一个 bar
        if symbol in self._current_bar_start:
            if bar_start > self._current_bar_start[symbol]:
                # 完成上一个 bar
                self._complete_bar(symbol, self._current_bar_start[symbol])
                self._buffers[symbol] = []

        self._current_bar_start[symbol] = bar_start
        self._buffers[symbol].append(snapshot)

    def _get_bar_start(self, ts: datetime) -> datetime:
        """计算 bar 开始时间"""
        if self.bar_freq == "1m":
            return ts.replace(second=0, microsecond=0)
        elif self.bar_freq == "5m":
            minute = (ts.minute // 5) * 5
            return ts.replace(minute=minute, second=0, microsecond=0)
        else:
            raise ValueError(f"Unsupported bar_freq: {self.bar_freq}")

    def _complete_bar(self, symbol: str, bar_time: datetime):
        """聚合并输出一个完整的 bar"""
        snapshots = self._buffers[symbol]
        if not snapshots:
            return

        # 计算聚合指标
        aggregated = self._aggregate_snapshots(symbol, bar_time, snapshots)

        # 回调
        if self.on_bar_complete:
            self.on_bar_complete(aggregated)

    def _aggregate_snapshots(self,
                             symbol: str,
                             bar_time: datetime,
                             snapshots: List[DepthSnapshot]) -> AggregatedDepthBar:
        """聚合快照为 bar 数据"""

        spreads = []
        imbalances = []
        depth_1pct_bids = []
        depth_1pct_asks = []
        slope_bids = []
        slope_asks = []
        impact_costs = {size: {"buy": [], "sell": []}
                       for size in self.IMPACT_SIZES}

        for snap in snapshots:
            mid = snap.mid_price
            if mid <= 0:
                continue

            # 1. Bid-Ask Spread (bps)
            spread_bps = (snap.best_ask - snap.best_bid) / mid * 10000
            spreads.append(spread_bps)

            # 2. Order Book Imbalance
            bid_qty = sum(qty for _, qty in snap.bids)
            ask_qty = sum(qty for _, qty in snap.asks)
            total = bid_qty + ask_qty
            imbalance = (bid_qty - ask_qty) / total if total > 0 else 0
            imbalances.append(imbalance)

            # 3. Depth within 1% (USD)
            depth_bid = self._calculate_depth_within_pct(snap.bids, mid, 0.01)
            depth_ask = self._calculate_depth_within_pct(snap.asks, mid, 0.01)
            depth_1pct_bids.append(depth_bid * mid)  # 转换为 USD
            depth_1pct_asks.append(depth_ask * mid)

            # 4. Depth Slope (衰减速度)
            slope_bid = self._calculate_depth_slope(snap.bids, mid)
            slope_ask = self._calculate_depth_slope(snap.asks, mid)
            slope_bids.append(slope_bid)
            slope_asks.append(slope_ask)

            # 5. Impact Cost (关键!)
            for size in self.IMPACT_SIZES:
                cost_buy = self._calculate_impact_cost(snap.asks, mid, size)
                cost_sell = self._calculate_impact_cost(snap.bids, mid, size)
                impact_costs[size]["buy"].append(cost_buy)
                impact_costs[size]["sell"].append(cost_sell)

        return AggregatedDepthBar(
            bar_time=bar_time,
            symbol=symbol.upper(),
            avg_bid_ask_spread=np.mean(spreads) if spreads else 0,
            avg_imbalance=np.mean(imbalances) if imbalances else 0,
            avg_depth_1pct_bid=np.mean(depth_1pct_bids) if depth_1pct_bids else 0,
            avg_depth_1pct_ask=np.mean(depth_1pct_asks) if depth_1pct_asks else 0,
            avg_depth_slope_bid=np.mean(slope_bids) if slope_bids else 0,
            avg_depth_slope_ask=np.mean(slope_asks) if slope_asks else 0,
            impact_cost_10k_buy=np.mean(impact_costs[10000]["buy"]),
            impact_cost_10k_sell=np.mean(impact_costs[10000]["sell"]),
            impact_cost_50k_buy=np.mean(impact_costs[50000]["buy"]),
            impact_cost_50k_sell=np.mean(impact_costs[50000]["sell"]),
            impact_cost_100k_buy=np.mean(impact_costs[100000]["buy"]),
            impact_cost_100k_sell=np.mean(impact_costs[100000]["sell"]),
            snapshot_count=len(snapshots),
        )

    def _calculate_depth_within_pct(self,
                                    levels: List[List[float]],
                                    mid: float,
                                    pct: float) -> float:
        """计算指定百分比范围内的深度"""
        total_qty = 0
        for price, qty in levels:
            if abs(price - mid) / mid <= pct:
                total_qty += qty
        return total_qty

    def _calculate_depth_slope(self,
                               levels: List[List[float]],
                               mid: float) -> float:
        """计算深度衰减斜率 (越陡峭说明流动性越集中在 best price)"""
        if len(levels) < 5:
            return 0

        distances = []
        quantities = []
        for price, qty in levels[:10]:  # 前10档
            dist = abs(price - mid) / mid * 100  # 百分比距离
            distances.append(dist)
            quantities.append(qty)

        if not distances:
            return 0

        # 简单线性回归斜率
        x = np.array(distances)
        y = np.array(quantities)
        if len(x) > 1 and np.std(x) > 0:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        return 0

    def _calculate_impact_cost(self,
                               levels: List[List[float]],
                               mid: float,
                               order_size_usd: float) -> float:
        """
        计算冲击成本 (bps)

        模拟吃掉订单簿，计算平均成交价与 mid 的偏离
        """
        remaining_usd = order_size_usd
        total_qty = 0
        total_cost = 0

        for price, qty in levels:
            level_usd = price * qty
            if remaining_usd <= 0:
                break

            fill_usd = min(remaining_usd, level_usd)
            fill_qty = fill_usd / price

            total_qty += fill_qty
            total_cost += fill_qty * price
            remaining_usd -= fill_usd

        if total_qty == 0:
            return 0

        avg_price = total_cost / total_qty
        impact_bps = abs(avg_price - mid) / mid * 10000
        return impact_bps

    async def stop(self):
        """停止采集"""
        self._running = False
        if self._ws:
            await self._ws.close()


# ============== TimescaleDB 存储 ==============

DEPTH_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS depth_bars (
    bar_time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- 流动性指标
    avg_bid_ask_spread DOUBLE PRECISION,
    avg_imbalance DOUBLE PRECISION,
    avg_depth_1pct_bid DOUBLE PRECISION,
    avg_depth_1pct_ask DOUBLE PRECISION,
    avg_depth_slope_bid DOUBLE PRECISION,
    avg_depth_slope_ask DOUBLE PRECISION,

    -- 冲击成本 (核心！用于滑点校准)
    impact_cost_10k_buy DOUBLE PRECISION,
    impact_cost_10k_sell DOUBLE PRECISION,
    impact_cost_50k_buy DOUBLE PRECISION,
    impact_cost_50k_sell DOUBLE PRECISION,
    impact_cost_100k_buy DOUBLE PRECISION,
    impact_cost_100k_sell DOUBLE PRECISION,

    -- 元数据
    snapshot_count INTEGER,

    PRIMARY KEY (bar_time, symbol)
);

-- 创建 hypertable (TimescaleDB)
SELECT create_hypertable('depth_bars', 'bar_time', if_not_exists => TRUE);

-- 索引
CREATE INDEX IF NOT EXISTS idx_depth_symbol ON depth_bars (symbol, bar_time DESC);
"""
```

#### 12.11.4 CalibratedSlippageModel 实现

**文件位置**: `algvex/core/backtest/calibrated_slippage.py`

```python
# algvex/core/backtest/calibrated_slippage.py

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import pandas as pd

from .slippage_model import DynamicSlippageModel  # Step 4 的基础模型


@dataclass
class SlippageEstimate:
    """滑点估算结果"""
    slippage_bps: float          # 估算滑点 (bps)
    confidence: str              # "high" / "medium" / "low"
    source: str                  # "depth_data" / "fallback_model"
    details: Dict                # 详细信息


class CalibratedSlippageModel:
    """
    校准滑点模型 - 基于真实 L2 深度数据

    升级路径:
    - Step 4 DynamicSlippageModel: 基于经验公式估算 (fallback)
    - Step 9 CalibratedSlippageModel: 基于真实深度数据校准 (primary)

    使用场景:
    - 回测: 使用历史 impact_cost 计算更真实的滑点
    - 实盘: 使用实时深度数据预估下单冲击
    """

    # 预设的订单规模档位 (与 DepthCollector 对齐)
    SIZE_TIERS = [10_000, 50_000, 100_000]

    def __init__(self,
                 data_manager,
                 fallback_model: Optional[DynamicSlippageModel] = None):
        """
        Args:
            data_manager: 数据管理器 (用于获取深度数据)
            fallback_model: 当没有深度数据时的回退模型 (Step 4)
        """
        self.data_manager = data_manager
        self.fallback_model = fallback_model or DynamicSlippageModel()

        # 校准系数 (可通过历史数据拟合)
        self.calibration_params = {
            "spread_weight": 0.5,      # spread 对滑点的贡献
            "impact_weight": 1.0,      # impact_cost 对滑点的贡献
            "volatility_adj": 0.3,     # 波动率调整系数
        }

    def estimate_slippage(self,
                         symbol: str,
                         order_size_usd: float,
                         bar_time: pd.Timestamp,
                         side: str = "buy",
                         use_fallback_if_missing: bool = True) -> SlippageEstimate:
        """
        估算滑点

        Args:
            symbol: 交易对 (如 "BTCUSDT")
            order_size_usd: 订单金额 (USD)
            bar_time: 当前 bar 时间 (用于获取 as-of 深度数据)
            side: "buy" 或 "sell"
            use_fallback_if_missing: 无深度数据时是否使用回退模型

        Returns:
            SlippageEstimate: 滑点估算结果
        """

        # 1. 尝试获取深度数据
        depth_data = self._get_depth_at_time(symbol, bar_time)

        if depth_data is None:
            # 无深度数据，使用回退模型
            if use_fallback_if_missing:
                fallback_slip = self.fallback_model.estimate_slippage(
                    symbol=symbol,
                    order_size_usd=order_size_usd,
                    market_conditions=self._get_market_conditions(symbol, bar_time)
                )
                return SlippageEstimate(
                    slippage_bps=fallback_slip * 10000,  # 转为 bps
                    confidence="low",
                    source="fallback_model",
                    details={"reason": "no_depth_data"}
                )
            else:
                raise ValueError(f"No depth data for {symbol} at {bar_time}")

        # 2. 根据订单规模插值计算冲击成本
        impact_bps = self._interpolate_impact_cost(
            depth_data, order_size_usd, side
        )

        # 3. 加入 spread 贡献
        spread_bps = depth_data.get("avg_bid_ask_spread", 0)
        spread_contribution = spread_bps * self.calibration_params["spread_weight"]

        # 4. 波动率调整 (高波动时滑点通常更大)
        volatility = self._get_volatility(symbol, bar_time)
        vol_adj = 1 + (volatility - 0.02) * self.calibration_params["volatility_adj"]
        vol_adj = max(0.5, min(vol_adj, 2.0))  # 限制在 [0.5, 2.0]

        # 5. 综合计算
        total_slippage_bps = (impact_bps + spread_contribution) * vol_adj

        return SlippageEstimate(
            slippage_bps=total_slippage_bps,
            confidence="high" if depth_data.get("snapshot_count", 0) > 30 else "medium",
            source="depth_data",
            details={
                "impact_bps": impact_bps,
                "spread_contribution": spread_contribution,
                "volatility_adj": vol_adj,
                "snapshot_count": depth_data.get("snapshot_count", 0),
            }
        )

    def _get_depth_at_time(self,
                          symbol: str,
                          bar_time: pd.Timestamp) -> Optional[Dict]:
        """
        获取指定时间的深度数据 (as-of query)

        ⚠️ 可见性规则: 只能获取 bar_time 之前已完成的 bar 数据
        """
        return self.data_manager.get_depth_bar(
            symbol=symbol,
            bar_time=bar_time,
            visibility_rule="bar_close"  # 确保不泄露未来信息
        )

    def _interpolate_impact_cost(self,
                                 depth_data: Dict,
                                 order_size_usd: float,
                                 side: str) -> float:
        """
        根据订单规模插值计算冲击成本

        预存的档位: 10k, 50k, 100k
        对于其他规模，使用线性/对数插值
        """
        suffix = "buy" if side.lower() == "buy" else "sell"

        # 获取各档位的冲击成本
        costs = {
            10_000: depth_data.get(f"impact_cost_10k_{suffix}", 0),
            50_000: depth_data.get(f"impact_cost_50k_{suffix}", 0),
            100_000: depth_data.get(f"impact_cost_100k_{suffix}", 0),
        }

        # 小于 10k: 直接用 10k 的值 (保守)
        if order_size_usd <= 10_000:
            # 线性缩放
            return costs[10_000] * (order_size_usd / 10_000)

        # 10k-50k: 线性插值
        if order_size_usd <= 50_000:
            t = (order_size_usd - 10_000) / (50_000 - 10_000)
            return costs[10_000] + t * (costs[50_000] - costs[10_000])

        # 50k-100k: 线性插值
        if order_size_usd <= 100_000:
            t = (order_size_usd - 50_000) / (100_000 - 50_000)
            return costs[50_000] + t * (costs[100_000] - costs[50_000])

        # 大于 100k: 外推 (假设线性增长)
        slope = (costs[100_000] - costs[50_000]) / 50_000
        extra = order_size_usd - 100_000
        return costs[100_000] + slope * extra

    def _get_volatility(self, symbol: str, bar_time: pd.Timestamp) -> float:
        """获取当前波动率 (用于调整滑点)"""
        # 从 DataManager 获取波动率因子
        try:
            vol = self.data_manager.get_feature(
                symbol=symbol,
                feature="volatility_24h",
                bar_time=bar_time
            )
            return vol if vol else 0.02  # 默认 2%
        except:
            return 0.02

    def _get_market_conditions(self,
                               symbol: str,
                               bar_time: pd.Timestamp) -> Dict:
        """获取市场条件 (用于 fallback 模型)"""
        return {
            "volatility": self._get_volatility(symbol, bar_time),
            "avg_daily_volume": 1e9,  # 默认值
            "bid_ask_spread": 0.0005,  # 默认 5bps
        }

    # ============== 校准方法 ==============

    def calibrate(self,
                 historical_trades: pd.DataFrame,
                 historical_depth: pd.DataFrame) -> Dict:
        """
        使用历史成交数据校准模型参数

        Args:
            historical_trades: 历史成交记录 (包含实际滑点)
                columns: [symbol, timestamp, side, size_usd, expected_price,
                          actual_avg_price, actual_slippage_bps]
            historical_depth: 历史深度数据
                columns: [symbol, timestamp, bids, asks]

        Returns:
            校准后的参数
        """
        import numpy as np
        from scipy import optimize

        calibration_results = {}

        for symbol in historical_trades["symbol"].unique():
            symbol_trades = historical_trades[
                historical_trades["symbol"] == symbol
            ]
            symbol_depth = historical_depth[
                historical_depth["symbol"] == symbol
            ]

            # 1. 计算每笔交易的预估滑点 vs 实际滑点
            errors = []
            for _, trade in symbol_trades.iterrows():
                # 找到对应时间点的深度数据
                depth_at_time = symbol_depth[
                    symbol_depth["timestamp"] <= trade["timestamp"]
                ].iloc[-1] if len(symbol_depth) > 0 else None

                if depth_at_time is not None:
                    estimated = self.estimate_slippage_from_depth(
                        size_usd=trade["size_usd"],
                        side=trade["side"],
                        depth=depth_at_time,
                    )
                    actual = trade["actual_slippage_bps"]
                    errors.append(actual - estimated)

            # 2. 计算校准因子
            if errors:
                mean_error = np.mean(errors)
                std_error = np.std(errors)

                # 校准参数: 偏移量 + 波动率调整
                calibration_results[symbol] = {
                    "bias_adjustment_bps": mean_error,
                    "volatility_multiplier": 1 + std_error / 10,
                    "sample_count": len(errors),
                    "calibration_date": pd.Timestamp.now(),
                }

        # 3. 更新内部参数
        self.calibration_params.update(calibration_results)

        return calibration_results

    def backtest_vs_actual(self,
                          symbol: str,
                          start_date: str,
                          end_date: str) -> pd.DataFrame:
        """
        对比回测滑点 vs 真实深度滑点

        用于验证模型准确性

        Returns:
            DataFrame with columns:
            - timestamp: 时间戳
            - size_usd: 订单大小
            - side: 方向
            - backtest_slippage_bps: 回测估算滑点
            - depth_slippage_bps: 真实深度计算滑点
            - error_bps: 误差
        """
        # 获取历史深度数据
        depth_data = self.data_manager.get_depth_history(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )

        # 测试不同订单大小
        test_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        results = []

        for _, depth_row in depth_data.iterrows():
            for size in test_sizes:
                for side in ["buy", "sell"]:
                    # 回测模型估算
                    backtest_slip = self.fallback_model.estimate_slippage(
                        symbol=symbol,
                        side=side,
                        order_size_usd=size,
                        bar_time=depth_row["timestamp"],
                    )

                    # 真实深度计算
                    depth_slip = self.estimate_slippage_from_depth(
                        size_usd=size,
                        side=side,
                        depth=depth_row,
                    )

                    results.append({
                        "timestamp": depth_row["timestamp"],
                        "size_usd": size,
                        "side": side,
                        "backtest_slippage_bps": backtest_slip * 10000,
                        "depth_slippage_bps": depth_slip,
                        "error_bps": (backtest_slip * 10000) - depth_slip,
                    })

        result_df = pd.DataFrame(results)

        # 添加统计摘要
        print(f"=== 滑点模型验证报告 ({symbol}) ===")
        print(f"时间范围: {start_date} ~ {end_date}")
        print(f"样本数: {len(result_df)}")
        print(f"平均误差: {result_df['error_bps'].mean():.2f} bps")
        print(f"误差标准差: {result_df['error_bps'].std():.2f} bps")
        print(f"最大低估: {result_df['error_bps'].min():.2f} bps")
        print(f"最大高估: {result_df['error_bps'].max():.2f} bps")

        return result_df


# ============== ExecutionModel 集成 ==============

class ExecutionModelV2:
    """
    执行模型 V2 - 支持校准滑点

    升级路径:
    - V1 (Step 4): DynamicSlippageModel (经验公式)
    - V2 (Step 9): CalibratedSlippageModel (真实深度)
    """

    def __init__(self,
                 config,
                 use_calibrated_slippage: bool = True):
        self.config = config
        self.fee_model = FeeModel(config.vip_level)

        # 滑点模型选择
        if use_calibrated_slippage:
            self.slippage_model = CalibratedSlippageModel(
                data_manager=config.data_manager,
                fallback_model=DynamicSlippageModel()
            )
        else:
            self.slippage_model = DynamicSlippageModel()

    def calculate_fill_price(self,
                            symbol: str,
                            side: str,
                            order_type: str,
                            order_size_usd: float,
                            bar_time: pd.Timestamp,
                            market_data: dict) -> float:
        """计算成交价格 (考虑真实滑点)"""

        if order_type == "MARKET":
            base_price = market_data['last_price']

            # 使用校准滑点模型
            if isinstance(self.slippage_model, CalibratedSlippageModel):
                estimate = self.slippage_model.estimate_slippage(
                    symbol=symbol,
                    order_size_usd=order_size_usd,
                    bar_time=bar_time,
                    side=side,
                )
                slippage = estimate.slippage_bps / 10000  # 转为小数
            else:
                slippage = self.slippage_model.estimate_slippage(
                    symbol=symbol,
                    order_size_usd=order_size_usd,
                    market_conditions=market_data,
                )

            if side == "BUY":
                return base_price * (1 + slippage)
            else:
                return base_price * (1 - slippage)
        else:
            return market_data['limit_price']
```

#### 12.11.5 数据可见性配置更新

**更新文件**: `algvex/core/data/visibility.py` (Section 11.1 定义的)

```python
# 新增深度数据的可见性规则
PUBLICATION_DELAYS = {
    # ... 已有配置 ...

    # Step 9: L2 深度数据 (C档, bar_close)
    "depth_bid_ask_spread": "bar_close",
    "depth_imbalance": "bar_close",
    "depth_1pct_bid": "bar_close",
    "depth_1pct_ask": "bar_close",
    "depth_slope_bid": "bar_close",
    "depth_slope_ask": "bar_close",
    "depth_impact_cost_buy": "bar_close",
    "depth_impact_cost_sell": "bar_close",
}

# 数据可得性分级
DATA_AVAILABILITY = {
    # ... 已有配置 ...

    # Step 9: L2 深度 (C档 - 必须自建落盘)
    "depth_bars": {
        "tier": "C",
        "history_window": "无 (必须自建)",
        "schema_stability": "★★☆",
        "notes": "WebSocket 深度数据，只能实时采集，无历史 API",
    },
}
```

#### 12.11.6 测试用例

**文件位置**: `tests/p0/test_depth_collector.py`

```python
# tests/p0/test_depth_collector.py

import pytest
from datetime import datetime, timezone
from algvex.core.data.collectors.depth import (
    DepthCollector, DepthSnapshot, AggregatedDepthBar
)


class TestDepthSnapshot:
    """深度快照测试"""

    def test_basic_metrics(self):
        """基础指标计算"""
        snapshot = DepthSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="btcusdt",
            bids=[[100000, 1.0], [99900, 2.0], [99800, 3.0]],
            asks=[[100100, 1.0], [100200, 2.0], [100300, 3.0]],
        )

        assert snapshot.best_bid == 100000
        assert snapshot.best_ask == 100100
        assert snapshot.mid_price == 100050

    def test_impact_cost_calculation(self):
        """冲击成本计算测试"""
        collector = DepthCollector(symbols=["btcusdt"])

        # 模拟订单簿
        asks = [
            [100000, 0.1],   # $10,000
            [100100, 0.2],   # $20,020
            [100200, 0.3],   # $30,060
        ]
        mid = 99950

        # 买入 $10,000 应该只吃第一档
        impact_10k = collector._calculate_impact_cost(asks, mid, 10000)
        assert impact_10k < 10  # 小于 10 bps

        # 买入 $50,000 需要吃掉多档
        impact_50k = collector._calculate_impact_cost(asks, mid, 50000)
        assert impact_50k > impact_10k  # 大单冲击更大


class TestCalibratedSlippageModel:
    """校准滑点模型测试"""

    def test_interpolation(self):
        """冲击成本插值测试"""
        model = CalibratedSlippageModel(data_manager=MockDataManager())

        # 模拟深度数据
        depth_data = {
            "impact_cost_10k_buy": 2.0,   # 2 bps
            "impact_cost_50k_buy": 5.0,   # 5 bps
            "impact_cost_100k_buy": 10.0, # 10 bps
        }

        # 测试插值
        assert model._interpolate_impact_cost(depth_data, 10000, "buy") == 2.0
        assert model._interpolate_impact_cost(depth_data, 30000, "buy") == 3.5  # 线性插值
        assert model._interpolate_impact_cost(depth_data, 100000, "buy") == 10.0

    def test_fallback_when_no_depth(self):
        """无深度数据时回退测试"""
        model = CalibratedSlippageModel(
            data_manager=MockDataManager(return_none=True),
            fallback_model=DynamicSlippageModel()
        )

        result = model.estimate_slippage(
            symbol="BTCUSDT",
            order_size_usd=10000,
            bar_time=pd.Timestamp.now(),
            use_fallback_if_missing=True
        )

        assert result.source == "fallback_model"
        assert result.confidence == "low"

    def test_visibility_compliance(self):
        """可见性规则合规测试"""
        model = CalibratedSlippageModel(data_manager=MockDataManager())

        # 不能使用未来的深度数据
        # (MockDataManager 应该只返回 bar_time 之前的数据)
        # ...
```

#### 12.11.7 验收标准

| 验收项 | 描述 | 测试方法 | 状态 |
|--------|------|----------|------|
| DepthCollector | WebSocket 连接稳定，能持续采集深度数据 | 运行 24h 无断连 | ⬜ |
| Bar 聚合 | 1m/5m 聚合逻辑正确，snapshot_count > 0 | 单元测试 | ⬜ |
| 8个深度因子 | 所有指标计算正确 (spread, imbalance, depth, slope, impact) | 单元测试 | ⬜ |
| 冲击成本 | impact_cost 与真实订单簿滑点一致 | 回放对比测试 | ⬜ |
| TimescaleDB 存储 | 数据正确写入，查询性能达标 | 压力测试 | ⬜ |
| CalibratedSlippageModel | 滑点估算比 DynamicSlippageModel 更准确 | 对比测试 | ⬜ |
| 可见性 | depth 数据使用 bar_close 规则，无未来泄露 | 泄露检测测试 | ⬜ |
| Fallback | 无深度数据时正确回退到 DynamicSlippageModel | 单元测试 | ⬜ |
| ExecutionModelV2 | 回测引擎正确使用 CalibratedSlippageModel | 集成测试 | ⬜ |
| 实盘预估 | 实盘下单前能预估冲击成本 | 手动验证 | ⬜ |

---

### 12.12 Step 10: 清算数据 (Liquidations)

> **增量价值**: 对"极端行情/瀑布/挤仓"预测比普通价量更敏感。清算级联是加密市场独特的风险特征。
>
> **工程复杂度**: 中等。WebSocket 实时采集 + bar 聚合，与 Step 9 结构类似。
>
> **数据可得性**: B档 (需自建落盘，币安有实时流但历史有限)

#### 12.12.1 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Step 10: 清算数据采集与因子计算                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    1. LiquidationCollector (WebSocket)              │   │
│  │                                                                      │   │
│  │   Binance WS ──→ 单笔清算事件 ──→ 1m/5m/1h 聚合 ──→ TimescaleDB    │   │
│  │   !forceOrder@arr   (实时推送)      (bar_close)       (持久化)       │   │
│  │                                                                      │   │
│  │   ⚠️ 可见性: bar_close (聚合后才可用于因子计算)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    2. 清算因子计算 (5个核心指标)                      │   │
│  │                                                                      │   │
│  │   liquidation_volume_long/short, liquidation_imbalance,             │   │
│  │   liquidation_spike, liquidation_momentum                           │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    3. 极端行情预警信号                                │   │
│  │                                                                      │   │
│  │   清算级联检测 → 可触发风控降仓 / 暂停开仓                            │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 12.12.2 数据源与获取方式

**币安强平数据流 (免费)**:

```python
# WebSocket 订阅地址
wss://fstream.binance.com/ws/!forceOrder@arr

# 返回数据格式
{
    "e": "forceOrder",                   # 事件类型
    "E": 1703001234567,                  # 事件时间
    "o": {
        "s": "BTCUSDT",                  # 交易对
        "S": "SELL",                     # 方向 (SELL=多头被清算, BUY=空头被清算)
        "o": "LIMIT",                    # 订单类型
        "f": "IOC",                      # 有效方式
        "q": "0.050",                    # 数量
        "p": "43000.00",                 # 价格
        "ap": "42980.00",                # 平均成交价
        "X": "FILLED",                   # 订单状态
        "l": "0.050",                    # 最新成交量
        "z": "0.050",                    # 累计成交量
        "T": 1703001234560               # 成交时间
    }
}
```

**数据可得性**:

| 项目 | 说明 |
|------|------|
| **费用** | 免费，无需 API Key |
| **延迟** | 实时推送 (<100ms) |
| **历史数据** | ❌ 无历史 API，必须自建落盘 (B档) |
| **数据量** | 平静期: ~100条/小时，极端行情: ~10000条/小时 |

#### 12.12.3 文件结构

```
algvex/core/data/collectors/
├── liquidation.py              # LiquidationCollector (新增)
│
algvex/core/data/features/
├── liquidation_features.py     # 5个清算因子计算 (新增)
│
algvex/core/risk/
├── liquidation_cascade.py      # 清算级联检测 (新增)
│
tests/p0/
├── test_liquidation_collector.py
├── test_liquidation_features.py
```

#### 12.12.4 LiquidationCollector 实现

**文件位置**: `algvex/core/data/collectors/liquidation.py`

```python
# algvex/core/data/collectors/liquidation.py

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import numpy as np
import pandas as pd
import websockets

from .base import BaseCollector


@dataclass
class LiquidationEvent:
    """单笔清算事件"""
    timestamp: datetime
    symbol: str
    side: str                    # "LONG" 或 "SHORT" (被清算方向)
    quantity: float              # 清算数量
    price: float                 # 清算价格
    notional_usd: float          # 清算金额 (USD)


@dataclass
class AggregatedLiquidationBar:
    """聚合后的清算 Bar"""
    bar_time: datetime
    symbol: str

    # 清算量统计
    volume_long: float           # 多头清算金额 (USD)
    volume_short: float          # 空头清算金额 (USD)
    count_long: int              # 多头清算笔数
    count_short: int             # 空头清算笔数

    # 派生指标
    total_volume: float          # 总清算金额
    imbalance: float             # 多空不平衡 (-1 to 1)
    avg_size: float              # 平均单笔清算金额

    # 极端行情标记
    is_spike: bool               # 是否为清算激增
    spike_ratio: float           # 相对于基准的倍数

    # 元数据
    event_count: int             # bar 内清算事件数
    visibility: str = "bar_close"


class LiquidationCollector(BaseCollector):
    """
    币安强平数据采集器 (WebSocket)

    ⚠️ 可见性规则: bar_close
    - 清算数据在 bar 结束后才能用于因子计算
    - 防止未来信息泄露

    ⚠️ 注意: 0 值是正常的 (没有清算发生)，不是缺失
    """

    # 币安 WebSocket 配置
    WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"

    # 聚合配置
    BAR_FREQUENCIES = ["1m", "5m", "1h"]

    # 清算激增阈值
    SPIKE_THRESHOLD = 3.0  # 超过24h均值的3倍视为 spike

    def __init__(self,
                 symbols: Optional[List[str]] = None,
                 bar_freq: str = "1h",
                 on_bar_complete: Optional[Callable] = None):
        """
        Args:
            symbols: 要追踪的交易对列表 (None=全部)
            bar_freq: 聚合频率 ("1m", "5m", "1h")
            on_bar_complete: bar 完成时的回调函数
        """
        self.symbols = [s.upper() for s in symbols] if symbols else None
        self.bar_freq = bar_freq
        self.on_bar_complete = on_bar_complete

        # 内存缓冲区: symbol -> 当前 bar 的事件列表
        self._buffers: Dict[str, List[LiquidationEvent]] = defaultdict(list)

        # 当前 bar 开始时间
        self._current_bar_start: Dict[str, datetime] = {}

        # 24h 滚动均值 (用于计算 spike)
        self._rolling_avg: Dict[str, float] = defaultdict(lambda: 0.0)
        self._rolling_count: Dict[str, int] = defaultdict(int)

        # 运行状态
        self._running = False
        self._ws = None

    async def start(self):
        """启动 WebSocket 连接和数据采集"""
        self._running = True

        while self._running:
            try:
                async with websockets.connect(self.WS_URL) as ws:
                    self._ws = ws
                    await self._receive_loop()
            except Exception as e:
                if self._running:
                    print(f"WebSocket disconnected: {e}, reconnecting in 5s...")
                    await asyncio.sleep(5)

    async def _receive_loop(self):
        """接收并处理清算事件"""
        async for message in self._ws:
            data = json.loads(message)

            # 解析清算事件
            order = data.get("o", {})
            symbol = order.get("s", "")

            # 过滤交易对
            if self.symbols and symbol not in self.symbols:
                continue

            # 解析方向: SELL = 多头被清算, BUY = 空头被清算
            side = "LONG" if order.get("S") == "SELL" else "SHORT"

            quantity = float(order.get("q", 0))
            price = float(order.get("ap", 0))  # 使用平均成交价
            notional = quantity * price

            event = LiquidationEvent(
                timestamp=datetime.fromtimestamp(
                    data.get("E", 0) / 1000, tz=timezone.utc
                ),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                notional_usd=notional,
            )

            # 添加到缓冲区
            self._add_to_buffer(event)

    def _add_to_buffer(self, event: LiquidationEvent):
        """添加事件到缓冲区，检查是否需要聚合"""
        symbol = event.symbol

        # 计算当前 bar 开始时间
        bar_start = self._get_bar_start(event.timestamp)

        # 检查是否需要完成上一个 bar
        if symbol in self._current_bar_start:
            if bar_start > self._current_bar_start[symbol]:
                # 完成上一个 bar
                self._complete_bar(symbol, self._current_bar_start[symbol])
                self._buffers[symbol] = []

        self._current_bar_start[symbol] = bar_start
        self._buffers[symbol].append(event)

    def _get_bar_start(self, ts: datetime) -> datetime:
        """计算 bar 开始时间"""
        if self.bar_freq == "1m":
            return ts.replace(second=0, microsecond=0)
        elif self.bar_freq == "5m":
            minute = (ts.minute // 5) * 5
            return ts.replace(minute=minute, second=0, microsecond=0)
        elif self.bar_freq == "1h":
            return ts.replace(minute=0, second=0, microsecond=0)
        else:
            raise ValueError(f"Unsupported bar_freq: {self.bar_freq}")

    def _complete_bar(self, symbol: str, bar_time: datetime):
        """聚合并输出一个完整的 bar"""
        events = self._buffers[symbol]

        # 计算聚合指标
        aggregated = self._aggregate_events(symbol, bar_time, events)

        # 更新滚动均值
        self._update_rolling_avg(symbol, aggregated.total_volume)

        # 回调
        if self.on_bar_complete:
            self.on_bar_complete(aggregated)

    def _aggregate_events(self,
                          symbol: str,
                          bar_time: datetime,
                          events: List[LiquidationEvent]) -> AggregatedLiquidationBar:
        """聚合清算事件为 bar 数据"""

        volume_long = sum(e.notional_usd for e in events if e.side == "LONG")
        volume_short = sum(e.notional_usd for e in events if e.side == "SHORT")
        count_long = sum(1 for e in events if e.side == "LONG")
        count_short = sum(1 for e in events if e.side == "SHORT")

        total_volume = volume_long + volume_short
        total_count = count_long + count_short

        # 计算不平衡度
        if total_volume > 0:
            imbalance = (volume_long - volume_short) / total_volume
        else:
            imbalance = 0.0

        # 计算平均单笔大小
        avg_size = total_volume / total_count if total_count > 0 else 0.0

        # 判断是否为 spike
        rolling_avg = self._rolling_avg.get(symbol, 0)
        if rolling_avg > 0:
            spike_ratio = total_volume / rolling_avg
            is_spike = spike_ratio >= self.SPIKE_THRESHOLD
        else:
            spike_ratio = 0.0
            is_spike = False

        return AggregatedLiquidationBar(
            bar_time=bar_time,
            symbol=symbol,
            volume_long=volume_long,
            volume_short=volume_short,
            count_long=count_long,
            count_short=count_short,
            total_volume=total_volume,
            imbalance=imbalance,
            avg_size=avg_size,
            is_spike=is_spike,
            spike_ratio=spike_ratio,
            event_count=len(events),
        )

    def _update_rolling_avg(self, symbol: str, new_volume: float):
        """更新24h滚动均值 (简化版: 指数移动平均)"""
        alpha = 0.01  # 平滑系数
        current = self._rolling_avg.get(symbol, new_volume)
        self._rolling_avg[symbol] = alpha * new_volume + (1 - alpha) * current

    async def stop(self):
        """停止采集"""
        self._running = False
        if self._ws:
            await self._ws.close()


# ============== TimescaleDB 存储 ==============

LIQUIDATION_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS liquidation_bars (
    bar_time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- 清算量
    volume_long DOUBLE PRECISION,
    volume_short DOUBLE PRECISION,
    count_long INTEGER,
    count_short INTEGER,

    -- 派生指标
    total_volume DOUBLE PRECISION,
    imbalance DOUBLE PRECISION,
    avg_size DOUBLE PRECISION,

    -- 极端行情标记
    is_spike BOOLEAN,
    spike_ratio DOUBLE PRECISION,

    -- 元数据
    event_count INTEGER,

    PRIMARY KEY (bar_time, symbol)
);

-- 创建 hypertable
SELECT create_hypertable('liquidation_bars', 'bar_time', if_not_exists => TRUE);

-- 索引
CREATE INDEX IF NOT EXISTS idx_liquidation_symbol ON liquidation_bars (symbol, bar_time DESC);
CREATE INDEX IF NOT EXISTS idx_liquidation_spike ON liquidation_bars (is_spike, bar_time DESC);
"""
```

#### 12.12.5 清算因子计算

**文件位置**: `algvex/core/data/features/liquidation_features.py`

```python
# algvex/core/data/features/liquidation_features.py

import pandas as pd
import numpy as np
from typing import Dict


class LiquidationFeatureCalculator:
    """
    清算因子计算器

    所有因子的可见性: bar_close
    """

    def calculate_features(self,
                          df: pd.DataFrame,
                          lookback_hours: int = 24) -> pd.DataFrame:
        """
        计算清算因子

        Args:
            df: 清算 bar 数据 (从 TimescaleDB 查询)
            lookback_hours: 滚动窗口小时数

        Returns:
            包含清算因子的 DataFrame
        """
        features = pd.DataFrame(index=df.index)

        # 1. 清算量 (归一化到日均值)
        features['liquidation_volume_long'] = self._normalize_volume(
            df['volume_long'], lookback_hours
        )
        features['liquidation_volume_short'] = self._normalize_volume(
            df['volume_short'], lookback_hours
        )

        # 2. 清算不平衡度 (-1 to 1)
        features['liquidation_imbalance'] = df['imbalance']

        # 3. 清算激增指标 (spike detection)
        features['liquidation_spike'] = df['spike_ratio'].clip(0, 10)  # 上限10倍

        # 4. 清算动量 (volume 变化趋势)
        features['liquidation_momentum'] = self._calculate_momentum(
            df['total_volume'], lookback_hours
        )

        return features

    def _normalize_volume(self,
                         series: pd.Series,
                         lookback_hours: int) -> pd.Series:
        """归一化清算量 (相对于滚动均值)"""
        # 假设 1h bar
        rolling_mean = series.rolling(window=lookback_hours, min_periods=1).mean()
        normalized = series / (rolling_mean + 1e-8)  # 避免除零
        return normalized.clip(0, 10)  # 上限10倍

    def _calculate_momentum(self,
                           series: pd.Series,
                           lookback_hours: int) -> pd.Series:
        """计算清算动量 (短期 vs 长期)"""
        short_window = max(1, lookback_hours // 6)  # 4h
        long_window = lookback_hours  # 24h

        short_ma = series.rolling(window=short_window, min_periods=1).mean()
        long_ma = series.rolling(window=long_window, min_periods=1).mean()

        momentum = (short_ma - long_ma) / (long_ma + 1e-8)
        return momentum.clip(-5, 5)  # 限制范围


# ============== 可见性配置 ==============

LIQUIDATION_VISIBILITY = {
    "liquidation_volume_long": "bar_close",
    "liquidation_volume_short": "bar_close",
    "liquidation_imbalance": "bar_close",
    "liquidation_spike": "bar_close",
    "liquidation_momentum": "bar_close",
}
```

#### 12.12.6 清算级联检测 (风控集成)

**文件位置**: `algvex/core/risk/liquidation_cascade.py`

```python
# algvex/core/risk/liquidation_cascade.py

from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd


@dataclass
class CascadeAlert:
    """清算级联告警"""
    timestamp: datetime
    symbol: str
    severity: str              # "warning" / "critical"
    spike_ratio: float
    imbalance: float
    recommendation: str        # "reduce_position" / "pause_new_orders"


class LiquidationCascadeDetector:
    """
    清算级联检测器

    用途:
    - 检测极端行情风险
    - 触发风控降仓 / 暂停开仓
    """

    # 告警阈值
    WARNING_SPIKE_RATIO = 3.0    # 3倍均值
    CRITICAL_SPIKE_RATIO = 5.0   # 5倍均值

    # 连续 spike 检测
    CONSECUTIVE_THRESHOLD = 3    # 连续3个bar都是spike

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self._recent_spikes: dict = {}  # symbol -> spike count

    def check(self, symbol: str) -> Optional[CascadeAlert]:
        """
        检查是否存在清算级联风险

        Returns:
            CascadeAlert if risk detected, None otherwise
        """
        # 获取最近的清算数据
        recent_bars = self.data_manager.get_liquidation_bars(
            symbol=symbol,
            lookback="3h",
        )

        if recent_bars.empty:
            return None

        latest = recent_bars.iloc[-1]

        # 检查 spike
        if latest['is_spike']:
            self._recent_spikes[symbol] = self._recent_spikes.get(symbol, 0) + 1
        else:
            self._recent_spikes[symbol] = 0

        # 判断告警级别
        spike_ratio = latest['spike_ratio']
        consecutive_count = self._recent_spikes.get(symbol, 0)

        if spike_ratio >= self.CRITICAL_SPIKE_RATIO or consecutive_count >= self.CONSECUTIVE_THRESHOLD:
            return CascadeAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                severity="critical",
                spike_ratio=spike_ratio,
                imbalance=latest['imbalance'],
                recommendation="pause_new_orders",
            )
        elif spike_ratio >= self.WARNING_SPIKE_RATIO:
            return CascadeAlert(
                timestamp=datetime.now(),
                symbol=symbol,
                severity="warning",
                spike_ratio=spike_ratio,
                imbalance=latest['imbalance'],
                recommendation="reduce_position",
            )

        return None


# ============== 与 RiskManager 集成 ==============

class RiskManagerWithLiquidation:
    """扩展 RiskManager 以支持清算级联检测"""

    def __init__(self, base_risk_manager, cascade_detector):
        self.base = base_risk_manager
        self.cascade_detector = cascade_detector

    def check_order(self, order) -> bool:
        """检查订单是否允许执行"""

        # 1. 基础风控检查
        if not self.base.check_order(order):
            return False

        # 2. 清算级联检查
        alert = self.cascade_detector.check(order.symbol)
        if alert:
            if alert.severity == "critical":
                # 暂停所有新订单
                return False
            elif alert.severity == "warning":
                # 只允许减仓订单
                if order.is_reduce_only:
                    return True
                return False

        return True
```

#### 12.12.7 测试用例

```python
# tests/p0/test_liquidation_collector.py

import pytest
from datetime import datetime, timezone
from algvex.core.data.collectors.liquidation import (
    LiquidationCollector, LiquidationEvent, AggregatedLiquidationBar
)


class TestLiquidationEvent:
    """清算事件测试"""

    def test_parse_long_liquidation(self):
        """多头清算解析"""
        raw = {
            "e": "forceOrder",
            "E": 1703001234567,
            "o": {
                "s": "BTCUSDT",
                "S": "SELL",  # 卖出 = 多头被清算
                "q": "0.1",
                "ap": "43000.00",
            }
        }
        # 解析后 side 应为 "LONG"
        # notional = 0.1 * 43000 = 4300 USD

    def test_parse_short_liquidation(self):
        """空头清算解析"""
        raw = {
            "o": {
                "s": "BTCUSDT",
                "S": "BUY",  # 买入 = 空头被清算
                "q": "0.2",
                "ap": "43500.00",
            }
        }
        # 解析后 side 应为 "SHORT"


class TestAggregation:
    """聚合逻辑测试"""

    def test_imbalance_calculation(self):
        """不平衡度计算"""
        events = [
            LiquidationEvent(..., side="LONG", notional_usd=100000),
            LiquidationEvent(..., side="SHORT", notional_usd=50000),
        ]
        # imbalance = (100000 - 50000) / 150000 = 0.333

    def test_spike_detection(self):
        """清算激增检测"""
        # 当 volume 超过 24h 均值的 3 倍时，is_spike = True


class TestCascadeDetector:
    """清算级联检测测试"""

    def test_warning_alert(self):
        """警告级别告警"""
        # spike_ratio >= 3.0 时触发 warning

    def test_critical_alert(self):
        """严重级别告警"""
        # spike_ratio >= 5.0 或连续 3 个 spike 时触发 critical
```

#### 12.12.8 验收标准

| 验收项 | 描述 | 测试方法 | 状态 |
|--------|------|----------|------|
| LiquidationCollector | WebSocket 连接稳定，能采集清算事件 | 运行 24h 验证 | ⬜ |
| 事件解析 | 正确区分多头/空头清算 | 单元测试 | ⬜ |
| Bar 聚合 | 1m/5m/1h 聚合逻辑正确 | 单元测试 | ⬜ |
| 5个清算因子 | 所有因子计算正确 | 单元测试 | ⬜ |
| Spike 检测 | 清算激增正确标记 | 回放测试 | ⬜ |
| 级联检测 | 连续 spike 正确触发 critical | 模拟测试 | ⬜ |
| 风控集成 | RiskManager 正确响应告警 | 集成测试 | ⬜ |
| 可见性 | bar_close 规则无泄露 | 泄露检测测试 | ⬜ |
| TimescaleDB | 数据正确写入和查询 | 压力测试 | ⬜ |
| 零值处理 | 无清算时正确记录 0 而非 NULL | 单元测试 | ⬜ |

---

### 12.13 Step 11: 多交易所 Basis/价差矩阵

> **增量价值**: 单一交易所 basis 容易被局部流动性扭曲，多交易所能检测"结构性偏离"与"套利压力"。
>
> **工程复杂度**: 低。REST API 轮询，无需 WebSocket。
>
> **数据可得性**: C档 (需自行计算和落盘)

#### 12.13.1 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Step 11: 多交易所 Basis/价差矩阵                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    1. MultiExchangeCollector (REST)                 │   │
│  │                                                                      │   │
│  │   Binance ─┐                                                        │   │
│  │   Bybit   ─┼──→ 价格对齐 (asof) ──→ Basis计算 ──→ TimescaleDB      │   │
│  │   OKX     ─┘      (UTC统一)          (bar_close)    (持久化)        │   │
│  │                                                                      │   │
│  │   ⚠️ 可见性: bar_close                                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    2. Basis 因子计算 (8个核心指标)                    │   │
│  │                                                                      │   │
│  │   basis_binance/bybit/okx, basis_consensus, basis_dispersion,       │   │
│  │   cross_exchange_spread, price_discovery_leader, arbitrage_pressure │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    3. 套利压力信号                                    │   │
│  │                                                                      │   │
│  │   跨所价差异常 → 可能预示大额资金流动 / 价格结构调整                   │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 12.13.2 数据源与获取方式

**三个交易所的价格 API (全部免费)**:

| 交易所 | 现货 API | 永续 API | 频率限制 |
|--------|----------|----------|----------|
| **Binance** | `GET /api/v3/ticker/price` | `GET /fapi/v1/ticker/price` | 1200/min |
| **Bybit** | `GET /v5/market/tickers?category=spot` | `GET /v5/market/tickers?category=linear` | 600/min |
| **OKX** | `GET /api/v5/market/ticker?instId=BTC-USDT` | `GET /api/v5/market/ticker?instId=BTC-USDT-SWAP` | 20/2s |

**Symbol 映射**:

```python
SYMBOL_MAPPING = {
    "BTCUSDT": {
        "binance_spot": "BTCUSDT",
        "binance_perp": "BTCUSDT",
        "bybit_spot": "BTCUSDT",
        "bybit_perp": "BTCUSDT",
        "okx_spot": "BTC-USDT",
        "okx_perp": "BTC-USDT-SWAP",
    },
    "ETHUSDT": {
        "binance_spot": "ETHUSDT",
        "binance_perp": "ETHUSDT",
        "bybit_spot": "ETHUSDT",
        "bybit_perp": "ETHUSDT",
        "okx_spot": "ETH-USDT",
        "okx_perp": "ETH-USDT-SWAP",
    },
    # ... 更多交易对
}
```

#### 12.13.3 文件结构

```
algvex/core/data/collectors/
├── multi_exchange.py           # MultiExchangeCollector (新增)
│
algvex/core/data/features/
├── basis_features.py           # 8个 basis 因子计算 (新增)
│
algvex/core/config/
├── exchange_symbols.py         # Symbol 映射配置 (新增)
│
tests/p0/
├── test_multi_exchange_collector.py
├── test_basis_features.py
```

#### 12.13.4 MultiExchangeCollector 实现

**文件位置**: `algvex/core/data/collectors/multi_exchange.py`

```python
# algvex/core/data/collectors/multi_exchange.py

import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
import pandas as pd

from .base import BaseCollector
from ..config.exchange_symbols import SYMBOL_MAPPING


@dataclass
class ExchangePrice:
    """单个交易所的价格快照"""
    timestamp: datetime
    exchange: str            # "binance" / "bybit" / "okx"
    symbol: str              # 统一 symbol (如 "BTCUSDT")
    spot_price: float
    perp_price: float
    basis_bps: float         # (spot - perp) / spot * 10000


@dataclass
class AggregatedBasisBar:
    """聚合后的 Basis Bar"""
    bar_time: datetime
    symbol: str

    # 各交易所 basis (bps)
    basis_binance: float
    basis_bybit: float
    basis_okx: float

    # 共识 basis
    basis_consensus: float   # median
    basis_dispersion: float  # std

    # 跨所价差
    cross_exchange_spread_spot: float   # max - min (bps)
    cross_exchange_spread_perp: float   # max - min (bps)

    # 套利压力指标
    arbitrage_pressure: float  # 价差回归速度

    # 元数据
    sample_count: int
    visibility: str = "bar_close"


class MultiExchangeCollector(BaseCollector):
    """
    多交易所价格采集器 (REST API 轮询)

    ⚠️ 可见性规则: bar_close
    - Basis 数据在 bar 结束后才能用于因子计算

    采集频率: 每分钟一次 (符合所有交易所的频率限制)
    """

    # 交易所 API 配置
    EXCHANGES = {
        "binance": {
            "spot_url": "https://api.binance.com/api/v3/ticker/price",
            "perp_url": "https://fapi.binance.com/fapi/v1/ticker/price",
        },
        "bybit": {
            "spot_url": "https://api.bybit.com/v5/market/tickers",
            "perp_url": "https://api.bybit.com/v5/market/tickers",
        },
        "okx": {
            "base_url": "https://www.okx.com/api/v5/market/ticker",
        },
    }

    def __init__(self,
                 symbols: List[str],
                 poll_interval: int = 60,
                 bar_freq: str = "1m"):
        """
        Args:
            symbols: 统一 symbol 列表 (如 ["BTCUSDT", "ETHUSDT"])
            poll_interval: 轮询间隔 (秒)
            bar_freq: 聚合频率
        """
        self.symbols = symbols
        self.poll_interval = poll_interval
        self.bar_freq = bar_freq

        # 内存缓冲区
        self._buffers: Dict[str, List[ExchangePrice]] = {}
        self._current_bar_start: Dict[str, datetime] = {}

        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """启动轮询采集"""
        self._running = True
        self._session = aiohttp.ClientSession()

        while self._running:
            try:
                await self._poll_all_exchanges()
                await asyncio.sleep(self.poll_interval)
            except Exception as e:
                print(f"Poll error: {e}")
                await asyncio.sleep(5)

    async def _poll_all_exchanges(self):
        """轮询所有交易所"""
        timestamp = datetime.now(timezone.utc)

        # 并发获取所有交易所价格
        tasks = [
            self._fetch_binance_prices(timestamp),
            self._fetch_bybit_prices(timestamp),
            self._fetch_okx_prices(timestamp),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并结果
        for result in results:
            if isinstance(result, list):
                for price in result:
                    self._add_to_buffer(price)

    async def _fetch_binance_prices(self, timestamp: datetime) -> List[ExchangePrice]:
        """获取币安价格"""
        prices = []

        try:
            # 获取现货价格
            async with self._session.get(self.EXCHANGES["binance"]["spot_url"]) as resp:
                spot_data = {item["symbol"]: float(item["price"])
                            for item in await resp.json()}

            # 获取永续价格
            async with self._session.get(self.EXCHANGES["binance"]["perp_url"]) as resp:
                perp_data = {item["symbol"]: float(item["price"])
                            for item in await resp.json()}

            # 计算 basis
            for symbol in self.symbols:
                mapping = SYMBOL_MAPPING.get(symbol, {})
                spot_sym = mapping.get("binance_spot", symbol)
                perp_sym = mapping.get("binance_perp", symbol)

                if spot_sym in spot_data and perp_sym in perp_data:
                    spot = spot_data[spot_sym]
                    perp = perp_data[perp_sym]
                    basis_bps = (spot - perp) / spot * 10000

                    prices.append(ExchangePrice(
                        timestamp=timestamp,
                        exchange="binance",
                        symbol=symbol,
                        spot_price=spot,
                        perp_price=perp,
                        basis_bps=basis_bps,
                    ))

        except Exception as e:
            print(f"Binance fetch error: {e}")

        return prices

    async def _fetch_bybit_prices(self, timestamp: datetime) -> List[ExchangePrice]:
        """获取 Bybit 价格"""
        prices = []

        try:
            # 现货
            url = f"{self.EXCHANGES['bybit']['spot_url']}?category=spot"
            async with self._session.get(url) as resp:
                data = await resp.json()
                spot_data = {item["symbol"]: float(item["lastPrice"])
                            for item in data.get("result", {}).get("list", [])}

            # 永续
            url = f"{self.EXCHANGES['bybit']['perp_url']}?category=linear"
            async with self._session.get(url) as resp:
                data = await resp.json()
                perp_data = {item["symbol"]: float(item["lastPrice"])
                            for item in data.get("result", {}).get("list", [])}

            for symbol in self.symbols:
                mapping = SYMBOL_MAPPING.get(symbol, {})
                spot_sym = mapping.get("bybit_spot", symbol)
                perp_sym = mapping.get("bybit_perp", symbol)

                if spot_sym in spot_data and perp_sym in perp_data:
                    spot = spot_data[spot_sym]
                    perp = perp_data[perp_sym]
                    basis_bps = (spot - perp) / spot * 10000

                    prices.append(ExchangePrice(
                        timestamp=timestamp,
                        exchange="bybit",
                        symbol=symbol,
                        spot_price=spot,
                        perp_price=perp,
                        basis_bps=basis_bps,
                    ))

        except Exception as e:
            print(f"Bybit fetch error: {e}")

        return prices

    async def _fetch_okx_prices(self, timestamp: datetime) -> List[ExchangePrice]:
        """获取 OKX 价格"""
        prices = []

        for symbol in self.symbols:
            try:
                mapping = SYMBOL_MAPPING.get(symbol, {})
                spot_sym = mapping.get("okx_spot")
                perp_sym = mapping.get("okx_perp")

                if not spot_sym or not perp_sym:
                    continue

                # 现货
                url = f"{self.EXCHANGES['okx']['base_url']}?instId={spot_sym}"
                async with self._session.get(url) as resp:
                    data = await resp.json()
                    spot = float(data["data"][0]["last"])

                # 永续
                url = f"{self.EXCHANGES['okx']['base_url']}?instId={perp_sym}"
                async with self._session.get(url) as resp:
                    data = await resp.json()
                    perp = float(data["data"][0]["last"])

                basis_bps = (spot - perp) / spot * 10000

                prices.append(ExchangePrice(
                    timestamp=timestamp,
                    exchange="okx",
                    symbol=symbol,
                    spot_price=spot,
                    perp_price=perp,
                    basis_bps=basis_bps,
                ))

            except Exception as e:
                print(f"OKX fetch error for {symbol}: {e}")

        return prices

    def _add_to_buffer(self, price: ExchangePrice):
        """添加到缓冲区"""
        key = f"{price.symbol}_{price.exchange}"
        bar_start = self._get_bar_start(price.timestamp)

        if key not in self._buffers:
            self._buffers[key] = []
            self._current_bar_start[key] = bar_start

        # 检查是否需要完成上一个 bar (简化: 由外部定时器触发)
        self._buffers[key].append(price)

    def _get_bar_start(self, ts: datetime) -> datetime:
        """计算 bar 开始时间"""
        if self.bar_freq == "1m":
            return ts.replace(second=0, microsecond=0)
        elif self.bar_freq == "5m":
            minute = (ts.minute // 5) * 5
            return ts.replace(minute=minute, second=0, microsecond=0)
        else:
            return ts.replace(minute=0, second=0, microsecond=0)

    async def stop(self):
        """停止采集"""
        self._running = False
        if self._session:
            await self._session.close()


# ============== TimescaleDB 存储 ==============

BASIS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS basis_bars (
    bar_time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,

    -- 各交易所 basis (bps)
    basis_binance DOUBLE PRECISION,
    basis_bybit DOUBLE PRECISION,
    basis_okx DOUBLE PRECISION,

    -- 共识 basis
    basis_consensus DOUBLE PRECISION,
    basis_dispersion DOUBLE PRECISION,

    -- 跨所价差
    cross_exchange_spread_spot DOUBLE PRECISION,
    cross_exchange_spread_perp DOUBLE PRECISION,

    -- 套利压力
    arbitrage_pressure DOUBLE PRECISION,

    -- 元数据
    sample_count INTEGER,

    PRIMARY KEY (bar_time, symbol)
);

SELECT create_hypertable('basis_bars', 'bar_time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_basis_symbol ON basis_bars (symbol, bar_time DESC);
"""
```

#### 12.13.5 Basis 因子计算

**文件位置**: `algvex/core/data/features/basis_features.py`

```python
# algvex/core/data/features/basis_features.py

import pandas as pd
import numpy as np
from typing import Dict, List


class BasisFeatureCalculator:
    """
    Basis 因子计算器

    所有因子的可见性: bar_close
    """

    def calculate_features(self,
                          df: pd.DataFrame,
                          lookback_hours: int = 24) -> pd.DataFrame:
        """
        计算 Basis 因子

        Args:
            df: Basis bar 数据
            lookback_hours: 滚动窗口

        Returns:
            包含 Basis 因子的 DataFrame
        """
        features = pd.DataFrame(index=df.index)

        # 1. 各交易所 basis (归一化)
        features['basis_binance'] = df['basis_binance']
        features['basis_bybit'] = df['basis_bybit']
        features['basis_okx'] = df['basis_okx']

        # 2. 共识 basis (中位数)
        basis_cols = ['basis_binance', 'basis_bybit', 'basis_okx']
        features['basis_consensus'] = df[basis_cols].median(axis=1)

        # 3. Basis 分散度 (标准差)
        features['basis_dispersion'] = df[basis_cols].std(axis=1)

        # 4. 跨所价差
        features['cross_exchange_spread'] = df['cross_exchange_spread_perp']

        # 5. 价格发现领导者 (哪个交易所的 basis 变化领先)
        features['price_discovery_leader'] = self._calculate_price_discovery(
            df, lookback_hours
        )

        # 6. 套利压力 (价差收敛速度)
        features['arbitrage_pressure'] = self._calculate_arbitrage_pressure(
            df['cross_exchange_spread_perp'], lookback_hours
        )

        return features

    def _calculate_price_discovery(self,
                                   df: pd.DataFrame,
                                   lookback_hours: int) -> pd.Series:
        """
        计算价格发现领导者

        使用各交易所 basis 变化的领先性 (简化版: 变化幅度最大的)
        """
        basis_changes = pd.DataFrame({
            'binance': df['basis_binance'].diff().abs(),
            'bybit': df['basis_bybit'].diff().abs(),
            'okx': df['basis_okx'].diff().abs(),
        })

        # 返回变化最大的交易所 (编码: binance=1, bybit=2, okx=3)
        leader_map = {'binance': 1, 'bybit': 2, 'okx': 3}
        leader = basis_changes.idxmax(axis=1)
        return leader.map(leader_map).fillna(0)

    def _calculate_arbitrage_pressure(self,
                                      spread: pd.Series,
                                      lookback_hours: int) -> pd.Series:
        """
        计算套利压力

        价差越大、收敛越慢 → 套利压力越大
        """
        # 计算价差的自相关衰减
        spread_ma = spread.rolling(window=lookback_hours, min_periods=1).mean()
        deviation = (spread - spread_ma).abs()

        # 归一化
        normalized = deviation / (spread_ma.abs() + 1e-8)
        return normalized.clip(0, 5)


# ============== 可见性配置 ==============

BASIS_VISIBILITY = {
    "basis_binance": "bar_close",
    "basis_bybit": "bar_close",
    "basis_okx": "bar_close",
    "basis_consensus": "bar_close",
    "basis_dispersion": "bar_close",
    "cross_exchange_spread": "bar_close",
    "price_discovery_leader": "bar_close",
    "arbitrage_pressure": "bar_close",
}
```

#### 12.13.6 测试用例

```python
# tests/p0/test_multi_exchange_collector.py

import pytest
from algvex.core.data.collectors.multi_exchange import (
    MultiExchangeCollector, ExchangePrice
)


class TestSymbolMapping:
    """Symbol 映射测试"""

    def test_binance_mapping(self):
        """币安 symbol 映射正确"""
        from algvex.core.config.exchange_symbols import SYMBOL_MAPPING
        assert SYMBOL_MAPPING["BTCUSDT"]["binance_spot"] == "BTCUSDT"
        assert SYMBOL_MAPPING["BTCUSDT"]["binance_perp"] == "BTCUSDT"

    def test_okx_mapping(self):
        """OKX symbol 映射正确"""
        from algvex.core.config.exchange_symbols import SYMBOL_MAPPING
        assert SYMBOL_MAPPING["BTCUSDT"]["okx_spot"] == "BTC-USDT"
        assert SYMBOL_MAPPING["BTCUSDT"]["okx_perp"] == "BTC-USDT-SWAP"


class TestBasisCalculation:
    """Basis 计算测试"""

    def test_basis_positive(self):
        """现货 > 永续时 basis 为正"""
        spot = 43500
        perp = 43400
        basis_bps = (spot - perp) / spot * 10000
        assert basis_bps > 0

    def test_basis_negative(self):
        """现货 < 永续时 basis 为负 (contango)"""
        spot = 43400
        perp = 43500
        basis_bps = (spot - perp) / spot * 10000
        assert basis_bps < 0


class TestConsensus:
    """共识 Basis 测试"""

    def test_median_calculation(self):
        """中位数计算正确"""
        import numpy as np
        basis_values = [10, 15, 12]  # bps
        consensus = np.median(basis_values)
        assert consensus == 12
```

#### 12.13.7 验收标准

| 验收项 | 描述 | 测试方法 | 状态 |
|--------|------|----------|------|
| Binance Collector | 能正确获取现货/永续价格 | API 测试 | ⬜ |
| Bybit Collector | 能正确获取现货/永续价格 | API 测试 | ⬜ |
| OKX Collector | 能正确获取现货/永续价格 | API 测试 | ⬜ |
| Symbol 映射 | 各交易所 symbol 正确映射 | 单元测试 | ⬜ |
| 时区对齐 | 所有价格统一为 UTC | 单元测试 | ⬜ |
| Basis 计算 | 各交易所 basis 计算正确 | 单元测试 | ⬜ |
| 共识 Basis | 中位数和标准差计算正确 | 单元测试 | ⬜ |
| 跨所价差 | spot/perp 价差计算正确 | 单元测试 | ⬜ |
| 8个 Basis 因子 | 所有因子计算正确 | 单元测试 | ⬜ |
| 可见性 | bar_close 规则无泄露 | 泄露检测测试 | ⬜ |
| 频率限制 | 不超过各交易所 API 限制 | 压力测试 | ⬜ |
| TimescaleDB | 数据正确写入和查询 | 集成测试 | ⬜ |

---

### 12.14 数据扩展路线图 (P2/P3 后续)

> **核心原则**: P1 数据扩展已纳入 Steps 9-11。P2/P3 待基础设施稳定后再实施。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        数据扩展状态 (P1 已完成规划)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✅ 【P1 已升级为 Steps 9-11】                                               │
│  ├─ Step 9: L2深度聚合 + 滑点校准 (Section 12.11)                           │
│  ├─ Step 10: 清算数据 + 级联检测 (Section 12.12)                            │
│  └─ Step 11: 多交易所Basis矩阵 (Section 12.13)                              │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ⏳ 【P2 中期扩展】有价值但工程量较大或口径需验证                              │
│  ├─ 链上流向交易所 (稳定币净流入, BTC/ETH大额转账)                            │
│  └─ 更细IV结构 (不同Delta/到期的skew, term structure)                        │
│                                                                             │
│  ⏸️ 【P3 谨慎扩展】免费数据不稳定，宁可晚做                                   │
│  └─ 社媒/新闻 (Reddit, Twitter, Telegram - 免费API受限严重)                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 12.14.1 P2-1: 链上流向交易所

**增量价值**: 比 DefiLlama 的供应量更接近"交易驱动"

**数据来源**: 公开链上数据 (需自建解析或使用免费API)

**可见性分级**: B/C档

| 指标 | 说明 | 难度 |
|------|------|------|
| stablecoin_exchange_netflow | 稳定币净流入交易所 | 中 |
| btc_exchange_netflow | BTC净流入交易所 | 中 |
| eth_exchange_netflow | ETH净流入交易所 | 中 |
| whale_transfer_count | 大额转账次数 (>$1M) | 低 |
| whale_transfer_volume | 大额转账金额 | 低 |

**难点**: 地址标签需手动维护

---

#### 12.14.2 P2-2: 更细IV结构

**增量价值**: 对"行情制度切换/尾部风险"更敏感

**数据来源**: Deribit API (已有)

| 指标 | 说明 |
|------|------|
| iv_25delta_put/call | 25-delta put/call的IV |
| iv_skew_25delta | 25-delta skew |
| iv_butterfly | 凸性 (wings vs ATM) |
| iv_term_slope | 近月 vs 远月 IV差 |
| vol_surface_pca_1 | 波动率曲面第一主成分 |

---

#### 12.14.3 P3: 社媒/新闻 (谨慎)

**建议**: 列为"可选/实验性C档"，Phase 3之后再考虑

---

#### 12.14.4 新增数据的准入检查清单

> **任何新增数据，都必须先回答以下四项**:

1. **可见性** - 这数据在T时刻"什么时候可见"？
2. **可得性** - 历史窗口属于A/B/C哪一档？
3. **口径稳定性** - 交易所会不会改API？
4. **增量验证** - ablation + walk-forward 验证增量存在

---

#### 12.14.5 数据扩展检查清单

| 优先级 | 数据源 | 状态 | 详细方案 |
|--------|--------|------|----------|
| **P1** | **L2深度聚合** | **→ Step 9** | Section 12.11 |
| **P1** | **清算数据** | **→ Step 10** | Section 12.12 |
| **P1** | **多交易所Basis** | **→ Step 11** | Section 12.13 |
| P2 | 链上流向交易所 | ⏳ 待实施 | Section 12.14.1 |
| P2 | 更细IV结构 | ⏳ 待实施 | Section 12.14.2 |
| P3 | 社媒/新闻 | ⏸️ 谨慎 | Section 12.14.3 |

---

## 文档总结

### 核心能力

1. **物理边界隔离** - production/ vs research/ 目录隔离 + CI导入扫描门禁
2. **DataManager唯一入口** - 禁止直接访问DB/Redis，依赖注入隔离连接信息
3. **Canonical Hashing** - 规范化序列化/排序/浮点精度，CI自动更新hash
4. **Replay确定性** - TimeProvider/SeededRandom/Decimal，消除非确定性
5. **MVP Scope配置开关** - mvp_scope.yaml 运行时强制检查因子/数据源边界
6. **双链路架构** - 生产链路不依赖Qlib，研究链路独立
7. **MVP范围明确** - 1时间框架 (5m) + 20-50标的 + 11核心因子
8. **版本化配置** - visibility.yaml + data_contracts/*.yaml + alignment.yaml
9. **哈希审计** - 所有配置有 config_version + config_hash，启动时校验
10. **Daily Replay对齐** - 可验证的回测-实盘闭环验证
11. **4轮迭代交付** - Iter-1契约 → Iter-2对齐 → Iter-3快照 → Iter-4执行层
12. **动态因子门槛** - IC基准相对化，适配加密货币高波动特性

### 完整系统能力 (研究侧)

1. **数据分级** - 6大类免费数据源，明确A/B/C三档历史可得性
2. **因子丰富** - 201个永续专用因子 (180核心 + 21 P1扩展)，**仅用于研究**
3. **P1扩展完成** - L2深度(8) + 清算(5) + 多交易所Basis(8) = 21个新因子
4. **回测可信** - 6项P0验收 + Steps 9-11 滑点校准
5. **执行可靠** - Hummingbot v2.11.0 企业级成熟度
6. **可复现** - 数据快照 + Trace Schema + 完整血缘链
7. **零数据成本** - 全部使用免费公开数据

> **MVP vs 完整系统**: MVP生产管道仅使用11个验证因子 + 3个数据源，201因子体系仅用于研究/回测，不进入生产代码。

---

*文档版本: v2.0.0 | 更新于 2025-12-31*