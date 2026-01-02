# AlgVex 核心方案 (P0 - MVP)

> **Qlib + Hummingbot 融合的专业加密货币量化交易平台**
>
> 本文档仅包含 MVP 核心功能，是实现优先级最高的部分。
>
> 相关文档：
> - [扩展功能 (P2)](./EXTENSION_PLAN.md) - 因子扩展、风控增强等
> - [未来规划 (P3)](./FUTURE_PLAN.md) - 开发路线图、更新日志等

---

## 目录

- [0.0 MVP Scope 定义](#00-mvp-scope-定义)
- [0.12 Iteration 交付计划](#012-iteration-1234-交付计划)
- [1. 项目概述](#1-项目概述)
- [2. 系统架构](#2-系统架构)
- [3. 数据层](#3-数据层)
- [5. 回测层](#5-回测层)
- [6. 执行层](#6-执行层)
- [11. P0 验收标准](#11-p0-验收标准-上线前必须完成)

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

