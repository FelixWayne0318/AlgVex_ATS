# AlgVex 扩展方案 (P2 - 增强功能)

> **Qlib + Hummingbot 融合的专业加密货币量化交易平台**
>
> 本文档包含扩展功能，在 MVP 完成后实现。
>
> 相关文档：
> - [核心功能 (P0)](./CORE_PLAN.md) - MVP 必须实现
> - [未来规划 (P3)](./FUTURE_PLAN.md) - 开发路线图、更新日志等

---

## 目录

- [0.1 系统规约原则 (P1-P10)](#01-系统规约原则-p1-p10--落地机制)
- [0.2 S1: 时间+快照契约](#02-s1-时间快照契约-time--snapshot-contract)
- [0.3 S2: 数据契约模板](#03-s2-数据契约模板-data-contract-template)
- [0.4 S3: 预算与降级策略](#04-s3-预算与降级策略-budget--degrade-policy)
- [0.5 S4: 因子治理](#05-s4-因子治理-factor-governance)
- [0.6 S5: 对齐与归因 + Daily Replay](#06-s5-对齐与归因--daily-replay-alignment--attribution)
- [0.7 S6: 验收测试](#07-s6-验收测试-acceptance-tests)
- [0.8 S7: 物理边界隔离](#08-s7-物理边界隔离-p0-1)
- [0.9 S8: 数据层唯一入口](#09-s8-datamanager唯一入口-p0-2)
- [0.10 S9: Canonical Hashing规范](#010-s9-canonical-hashing规范-p0-3)
- [0.11 S10: Replay确定性保障](#011-s10-replay确定性保障-p0-4)
- [0.13 硬约束层检查清单](#013-硬约束层检查清单)
- [0.14 逻辑一致性审查](#014-逻辑一致性审查v397-增补)
- [0.15 复杂度证据化](#015-复杂度证据化把复杂写成可验证的代价与收益)
- [0.16 合理决策](#016-合理决策如何保证复杂度不是乱加的不砍功能但把高配变成可控开关)
- [4. 信号层](#4-信号层)
- [7. 风控层](#7-风控层)
- [8. 技术栈](#8-技术栈)
- [9. 目录结构](#9-目录结构)
- [10. 部署方案](#10-部署方案)

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

