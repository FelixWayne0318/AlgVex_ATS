# AlgVex 未来规划 (P3 - 路线图与历史)

> **Qlib + Hummingbot 融合的专业加密货币量化交易平台**
>
> 本文档包含开发路线图、更新日志和文档总结。
>
> 相关文档：
> - [核心功能 (P0)](./CORE_PLAN.md) - MVP 必须实现
> - [扩展功能 (P2)](./EXTENSION_PLAN.md) - 因子扩展、风控增强等

---

## 目录

- [更新日志](#📋-v200-更新日志-2025-12-31)
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
    BinancePerpetualConnector, BybitPerpetualConnector,
    ExchangeConfig, ExchangeType, OrderRequest, OrderSide, OrderType
)
from decimal import Decimal

# 创建配置
config = ExchangeConfig(
    exchange_type=ExchangeType.BINANCE_PERPETUAL,
    api_key="your_api_key",
    api_secret="your_api_secret",
    testnet=True  # 使用测试网
)

# 创建连接器
connector = BinancePerpetualConnector(config)
await connector.connect()

# 创建订单
order_request = OrderRequest(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal("0.01")
)
order = await connector.create_order(order_request)

# 查询持仓
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
from algvex.core.execution.exchange_connectors import OrderSide
from decimal import Decimal

# TWAP 执行
executor = TWAPExecutor(
    connector=connector,
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    total_quantity=Decimal("0.1"),
    duration_minutes=60,
    num_slices=12
)
result = await executor.execute()

# 网格交易
executor = GridExecutor(
    connector=connector,
    symbol="BTCUSDT",
    total_quantity=Decimal("1.0"),
    lower_price=Decimal("40000"),
    upper_price=Decimal("45000"),
    num_grids=10
)
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