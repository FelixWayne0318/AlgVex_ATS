# AlgVex 系统交接文档

> 生成时间: 2024-12-23
> 目的: 供新对话快速了解系统现状

---

## 1. 系统概述

**AlgVex** = Qlib + Hummingbot 融合的加密货币永续合约量化交易平台

### 核心架构 (4层迭代)
```
Iteration-1: 契约层 (Data Contract)     ← 数据可见性规则
Iteration-2: 对齐层 (Alignment)         ← 回测/实盘配置对齐
Iteration-3: 快照层 (Snapshot)          ← 数据版本与重放
Iteration-4: 执行层 (Execution)         ← Hummingbot 集成
```

### 技术栈
- **研究层**: Qlib (因子研究、模型训练、回测)
- **执行层**: Hummingbot (交易所连接、订单执行)
- **API层**: FastAPI + WebSocket
- **数据层**: 多源数据采集 + 规范化存储

---

## 2. 仓库结构

```
algvex/
├── api/                          # FastAPI REST API
│   ├── main.py                   # 入口: uvicorn api.main:app
│   ├── routers/                  # 路由: auth, users, strategies, backtests, trades, market
│   ├── models/                   # SQLAlchemy ORM
│   ├── schemas/                  # Pydantic 模型
│   ├── services/                 # 业务逻辑
│   ├── tasks/                    # Celery 异步任务
│   └── websocket/                # WebSocket 管理

├── core/                         # 核心业务逻辑
│   ├── alignment_checker.py      # 回测/实盘对齐检查
│   ├── canonical_hash.py         # 规范化哈希
│   ├── mvp_scope_enforcer.py     # MVP 边界强制
│   ├── data/                     # 数据管理
│   │   ├── collector.py          # 数据采集 (Binance, Bybit, OKX)
│   │   ├── handler.py            # 数据处理
│   │   ├── multi_source_manager.py # 多源管理
│   │   ├── realtime.py           # 实时数据流
│   │   ├── snapshot_manager.py   # 快照管理 (P0-5)
│   │   └── validator.py          # 数据验证
│   ├── execution/                # 执行层 (Iteration-4)
│   │   ├── hummingbot_bridge.py  # Qlib→Hummingbot 桥接
│   │   ├── order_tracker.py      # 订单追踪 (P0-3)
│   │   ├── state_synchronizer.py # 状态同步 (P0-3)
│   │   ├── position_manager.py   # 持仓管理
│   │   ├── risk_manager.py       # 风控模块
│   │   └── controllers/          # Hummingbot 控制器
│   │       └── algvex_controller.py
│   ├── backtest/                 # [空目录 - 待实现]
│   ├── factor/                   # [空目录 - 待实现 180 因子]
│   ├── model/                    # [空目录 - 待实现模型训练]
│   └── strategy/                 # [空目录 - 待实现策略框架]

├── production/                   # 生产环境组件
│   ├── signal_generator.py       # 信号生成器
│   ├── factor_engine.py          # 因子计算引擎 (11个 MVP 因子)
│   └── model_loader.py           # 模型加载器

├── research/                     # 研究环境
│   ├── qlib_adapter.py           # Qlib 适配器
│   └── factor_research.py        # 因子研究工具

├── shared/                       # 共享模块 (跨层使用)
│   ├── visibility_checker.py     # 数据可见性 (P0-1)
│   ├── price_semantics.py        # 价格语义 (P0-2) ✅ 新增
│   ├── funding_rate.py           # 资金费率 (P0-2) ✅ 新增
│   ├── walk_forward.py           # Walk-Forward验证 (P0-4) ✅ 新增
│   ├── execution_models.py       # 执行模型 (P0-6) ✅ 新增
│   ├── time_provider.py          # 时间提供器
│   ├── trace_logger.py           # Trace 日志
│   ├── seeded_random.py          # 确定性随机
│   ├── config_validator.py       # 配置验证
│   └── data_service.py           # 数据服务

├── tests/                        # 测试套件
│   ├── p0/                       # P0 验收测试 ✅ 63个测试全部通过
│   │   ├── test_alignment.py
│   │   ├── test_determinism.py
│   │   ├── test_execution_models.py  ✅ 新增
│   │   ├── test_funding_rate.py      ✅ 新增
│   │   ├── test_price_semantics.py   ✅ 新增
│   │   ├── test_walk_forward.py      ✅ 新增
│   │   └── ...
│   ├── execution/                # 执行层测试
│   ├── integration/              # 集成测试
│   └── unit/                     # 单元测试

├── scripts/                      # 脚本工具
│   ├── run_backtest.py           # 回测运行
│   ├── paper_trading.py          # 模拟交易
│   ├── daily_alignment.py        # 每日对齐检查
│   └── export_qlib_model.py      # 模型导出

├── config/                       # 配置文件
│   └── data_contracts/           # 数据契约 YAML

├── data/                         # 数据目录
│   ├── snapshots/                # 数据快照
│   ├── live_outputs/             # 实盘输出
│   └── replay_outputs/           # 重放输出

└── docs/                         # 文档
```

---

## 3. 关键入口点

### 3.1 API 服务
```bash
# 启动 API
cd algvex && uvicorn api.main:app --host 0.0.0.0 --port 8000

# 路由:
# POST /api/auth/login          - 用户登录
# GET  /api/strategies          - 策略列表
# POST /api/backtests           - 创建回测
# GET  /api/market/klines       - K线数据
# WS   /ws                      - WebSocket 实时推送
```

### 3.2 信号生成
```python
# production/signal_generator.py
from production.signal_generator import SignalGenerator

generator = SignalGenerator()
signals = generator.generate(
    symbols=["BTCUSDT", "ETHUSDT"],
    klines_data=klines_data,  # Dict[symbol, DataFrame]
)
```

### 3.3 Hummingbot 执行
```python
# core/execution/hummingbot_bridge.py
from core.execution.hummingbot_bridge import HummingbotBridge

bridge = HummingbotBridge(exchange="binance_perpetual", testnet=True)
await bridge.process_signal(signal)  # Qlib信号 → 交易订单
```

### 3.4 数据采集
```python
# core/data/collector.py
from core.data.collector import BinanceCollector

collector = BinanceCollector()
klines = await collector.fetch_klines("BTCUSDT", "5m", limit=1000)
```

### 3.5 P0 组件使用
```python
# 价格语义 (P0-2)
from shared.price_semantics import PriceSemantics, PriceScenario, PriceData
semantics = PriceSemantics()
price = semantics.get_price(PriceScenario.PNL_CALCULATION, price_data)

# 资金费率 (P0-2)
from shared.funding_rate import FundingRateHandler
handler = FundingRateHandler()
next_settlement = handler.get_next_settlement_time(current_time)

# Walk-Forward 验证 (P0-4)
from shared.walk_forward import WalkForwardValidator
validator = WalkForwardValidator(train_months=12, test_months=3)
folds = validator.create_folds(data)

# 执行模型 (P0-6)
from shared.execution_models import DynamicSlippageModel, FeeModel, VIPLevel
slippage_model = DynamicSlippageModel()
fee_model = FeeModel(exchange="binance", vip_level=VIPLevel.VIP0)
```

---

## 4. 当前进度

### 4.1 P0 验收标准 (上线必须) ✅ 100% 完成

| P0 | 内容 | 状态 | 关键文件 |
|----|------|------|----------|
| P0-1 | 数据可见性 | ✅ 100% | `shared/visibility_checker.py` |
| P0-2 | 价格语义统一 | ✅ 100% | `shared/price_semantics.py`, `shared/funding_rate.py` |
| P0-3 | 订单一致性 | ✅ 100% | `core/execution/order_tracker.py`, `state_synchronizer.py` |
| P0-4 | Walk-Forward验证 | ✅ 100% | `shared/walk_forward.py` |
| P0-5 | 数据Lineage | ✅ 100% | `core/data/snapshot_manager.py` |
| P0-6 | 执行对齐 | ✅ 100% | `shared/execution_models.py` |

**测试状态**: 63/63 通过
```bash
cd algvex && python -m pytest tests/p0/ -v
```

### 4.2 整体系统完成度

| 模块 | 完成度 | 说明 |
|------|--------|------|
| P0 验收标准 | ✅ 100% | 全部实现并测试通过 |
| API 层 | ✅ 90% | FastAPI 框架完整 |
| 执行层 | ✅ 88% | Hummingbot 集成完成 |
| 数据采集 | ⚠️ 50% | 缺 Deribit, DeFiLlama, 宏观数据 |
| 因子层 | ⚠️ 5% | 仅 11 个 MVP 因子，需 180 个 |
| 回测层 | ❌ 0% | 完全未实现 (CRITICAL) |
| 模型训练 | ⚠️ 10% | 仅有加载器，缺训练管道 |

**综合完成度: ~33%**

---

## 5. 待完成任务 (按优先级)

### 5.1 CRITICAL - 阻塞 MVP
```
1. CryptoPerpetualBacktest 回测引擎
   - 位置: core/backtest/ (空目录)
   - 需求: 永续合约回测、资金费率模拟、强平逻辑
   - 参考: AlgVex_Qlib_Hummingbot_Platform.md Section 4.2

2. 180 因子系统完整实现
   - 位置: core/factor/ (空目录)
   - 现状: 仅 11 个 MVP 因子在 production/factor_engine.py
   - 需求: 7 大类共 180 个因子
```

### 5.2 HIGH - 核心功能
```
3. 模型训练管道
   - 位置: core/model/ (空目录)
   - 需求: QlibModel 封装、训练流程、交叉验证

4. 数据采集器补全
   - DeribitCollector (期权数据)
   - DeFiLlamaCollector (链上数据)
   - MacroCollector (宏观指标)
```

### 5.3 MEDIUM - 增强功能
```
5. 策略框架
   - 位置: core/strategy/ (空目录)

6. 完整测试覆盖
   - 集成测试补全
   - 性能测试
```

---

## 6. 关键技术细节

### 6.1 价格语义映射
```python
# 永续合约不同场景使用不同价格
PNL_CALCULATION    → mark_price   # 防止操纵
LIQUIDATION_CHECK  → mark_price
ENTRY_EXIT_SIGNAL  → close_price  # K线收盘
ORDER_EXECUTION    → last_price   # 最新成交
FUNDING_SETTLEMENT → mark_price
BACKTEST_FILL      → close_price
```

### 6.2 资金费率结算
```python
# 币安永续合约结算时间 (UTC)
SETTLEMENT_HOURS = [0, 8, 16]

# 只有持仓跨越结算时间点才支付资金费
if entry_time < settlement < exit_time:
    payment = position_value * funding_rate * direction
```

### 6.3 Walk-Forward 验证
```python
# 禁止随机切分时序数据
if shuffle:
    raise RandomSplitForbiddenError("禁止随机切分时序数据!")

# 滚动窗口: 12个月训练 + 3个月测试
# 过拟合检测: train_sharpe - test_sharpe > 0.3 则报警
```

### 6.4 VIP 费率等级
```python
# 币安永续合约费率
VIP0: maker=0.02%, taker=0.04%
VIP3: maker=0.016%, taker=0.032%
VIP9: maker=0%, taker=0.017%
```

---

## 7. 用户未回答的问题

在上一对话中，用户询问了:
> "Qlib是不是也有回测层，方案中的回测层和Qlib回测层有什么不同，是在原有基础上修改，还是单独建回测层？会不会形成两套回测层？会不会造成干扰。"

**答案要点**:
1. Qlib 有内置回测 (`qlib.backtest`)，但针对股票市场
2. AlgVex 需要自建 `CryptoPerpetualBacktest` 因为:
   - 永续合约无到期日
   - 需要资金费率模拟
   - 需要强平逻辑
   - 杠杆保证金计算不同
3. 建议: 继承 Qlib Executor 接口但替换核心逻辑，不会冲突

---

## 8. 配置参考

### 8.1 环境变量 (.env)
```env
APP_ENV=development
DEBUG=true
SECRET_KEY=xxx
DATABASE_URL=postgresql+asyncpg://...
REDIS_URL=redis://localhost:6379
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
```

### 8.2 数据契约 (config/data_contracts/*.yaml)
```yaml
# 定义数据源、字段映射、验证规则
data_source: binance
asset_type: perpetual
fields:
  - name: close
    type: float
    required: true
```

---

## 9. 运行测试

```bash
# P0 验收测试 (全部通过)
cd algvex && python -m pytest tests/p0/ -v

# 执行层测试
python -m pytest tests/execution/ -v

# 全部测试
python -m pytest tests/ -v
```

---

## 10. 重要文件清单

| 文件 | 用途 | 行数 |
|------|------|------|
| `AlgVex_Qlib_Hummingbot_Platform.md` | 完整设计文档 | 8896 |
| `api/main.py` | API 入口 | 274 |
| `production/signal_generator.py` | 信号生成核心 | 433 |
| `core/execution/hummingbot_bridge.py` | 执行桥接 | ~300 |
| `shared/execution_models.py` | 执行模型 | ~350 |
| `shared/walk_forward.py` | WF验证 | ~280 |
| `shared/price_semantics.py` | 价格语义 | ~200 |
| `shared/funding_rate.py` | 资金费率 | ~300 |

---

**最后更新**: 2024-12-23
**P0 状态**: ✅ 全部通过 (63/63)
**下一步**: 实现 CryptoPerpetualBacktest 回测引擎
