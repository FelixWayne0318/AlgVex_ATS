# 币安USDT永续合约交易系统选型评估报告

**评估日期**: 2025-12-21
**评估范围**: Qlib, ValueCell, AI-Trader
**目标需求**: 币安USDT永续合约交易，支持多空双向，信号精准可复现

---

## 一、静态能力核查汇总

### 1.1 核心能力对比矩阵

| 能力维度 | Qlib | ValueCell | AI-Trader |
|---------|------|-----------|-----------|
| **1.1 交易所接入** | ❌ CoinGecko数据 | ✅ Binance USDT-M | ❌ Alpha Vantage数据 |
| **1.2 做空支持** | ❌ TODO未实现 | ✅ open_short/close_short | ❌ 仅买卖现货 |
| **1.3 杠杆/保证金** | ❌ 无 | ✅ 1-125x杠杆 | ❌ 无 |
| **1.4 资金费率** | ❌ 无 | ⚠️ 需自行获取 | ❌ 无 |
| **1.5 滑点/冲击成本** | ✅ impact_cost | ✅ CCXT市价单 | ❌ 无 |
| **1.6 信号可复现** | ✅ ML模型可复现 | ❌ LLM prompt不可复现 | ❌ LLM agent不可复现 |
| **1.7 回测能力** | ✅ 完整回测框架 | ⚠️ 仅Paper Trading | ⚠️ Historical Replay |
| **1.8 WFA/OOS验证** | ✅ RollingGen支持 | ❌ 无 | ❌ 无 |
| **1.9 评估指标** | ✅ 完整(IC/IR/MDD等) | ⚠️ 基础 | ✅ CR/SR/Vol/MDD |

### 1.2 详细证据

#### Qlib

**交易所接入** - 不支持币安
```
# 文件: scripts/data_collector/crypto/README.md
"Crypto dataset only support Data retrieval function but not support backtest function"
数据源仅支持CoinGecko,不支持交易所API
```

**做空支持** - 明确不支持
```python
# 文件: qlib/backtest/exchange.py:900
# TODO: make the trading shortable
current_amount = (
    position.get_stock_amount(order.stock_id) if position.check_stock(order.stock_id) else 0
)
```

```python
# 文件: qlib/contrib/strategy/optimizer.py:31
# The current implementation of the strategy is designed for stocks without shorting and leverage.
```

**回测能力** - 完整
```python
# 文件: qlib/backtest/exchange.py
class Exchange:
    def __init__(self, ..., impact_cost=0.0005, subscribe_fields=[]):
        # 支持滑点/冲击成本
```

**WFA支持** - 完整
```python
# 文件: qlib/workflow/task/gen.py
class RollingGen(TaskGen):
    ROLL_EX = TimeAdjuster.SHIFT_EX  # 扩展窗口
    ROLL_SD = TimeAdjuster.SHIFT_SD  # 滑动窗口
```

#### ValueCell

**币安接入** - 完整支持
```python
# 文件: python/valuecell/agents/common/trading/execution/ccxt_trading.py
class CCXTExecutionGateway(BaseExecutionGateway):
    """Supports spot, futures, and perpetual contracts"""
    def __init__(self, exchange_id, ..., default_type="swap",
                 margin_mode="cross", position_mode="oneway", ...)
```

```markdown
# 文件: README.md:113
| Binance | Only supports international site binance.com...
          Uses USDT-M futures (USDT-margined contracts). | ✅ Tested |
```

**做空支持** - 完整
```python
# 文件: python/valuecell/agents/common/trading/decision/prompt_based/system_prompt.py
ACTION SEMANTICS
- action must be one of: open_long, open_short, close_long, close_short, noop.
```

**杠杆支持**
```python
# 文件: ccxt_trading.py
default_leverage: int = 1  # 默认1x,可配置到125x
margin_mode: str = "cross"  # cross/isolated
position_mode: str = "oneway"  # oneway/hedged
```

**信号生成** - LLM不可复现
```python
# 文件: system_prompt.py
SYSTEM_PROMPT = """You are a crypto trading agent..."""
# 依赖LLM推理,同一输入可能产生不同输出
```

#### AI-Trader

**定位** - 基准测试平台,非生产交易系统
```markdown
# 文件: README.md:11
# AI-Trader: Can AI Beat the Market?
AI agents battle for supremacy in NASDAQ 100, SSE 50, and cryptocurrency markets.
Zero human input. Pure competition.
```

**加密货币交易** - 仅现货模拟
```python
# 文件: agent_tools/tool_crypto_trade.py
@mcp.tool()
def buy_crypto(symbol: str, amount: float) -> Dict[str, Any]:
    """Buy cryptocurrency function"""

@mcp.tool()
def sell_crypto(symbol: str, amount: float) -> Dict[str, Any]:
    """Sell cryptocurrency function"""
# 无short_crypto函数
```

**数据源** - Alpha Vantage
```markdown
# 文件: README.md:95
Data Integration: Alpha Vantage API combined with Jina AI market intelligence
```

**信号生成** - LLM Agent
```python
# 文件: agent/base_agent_crypto/base_agent_crypto.py
self.agent = create_agent(
    self.model,
    tools=self.tools,
    system_prompt=get_agent_system_prompt_crypto(...)
)
# 完全依赖LLM决策,不可复现
```

---

## 二、最小可运行验证

### 2.1 依赖安装状态

| 项目 | 导入测试 | 缺失依赖 | 修复路径 |
|------|---------|---------|---------|
| Qlib | ❌ 失败 | setuptools_scm | `pip install qlib` |
| ValueCell | ❌ 失败 | ccxt | `pip install ccxt` |
| AI-Trader | ❌ 失败 | numpy | `pip install -r requirements.txt` |

### 2.2 运行路径评估

**Qlib**: 可通过 `pip install qlib` 快速安装,但需要额外开发币安合约模块
**ValueCell**: 安装 `ccxt` 后可直接运行币安交易,但信号模块依赖LLM
**AI-Trader**: 主要用于研究benchmark,不适合生产环境

---

## 三、选型建议

### 3.1 需求符合度评分

| 需求项 | 权重 | Qlib | ValueCell | AI-Trader |
|-------|------|------|-----------|-----------|
| A. 币安USDT永续合约 | 30% | 0 | 100 | 0 |
| A. 做多做空 | 20% | 0 | 100 | 0 |
| B. 信号可复现/可回测 | 35% | 100 | 20 | 20 |
| C. 个人使用维护成本 | 15% | 60 | 80 | 70 |
| **加权总分** | 100% | **41** | **64** | **17** |

### 3.2 选型结论

**推荐方案: 混合架构 (Qlib信号 + ValueCell执行)**

| 模块 | 来源 | 说明 |
|-----|------|------|
| 信号生成 | Qlib | ML/DL模型,可复现,可回测 |
| 交易执行 | ValueCell | CCXT网关,支持币安USDT永续 |
| 回测验证 | Qlib | RollingGen滚动验证 |
| 实盘下单 | ValueCell | CCXTExecutionGateway |

**理由**:
1. **Qlib优势**: 完整的量化回测框架,ML模型可复现,RollingGen支持WFA验证
2. **ValueCell优势**: 生产级CCXT交易网关,已测试通过Binance USDT-M
3. **AI-Trader不适合**: 定位是研究benchmark,无生产级交易能力

---

## 四、改造方案

### 4.1 P0 (必须) - 信号+执行打通

**目标**: 将Qlib信号输出对接ValueCell执行层

**工作量**: 约 500-800 行代码

| 任务 | 说明 | 预估代码量 |
|------|------|-----------|
| 信号格式转换 | Qlib预测分数 → ValueCell动作格式 | 100行 |
| 持仓同步 | Qlib Position → ValueCell Position | 150行 |
| 做空逻辑适配 | 扩展Qlib策略支持空头信号 | 200行 |
| 杠杆计算 | 根据信号强度计算开仓量 | 150行 |

**关键文件**:
- 新建: `adapter/qlib_to_valuecell.py`
- 修改: Qlib策略输出格式

### 4.2 P1 (推荐) - 完善回测

**目标**: 在Qlib回测中模拟永续合约特性

**工作量**: 约 800-1200 行代码

| 任务 | 说明 | 预估代码量 |
|------|------|-----------|
| 做空回测 | 扩展Exchange支持负持仓 | 300行 |
| 资金费率 | 定期扣除/收取funding | 200行 |
| 杠杆保证金 | 爆仓检测/强平逻辑 | 300行 |
| 永续合约指标 | 资金费率收益统计 | 200行 |

**关键文件**:
- 修改: `qlib/backtest/exchange.py`
- 修改: `qlib/backtest/position.py`
- 新建: `qlib/contrib/strategy/perpetual_strategy.py`

### 4.3 P2 (可选) - 生产级增强

**目标**: 提升系统稳定性和可观测性

| 任务 | 说明 |
|------|------|
| 风控模块 | 最大持仓/最大亏损/频率限制 |
| 监控告警 | 持仓异常/执行失败告警 |
| 数据持久化 | 交易记录入库分析 |
| 多账户支持 | 子账户/跟单功能 |

---

## 五、改造优先级路线图

```
阶段一 (P0): Qlib信号 → ValueCell执行
  │
  ├── 1. 安装Qlib + ValueCell依赖
  ├── 2. 实现信号格式转换器
  ├── 3. 测试现货买卖流程
  └── 4. 扩展支持开空/平空

阶段二 (P1): 永续合约回测
  │
  ├── 1. 扩展Qlib Exchange做空
  ├── 2. 添加资金费率模拟
  ├── 3. 添加杠杆保证金计算
  └── 4. 完善评估指标

阶段三 (P2): 生产级增强
  │
  ├── 1. 风控规则引擎
  ├── 2. 告警通知系统
  └── 3. 交易分析仪表盘
```

---

## 六、结论

### 6.1 最终推荐

**采用 Qlib + ValueCell 混合架构**

- 使用 **Qlib** 进行策略研究、信号生成和历史回测
- 使用 **ValueCell CCXT网关** 进行币安合约实盘执行
- 开发一个 **适配器层** 连接两者

### 6.2 不推荐 AI-Trader 的原因

1. 定位是AI模型竞技平台,非生产交易系统
2. 仅支持加密货币现货,不支持永续合约
3. 依赖LLM决策,信号不可复现
4. 无真实交易所接入能力

### 6.3 风险提示

1. Qlib做空功能需要自行开发 (约300行代码)
2. 资金费率需要从交易所API获取并模拟
3. 杠杆交易有爆仓风险,需完善风控
4. 生产环境需要充分测试后再上线

---

**评估完成**

如需详细的代码实现指导,请进一步说明具体模块。
