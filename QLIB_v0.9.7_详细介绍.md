# Qlib v0.9.7 深度解析文档

> **版本**: v0.9.7 (2024年8月15日发布)
> **开发者**: Microsoft Research Asia
> **许可证**: MIT License
> **仓库**: https://github.com/microsoft/qlib

---

## 目录

1. [概述](#1-概述)
2. [系统架构](#2-系统架构)
3. [数据处理模块](#3-数据处理模块)
4. [模型模块](#4-模型模块)
5. [回测模块](#5-回测模块)
6. [策略与投资组合模块](#6-策略与投资组合模块)
7. [工作流与实验管理](#7-工作流与实验管理)
8. [快速开始](#8-快速开始)
9. [目录结构](#9-目录结构)

---

## 1. 概述

### 1.1 什么是 Qlib？

**Qlib** 是微软亚洲研究院开发的 **AI 导向量化投资平台**，旨在利用人工智能技术赋能量化研究，从探索想法到生产实现提供全栈支持。

### 1.2 核心特性

| 特性 | 描述 |
|------|------|
| **完整 ML 流水线** | 数据处理、模型训练、回测、分析一站式解决 |
| **多种 ML 范式** | 监督学习、市场动态建模、强化学习 |
| **量化投资全链条** | Alpha 因子挖掘、风险建模、组合优化、订单执行 |
| **高性能数据引擎** | 多层缓存、表达式计算、并行处理 |
| **实验管理** | MLflow 集成、分布式任务管理 |

### 1.3 支持的市场

- 🇨🇳 中国 A 股 (CSI300, CSI500, 全市场)
- 🇺🇸 美国股市
- 🇹🇼 台湾股市

### 1.4 v0.9.7 新特性

- 数据改进，支持 Parquet 格式
- 使用 pydantic-settings 配置 MLflow
- 多项 Bug 修复和文档更新

---

## 2. 系统架构

### 2.1 分层架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     用户应用层                              │
│  (策略开发、模型训练、回测分析)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  工作流层 (Workflow)                         │
│  - 实验管理 (MLflowExpManager)                              │
│  - 记录器 (QlibRecorder)                                    │
│  - 任务管理 (TaskManager)                                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              核心功能模块层                                 │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │   数据模块   │   模型模块   │  强化学习    │            │
│  │  (Data)      │  (Model)     │  (RL)        │            │
│  └──────────────┴──────────────┴──────────────┘            │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │   回测模块   │   策略模块   │  Contrib     │            │
│  │ (Backtest)   │ (Strategy)   │  (插件扩展)  │            │
│  └──────────────┴──────────────┴──────────────┘            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│            配置与基础设施层                                  │
│  - 配置系统 (QlibConfig)                                    │
│  - 日志系统 (QlibLogger)                                    │
│  - 缓存系统 (MemCache/DiskCache)                           │
│  - 数据提供者 (Provider)                                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 初始化流程

```python
import qlib
from qlib.config import C

# 标准初始化
qlib.init(
    provider_uri="~/.qlib/qlib_data/cn_data",  # 数据路径
    region="cn",                                 # 地区
    exp_manager={                                # 实验管理器配置
        "class": "MLflowExpManager",
        "kwargs": {"uri": "mlruns", "default_exp_name": "Experiment"}
    }
)
```

### 2.3 全局对象

| 对象 | 说明 | 用途 |
|------|------|------|
| `C` | QlibConfig | 全局配置 |
| `D` | DataProvider | 数据访问 |
| `H` | MemCache | 内存缓存 |
| `R` | QlibRecorder | 实验记录 |

---

## 3. 数据处理模块

### 3.1 数据流水线架构

```
Raw Data Source
    ↓
[Data Loader] → 加载股票、时间范围
    ↓
[Data Handler] → 管理 DataFrame，处理数据获取
    ↓
[Processors Pipeline] → 数据转换（标准化、过滤等）
    ↓
[Dataset] → 为模型准备最终数据
    ↓
[Model Training/Inference]
```

### 3.2 核心组件

#### 3.2.1 数据加载器 (DataLoader)

| 类名 | 功能 |
|------|------|
| `QlibDataLoader` | 支持表达式计算的数据加载 |
| `StaticDataLoader` | 从文件/DataFrame 加载静态数据 |
| `NestedDataLoader` | 多个 DataLoader 组合 |

#### 3.2.2 数据处理器 (Processor)

| 处理器 | 功能 |
|--------|------|
| `DropnaProcessor` | 删除缺失值 |
| `MinMaxNorm` | 最小-最大归一化 |
| `ZScoreNorm` | Z-Score 标准化 |
| `RobustZScoreNorm` | 鲁棒 Z-Score (基于中位数/MAD) |
| `CSZScoreNorm` | 截面 Z-Score (按日期分组) |
| `CSRankNorm` | 截面排名归一化 |
| `Fillna` | 缺失值填充 |

#### 3.2.3 数据集 (Dataset)

| 类名 | 功能 |
|------|------|
| `DatasetH` | 基于 DataHandler 的数据集 |
| `TSDatasetH` | 时间序列数据集 |
| `TSDataSampler` | 时间序列采样器 |

### 3.3 Alpha 因子表达式系统

Qlib 提供强大的表达式引擎，支持快速定义 Alpha 因子：

```python
# 基础特征
"$close"                    # 收盘价
"$volume"                   # 成交量

# 滚动窗口操作
"Mean($close, 5)"           # 5日均价
"Std($volume, 20)"          # 20日成交量标准差
"Max($high, 10)"            # 10日最高价
"Ref($close, 1)"            # 前一日收盘价

# 算术运算
"$close - $open"            # 日内涨跌
"$high / $low"              # 振幅

# 回归分析
"Slope($close, 5)"          # 5日价格斜率
"Rsquare($close, 5)"        # 5日拟合优度
"Resi($close, 5)"           # 5日残差

# 加权移动平均
"EMA($close, 10)"           # 10日指数加权移动平均
"WMA($close, 5)"            # 5日加权移动平均

# 复杂因子示例
"(Close - Mean(Close, 5)) / Std(Close, 5)"  # 标准化价格偏离
```

### 3.4 多层缓存策略

```
┌─────────────────────────────────────┐
│  L1: 内存缓存 (MemCache)            │
│  - LRU 淘汰策略                      │
│  - 可配置大小限制和过期时间          │
├─────────────────────────────────────┤
│  L2: 表达式缓存 (DiskExpressionCache)│
│  - 按 instrument 分目录存储          │
│  - 增量更新支持                      │
├─────────────────────────────────────┤
│  L3: 数据集缓存 (DiskDatasetCache)   │
│  - HDF5 格式存储                     │
│  - 时间索引快速查询                  │
└─────────────────────────────────────┘
```

---

## 4. 模型模块

### 4.1 模型基类设计

```python
class Model:
    def fit(self, dataset, reweighter=None):
        """训练模型"""
        pass

    def predict(self, dataset, segment="test"):
        """预测"""
        pass
```

### 4.2 机器学习模型 (6种)

| 模型 | 类名 | 特点 |
|------|------|------|
| **LightGBM** | `LGBModel` | GBDT, 可微调, 早停, 特征重要性 |
| **XGBoost** | `XGBModel` | GBDT, 特征重要性 |
| **CatBoost** | `CatBoostModel` | GBDT, 自动 GPU 支持 |
| **线性回归** | `LinearModel` | OLS/NNLS/Ridge/Lasso |
| **双重集合** | `DEnsembleModel` | 样本-特征双重重采样 |
| **高频 GBDT** | `HighfreqGDBTModel` | 高频交易特化 |

### 4.3 深度学习模型 (29+种)

#### 基础 RNN

| 模型 | 特点 |
|------|------|
| `LSTM` | 长短期记忆网络，标准序列建模 |
| `GRU` | 门控循环单元，更轻量 |

#### 注意力模型

| 模型 | 特点 |
|------|------|
| `GATs` | 图注意力网络，跨股票关联 |
| `ALSTM` | 注意力增强 LSTM |
| `Transformer` | 自注意力机制 |
| `Localformer` | 本地注意力，更高效 |

#### 混合架构

| 模型 | 特点 |
|------|------|
| `TCN` | 时间卷积网络 |
| `Sandwich` | CNN-KRNN-CNN 三明治结构 |
| `SFM` | 频域特征融合 |
| `TabNet` | 可解释的树形注意力 |

#### 高级模型

| 模型 | 特点 |
|------|------|
| `HIST` | 股票-概念图网络 |
| `TRA` | 最优传输理论应用 |
| `ADARNN` | 自适应 RNN |
| `TCTS` | 时间相关变换 |

### 4.4 模型训练示例

```python
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH

# 创建数据集
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2017-01-01", "2020-12-31"),
        "valid": ("2021-01-01", "2021-06-30"),
        "test": ("2021-07-01", "2021-12-31"),
    }
)

# 训练模型
model = LGBModel(
    loss="mse",
    num_boost_round=1000,
    early_stopping_rounds=50,
)
model.fit(dataset)

# 预测
predictions = model.predict(dataset, segment="test")
```

---

## 5. 回测模块

### 5.1 回测框架架构

```
Strategy (策略层)
    ↓ generate_trade_decision()
Decision (决策层) - Order 列表
    ↓ get_decision()
Executor (执行层)
    ↓ execute/collect_data()
Exchange (交易所层)
    ↓ deal_order()
Account (账户层)
    └─ Position (持仓层)
```

### 5.2 执行器类型

| 执行器 | 功能 |
|--------|------|
| `SimulatorExecutor` | 原子执行器（最低层） |
| `NestedExecutor` | 嵌套执行器，支持多频率 |

**执行模式**：
- **TT_SERIAL**：串行执行（卖出 → 收益 → 买入）
- **TT_PARAL**：并行执行（先卖 → 后买）

### 5.3 交易成本模型

```python
# 手续费配置
open_cost = 0.0015      # 买入手续费 0.15%
close_cost = 0.0025     # 卖出手续费 0.25%
min_cost = 5.0          # 最低手续费 5元
impact_cost = 0.001     # 市场冲击系数

# 市场冲击计算 (平方关系)
adj_cost = impact_cost * (trade_val / total_market_val) ** 2
```

### 5.4 交易限制

| 限制类型 | 说明 |
|----------|------|
| 涨跌停检查 | 默认 ±10% (A股) |
| 停牌检查 | 自动跳过停牌股票 |
| 成交量限制 | 可配置累计/实时限制 |
| 交易单位 | 中国 100 股，可配置 |
| T+1 清算 | 卖出资金次日可用 |

### 5.5 持仓管理

```python
Position = {
    "stock_id": {
        "amount": 100,       # 持仓数量
        "price": 10.5,       # 最新价格
        "weight": 0.3,       # 权重
        "count_day": 5,      # 持仓天数
    },
    "cash": 1e9,             # 现金
    "now_account_value": 1e9 # 账户总值
}
```

---

## 6. 策略与投资组合模块

### 6.1 策略类型

#### 6.1.1 基于信号的策略

| 策略 | 功能 |
|------|------|
| `TopkDropoutStrategy` | Top-K 替换策略 |
| `WeightStrategyBase` | 权重策略基类 |
| `EnhancedIndexingStrategy` | 增强指数策略 |
| `SoftTopkStrategy` | 柔性 Top-K，成本控制 |

#### 6.1.2 规则执行策略

| 策略 | 功能 |
|------|------|
| `TWAPStrategy` | 时间加权平均价格执行 |
| `SBBStrategyEMA` | 两根 K 线选最优 |
| `ACStrategy` | 波动率自适应交易 |

#### 6.1.3 强化学习策略

| 策略 | 功能 |
|------|------|
| `PPO` | 近端策略优化 |
| `DQN` | 深度 Q 网络 |

### 6.2 投资组合优化方法

| 方法 | 目标 | 需要收益 |
|------|------|----------|
| **GMV** | 全局最小方差 | ❌ |
| **MVO** | 均值方差优化 (最大夏普率) | ✅ |
| **RP** | 风险平价 (等风险贡献) | ❌ |
| **INV** | 反向波动率加权 | ❌ |
| **EnhancedIndexing** | 增强指数跟踪 | ✅ |

### 6.3 策略使用示例

```python
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

strategy = TopkDropoutStrategy(
    signal=predictions,        # 模型预测
    topk=50,                   # 持有 50 只股票
    n_drop=5,                  # 每次调整 5 只
    method_sell="bottom",      # 卖出排名最差的
    method_buy="top",          # 买入排名最好的
    hold_thresh=1,             # 最少持有 1 天
)
```

---

## 7. 工作流与实验管理

### 7.1 核心组件

```
QlibRecorder (R) - 全局接口
    ↓
ExpManager - 实验管理器
    ↓
Experiment - 单个实验
    ↓
Recorder - 运行记录
```

### 7.2 实验记录

```python
from qlib.workflow import R

# 使用上下文管理器（推荐）
with R.start(experiment_name="backtest_v1", recorder_name="xgboost"):
    # 记录参数
    R.log_params(lr=0.01, max_depth=10)

    # 训练模型
    model.fit(dataset)

    # 记录指标
    R.log_metrics(train_loss=0.25, step=0)

    # 保存对象
    R.save_objects(model=model, artifact_path="models")

    # 设置标签
    R.set_tags(model_type="XGBoost", version="v1.0")
```

### 7.3 任务管理

```python
from qlib.workflow.task.manage import TaskManager

# 创建任务管理器 (MongoDB)
manager = TaskManager("rolling_tasks")

# 插入任务
manager.insert(task_config)

# 获取并执行任务
task = manager.fetch_task()
result = execute_task(task)
manager.update_task(task["_id"], result)
```

### 7.4 MLflow 集成

- 自动记录代码 diff (`git diff`, `git status`)
- 自动记录环境变量
- 异步日志提高性能
- 支持对象序列化/反序列化

---

## 8. 快速开始

### 8.1 安装

```bash
# 使用 pip 安装
pip install pyqlib

# 或从源码安装
git clone https://github.com/microsoft/qlib.git
cd qlib
pip install -e .
```

### 8.2 下载数据

```bash
# 下载中国 A 股数据
python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### 8.3 完整示例

```python
import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.contrib.data.handler import Alpha158

# 1. 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

# 2. 数据处理
handler = Alpha158(
    instruments="csi300",
    start_time="2017-01-01",
    end_time="2021-12-31",
)

# 3. 创建数据集
from qlib.data.dataset import DatasetH
dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2017-01-01", "2019-12-31"),
        "valid": ("2020-01-01", "2020-06-30"),
        "test": ("2020-07-01", "2021-12-31"),
    }
)

# 4. 训练模型
from qlib.contrib.model.gbdt import LGBModel
model = LGBModel()

with R.start(experiment_name="quick_start"):
    model.fit(dataset)
    R.save_objects(model=model)

    # 5. 预测
    predictions = model.predict(dataset)

    # 6. 回测
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
    from qlib.backtest import backtest

    strategy = TopkDropoutStrategy(signal=predictions, topk=50, n_drop=5)
    report, positions = backtest(strategy, dataset)

    R.log_metrics(annualized_return=report["annualized_return"])
    R.save_objects(report=report, positions=positions)
```

---

## 9. 目录结构

```
qlib-0.9.7/
├── qlib/                      # 核心库代码
│   ├── __init__.py           # 初始化和配置
│   ├── config.py             # 配置系统 (QlibConfig)
│   ├── constant.py           # 常量定义
│   ├── log.py                # 日志系统
│   │
│   ├── data/                 # 数据处理模块
│   │   ├── data.py           # 数据提供者
│   │   ├── cache.py          # 缓存系统
│   │   ├── ops.py            # 表达式引擎 (Alpha 因子)
│   │   └── dataset/          # 数据集实现
│   │
│   ├── model/                # 模型基础架构
│   │   ├── base.py           # Model 基类
│   │   └── trainer.py        # 训练器
│   │
│   ├── backtest/             # 回测模块
│   │   ├── backtest.py       # 回测引擎
│   │   ├── exchange.py       # 交易所模拟
│   │   ├── account.py        # 账户管理
│   │   ├── position.py       # 持仓管理
│   │   ├── executor.py       # 执行器
│   │   └── decision.py       # 决策系统
│   │
│   ├── strategy/             # 策略基础架构
│   │   └── base.py           # BaseStrategy
│   │
│   ├── rl/                   # 强化学习模块
│   │   ├── simulator.py      # 模拟器
│   │   ├── reward.py         # 奖励计算
│   │   └── order_execution/  # 订单执行 RL
│   │
│   ├── workflow/             # 工作流模块
│   │   ├── __init__.py       # QlibRecorder (R)
│   │   ├── expm.py           # ExpManager
│   │   ├── exp.py            # Experiment
│   │   ├── recorder.py       # Recorder
│   │   └── task/             # 任务管理
│   │
│   ├── contrib/              # 扩展模块
│   │   ├── model/            # 内置模型 (35+)
│   │   ├── strategy/         # 内置策略
│   │   ├── data/             # 数据处理扩展
│   │   └── report/           # 报告生成
│   │
│   └── utils/                # 工具函数
│
├── examples/                  # 使用示例
│   ├── benchmarks/           # 模型基准测试配置
│   ├── rl_order_execution/   # RL 订单执行示例
│   └── workflow_by_code.py   # 代码工作流示例
│
├── docs/                      # 文档
├── tests/                     # 测试代码
├── scripts/                   # 脚本工具
│
├── pyproject.toml            # 项目配置
├── setup.py                  # 安装脚本
└── README.md                 # 项目说明
```

---

## 附录

### A. 支持的模型完整列表

#### 机器学习模型
1. LGBModel (LightGBM)
2. XGBModel (XGBoost)
3. CatBoostModel
4. LinearModel
5. DEnsembleModel
6. HighfreqGDBTModel

#### 深度学习模型
1. LSTM / LSTM_TS
2. GRU / GRU_TS
3. ALSTM / ALSTM_TS
4. GATs / GATs_TS
5. Transformer / Transformer_TS
6. Localformer / Localformer_TS
7. TCN / TCN_TS
8. Sandwich
9. SFM
10. TabNet
11. HIST
12. TRA
13. ADARNN
14. TCTS
15. ADD
16. IGMTF
17. KRNN
18. DNNModelPytorch
19. GeneralPTNN

### B. 配置参数参考

```python
# 完整配置示例
qlib.init(
    # 数据配置
    provider_uri="~/.qlib/qlib_data/cn_data",
    region="cn",                    # cn/us/tw

    # 缓存配置
    expression_cache="DiskExpressionCache",
    dataset_cache="DiskDatasetCache",
    mem_cache_size_limit=500,
    mem_cache_expire=3600,

    # 并行配置
    kernels=8,
    joblib_backend="multiprocessing",

    # 实验管理
    exp_manager={
        "class": "MLflowExpManager",
        "kwargs": {
            "uri": "mlruns",
            "default_exp_name": "Experiment"
        }
    },

    # Redis 配置 (可选)
    redis_host="127.0.0.1",
    redis_port=6379,

    # MongoDB 配置 (可选)
    mongo={
        "task_url": "mongodb://localhost:27017/",
        "task_db_name": "qlib_tasks"
    }
)
```

### C. 学习资源

- **官方文档**: https://qlib.readthedocs.io/
- **GitHub**: https://github.com/microsoft/qlib
- **论文**: [Qlib: An AI-oriented Quantitative Investment Platform](https://arxiv.org/abs/2009.11189)
- **RD-Agent**: https://github.com/microsoft/RD-Agent (LLM 驱动的因子挖掘)

---

> **文档生成时间**: 2026年1月
> **Qlib 版本**: v0.9.7
> **AlgVex 版本**: 2.0.0
> **分析深度**: 源代码级别
