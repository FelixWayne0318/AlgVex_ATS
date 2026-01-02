# Qlib 参考文档

> **版本**: 1.0.0 (2026-01-02)
> **来源**: [Qlib 官方文档](https://qlib.readthedocs.io/en/latest/) | [GitHub](https://github.com/microsoft/qlib)
> **适用**: AlgVex 项目 Qlib 集成参考

---

## 目录

- [1. Qlib 概述](#1-qlib-概述)
- [2. 数据层 (Data Layer)](#2-数据层-data-layer)
- [3. 数据处理器 (Processors)](#3-数据处理器-processors)
- [4. 模型层 (Model Layer)](#4-模型层-model-layer)
- [5. 工作流 (Workflow)](#5-工作流-workflow)
- [6. 回测与策略 (Backtest & Strategy)](#6-回测与策略-backtest--strategy)
- [7. 在线服务 (Online Serving)](#7-在线服务-online-serving)
- [8. 强化学习 (Reinforcement Learning)](#8-强化学习-reinforcement-learning)
- [9. 实验管理 (Recorder)](#9-实验管理-recorder)
- [10. 与 AlgVex 项目的关系](#10-与-algvex-项目的关系)
- [附录 A: API 速查](#附录-a-api-速查)
- [附录 B: 常见问题](#附录-b-常见问题)

---

## 1. Qlib 概述

### 1.1 什么是 Qlib

Qlib 是微软研究院开发的 **AI 驱动的量化投资平台**，旨在利用 AI 技术赋能量化研究，
从创意探索到生产实现的全流程。

```
┌─────────────────────────────────────────────────────────────┐
│                      Qlib 框架层次                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Infrastructure Layer (基础设施层)                          │
│  └── 数据存储、缓存、并行计算                               │
│                                                             │
│  Data Layer (数据层)                                        │
│  └── DataHandler, Processor, Alpha158/Alpha360             │
│                                                             │
│  Interday Model Layer (日间模型层)                          │
│  └── LGBModel, LSTM, Transformer 等                        │
│                                                             │
│  Interday Strategy Layer (日间策略层)                       │
│  └── TopkDropoutStrategy, EnhancedIndexingStrategy         │
│                                                             │
│  Intraday Trading Layer (日内交易层)                        │
│  └── 嵌套决策执行框架 (Nested Decision Execution)          │
│                                                             │
│  Analysis Layer (分析层)                                    │
│  └── Recorder, Report, Visualization                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| **AI 驱动** | 支持监督学习、市场动态建模、强化学习 |
| **全栈平台** | 数据处理 → 模型训练 → 回测 → 在线服务 |
| **模块化** | 各组件可独立使用或组合 |
| **可扩展** | 支持自定义模型、数据处理器、策略 |
| **RD-Agent** | 集成 LLM 驱动的自动化研发 (2025) |

### 1.3 官方资源

- **文档**: https://qlib.readthedocs.io/en/latest/
- **GitHub**: https://github.com/microsoft/qlib
- **Qlib-Server**: https://github.com/microsoft/qlib-server
- **论文**: [Qlib: An AI-oriented Quantitative Investment Platform](https://www.microsoft.com/en-us/research/publication/qlib-an-ai-oriented-quantitative-investment-platform/)

---

## 2. 数据层 (Data Layer)

### 2.1 .bin 文件格式

Qlib 使用专门设计的二进制格式 (`.bin`) 存储金融数据，针对科学计算优化。

**文件结构**:
```
[start_index (float32)] + [data (float32...)]
```

- `start_index`: 数据在日历中的起始位置索引
- `data`: 实际数据值数组

**必需字段** (官方要求):
```
open, close, high, low, volume, factor
```

> **AlgVex 注**: 我们在 `prepare_crypto_data.py` 中正确实现了此格式，
> 包括 `factor` 字段 (加密货币设为 1.0)。

### 2.2 目录结构

```
~/.qlib/qlib_data/{dataset}/
├── calendars/           # 交易日历
│   ├── day.txt          # 日线日历 (格式: YYYY-MM-DD)
│   └── 1h.txt           # 小时级日历 (格式: YYYY-MM-DD HH:MM:SS)
├── instruments/         # 交易品种
│   └── all.txt          # 格式: {symbol}\t{start_date}\t{end_date}
├── features/            # 原始特征
│   ├── btcusdt/
│   │   ├── open.bin
│   │   ├── close.bin
│   │   ├── high.bin
│   │   ├── low.bin
│   │   ├── volume.bin
│   │   └── factor.bin   # 复权因子
│   └── ethusdt/
│       └── ...
└── cache/               # 计算缓存 (自动生成)
```

### 2.3 数据准备工具

**dump_bin.py** - 官方 CSV 转 .bin 工具:

```bash
# 将 CSV 数据转换为 Qlib 格式
python scripts/dump_bin.py dump_all \
    --csv_path ~/.qlib/csv_data/my_data \
    --qlib_dir ~/.qlib/qlib_data/my_data \
    --include_fields open,close,high,low,volume,factor
```

**CSV 格式要求**:
- 文件名为股票代码: `SH600000.csv`, `BTCUSDT.csv`
- 或包含 `symbol` 列
- 必须包含: open, close, high, low, volume, factor

**数据健康检查**:
```bash
python scripts/check_data_health.py check_data \
    --qlib_dir ~/.qlib/qlib_data/crypto_data \
    --freq 1h
```

### 2.4 DataHandler

DataHandler 负责数据加载和预处理。

**Alpha158** - 158 个人工设计因子:
```python
from qlib.contrib.data.handler import Alpha158

handler = Alpha158(
    instruments=["btcusdt", "ethusdt"],
    start_time="2023-01-01",
    end_time="2024-12-31",
    freq="1h",
    infer_processors=[],
    learn_processors=[
        {"class": "DropnaLabel"},
        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "label", "clip_outlier": True}},
    ],
)
```

**Alpha360** - 360 个原始价量特征:
- 保留原始时序结构
- 适合深度学习模型 (LSTM, Transformer)
- 特征间有强时间维度关系

**默认 Label 定义**:
```python
label = Ref($close, -2) / Ref($close, -1) - 1
# 即: T+2 相对于 T+1 的收益率
# 原因: A股 T+1 交易制度，T日收盘买入，T+1可卖出
```

> **AlgVex 注**: 加密货币无 T+1 限制，但保持此 label 定义有助于模型学习。

### 2.5 DatasetH

DatasetH 是 Qlib 的核心数据集类:

```python
from qlib.data.dataset import DatasetH

dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2023-01-01", "2024-06-30"),
        "valid": ("2024-07-01", "2024-09-30"),
        "test": ("2024-10-01", "2024-12-31"),
    },
)
```

---

## 3. 数据处理器 (Processors)

### 3.1 处理器类型

| 处理器 | 类别 | 说明 |
|--------|------|------|
| **DropnaProcessor** | 特征处理 | 删除包含 NaN 的特征 |
| **DropnaLabel** | 标签处理 | 删除包含 NaN 的标签行 (仅训练用) |
| **Fillna** | 特征处理 | 用 0 或指定值填充 NaN |
| **ProcessInf** | 特征处理 | 用列均值替换无穷值 |
| **TanhProcess** | 特征处理 | 使用 tanh 处理噪声数据 |
| **CSRankNorm** | 标准化 | 截面排名标准化 (跨股票) |
| **RobustZScoreNorm** | 标准化 | 鲁棒 Z-Score (对异常值不敏感) |

### 3.2 处理器分类

**infer_processors** (推理时使用):
- 用于预测/推理阶段
- 不处理 label
- 典型: RobustZScoreNorm + Fillna

**learn_processors** (训练时使用):
- 用于训练阶段
- 处理 label
- 典型: DropnaLabel + CSRankNorm

### 3.3 CSRankNorm vs RobustZScoreNorm

| 对比 | CSRankNorm | RobustZScoreNorm |
|------|------------|------------------|
| **原理** | 截面排名后归一化 | (x - median) / MAD |
| **适用场景** | 多品种 (>10 个) | 少量品种 (1-5 个) |
| **异常值敏感度** | 低 | 低 |
| **信息保留** | 仅保留排名 | 保留相对大小 |

> **AlgVex 决策**: 我们使用 RobustZScoreNorm，因为只交易少量加密货币。

### 3.4 配置示例

```yaml
data_handler_config:
  start_time: 2023-01-01
  end_time: 2024-12-31
  fit_start_time: 2023-01-01
  fit_end_time: 2024-06-30
  instruments: ["btcusdt", "ethusdt"]

  infer_processors:
    - class: RobustZScoreNorm
      kwargs:
        fields_group: feature
        clip_outlier: true
    - class: Fillna
      kwargs:
        fields_group: feature

  learn_processors:
    - class: DropnaLabel
    - class: RobustZScoreNorm
      kwargs:
        fields_group: label
        clip_outlier: true
```

---

## 4. 模型层 (Model Layer)

### 4.1 Model Zoo

Qlib 提供丰富的预置模型:

| 模型 | 类型 | 说明 |
|------|------|------|
| **LGBModel** | Tree | LightGBM，快速、可解释 |
| **XGBModel** | Tree | XGBoost |
| **CatBoostModel** | Tree | CatBoost，处理类别特征 |
| **LinearModel** | Linear | 线性回归 |
| **DNNModelPytorch** | DNN | 多层感知机 |
| **LSTM** | RNN | 长短期记忆网络 |
| **GRU** | RNN | 门控循环单元 |
| **ALSTM** | RNN | 注意力 LSTM |
| **Transformer** | Attention | 自注意力模型 |
| **TCN** | CNN | 时序卷积网络 |
| **GATs** | GNN | 图注意力网络 |
| **SFM** | Custom | 股票预测模型 |

### 4.2 Benchmark 性能 (Alpha158, CSI300)

| 模型 | IC | ICIR | Rank IC | Rank ICIR |
|------|-----|------|---------|-----------|
| **LightGBM** | 0.0448±0.00 | 0.3660±0.00 | 0.0469±0.00 | 0.3877±0.00 |
| **DoubleEnsemble** | 0.0521±0.00 | 0.4223±0.01 | 0.0502±0.00 | 0.4117±0.01 |
| **MLP** | 0.0376±0.00 | 0.2846±0.02 | 0.0429±0.00 | 0.3220±0.01 |
| **Linear** | 0.0332±0.00 | 0.3044±0.00 | 0.0462±0.00 | 0.4326±0.00 |

> 数据来源: [Qlib Benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)

### 4.3 LGBModel 使用

```python
from qlib.contrib.model.gbdt import LGBModel

model = LGBModel(
    loss="mse",                    # 损失函数
    early_stopping_rounds=50,      # 早停轮数
    num_boost_round=500,           # 最大迭代次数
    num_leaves=63,                 # 叶子节点数
    learning_rate=0.05,            # 学习率
    feature_fraction=0.8,          # 特征采样比例
    bagging_fraction=0.8,          # 样本采样比例
    bagging_freq=5,                # 采样频率
)

# 训练
model.fit(dataset)

# 预测
predictions = model.predict(dataset, segment="test")
```

### 4.4 自定义模型集成

继承 `ModelFT` 或 `Model` 基类:

```python
from qlib.model.base import Model

class MyModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        # 初始化模型

    def fit(self, dataset):
        # 训练逻辑
        x_train, y_train = dataset.prepare("train", col_set=["feature", "label"])
        # ...

    def predict(self, dataset, segment="test"):
        # 预测逻辑
        x_test = dataset.prepare(segment, col_set=["feature"])
        # ...
        return predictions
```

---

## 5. 工作流 (Workflow)

### 5.1 qrun 命令

`qrun` 是 Qlib 的工作流自动化工具:

```bash
# 运行完整工作流
cd examples
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
```

### 5.2 YAML 配置结构

```yaml
qlib_init:
  provider_uri: "~/.qlib/qlib_data/crypto_data"
  region: "crypto"

market: &market btcusdt ethusdt

task:
  model:
    class: LGBModel
    module_path: qlib.contrib.model.gbdt
    kwargs:
      loss: mse
      early_stopping_rounds: 50

  dataset:
    class: DatasetH
    module_path: qlib.data.dataset
    kwargs:
      handler:
        class: Alpha158
        module_path: qlib.contrib.data.handler
        kwargs:
          instruments: *market
          start_time: 2023-01-01
          end_time: 2024-12-31
      segments:
        train: [2023-01-01, 2024-06-30]
        valid: [2024-07-01, 2024-09-30]
        test: [2024-10-01, 2024-12-31]

  record:
    - class: SignalRecord
      module_path: qlib.workflow.record_temp
    - class: SigAnaRecord
      module_path: qlib.workflow.record_temp
```

### 5.3 工作流步骤

```
┌─────────────────────────────────────────────────────────────┐
│                    Qlib 工作流                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. qlib.init()           初始化 Qlib 环境                  │
│         ↓                                                   │
│  2. DataHandler           加载和处理数据                    │
│         ↓                                                   │
│  3. DatasetH              创建数据集 (train/valid/test)     │
│         ↓                                                   │
│  4. Model.fit()           模型训练                          │
│         ↓                                                   │
│  5. Model.predict()       生成预测                          │
│         ↓                                                   │
│  6. Strategy              策略生成交易信号                  │
│         ↓                                                   │
│  7. Backtest              回测评估                          │
│         ↓                                                   │
│  8. Record                记录结果到 MLflow                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. 回测与策略 (Backtest & Strategy)

### 6.1 TopkDropoutStrategy

Qlib 的核心策略实现:

```python
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily

strategy = TopkDropoutStrategy(
    signal=predictions,       # 模型预测值
    topk=50,                  # 持仓数量
    n_drop=5,                 # 每次调仓替换数量
    only_tradable=True,       # 只交易可交易品种
    hold_thresh=1,            # 最少持有天数
)

report, positions = backtest_daily(
    start_time="2024-07-01",
    end_time="2024-12-31",
    strategy=strategy,
)
```

**策略逻辑**:
1. 每日按预测分数排名所有品种
2. 卖出排名低于 K 的持仓 (n_drop 个)
3. 买入排名最高的未持仓品种 (n_drop 个)
4. 换手率 ≈ 2 × n_drop / topk

### 6.2 回测 API

```python
from qlib.backtest import backtest, executor

# 详细回测
portfolio_metric, indicator = backtest(
    executor=executor.SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
    ),
    strategy=strategy,
    start_time="2024-07-01",
    end_time="2024-12-31",
    benchmark="btcusdt",
)
```

### 6.3 评估指标

| 指标 | 含义 | 通过标准 |
|------|------|----------|
| **IC** | 预测与实际收益相关系数 | > 0.02 |
| **ICIR** | IC 的信息比率 (IC均值/IC标准差) | > 0.1 |
| **Rank IC** | 预测排名与实际排名相关系数 | > 0.02 |
| **年化收益率** | 年化后策略收益 | > 0 |
| **夏普比率** | 风险调整后收益 | > 0.5 |
| **最大回撤** | 最大亏损幅度 | < 30% |
| **换手率** | 平均每日换手 | 视策略而定 |

### 6.4 风险分析

```python
from qlib.contrib.evaluate import risk_analysis

analysis = risk_analysis(report)
print(f"Annual Return: {analysis['annual_return']:.2%}")
print(f"Sharpe Ratio: {analysis['sharpe']:.2f}")
print(f"Max Drawdown: {analysis['max_drawdown']:.2%}")
```

---

## 7. 在线服务 (Online Serving)

### 7.1 概述

Online Serving 是 Qlib 用于生产环境部署的模块集:

- **OnlineManager**: 管理多个在线策略
- **Online Strategy**: 在线策略实现
- **Online Tool**: 在线工具集
- **Updater**: 数据和模型更新器

### 7.2 在线更新流程

```
每日更新流程:
1. Update predictions    → 更新最新预测
2. Prepare tasks         → 准备任务
3. Prepare online models → 准备在线模型
4. Prepare signals       → 准备交易信号
```

### 7.3 Qlib-Server

[Qlib-Server](https://github.com/microsoft/qlib-server) 提供数据服务器部署:

**优势**:
- 集中式数据管理
- 共享缓存，减少磁盘占用
- 客户端轻量化
- 支持远程计算资源

**架构**:
```
┌─────────────┐         ┌─────────────┐
│   Client    │◀═══════▶│   Server    │
│  (Qlib)     │WebSocket│ (Flask +    │
│             │         │  SocketIO)  │
└─────────────┘         └─────────────┘
```

**部署方式**:
- 本地部署: Docker / 手动安装
- 云部署: Azure CLI 自动化脚本

### 7.4 当前限制

- 支持日级预测更新
- 不支持自动生成次日订单 (公开数据限制)

> **AlgVex 方案**: 我们使用 Hummingbot 处理订单执行，
> Qlib 仅负责信号生成。

---

## 8. 强化学习 (Reinforcement Learning)

### 8.1 RL 框架概述

Qlib 的 RL 框架支持嵌套决策执行 (Nested Decision Execution):

```
┌─────────────────────────────────────────────────────────────┐
│                    嵌套决策执行框架                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Daily Strategy (日间策略)                                  │
│  └── 生成粗粒度决策 (如: "今日买入 X 股")                  │
│          ↓                                                  │
│  Executor (执行器)                                          │
│  └── 将决策拆分为小动作                                     │
│          ↓                                                  │
│  RL Policy (强化学习策略)                                   │
│  └── 分钟级/秒级执行决策                                   │
│          ↓                                                  │
│  Simulator/Environment (模拟器)                             │
│  └── 提供日内数据、模拟成交、计算奖励                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 RL 组件

**State (状态)**:
- 订单簿状态 (买卖盘深度)
- 历史价格数据
- 历史成交量
- 市场波动率
- 部分成交统计

**Action (动作)**:
- 下限价单
- 下市价单
- 取消订单
- 跳过

**Reward (奖励)**:
- 负交易成本
- 实现 PnL 改善

### 8.3 使用场景

| 场景 | 说明 |
|------|------|
| **订单执行优化** | 最小化市场冲击成本 |
| **日内交易** | 高频交易决策 |
| **做市策略** | 动态调整买卖报价 |
| **组合再平衡** | 优化再平衡时机和方式 |

### 8.4 NestedExecutor

```python
from qlib.backtest.executor import NestedExecutor

# 嵌套执行器允许日间和日内策略联合优化
nested_executor = NestedExecutor(
    outer_executor=daily_executor,
    inner_executor=intraday_executor,
)
```

> **AlgVex 未来方向**: RL 可用于优化 Hummingbot 的订单执行策略。

---

## 9. 实验管理 (Recorder)

### 9.1 QlibRecorder

Qlib 内置实验管理系统，支持 MLflow 后端:

```python
from qlib.workflow import R

# 开始记录
with R.start(experiment_name="crypto_experiment"):
    # 训练和评估
    model.fit(dataset)
    predictions = model.predict(dataset)

    # 记录参数
    R.log_params(model_type="LGBModel", topk=50)

    # 记录指标
    R.log_metrics(IC=0.045, Sharpe=1.2)

    # 保存模型
    R.save_objects(model=model)
```

### 9.2 Record 类型

| Record | 说明 |
|--------|------|
| **SignalRecord** | 记录预测信号 |
| **SigAnaRecord** | 信号分析记录 (IC, ICIR) |
| **PortAnaRecord** | 组合分析记录 |

### 9.3 MLflow 集成

```bash
# 启动 MLflow UI
mlflow ui

# 访问 http://localhost:5000 查看实验结果
```

**功能**:
- 可视化 IC、Sharpe 等指标变化
- 对比不同实验运行
- 模型版本管理 (Model Registry)
- 存储预测结果、图表等 artifacts

### 9.4 兼容性注意

- 推荐 MLflow 版本: 1.27.0
- 部分版本 (如 1.28.0) 可能有兼容性问题
- 长配置参数可能超过 MLflow 500 字符限制

---

## 10. 与 AlgVex 项目的关系

### 10.1 已采用的 Qlib 组件

| 组件 | AlgVex 中的使用 |
|------|----------------|
| **REG_CRYPTO** | 自定义加密货币区域配置 |
| **Alpha158** | 特征工程 (训练阶段) |
| **LGBModel** | 价格预测模型 |
| **DatasetH** | 数据集管理 |
| **RobustZScoreNorm** | 标签标准化 |
| **TopkDropoutStrategy** | 回测策略 |

### 10.2 未采用的组件

| 组件 | 原因 |
|------|------|
| **官方 Crypto Collector** | 仅支持日线，不支持回测 |
| **CSRankNorm** | 不适合少量品种 |
| **Qlib-Server** | MVP 阶段使用本地部署 |
| **RL 框架** | 未来增强方向 |

### 10.3 AlgVex 定制

```
┌─────────────────────────────────────────────────────────────┐
│                    AlgVex 数据流                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Binance API                                                │
│      ↓                                                      │
│  prepare_crypto_data.py    → 自定义数据准备                 │
│      ↓                                                      │
│  Qlib .bin 格式            → 标准 Qlib 数据                 │
│      ↓                                                      │
│  Alpha158 + LGBModel       → Qlib 模型训练                  │
│      ↓                                                      │
│  backtest_model.py         → Qlib 回测验证                  │
│      ↓                                                      │
│  QlibAlphaController       → 实盘信号生成                   │
│      ↓                                                      │
│  Hummingbot V2             → 订单执行                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 10.4 未来增强方向

| 优先级 | 增强项 | Qlib 组件 |
|--------|--------|-----------|
| P2 | 深度学习模型 | LSTM, Transformer |
| P2 | 在线更新 | OnlineManager |
| P3 | 执行优化 | RL Framework |
| P3 | 分布式计算 | Qlib-Server |

---

## 附录 A: API 速查

### 初始化

```python
import qlib
from qlib.constant import REG_CRYPTO

qlib.init(
    provider_uri="~/.qlib/qlib_data/crypto_data",
    region=REG_CRYPTO,
)
```

### 数据加载

```python
from qlib.data import D

# 获取特征数据
df = D.features(
    instruments=["btcusdt"],
    fields=["$close", "$volume"],
    start_time="2024-01-01",
    end_time="2024-12-31",
    freq="1h",
)
```

### 模型训练

```python
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

handler = Alpha158(instruments=["btcusdt"], ...)
dataset = DatasetH(handler=handler, segments={...})
model = LGBModel(...)
model.fit(dataset)
```

### 回测

```python
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily

strategy = TopkDropoutStrategy(signal=predictions, topk=1)
report, positions = backtest_daily(strategy=strategy, ...)
```

---

## 附录 B: 常见问题

### Q1: REG_CRYPTO 未定义

**原因**: 未修改 Qlib 源码
**解决**: 按 CORE_PLAN.md 第 3 节修改 `constant.py` 和 `config.py`

### Q2: 数据加载失败

**原因**: 目录结构不正确或 .bin 格式错误
**解决**:
- 检查 `calendars/`, `instruments/`, `features/` 目录
- 验证 .bin 文件包含 start_index 前缀

### Q3: Alpha158 返回空数据

**原因**: 数据不足 (需要至少 60 个时间步)
**解决**: 增加历史数据量

### Q4: IC 值过低 (< 0.02)

**可能原因**:
- 特征与标签不对齐
- 市场状态变化 (模型过时)
- 数据质量问题

**解决**:
- 检查 freq 参数一致性
- 增加训练数据
- 尝试不同模型

### Q5: MLflow 记录失败

**原因**: 版本兼容性或参数过长
**解决**:
- 使用 MLflow 1.27.0
- 简化配置参数

---

## 参考链接

- [Qlib 官方文档](https://qlib.readthedocs.io/en/latest/)
- [Qlib GitHub](https://github.com/microsoft/qlib)
- [Qlib Benchmarks](https://github.com/microsoft/qlib/tree/main/examples/benchmarks)
- [Qlib-Server](https://github.com/microsoft/qlib-server)
- [Qlib Model Zoo 文档](https://qlib.readthedocs.io/en/latest/component/model.html)
- [Qlib RL 文档](https://qlib.readthedocs.io/en/latest/component/rl/overall.html)
- [Qlib Online Serving](https://qlib.readthedocs.io/en/latest/component/online.html)
- [Vadim's Qlib Blog Series](https://www.vadim.blog/tags/quantitative-trading)
