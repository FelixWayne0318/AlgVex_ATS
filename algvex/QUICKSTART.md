# AlgVex 快速启动指南

> 5分钟内运行你的第一个加密货币量化回测

---

## 📋 前置要求

```bash
Python 3.10+
pip (Python包管理器)
```

---

## 🚀 快速开始

### Step 1: 安装依赖

```bash
cd algvex
pip install -r requirements.txt
```

### Step 2: 运行回测

```bash
# 使用模拟数据回测 (无需API)
python scripts/run_backtest.py --symbols BTCUSDT,ETHUSDT

# 使用真实数据 (自动从币安采集)
python scripts/run_backtest.py --symbols BTCUSDT --start 2024-01-01 --end 2024-06-30
```

### Step 3: 查看结果

```
==============================================================
📊 回测报告
==============================================================
总收益率: 15.32%
年化收益率: 28.45%
夏普比率: 1.85
最大回撤: -8.23%
胜率: 54.12%
==============================================================
```

---

## 📁 项目结构

```
algvex/
├── config/                 # 配置文件
│   ├── visibility.yaml    # 可见性规则 (防止未来信息泄露)
│   └── data_contracts/    # 数据契约
│
├── production/            # 生产模块 (不依赖Qlib)
│   ├── factor_engine.py   # 11个MVP因子
│   ├── model_loader.py    # 模型加载
│   └── signal_generator.py # 信号生成
│
├── research/              # 研究模块 (可用Qlib)
│   ├── qlib_adapter.py    # Qlib适配器
│   └── factor_research.py # 因子研究
│
├── shared/                # 共享模块
│   ├── visibility_checker.py  # 可见性检查
│   ├── trace_logger.py        # 信号追溯
│   └── data_service.py        # 数据服务
│
├── core/                  # 核心引擎
│   ├── data/
│   │   ├── collector.py       # 数据采集
│   │   └── snapshot_manager.py # 快照管理
│   └── replay/
│       └── replay_runner.py   # 回放运行
│
└── scripts/               # 脚本
    ├── run_backtest.py        # 回测脚本
    └── daily_alignment.py     # 对齐检查
```

---

## 🎯 MVP 11因子

| 因子族 | 因子ID | 说明 |
|--------|--------|------|
| **动量** | return_5m | 5分钟收益率 |
| | return_1h | 1小时收益率 |
| | ma_cross | 均线交叉 (MA5/MA20) |
| | breakout_20d | 20日突破 |
| | trend_strength | 趋势强度 (ADX) |
| **波动率** | atr_288 | 1日ATR |
| | realized_vol_1d | 1日已实现波动率 |
| | vol_regime | 波动率状态 |
| **订单流** | oi_change_rate | 持仓量变化率 |
| | funding_momentum | 资金费率动量 |
| | oi_funding_divergence | OI-Funding背离 |

---

## 🔧 配置

### 修改交易对

```python
# scripts/run_backtest.py
symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
```

### 修改回测参数

```python
# config/mvp_scope.yaml
mvp_backtest:
  initial_capital: 100000
  strategy:
    topk: 10
    n_drop: 3
  costs:
    open_cost: 0.0004
    close_cost: 0.0004
```

---

## 📊 使用 Qlib 训练模型 (进阶)

```python
from research.qlib_adapter import QlibAdapter

# 初始化
adapter = QlibAdapter()
adapter.init_qlib(data_path="~/.qlib/qlib_data/us_data")

# 创建数据集
dataset = adapter.create_dataset(
    instruments=["AAPL", "GOOGL"],
    start_time="2020-01-01",
    end_time="2023-12-31",
    train_end="2022-12-31",
    test_start="2023-01-01",
)

# 训练模型
model = adapter.train_model(dataset, model_type="lightgbm")

# 导出模型用于生产
adapter.export_model(model, "models/lgb_v1.pkl", features=factor_names)
```

---

## ❓ 常见问题

### Q: 如何防止未来信息泄露?

AlgVex 使用 **可见性检查器** 自动检测:

```python
from shared.visibility_checker import check_visibility

# 自动检查数据是否在信号时间可见
is_ok = check_visibility(
    source_id="open_interest_5m",  # OI有5分钟延迟
    data_time=datetime(2024, 1, 1, 10, 0),
    signal_time=datetime(2024, 1, 1, 10, 5),
)
```

### Q: 如何验证回测与实盘一致?

使用 **每日对齐检查**:

```bash
python scripts/daily_alignment.py --date 2024-01-15
```

---

## 📚 下一步

1. **学习 Qlib**: 参考 `Qlib_完整教程_入门到进阶.ipynb`
2. **自定义因子**: 修改 `production/factor_engine.py`
3. **训练模型**: 使用 `research/qlib_adapter.py`
4. **模拟交易**: 接入 Hummingbot (Phase 3)

---

## 🆘 获取帮助

- 文档: `algvex/AlgVex_Qlib_Hummingbot_Platform.md`
- 问题反馈: GitHub Issues
