# AlgVex v10.0.4 Windows 本地部署指南

> **目标**: 在 Windows 本地电脑上安装 AlgVex 并运行教程

---

## 1. 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **操作系统** | Windows 10 | Windows 11 |
| **Python** | 3.9+ | 3.11 |
| **内存** | 8GB | 16GB |
| **硬盘** | 10GB 可用空间 | SSD |

---

## 2. 安装步骤

### Step 1: 安装 Python

1. 下载 Python 3.11: https://www.python.org/downloads/
2. 安装时勾选 **"Add Python to PATH"**
3. 验证安装:
```cmd
python --version
```

### Step 2: 克隆项目

```cmd
cd C:\Users\你的用户名
git clone https://github.com/FelixWayne0318/AlgVex_ATS.git
cd AlgVex_ATS
```

### Step 3: 创建虚拟环境

```cmd
python -m venv venv
venv\Scripts\activate
```

> 激活后命令行前面会显示 `(venv)`

### Step 4: 安装依赖

```cmd
pip install --upgrade pip
pip install numpy pandas lightgbm scikit-learn pyarrow plotly jupyter
pip install qlib
```

### Step 5: 创建数据目录

```cmd
mkdir %USERPROFILE%\.algvex\data\1h
mkdir %USERPROFILE%\.algvex\models\qlib_alpha
```

---

## 3. 准备数据

### 方法 A: 使用系统脚本获取真实数据

```cmd
python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT --start-date 2023-01-01
```

### 方法 B: 生成模拟数据 (快速测试)

```cmd
python -c "
import pandas as pd
import numpy as np
from pathlib import Path

data_dir = Path.home() / '.algvex' / 'data' / '1h'
data_dir.mkdir(parents=True, exist_ok=True)

# 生成 2 年模拟数据
dates = pd.date_range('2023-01-01', '2024-12-31', freq='1h', tz='UTC')
n = len(dates)

for symbol, base_price in [('btcusdt', 30000), ('ethusdt', 2000)]:
    np.random.seed(42 if symbol == 'btcusdt' else 43)
    returns = np.random.randn(n) * 0.02
    close = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(n) * 0.005),
        'high': close * (1 + np.abs(np.random.randn(n)) * 0.01),
        'low': close * (1 - np.abs(np.random.randn(n)) * 0.01),
        'close': close,
        'volume': np.random.uniform(100, 1000, n),
    }, index=dates)

    df.to_parquet(data_dir / f'{symbol}.parquet')
    print(f'Created {symbol}.parquet: {len(df)} bars')
"
```

---

## 4. 训练模型

```cmd
python scripts/train_model.py --instruments btcusdt ethusdt --train-start 2023-01-01 --train-end 2024-06-30
```

训练完成后会在 `~/.algvex/models/qlib_alpha/` 生成:
- `lgb_model.txt` - LightGBM 模型
- `normalizer.pkl` - 归一化参数
- `feature_columns.pkl` - 特征列顺序
- `metadata.json` - 训练元信息

---

## 5. 运行教程

### 启动 Jupyter

```cmd
cd C:\Users\你的用户名\AlgVex_ATS
venv\Scripts\activate
jupyter notebook
```

### 打开教程

在浏览器中打开: `AlgVex_v10_教程_详细版.ipynb`

### 运行顺序

按顺序运行每个 Cell (Shift + Enter):

| Part | 内容 | 预计时间 |
|------|------|----------|
| 0 | 环境验证 | 5 分钟 |
| 1 | 配置详解 | 10 分钟 |
| 2 | 数据准备 | 15 分钟 |
| 3 | 59 因子 | 20 分钟 |
| 4 | 模型训练 | 15 分钟 |
| 5 | 离线回测 | 15 分钟 |
| 6 | 风险管理 | 10 分钟 |
| 7 | 验收测试 | 10 分钟 |

---

## 6. 验证安装

运行系统验证脚本:

```cmd
python scripts/verify_integration.py
```

预期输出:
```
1. Testing Qlib runtime config (optional)...
2. Checking Parquet data...
3. Testing unified_features...
4. Testing FeatureNormalizer...
5. Testing model loading...
✅ All tests passed!
```

---

## 7. 目录结构

安装完成后的目录结构:

```
C:\Users\你的用户名\
├── AlgVex_ATS/                    # 项目代码
│   ├── scripts/                   # 核心脚本
│   │   ├── unified_features.py    # 59 因子计算
│   │   ├── prepare_crypto_data.py # 数据准备
│   │   ├── train_model.py         # 模型训练
│   │   ├── backtest_offline.py    # 离线回测
│   │   └── verify_integration.py  # 集成验证
│   ├── controllers/               # 控制器
│   │   └── qlib_alpha_controller.py
│   ├── AlgVex_v10_教程_详细版.ipynb
│   └── venv/                      # 虚拟环境
│
└── .algvex/                       # 数据和模型
    ├── data/
    │   └── 1h/
    │       ├── btcusdt.parquet
    │       └── ethusdt.parquet
    └── models/
        └── qlib_alpha/
            ├── lgb_model.txt
            ├── normalizer.pkl
            ├── feature_columns.pkl
            └── metadata.json
```

---

## 8. 常见问题

### Q1: pip 安装失败

```cmd
pip install --upgrade pip
pip install wheel
pip install 包名
```

### Q2: 找不到 qlib 模块

```cmd
pip install pyqlib
```

### Q3: Jupyter 无法启动

```cmd
pip install notebook
jupyter notebook
```

### Q4: 数据文件不存在

确保已运行 Step 3 (准备数据) 中的任一方法。

### Q5: 模型文件不存在

确保已运行 Step 4 (训练模型)。

---

## 9. 快速启动脚本

创建 `start_algvex.bat`:

```batch
@echo off
cd /d C:\Users\%USERNAME%\AlgVex_ATS
call venv\Scripts\activate
jupyter notebook AlgVex_v10_教程_详细版.ipynb
```

双击运行即可启动教程。

---

## 10. 下一步

完成教程后:

1. **Paper Trading**: 使用 Hummingbot 模拟交易
2. **参数优化**: 调整信号阈值、止损止盈
3. **多币种扩展**: 添加更多交易对
4. **实盘部署**: 配置 Binance API

---

**版本**: v10.0.4
**更新日期**: 2026-01-03
