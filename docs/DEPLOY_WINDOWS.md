# AlgVex v10.0.4 Windows 本地部署指南

> **目标**: 在 Windows 本地电脑上从零开始部署 AlgVex 并运行教程

---

## 依赖关系说明

```
┌─────────────────────────────────────────────────────────────┐
│                    AlgVex v10.0.4 依赖结构                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【基础层】 numpy, pandas, lightgbm, pyarrow                │
│      ↓                                                      │
│  【核心脚本】 scripts/unified_features.py                   │
│              scripts/train_model.py                         │
│              scripts/backtest_offline.py                    │
│      ↓                                                      │
│  【教程 Part 0-5】 可独立运行 ✅                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  【可选】 Hummingbot (实盘交易)                             │
│      ↓                                                      │
│  【控制器】 controllers/qlib_alpha_controller.py            │
│      ↓                                                      │
│  【教程 Controller 测试】 需要 Hummingbot                   │
│  【实盘交易】 需要 Hummingbot + Binance API                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**结论**:
- **运行教程学习**: 只需基础依赖，不需要 Hummingbot
- **实盘交易**: 需要安装 Hummingbot

---

## 1. 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **操作系统** | Windows 10 | Windows 11 |
| **Python** | 3.9+ | 3.11 |
| **内存** | 8GB | 16GB |
| **硬盘** | 10GB 可用空间 | SSD |

---

## 2. 基础安装 (运行教程)

### Step 1: 安装 Python

1. 下载 Python 3.11: https://www.python.org/downloads/
2. **重要**: 安装时勾选 **"Add Python to PATH"**
3. 验证安装:
```cmd
python --version
pip --version
```

### Step 2: 安装 Git

1. 下载 Git: https://git-scm.com/download/win
2. 安装 (默认选项即可)
3. 验证安装:
```cmd
git --version
```

### Step 3: 克隆项目

```cmd
cd %USERPROFILE%
git clone https://github.com/FelixWayne0318/AlgVex_ATS.git
cd AlgVex_ATS
```

### Step 4: 创建虚拟环境

```cmd
python -m venv venv
venv\Scripts\activate
```

> 激活成功后，命令行前面会显示 `(venv)`

### Step 5: 安装基础依赖

```cmd
python -m pip install --upgrade pip
python -m pip install numpy pandas pyarrow lightgbm scikit-learn plotly jupyter notebook
```

> **注意**: Windows 上必须使用 `python -m pip` 而不是直接 `pip`，否则升级 pip 时会报错。

### Step 6: 创建数据目录

```cmd
mkdir %USERPROFILE%\.algvex\data\1h
mkdir %USERPROFILE%\.algvex\models\qlib_alpha
```

---

## 3. 准备数据

### 方法 A: 生成模拟数据 (推荐新手)

创建文件 `generate_mock_data.py`:

```python
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
    print(f'✅ Created {symbol}.parquet: {len(df)} bars')

print(f'\n数据目录: {data_dir}')
```

运行:
```cmd
python generate_mock_data.py
```

### 方法 B: 获取真实数据

```cmd
pip install ccxt
python scripts/prepare_crypto_data.py --trading-pairs BTC-USDT ETH-USDT --start-date 2023-01-01
```

---

## 4. 训练模型

```cmd
python scripts/train_model.py --instruments btcusdt ethusdt --train-start 2023-01-01 --train-end 2024-06-30
```

训练完成后检查模型文件:
```cmd
dir %USERPROFILE%\.algvex\models\qlib_alpha
```

应该看到:
```
lgb_model.txt        - LightGBM 模型
normalizer.pkl       - 归一化参数
feature_columns.pkl  - 特征列顺序
metadata.json        - 训练元信息
```

---

## 5. 运行教程

### 启动 Jupyter

```cmd
cd %USERPROFILE%\AlgVex_ATS
venv\Scripts\activate
jupyter notebook
```

### 打开教程

浏览器会自动打开，点击 `AlgVex_v10_教程_详细版.ipynb`

### 运行顺序

按 `Shift + Enter` 逐个运行 Cell:

| Part | 内容 | 依赖 | 预计时间 |
|------|------|------|----------|
| 0 | 环境搭建 | 基础 | 5 分钟 |
| 1 | 配置详解 | 基础 | 10 分钟 |
| 2 | 数据准备 | 基础 | 15 分钟 |
| 3 | 59 因子 | 基础 | 20 分钟 |
| 4 | 模型训练 | 基础 | 15 分钟 |
| 5 | 离线回测 | 基础 | 15 分钟 |
| 6 | 风险管理 | 基础 | 10 分钟 |
| 7 | Hummingbot 部署 | **需要 Hummingbot** | 可跳过 |

> **注意**: Part 7 (Hummingbot 部署) 和 Controller 测试需要安装 Hummingbot，初学者可以跳过。

---

## 6. 快速启动脚本

创建 `start_algvex.bat`:

```batch
@echo off
echo ========================================
echo   AlgVex v10.0.4 启动脚本
echo ========================================
cd /d %USERPROFILE%\AlgVex_ATS
call venv\Scripts\activate
echo.
echo 虚拟环境已激活
echo 正在启动 Jupyter Notebook...
echo.
jupyter notebook AlgVex_v10_教程_详细版.ipynb
```

双击运行即可启动教程。

---

## 7. 完整安装 (实盘交易)

如果需要实盘交易，需要额外安装 Hummingbot:

### Step 1: 安装 Hummingbot 依赖

```cmd
cd %USERPROFILE%\AlgVex_ATS\hummingbot
pip install -e .
```

### Step 2: 配置 Binance API

1. 登录 Binance，创建 API Key
2. 在 Hummingbot 中配置:
```
>>> connect binance
```

### Step 3: 启动 Paper Trading

```cmd
cd %USERPROFILE%\AlgVex_ATS\hummingbot
python bin\hummingbot.py
```

---

## 8. 验证安装

### 基础验证

```cmd
python -c "
import numpy as np
import pandas as pd
import lightgbm as lgb
print('✅ 基础依赖正常')

import sys
sys.path.insert(0, '.')
from scripts.unified_features import compute_unified_features, FEATURE_COLUMNS
print(f'✅ 统一特征模块正常 ({len(FEATURE_COLUMNS)} 个因子)')

from scripts.backtest_offline import BacktestConfig, run_backtest
print('✅ 回测模块正常')
"
```

### 完整验证

```cmd
python scripts/verify_integration.py
```

---

## 9. 目录结构

```
%USERPROFILE%\
├── AlgVex_ATS\                    # 项目代码
│   ├── scripts\                   # 核心脚本 (不依赖 Hummingbot)
│   │   ├── unified_features.py    # 59 因子计算
│   │   ├── prepare_crypto_data.py # 数据准备
│   │   ├── train_model.py         # 模型训练
│   │   ├── backtest_offline.py    # 离线回测
│   │   └── verify_integration.py  # 集成验证
│   │
│   ├── controllers\               # 控制器 (需要 Hummingbot)
│   │   └── qlib_alpha_controller.py
│   │
│   ├── hummingbot\                # Hummingbot 框架 (可选)
│   │
│   ├── AlgVex_v10_教程_详细版.ipynb
│   ├── start_algvex.bat           # 快速启动脚本
│   └── venv\                      # 虚拟环境
│
└── .algvex\                       # 数据和模型
    ├── data\
    │   └── 1h\
    │       ├── btcusdt.parquet
    │       └── ethusdt.parquet
    └── models\
        └── qlib_alpha\
            ├── lgb_model.txt
            ├── normalizer.pkl
            ├── feature_columns.pkl
            └── metadata.json
```

---

## 10. 常见问题

### Q1: pip 安装失败

```cmd
pip install --upgrade pip setuptools wheel
pip install 包名
```

### Q2: lightgbm 安装失败

```cmd
pip install lightgbm --no-binary lightgbm
```

或使用 conda:
```cmd
conda install -c conda-forge lightgbm
```

### Q3: Jupyter 无法启动

```cmd
pip uninstall notebook jupyter
pip install notebook jupyter
jupyter notebook
```

### Q4: 数据文件不存在

运行 Step 3 生成模拟数据。

### Q5: 模型文件不存在

运行 Step 4 训练模型。

### Q6: ImportError: No module named 'hummingbot'

这是正常的！基础教程不需要 Hummingbot。
跳过 Controller 测试 Cell 即可。

---

## 11. 学习路径

```
第 1 天: 环境搭建 + 数据准备
         ↓
第 2 天: 学习 59 因子 + 模型训练
         ↓
第 3 天: 离线回测 + 分析结果
         ↓
第 4 天: (可选) 安装 Hummingbot + Paper Trading
         ↓
第 5 天: (可选) 实盘部署
```

---

**版本**: v10.0.4
**更新日期**: 2026-01-03
