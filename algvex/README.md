# AlgVex

> Qlib + Hummingbot 融合的专业加密货币量化交易平台
>
> **v2.0 更新**: 执行层升级为 Hummingbot (15k stars)，大幅提升系统成熟度和稳定性

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-orange.svg)](https://fastapi.tiangolo.com)

## 简介

AlgVex 是一个融合 Microsoft Qlib (量化研究框架) 和 Hummingbot (企业级执行引擎) 的专业量化交易平台，专注于币安永续合约交易。

### 为什么选择 Qlib + Hummingbot?

| 层级 | 框架 | 优势 |
|------|------|------|
| **信号层** | Qlib | 158+因子库、强大ML能力、完善回测 |
| **执行层** | Hummingbot | 15k stars、企业级风控、多策略支持 |

### 核心特性

- **数据采集**: 自动采集币安K线、资金费率、持仓量、多空比等数据
- **因子引擎**: 150+ 加密货币特有因子，支持自定义因子
- **机器学习**: 集成 LightGBM、XGBoost 等模型，一键训练
- **回测引擎**: 支持做空、杠杆、资金费率模拟
- **实盘交易**: 模拟盘 + 实盘，完整风控体系
- **Web界面**: 现代化响应式界面，实时图表

### 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      AlgVex Platform                        │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React)  │  API (FastAPI)  │  WebSocket           │
├─────────────────────────────────────────────────────────────┤
│  Celery Workers    │  Redis          │  PostgreSQL          │
├─────────────────────────────────────────────────────────────┤
│                    AlgVex Core                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │
│  │  Data   │ │ Factor  │ │  Model  │ │Execution│           │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Binance API  │  Alternative.me  │  DefiLlama              │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求

- Ubuntu 22.04 LTS / Windows 10+
- Python 3.11+
- Node.js 20+ (可选，用于前端)

### 1. 服务器初始化

```bash
# 下载并运行初始化脚本
curl -fsSL https://raw.githubusercontent.com/your-org/algvex/main/scripts/init_server.sh | sudo bash
```

### 2. 克隆项目

```bash
cd /opt/algvex
git clone https://github.com/your-org/algvex.git
cd algvex
```

### 3. 配置环境变量

```bash
cp .env.example .env
vim .env  # 编辑配置
```

### 4. 设置环境并启动

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 运行回测
python scripts/run_backtest.py --symbols BTCUSDT,ETHUSDT

# 启动 API 服务 (可选)
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 5. 访问

- 前端: http://localhost:3000
- API: http://localhost:8000
- API文档: http://localhost:8000/docs

## 项目结构

```
algvex/
├── core/                    # 量化引擎核心
│   ├── data/               # 数据采集与处理
│   ├── factor/             # 因子引擎
│   ├── model/              # 机器学习模型
│   ├── backtest/           # 回测引擎
│   ├── execution/          # 交易执行
│   └── strategy/           # 策略管理
├── api/                     # FastAPI 后端
│   ├── routers/            # API 路由
│   ├── models/             # 数据模型
│   ├── services/           # 业务逻辑
│   └── tasks/              # Celery 任务
├── web/                     # React 前端
├── config/                  # 配置文件
├── scripts/                 # 运维脚本
├── docs/                    # 文档
└── tests/                   # 测试
```

## 使用指南

### 数据采集

```python
from core import AlgVex

engine = AlgVex()

# 采集90天历史数据
engine.collect_data(
    symbols=['btcusdt', 'ethusdt'],
    days=90,
    interval='1h'
)
```

### 训练模型

```python
# 训练模型
result = engine.train_model(
    symbols=['btcusdt'],
    start_date='2023-01-01',
    end_date='2024-01-01',
)

print(f"准确率: {result['accuracy']:.2%}")
print(f"夏普比: {result['sharpe']:.2f}")
```

### 回测

```python
# 运行回测
backtest = engine.backtest(
    start_date='2024-01-01',
    end_date='2024-06-01',
    leverage=3,
)

print(f"总收益: {backtest['total_return']:.2%}")
print(f"最大回撤: {backtest['max_drawdown']:.2%}")
```

### 实盘交易

```python
import asyncio

# 启动模拟交易
asyncio.run(engine.start_paper_trading(
    symbols=['btcusdt'],
    rebalance_interval=3600,  # 1小时
))
```

## API 文档

### 认证

所有 API 请求需要 JWT Token:

```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -d "username=admin@algvex.com&password=admin123"
```

### 主要接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/auth/login` | POST | 用户登录 |
| `/api/strategies` | GET/POST | 策略管理 |
| `/api/backtests` | GET/POST | 回测管理 |
| `/api/trades/tasks` | GET/POST | 交易任务 |
| `/api/market/klines/{symbol}` | GET | K线数据 |

完整文档: http://localhost:8000/docs

## 部署

### 生产环境部署

```bash
# 配置 SSL 证书
sudo bash scripts/setup_ssl.sh

# 部署
bash scripts/deploy.sh --build --migrate --restart --prod
```

### 监控 (可选)

```bash
# 如需监控，可本地安装 Prometheus + Grafana
# 或使用云监控服务

# Grafana: http://localhost:3001
# Prometheus: http://localhost:9090
```

## 安全

- 所有 API 密钥加密存储
- JWT 认证 + HTTPS
- 完整的风控体系
- 定期安全审计

## 贡献

欢迎提交 Issue 和 Pull Request!

## 许可证

MIT License

## 联系

- 网站: https://algvex.com
- 邮箱: admin@algvex.com
