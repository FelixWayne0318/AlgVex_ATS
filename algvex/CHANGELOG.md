# AlgVex 变更日志

## [2.0.0] - 2025-01

### 新功能

#### Qlib 深度学习模型完整封装
- 新增 `research/qlib_models.py` - 封装 Qlib 0.9.7 全部 25+ 模型
- **GBDT 模型**: LightGBM, XGBoost, CatBoost
- **线性模型**: Linear, Ridge, Lasso
- **深度学习模型**:
  - LSTM, GRU, MLP, TCN
  - Transformer, ALSTM, TabNet
  - GATs, SFM, HIST, TRA
  - AdaRNN, IGMTF, KRNN
  - Localformer, TCTS, ADD, Sandwich
- **集成模型**: DoubleEnsemble
- 统一的 `ModelFactory` 工厂类
- GPU 加速支持

#### 多交易所连接器
- 新增 `core/execution/exchange_connectors.py`
- **支持交易所**:
  - Binance Perpetual (主网/测试网)
  - Bybit Perpetual (主网/测试网)
  - OKX Perpetual (规划中)
  - Gate.io Perpetual (规划中)
- 统一的连接器接口 (`BaseExchangeConnector`)
- 完整的订单生命周期管理
- 仓位同步和余额查询

#### 高级执行策略
- 新增 `core/execution/executors.py`
- **TWAP** (时间加权平均价格) - 大单拆分均匀执行
- **VWAP** (成交量加权平均价格) - 根据成交量分布执行
- **Grid** (网格交易) - 低买高卖自动化
- **DCA** (定投策略) - 定期定额买入
- **Iceberg** (冰山订单) - 隐藏真实订单量

#### HummingbotBridge v2.0.0
- 完全重写 `core/execution/hummingbot_bridge.py`
- 集成多交易所连接器
- 集成高级执行策略
- 幂等执行 (重放安全)
- 风控集成
- 统计和回调系统

### 改进

#### 可见性规则 (v1.1.0 → v2.0.0)
- `safety_margin` 修正为 `0s`，确保 bar_close 数据可见
- OI 延迟可见性规则优化
- `oi_change_rate` 公式修正为 `(OI[t-1] - OI[t-2]) / OI[t-2]`
- `vol_regime` 窗口修正为 8640 bars (30 天)

#### 代码清理
- 删除 `crypto_quant/` 旧版原型 (v0.1.0)
- 保留 `qlib-0.9.7/` 和 `hummingbot-2.11.0/` 作为参考

### 文件变更

#### 新增文件
```
algvex/research/qlib_models.py          # Qlib 模型封装
algvex/core/execution/exchange_connectors.py  # 交易所连接器
algvex/core/execution/executors.py      # 执行策略
algvex/CHANGELOG.md                      # 变更日志
```

#### 修改文件
```
algvex/__init__.py                       # 版本 → 2.0.0
algvex/config/visibility.yaml            # 版本 → 2.0.0
algvex/requirements.txt                  # 版本 → 2.0.0
algvex/core/execution/hummingbot_bridge.py  # 完全重写
algvex/production/factor_engine.py       # oi_change_rate 修正
algvex/shared/visibility_checker.py      # safety_margin 修正
algvex/tests/test_visibility_checker.py  # 测试更新
```

#### 删除文件
```
crypto_quant/                            # 旧版原型
```

---

## [1.1.0] - 2025-01

### 修复
- 修复 5 个设计文档与代码的逻辑矛盾
- S1: `snapshot_cutoff` vs `bar_close` 可见性冲突
- S2: OI t-1 可用性描述错误
- S3: `oi_change_rate` 公式使用不可用数据
- S8: DataManager/DataService 术语混淆
- S9: `vol_regime` MA(30) 窗口语义 (30 bars vs 30 days)

### 新增
- GitHub Actions 工作流配置
- Python 测试工作流
- 可见性配置验证

---

## [1.0.0] - 2024-12

### 初始版本
- AlgVex 核心框架
- Qlib 研究适配器
- Hummingbot 桥接器 (基础版)
- 11 个 MVP 因子
- 可见性规则系统
- 回测引擎
