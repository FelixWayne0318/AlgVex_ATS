# Qlib 进阶教程：可视化分析、Alpha158因子、回测评估

> 📚 本教程是《Qlib小白入门指南》的延续，涵盖 Step 2-4 的详细内容

---

## 目录

- [Step 2: Jupyter 可视化分析](#step-2-jupyter-可视化分析)
  - [2.1 数据可视化基础](#21-数据可视化基础)
  - [2.2 股票行情可视化](#22-股票行情可视化)
  - [2.3 因子分布可视化](#23-因子分布可视化)
  - [2.4 预测结果可视化](#24-预测结果可视化)
  - [2.5 交互式图表](#25-交互式图表)
- [Step 3: 理解 Alpha158 因子](#step-3-理解-alpha158-因子)
  - [3.1 什么是因子](#31-什么是因子)
  - [3.2 Alpha158 因子库概览](#32-alpha158-因子库概览)
  - [3.3 六大类因子详解](#33-六大类因子详解)
  - [3.4 因子计算实战](#34-因子计算实战)
  - [3.5 自定义因子](#35-自定义因子)
- [Step 4: 回测与评估指标](#step-4-回测与评估指标)
  - [4.1 回测基础概念](#41-回测基础概念)
  - [4.2 Qlib 回测框架](#42-qlib-回测框架)
  - [4.3 核心评估指标详解](#43-核心评估指标详解)
  - [4.4 完整回测实战](#44-完整回测实战)
  - [4.5 回测结果分析](#45-回测结果分析)

---

# Step 2: Jupyter 可视化分析

## 2.1 数据可视化基础

### 为什么要可视化？

- 📊 直观理解数据分布和趋势
- 🔍 快速发现异常值和数据问题
- 📈 展示模型预测效果
- 💡 辅助投资决策

### 常用可视化库

```python
# 基础绑定
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
```

---

## 2.2 股票行情可视化

### 2.2.1 K线图（蜡烛图）

```python
import qlib
from qlib.data import D

# 初始化 Qlib
qlib.init(provider_uri="~/.qlib/qlib_data/us_data")

# 获取苹果股票数据
df = D.features(
    instruments=["AAPL"],
    fields=["$open", "$high", "$low", "$close", "$volume"],
    start_time="2024-01-01",
    end_time="2024-06-30"
)

# 整理数据
df = df.reset_index()
df.columns = ['instrument', 'datetime', 'open', 'high', 'low', 'close', 'volume']
df = df.set_index('datetime')
```

```python
# 绘制 K 线图
import mplfinance as mpf

# 准备数据格式
ohlc_data = df[['open', 'high', 'low', 'close', 'volume']].copy()
ohlc_data.index = pd.to_datetime(ohlc_data.index)

# 绘制
mpf.plot(ohlc_data,
         type='candle',           # 蜡烛图
         volume=True,             # 显示成交量
         title='AAPL K线图',
         style='charles',         # 图表风格
         figsize=(14, 8))
```

**输出效果：**
- 🟢 绿色/空心：收盘价 > 开盘价（上涨）
- 🔴 红色/实心：收盘价 < 开盘价（下跌）
- 上下影线：最高价和最低价

### 2.2.2 移动平均线

```python
# 计算移动平均线
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA20'] = df['close'].rolling(window=20).mean()
df['MA60'] = df['close'].rolling(window=60).mean()

# 绘制
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['close'], label='收盘价', linewidth=1.5)
plt.plot(df.index, df['MA5'], label='MA5 (5日均线)', linewidth=1, alpha=0.8)
plt.plot(df.index, df['MA20'], label='MA20 (20日均线)', linewidth=1, alpha=0.8)
plt.plot(df.index, df['MA60'], label='MA60 (60日均线)', linewidth=1, alpha=0.8)

plt.title('AAPL 股价与移动平均线')
plt.xlabel('日期')
plt.ylabel('价格 ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 2.2.3 成交量分析

```python
fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

# 上图：价格
axes[0].plot(df.index, df['close'], color='blue', linewidth=1.5)
axes[0].set_ylabel('价格 ($)')
axes[0].set_title('AAPL 价格与成交量')
axes[0].grid(True, alpha=0.3)

# 下图：成交量柱状图
colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
          for i in range(len(df))]
axes[1].bar(df.index, df['volume'], color=colors, alpha=0.7)
axes[1].set_ylabel('成交量')
axes[1].set_xlabel('日期')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2.3 因子分布可视化

### 2.3.1 单因子分布

```python
from qlib.contrib.data.handler import Alpha158

# 创建因子处理器
handler = Alpha158(
    instruments="sp500",
    start_time="2023-01-01",
    end_time="2023-12-31"
)

# 获取因子数据
factor_data = handler.fetch()
print(f"因子数据形状: {factor_data.shape}")
print(f"因子列表: {list(factor_data.columns[:10])}...")  # 前10个因子
```

```python
# 选择一个因子进行可视化
factor_name = 'KMID'  # 价格动量因子

plt.figure(figsize=(14, 5))

# 子图1：直方图
plt.subplot(1, 2, 1)
plt.hist(factor_data[factor_name].dropna(), bins=50, edgecolor='black', alpha=0.7)
plt.xlabel(factor_name)
plt.ylabel('频数')
plt.title(f'{factor_name} 因子分布直方图')
plt.axvline(x=0, color='red', linestyle='--', label='零点')
plt.legend()

# 子图2：箱线图
plt.subplot(1, 2, 2)
plt.boxplot(factor_data[factor_name].dropna(), vert=True)
plt.ylabel(factor_name)
plt.title(f'{factor_name} 因子箱线图')

plt.tight_layout()
plt.show()

# 统计信息
print(f"\n{factor_name} 因子统计:")
print(factor_data[factor_name].describe())
```

### 2.3.2 因子相关性热力图

```python
# 选取部分因子计算相关性
selected_factors = ['KMID', 'KLEN', 'KMID2', 'KUP', 'KUP2',
                    'KLOW', 'KLOW2', 'KSFT', 'KSFT2', 'ROC5']

corr_matrix = factor_data[selected_factors].corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,           # 显示数值
            fmt='.2f',            # 保留2位小数
            cmap='RdBu_r',        # 颜色方案
            center=0,             # 中心点为0
            square=True,          # 方形格子
            linewidths=0.5)
plt.title('因子相关性热力图')
plt.tight_layout()
plt.show()
```

**解读：**
- 🔴 红色：正相关（接近 +1）
- 🔵 蓝色：负相关（接近 -1）
- ⚪ 白色：无相关（接近 0）

### 2.3.3 因子时序变化

```python
# 查看某只股票的因子时序变化
stock = 'AAPL'
stock_factors = factor_data.loc[stock]

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# ROC（涨跌幅）
axes[0].plot(stock_factors.index, stock_factors['ROC5'], label='ROC5')
axes[0].plot(stock_factors.index, stock_factors['ROC10'], label='ROC10')
axes[0].set_ylabel('ROC')
axes[0].set_title(f'{stock} 因子时序变化')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RSQR（波动率）
axes[1].plot(stock_factors.index, stock_factors['RSQR5'], label='RSQR5', color='orange')
axes[1].plot(stock_factors.index, stock_factors['RSQR10'], label='RSQR10', color='red')
axes[1].set_ylabel('RSQR')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 成交量因子
axes[2].plot(stock_factors.index, stock_factors['VSTD5'], label='VSTD5', color='green')
axes[2].set_ylabel('VSTD')
axes[2].set_xlabel('日期')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2.4 预测结果可视化

### 2.4.1 预测值 vs 真实值散点图

```python
# 假设我们已经有了预测结果
# pred_df 包含 'score'(预测值) 和 'label'(真实收益率) 列

# 模拟数据（实际使用时替换为真实预测结果）
np.random.seed(42)
n_samples = 1000
pred_df = pd.DataFrame({
    'score': np.random.randn(n_samples) * 0.1,
    'label': np.random.randn(n_samples) * 0.05
})
pred_df['label'] = pred_df['score'] * 0.3 + np.random.randn(n_samples) * 0.03

# 绘制散点图
plt.figure(figsize=(10, 8))
plt.scatter(pred_df['score'], pred_df['label'], alpha=0.5, s=10)
plt.xlabel('预测得分 (score)')
plt.ylabel('实际收益率 (label)')
plt.title('预测值 vs 真实值')

# 添加回归线
z = np.polyfit(pred_df['score'], pred_df['label'], 1)
p = np.poly1d(z)
x_line = np.linspace(pred_df['score'].min(), pred_df['score'].max(), 100)
plt.plot(x_line, p(x_line), "r--", linewidth=2, label=f'回归线: y={z[0]:.3f}x+{z[1]:.3f}')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 计算相关系数
corr = pred_df['score'].corr(pred_df['label'])
print(f"预测值与真实值相关系数: {corr:.4f}")
```

### 2.4.2 分组累计收益图

```python
def plot_group_returns(pred_df, n_groups=5):
    """
    按预测得分分组，绘制各组累计收益
    """
    # 按日期分组，计算每日各组收益
    pred_df = pred_df.copy()
    pred_df['group'] = pd.qcut(pred_df['score'], q=n_groups,
                                labels=[f'G{i+1}' for i in range(n_groups)],
                                duplicates='drop')

    # 计算各组平均收益
    group_returns = pred_df.groupby(['datetime', 'group'])['label'].mean().unstack()

    # 计算累计收益
    cumulative_returns = (1 + group_returns).cumprod()

    # 绘图
    plt.figure(figsize=(14, 6))

    color_map = {
        'G1': 'red',      # 预测最低组
        'G2': 'orange',
        'G3': 'gray',
        'G4': 'lightgreen',
        'G5': 'green'     # 预测最高组
    }

    for col in cumulative_returns.columns:
        color = color_map.get(col, 'blue')
        plt.plot(cumulative_returns.index, cumulative_returns[col],
                 label=col, linewidth=2, color=color)

    plt.xlabel('日期')
    plt.ylabel('累计收益')
    plt.title('按预测得分分组的累计收益')
    plt.legend(title='分组')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 打印结论
    print("\n📊 分析结论:")
    print("如果 G5 (预测最高组) 的累计收益明显高于 G1 (预测最低组),")
    print("说明模型具有良好的预测能力。")

    return cumulative_returns
```

### 2.4.3 IC 时序图

```python
def plot_ic_series(pred_df):
    """
    绘制 IC (Information Coefficient) 时序图
    IC = 预测得分与实际收益的秩相关系数
    """
    from scipy.stats import spearmanr

    # 按日期计算 IC
    ic_series = pred_df.groupby('datetime').apply(
        lambda x: spearmanr(x['score'], x['label'])[0]
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 上图：IC 时序
    axes[0].bar(ic_series.index, ic_series.values,
                color=['green' if x > 0 else 'red' for x in ic_series.values],
                alpha=0.7)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].axhline(y=ic_series.mean(), color='blue', linestyle='--',
                    label=f'均值: {ic_series.mean():.4f}')
    axes[0].set_ylabel('IC')
    axes[0].set_title('每日 IC (Information Coefficient)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 下图：累计 IC
    cumulative_ic = ic_series.cumsum()
    axes[1].plot(cumulative_ic.index, cumulative_ic.values,
                 color='blue', linewidth=2)
    axes[1].fill_between(cumulative_ic.index, 0, cumulative_ic.values, alpha=0.3)
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('累计 IC')
    axes[1].set_title('累计 IC')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 统计信息
    print(f"\nIC 统计:")
    print(f"  均值 (IC Mean): {ic_series.mean():.4f}")
    print(f"  标准差 (IC Std): {ic_series.std():.4f}")
    print(f"  IR (IC Mean / IC Std): {ic_series.mean() / ic_series.std():.4f}")
    print(f"  IC > 0 比例: {(ic_series > 0).mean():.2%}")

    return ic_series
```

---

## 2.5 交互式图表

### 使用 Plotly 创建交互式图表

```python
# 安装：pip install plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_interactive_candlestick(df):
    """
    创建交互式 K 线图
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_heights=[0.7, 0.3])

    # K 线图
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线'
    ), row=1, col=1)

    # 成交量
    colors = ['green' if row['close'] >= row['open'] else 'red'
              for _, row in df.iterrows()]
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        marker_color=colors,
        name='成交量'
    ), row=2, col=1)

    # 布局设置
    fig.update_layout(
        title='交互式 K 线图（可缩放、拖动）',
        yaxis_title='价格',
        yaxis2_title='成交量',
        xaxis_rangeslider_visible=False,
        height=600
    )

    fig.show()

# 使用示例
# plot_interactive_candlestick(df)
```

---

# Step 3: 理解 Alpha158 因子

## 3.1 什么是因子

### 因子的定义

**因子（Factor）** 是用于预测股票未来收益的特征变量。

```
因子 = 从历史数据中提取的、可能与未来收益相关的数值特征
```

### 因子投资的核心思想

```
高质量因子 → 预测股票收益 → 构建投资组合 → 获取超额收益
```

### 常见因子类型

| 类型 | 描述 | 示例 |
|------|------|------|
| **价值因子** | 衡量股票便宜程度 | 市盈率、市净率 |
| **动量因子** | 过去涨跌趋势的延续 | 过去N日涨幅 |
| **波动率因子** | 价格波动程度 | 收益率标准差 |
| **流动性因子** | 交易活跃程度 | 换手率、成交量 |
| **技术因子** | 技术分析指标 | MACD、RSI、布林带 |

---

## 3.2 Alpha158 因子库概览

### 什么是 Alpha158？

Alpha158 是 Qlib 内置的因子库，包含 **158 个** 经过验证的技术因子。

```python
from qlib.contrib.data.handler import Alpha158

# 查看 Alpha158 的因子配置
import inspect
print(inspect.getsourcefile(Alpha158))
```

### 因子数量统计

| 大类 | 数量 | 说明 |
|------|------|------|
| KBAR 类 | 18 | K线形态特征 |
| PRICE 类 | 15 | 价格变化特征 |
| VOLUME 类 | 15 | 成交量特征 |
| STD 类 | 6 | 波动率特征 |
| BETA 类 | 6 | 市场相关性 |
| RSQR 类 | 6 | 拟合度特征 |
| RESI 类 | 6 | 残差特征 |
| MAX/MIN 类 | 12 | 极值特征 |
| QTLU/QTLD 类 | 12 | 分位数特征 |
| RANK 类 | 6 | 排名特征 |
| RSV 类 | 6 | 相对强弱 |
| CORR 类 | 6 | 相关性特征 |
| CORD 类 | 6 | 协方差特征 |
| CNTP/CNTN/CNTD 类 | 18 | 计数特征 |
| ROC 类 | 6 | 变化率 |
| WVMA 类 | 6 | 加权波动 |
| VMA 类 | 6 | 成交量移动平均 |
| **总计** | **158** | - |

---

## 3.3 六大类因子详解

### 3.3.1 KBAR 类因子（K线形态）

K线因子从单根K线中提取信息：

```python
# KBAR 因子公式
KMID = (close - open) / open                    # 中间位置（涨跌幅）
KLEN = (high - low) / open                      # K线长度（振幅）
KMID2 = (close - open) / (high - low + 1e-12)   # 相对位置
KUP = (high - max(open, close)) / open          # 上影线
KUP2 = (high - max(open, close)) / (high - low + 1e-12)
KLOW = (min(open, close) - low) / open          # 下影线
KLOW2 = (min(open, close) - low) / (high - low + 1e-12)
KSFT = (2*close - high - low) / open            # 收盘位置偏移
KSFT2 = (2*close - high - low) / (high - low + 1e-12)
```

**图解：**
```
    ┌── high (最高价)
    │   ← KUP (上影线)
    ├── max(open, close)
    │
    │   ← KMID (实体)
    │
    ├── min(open, close)
    │   ← KLOW (下影线)
    └── low (最低价)

    ←─────── KLEN ────────→
```

**使用场景：**
- `KMID > 0`：收阳线（上涨）
- `KUP 很大`：上方压力大，可能反转
- `KLOW 很大`：下方支撑强

### 3.3.2 PRICE 类因子（价格变化）

```python
# 价格因子 - 反映不同时间窗口的价格变化
OPEN0 = open / close          # 开盘相对位置
HIGH0 = high / close          # 最高价相对位置
LOW0 = low / close            # 最低价相对位置
VWAP0 = vwap / close          # 成交均价相对位置 (如果有 VWAP)

# 滞后价格因子
CLOSE1 = Ref(close, 1) / close    # 昨收 / 今收
CLOSE2 = Ref(close, 2) / close    # 前天收 / 今收
...
```

**解读：**
- `CLOSE1 > 1`：今日下跌
- `CLOSE1 < 1`：今日上涨

### 3.3.3 VOLUME 类因子（成交量）

```python
# 成交量因子
VOLUME1 = Ref(volume, 1) / (volume + 1e-12)  # 昨日成交量 / 今日成交量
VOLUME5 = Mean(volume, 5) / (volume + 1e-12)  # 5日均量 / 今日成交量

# 成交量变化率
VMA5 = Mean(volume, 5)
VMA10 = Mean(volume, 10)
VSTD5 = Std(volume, 5)   # 成交量波动
VSTD10 = Std(volume, 10)
```

**解读：**
- `VOLUME5 > 1`：今日缩量
- `VOLUME5 < 1`：今日放量
- `VSTD` 大：成交量波动大

### 3.3.4 ROC 类因子（变化率）

```python
# ROC = Rate of Change
ROC5 = Ref(close, 5) / close - 1     # 5日涨跌幅
ROC10 = Ref(close, 10) / close - 1   # 10日涨跌幅
ROC20 = Ref(close, 20) / close - 1   # 20日涨跌幅
ROC60 = Ref(close, 60) / close - 1   # 60日涨跌幅 (季度)
```

**使用场景：**
- **动量策略**：买入 ROC > 0 的股票（趋势跟随）
- **反转策略**：买入 ROC < 0 的股票（均值回归）

### 3.3.5 STD/RSQR 类因子（波动率）

```python
# 收益率波动
STD5 = Std(close/Ref(close,1)-1, 5)    # 5日收益率标准差
STD10 = Std(close/Ref(close,1)-1, 10)
STD20 = Std(close/Ref(close,1)-1, 20)
STD60 = Std(close/Ref(close,1)-1, 60)

# 拟合残差（与市场的偏离度）
RSQR5 = 对过去5日收益率做线性回归的 R²
RESI5 = 回归残差的均值
```

**解读：**
- `STD` 高：高波动，高风险
- `RSQR` 高：与市场相关性高
- `RESI` 正：跑赢市场

### 3.3.6 CORR/CORD 类因子（相关性）

```python
# 价量相关性
CORR5 = Corr(close, volume, 5)    # 5日价量相关系数
CORR10 = Corr(close, volume, 10)
CORR20 = Corr(close, volume, 20)

# 价量协方差
CORD5 = Cov(close, volume, 5) / (Std(close,5) * Std(volume,5))
```

**解读：**
- `CORR > 0`：量价齐升或齐跌
- `CORR < 0`：量价背离

---

## 3.4 因子计算实战

### 3.4.1 使用 Alpha158 获取因子

```python
import qlib
from qlib.data import D
from qlib.contrib.data.handler import Alpha158

# 初始化
qlib.init(provider_uri="~/.qlib/qlib_data/us_data")

# 创建 Alpha158 因子处理器
handler = Alpha158(
    instruments="sp500",          # 股票池
    start_time="2023-01-01",
    end_time="2023-12-31",
    fit_start_time="2022-01-01",  # 拟合开始时间
    fit_end_time="2022-12-31"     # 拟合结束时间
)

# 获取因子数据
factor_df = handler.fetch()

print(f"数据形状: {factor_df.shape}")
print(f"股票数量: {factor_df.index.get_level_values(0).nunique()}")
print(f"日期范围: {factor_df.index.get_level_values(1).min()} ~ {factor_df.index.get_level_values(1).max()}")
print(f"\n因子列表 (前20个):")
for i, col in enumerate(factor_df.columns[:20]):
    print(f"  {i+1}. {col}")
```

### 3.4.2 因子有效性检验

```python
from scipy.stats import spearmanr

def check_factor_effectiveness(factor_df, factor_name, label_col='label'):
    """
    检验单个因子的有效性
    """
    # 准备数据
    data = factor_df[[factor_name, label_col]].dropna()

    # 1. 计算 IC (Spearman 相关系数)
    ic_values = data.groupby(level=1).apply(
        lambda x: spearmanr(x[factor_name], x[label_col])[0]
    )

    ic_mean = ic_values.mean()
    ic_std = ic_values.std()
    ir = ic_mean / ic_std if ic_std > 0 else 0

    # 2. 分组回测
    data['group'] = data.groupby(level=1)[factor_name].transform(
        lambda x: pd.qcut(x, q=5, labels=['G1','G2','G3','G4','G5'], duplicates='drop')
    )
    group_returns = data.groupby('group')[label_col].mean()

    # 3. 输出结果
    print(f"\n{'='*50}")
    print(f"因子有效性检验: {factor_name}")
    print(f"{'='*50}")
    print(f"IC 均值: {ic_mean:.4f}")
    print(f"IC 标准差: {ic_std:.4f}")
    print(f"IR (IC_mean/IC_std): {ir:.4f}")
    print(f"IC > 0 比例: {(ic_values > 0).mean():.2%}")
    print(f"\n分组平均收益:")
    print(group_returns)
    print(f"\nG5 - G1 (多空收益): {group_returns['G5'] - group_returns['G1']:.4f}")

    # 评价
    if abs(ic_mean) > 0.03 and abs(ir) > 0.3:
        print("\n✅ 该因子具有较好的预测能力")
    elif abs(ic_mean) > 0.02:
        print("\n⚠️ 该因子具有一定的预测能力")
    else:
        print("\n❌ 该因子预测能力较弱")

    return ic_values, group_returns

# 使用示例
# ic_values, group_returns = check_factor_effectiveness(factor_df, 'ROC5')
```

### 3.4.3 多因子相关性分析

```python
def analyze_factor_correlation(factor_df, factor_list=None):
    """
    分析因子之间的相关性，找出冗余因子
    """
    if factor_list is None:
        factor_list = ['KMID', 'ROC5', 'ROC10', 'ROC20', 'STD5', 'STD10',
                       'CORR5', 'VOLUME5', 'VSUMP5']

    # 计算相关性矩阵
    corr_matrix = factor_df[factor_list].corr()

    # 可视化
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 上三角遮罩
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, square=True)
    plt.title('因子相关性矩阵')
    plt.tight_layout()
    plt.show()

    # 找出高相关因子对
    high_corr_pairs = []
    for i in range(len(factor_list)):
        for j in range(i+1, len(factor_list)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append((factor_list[i], factor_list[j], corr))

    if high_corr_pairs:
        print("\n⚠️ 高相关因子对 (|corr| > 0.7):")
        for f1, f2, corr in high_corr_pairs:
            print(f"  {f1} <-> {f2}: {corr:.3f}")
        print("\n建议：高相关因子可能存在信息冗余，可考虑只保留其中一个")
    else:
        print("\n✅ 没有发现高相关因子对")

    return corr_matrix
```

---

## 3.5 自定义因子

### 3.5.1 使用 Qlib 表达式定义因子

```python
from qlib.data import D

# 定义自定义因子
custom_factors = [
    # 自定义动量因子：20日涨幅
    ("MOM20", "($close - Ref($close, 20)) / Ref($close, 20)"),

    # 自定义波动率因子：20日振幅均值
    ("SWING20", "Mean(($high - $low) / $close, 20)"),

    # 自定义量价因子：5日量价相关
    ("VP_CORR5", "Corr($close, $volume, 5)"),

    # 自定义趋势因子：价格在布林带中的位置
    ("BOLL_POS", "($close - Mean($close, 20)) / (Std($close, 20) + 1e-12)"),

    # 自定义换手率因子 (如果有流通股数据)
    # ("TURN5", "Mean($volume / $float_share, 5)"),
]

# 获取自定义因子数据
field_names = [f[0] for f in custom_factors]
field_exprs = [f[1] for f in custom_factors]

custom_df = D.features(
    instruments=["AAPL", "MSFT", "GOOGL"],
    fields=field_exprs,
    start_time="2023-01-01",
    end_time="2023-12-31"
)
custom_df.columns = field_names

print("自定义因子数据:")
print(custom_df.head(10))
```

### 3.5.2 常用因子表达式语法

```python
"""
Qlib 因子表达式语法速查表
"""

# 基础运算
"$close + $open"           # 加法
"$close - $open"           # 减法
"$close * $volume"         # 乘法
"$close / $open"           # 除法
"$close ** 2"              # 幂运算

# 引用函数
"Ref($close, 1)"           # 前1日收盘价
"Ref($close, 5)"           # 前5日收盘价
"Ref($close, -1)"          # 后1日收盘价（仅用于标签）

# 统计函数
"Mean($close, 5)"          # 5日均值
"Sum($volume, 5)"          # 5日成交量之和
"Std($close, 10)"          # 10日标准差
"Var($close, 10)"          # 10日方差
"Max($high, 20)"           # 20日最高价
"Min($low, 20)"            # 20日最低价
"Median($close, 10)"       # 10日中位数
"Prod($close/$ref($close,1), 5)"  # 5日累计收益率

# 排名函数
"Rank($close)"             # 截面排名 (0~1)

# 相关性函数
"Corr($close, $volume, 5)" # 5日价量相关系数
"Cov($close, $volume, 5)"  # 5日价量协方差

# 条件函数
"If($close > $open, 1, 0)" # 条件判断
"If($close > Ref($close, 1), $volume, 0)"  # 上涨时的成交量

# 符号函数
"Abs($close - $open)"      # 绝对值
"Sign($close - $open)"     # 符号 (-1, 0, 1)
"Log($volume)"             # 对数

# 组合示例：RSI 相对强弱指标
"""
up = If($close > Ref($close, 1), $close - Ref($close, 1), 0)
down = If($close < Ref($close, 1), Ref($close, 1) - $close, 0)
RSI = 100 * Mean(up, 14) / (Mean(up, 14) + Mean(down, 14) + 1e-12)
"""
```

### 3.5.3 创建自定义因子处理器

```python
from qlib.contrib.data.handler import DataHandlerLP

class MyAlphaHandler(DataHandlerLP):
    """
    自定义因子处理器
    """

    def __init__(self, instruments, start_time, end_time, **kwargs):
        # 定义特征
        self.feature_config = [
            # (因子名, 表达式)
            ("KMID", "($close-$open)/$open"),
            ("MOM5", "($close-Ref($close,5))/Ref($close,5)"),
            ("MOM20", "($close-Ref($close,20))/Ref($close,20)"),
            ("VOL5", "Std($close/Ref($close,1)-1, 5)"),
            ("VOL20", "Std($close/Ref($close,1)-1, 20)"),
            ("VWAP_RATIO", "Sum($close*$volume,5)/Sum($volume,5)/$close"),
            ("RSI", "Mean(If($close>Ref($close,1),$close-Ref($close,1),0),14)/"
                    "(Mean(If($close>Ref($close,1),$close-Ref($close,1),0),14)+"
                    "Mean(If($close<Ref($close,1),Ref($close,1)-$close,0),14)+1e-12)"),
        ]

        # 定义标签
        self.label_config = [
            ("label", "Ref($close,-2)/Ref($close,-1)-1"),  # 未来收益率
        ]

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.feature_config,
                    "label": self.label_config,
                },
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            **kwargs
        )

# 使用示例
# handler = MyAlphaHandler(
#     instruments="sp500",
#     start_time="2023-01-01",
#     end_time="2023-12-31"
# )
# data = handler.fetch()
```

---

# Step 4: 回测与评估指标

## 4.1 回测基础概念

### 什么是回测？

**回测（Backtesting）** 是使用历史数据模拟交易策略，评估其过去表现的过程。

```
历史数据 + 交易策略 → 模拟交易 → 评估指标 → 策略优化
```

### 回测的重要性

| 目的 | 说明 |
|------|------|
| **验证策略** | 检验策略在历史数据上是否有效 |
| **风险评估** | 了解最大回撤、波动率等风险指标 |
| **参数优化** | 找到最优策略参数 |
| **避免过拟合** | 通过样本外测试验证泛化能力 |

### 回测的陷阱

```
⚠️ 常见回测陷阱：

1. 未来信息泄露（Look-Ahead Bias）
   - 错误：使用未来数据计算因子
   - 正确：只使用当前及历史数据

2. 幸存者偏差（Survivorship Bias）
   - 错误：只回测现存股票
   - 正确：包含已退市股票

3. 过度拟合（Overfitting）
   - 错误：在同一数据上反复优化
   - 正确：使用独立的测试集

4. 忽略交易成本
   - 错误：假设零成本交易
   - 正确：计入手续费、滑点、冲击成本
```

---

## 4.2 Qlib 回测框架

### 4.2.1 回测流程

```
┌─────────────────────────────────────────────────────────────┐
│                      Qlib 回测流程                          │
├─────────────────────────────────────────────────────────────┤
│  1. 数据准备                                                │
│     └── DataHandler (Alpha158) → 因子数据                   │
│                                                             │
│  2. 模型预测                                                │
│     └── Model (LGBModel) → 预测得分                         │
│                                                             │
│  3. 信号生成                                                │
│     └── Strategy (TopkDropoutStrategy) → 买卖信号           │
│                                                             │
│  4. 订单执行                                                │
│     └── Executor → 模拟成交                                 │
│                                                             │
│  5. 绩效评估                                                │
│     └── Backtest → 收益、风险指标                           │
└─────────────────────────────────────────────────────────────┘
```

### 4.2.2 关键组件

```python
# Qlib 回测核心组件
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.evaluate import backtest_daily
from qlib.contrib.evaluate import risk_analysis

# 1. 策略 (Strategy)
strategy_config = {
    "class": "TopkDropoutStrategy",
    "kwargs": {
        "model": model,           # 预测模型
        "dataset": dataset,       # 数据集
        "topk": 50,               # 持仓股票数量
        "n_drop": 5,              # 每次调仓卖出数量
    },
}

# 2. 执行器 (Executor)
executor_config = {
    "class": "SimulatorExecutor",
    "kwargs": {
        "time_per_step": "day",   # 每日调仓
        "generate_portfolio_metrics": True,
    },
}

# 3. 回测配置
backtest_config = {
    "start_time": "2023-01-01",
    "end_time": "2023-12-31",
    "account": 1000000,          # 初始资金 100万
    "benchmark": "SH000300",     # 基准指数
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": 0.095,  # 涨跌停限制
        "deal_price": "close",     # 成交价格
        "open_cost": 0.0005,       # 买入手续费 0.05%
        "close_cost": 0.0015,      # 卖出手续费 0.15%
        "min_cost": 5,             # 最低手续费
    },
}
```

---

## 4.3 核心评估指标详解

### 4.3.1 收益类指标

#### 1. 年化收益率 (Annualized Return)

```python
def annualized_return(returns, periods_per_year=252):
    """
    年化收益率

    公式: (1 + 总收益率)^(252/交易天数) - 1

    Args:
        returns: 每日收益率序列
        periods_per_year: 每年交易日数 (默认252)

    解读:
        > 15%: 优秀
        10-15%: 良好
        5-10%: 一般
        < 5%: 较差
    """
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    ann_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    return ann_return

# 示例
daily_returns = pd.Series([0.01, -0.005, 0.02, 0.003, -0.01])  # 5天收益率
ann_ret = annualized_return(daily_returns)
print(f"年化收益率: {ann_ret:.2%}")
```

#### 2. 累计收益率 (Cumulative Return)

```python
def cumulative_return(returns):
    """
    累计收益率

    公式: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    """
    return (1 + returns).prod() - 1

# 可视化累计收益
cumulative = (1 + returns).cumprod()
plt.plot(cumulative)
plt.title('累计收益曲线')
```

#### 3. 超额收益 (Excess Return)

```python
def excess_return(strategy_returns, benchmark_returns):
    """
    超额收益 = 策略收益 - 基准收益

    衡量策略相对于基准的表现
    """
    return strategy_returns - benchmark_returns
```

### 4.3.2 风险类指标

#### 1. 最大回撤 (Maximum Drawdown)

```python
def max_drawdown(returns):
    """
    最大回撤：从历史最高点下跌的最大幅度

    公式: max((peak - trough) / peak)

    解读:
        < 10%: 低风险
        10-20%: 中等风险
        20-30%: 较高风险
        > 30%: 高风险
    """
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    max_dd = drawdown.min()
    return abs(max_dd)

# 示例
returns = pd.Series([0.1, 0.05, -0.15, -0.1, 0.2, 0.1])
mdd = max_drawdown(returns)
print(f"最大回撤: {mdd:.2%}")
```

**可视化最大回撤：**

```python
def plot_drawdown(returns):
    """绘制回撤图"""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 上图：累计收益
    axes[0].plot(cumulative.index, cumulative.values, label='累计收益')
    axes[0].plot(peak.index, peak.values, '--', label='历史最高')
    axes[0].fill_between(cumulative.index, cumulative.values, peak.values,
                         alpha=0.3, color='red')
    axes[0].set_ylabel('累计收益')
    axes[0].legend()

    # 下图：回撤
    axes[1].fill_between(drawdown.index, 0, drawdown.values,
                         color='red', alpha=0.5)
    axes[1].set_ylabel('回撤')
    axes[1].set_xlabel('日期')

    plt.tight_layout()
    plt.show()
```

#### 2. 波动率 (Volatility)

```python
def annualized_volatility(returns, periods_per_year=252):
    """
    年化波动率

    公式: std(daily_returns) * sqrt(252)

    解读:
        < 15%: 低波动
        15-25%: 中等波动
        > 25%: 高波动
    """
    return returns.std() * np.sqrt(periods_per_year)
```

#### 3. 下行波动率 (Downside Volatility)

```python
def downside_volatility(returns, threshold=0, periods_per_year=252):
    """
    下行波动率：只计算负收益的波动

    更关注下跌风险
    """
    downside_returns = returns[returns < threshold]
    return downside_returns.std() * np.sqrt(periods_per_year)
```

### 4.3.3 风险调整收益指标

#### 1. 夏普比率 (Sharpe Ratio) ⭐最重要

```python
def sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    夏普比率 = (年化收益 - 无风险利率) / 年化波动率

    衡量每承担1单位风险获得的超额收益

    解读:
        > 2.0: 优秀
        1.0-2.0: 良好
        0.5-1.0: 一般
        < 0.5: 较差
        < 0: 亏损
    """
    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)

    sharpe = (ann_return - risk_free_rate) / ann_vol
    return sharpe

# 示例
sharpe = sharpe_ratio(daily_returns)
print(f"夏普比率: {sharpe:.2f}")
```

#### 2. 卡尔玛比率 (Calmar Ratio)

```python
def calmar_ratio(returns, periods_per_year=252):
    """
    卡尔玛比率 = 年化收益 / 最大回撤

    衡量收益与最大损失的关系

    解读:
        > 3.0: 优秀
        1.0-3.0: 良好
        < 1.0: 需要改进
    """
    ann_return = annualized_return(returns, periods_per_year)
    mdd = max_drawdown(returns)

    return ann_return / mdd if mdd > 0 else np.inf
```

#### 3. 索提诺比率 (Sortino Ratio)

```python
def sortino_ratio(returns, risk_free_rate=0.02, periods_per_year=252):
    """
    索提诺比率 = (年化收益 - 无风险利率) / 下行波动率

    类似夏普比率，但只关注下行风险
    比夏普比率更合理，因为投资者主要担心亏损
    """
    ann_return = annualized_return(returns, periods_per_year)
    down_vol = downside_volatility(returns, 0, periods_per_year)

    return (ann_return - risk_free_rate) / down_vol if down_vol > 0 else np.inf
```

### 4.3.4 预测能力指标

#### 1. IC (Information Coefficient)

```python
def information_coefficient(pred_score, actual_return):
    """
    IC = 预测得分与实际收益的秩相关系数 (Spearman)

    衡量预测的排序能力

    解读:
        > 0.05: 优秀
        0.03-0.05: 良好
        0.01-0.03: 一般
        < 0.01: 较差
    """
    from scipy.stats import spearmanr
    ic, p_value = spearmanr(pred_score, actual_return)
    return ic, p_value
```

#### 2. IR (Information Ratio)

```python
def information_ratio(ic_series):
    """
    IR = IC均值 / IC标准差

    衡量IC的稳定性

    解读:
        > 0.5: 优秀
        0.3-0.5: 良好
        < 0.3: 一般
    """
    return ic_series.mean() / ic_series.std()
```

#### 3. 胜率 (Win Rate)

```python
def win_rate(returns):
    """
    胜率 = 盈利天数 / 总交易天数

    解读:
        > 55%: 良好
        50-55%: 一般
        < 50%: 需要改进
    """
    wins = (returns > 0).sum()
    total = len(returns)
    return wins / total
```

#### 4. 盈亏比 (Profit/Loss Ratio)

```python
def profit_loss_ratio(returns):
    """
    盈亏比 = 平均盈利 / 平均亏损

    解读:
        > 2.0: 优秀
        1.5-2.0: 良好
        1.0-1.5: 一般
        < 1.0: 较差
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1

    return avg_win / avg_loss
```

### 4.3.5 指标总结表

```
┌─────────────────────────────────────────────────────────────────────┐
│                       回测评估指标速查表                             │
├─────────────────┬───────────────────┬───────────────────────────────┤
│ 指标            │ 公式               │ 优秀标准                      │
├─────────────────┼───────────────────┼───────────────────────────────┤
│ 年化收益率      │ (1+总收益)^(252/n)-1│ > 15%                        │
│ 最大回撤        │ max((峰值-谷值)/峰值)│ < 20%                        │
│ 夏普比率 ⭐     │ (收益-无风险)/波动率 │ > 1.5                        │
│ 卡尔玛比率      │ 年化收益/最大回撤   │ > 2.0                         │
│ 索提诺比率      │ 收益/下行波动率     │ > 2.0                         │
│ IC             │ Spearman相关系数    │ > 0.03                        │
│ IR             │ IC均值/IC标准差     │ > 0.5                         │
│ 胜率            │ 盈利天数/总天数     │ > 55%                         │
│ 盈亏比          │ 平均盈利/平均亏损   │ > 1.5                         │
└─────────────────┴───────────────────┴───────────────────────────────┘
```

---

## 4.4 完整回测实战

### 4.4.1 完整回测代码

```python
import qlib
from qlib.constant import REG_CN, REG_US
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy

# ==================== 1. 初始化 Qlib ====================
qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region=REG_US)

# ==================== 2. 数据处理器配置 ====================
data_handler_config = {
    "class": "Alpha158",
    "module_path": "qlib.contrib.data.handler",
    "kwargs": {
        "instruments": "sp500",
        "start_time": "2020-01-01",
        "end_time": "2023-12-31",
        "fit_start_time": "2020-01-01",
        "fit_end_time": "2022-12-31",
        "infer_processors": [
            {"class": "RobustZScoreNorm", "kwargs": {"clip_outlier": True}},
            {"class": "Fillna", "kwargs": {"fill_value": 0}},
        ],
        "learn_processors": [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm"},
        ],
    },
}

# ==================== 3. 数据集配置 ====================
dataset_config = {
    "class": "DatasetH",
    "module_path": "qlib.data.dataset",
    "kwargs": {
        "handler": data_handler_config,
        "segments": {
            "train": ("2020-01-01", "2021-12-31"),
            "valid": ("2022-01-01", "2022-06-30"),
            "test": ("2022-07-01", "2023-12-31"),
        },
    },
}

# ==================== 4. 模型配置 ====================
model_config = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
        "loss": "mse",
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "lambda_l1": 200,
        "lambda_l2": 200,
        "max_depth": 8,
        "num_leaves": 64,
        "num_boost_round": 500,
        "early_stopping_rounds": 50,
    },
}

# ==================== 5. 训练模型 ====================
dataset = init_instance_by_config(dataset_config)
model = init_instance_by_config(model_config)

# 训练
model.fit(dataset)

# 预测
pred = model.predict(dataset)
print(f"预测结果形状: {pred.shape}")

# ==================== 6. 策略配置 ====================
strategy_config = {
    "class": "TopkDropoutStrategy",
    "module_path": "qlib.contrib.strategy",
    "kwargs": {
        "signal": pred,
        "topk": 30,              # 持有前30只股票
        "n_drop": 5,             # 每次调仓卖出5只
        "only_tradable": True,   # 只交易可交易股票
    },
}

# ==================== 7. 回测配置 ====================
backtest_config = {
    "start_time": "2022-07-01",
    "end_time": "2023-12-31",
    "account": 1000000,          # 100万初始资金
    "benchmark": "^GSPC",        # S&P 500 指数
    "exchange_kwargs": {
        "freq": "day",
        "limit_threshold": None,  # 美股无涨跌停
        "deal_price": "close",
        "open_cost": 0.0005,      # 买入费率 0.05%
        "close_cost": 0.0015,     # 卖出费率 0.15%
        "min_cost": 5,            # 最低手续费 $5
    },
}

# ==================== 8. 执行回测 ====================
strategy = init_instance_by_config(strategy_config)

portfolio_metric, indicator_dict = backtest_daily(
    start_time=backtest_config["start_time"],
    end_time=backtest_config["end_time"],
    strategy=strategy,
    account=backtest_config["account"],
    benchmark=backtest_config["benchmark"],
    exchange_kwargs=backtest_config["exchange_kwargs"],
)

# ==================== 9. 分析结果 ====================
analysis_result = risk_analysis(portfolio_metric["return"])
print("\n" + "="*60)
print("回测结果分析")
print("="*60)
print(analysis_result)
```

### 4.4.2 输出结果解读

```python
# 典型输出示例
"""
回测结果分析
============================================================
                  risk
mean              0.000821    # 日均收益率 0.0821%
std               0.012345    # 日波动率 1.23%
annualized_return 0.206789    # 年化收益率 20.68%
information_ratio 1.234567    # 信息比率
max_drawdown     -0.156789    # 最大回撤 -15.68%
sharpe_ratio      1.567890    # 夏普比率 1.57
calmar_ratio      1.318765    # 卡尔玛比率
sortino_ratio     2.123456    # 索提诺比率
"""
```

---

## 4.5 回测结果分析

### 4.5.1 回测报告生成

```python
def generate_backtest_report(portfolio_metric, benchmark_returns=None, save_path=None):
    """
    生成完整的回测报告
    """
    returns = portfolio_metric["return"]

    # 计算各项指标
    report = {
        "收益指标": {
            "累计收益率": f"{cumulative_return(returns):.2%}",
            "年化收益率": f"{annualized_return(returns):.2%}",
            "日均收益率": f"{returns.mean():.4%}",
        },
        "风险指标": {
            "最大回撤": f"{max_drawdown(returns):.2%}",
            "年化波动率": f"{annualized_volatility(returns):.2%}",
            "下行波动率": f"{downside_volatility(returns):.2%}",
        },
        "风险调整收益": {
            "夏普比率": f"{sharpe_ratio(returns):.2f}",
            "卡尔玛比率": f"{calmar_ratio(returns):.2f}",
            "索提诺比率": f"{sortino_ratio(returns):.2f}",
        },
        "交易统计": {
            "胜率": f"{win_rate(returns):.2%}",
            "盈亏比": f"{profit_loss_ratio(returns):.2f}",
            "最大连续盈利天数": f"{max_consecutive_wins(returns)} 天",
            "最大连续亏损天数": f"{max_consecutive_losses(returns)} 天",
        },
    }

    # 打印报告
    print("\n" + "="*60)
    print("📊 回测绩效报告")
    print("="*60)

    for category, metrics in report.items():
        print(f"\n【{category}】")
        for name, value in metrics.items():
            print(f"  {name}: {value}")

    # 与基准比较
    if benchmark_returns is not None:
        excess = cumulative_return(returns) - cumulative_return(benchmark_returns)
        print(f"\n【相对基准】")
        print(f"  超额收益: {excess:.2%}")
        print(f"  基准收益: {cumulative_return(benchmark_returns):.2%}")

    return report


def max_consecutive_wins(returns):
    """计算最大连续盈利天数"""
    wins = (returns > 0).astype(int)
    groups = (wins != wins.shift()).cumsum()
    return wins.groupby(groups).sum().max()


def max_consecutive_losses(returns):
    """计算最大连续亏损天数"""
    losses = (returns < 0).astype(int)
    groups = (losses != losses.shift()).cumsum()
    return losses.groupby(groups).sum().max()
```

### 4.5.2 可视化回测结果

```python
def plot_backtest_result(portfolio_metric, benchmark_returns=None):
    """
    可视化回测结果
    """
    returns = portfolio_metric["return"]
    cumulative = (1 + returns).cumprod()

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 1. 累计收益曲线
    ax1 = axes[0, 0]
    ax1.plot(cumulative.index, cumulative.values, label='策略', linewidth=2)
    if benchmark_returns is not None:
        bench_cumulative = (1 + benchmark_returns).cumprod()
        ax1.plot(bench_cumulative.index, bench_cumulative.values,
                 label='基准', linewidth=2, linestyle='--')
    ax1.set_title('累计收益曲线')
    ax1.set_ylabel('累计收益')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 回撤曲线
    ax2 = axes[0, 1]
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    ax2.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.5)
    ax2.set_title('回撤曲线')
    ax2.set_ylabel('回撤')
    ax2.grid(True, alpha=0.3)

    # 3. 日收益分布
    ax3 = axes[1, 0]
    ax3.hist(returns, bins=50, edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--')
    ax3.axvline(x=returns.mean(), color='blue', linestyle='--', label=f'均值: {returns.mean():.4f}')
    ax3.set_title('日收益分布')
    ax3.set_xlabel('日收益率')
    ax3.set_ylabel('频数')
    ax3.legend()

    # 4. 月度收益热力图
    ax4 = axes[1, 1]
    monthly_returns = returns.resample('M').apply(lambda x: (1+x).prod() - 1)
    monthly_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values
    })
    monthly_pivot = monthly_df.pivot(index='year', columns='month', values='return')
    sns.heatmap(monthly_pivot, annot=True, fmt='.1%', cmap='RdYlGn',
                center=0, ax=ax4, cbar_kws={'label': '收益率'})
    ax4.set_title('月度收益热力图')
    ax4.set_xlabel('月份')
    ax4.set_ylabel('年份')

    # 5. 滚动夏普比率
    ax5 = axes[2, 0]
    rolling_sharpe = returns.rolling(60).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    ax5.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5)
    ax5.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe=1')
    ax5.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax5.set_title('60日滚动夏普比率')
    ax5.set_ylabel('夏普比率')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 超额收益曲线
    ax6 = axes[2, 1]
    if benchmark_returns is not None:
        excess_returns = returns - benchmark_returns
        excess_cumulative = (1 + excess_returns).cumprod() - 1
        ax6.fill_between(excess_cumulative.index, 0, excess_cumulative.values,
                         where=(excess_cumulative.values >= 0), color='green', alpha=0.5)
        ax6.fill_between(excess_cumulative.index, 0, excess_cumulative.values,
                         where=(excess_cumulative.values < 0), color='red', alpha=0.5)
        ax6.set_title('累计超额收益')
        ax6.set_ylabel('超额收益')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, '无基准数据', ha='center', va='center', transform=ax6.transAxes)

    plt.tight_layout()
    plt.show()
```

### 4.5.3 策略对比分析

```python
def compare_strategies(strategy_results, names=None):
    """
    对比多个策略的表现

    Args:
        strategy_results: list of portfolio_metrics
        names: list of strategy names
    """
    if names is None:
        names = [f'策略{i+1}' for i in range(len(strategy_results))]

    # 计算各策略指标
    comparison = []
    for returns, name in zip(strategy_results, names):
        metrics = {
            '策略': name,
            '年化收益': annualized_return(returns),
            '最大回撤': max_drawdown(returns),
            '夏普比率': sharpe_ratio(returns),
            '卡尔玛比率': calmar_ratio(returns),
            '胜率': win_rate(returns),
        }
        comparison.append(metrics)

    df = pd.DataFrame(comparison)

    # 格式化显示
    print("\n策略对比:")
    print("="*80)

    formatted = df.copy()
    formatted['年化收益'] = formatted['年化收益'].apply(lambda x: f'{x:.2%}')
    formatted['最大回撤'] = formatted['最大回撤'].apply(lambda x: f'{x:.2%}')
    formatted['夏普比率'] = formatted['夏普比率'].apply(lambda x: f'{x:.2f}')
    formatted['卡尔玛比率'] = formatted['卡尔玛比率'].apply(lambda x: f'{x:.2f}')
    formatted['胜率'] = formatted['胜率'].apply(lambda x: f'{x:.2%}')

    print(formatted.to_string(index=False))

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 累计收益对比
    ax1 = axes[0]
    for returns, name in zip(strategy_results, names):
        cumulative = (1 + returns).cumprod()
        ax1.plot(cumulative.index, cumulative.values, label=name, linewidth=2)
    ax1.set_title('累计收益对比')
    ax1.set_ylabel('累计收益')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 指标雷达图
    ax2 = axes[1]
    categories = ['年化收益', '夏普比率', '卡尔玛比率', '胜率']
    # ... (雷达图实现)

    plt.tight_layout()
    plt.show()

    return df
```

---

## 附录：常用代码片段

### A1. 快速回测模板

```python
# 最简回测模板
import qlib
from qlib.workflow import R

qlib.init(provider_uri="~/.qlib/qlib_data/us_data")

# 使用工作流自动执行
with R.start(experiment_name="quick_backtest"):
    R.record(**{
        "model": {"class": "LGBModel"},
        "dataset": {"class": "Alpha158"},
        "strategy": {"class": "TopkDropoutStrategy", "topk": 30},
        "backtest": {"start_time": "2023-01-01", "end_time": "2023-12-31"},
    })
```

### A2. 指标计算工具类

```python
class PerformanceMetrics:
    """绩效指标计算工具类"""

    def __init__(self, returns, benchmark=None, risk_free_rate=0.02):
        self.returns = returns
        self.benchmark = benchmark
        self.rf = risk_free_rate

    @property
    def annual_return(self):
        return annualized_return(self.returns)

    @property
    def max_dd(self):
        return max_drawdown(self.returns)

    @property
    def sharpe(self):
        return sharpe_ratio(self.returns, self.rf)

    def summary(self):
        return {
            "年化收益": f"{self.annual_return:.2%}",
            "最大回撤": f"{self.max_dd:.2%}",
            "夏普比率": f"{self.sharpe:.2f}",
        }

# 使用
# pm = PerformanceMetrics(returns)
# print(pm.summary())
```

---

## 结语

恭喜你完成了 Qlib 进阶教程！现在你已经掌握了：

| 模块 | 核心技能 |
|------|----------|
| **可视化分析** | K线图、因子分布、预测结果、IC时序 |
| **Alpha158因子** | 6大类因子原理、因子有效性检验、自定义因子 |
| **回测评估** | 夏普比率、最大回撤、IC/IR、完整回测流程 |

### 下一步学习建议

1. **实战练习**：用真实数据跑完整回测
2. **因子研究**：尝试自定义新因子
3. **策略优化**：调整参数提升夏普比率
4. **进入 Step 5**：加密货币数据整合

---

*教程版本: v2.0 | 更新日期: 2026-01 | 对应 AlgVex 2.0.0*
