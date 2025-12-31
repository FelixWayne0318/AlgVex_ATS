"""
AlgVex 评估模块 (Qlib 风格)

实现 Qlib 的评估函数:
- risk_analysis: 风险分析 (年化收益、夏普比率、最大回撤)
- calc_ic: IC 和 Rank IC 计算
- calc_long_short_return: 多空收益计算
- calc_long_short_prec: 多空精度计算
- indicator_analysis: 指标分析

用法:
    from algvex.core.evaluate import risk_analysis, calc_ic

    # 风险分析
    metrics = risk_analysis(returns, freq='day')

    # IC 分析
    ic, ric = calc_ic(pred, label)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Union, Literal, Dict, Optional

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


# ============================================================
# 风险分析 (Qlib risk_analysis)
# ============================================================

def risk_analysis(
    r: pd.Series,
    N: int = None,
    freq: str = "day",
    mode: Literal["sum", "product"] = "sum",
) -> pd.Series:
    """
    风险分析 (Qlib 原版)

    计算收益序列的风险指标

    Args:
        r: 收益率序列 (日收益率)
        N: 年化系数 (None=自动计算)
        freq: 频率 ('minute', 'day', 'week', 'month')
        mode: 累积方式 ('sum'=算术, 'product'=几何)

    Returns:
        pd.Series: 包含以下指标
            - mean: 平均收益
            - std: 收益波动率
            - annualized_return: 年化收益率
            - information_ratio: 信息比率 (夏普)
            - max_drawdown: 最大回撤
    """
    # 计算年化系数
    if N is None:
        freq_map = {
            'minute': 252 * 240,  # 每天240分钟
            'day': 252,
            'week': 50,
            'month': 12,
        }
        N = freq_map.get(freq, 252)

    # 清理数据
    r = r.dropna()

    if len(r) == 0:
        return pd.Series({
            'mean': np.nan,
            'std': np.nan,
            'annualized_return': np.nan,
            'information_ratio': np.nan,
            'max_drawdown': np.nan,
        }, name='risk')

    if mode == "sum":
        # 算术累积
        cum_ret = r.cumsum()
        mean_ret = r.mean()
        std_ret = r.std()
        ann_ret = mean_ret * N
        ir = mean_ret / std_ret * np.sqrt(N) if std_ret > 0 else np.nan

        # 最大回撤
        max_so_far = cum_ret.cummax()
        dd = max_so_far - cum_ret
        max_dd = dd.max()

    else:  # product
        # 几何累积
        cum_ret = (1 + r).cumprod()

        # 几何平均 (CAGR)
        total_ret = cum_ret.iloc[-1] - 1
        n_periods = len(r)
        ann_ret = (1 + total_ret) ** (N / n_periods) - 1 if n_periods > 0 else np.nan

        # 对数收益的波动率
        log_ret = np.log(1 + r)
        mean_ret = log_ret.mean()
        std_ret = log_ret.std()
        ir = mean_ret / std_ret * np.sqrt(N) if std_ret > 0 else np.nan

        # 最大回撤 (百分比)
        max_so_far = cum_ret.cummax()
        dd = (max_so_far - cum_ret) / max_so_far
        max_dd = dd.max()

    return pd.Series({
        'mean': mean_ret,
        'std': std_ret,
        'annualized_return': ann_ret,
        'information_ratio': ir,
        'max_drawdown': max_dd,
    }, name='risk')


# ============================================================
# IC 计算 (Qlib calc_ic)
# ============================================================

def calc_ic(
    pred: pd.Series,
    label: pd.Series,
    date_col: str = "datetime",
    dropna: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    计算 IC 和 Rank IC (Qlib 原版)

    IC = Pearson 相关系数
    Rank IC = Spearman 秩相关系数

    Args:
        pred: 预测值 (MultiIndex: [datetime, instrument])
        label: 实际值 (MultiIndex: [datetime, instrument])
        date_col: 时间列名
        dropna: 是否删除缺失值

    Returns:
        Tuple[pd.Series, pd.Series]: (IC序列, Rank IC序列)
    """
    # 对齐数据
    df = pd.DataFrame({'pred': pred, 'label': label})

    if dropna:
        df = df.dropna()

    # 按时间分组计算相关系数
    def calc_pearson(x):
        if len(x) < 2:
            return np.nan
        return x['pred'].corr(x['label'])

    def calc_spearman(x):
        if len(x) < 2:
            return np.nan
        return x['pred'].corr(x['label'], method='spearman')

    # 获取时间索引
    if isinstance(df.index, pd.MultiIndex):
        ic = df.groupby(level=0).apply(calc_pearson)
        ric = df.groupby(level=0).apply(calc_spearman)
    else:
        # 单一索引，假设就是时间
        ic = pd.Series([calc_pearson(df)], index=[df.index[0]])
        ric = pd.Series([calc_spearman(df)], index=[df.index[0]])

    return ic, ric


def calc_ic_summary(ic: pd.Series, ric: pd.Series) -> Dict[str, float]:
    """
    IC 汇总统计

    Args:
        ic: IC 序列
        ric: Rank IC 序列

    Returns:
        Dict: IC统计指标
    """
    return {
        'IC': ic.mean(),
        'ICIR': ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        'Rank IC': ric.mean(),
        'Rank ICIR': ric.mean() / ric.std() if ric.std() > 0 else np.nan,
        'IC > 0': (ic > 0).mean(),
    }


# ============================================================
# 多空收益计算 (Qlib calc_long_short_return)
# ============================================================

def calc_long_short_return(
    pred: pd.Series,
    label: pd.Series,
    date_col: str = "datetime",
    quantile: float = 0.2,
    dropna: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    计算多空收益 (Qlib 原版)

    选择预测得分最高的 quantile 做多，最低的 quantile 做空

    Args:
        pred: 预测值 (MultiIndex: [datetime, instrument])
        label: 实际收益 (MultiIndex: [datetime, instrument])
        date_col: 时间列名
        quantile: 分位数 (默认 0.2 = 20%)
        dropna: 是否删除缺失值

    Returns:
        Tuple[pd.Series, pd.Series]:
            - long_short_return: 多空收益 = (long - short) / 2
            - long_avg_return: 多头平均收益
    """
    df = pd.DataFrame({'pred': pred, 'label': label})

    if dropna:
        df = df.dropna()

    def calc_ls_return(x):
        if len(x) < 5:  # 至少5个样本
            return np.nan, np.nan

        n = len(x)
        top_n = max(1, int(n * quantile))

        # 排序
        sorted_df = x.sort_values('pred', ascending=False)

        # 多头: 预测最高的
        long_ret = sorted_df.head(top_n)['label'].mean()

        # 空头: 预测最低的
        short_ret = sorted_df.tail(top_n)['label'].mean()

        return (long_ret - short_ret) / 2, long_ret

    # 按时间分组
    if isinstance(df.index, pd.MultiIndex):
        results = df.groupby(level=0).apply(lambda x: pd.Series(calc_ls_return(x)))
        long_short = results[0]
        long_avg = results[1]
    else:
        ls, la = calc_ls_return(df)
        long_short = pd.Series([ls], index=[df.index[0]])
        long_avg = pd.Series([la], index=[df.index[0]])

    return long_short, long_avg


# ============================================================
# 多空精度计算 (Qlib calc_long_short_prec)
# ============================================================

def calc_long_short_prec(
    pred: pd.Series,
    label: pd.Series,
    date_col: str = "datetime",
    quantile: float = 0.2,
    dropna: bool = False,
    is_alpha: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """
    计算多空精度 (Qlib 原版)

    多头精度: 预测最高的股票中，实际收益为正的比例
    空头精度: 预测最低的股票中，实际收益为负的比例

    Args:
        pred: 预测值
        label: 实际收益
        date_col: 时间列名
        quantile: 分位数
        dropna: 是否删除缺失值
        is_alpha: 是否减去截面均值

    Returns:
        Tuple[pd.Series, pd.Series]: (多头精度, 空头精度)
    """
    df = pd.DataFrame({'pred': pred, 'label': label})

    if dropna:
        df = df.dropna()

    def calc_prec(x):
        if len(x) < 5:
            return np.nan, np.nan

        label_col = x['label']

        # 如果是 alpha，减去均值
        if is_alpha:
            label_col = label_col - label_col.mean()

        n = len(x)
        top_n = max(1, int(n * quantile))

        # 排序
        sorted_idx = x['pred'].sort_values(ascending=False).index

        # 多头精度: 预测最高的中，正收益的比例
        long_idx = sorted_idx[:top_n]
        long_prec = (label_col.loc[long_idx] > 0).mean()

        # 空头精度: 预测最低的中，负收益的比例
        short_idx = sorted_idx[-top_n:]
        short_prec = (label_col.loc[short_idx] < 0).mean()

        return long_prec, short_prec

    # 按时间分组
    if isinstance(df.index, pd.MultiIndex):
        results = df.groupby(level=0).apply(lambda x: pd.Series(calc_prec(x)))
        long_prec = results[0]
        short_prec = results[1]
    else:
        lp, sp = calc_prec(df)
        long_prec = pd.Series([lp], index=[df.index[0]])
        short_prec = pd.Series([sp], index=[df.index[0]])

    return long_prec, short_prec


# ============================================================
# 指标分析 (Qlib indicator_analysis)
# ============================================================

def indicator_analysis(
    df: pd.DataFrame,
    method: str = "mean",
) -> pd.DataFrame:
    """
    交易指标分析 (Qlib 原版)

    Args:
        df: 包含以下列的 DataFrame:
            - pa: 价格优势
            - pos: 胜率
            - ffr: 成交率
            - count: 订单数
            - deal_amount (可选): 成交量 (amount_weighted 需要)
            - value (可选): 成交额 (value_weighted 需要)
        method: 统计方法 ('mean', 'amount_weighted', 'value_weighted')

    Returns:
        pd.DataFrame: 各指标的统计值
    """
    result = {}

    indicators = ['pa', 'ffr', 'pos']

    for ind in indicators:
        if ind not in df.columns:
            result[ind] = np.nan
            continue

        if method == 'mean':
            result[ind] = df[ind].mean()
        elif method == 'amount_weighted' and 'deal_amount' in df.columns:
            weights = df['deal_amount']
            result[ind] = (df[ind] * weights).sum() / weights.sum()
        elif method == 'value_weighted' and 'value' in df.columns:
            weights = df['value']
            result[ind] = (df[ind] * weights).sum() / weights.sum()
        else:
            result[ind] = df[ind].mean()

    return pd.DataFrame({'value': result})


# ============================================================
# 预测自相关 (Qlib pred_autocorr)
# ============================================================

def pred_autocorr(
    pred: pd.Series,
    lag: int = 1,
    inst_col: str = "instrument",
    date_col: str = "datetime",
) -> pd.Series:
    """
    预测自相关 (Qlib 原版)

    计算预测序列在时间上的自相关性

    Args:
        pred: 预测序列 (MultiIndex: [instrument, datetime])
        lag: 滞后期数
        inst_col: 股票列名
        date_col: 时间列名

    Returns:
        pd.Series: 每个时间点的自相关系数
    """
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]

    # 确保索引顺序正确
    if isinstance(pred.index, pd.MultiIndex):
        pred = pred.unstack(level=0)  # 转为 datetime x instrument

    # 计算滞后相关
    result = {}
    dates = pred.index.tolist()

    for i in range(lag, len(dates)):
        curr = pred.loc[dates[i]]
        prev = pred.loc[dates[i - lag]]

        # 取交集
        common = curr.index.intersection(prev.index)
        if len(common) < 2:
            continue

        corr = curr.loc[common].corr(prev.loc[common])
        result[dates[i]] = corr

    return pd.Series(result)


# ============================================================
# 综合评估报告
# ============================================================

def generate_report(
    pred: pd.Series,
    label: pd.Series,
    returns: pd.Series = None,
    freq: str = 'day',
) -> Dict[str, Union[float, pd.Series]]:
    """
    生成综合评估报告

    Args:
        pred: 预测值
        label: 标签值
        returns: 收益序列 (可选)
        freq: 频率

    Returns:
        Dict: 评估指标
    """
    report = {}

    # IC 分析
    ic, ric = calc_ic(pred, label)
    report['ic'] = ic
    report['ric'] = ric
    report['ic_summary'] = calc_ic_summary(ic, ric)

    # 多空分析
    ls_ret, long_ret = calc_long_short_return(pred, label)
    report['long_short_return'] = ls_ret
    report['long_return'] = long_ret

    long_prec, short_prec = calc_long_short_prec(pred, label)
    report['long_precision'] = long_prec
    report['short_precision'] = short_prec

    # 风险分析 (如果有收益序列)
    if returns is not None:
        report['risk'] = risk_analysis(returns, freq=freq)

    return report
