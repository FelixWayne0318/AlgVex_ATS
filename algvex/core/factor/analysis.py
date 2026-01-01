"""
AlgVex 因子分析工具 (Qlib 风格)

实现 Qlib 的完整因子分析功能:
- IC/Rank IC 计算
- 因子收益分析
- 多空收益计算
- 分组回测
- 相关性矩阵
- 因子衰减分析

用法:
    from algvex.core.factor.analysis import FactorAnalyzer

    analyzer = FactorAnalyzer()
    ic_result = analyzer.calc_ic(predictions, labels)
    report = analyzer.generate_report(predictions, labels, returns)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ICResult:
    """IC 计算结果"""
    ic: float                       # Pearson IC
    rank_ic: float                  # Spearman Rank IC
    ic_mean: float                  # IC 均值
    ic_std: float                   # IC 标准差
    icir: float                     # IC 信息比率 (IC/std)
    rank_icir: float                # Rank IC 信息比率
    positive_ratio: float           # IC > 0 的比率
    ic_series: pd.Series            # 时序 IC
    rank_ic_series: pd.Series       # 时序 Rank IC


@dataclass
class FactorReport:
    """因子分析报告"""
    ic_result: ICResult
    long_short_return: float        # 多空收益
    long_return: float              # 多头收益
    short_return: float             # 空头收益
    turnover: float                 # 换手率
    quantile_returns: pd.DataFrame  # 分组收益
    decay_ic: pd.Series             # IC 衰减


class FactorAnalyzer:
    """
    因子分析器 (Qlib 原版)

    提供完整的因子评估功能
    """

    def __init__(self, n_quantiles: int = 5):
        """
        初始化分析器

        Args:
            n_quantiles: 分组数量
        """
        self.n_quantiles = n_quantiles

    def calc_ic(
        self,
        predictions: Union[pd.Series, pd.DataFrame],
        labels: Union[pd.Series, pd.DataFrame],
        method: str = "pearson",
    ) -> Union[float, Tuple[float, float]]:
        """
        计算 IC (Information Coefficient)

        Args:
            predictions: 预测值
            labels: 实际标签
            method: 'pearson' 或 'spearman'

        Returns:
            IC 值 (如果 method='both' 返回元组)
        """
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.iloc[:, 0]
        if isinstance(labels, pd.DataFrame):
            labels = labels.iloc[:, 0]

        # 对齐索引
        common_idx = predictions.index.intersection(labels.index)
        pred = predictions.loc[common_idx].values
        label = labels.loc[common_idx].values

        # 移除 NaN
        mask = ~(np.isnan(pred) | np.isnan(label))
        pred = pred[mask]
        label = label[mask]

        if len(pred) < 2:
            return np.nan if method != "both" else (np.nan, np.nan)

        if method == "pearson":
            return np.corrcoef(pred, label)[0, 1]
        elif method == "spearman":
            if SCIPY_AVAILABLE:
                return stats.spearmanr(pred, label)[0]
            else:
                # 手动计算 Spearman
                pred_rank = pd.Series(pred).rank().values
                label_rank = pd.Series(label).rank().values
                return np.corrcoef(pred_rank, label_rank)[0, 1]
        elif method == "both":
            pearson_ic = np.corrcoef(pred, label)[0, 1]
            if SCIPY_AVAILABLE:
                spearman_ic = stats.spearmanr(pred, label)[0]
            else:
                pred_rank = pd.Series(pred).rank().values
                label_rank = pd.Series(label).rank().values
                spearman_ic = np.corrcoef(pred_rank, label_rank)[0, 1]
            return pearson_ic, spearman_ic
        else:
            raise ValueError(f"Unknown method: {method}")

    def calc_ic_series(
        self,
        predictions: pd.DataFrame,
        labels: pd.DataFrame,
        group_col: str = None,
    ) -> ICResult:
        """
        计算时序 IC

        Args:
            predictions: 预测值 (index=datetime, columns=assets 或单列)
            labels: 实际标签
            group_col: 分组列 (如日期)

        Returns:
            ICResult 对象
        """
        # 确保是 DataFrame
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame('prediction')
        if isinstance(labels, pd.Series):
            labels = labels.to_frame('label')

        # 如果有 MultiIndex，提取日期作为分组
        if isinstance(predictions.index, pd.MultiIndex):
            dates = predictions.index.get_level_values(0).unique()
        elif group_col is not None:
            dates = predictions[group_col].unique()
        else:
            # 假设索引是 datetime
            dates = predictions.index.unique()

        ic_list = []
        rank_ic_list = []

        for date in dates:
            try:
                if isinstance(predictions.index, pd.MultiIndex):
                    pred_slice = predictions.loc[date]
                    label_slice = labels.loc[date]
                else:
                    pred_slice = predictions.loc[[date]]
                    label_slice = labels.loc[[date]]

                if len(pred_slice) < 2:
                    continue

                ic, rank_ic = self.calc_ic(
                    pred_slice.iloc[:, 0],
                    label_slice.iloc[:, 0],
                    method="both"
                )

                ic_list.append((date, ic))
                rank_ic_list.append((date, rank_ic))
            except Exception as e:
                logger.debug(f"Skipping date {date}: {e}")
                continue

        if not ic_list:
            return ICResult(
                ic=np.nan, rank_ic=np.nan, ic_mean=np.nan, ic_std=np.nan,
                icir=np.nan, rank_icir=np.nan, positive_ratio=np.nan,
                ic_series=pd.Series(), rank_ic_series=pd.Series()
            )

        ic_series = pd.Series(dict(ic_list))
        rank_ic_series = pd.Series(dict(rank_ic_list))

        ic_mean = ic_series.mean()
        ic_std = ic_series.std()
        rank_ic_mean = rank_ic_series.mean()
        rank_ic_std = rank_ic_series.std()

        return ICResult(
            ic=ic_mean,
            rank_ic=rank_ic_mean,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=ic_mean / ic_std if ic_std > 0 else np.nan,
            rank_icir=rank_ic_mean / rank_ic_std if rank_ic_std > 0 else np.nan,
            positive_ratio=(ic_series > 0).mean(),
            ic_series=ic_series,
            rank_ic_series=rank_ic_series,
        )

    def calc_long_short_return(
        self,
        predictions: pd.Series,
        returns: pd.Series,
        n_quantiles: int = None,
    ) -> Dict[str, float]:
        """
        计算多空收益

        Args:
            predictions: 预测值
            returns: 实际收益率
            n_quantiles: 分组数

        Returns:
            多空收益指标
        """
        n_quantiles = n_quantiles or self.n_quantiles

        # 对齐
        common_idx = predictions.index.intersection(returns.index)
        pred = predictions.loc[common_idx]
        ret = returns.loc[common_idx]

        # 去除 NaN
        mask = ~(pred.isna() | ret.isna())
        pred = pred[mask]
        ret = ret[mask]

        if len(pred) < n_quantiles:
            return {'long_short': np.nan, 'long': np.nan, 'short': np.nan}

        # 分组
        try:
            quantiles = pd.qcut(pred, n_quantiles, labels=False, duplicates='drop')
        except ValueError:
            # 如果分组失败，使用百分位
            quantiles = pd.cut(pred.rank(pct=True), n_quantiles, labels=False)

        # 计算各组收益
        group_returns = ret.groupby(quantiles).mean()

        # 多空收益
        long_return = group_returns.iloc[-1] if len(group_returns) > 0 else np.nan
        short_return = group_returns.iloc[0] if len(group_returns) > 0 else np.nan
        long_short = long_return - short_return

        return {
            'long_short': long_short,
            'long': long_return,
            'short': short_return,
            'quantile_returns': group_returns,
        }

    def calc_quantile_returns(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        n_quantiles: int = None,
    ) -> pd.DataFrame:
        """
        计算分组收益

        Args:
            predictions: 预测值 (每行是一个时间点)
            returns: 实际收益率
            n_quantiles: 分组数

        Returns:
            分组收益 DataFrame
        """
        n_quantiles = n_quantiles or self.n_quantiles

        # 确保是 DataFrame
        if isinstance(predictions, pd.Series):
            predictions = predictions.to_frame('prediction')
        if isinstance(returns, pd.Series):
            returns = returns.to_frame('return')

        results = []

        for idx in predictions.index:
            if idx not in returns.index:
                continue

            pred = predictions.loc[idx]
            ret = returns.loc[idx]

            if isinstance(pred, pd.Series) and isinstance(ret, pd.Series):
                # 单资产
                continue

            # 对齐资产
            common = pred.index.intersection(ret.index)
            if len(common) < n_quantiles:
                continue

            pred_slice = pred[common]
            ret_slice = ret[common]

            # 分组
            try:
                quantiles = pd.qcut(pred_slice, n_quantiles, labels=False, duplicates='drop')
            except ValueError:
                continue

            # 各组平均收益
            group_ret = ret_slice.groupby(quantiles).mean()
            group_ret.name = idx
            results.append(group_ret)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)

    def calc_factor_decay(
        self,
        predictions: pd.Series,
        future_returns: Dict[int, pd.Series],
        max_lag: int = 20,
    ) -> pd.Series:
        """
        计算因子衰减

        Args:
            predictions: 预测值
            future_returns: {lag: 未来 lag 期的收益率}
            max_lag: 最大滞后期

        Returns:
            各滞后期的 IC
        """
        decay_ic = {}

        for lag in range(1, max_lag + 1):
            if lag not in future_returns:
                continue

            ret = future_returns[lag]
            common_idx = predictions.index.intersection(ret.index)

            if len(common_idx) < 10:
                continue

            ic = self.calc_ic(predictions.loc[common_idx], ret.loc[common_idx])
            decay_ic[lag] = ic

        return pd.Series(decay_ic)

    def calc_turnover(
        self,
        positions_series: List[pd.Series],
    ) -> float:
        """
        计算换手率

        Args:
            positions_series: 持仓权重序列

        Returns:
            平均换手率
        """
        if len(positions_series) < 2:
            return 0.0

        turnovers = []
        for i in range(1, len(positions_series)):
            prev = positions_series[i-1]
            curr = positions_series[i]

            # 对齐
            all_assets = prev.index.union(curr.index)
            prev_aligned = prev.reindex(all_assets, fill_value=0)
            curr_aligned = curr.reindex(all_assets, fill_value=0)

            # 换手率 = |Δw| / 2
            turnover = np.abs(curr_aligned - prev_aligned).sum() / 2
            turnovers.append(turnover)

        return np.mean(turnovers)

    def calc_factor_correlation(
        self,
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        计算因子相关性矩阵

        Args:
            factors: 因子数据 (columns=因子名)

        Returns:
            相关性矩阵
        """
        return factors.corr()

    def calc_factor_rank_correlation(
        self,
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        计算因子秩相关性矩阵

        Args:
            factors: 因子数据

        Returns:
            秩相关性矩阵
        """
        if SCIPY_AVAILABLE:
            n = factors.shape[1]
            corr_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i <= j:
                        mask = ~(factors.iloc[:, i].isna() | factors.iloc[:, j].isna())
                        if mask.sum() > 2:
                            corr, _ = stats.spearmanr(
                                factors.iloc[:, i][mask],
                                factors.iloc[:, j][mask]
                            )
                            corr_matrix[i, j] = corr
                            corr_matrix[j, i] = corr
            return pd.DataFrame(
                corr_matrix,
                index=factors.columns,
                columns=factors.columns
            )
        else:
            # 使用排名后的 Pearson 相关
            return factors.rank().corr()

    def generate_report(
        self,
        predictions: pd.Series,
        labels: pd.Series,
        returns: pd.Series = None,
    ) -> FactorReport:
        """
        生成完整因子报告

        Args:
            predictions: 预测值
            labels: 实际标签
            returns: 实际收益率 (可选，用于多空分析)

        Returns:
            FactorReport 对象
        """
        # IC 分析
        ic, rank_ic = self.calc_ic(predictions, labels, method="both")

        # 创建简化的 ICResult
        ic_result = ICResult(
            ic=ic,
            rank_ic=rank_ic,
            ic_mean=ic,
            ic_std=0.0,
            icir=np.nan,
            rank_icir=np.nan,
            positive_ratio=1.0 if ic > 0 else 0.0,
            ic_series=pd.Series([ic]),
            rank_ic_series=pd.Series([rank_ic]),
        )

        # 多空收益
        if returns is not None:
            ls_result = self.calc_long_short_return(predictions, returns)
            long_short_return = ls_result['long_short']
            long_return = ls_result['long']
            short_return = ls_result['short']
            quantile_returns = ls_result.get('quantile_returns', pd.DataFrame())
        else:
            long_short_return = np.nan
            long_return = np.nan
            short_return = np.nan
            quantile_returns = pd.DataFrame()

        return FactorReport(
            ic_result=ic_result,
            long_short_return=long_short_return,
            long_return=long_return,
            short_return=short_return,
            turnover=0.0,
            quantile_returns=quantile_returns,
            decay_ic=pd.Series(),
        )

    def print_report(self, report: FactorReport):
        """打印因子报告"""
        print("=" * 60)
        print("因子分析报告")
        print("=" * 60)
        print(f"\n📊 IC 分析:")
        print(f"  Pearson IC:   {report.ic_result.ic:.4f}")
        print(f"  Rank IC:      {report.ic_result.rank_ic:.4f}")
        print(f"  IC IR:        {report.ic_result.icir:.4f}")
        print(f"  Rank IC IR:   {report.ic_result.rank_icir:.4f}")
        print(f"  IC 正向比例:  {report.ic_result.positive_ratio:.2%}")

        if not np.isnan(report.long_short_return):
            print(f"\n📈 多空收益:")
            print(f"  多空收益:     {report.long_short_return:.4f}")
            print(f"  多头收益:     {report.long_return:.4f}")
            print(f"  空头收益:     {report.short_return:.4f}")

        if not report.quantile_returns.empty:
            print(f"\n📊 分组收益:")
            print(report.quantile_returns)

        print("=" * 60)


# ============================================================
# 风险分析 (Qlib 原版)
# ============================================================

def risk_analysis(
    returns: pd.Series,
    rf: float = 0.0,
    freq: str = 'day',
) -> Dict[str, float]:
    """
    风险分析

    Args:
        returns: 收益率序列
        rf: 无风险利率 (年化)
        freq: 频率 ('day', 'hour', 'minute', '5min')

    Returns:
        风险指标字典
    """
    # 频率转换因子
    freq_map = {
        'day': 252,
        'hour': 252 * 24,
        'minute': 252 * 24 * 60,
        '5min': 252 * 24 * 12,
        '1h': 252 * 24,
        '5m': 252 * 24 * 12,
    }
    ann_factor = freq_map.get(freq, 252)

    # 基础统计
    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
    ann_volatility = returns.std() * np.sqrt(ann_factor)

    # 夏普比率
    excess_return = ann_return - rf
    sharpe_ratio = excess_return / ann_volatility if ann_volatility > 0 else np.nan

    # 最大回撤
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar 比率
    calmar_ratio = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # 胜率
    win_rate = (returns > 0).mean()

    # 盈亏比
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_loss_ratio = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.nan

    return {
        'total_return': total_return,
        'annual_return': ann_return,
        'annual_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'profit_loss_ratio': profit_loss_ratio,
        'n_trades': len(returns),
    }


def print_risk_analysis(metrics: Dict[str, float]):
    """打印风险分析结果"""
    print("=" * 60)
    print("风险分析报告")
    print("=" * 60)
    print(f"  总收益:       {metrics['total_return']:.2%}")
    print(f"  年化收益:     {metrics['annual_return']:.2%}")
    print(f"  年化波动率:   {metrics['annual_volatility']:.2%}")
    print(f"  夏普比率:     {metrics['sharpe_ratio']:.2f}")
    print(f"  最大回撤:     {metrics['max_drawdown']:.2%}")
    print(f"  卡玛比率:     {metrics['calmar_ratio']:.2f}")
    print(f"  胜率:         {metrics['win_rate']:.2%}")
    print(f"  盈亏比:       {metrics['profit_loss_ratio']:.2f}")
    print("=" * 60)


# ============================================================
# 便捷函数
# ============================================================

def calc_ic(
    predictions: pd.Series,
    labels: pd.Series,
) -> Tuple[float, float]:
    """
    计算 IC (便捷函数)

    Returns:
        (Pearson IC, Rank IC)
    """
    analyzer = FactorAnalyzer()
    return analyzer.calc_ic(predictions, labels, method="both")


def generate_factor_report(
    predictions: pd.Series,
    labels: pd.Series,
    returns: pd.Series = None,
) -> FactorReport:
    """
    生成因子报告 (便捷函数)
    """
    analyzer = FactorAnalyzer()
    return analyzer.generate_report(predictions, labels, returns)


# 导出
__all__ = [
    'ICResult',
    'FactorReport',
    'FactorAnalyzer',
    'risk_analysis',
    'print_risk_analysis',
    'calc_ic',
    'generate_factor_report',
]
