"""
AlgVex 投资组合优化器 (Qlib 风格)

实现 Qlib 的 4 种投资组合优化方法:
- GMV: 全局最小方差组合 (Global Minimum Variance)
- MVO: 均值-方差优化 (Mean-Variance Optimization)
- Risk Parity: 风险平价
- Inverse Volatility: 反向波动率加权

用法:
    from algvex.core.strategy.portfolio_optimizer import PortfolioOptimizer

    # 创建优化器
    optimizer = PortfolioOptimizer(method='risk_parity')

    # 优化权重
    weights = optimizer.optimize(returns_df, expected_returns=pred)
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, List, Literal
from enum import Enum
import warnings

try:
    from scipy.optimize import minimize
    from scipy.linalg import inv, cholesky
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class OptMethod(str, Enum):
    """优化方法枚举"""
    GMV = "gmv"                     # 全局最小方差
    MVO = "mvo"                     # 均值-方差优化
    RISK_PARITY = "risk_parity"     # 风险平价
    INV_VOL = "inv_vol"             # 反向波动率


class PortfolioOptimizer:
    """
    投资组合优化器 (Qlib 原版)

    支持多种优化方法和约束条件

    Args:
        method: 优化方法 ('gmv', 'mvo', 'risk_parity', 'inv_vol')
        risk_aversion: 风险厌恶系数 (仅 MVO)
        target_return: 目标收益率 (仅 MVO)
        long_only: 是否只做多
        max_weight: 单一资产最大权重
        min_weight: 单一资产最小权重
        regularization: 正则化参数
    """

    def __init__(
        self,
        method: str = "risk_parity",
        risk_aversion: float = 1.0,
        target_return: Optional[float] = None,
        long_only: bool = True,
        max_weight: float = 1.0,
        min_weight: float = 0.0,
        regularization: float = 0.0,
    ):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for portfolio optimization")

        self.method = OptMethod(method.lower())
        self.risk_aversion = risk_aversion
        self.target_return = target_return
        self.long_only = long_only
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.regularization = regularization

    def optimize(
        self,
        returns: Union[pd.DataFrame, np.ndarray],
        expected_returns: Optional[Union[pd.Series, np.ndarray]] = None,
        cov_matrix: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> Union[pd.Series, np.ndarray]:
        """
        优化投资组合权重

        Args:
            returns: 历史收益率 (用于估计协方差)
            expected_returns: 预期收益率 (用于 MVO)
            cov_matrix: 协方差矩阵 (可选，否则从 returns 估计)

        Returns:
            优化后的权重
        """
        # 转换为 numpy
        if isinstance(returns, pd.DataFrame):
            assets = returns.columns.tolist()
            returns_np = returns.values
        else:
            assets = None
            returns_np = returns

        n_assets = returns_np.shape[1]

        # 估计协方差矩阵
        if cov_matrix is None:
            cov_matrix = np.cov(returns_np.T)
            # 添加正则化
            if self.regularization > 0:
                cov_matrix = cov_matrix + self.regularization * np.eye(n_assets)
        elif isinstance(cov_matrix, pd.DataFrame):
            cov_matrix = cov_matrix.values

        # 估计预期收益
        if expected_returns is None:
            expected_returns = np.mean(returns_np, axis=0)
        elif isinstance(expected_returns, pd.Series):
            expected_returns = expected_returns.values

        # 根据方法优化
        if self.method == OptMethod.GMV:
            weights = self._optimize_gmv(cov_matrix)
        elif self.method == OptMethod.MVO:
            weights = self._optimize_mvo(cov_matrix, expected_returns)
        elif self.method == OptMethod.RISK_PARITY:
            weights = self._optimize_risk_parity(cov_matrix)
        elif self.method == OptMethod.INV_VOL:
            weights = self._optimize_inv_vol(cov_matrix)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 返回格式
        if assets is not None:
            return pd.Series(weights, index=assets)
        return weights

    def _optimize_gmv(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        全局最小方差组合

        min w'Σw
        s.t. sum(w) = 1
        """
        n = cov_matrix.shape[0]

        # 目标函数: 组合方差
        def objective(w):
            return w @ cov_matrix @ w

        # 梯度
        def gradient(w):
            return 2 * cov_matrix @ w

        # 约束
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # 边界
        if self.long_only:
            bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        else:
            bounds = [(-self.max_weight, self.max_weight) for _ in range(n)]

        # 初始权重
        w0 = np.ones(n) / n

        # 优化
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            jac=gradient,
            constraints=constraints,
            bounds=bounds
        )

        if not result.success:
            logger.warning(f"GMV optimization did not converge: {result.message}")

        return result.x

    def _optimize_mvo(
        self,
        cov_matrix: np.ndarray,
        expected_returns: np.ndarray
    ) -> np.ndarray:
        """
        均值-方差优化

        max μ'w - (λ/2) * w'Σw
        s.t. sum(w) = 1
        """
        n = cov_matrix.shape[0]
        lambda_param = self.risk_aversion

        # 目标函数: 负效用 (最大化转最小化)
        def objective(w):
            return -(expected_returns @ w) + (lambda_param / 2) * (w @ cov_matrix @ w)

        # 梯度
        def gradient(w):
            return -expected_returns + lambda_param * (cov_matrix @ w)

        # 约束
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # 如果有目标收益约束
        if self.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: expected_returns @ w - self.target_return
            })

        # 边界
        if self.long_only:
            bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        else:
            bounds = [(-self.max_weight, self.max_weight) for _ in range(n)]

        # 初始权重
        w0 = np.ones(n) / n

        # 优化
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            jac=gradient,
            constraints=constraints,
            bounds=bounds
        )

        if not result.success:
            logger.warning(f"MVO optimization did not converge: {result.message}")

        return result.x

    def _optimize_risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        风险平价

        使每个资产的风险贡献相等
        """
        n = cov_matrix.shape[0]

        # 目标函数: 使风险贡献差异最小化
        def risk_contribution(w):
            sigma = np.sqrt(w @ cov_matrix @ w)
            marginal_risk = cov_matrix @ w
            risk_contrib = w * marginal_risk / sigma
            return risk_contrib

        def objective(w):
            rc = risk_contribution(w)
            target_rc = 1.0 / n  # 等风险贡献
            return np.sum((rc - target_rc) ** 2)

        # 约束
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        # 边界 (风险平价通常要求正权重)
        bounds = [(1e-6, self.max_weight) for _ in range(n)]

        # 初始权重
        w0 = np.ones(n) / n

        # 优化
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )

        if not result.success:
            logger.warning(f"Risk Parity optimization did not converge: {result.message}")

        return result.x

    def _optimize_inv_vol(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        反向波动率加权

        w_i = (1/σ_i) / sum(1/σ_j)
        """
        # 提取波动率 (对角线的平方根)
        volatilities = np.sqrt(np.diag(cov_matrix))

        # 计算反向波动率权重
        inv_vol = 1.0 / volatilities
        weights = inv_vol / np.sum(inv_vol)

        # 应用权重限制
        weights = np.clip(weights, self.min_weight, self.max_weight)

        # 重新归一化
        weights = weights / np.sum(weights)

        return weights


class RiskModel:
    """
    风险模型 (Qlib 原版)

    估计协方差矩阵的多种方法
    """

    @staticmethod
    def sample_cov(returns: np.ndarray) -> np.ndarray:
        """样本协方差"""
        return np.cov(returns.T)

    @staticmethod
    def shrinkage_cov(returns: np.ndarray, shrinkage: float = 0.5) -> np.ndarray:
        """
        收缩协方差估计

        Σ_shrink = shrinkage * diag(Σ) + (1-shrinkage) * Σ
        """
        sample_cov = np.cov(returns.T)
        target = np.diag(np.diag(sample_cov))
        return shrinkage * target + (1 - shrinkage) * sample_cov

    @staticmethod
    def exponential_cov(returns: np.ndarray, halflife: int = 60) -> np.ndarray:
        """
        指数加权协方差

        使用 EWM 方法，更重视近期数据
        """
        T = returns.shape[0]
        decay = 0.5 ** (1 / halflife)

        # 计算权重
        weights = decay ** np.arange(T-1, -1, -1)
        weights = weights / weights.sum()

        # 去均值
        mean = np.average(returns, axis=0, weights=weights)
        centered = returns - mean

        # 加权协方差
        cov = np.zeros((returns.shape[1], returns.shape[1]))
        for t in range(T):
            cov += weights[t] * np.outer(centered[t], centered[t])

        return cov


class TopkDropoutStrategy:
    """
    TopkDropout 策略 (Qlib 原版)

    选择 Top-K 资产，并使用 Dropout 机制减少换手

    Args:
        topk: 选择的资产数量
        n_drop: 每期最多替换的资产数
        method: 权重分配方法 ('equal', 'score', 'inv_vol')
        only_tradable_days: 仅在可交易日交易
    """

    def __init__(
        self,
        topk: int = 20,
        n_drop: int = 3,
        method: str = "equal",
        only_tradable_days: bool = True,
    ):
        self.topk = topk
        self.n_drop = n_drop
        self.method = method
        self.only_tradable_days = only_tradable_days

        self.current_holdings: List[str] = []

    def generate_signal(
        self,
        predictions: pd.Series,
        current_holdings: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        生成交易信号

        Args:
            predictions: 预测分数 (资产 -> 分数)
            current_holdings: 当前持仓 (可选)

        Returns:
            目标权重 (资产 -> 权重)
        """
        if current_holdings is not None:
            self.current_holdings = list(current_holdings)

        # 排序
        sorted_pred = predictions.sort_values(ascending=False)

        # 如果没有当前持仓，选择 Top-K
        if not self.current_holdings:
            selected = sorted_pred.head(self.topk).index.tolist()
        else:
            # 计算需要替换的资产
            current_set = set(self.current_holdings)
            top_k_candidates = sorted_pred.head(self.topk).index.tolist()

            # 在当前持仓中，不在 Top-K 的资产
            to_drop = [h for h in self.current_holdings if h not in top_k_candidates]

            # 限制最多 drop n_drop 个
            to_drop = to_drop[:self.n_drop]

            # 在 Top-K 中，不在当前持仓的资产
            to_add_candidates = [c for c in top_k_candidates if c not in current_set]

            # 添加相同数量
            to_add = to_add_candidates[:len(to_drop)]

            # 更新持仓
            selected = [h for h in self.current_holdings if h not in to_drop] + to_add

            # 如果还不够 topk，补充
            while len(selected) < self.topk and len(to_add_candidates) > len(to_add):
                next_add = to_add_candidates[len(to_add)]
                if next_add not in selected:
                    selected.append(next_add)
                to_add_candidates = to_add_candidates[1:]

        self.current_holdings = selected[:self.topk]

        # 分配权重
        if self.method == "equal":
            weight = 1.0 / len(self.current_holdings)
            weights = pd.Series(weight, index=self.current_holdings)
        elif self.method == "score":
            scores = predictions[self.current_holdings]
            scores = scores - scores.min() + 1e-6  # 确保正数
            weights = scores / scores.sum()
        else:
            weights = pd.Series(1.0 / len(self.current_holdings), index=self.current_holdings)

        return weights

    def reset(self):
        """重置策略状态"""
        self.current_holdings = []


# ============================================================
# 便捷函数
# ============================================================

def optimize_portfolio(
    returns: pd.DataFrame,
    method: str = "risk_parity",
    expected_returns: Optional[pd.Series] = None,
    **kwargs
) -> pd.Series:
    """
    优化投资组合权重 (便捷函数)

    Args:
        returns: 历史收益率
        method: 优化方法
        expected_returns: 预期收益率
        **kwargs: 其他参数

    Returns:
        权重
    """
    optimizer = PortfolioOptimizer(method=method, **kwargs)
    return optimizer.optimize(returns, expected_returns)


def calculate_portfolio_metrics(
    weights: pd.Series,
    returns: pd.DataFrame,
    cov_matrix: Optional[pd.DataFrame] = None,
    expected_returns: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    计算组合指标

    Args:
        weights: 权重
        returns: 历史收益率
        cov_matrix: 协方差矩阵
        expected_returns: 预期收益率

    Returns:
        指标字典
    """
    if cov_matrix is None:
        cov_matrix = returns.cov()

    if expected_returns is None:
        expected_returns = returns.mean()

    # 对齐资产
    common_assets = weights.index.intersection(cov_matrix.index)
    w = weights[common_assets].values
    cov = cov_matrix.loc[common_assets, common_assets].values
    mu = expected_returns[common_assets].values

    # 组合收益
    portfolio_return = np.dot(w, mu)

    # 组合波动率
    portfolio_vol = np.sqrt(np.dot(w, np.dot(cov, w)))

    # 夏普比率 (假设无风险利率为 0)
    sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

    # 最大权重
    max_weight = np.max(np.abs(w))

    # 有效资产数 (Herfindahl Index 的倒数)
    hhi = np.sum(w ** 2)
    effective_n = 1 / hhi if hhi > 0 else len(w)

    return {
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe_ratio': sharpe,
        'max_weight': max_weight,
        'effective_n': effective_n,
        'n_assets': len(w),
    }


# 导出
__all__ = [
    'OptMethod',
    'PortfolioOptimizer',
    'RiskModel',
    'TopkDropoutStrategy',
    'optimize_portfolio',
    'calculate_portfolio_metrics',
]
