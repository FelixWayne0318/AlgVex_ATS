"""
AlgVex 样本重新加权器 (Qlib 风格)

实现 Qlib 的 Reweighter 模式:
- 时间衰减加权 (近期数据更重要)
- 波动率加权 (高波动时期降权)
- 自定义加权

用法:
    from algvex.core.data.reweighter import TimeDecayReweighter

    # 创建重新加权器
    reweighter = TimeDecayReweighter(decay_rate=0.99)

    # 计算权重
    weights = reweighter.reweight(df)

    # 在模型训练中使用
    model.fit(X, y, sample_weight=weights)
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd
import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class Reweighter(ABC):
    """
    重新加权器基类 (Qlib 风格)

    所有重新加权器必须实现:
    - reweight(): 计算样本权重
    """

    @abstractmethod
    def reweight(self, df: pd.DataFrame) -> np.ndarray:
        """
        计算样本权重

        Args:
            df: 数据 DataFrame (必须有 DatetimeIndex)

        Returns:
            np.ndarray: 样本权重数组, 长度与 df 相同
        """
        pass


class TimeDecayReweighter(Reweighter):
    """
    时间衰减重新加权器

    给近期数据更高的权重，用于:
    - 适应市场regime变化
    - 减少过时数据的影响
    - 处理概念漂移

    权重计算:
        w[i] = decay_rate^(n-1-i)
        归一化: w = w / sum(w) * n

    Args:
        decay_rate: 衰减率 (0-1), 越小衰减越快
            - 0.99: 轻微衰减, 1000天前的数据权重约为 0.00004
            - 0.95: 中等衰减, 100天前的数据权重约为 0.006
            - 0.90: 强衰减, 50天前的数据权重约为 0.005
        min_weight: 最小权重 (防止过小)
    """

    def __init__(self, decay_rate: float = 0.99, min_weight: float = 0.01):
        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be in (0, 1]")

        self.decay_rate = decay_rate
        self.min_weight = min_weight

    def reweight(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if n == 0:
            return np.array([])

        # 计算指数衰减权重 (最新的数据权重最大)
        # w[i] = decay_rate^(n-1-i), 所以 w[n-1]=1, w[0]=decay_rate^(n-1)
        weights = self.decay_rate ** np.arange(n - 1, -1, -1)

        # 应用最小权重
        weights = np.maximum(weights, self.min_weight)

        # 归一化: 使权重总和等于样本数 (保持与无权重时的scale一致)
        weights = weights / weights.sum() * n

        return weights


class VolatilityReweighter(Reweighter):
    """
    波动率重新加权器

    在高波动时期降低权重，用于:
    - 减少极端市场条件的影响
    - 使模型更稳健
    - 降低噪音影响

    权重计算:
        vol = rolling_std(returns, window)
        w = 1 / (1 + scale * vol)
        归一化: w = w / sum(w) * n

    Args:
        return_col: 收益率列名
        window: 波动率计算窗口
        scale: 波动率影响系数
    """

    def __init__(
        self,
        return_col: str = 'label',
        window: int = 20,
        scale: float = 10.0,
    ):
        self.return_col = return_col
        self.window = window
        self.scale = scale

    def reweight(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if n == 0:
            return np.array([])

        # 计算滚动波动率
        if self.return_col in df.columns:
            returns = df[self.return_col]
        else:
            # 假设有 close 列
            if 'close' in df.columns:
                returns = df['close'].pct_change()
            else:
                logger.warning(f"Cannot find return column, using uniform weights")
                return np.ones(n)

        vol = returns.rolling(self.window, min_periods=1).std().fillna(0)

        # 计算权重 (高波动 -> 低权重)
        weights = 1.0 / (1.0 + self.scale * vol.values)

        # 归一化
        weights = weights / weights.sum() * n

        return weights


class CombinedReweighter(Reweighter):
    """
    组合重新加权器

    组合多个重新加权器的效果

    Args:
        reweighters: 重新加权器列表
        method: 组合方法 ('multiply', 'average')
    """

    def __init__(
        self,
        reweighters: list,
        method: str = 'multiply',
    ):
        self.reweighters = reweighters
        self.method = method

    def reweight(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        if n == 0:
            return np.array([])

        # 收集所有权重
        all_weights = [rw.reweight(df) for rw in self.reweighters]

        if self.method == 'multiply':
            # 乘积 (更激进)
            weights = np.prod(all_weights, axis=0)
        elif self.method == 'average':
            # 平均 (更温和)
            weights = np.mean(all_weights, axis=0)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # 归一化
        weights = weights / weights.sum() * n

        return weights


class CustomReweighter(Reweighter):
    """
    自定义重新加权器

    使用用户提供的函数计算权重

    Args:
        weight_func: 权重计算函数, 签名: (df) -> np.ndarray
    """

    def __init__(self, weight_func):
        self.weight_func = weight_func

    def reweight(self, df: pd.DataFrame) -> np.ndarray:
        weights = self.weight_func(df)

        # 确保是 numpy 数组
        if isinstance(weights, pd.Series):
            weights = weights.values

        # 归一化
        n = len(df)
        weights = weights / weights.sum() * n

        return weights


# ============================================================
# 便捷函数
# ============================================================

def get_default_reweighter() -> Reweighter:
    """获取默认的重新加权器 (时间衰减)"""
    return TimeDecayReweighter(decay_rate=0.995)


def get_crypto_reweighter() -> Reweighter:
    """获取加密货币专用重新加权器 (时间衰减 + 波动率)"""
    return CombinedReweighter([
        TimeDecayReweighter(decay_rate=0.99),
        VolatilityReweighter(scale=5.0),
    ], method='multiply')
