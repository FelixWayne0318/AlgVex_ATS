"""
因子处理器 (Qlib 风格)

实现 Qlib 的处理器链模式:
- 学习处理器 (learn_processors): 用于训练阶段
- 推理处理器 (infer_processors): 用于推理阶段

用法:
    from algvex.core.factor.processor import ProcessorChain, ZScoreNormalizer, DropnaProcessor

    # 创建处理器链
    chain = ProcessorChain([
        DropnaProcessor(),
        ZScoreNormalizer(clip=3.0),
    ])

    # 学习阶段 (计算统计量)
    chain.fit(train_df)

    # 应用处理
    processed_df = chain.transform(df)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """
    处理器基类 (Qlib 风格)

    所有处理器必须实现:
    - fit(): 学习统计量
    - transform(): 应用变换

    可选实现:
    - is_for_infer(): 是否用于推理阶段
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._fitted = False

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """学习处理参数"""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用处理"""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """学习并应用"""
        self.fit(df)
        return self.transform(df)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def is_for_infer(self) -> bool:
        """
        是否用于推理阶段 (Qlib 风格)

        某些处理器仅在训练时使用，如:
        - DropnaLabel: 根据标签删除样本
        - 某些特征工程处理器

        Returns:
            True 如果可用于推理
        """
        return True

    def readonly(self) -> bool:
        """
        是否只读 (不修改原数据)

        用于优化，避免不必要的拷贝

        Returns:
            True 如果不修改输入数据
        """
        return False


class DropnaProcessor(BaseProcessor):
    """
    删除缺失值处理器

    Args:
        subset: 检查缺失值的列 (None=所有列)
        how: 'any' 或 'all'
    """

    def __init__(self, subset: List[str] = None, how: str = 'any'):
        super().__init__()
        self.subset = subset
        self.how = how

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)
        result = df.dropna(subset=self.subset, how=self.how)
        dropped = original_len - len(result)
        if dropped > 0:
            logger.debug(f"DropnaProcessor: dropped {dropped} rows ({dropped/original_len*100:.1f}%)")
        return result

    def readonly(self) -> bool:
        return True


class DropnaLabel(DropnaProcessor):
    """
    删除标签缺失的样本 (Qlib 风格)

    仅在训练时使用，推理时不应删除样本

    Args:
        label_col: 标签列名
    """

    def __init__(self, label_col: str = 'label'):
        super().__init__(subset=[label_col], how='any')
        self.label_col = label_col

    def is_for_infer(self) -> bool:
        """推理时不删除样本"""
        return False


class FillnaProcessor(BaseProcessor):
    """
    填充缺失值处理器

    Args:
        value: 填充值 (数值或 'mean', 'median', 'ffill', 'bfill')
        columns: 要填充的列 (None=所有列)
    """

    def __init__(self, value: Union[float, str] = 0, columns: List[str] = None):
        super().__init__()
        self.value = value
        self.columns = columns
        self._fill_values = {}

    def fit(self, df: pd.DataFrame):
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()

        if self.value == 'mean':
            self._fill_values = df[cols].mean().to_dict()
        elif self.value == 'median':
            self._fill_values = df[cols].median().to_dict()
        else:
            self._fill_values = {c: self.value for c in cols}

        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        if self.value in ['ffill', 'bfill']:
            result = result.fillna(method=self.value)
        else:
            for col, val in self._fill_values.items():
                if col in result.columns:
                    result[col] = result[col].fillna(val)

        return result


class ZScoreNormalizer(BaseProcessor):
    """
    Z-Score 标准化处理器 (Qlib 的 CSZScoreNorm)

    Args:
        clip: 截断范围 (默认 3.0, 即 [-3, 3])
        columns: 要标准化的列 (None=所有数值列)
    """

    def __init__(self, clip: float = 3.0, columns: List[str] = None):
        super().__init__()
        self.clip = clip
        self.columns = columns
        self._mean = {}
        self._std = {}

    def fit(self, df: pd.DataFrame):
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            self._mean[col] = df[col].mean()
            self._std[col] = df[col].std()
            # 避免除零
            if self._std[col] == 0 or np.isnan(self._std[col]):
                self._std[col] = 1.0

        self._fitted = True
        logger.debug(f"ZScoreNormalizer fitted on {len(cols)} columns")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col in self._mean.keys():
            if col in result.columns:
                # Z-Score
                result[col] = (result[col] - self._mean[col]) / self._std[col]
                # 截断
                if self.clip:
                    result[col] = result[col].clip(-self.clip, self.clip)

        return result


class MinMaxNormalizer(BaseProcessor):
    """
    Min-Max 标准化处理器

    Args:
        feature_range: 目标范围 (默认 (0, 1))
        columns: 要标准化的列
    """

    def __init__(self, feature_range: tuple = (0, 1), columns: List[str] = None):
        super().__init__()
        self.feature_range = feature_range
        self.columns = columns
        self._min = {}
        self._max = {}

    def fit(self, df: pd.DataFrame):
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            self._min[col] = df[col].min()
            self._max[col] = df[col].max()

        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        min_val, max_val = self.feature_range

        for col in self._min.keys():
            if col in result.columns:
                col_min = self._min[col]
                col_max = self._max[col]
                if col_max - col_min != 0:
                    result[col] = (result[col] - col_min) / (col_max - col_min)
                    result[col] = result[col] * (max_val - min_val) + min_val

        return result


class RobustScaler(BaseProcessor):
    """
    鲁棒标准化处理器 (使用中位数和四分位数)

    对异常值更鲁棒
    """

    def __init__(self, columns: List[str] = None):
        super().__init__()
        self.columns = columns
        self._median = {}
        self._iqr = {}

    def fit(self, df: pd.DataFrame):
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            self._median[col] = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            self._iqr[col] = q3 - q1
            if self._iqr[col] == 0:
                self._iqr[col] = 1.0

        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col in self._median.keys():
            if col in result.columns:
                result[col] = (result[col] - self._median[col]) / self._iqr[col]

        return result


class WinsorizeProcessor(BaseProcessor):
    """
    Winsorize 处理器 (缩尾处理)

    将极端值替换为指定分位数的值
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99, columns: List[str] = None):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.columns = columns
        self._bounds = {}

    def fit(self, df: pd.DataFrame):
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            self._bounds[col] = (
                df[col].quantile(self.lower),
                df[col].quantile(self.upper)
            )

        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col, (lower, upper) in self._bounds.items():
            if col in result.columns:
                result[col] = result[col].clip(lower, upper)

        return result


class LagProcessor(BaseProcessor):
    """
    滞后特征处理器

    创建滞后特征
    """

    def __init__(self, columns: List[str], lags: List[int] = [1, 2, 3]):
        super().__init__()
        self.columns = columns
        self.lags = lags

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col in self.columns:
            if col in result.columns:
                for lag in self.lags:
                    result[f"{col}_lag{lag}"] = result[col].shift(lag)

        return result


class DiffProcessor(BaseProcessor):
    """
    差分处理器

    创建差分特征
    """

    def __init__(self, columns: List[str], periods: List[int] = [1]):
        super().__init__()
        self.columns = columns
        self.periods = periods

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col in self.columns:
            if col in result.columns:
                for period in self.periods:
                    result[f"{col}_diff{period}"] = result[col].diff(period)

        return result


class RollingProcessor(BaseProcessor):
    """
    滚动窗口特征处理器

    创建滚动统计特征
    """

    def __init__(
        self,
        columns: List[str],
        windows: List[int] = [5, 10, 20],
        functions: List[str] = ['mean', 'std']
    ):
        super().__init__()
        self.columns = columns
        self.windows = windows
        self.functions = functions

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col in self.columns:
            if col in result.columns:
                for window in self.windows:
                    rolling = result[col].rolling(window)
                    for func in self.functions:
                        if func == 'mean':
                            result[f"{col}_ma{window}"] = rolling.mean()
                        elif func == 'std':
                            result[f"{col}_std{window}"] = rolling.std()
                        elif func == 'min':
                            result[f"{col}_min{window}"] = rolling.min()
                        elif func == 'max':
                            result[f"{col}_max{window}"] = rolling.max()
                        elif func == 'sum':
                            result[f"{col}_sum{window}"] = rolling.sum()

        return result


# ============================================================
# 跨截面处理器 (Qlib Cross-Sectional Processors)
# ============================================================

class CSZScoreNorm(BaseProcessor):
    """
    跨截面 Z-Score 标准化 (Qlib 原版)

    按时间分组，对每个时间点的所有样本进行 Z-Score 标准化

    Args:
        columns: 要标准化的列 (None=所有数值列)
        method: 'zscore' 或 'robust'
    """

    def __init__(self, columns: List[str] = None, method: str = 'zscore'):
        super().__init__()
        self.columns = columns
        self.method = method

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in result.columns:
                continue

            if self.method == 'robust':
                # 使用中位数和 MAD
                def robust_zscore(x):
                    median = x.median()
                    mad = (x - median).abs().median()
                    return (x - median) / (mad * 1.4826 + 1e-12)
                result[col] = result.groupby(level=0)[col].transform(robust_zscore)
            else:
                # 标准 z-score
                def zscore(x):
                    return (x - x.mean()) / (x.std() + 1e-12)
                result[col] = result.groupby(level=0)[col].transform(zscore)

        return result


class CSRankNorm(BaseProcessor):
    """
    跨截面排名标准化 (Qlib 原版)

    将每个时间点的样本转换为排名百分位，然后映射到 [-1.73, 1.73]

    公式: (rank(pct=True) - 0.5) * 3.46

    Args:
        columns: 要标准化的列
    """

    def __init__(self, columns: List[str] = None):
        super().__init__()
        self.columns = columns

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in result.columns:
                continue

            def rank_norm(x):
                return (x.rank(pct=True) - 0.5) * 3.46

            result[col] = result.groupby(level=0)[col].transform(rank_norm)

        return result


class CSFillna(BaseProcessor):
    """
    跨截面缺失值填充 (Qlib 原版)

    用同一时间点的均值填充缺失值

    Args:
        columns: 要填充的列
    """

    def __init__(self, columns: List[str] = None):
        super().__init__()
        self.columns = columns

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in result.columns:
                continue

            def fill_with_mean(x):
                return x.fillna(x.mean())

            result[col] = result.groupby(level=0)[col].transform(fill_with_mean)

        return result


class TanhProcess(BaseProcessor):
    """
    Tanh 去噪处理器 (Qlib 原版)

    对特征应用 tanh 变换，压缩极端值

    不处理 label 列
    """

    def __init__(self, label_col: str = 'label'):
        super().__init__()
        self.label_col = label_col

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()

        for col in result.columns:
            if col == self.label_col:
                continue
            if result[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                result[col] = np.tanh(result[col])

        return result


class ProcessInf(BaseProcessor):
    """
    无穷值处理器 (Qlib 原版)

    将 inf/-inf 替换为同一时间点的均值
    """

    def __init__(self, columns: List[str] = None):
        super().__init__()
        self.columns = columns

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        cols = self.columns or df.select_dtypes(include=[np.number]).columns.tolist()

        for col in cols:
            if col not in result.columns:
                continue

            # 替换 inf 为 NaN
            result[col] = result[col].replace([np.inf, -np.inf], np.nan)

            # 用同一时间点的均值填充
            def fill_with_mean(x):
                return x.fillna(x.mean())

            result[col] = result.groupby(level=0)[col].transform(fill_with_mean)

        return result


class FilterCol(BaseProcessor):
    """
    列过滤器 (Qlib 原版)

    只保留指定的列

    Args:
        columns: 要保留的列列表
    """

    def __init__(self, columns: List[str]):
        super().__init__()
        self.columns = columns

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        existing_cols = [c for c in self.columns if c in df.columns]
        return df[existing_cols].copy()

    def readonly(self) -> bool:
        return True


class DropCol(BaseProcessor):
    """
    列删除器 (Qlib 原版)

    删除指定的列

    Args:
        columns: 要删除的列列表
    """

    def __init__(self, columns: List[str]):
        super().__init__()
        self.columns = columns

    def fit(self, df: pd.DataFrame):
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.columns if c in df.columns]
        return df.drop(columns=cols_to_drop).copy()

    def readonly(self) -> bool:
        return True


class ProcessorChain:
    """
    处理器链 (Qlib 风格)

    按顺序应用多个处理器

    特性:
    - 支持 fit/transform 分离
    - 支持 for_inference 模式 (跳过非推理处理器)
    - 支持序列化
    """

    def __init__(self, processors: List[BaseProcessor] = None):
        self.processors = processors or []
        self._fitted = False

    def add(self, processor: BaseProcessor):
        """添加处理器"""
        self.processors.append(processor)
        self._fitted = False
        return self

    def fit(self, df: pd.DataFrame):
        """学习所有处理器"""
        current_df = df.copy()

        for processor in self.processors:
            processor.fit(current_df)
            current_df = processor.transform(current_df)

        self._fitted = True
        logger.info(f"ProcessorChain fitted with {len(self.processors)} processors")

    def transform(self, df: pd.DataFrame, for_inference: bool = False) -> pd.DataFrame:
        """
        应用所有处理器

        Args:
            df: 输入数据
            for_inference: 是否为推理模式 (跳过非推理处理器)

        Returns:
            处理后的数据
        """
        result = df.copy()

        for processor in self.processors:
            # 推理模式下跳过非推理处理器
            if for_inference and not processor.is_for_infer():
                logger.debug(f"Skipping {type(processor).__name__} (not for inference)")
                continue

            if not processor.is_fitted:
                raise RuntimeError(f"Processor {type(processor).__name__} not fitted")
            result = processor.transform(result)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """学习并应用"""
        self.fit(df)
        return self.transform(df)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_learn_processors(self) -> List[BaseProcessor]:
        """获取学习阶段的处理器"""
        return self.processors

    def get_infer_processors(self) -> List[BaseProcessor]:
        """获取推理阶段的处理器"""
        return [p for p in self.processors if p.is_for_infer()]


# ============================================================
# 预定义处理器链
# ============================================================

def get_default_learn_processors() -> ProcessorChain:
    """获取默认的学习处理器链"""
    return ProcessorChain([
        DropnaProcessor(),
        WinsorizeProcessor(lower=0.01, upper=0.99),
        ZScoreNormalizer(clip=3.0),
    ])


def get_default_infer_processors() -> ProcessorChain:
    """获取默认的推理处理器链"""
    return ProcessorChain([
        DropnaProcessor(),
    ])


def get_crypto_processors() -> ProcessorChain:
    """获取加密货币专用处理器链"""
    return ProcessorChain([
        DropnaProcessor(),
        WinsorizeProcessor(lower=0.005, upper=0.995),  # 更激进的缩尾
        RobustScaler(),  # 对加密货币异常值更鲁棒
    ])
