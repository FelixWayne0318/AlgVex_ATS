"""
统一特征计算模块

训练和实盘共用此模块，确保特征计算完全一致。

用法:
    from unified_features import compute_unified_features, FEATURE_COLUMNS
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

# 固定的特征列顺序 (训练和预测必须一致)
FEATURE_COLUMNS = [
    # KBAR 类 (9)
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
    # ROC 类 (5)
    "ROC5", "ROC10", "ROC20", "ROC30", "ROC60",
    # MA 类 (5)
    "MA5", "MA10", "MA20", "MA30", "MA60",
    # STD 类 (5)
    "STD5", "STD10", "STD20", "STD30", "STD60",
    # MAX 类 (5)
    "MAX5", "MAX10", "MAX20", "MAX30", "MAX60",
    # MIN 类 (5)
    "MIN5", "MIN10", "MIN20", "MIN30", "MIN60",
    # QTLU 类 (5)
    "QTLU5", "QTLU10", "QTLU20", "QTLU30", "QTLU60",
    # QTLD 类 (5)
    "QTLD5", "QTLD10", "QTLD20", "QTLD30", "QTLD60",
    # RSV 类 (5)
    "RSV5", "RSV10", "RSV20", "RSV30", "RSV60",
    # CORR 类 (5)
    "CORR5", "CORR10", "CORR20", "CORR30", "CORR60",
    # CORD 类 (5)
    "CORD5", "CORD10", "CORD20", "CORD30", "CORD60",
]


def compute_unified_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算统一特征集

    Parameters
    ----------
    df : pd.DataFrame
        必须包含: open, high, low, close, volume 列

    Returns
    -------
    pd.DataFrame
        特征矩阵，列顺序固定为 FEATURE_COLUMNS
    """
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    features = pd.DataFrame(index=df.index)

    # KBAR 类因子
    features["KMID"] = (close - open_) / open_
    features["KLEN"] = (high - low) / open_
    features["KMID2"] = (close - open_) / (high - low + 1e-12)
    features["KUP"] = (high - np.maximum(open_, close)) / open_
    features["KUP2"] = (high - np.maximum(open_, close)) / (high - low + 1e-12)
    features["KLOW"] = (np.minimum(open_, close) - low) / open_
    features["KLOW2"] = (np.minimum(open_, close) - low) / (high - low + 1e-12)
    features["KSFT"] = (2 * close - high - low) / open_
    features["KSFT2"] = (2 * close - high - low) / (high - low + 1e-12)

    # 多周期因子
    for d in [5, 10, 20, 30, 60]:
        features[f"ROC{d}"] = close / close.shift(d) - 1
        ma = close.rolling(d).mean()
        features[f"MA{d}"] = close / ma - 1
        features[f"STD{d}"] = close.rolling(d).std() / close
        features[f"MAX{d}"] = close / high.rolling(d).max() - 1
        features[f"MIN{d}"] = close / low.rolling(d).min() - 1
        features[f"QTLU{d}"] = close / close.rolling(d).quantile(0.8) - 1
        features[f"QTLD{d}"] = close / close.rolling(d).quantile(0.2) - 1
        hh = high.rolling(d).max()
        ll = low.rolling(d).min()
        features[f"RSV{d}"] = (close - ll) / (hh - ll + 1e-12)
        features[f"CORR{d}"] = close.rolling(d).corr(volume)
        ret = close.pct_change()
        features[f"CORD{d}"] = ret.rolling(d).corr(volume.pct_change())

    # 确保列顺序一致
    return features[FEATURE_COLUMNS]


def compute_label(df: pd.DataFrame) -> pd.Series:
    """
    计算标签: t+1 时刻相对于当前的收益率

    与 Qlib Alpha158 不同，我们使用 t+1 而非 t+2
    因为加密货币没有 T+1 交易限制
    """
    close = df["close"].astype(float)
    return close.shift(-1) / close - 1


class FeatureNormalizer:
    """
    特征归一化器

    训练时: fit_transform() 计算并保存均值/标准差
    预测时: transform() 使用保存的参数
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.feature_columns = None  # 训练时的特征列顺序
        self.fitted = False

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        训练时使用: 计算统计量并归一化

        ⚠️ 防泄漏注意: 仅用训练集调用此方法!
        验证集/测试集应使用 transform()，避免未来数据泄漏到统计量中。
        """
        self.mean = features.mean()
        self.std = features.std() + 1e-8  # 避免除零
        self.fitted = True
        self.feature_columns = list(features.columns)  # 记录训练时的特征列
        return (features - self.mean) / self.std

    def transform(self, features: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        """
        预测时使用: 使用保存的统计量归一化

        Parameters
        ----------
        features : pd.DataFrame
            输入特征
        strict : bool
            严格模式 (实盘/回测必须为 True)
            - True: 缺失列或 NaN 直接抛异常
            - False: 填充 NaN + 告警 (仅用于调试)

        包含特征对齐防呆机制:
        - 缺失列: strict=True 抛异常, strict=False 填充 NaN
        - 多余列: 丢弃 + 告警
        - 顺序: 强制重排为训练时顺序
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit_transform first.")

        # 特征对齐防呆机制
        expected_cols = set(self.feature_columns)
        actual_cols = set(features.columns)

        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols

        if missing_cols:
            if strict:
                raise ValueError(f"❌ 严格模式: 缺失特征列 {missing_cols}")
            else:
                import warnings
                warnings.warn(f"⚠️ 缺失特征列 (将填充 NaN): {missing_cols}")
                for col in missing_cols:
                    features[col] = np.nan

        if extra_cols:
            import warnings
            warnings.warn(f"⚠️ 多余特征列 (将丢弃): {extra_cols}")
            features = features.drop(columns=list(extra_cols))

        # 强制重排为训练时顺序
        features = features[self.feature_columns]

        # 严格模式检查 NaN
        if strict and features.isna().any().any():
            nan_cols = features.columns[features.isna().any()].tolist()
            raise ValueError(f"❌ 严格模式: 特征包含 NaN {nan_cols}")

        return (features - self.mean) / self.std

    def save(self, path: str):
        """保存归一化参数 (含特征列顺序)"""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "mean": self.mean,
                "std": self.std,
                "feature_columns": self.feature_columns,
            }, f)

    def load(self, path: str):
        """加载归一化参数 (含特征列顺序)"""
        import pickle
        with open(path, "rb") as f:
            params = pickle.load(f)
        self.mean = params["mean"]
        self.std = params["std"]
        self.feature_columns = params.get("feature_columns", list(self.mean.index))
        self.fitted = True
