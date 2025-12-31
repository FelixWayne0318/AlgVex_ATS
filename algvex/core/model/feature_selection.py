"""
AlgVex 特征选择

设计文档参考: Section 4.3 feature_selection

特征选择方法:
- importance: 基于模型特征重要性
- correlation: 基于相关性过滤
- forward: 前向选择
- recursive: 递归特征消除

使用示例:
    selector = FeatureSelector(method="importance", top_k=50)
    selected = selector.select(X, y, model_config)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class FeatureImportance:
    """
    特征重要性

    Attributes:
        feature_name: 特征名称
        importance: 重要性分数
        rank: 排名
    """
    feature_name: str
    importance: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "feature_name": self.feature_name,
            "importance": self.importance,
            "rank": self.rank,
        }


class FeatureSelector:
    """
    特征选择器

    使用示例:
        selector = FeatureSelector(method="importance", top_k=50)

        # 使用模型重要性选择
        selected = selector.select(X, y, model_config)

        # 获取选中的特征
        X_selected = X[selected]
    """

    def __init__(
        self,
        method: str = "importance",
        top_k: int = 50,
        correlation_threshold: float = 0.7,
    ):
        """
        初始化特征选择器

        Args:
            method: 选择方法 (importance, correlation, forward)
            top_k: 选择前 k 个特征
            correlation_threshold: 相关性阈值 (用于过滤高度相关的特征)
        """
        self.method = method
        self.top_k = top_k
        self.correlation_threshold = correlation_threshold
        self.selected_features: Optional[List[str]] = None
        self.feature_importance: Optional[List[FeatureImportance]] = None

    def select(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        model_config: Optional[Any] = None,
    ) -> List[str]:
        """
        选择特征

        Args:
            X: 特征数据
            y: 标签
            model_config: 模型配置 (用于 importance 方法)

        Returns:
            选中的特征名称列表
        """
        if isinstance(X, np.ndarray):
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=feature_names)
        else:
            feature_names = X.columns.tolist()

        if isinstance(y, pd.Series):
            y = y.values

        if self.method == "importance":
            selected = self._select_by_importance(X, y, model_config)
        elif self.method == "correlation":
            selected = self._select_by_correlation(X, y)
        elif self.method == "forward":
            selected = self._select_forward(X, y, model_config)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        self.selected_features = selected
        return selected

    def _select_by_importance(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_config: Optional[Any] = None,
    ) -> List[str]:
        """基于模型重要性选择"""
        from .trainer import ModelConfig, ModelTrainer

        if model_config is None:
            model_config = ModelConfig(model_type="lightgbm")

        # 训练模型
        trainer = ModelTrainer(model_config)
        result = trainer.train(X, y)

        if result.feature_importance is None:
            # 回退到相关性
            return self._select_by_correlation(X, y)

        # 排序
        sorted_features = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # 记录特征重要性
        self.feature_importance = [
            FeatureImportance(
                feature_name=name,
                importance=importance,
                rank=i + 1,
            )
            for i, (name, importance) in enumerate(sorted_features)
        ]

        # 选择 top_k
        selected = [name for name, _ in sorted_features[:self.top_k]]

        return selected

    def _select_by_correlation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[str]:
        """基于相关性选择"""
        feature_names = X.columns.tolist()

        # 1. 计算与目标的相关性
        correlations = {}
        for col in feature_names:
            corr = np.corrcoef(X[col].values, y)[0, 1]
            if not np.isnan(corr):
                correlations[col] = abs(corr)
            else:
                correlations[col] = 0

        # 排序
        sorted_features = sorted(
            correlations.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # 2. 过滤高度相关的特征
        selected = []
        for name, _ in sorted_features:
            if len(selected) >= self.top_k:
                break

            # 检查与已选特征的相关性
            is_redundant = False
            for selected_name in selected:
                corr = np.corrcoef(X[name].values, X[selected_name].values)[0, 1]
                if abs(corr) > self.correlation_threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                selected.append(name)

        # 记录特征重要性
        self.feature_importance = [
            FeatureImportance(
                feature_name=name,
                importance=correlations[name],
                rank=i + 1,
            )
            for i, (name, _) in enumerate(sorted_features)
        ]

        return selected

    def _select_forward(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        model_config: Optional[Any] = None,
    ) -> List[str]:
        """前向选择"""
        from .trainer import ModelConfig, ModelTrainer

        if model_config is None:
            model_config = ModelConfig(model_type="ridge")

        feature_names = X.columns.tolist()
        selected = []
        remaining = feature_names.copy()

        best_score = -np.inf

        while len(selected) < self.top_k and remaining:
            scores = []

            for feature in remaining:
                candidate = selected + [feature]
                X_subset = X[candidate].values

                # 使用交叉验证评估
                trainer = ModelTrainer(model_config)
                try:
                    result = trainer.train(X_subset, y)
                    score = result.val_metrics.get("ic", result.train_metrics.get("ic", 0))
                except Exception:
                    score = -np.inf

                scores.append((feature, score))

            # 选择最佳特征
            best_feature, best_feature_score = max(scores, key=lambda x: x[1])

            if best_feature_score > best_score:
                selected.append(best_feature)
                remaining.remove(best_feature)
                best_score = best_feature_score
            else:
                break

        return selected

    def get_importance_report(self) -> str:
        """生成特征重要性报告"""
        if self.feature_importance is None:
            return "No feature importance available"

        lines = [
            "=" * 50,
            "Feature Importance Report",
            "=" * 50,
            "",
            f"{'Rank':<6}{'Feature':<30}{'Importance':<15}",
            "-" * 50,
        ]

        for fi in self.feature_importance[:min(20, len(self.feature_importance))]:
            lines.append(f"{fi.rank:<6}{fi.feature_name:<30}{fi.importance:<15.6f}")

        if len(self.feature_importance) > 20:
            lines.append(f"... and {len(self.feature_importance) - 20} more features")

        return "\n".join(lines)


def compute_feature_correlations(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算特征相关性矩阵

    Args:
        X: 特征数据

    Returns:
        相关性矩阵
    """
    return X.corr()


def find_redundant_features(
    X: pd.DataFrame,
    threshold: float = 0.9,
) -> List[Tuple[str, str, float]]:
    """
    找出高度相关的特征对

    Args:
        X: 特征数据
        threshold: 相关性阈值

    Returns:
        [(feature1, feature2, correlation), ...]
    """
    corr_matrix = X.corr()
    feature_names = X.columns.tolist()

    redundant = []
    for i, f1 in enumerate(feature_names):
        for f2 in feature_names[i + 1:]:
            corr = corr_matrix.loc[f1, f2]
            if abs(corr) > threshold:
                redundant.append((f1, f2, corr))

    return sorted(redundant, key=lambda x: abs(x[2]), reverse=True)
