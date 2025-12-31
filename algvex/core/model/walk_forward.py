"""
AlgVex 滚动验证 (Walk-Forward Validation)

设计文档参考: Section 12.1 walk_forward.py

滚动验证确保:
- 训练集和测试集严格按时间分离
- 避免未来数据泄露
- 模拟真实交易环境

使用示例:
    config = WalkForwardConfig(
        train_window=252,
        test_window=21,
        step_size=21,
    )

    validator = WalkForwardValidator(config)
    results = validator.validate(data, labels, model_config)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .trainer import ModelConfig, ModelTrainer, TrainingResult


@dataclass
class WalkForwardConfig:
    """
    滚动验证配置

    Attributes:
        train_window: 训练窗口大小 (bars)
        test_window: 测试窗口大小 (bars)
        step_size: 步进大小 (bars)
        min_train_samples: 最小训练样本数
        purge_gap: 训练集和测试集之间的间隔 (防止数据泄露)
    """
    train_window: int = 252 * 288  # 252天 (5分钟频率)
    test_window: int = 21 * 288     # 21天
    step_size: int = 21 * 288       # 每21天滚动
    min_train_samples: int = 1000
    purge_gap: int = 1              # 1个 bar 间隔

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "train_window": self.train_window,
            "test_window": self.test_window,
            "step_size": self.step_size,
            "min_train_samples": self.min_train_samples,
            "purge_gap": self.purge_gap,
        }


@dataclass
class FoldResult:
    """
    单个 Fold 的结果

    Attributes:
        fold_idx: Fold 索引
        train_start_idx: 训练起始索引
        train_end_idx: 训练结束索引
        test_start_idx: 测试起始索引
        test_end_idx: 测试结束索引
        train_metrics: 训练集指标
        test_metrics: 测试集指标
        predictions: 测试集预测
        feature_importance: 特征重要性
    """
    fold_idx: int
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_time: float = 0.0


@dataclass
class WalkForwardResult:
    """
    滚动验证结果

    Attributes:
        folds: 各 Fold 结果
        aggregate_metrics: 聚合指标
        total_time: 总训练时间
        config: 验证配置
    """
    folds: List[FoldResult]
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    config: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "num_folds": len(self.folds),
            "aggregate_metrics": self.aggregate_metrics,
            "total_time": self.total_time,
            "config": self.config,
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "train_range": (f.train_start_idx, f.train_end_idx),
                    "test_range": (f.test_start_idx, f.test_end_idx),
                    "train_metrics": f.train_metrics,
                    "test_metrics": f.test_metrics,
                }
                for f in self.folds
            ],
        }

    def summary(self) -> str:
        """生成摘要"""
        lines = [
            "=" * 50,
            "Walk-Forward Validation Results",
            "=" * 50,
            f"Number of Folds: {len(self.folds)}",
            f"Total Time: {self.total_time:.2f}s",
            "",
            "Aggregate Metrics:",
        ]

        for metric, value in self.aggregate_metrics.items():
            lines.append(f"  {metric}: {value:.4f}")

        return "\n".join(lines)


class WalkForwardValidator:
    """
    滚动验证器

    使用示例:
        config = WalkForwardConfig(
            train_window=252 * 288,
            test_window=21 * 288,
        )

        validator = WalkForwardValidator(config)

        results = validator.validate(
            X=features_df,
            y=labels_series,
            model_config=ModelConfig(model_type="lightgbm"),
        )

        print(results.summary())
    """

    def __init__(self, config: WalkForwardConfig):
        """
        初始化验证器

        Args:
            config: 验证配置
        """
        self.config = config

    def generate_folds(
        self,
        n_samples: int,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        生成滚动验证的 folds

        Args:
            n_samples: 样本总数

        Returns:
            [(train_range, test_range), ...]
        """
        folds = []

        train_start = 0
        while True:
            train_end = train_start + self.config.train_window

            # 测试集起点 (加上 purge gap)
            test_start = train_end + self.config.purge_gap
            test_end = test_start + self.config.test_window

            # 检查是否超出范围
            if test_end > n_samples:
                break

            # 检查最小训练样本
            if train_end - train_start < self.config.min_train_samples:
                train_start += self.config.step_size
                continue

            folds.append((
                (train_start, train_end),
                (test_start, test_end),
            ))

            # 滚动
            train_start += self.config.step_size

        return folds

    def validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        model_config: ModelConfig,
        verbose: bool = True,
    ) -> WalkForwardResult:
        """
        执行滚动验证

        Args:
            X: 特征
            y: 标签
            model_config: 模型配置
            verbose: 是否打印进度

        Returns:
            WalkForwardResult
        """
        import time
        total_start = time.time()

        # 转换为 numpy
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = None

        if isinstance(y, pd.Series):
            y = y.values

        n_samples = len(X)
        folds = self.generate_folds(n_samples)

        if len(folds) == 0:
            raise ValueError("No valid folds generated. Check your data size and config.")

        if verbose:
            print(f"Generated {len(folds)} folds for walk-forward validation")

        fold_results = []
        all_test_metrics = []

        for fold_idx, ((train_start, train_end), (test_start, test_end)) in enumerate(folds):
            if verbose:
                print(f"Fold {fold_idx + 1}/{len(folds)}: "
                      f"Train [{train_start}:{train_end}], Test [{test_start}:{test_end}]")

            # 提取数据
            X_train = X[train_start:train_end]
            y_train = y[train_start:train_end]
            X_test = X[test_start:test_end]
            y_test = y[test_start:test_end]

            # 训练
            trainer = ModelTrainer(model_config)
            result = trainer.train(X_train, y_train)

            # 预测
            predictions = trainer.predict(X_test)

            # 计算测试指标
            test_metrics = self._calculate_metrics(y_test, predictions)

            fold_result = FoldResult(
                fold_idx=fold_idx,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_start_idx=test_start,
                test_end_idx=test_end,
                train_metrics=result.train_metrics,
                test_metrics=test_metrics,
                predictions=predictions,
                feature_importance=result.feature_importance,
                training_time=result.training_time,
            )

            fold_results.append(fold_result)
            all_test_metrics.append(test_metrics)

        # 聚合指标
        aggregate_metrics = self._aggregate_metrics(all_test_metrics)

        total_time = time.time() - total_start

        return WalkForwardResult(
            folds=fold_results,
            aggregate_metrics=aggregate_metrics,
            total_time=total_time,
            config=self.config.to_dict(),
        )

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """计算评估指标"""
        # MSE and RMSE (numpy implementation)
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = np.sqrt(mse)

        # R2 score (numpy implementation)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # IC
        ic = np.corrcoef(y_true, y_pred)[0, 1]

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "ic": float(ic) if not np.isnan(ic) else 0.0,
        }

    def _aggregate_metrics(
        self,
        fold_metrics: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """聚合各 fold 的指标"""
        if not fold_metrics:
            return {}

        metric_names = fold_metrics[0].keys()
        aggregate = {}

        for name in metric_names:
            values = [m[name] for m in fold_metrics]
            aggregate[f"{name}_mean"] = float(np.mean(values))
            aggregate[f"{name}_std"] = float(np.std(values))

        # 计算 IR (IC / IC_std)
        ic_values = [m["ic"] for m in fold_metrics]
        ic_mean = np.mean(ic_values)
        ic_std = np.std(ic_values)
        aggregate["ir"] = float(ic_mean / ic_std) if ic_std > 0 else 0.0

        return aggregate


def create_expanding_folds(
    n_samples: int,
    initial_train_size: int,
    test_window: int,
    step_size: int,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    创建扩展窗口的 folds (训练集不断增大)

    Args:
        n_samples: 样本总数
        initial_train_size: 初始训练集大小
        test_window: 测试窗口大小
        step_size: 步进大小

    Returns:
        [(train_range, test_range), ...]
    """
    folds = []

    train_end = initial_train_size
    while True:
        test_start = train_end
        test_end = test_start + test_window

        if test_end > n_samples:
            break

        folds.append((
            (0, train_end),  # 训练集从头开始
            (test_start, test_end),
        ))

        train_end += step_size

    return folds
