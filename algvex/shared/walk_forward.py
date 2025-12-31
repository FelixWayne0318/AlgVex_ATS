"""
AlgVex Walk-Forward 验证器 (P0-4)

功能:
- Walk-Forward 时序验证
- 禁止随机切分时序数据
- 滚动窗口训练/测试
- 过拟合检测

关键原则:
- 时序数据不能随机切分
- 必须使用 Walk-Forward 验证
- 训练集和测试集夏普比差距不超过 0.3
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RandomSplitForbiddenError(Exception):
    """禁止随机切分错误"""
    pass


@dataclass
class WalkForwardFold:
    """Walk-Forward 折叠"""
    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    test_samples: int


@dataclass
class WalkForwardResult:
    """Walk-Forward 验证结果"""
    fold_index: int
    train_period: str
    test_period: str
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    is_overfitting: bool = False
    overfitting_score: float = 0.0


@dataclass
class WalkForwardReport:
    """Walk-Forward 验证报告"""
    total_folds: int
    results: List[WalkForwardResult]
    avg_train_sharpe: float
    avg_test_sharpe: float
    sharpe_gap: float
    overfitting_ratio: float
    is_valid: bool
    failure_reasons: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class WalkForwardValidator:
    """
    Walk-Forward 验证器 - 防止过拟合

    使用方法:
        validator = WalkForwardValidator(train_months=12, test_months=3)

        # 创建折叠
        folds = validator.create_folds(data)

        # 验证模型
        report = validator.validate_model(model, data)
    """

    # 默认配置
    DEFAULT_TRAIN_MONTHS = 12
    DEFAULT_TEST_MONTHS = 3
    DEFAULT_MIN_TRAIN_SAMPLES = 1000
    DEFAULT_STEP_MONTHS = 3

    def __init__(
        self,
        train_months: int = DEFAULT_TRAIN_MONTHS,
        test_months: int = DEFAULT_TEST_MONTHS,
        min_train_samples: int = DEFAULT_MIN_TRAIN_SAMPLES,
        step_months: Optional[int] = None,
        purge_days: int = 0,  # 训练和测试之间的间隔天数
    ):
        """
        初始化 Walk-Forward 验证器

        Args:
            train_months: 训练窗口长度 (月)
            test_months: 测试窗口长度 (月)
            min_train_samples: 最小训练样本数
            step_months: 滚动步长 (月)，默认等于 test_months
            purge_days: 训练和测试之间的间隔天数
        """
        self.train_months = train_months
        self.test_months = test_months
        self.min_train_samples = min_train_samples
        self.step_months = step_months or test_months
        self.purge_days = purge_days

        logger.info(
            f"WalkForwardValidator initialized: "
            f"train={train_months}m, test={test_months}m, step={self.step_months}m"
        )

    def create_folds(
        self,
        data: pd.DataFrame,
        time_column: Optional[str] = None,
    ) -> List[WalkForwardFold]:
        """
        创建 Walk-Forward 折叠 - 严禁随机切分

        Args:
            data: 时序数据 (index 为时间戳或指定时间列)
            time_column: 时间列名 (如果不使用 index)

        Returns:
            折叠列表
        """
        # 获取时间索引
        if time_column:
            times = pd.to_datetime(data[time_column])
        else:
            times = pd.to_datetime(data.index)

        start_date = times.min()
        end_date = times.max()

        folds = []
        fold_index = 0
        current_train_start = start_date

        while True:
            # 计算训练结束时间
            train_end = current_train_start + pd.DateOffset(months=self.train_months)

            # 计算测试时间范围 (考虑 purge)
            test_start = train_end + pd.DateOffset(days=self.purge_days)
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # 检查是否超出数据范围
            if test_end > end_date:
                break

            # 获取训练和测试数据
            if time_column:
                train_mask = (times >= current_train_start) & (times < train_end)
                test_mask = (times >= test_start) & (times < test_end)
            else:
                train_mask = (times >= current_train_start) & (times < train_end)
                test_mask = (times >= test_start) & (times < test_end)

            train_samples = train_mask.sum()
            test_samples = test_mask.sum()

            # 验证训练样本量
            if train_samples >= self.min_train_samples:
                fold = WalkForwardFold(
                    fold_index=fold_index,
                    train_start=current_train_start.to_pydatetime(),
                    train_end=train_end.to_pydatetime(),
                    test_start=test_start.to_pydatetime(),
                    test_end=test_end.to_pydatetime(),
                    train_samples=train_samples,
                    test_samples=test_samples,
                )
                folds.append(fold)
                fold_index += 1

            # 滚动向前
            current_train_start += pd.DateOffset(months=self.step_months)

        logger.info(f"Created {len(folds)} Walk-Forward folds")
        return folds

    def get_fold_data(
        self,
        data: pd.DataFrame,
        fold: WalkForwardFold,
        time_column: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取折叠的训练和测试数据

        Args:
            data: 原始数据
            fold: 折叠信息
            time_column: 时间列名

        Returns:
            (训练数据, 测试数据)
        """
        if time_column:
            times = pd.to_datetime(data[time_column])
            train_mask = (times >= fold.train_start) & (times < fold.train_end)
            test_mask = (times >= fold.test_start) & (times < fold.test_end)
        else:
            times = pd.to_datetime(data.index)
            train_mask = (times >= fold.train_start) & (times < fold.train_end)
            test_mask = (times >= fold.test_start) & (times < fold.test_end)

        train_data = data[train_mask].copy()
        test_data = data[test_mask].copy()

        return train_data, test_data

    def validate_model(
        self,
        model: Any,
        data: pd.DataFrame,
        fit_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        time_column: Optional[str] = None,
    ) -> WalkForwardReport:
        """
        执行 Walk-Forward 验证

        Args:
            model: 模型对象
            data: 完整数据
            fit_fn: 训练函数 fit_fn(model, train_data) -> model
            evaluate_fn: 评估函数 evaluate_fn(model, test_data) -> metrics
            time_column: 时间列名

        Returns:
            验证报告
        """
        # 创建折叠
        folds = self.create_folds(data, time_column)

        if len(folds) < 2:
            logger.warning("Not enough data for Walk-Forward validation")
            return WalkForwardReport(
                total_folds=0,
                results=[],
                avg_train_sharpe=0,
                avg_test_sharpe=0,
                sharpe_gap=0,
                overfitting_ratio=0,
                is_valid=False,
                failure_reasons=["Insufficient data for validation"],
            )

        # 默认函数
        if fit_fn is None:
            fit_fn = lambda m, d: m.fit(d) if hasattr(m, 'fit') else m
        if evaluate_fn is None:
            evaluate_fn = lambda m, d: m.evaluate(d) if hasattr(m, 'evaluate') else {}

        results = []
        train_sharpes = []
        test_sharpes = []

        for fold in folds:
            train_data, test_data = self.get_fold_data(data, fold, time_column)

            # 训练模型
            model = fit_fn(model, train_data)

            # 评估
            train_metrics = evaluate_fn(model, train_data)
            test_metrics = evaluate_fn(model, test_data)

            # 提取夏普比
            train_sharpe = train_metrics.get("sharpe", train_metrics.get("sharpe_ratio", 0))
            test_sharpe = test_metrics.get("sharpe", test_metrics.get("sharpe_ratio", 0))

            train_sharpes.append(train_sharpe)
            test_sharpes.append(test_sharpe)

            # 检测过拟合
            overfitting_score = train_sharpe - test_sharpe
            is_overfitting = overfitting_score > 0.3

            result = WalkForwardResult(
                fold_index=fold.fold_index,
                train_period=f"{fold.train_start.date()} ~ {fold.train_end.date()}",
                test_period=f"{fold.test_start.date()} ~ {fold.test_end.date()}",
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                is_overfitting=is_overfitting,
                overfitting_score=overfitting_score,
            )
            results.append(result)

        # 汇总统计
        avg_train_sharpe = np.mean(train_sharpes) if train_sharpes else 0
        avg_test_sharpe = np.mean(test_sharpes) if test_sharpes else 0
        sharpe_gap = avg_train_sharpe - avg_test_sharpe
        overfitting_count = sum(1 for r in results if r.is_overfitting)
        overfitting_ratio = overfitting_count / len(results) if results else 0

        # 验证是否通过
        failure_reasons = []
        if sharpe_gap > 0.3:
            failure_reasons.append(f"Average Sharpe gap too large: {sharpe_gap:.3f} > 0.3")
        if overfitting_ratio > 0.5:
            failure_reasons.append(f"Too many overfitting folds: {overfitting_ratio:.1%}")
        if avg_test_sharpe < 0:
            failure_reasons.append(f"Negative average test Sharpe: {avg_test_sharpe:.3f}")

        is_valid = len(failure_reasons) == 0

        return WalkForwardReport(
            total_folds=len(folds),
            results=results,
            avg_train_sharpe=avg_train_sharpe,
            avg_test_sharpe=avg_test_sharpe,
            sharpe_gap=sharpe_gap,
            overfitting_ratio=overfitting_ratio,
            is_valid=is_valid,
            failure_reasons=failure_reasons,
        )

    def split_data(
        self,
        data: pd.DataFrame,
        shuffle: bool = False,
        **kwargs,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        切分数据 - 禁止随机切分

        Args:
            data: 数据
            shuffle: 是否随机打乱 (必须为 False)

        Raises:
            RandomSplitForbiddenError: 如果 shuffle=True
        """
        if shuffle:
            raise RandomSplitForbiddenError(
                "禁止随机切分时序数据! 请使用 Walk-Forward 验证。"
            )

        # 简单的时序切分
        split_idx = int(len(data) * 0.8)
        return data.iloc[:split_idx], data.iloc[split_idx:]


class OverfittingDetector:
    """
    过拟合检测器

    功能:
    - 检测训练和测试夏普比差距
    - 计算 Deflated Sharpe Ratio
    - 评估数据挖掘偏差
    """

    # 最大训练/测试夏普差距
    MAX_TRAIN_TEST_GAP = 0.3

    def __init__(
        self,
        max_gap: float = MAX_TRAIN_TEST_GAP,
        significance_level: float = 0.05,
    ):
        """
        初始化过拟合检测器

        Args:
            max_gap: 最大训练/测试夏普差距
            significance_level: 显著性水平
        """
        self.max_gap = max_gap
        self.significance_level = significance_level

    def check_overfitting(
        self,
        train_sharpe: float,
        test_sharpe: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        检测是否过拟合

        Args:
            train_sharpe: 训练集夏普比
            test_sharpe: 测试集夏普比

        Returns:
            (是否过拟合, 详细信息)
        """
        gap = train_sharpe - test_sharpe

        is_overfitting = gap > self.max_gap

        details = {
            "train_sharpe": train_sharpe,
            "test_sharpe": test_sharpe,
            "gap": gap,
            "max_gap": self.max_gap,
            "is_overfitting": is_overfitting,
        }

        if is_overfitting:
            logger.warning(
                f"疑似过拟合! 训练夏普={train_sharpe:.2f}, "
                f"测试夏普={test_sharpe:.2f}, 差距={gap:.2f}"
            )

        return is_overfitting, details

    def calculate_deflated_sharpe(
        self,
        sharpe: float,
        num_trials: int,
        sample_length: int = 252,  # 年交易日
        skewness: float = 0,
        kurtosis: float = 3,
    ) -> float:
        """
        计算 Deflated Sharpe Ratio (调整后的夏普比)

        考虑多次尝试的影响，防止数据挖掘偏差。
        使用 Bailey-Lopez de Prado 公式。

        Args:
            sharpe: 原始夏普比
            num_trials: 尝试次数
            sample_length: 样本长度
            skewness: 偏度
            kurtosis: 峰度

        Returns:
            调整后的夏普比
        """
        try:
            from scipy import stats
            has_scipy = True
        except ImportError:
            has_scipy = False

        if num_trials <= 1:
            return sharpe

        # 计算预期最大夏普比 (在多次尝试下)
        # E[max(SR)] ≈ sqrt(2*log(N)) for N trials
        expected_max_sharpe = np.sqrt(2 * np.log(num_trials))

        # 方差调整
        var_sharpe = (1 + 0.5 * sharpe**2 - skewness * sharpe +
                      (kurtosis - 3) / 4 * sharpe**2) / sample_length

        # 计算 Deflated Sharpe
        if var_sharpe > 0:
            deflated = (sharpe - expected_max_sharpe) / np.sqrt(var_sharpe)
            # 转换为概率
            if has_scipy:
                prob = stats.norm.cdf(deflated)
            else:
                # 简单近似: 使用正态分布的近似公式
                # For large |z|, Φ(z) ≈ 1 if z > 0, 0 if z < 0
                prob = 0.5 * (1 + np.tanh(deflated * 0.7978845608))
        else:
            prob = 0

        # 返回有效夏普 (如果 prob < 0.05，则策略可能是偶然发现)
        if prob < self.significance_level:
            logger.warning(
                f"Deflated Sharpe test failed: prob={prob:.4f} < {self.significance_level}"
            )
            return 0

        return max(0, sharpe - expected_max_sharpe)

    def calculate_minimum_backtest_length(
        self,
        target_sharpe: float,
        num_trials: int = 1,
        significance_level: float = 0.05,
    ) -> int:
        """
        计算最小回测长度

        Args:
            target_sharpe: 目标夏普比
            num_trials: 尝试次数
            significance_level: 显著性水平

        Returns:
            最小回测长度 (年)
        """
        try:
            from scipy import stats
            has_scipy = True
        except ImportError:
            has_scipy = False

        if target_sharpe <= 0:
            return float('inf')

        # z_score for 95% confidence (significance_level=0.05) is ~1.645
        if has_scipy:
            z_score = stats.norm.ppf(1 - significance_level)
        else:
            # Approximate z_score for common significance levels
            z_scores = {0.01: 2.326, 0.05: 1.645, 0.10: 1.282}
            z_score = z_scores.get(significance_level, 1.645)

        # 考虑多次尝试
        if num_trials > 1:
            expected_max_sharpe = np.sqrt(2 * np.log(num_trials))
            effective_sharpe = target_sharpe - expected_max_sharpe
            if effective_sharpe <= 0:
                return float('inf')
        else:
            effective_sharpe = target_sharpe

        min_length = (z_score / effective_sharpe) ** 2

        return int(np.ceil(min_length))

    def evaluate_strategy(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        num_trials: int = 1,
        sample_length: int = 252,
    ) -> Dict[str, Any]:
        """
        全面评估策略

        Args:
            train_metrics: 训练集指标
            test_metrics: 测试集指标
            num_trials: 尝试次数
            sample_length: 样本长度

        Returns:
            评估报告
        """
        train_sharpe = train_metrics.get("sharpe", train_metrics.get("sharpe_ratio", 0))
        test_sharpe = test_metrics.get("sharpe", test_metrics.get("sharpe_ratio", 0))

        # 检测过拟合
        is_overfitting, overfit_details = self.check_overfitting(train_sharpe, test_sharpe)

        # 计算 Deflated Sharpe
        deflated_sharpe = self.calculate_deflated_sharpe(
            test_sharpe, num_trials, sample_length
        )

        # 计算最小回测长度
        min_length = self.calculate_minimum_backtest_length(test_sharpe, num_trials)

        return {
            "train_sharpe": train_sharpe,
            "test_sharpe": test_sharpe,
            "sharpe_gap": train_sharpe - test_sharpe,
            "is_overfitting": is_overfitting,
            "deflated_sharpe": deflated_sharpe,
            "min_backtest_years": min_length,
            "num_trials": num_trials,
            "is_statistically_significant": deflated_sharpe > 0,
            "recommendation": self._get_recommendation(
                is_overfitting, deflated_sharpe, test_sharpe
            ),
        }

    def _get_recommendation(
        self,
        is_overfitting: bool,
        deflated_sharpe: float,
        test_sharpe: float,
    ) -> str:
        """生成建议"""
        if is_overfitting:
            return "⚠️ 疑似过拟合，建议简化模型或增加正则化"
        if deflated_sharpe <= 0:
            return "❌ 策略可能是偶然发现，建议增加样本或减少尝试次数"
        if test_sharpe < 0.5:
            return "⚡ 夏普比较低，建议优化策略"
        if test_sharpe >= 1.0:
            return "✅ 策略表现良好，可以进入下一阶段验证"
        return "✓ 策略表现中等，建议继续观察"


# 全局单例
_walk_forward_validator: Optional[WalkForwardValidator] = None
_overfitting_detector: Optional[OverfittingDetector] = None


def get_walk_forward_validator(**kwargs) -> WalkForwardValidator:
    """获取全局 WalkForwardValidator 实例"""
    global _walk_forward_validator
    if _walk_forward_validator is None:
        _walk_forward_validator = WalkForwardValidator(**kwargs)
    return _walk_forward_validator


def get_overfitting_detector(**kwargs) -> OverfittingDetector:
    """获取全局 OverfittingDetector 实例"""
    global _overfitting_detector
    if _overfitting_detector is None:
        _overfitting_detector = OverfittingDetector(**kwargs)
    return _overfitting_detector


def reset_validators():
    """重置全局实例 (用于测试)"""
    global _walk_forward_validator, _overfitting_detector
    _walk_forward_validator = None
    _overfitting_detector = None
