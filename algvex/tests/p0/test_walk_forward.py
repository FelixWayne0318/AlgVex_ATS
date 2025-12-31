"""
P0-4 验收测试: Walk-Forward 验证

验收标准:
- 禁止随机切分时序数据
- Walk-Forward 滚动验证正确
- 过拟合检测有效
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.walk_forward import (
    WalkForwardValidator,
    OverfittingDetector,
    RandomSplitForbiddenError,
    get_walk_forward_validator,
    get_overfitting_detector,
    reset_validators,
)


class TestWalkForwardValidator:
    """测试 Walk-Forward 验证器"""

    def setup_method(self):
        """每个测试前重置"""
        reset_validators()

    @pytest.fixture
    def sample_data(self):
        """创建测试数据 (2年)"""
        dates = pd.date_range("2022-01-01", "2024-01-01", freq="1D")
        data = pd.DataFrame({
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, len(dates)),
        }, index=dates)
        return data

    def test_create_folds(self, sample_data):
        """测试创建折叠"""
        # 使用较小的 min_train_samples，因为日数据12个月约365个样本
        validator = WalkForwardValidator(train_months=12, test_months=3, min_train_samples=300)
        folds = validator.create_folds(sample_data)

        # 2年数据，12个月训练 + 3个月测试，步长3个月
        # 应该有约 3-4 个折叠
        assert len(folds) >= 2

        for fold in folds:
            # 训练集必须在测试集之前
            assert fold.train_end <= fold.test_start

            # 验证样本数
            assert fold.train_samples > 0
            assert fold.test_samples > 0

    def test_no_overlap(self, sample_data):
        """测试训练和测试不重叠"""
        validator = WalkForwardValidator(train_months=12, test_months=3, min_train_samples=300)
        folds = validator.create_folds(sample_data)

        for fold in folds:
            train_data, test_data = validator.get_fold_data(sample_data, fold)

            # 检查没有重叠
            train_dates = set(train_data.index)
            test_dates = set(test_data.index)
            overlap = train_dates & test_dates

            assert len(overlap) == 0, "训练和测试集存在重叠!"

    def test_train_before_test(self, sample_data):
        """测试训练集在测试集之前"""
        validator = WalkForwardValidator(train_months=12, test_months=3, min_train_samples=300)
        folds = validator.create_folds(sample_data)

        for fold in folds:
            train_data, test_data = validator.get_fold_data(sample_data, fold)

            # 训练集最大日期 < 测试集最小日期
            assert train_data.index.max() < test_data.index.min()

    def test_forbid_random_split(self, sample_data):
        """测试禁止随机切分"""
        validator = WalkForwardValidator()

        # shuffle=True 应该抛出异常
        with pytest.raises(RandomSplitForbiddenError, match="禁止随机切分"):
            validator.split_data(sample_data, shuffle=True)

    def test_allow_sequential_split(self, sample_data):
        """测试允许时序切分"""
        validator = WalkForwardValidator()

        # shuffle=False 应该正常工作
        train, test = validator.split_data(sample_data, shuffle=False)

        assert len(train) > 0
        assert len(test) > 0
        assert train.index.max() < test.index.min()

    def test_purge_days(self, sample_data):
        """测试训练测试间隔"""
        validator = WalkForwardValidator(
            train_months=12,
            test_months=3,
            purge_days=7,  # 7天间隔
            min_train_samples=300,
        )
        folds = validator.create_folds(sample_data)

        for fold in folds:
            gap = (fold.test_start - fold.train_end).days
            assert gap >= 7, f"间隔不足: {gap} 天"

    def test_min_train_samples(self):
        """测试最小训练样本"""
        # 创建较少数据
        dates = pd.date_range("2023-01-01", "2023-06-01", freq="1D")
        data = pd.DataFrame({"close": range(len(dates))}, index=dates)

        validator = WalkForwardValidator(
            train_months=12,  # 需要12个月，但只有5个月数据
            min_train_samples=1000,
        )

        folds = validator.create_folds(data)
        # 数据不足，应该没有折叠
        assert len(folds) == 0

    def test_validate_model(self, sample_data):
        """测试模型验证"""

        # 创建简单模型
        class SimpleModel:
            def fit(self, data):
                return self

            def evaluate(self, data):
                return {
                    "sharpe": np.random.uniform(0.5, 1.5),
                    "return": np.random.uniform(0.1, 0.3),
                }

        # 使用较小的 min_train_samples 以适配日数据
        validator = WalkForwardValidator(train_months=6, test_months=2, min_train_samples=150)
        model = SimpleModel()

        report = validator.validate_model(
            model=model,
            data=sample_data,
            fit_fn=lambda m, d: m.fit(d),
            evaluate_fn=lambda m, d: m.evaluate(d),
        )

        assert report.total_folds > 0
        assert len(report.results) == report.total_folds
        assert 0 <= report.overfitting_ratio <= 1

    def test_global_singleton(self):
        """测试全局单例"""
        reset_validators()

        v1 = get_walk_forward_validator()
        v2 = get_walk_forward_validator()

        assert v1 is v2


class TestOverfittingDetector:
    """测试过拟合检测器"""

    def setup_method(self):
        """每个测试前重置"""
        reset_validators()

    def test_detect_overfitting(self):
        """测试过拟合检测"""
        detector = OverfittingDetector(max_gap=0.3)

        # 正常情况
        is_overfit, details = detector.check_overfitting(
            train_sharpe=1.0,
            test_sharpe=0.8,
        )
        assert is_overfit is False
        assert abs(details["gap"] - 0.2) < 1e-10  # 浮点数比较

        # 过拟合
        is_overfit, details = detector.check_overfitting(
            train_sharpe=1.5,
            test_sharpe=1.0,
        )
        assert is_overfit is True
        assert abs(details["gap"] - 0.5) < 1e-10  # 浮点数比较

    def test_deflated_sharpe(self):
        """测试 Deflated Sharpe Ratio"""
        detector = OverfittingDetector()

        # 单次尝试
        deflated = detector.calculate_deflated_sharpe(
            sharpe=1.0,
            num_trials=1,
        )
        assert deflated == 1.0  # 单次尝试不调整

        # 多次尝试应该降低
        deflated_multi = detector.calculate_deflated_sharpe(
            sharpe=1.0,
            num_trials=100,
        )
        assert deflated_multi < 1.0

    def test_min_backtest_length(self):
        """测试最小回测长度"""
        detector = OverfittingDetector()

        # 高夏普比需要较短回测
        min_len_high = detector.calculate_minimum_backtest_length(
            target_sharpe=2.0,
            num_trials=1,
        )

        # 低夏普比需要较长回测
        min_len_low = detector.calculate_minimum_backtest_length(
            target_sharpe=0.5,
            num_trials=1,
        )

        assert min_len_high < min_len_low

    def test_evaluate_strategy(self):
        """测试策略评估"""
        detector = OverfittingDetector()

        train_metrics = {"sharpe": 1.2, "return": 0.25}
        test_metrics = {"sharpe": 1.0, "return": 0.20}

        evaluation = detector.evaluate_strategy(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            num_trials=10,
        )

        assert "train_sharpe" in evaluation
        assert "test_sharpe" in evaluation
        assert "sharpe_gap" in evaluation
        assert "is_overfitting" in evaluation
        assert "deflated_sharpe" in evaluation
        assert "recommendation" in evaluation

    def test_recommendation(self):
        """测试建议生成"""
        detector = OverfittingDetector()

        # 过拟合情况
        eval1 = detector.evaluate_strategy(
            train_metrics={"sharpe": 2.0},
            test_metrics={"sharpe": 1.0},
        )
        assert "过拟合" in eval1["recommendation"]

        # 良好情况
        eval2 = detector.evaluate_strategy(
            train_metrics={"sharpe": 1.2},
            test_metrics={"sharpe": 1.1},
        )
        assert eval2["is_overfitting"] is False

    def test_global_singleton(self):
        """测试全局单例"""
        reset_validators()

        d1 = get_overfitting_detector()
        d2 = get_overfitting_detector()

        assert d1 is d2


class TestWalkForwardIntegration:
    """Walk-Forward 集成测试"""

    def test_complete_workflow(self):
        """测试完整工作流"""
        # 创建测试数据
        dates = pd.date_range("2020-01-01", "2024-01-01", freq="1D")
        data = pd.DataFrame({
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "feature": np.random.randn(len(dates)),
        }, index=dates)

        # 创建验证器
        validator = WalkForwardValidator(
            train_months=12,
            test_months=3,
            min_train_samples=200,
        )

        # 创建折叠
        folds = validator.create_folds(data)
        assert len(folds) >= 3

        # 检测器
        detector = OverfittingDetector()

        # 模拟评估
        for fold in folds:
            train_data, test_data = validator.get_fold_data(data, fold)

            # 模拟指标
            train_sharpe = np.random.uniform(0.8, 1.5)
            test_sharpe = np.random.uniform(0.5, 1.2)

            is_overfit, _ = detector.check_overfitting(train_sharpe, test_sharpe)
            # 只是检查运行，不检查结果
