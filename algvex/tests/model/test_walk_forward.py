"""
WalkForwardValidator 单元测试

测试滚动验证器的核心功能:
- 窗口生成
- 折叠验证
- 指标计算
"""

import pytest
import numpy as np
import pandas as pd

from algvex.core.model import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardResult,
)


class TestWalkForwardConfig:
    """测试滚动验证配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = WalkForwardConfig()
        assert config.train_window == 252 * 288  # 252天 (5分钟频率)
        assert config.test_window == 21 * 288     # 21天
        assert config.step_size == 21 * 288       # 每21天滚动

    def test_custom_config(self):
        """测试自定义配置"""
        config = WalkForwardConfig(
            train_window=500,
            test_window=100,
            step_size=50,
        )
        assert config.train_window == 500
        assert config.test_window == 100
        assert config.step_size == 50

    def test_config_to_dict(self):
        """测试配置转字典"""
        config = WalkForwardConfig(
            train_window=1000,
            test_window=200,
        )
        data = config.to_dict()
        assert data["train_window"] == 1000
        assert data["test_window"] == 200


class TestWalkForwardValidator:
    """测试滚动验证器"""

    @pytest.fixture
    def validator(self):
        """创建验证器"""
        config = WalkForwardConfig(
            train_window=100,
            test_window=20,
            step_size=20,
            min_train_samples=50,  # Lower for testing
        )
        return WalkForwardValidator(config)

    def test_validator_init(self, validator):
        """测试验证器初始化"""
        assert validator.config is not None
        assert validator.config.train_window == 100
        assert validator.config.test_window == 20

    def test_generate_folds(self, validator):
        """测试生成折叠"""
        n_samples = 200
        folds = validator.generate_folds(n_samples)

        assert len(folds) > 0

        # 验证每个折叠 - folds are [(train_range, test_range), ...]
        for train_range, test_range in folds:
            train_start, train_end = train_range
            test_start, test_end = test_range
            assert train_start >= 0
            assert train_end > train_start
            assert test_start >= train_end
            assert test_end > test_start
            assert test_end <= n_samples

    def test_generate_folds_small_data(self, validator):
        """测试小数据集生成折叠"""
        n_samples = 50  # 小于 train_window
        folds = validator.generate_folds(n_samples)

        # 数据太小,应该没有折叠
        assert len(folds) == 0

    def test_fold_boundaries(self, validator):
        """测试折叠边界"""
        n_samples = 300
        folds = validator.generate_folds(n_samples)

        # 检查第一个折叠
        if folds:
            train_range, test_range = folds[0]
            train_start, train_end = train_range
            test_start, test_end = test_range
            assert train_start == 0
            assert train_end - train_start == validator.config.train_window
            assert test_end - test_start == validator.config.test_window


class TestWalkForwardResult:
    """测试滚动验证结果"""

    def test_result_creation(self):
        """测试创建结果"""
        from algvex.core.model.walk_forward import FoldResult

        # 创建 FoldResult 对象
        fold1 = FoldResult(
            fold_idx=0,
            train_start_idx=0,
            train_end_idx=100,
            test_start_idx=100,
            test_end_idx=120,
            train_metrics={"mse": 0.1, "r2": 0.9},
            test_metrics={"mse": 0.11, "r2": 0.88},
        )
        fold2 = FoldResult(
            fold_idx=1,
            train_start_idx=20,
            train_end_idx=120,
            test_start_idx=120,
            test_end_idx=140,
            train_metrics={"mse": 0.12, "r2": 0.88},
            test_metrics={"mse": 0.13, "r2": 0.86},
        )

        result = WalkForwardResult(
            folds=[fold1, fold2],
            aggregate_metrics={"mse": 0.11, "r2": 0.87},
            total_time=5.0,
        )
        assert len(result.folds) == 2
        assert result.total_time == 5.0
        assert result.aggregate_metrics["mse"] == 0.11

    def test_result_to_dict(self):
        """测试结果转字典"""
        from algvex.core.model.walk_forward import FoldResult

        fold1 = FoldResult(
            fold_idx=0,
            train_start_idx=0,
            train_end_idx=100,
            test_start_idx=100,
            test_end_idx=120,
            train_metrics={"mse": 0.1},
            test_metrics={"mse": 0.12},
        )

        result = WalkForwardResult(
            folds=[fold1],
            aggregate_metrics={"mse": 0.1},
            total_time=1.0,
        )
        data = result.to_dict()
        assert "num_folds" in data
        assert "aggregate_metrics" in data
        assert "total_time" in data
