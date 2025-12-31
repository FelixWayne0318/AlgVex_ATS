"""
ModelTrainer 单元测试

测试模型训练器的核心功能:
- 配置验证
- 训练流程
- 预测功能
- 特征重要性
"""

import pytest
import numpy as np
import pandas as pd

from algvex.core.model import (
    ModelTrainer,
    ModelConfig,
    TrainingResult,
)


class TestModelConfig:
    """测试模型配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = ModelConfig()
        assert config.model_type == "lightgbm"
        assert config.normalize is True
        assert config.early_stopping_rounds == 50

    def test_custom_config(self):
        """测试自定义配置"""
        config = ModelConfig(
            model_type="xgboost",
            params={"n_estimators": 200, "learning_rate": 0.1},
        )
        assert config.model_type == "xgboost"
        assert config.params["n_estimators"] == 200
        assert config.params["learning_rate"] == 0.1

    def test_config_to_dict(self):
        """测试配置转字典"""
        config = ModelConfig(
            model_type="ridge",
            params={"alpha": 0.5},
        )
        data = config.to_dict()
        assert "model_type" in data
        assert data["params"]["alpha"] == 0.5


class TestModelTrainer:
    """测试模型训练器"""

    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        np.random.seed(42)
        n = 500
        X = pd.DataFrame({
            f"feature_{i}": np.random.randn(n)
            for i in range(10)
        })
        # 简单线性关系
        y = X["feature_0"] * 0.5 + X["feature_1"] * 0.3 + np.random.randn(n) * 0.1
        return X, y.values

    @pytest.fixture
    def trainer(self):
        """创建训练器 (使用不依赖外部库的配置)"""
        # 由于可能没有lightgbm,我们测试基础功能
        config = ModelConfig(
            model_type="lightgbm",
            params={"n_estimators": 10},
            normalize=False,  # 避免需要sklearn
        )
        return ModelTrainer(config)

    def test_trainer_init(self, trainer):
        """测试训练器初始化"""
        assert trainer.config is not None
        assert trainer.model is None

    def test_config_validation(self):
        """测试配置验证"""
        # 有效配置
        config = ModelConfig(model_type="lightgbm")
        assert config.model_type == "lightgbm"

    def test_predict_without_training_raises(self, trainer, sample_data):
        """测试未训练时预测应该报错"""
        X, _ = sample_data
        with pytest.raises(ValueError, match="Model not trained"):
            trainer.predict(X)


class TestTrainingResult:
    """测试训练结果"""

    def test_training_result_creation(self):
        """测试创建训练结果"""
        result = TrainingResult(
            model=None,  # Mock model
            train_metrics={"mse": 0.1, "r2": 0.9},
            val_metrics={"mse": 0.15, "r2": 0.85},
            feature_importance={"feature_0": 0.5, "feature_1": 0.3},
            training_time=1.5,
        )
        assert result.train_metrics["mse"] == 0.1
        assert result.val_metrics["r2"] == 0.85
        assert result.training_time == 1.5

    def test_training_result_to_dict(self):
        """测试结果转字典"""
        result = TrainingResult(
            model=None,  # Mock model
            train_metrics={"mse": 0.1},
            val_metrics={"mse": 0.15},
            training_time=1.0,
        )
        data = result.to_dict()
        assert "train_metrics" in data
        assert "val_metrics" in data
        assert "training_time" in data
