"""
AlgVex 模型训练层

提供 ML 模型训练和预测功能:
- ModelTrainer: 模型训练器
- WalkForwardValidator: 滚动验证
- FeatureSelector: 特征选择
- ModelRegistry: 模型注册表

Qlib 风格模型 (直接使用):
- LGBModel: LightGBM 模型
- XGBModel: XGBoost 模型
- LinearModel: 线性模型 (OLS, NNLS, Ridge, Lasso)

使用示例:
    # 方式1: 使用 ModelTrainer
    from algvex.core.model import ModelTrainer, ModelConfig

    config = ModelConfig(
        model_type="lightgbm",
        params={"n_estimators": 500, "learning_rate": 0.05},
    )
    trainer = ModelTrainer(config)
    model = trainer.train(X_train, y_train)

    # 方式2: 直接使用 Qlib 风格模型
    from algvex.core.model import LGBModel

    model = LGBModel(num_leaves=64, learning_rate=0.05)
    model.fit(dataset)
    predictions = model.predict(dataset, segment='test')
"""

from .trainer import (
    ModelTrainer,
    ModelConfig,
    TrainingResult,
)

from .walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardResult,
)

from .feature_selection import (
    FeatureSelector,
    FeatureImportance,
)

# Qlib 风格模型
from .qlib_models import (
    BaseModel,
    Model,
    ModelFT,
    LGBModel,
    XGBModel,
    LinearModel,
    get_model,
)

__all__ = [
    # Trainer
    "ModelTrainer",
    "ModelConfig",
    "TrainingResult",
    # Walk Forward
    "WalkForwardValidator",
    "WalkForwardConfig",
    "WalkForwardResult",
    # Feature Selection
    "FeatureSelector",
    "FeatureImportance",
    # Qlib 风格模型
    "BaseModel",
    "Model",
    "ModelFT",
    "LGBModel",
    "XGBModel",
    "LinearModel",
    "get_model",
]
