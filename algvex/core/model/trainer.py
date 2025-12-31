"""
AlgVex 模型训练器

设计文档参考: Section 4.3 ML模型配置

支持的模型:
- LightGBM (推荐)
- XGBoost
- Linear (Ridge, Lasso)

使用示例:
    config = ModelConfig(
        model_type="lightgbm",
        params={"n_estimators": 500},
    )
    trainer = ModelTrainer(config)
    result = trainer.train(X_train, y_train, X_val, y_val)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.linear_model import Ridge, Lasso, LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    Ridge = None
    Lasso = None
    LinearRegression = None


class ModelType(Enum):
    """模型类型"""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    RIDGE = "ridge"
    LASSO = "lasso"
    LINEAR = "linear"


@dataclass
class ModelConfig:
    """
    模型配置

    Attributes:
        model_type: 模型类型
        params: 模型参数
        early_stopping_rounds: 早停轮数
        validation_split: 验证集比例
        feature_selection: 是否启用特征选择
        top_k_features: 选择前 k 个特征
        normalize: 是否标准化特征
        random_state: 随机种子
    """
    model_type: str = "lightgbm"
    params: Dict[str, Any] = field(default_factory=dict)
    early_stopping_rounds: int = 50
    validation_split: float = 0.2
    feature_selection: bool = True
    top_k_features: int = 50
    normalize: bool = True
    random_state: int = 42

    def __post_init__(self):
        # 默认参数
        if self.model_type == "lightgbm" and not self.params:
            self.params = {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 8,
                "num_leaves": 64,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "min_child_samples": 50,
                "verbose": -1,
            }
        elif self.model_type == "xgboost" and not self.params:
            self.params = {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "verbosity": 0,
            }
        elif self.model_type in ["ridge", "lasso"] and not self.params:
            self.params = {"alpha": 1.0}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_type": self.model_type,
            "params": self.params,
            "early_stopping_rounds": self.early_stopping_rounds,
            "validation_split": self.validation_split,
            "feature_selection": self.feature_selection,
            "top_k_features": self.top_k_features,
            "normalize": self.normalize,
            "random_state": self.random_state,
        }


@dataclass
class TrainingResult:
    """
    训练结果

    Attributes:
        model: 训练好的模型
        train_metrics: 训练集指标
        val_metrics: 验证集指标
        feature_importance: 特征重要性
        selected_features: 选中的特征
        training_time: 训练时间
        config: 模型配置
    """
    model: Any
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None
    selected_features: Optional[List[str]] = None
    training_time: float = 0.0
    config: Optional[Dict[str, Any]] = None
    scaler: Optional[Any] = None  # StandardScaler if sklearn available

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "feature_importance": self.feature_importance,
            "selected_features": self.selected_features,
            "training_time": self.training_time,
            "config": self.config,
        }


class ModelTrainer:
    """
    模型训练器

    使用示例:
        config = ModelConfig(model_type="lightgbm")
        trainer = ModelTrainer(config)

        # 训练
        result = trainer.train(X_train, y_train, X_val, y_val)

        # 预测
        predictions = trainer.predict(X_test)
    """

    def __init__(self, config: ModelConfig):
        """
        初始化训练器

        Args:
            config: 模型配置
        """
        self.config = config
        self.model = None
        self.scaler = StandardScaler() if (config.normalize and HAS_SKLEARN) else None
        self.selected_features = None
        self.feature_names = None

    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> TrainingResult:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征 (可选)
            y_val: 验证标签 (可选)

        Returns:
            TrainingResult
        """
        import time
        start_time = time.time()

        # 转换为 numpy
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if isinstance(y_val, pd.Series):
            y_val = y_val.values

        # 自动分割验证集
        if X_val is None and self.config.validation_split > 0:
            split_idx = int(len(X_train) * (1 - self.config.validation_split))
            X_train, X_val = X_train[:split_idx], X_train[split_idx:]
            y_train, y_val = y_train[:split_idx], y_train[split_idx:]

        # 标准化
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)

        # 训练模型
        model_type = self.config.model_type.lower()

        if model_type == "lightgbm":
            self.model = self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif model_type == "xgboost":
            self.model = self._train_xgboost(X_train, y_train, X_val, y_val)
        elif model_type == "ridge":
            self.model = self._train_ridge(X_train, y_train)
        elif model_type == "lasso":
            self.model = self._train_lasso(X_train, y_train)
        elif model_type == "linear":
            self.model = self._train_linear(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 计算指标
        train_pred = self._predict_internal(X_train)
        train_metrics = self._calculate_metrics(y_train, train_pred)

        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_pred = self._predict_internal(X_val)
            val_metrics = self._calculate_metrics(y_val, val_pred)

        # 特征重要性
        feature_importance = self._get_feature_importance()

        training_time = time.time() - start_time

        return TrainingResult(
            model=self.model,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            feature_importance=feature_importance,
            selected_features=self.selected_features,
            training_time=training_time,
            config=self.config.to_dict(),
            scaler=self.scaler,
        )

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        预测

        Args:
            X: 特征

        Returns:
            预测值
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.scaler:
            X = self.scaler.transform(X)

        return self._predict_internal(X)

    def _train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ):
        """训练 LightGBM"""
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")

        params = self.config.params.copy()
        params["random_state"] = self.config.random_state

        model = lgb.LGBMRegressor(**params)

        callbacks = []
        if self.config.early_stopping_rounds > 0 and X_val is not None:
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds))

        eval_set = [(X_val, y_val)] if X_val is not None else None

        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )

        return model

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ):
        """训练 XGBoost"""
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed")

        params = self.config.params.copy()
        params["random_state"] = self.config.random_state

        model = xgb.XGBRegressor(**params)

        eval_set = [(X_val, y_val)] if X_val is not None else None

        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=self.config.early_stopping_rounds if X_val is not None else None,
            verbose=False,
        )

        return model

    def _train_ridge(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练 Ridge"""
        params = self.config.params.copy()
        model = Ridge(**params)
        model.fit(X_train, y_train)
        return model

    def _train_lasso(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练 Lasso"""
        params = self.config.params.copy()
        model = Lasso(**params)
        model.fit(X_train, y_train)
        return model

    def _train_linear(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练 Linear"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """内部预测方法"""
        return self.model.predict(X)

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """计算评估指标"""
        # MSE and RMSE
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = np.sqrt(mse)

        # R2 score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # 计算 IC
        ic = np.corrcoef(y_true, y_pred)[0, 1]

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "ic": float(ic) if not np.isnan(ic) else 0.0,
        }

    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        if self.model is None:
            return None

        importance = None

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_)

        if importance is None:
            return None

        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return {f"feature_{i}": float(v) for i, v in enumerate(importance)}

    def save(self, path: str):
        """保存模型"""
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "config": self.config.to_dict(),
                "feature_names": self.feature_names,
            }, f)

    @classmethod
    def load(cls, path: str) -> "ModelTrainer":
        """加载模型"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        config = ModelConfig(**data["config"])
        trainer = cls(config)
        trainer.model = data["model"]
        trainer.scaler = data["scaler"]
        trainer.feature_names = data["feature_names"]

        return trainer
